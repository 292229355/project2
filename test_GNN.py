import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader as GeoDataLoader
from torch_geometric.data import Dataset, Data
from torch_geometric.nn import GATConv, global_mean_pool
from preprocess2 import extract_descriptors_and_build_graph2
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, classification_report


class DescriptorGraphDataset(Dataset):
    def __init__(self, path, mode="train"):
        super(DescriptorGraphDataset, self).__init__()
        self.path = path
        self.mode = mode
        self.files = sorted(
            [
                os.path.join(path, x)
                for x in os.listdir(path)
                if x.endswith(".jpg") or x.endswith(".png")
            ]
        )

        if len(self.files) == 0:
            print(f"No image files found in {path}.")
            exit()

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        data = extract_descriptors_and_build_graph2(fname)
        x, edge_index, pos, edge_attr = (
            data  # x 是特徵 edge_index 是紀錄兩節點是否有edge, pos是座標 ,edge_attr 是邊權重
        )

        # 由於資料切割方式，檔案命名為 編號_label
        try:
            file_parts = os.path.basename(fname).split("_")
            label_str = file_parts[1].split(".")[0]
            label = int(label_str)
        except:
            label = -1  # 以防萬一

        # 創建PyTorch Geometric 的graph型態
        graph_data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=torch.tensor([label], dtype=torch.float),  # pytorch強制用的
            pos=pos,
        )

        return graph_data


class GATClassifier(nn.Module):
    def __init__(self, input_dim):
        super(GATClassifier, self).__init__()
        self.conv1 = GATConv(input_dim, 128, heads=4, concat=True, edge_dim=1)
        self.bn1 = nn.BatchNorm1d(128 * 4)
        self.conv2 = GATConv(128 * 4, 64, heads=4, concat=False, edge_dim=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, 1)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x, edge_index, batch, edge_attr=None):
        if edge_attr is not None and edge_attr.size(0) == 0:
            edge_attr = None  # 如果沒有就設置成None
        x = self.conv1(x, edge_index, edge_attr=edge_attr)
        x = self.bn1(x)
        x = F.elu(x)
        x = self.conv2(x, edge_index, edge_attr=edge_attr)
        x = self.bn2(x)
        x = F.elu(x)
        x = global_mean_pool(x, batch)
        x = self.dropout(x)
        x = self.fc1(x)
        x = F.elu(x)
        x = self.fc2(x)
        return x  # 返回 logits


# 載入測試集資料夾
dataset_dir = "dataset/Inpaint_dataset/test"

# 準備模型與設備
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 初始化測試資料集和 DataLoader
test_set = DescriptorGraphDataset(dataset_dir, mode="test")
batch_size = 512  # 根據顯存大小調整
test_loader = GeoDataLoader(test_set, batch_size=batch_size, shuffle=False)

# 加載之前訓練保存的模型
input_dim = test_set[0].num_node_features  # 假設測試集與訓練集具有相同的特徵維度
model_file = "models/Inpaint_best.ckpt"  # 模型文件路徑
model = GATClassifier(input_dim).to(device)
model.load_state_dict(torch.load(model_file))
model.eval()

# 計算測試集預測結果與 AUC
all_labels = []
all_probs = []
all_preds = []

with torch.no_grad():
    for batch in test_loader:
        batch = batch.to(device)
        logits = model(
            batch.x, batch.edge_index, batch.batch, edge_attr=batch.edge_attr
        ).squeeze()
        preds = torch.sigmoid(logits)
        all_probs.extend(preds.cpu().numpy())  # 預測機率
        all_labels.extend(batch.y.cpu().numpy())  # 真實標籤
        all_preds.extend((preds > 0.5).long().cpu().numpy())  # 將機率轉換為二元標籤

# 確保存在有效的標籤和預測
if len(all_labels) > 0 and len(all_probs) > 0:
    # 計算 ROC 曲線和 AUC 值
    fpr, tpr, thresholds = roc_curve(all_labels, all_probs)
    roc_auc = auc(fpr, tpr)

    # 繪製 ROC 曲線
    plt.figure()
    lw = 2
    plt.plot(
        fpr,
        tpr,
        color="darkorange",
        lw=lw,
        label="ROC curve (AUC = %0.4f)" % roc_auc,
    )
    plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("(ROC) - Inpaint Test Set")
    plt.legend(loc="lower right")
    plt.show()

    print(f"AUC: {roc_auc:.4f}")

    # 打印分類報告
    from sklearn.metrics import classification_report

    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=["Real", "Fake"]))

else:
    print("No valid data available to compute AUC.")
