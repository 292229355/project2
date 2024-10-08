import os
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch_geometric.loader import DataLoader as GeoDataLoader
from torch_geometric.data import Data
from torch_geometric.nn import GATConv, global_mean_pool
from tqdm.auto import tqdm
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from preprocess2 import extract_descriptors_and_build_graph2

myseed = 6666
torch.manual_seed(myseed)
torch.cuda.manual_seed_all(myseed)
np.random.seed(myseed)
random.seed(myseed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


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
            edge_attr = None  # 如果 沒有就設置成NOne
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


def train(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    total_acc = 0
    for batch in tqdm(loader, desc="Training"):
        batch = batch.to(device)
        optimizer.zero_grad()
        logits = model(
            batch.x, batch.edge_index, batch.batch, edge_attr=batch.edge_attr
        ).squeeze()
        loss = criterion(logits, batch.y)
        loss.backward()
        optimizer.step()
        preds = torch.sigmoid(logits)
        acc = ((preds > 0.5).float() == batch.y).float().mean()
        total_loss += loss.item()
        total_acc += acc.item()
    return total_loss / len(loader), total_acc / len(loader)


def validate(model, loader, criterion):
    model.eval()
    total_loss = 0
    total_acc = 0
    all_labels = []
    all_probs = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Validating"):
            batch = batch.to(device)
            logits = model(
                batch.x, batch.edge_index, batch.batch, edge_attr=batch.edge_attr
            ).squeeze()
            loss = criterion(logits, batch.y)
            preds = torch.sigmoid(logits)
            acc = ((preds > 0.5).float() == batch.y).float().mean()
            total_loss += loss.item()
            total_acc += acc.item()
            all_labels.extend(batch.y.cpu().numpy())
            all_probs.extend(preds.cpu().numpy())
    avg_loss = total_loss / len(loader)
    avg_acc = total_acc / len(loader)
    return avg_loss, avg_acc, all_labels, all_probs


if __name__ == "__main__":
    dataset_dir = "dataset/Inpaint_dataset"
    exp_name = "Inpaint"

    os.makedirs("models", exist_ok=True)

    train_set = DescriptorGraphDataset(os.path.join(dataset_dir, "train"), mode="train")
    valid_set = DescriptorGraphDataset(os.path.join(dataset_dir, "valid"), mode="valid")

    # DataLoader
    batch_size = 512
    train_loader = GeoDataLoader(train_set, batch_size=batch_size, shuffle=True)
    valid_loader = GeoDataLoader(valid_set, batch_size=batch_size, shuffle=False)

    sample_data = None
    for data in train_set:
        if data is not None:
            sample_data = data
            break
    if sample_data is None:
        print("No valid data found in training set.")
        exit()
    input_dim = sample_data.num_node_features

    model = GATClassifier(input_dim).to(device)
    # loss值我看大家都用這個，適用二元分類任務
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    n_epochs = 10
    patience = 10
    best_acc = 0
    stale = 0
    train_losses, valid_losses, train_accuracies, valid_accuracies = [], [], [], []

    for epoch in range(n_epochs):
        train_loss, train_acc = train(model, train_loader, optimizer, criterion)
        valid_loss, valid_acc, valid_labels, valid_probs = validate(
            model, valid_loader, criterion
        )

        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
        train_accuracies.append(train_acc)
        valid_accuracies.append(valid_acc)

        print(
            f"[Epoch {epoch + 1:03d}/{n_epochs:03d}] "
            f"Train Loss: {train_loss:.5f}, Train Acc: {train_acc:.5f} | "
            f"Valid Loss: {valid_loss:.5f}, Valid Acc: {valid_acc:.5f}"
        )

        scheduler.step()

        if valid_acc > best_acc:
            print(f"Best model found at epoch {epoch + 1}, saving model")
            torch.save(model.state_dict(), f"models/{exp_name}_best.ckpt")
            best_acc = valid_acc
            stale = 0
            # 用來繪製ROC
            best_valid_labels = valid_labels
            best_valid_probs = valid_probs
        else:
            stale += 1
            if stale > patience:
                print(
                    f"No improvement in {patience} consecutive epochs, early stopping"
                )
                break

    test_set = DescriptorGraphDataset(os.path.join(dataset_dir, "test"), mode="test")
    test_loader = GeoDataLoader(test_set, batch_size=batch_size, shuffle=False)

    model_file = f"models/{exp_name}_best.ckpt"
    model_best = GATClassifier(input_dim).to(device)
    model_best.load_state_dict(torch.load(model_file))
    model_best.eval()

    prediction = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            batch = batch.to(device)
            logits = model_best(
                batch.x, batch.edge_index, batch.batch, edge_attr=batch.edge_attr
            ).squeeze()
            preds = torch.sigmoid(logits)
            prediction.extend((preds > 0.5).long().cpu().numpy())

    df = pd.DataFrame()
    df["Id"] = [str(i).zfill(4) for i in range(1, len(test_set) + 1)]
    df["Category"] = prediction
    df.to_csv("submission.csv", index=False)
    print("Prediction results saved to submission.csv")

if best_valid_labels is not None and best_valid_probs is not None:
    from sklearn.metrics import roc_curve, auc

    best_valid_labels = np.array(best_valid_labels)
    best_valid_probs = np.array(best_valid_probs)

    fpr, tpr, thresholds = roc_curve(best_valid_labels, best_valid_probs)
    roc_auc = auc(fpr, tpr)

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
    plt.title("Receiver Operating Characteristic (ROC)")
    plt.legend(loc="lower right")
    plt.show()
else:
    print("No validation data available to compute ROC curve.")
