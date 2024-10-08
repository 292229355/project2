import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, losses
from spektral.layers import GATConv, GlobalAvgPool
from spektral.data.dataset import Dataset, Graph
from spektral.data import DisjointLoader
import scipy.sparse as sp
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import random
import shutil


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


set_seed(42)


class MyDataset(Dataset):
    def __init__(self, path=None, force_download=False, **kwargs):
        self.amount = 0  # 初始化計數器
        self._path = path  # 設定 _path 屬性

        # 如果 force_download 為 True，或路徑不存在，則調用 download()
        if force_download and os.path.exists(self.path):
            shutil.rmtree(self.path)

        super().__init__(auto_download=True, **kwargs)  # 傳遞 auto_download=True

    @property
    def path(self):
        # 定義 path 屬性，返回 _path
        return self._path

    def read(self):
        print(f"從 {self.path} 讀取資料集")
        output = []
        # 從文件中讀取資料，返回 Graph 物件列表
        files = sorted(os.listdir(self.path))
        for filename in files:
            if filename.endswith(".npz"):
                filepath = os.path.join(self.path, filename)
                data = np.load(filepath, allow_pickle=True)
                x = data["x"]
                adj_data = data["adj_data"]
                adj_indices = data["adj_indices"]
                adj_indptr = data["adj_indptr"]
                adj_shape = data["adj_shape"]
                a = sp.csr_matrix((adj_data, adj_indices, adj_indptr), shape=adj_shape)
                e = data["e"]
                y = data["y"]
                graph = Graph(x=x, a=a, e=e, y=y)
                output.append(graph)
        return output


if __name__ == "__main__":
    # 創建資料集實例，並指定資料集路徑
    train_dataset = MyDataset(path="Datasets\\train")
    val_dataset = MyDataset(path="Datasets\\val")
    test_dataset = MyDataset(path="Datasets\\test")

    # 創建資料加載器
    batch_size = 16  # 根據您的顯存大小調整
    train_loader = DisjointLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DisjointLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DisjointLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 檢查輸入特徵維度
    sample_graph = train_dataset[0]
    input_dim = sample_graph.n_node_features
    # edge_dim = sample_graph.e.shape[-1] if sample_graph.e is not None else None

    # 定義模型
    class GATClassifier(models.Model):
        def __init__(self, input_dim):
            super(GATClassifier, self).__init__()
            self.conv1 = GATConv(
                channels=128,
                activation="elu",
                attn_heads=4,
                concat_heads=True,
            )
            self.bn1 = layers.BatchNormalization()
            self.conv2 = GATConv(
                channels=64,
                activation="elu",
                attn_heads=4,
                concat_heads=False,
            )
            self.bn2 = layers.BatchNormalization()
            self.global_pool = GlobalAvgPool()
            self.dropout = layers.Dropout(0.5)
            self.fc1 = layers.Dense(32, activation="elu")
            self.fc2 = layers.Dense(1, activation=None)  # 最後一層不加激活函數

        def call(self, inputs):
            x, a, i = inputs  # 移除 e
            x = self.conv1([x, a])
            x = self.bn1(x)
            x = self.conv2([x, a])
            x = self.bn2(x)
            x = self.global_pool([x, i])
            x = self.dropout(x)
            x = self.fc1(x)
            x = self.fc2(x)
            return x

    # 準備模型
    model = GATClassifier(input_dim)

    # 定義損失函數和優化器
    loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4, decay=1e-5)

    # 定義評估指標
    train_loss_metric = tf.keras.metrics.Mean()
    train_accuracy_metric = tf.keras.metrics.BinaryAccuracy()
    val_loss_metric = tf.keras.metrics.Mean()
    val_accuracy_metric = tf.keras.metrics.BinaryAccuracy()

    # 訓練參數
    n_epochs = 10
    patience = 10
    best_val_acc = 0
    stale = 0

    # 保存最佳模型
    checkpoint = tf.train.Checkpoint(model=model)
    manager = tf.train.CheckpointManager(
        checkpoint, directory="./models", max_to_keep=1
    )

    for epoch in range(1, n_epochs + 1):
        print(f"Epoch {epoch}/{n_epochs}")
        # 重置評估指標
        train_loss_metric.reset_states()
        train_accuracy_metric.reset_states()
        val_loss_metric.reset_states()
        val_accuracy_metric.reset_states()

        # 訓練階段
        for batch in train_loader:
            inputs, targets = batch
            x, a, _, i = inputs  # 忽略 e
            inputs = (x, a, i)
            with tf.GradientTape() as tape:
                predictions = model(inputs, training=True)
                loss = loss_fn(targets, predictions)
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            train_loss_metric.update_state(loss)
            train_accuracy_metric.update_state(targets, tf.sigmoid(predictions))

        # 驗證階段
        all_val_targets = []
        all_val_predictions = []
        for batch in val_loader:
            inputs, targets = batch
            x, a, _, i = inputs  # 忽略 e
            inputs = (x, a, i)
            predictions = model(inputs, training=False)
            loss = loss_fn(targets, predictions)
            val_loss_metric.update_state(loss)
            val_accuracy_metric.update_state(targets, tf.sigmoid(predictions))
            all_val_targets.extend(targets.numpy())
            all_val_predictions.extend(tf.sigmoid(predictions).numpy())

        # 計算平均損失和準確率
        train_loss = train_loss_metric.result().numpy()
        train_acc = train_accuracy_metric.result().numpy()
        val_loss = val_loss_metric.result().numpy()
        val_acc = val_accuracy_metric.result().numpy()

        print(f"訓練損失: {train_loss:.4f}, 準確率: {train_acc:.4f}")
        print(f"驗證損失: {val_loss:.4f}, 準確率: {val_acc:.4f}")

        # 保存最佳模型
        if val_acc > best_val_acc:
            print(f"驗證準確率從 {best_val_acc:.4f} 提高到 {val_acc:.4f}，保存模型。")
            best_val_acc = val_acc
            manager.save()
            stale = 0
            best_val_targets = all_val_targets
            best_val_predictions = all_val_predictions
        else:
            stale += 1
            if stale >= patience:
                print(f"{patience} 個 epoch 未提升，停止訓練。")
                break

    # 加載最佳模型
    checkpoint.restore(manager.latest_checkpoint)

    # 測試模型
    test_loss_metric = tf.keras.metrics.Mean()
    test_accuracy_metric = tf.keras.metrics.BinaryAccuracy()

    all_test_targets = []
    all_test_predictions = []

    for batch in test_loader:
        inputs, targets = batch
        x, a, _, i = inputs  # 忽略 e
        inputs = (x, a, i)
        predictions = model(inputs, training=False)
        loss = loss_fn(targets, predictions)
        test_loss_metric.update_state(loss)
        test_accuracy_metric.update_state(targets, tf.sigmoid(predictions))
        all_test_targets.extend(targets.numpy())
        all_test_predictions.extend(tf.sigmoid(predictions).numpy())

    test_loss = test_loss_metric.result().numpy()
    test_acc = test_accuracy_metric.result().numpy()

    print(f"測試損失: {test_loss:.4f}, 準確率: {test_acc:.4f}")

    # 繪製 ROC 曲線
    fpr, tpr, thresholds = roc_curve(all_test_targets, all_test_predictions)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    lw = 2
    plt.plot(
        fpr,
        tpr,
        color="darkorange",
        lw=lw,
        label="ROC 曲線 (AUC = %0.4f)" % roc_auc,
    )
    plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("接收者操作特徵曲線 (ROC)")
    plt.legend(loc="lower right")
    plt.show()
