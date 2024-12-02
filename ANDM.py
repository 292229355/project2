import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
import cv2
import dlib
from PIL import Image
from sklearn.preprocessing import StandardScaler
import os
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from transformers import (
    AutoImageProcessor,
    SuperPointForKeypointDetection,
)
import tensorflow as tf

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
processor = AutoImageProcessor.from_pretrained("magic-leap-community/superpoint")
superpoint_model = SuperPointForKeypointDetection.from_pretrained(
    "magic-leap-community/superpoint"
)
superpoint_model.eval()
superpoint_model.to(device)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")


def extract_descriptors_and_build_graph2(
    img_pth,
    max_num_nodes=500,  # 限制節點數
    feature_dim=256,  # SuperPoint 的descriptor維度256
):
    img = cv2.imread(img_pth)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = detector(gray)
    face = faces[0]
    landmarks = predictor(gray, face)
    # 使用dlib提取人臉68個關鍵點
    points = np.array(
        [(landmarks.part(n).x, landmarks.part(n).y) for n in range(68)],
        dtype=np.int32,
    )

    mask = np.zeros_like(gray)
    cv2.fillConvexPoly(mask, points, 255)
    face_img = cv2.bitwise_and(gray, gray, mask=mask)

    # 將人臉改回RGB
    face_img_pil = Image.fromarray(face_img).convert("RGB")

    # 使用 SuperPoint 進行特徵提取
    inputs = processor(face_img_pil, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = superpoint_model(**inputs)

    image_mask = outputs.mask[0]
    image_indices = torch.nonzero(image_mask).squeeze()
    if image_indices.numel() == 0:
        print(f"No keypoints detected in image {img_pth}.")
        # 如果偵測不到一樣回傳0向量
        x = torch.zeros((1, feature_dim), dtype=torch.float)
        edge_index = torch.empty((2, 0), dtype=torch.long)
        y = torch.zeros((1, 2), dtype=torch.float)
        edge_attr = torch.empty((0, 1), dtype=torch.float)
        return x, edge_index, y, edge_attr

    # 獲取關鍵點座標
    keypoints = outputs.keypoints[0][image_indices]
    descriptors = outputs.descriptors[0][image_indices]

    keypoint_coords = keypoints.cpu().numpy()
    descriptors = descriptors.cpu().numpy()

    # 標準化描述子
    scaler = StandardScaler()
    descriptors = scaler.fit_transform(descriptors)

    # 限制節點數量
    if descriptors.shape[0] > max_num_nodes:
        indices = np.random.choice(descriptors.shape[0], max_num_nodes, replace=False)
        descriptors = descriptors[indices]
        keypoint_coords = keypoint_coords[indices]

    num_nodes = descriptors.shape[0]

    # 邊索引和邊特徵
    edge_index = []
    edge_attr = []
    """
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            # 這裡之後可以再做修改，不用distance衡量
            distance = np.linalg.norm(keypoint_coords[i] - keypoint_coords[j])
            if distance < distance_threshold:
                edge_index.append([i, j])
                edge_index.append([j, i])
                edge_attr.append([1 / (distance + 1e-5)])
                edge_attr.append([1 / (distance + 1e-5)])

    # 以防沒有一個點鄰近
    if len(edge_index) == 0:
        print(f"No edges formed in graph for image {img_pth}.")
        x = torch.zeros((1, feature_dim), dtype=torch.float)
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 1), dtype=torch.float)
        y = torch.zeros((1, 2), dtype=torch.float)
        return x, edge_index, y, edge_attr
    """
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)

    x = torch.from_numpy(descriptors).float()
    y = torch.from_numpy(keypoint_coords).float()  # 關鍵點座標

    return (
        x,
        edge_index,
        y,
        edge_attr,
    )  # 節點特徵 x、邊索引 edge_index、關鍵點座標 y 和邊特徵 edge_attr


def stable_sigmoid(x):
    """
    Stable sigmoid function to handle potential overflow.
    """
    sig = np.where(x < 0, np.exp(x) / (1 + np.exp(x)), 1 / (1 + np.exp(-x)))
    return sig


def scm(des1, des2):
    """
    Similarity Calculation Module (SCM) for calculating similarity between two descriptors.
    """
    dotproduct = np.dot(des1, des2) / (
        np.linalg.norm(des1) * np.linalg.norm(des2)
    )  # Cosine similarity
    x = dotproduct / (np.linalg.norm(des1) ** 0.5)
    similarity = stable_sigmoid(x)
    return similarity


def andm(A, gamma, beta):
    """
    Adaptive Neighbour Discovery Module (ANDM) for pruning and thresholding the adjacency matrix.
    """
    n = A.shape[0]

    # Calculate the adaptive threshold Ti for each node
    mean_A = np.mean(A, axis=1, keepdims=True)
    Ti = gamma * mean_A + beta

    # Apply the threshold to prune the edges
    AT = np.where(A > Ti, A, 0)

    # Normalize the adjacency matrix
    AN = np.zeros_like(AT)
    for i in range(n):
        neighbors = np.where(AT[i] > 0)[0]
        if len(neighbors) > 0:
            AN[i, neighbors] = np.exp(AT[i, neighbors]) / np.sum(
                np.exp(AT[i, neighbors])
            )

    return AN


def calculate_adjacency_matrix_with_andm(X, eta=0.5, dk=64):
    """
    Calculate and optimize adjacency matrix using SCM, SAM, and ANDM.
    """
    n, d = X.shape

    # SCM: Calculate similarity matrix
    Msim = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            Msim[i, j] = scm(X[i], X[j])

    # SAM: Self-attention mechanism using TensorFlow
    X_tensor = tf.convert_to_tensor(X, dtype=tf.float32)
    Q = tf.keras.layers.Dense(dk)(X_tensor)  # Query
    S = tf.keras.layers.Dense(dk)(X_tensor)  # Key
    O = tf.keras.layers.Dense(n)(X_tensor)  # Value
    attention_scores = tf.nn.softmax(
        tf.matmul(Q, S, transpose_b=True) / tf.sqrt(float(dk))
    )
    Matt = tf.sigmoid(tf.matmul(attention_scores, O)).numpy()

    # Combine SCM and SAM
    A = eta * Msim + (1 - eta) * Matt

    # Define learnable parameters for ANDM
    gamma = np.random.rand(n, 1)  # Placeholder for learnable gamma
    beta = np.random.rand(n, 1)  # Placeholder for learnable beta

    # Apply ANDM
    AN = andm(A, gamma, beta)

    return AN


def extract_descriptors_and_build_graph_with_andm(
    img_pth, max_num_nodes=500, feature_dim=256, eta=0.5, dk=64
):
    """
    Extract features, build graph, and calculate adjacency matrix with ANDM.
    """
    # Step 1: Extract node features using the provided function
    x, edge_index, y, edge_attr = extract_descriptors_and_build_graph2(
        img_pth, max_num_nodes=max_num_nodes, feature_dim=feature_dim
    )

    # If no nodes are detected, return empty results
    if x.size(0) == 0:
        print("No nodes detected, returning zero adjacency matrix.")
        return x, edge_index, y, edge_attr, np.zeros((1, 1))

    # Convert node features to NumPy array
    feature_matrix = x.numpy()

    # Step 2: Calculate adjacency matrix using ANDM
    adjacency_matrix = calculate_adjacency_matrix_with_andm(
        feature_matrix, eta=eta, dk=dk
    )

    return x, edge_index, y, edge_attr, adjacency_matrix


class DescriptorGraphDataset(Dataset):
    def __init__(
        self, path, mode="train", max_num_nodes=500, feature_dim=256, eta=0.5, dk=64
    ):
        """
        Dataset for extracting graph-based descriptors from images.

        Args:
            path (str): Path to the directory containing image files.
            mode (str): Dataset mode ('train', 'val', or 'test').
            max_num_nodes (int): Maximum number of nodes in the graph.
            feature_dim (int): Dimensionality of the feature descriptors.
            eta (float): Balance parameter for combining SCM and SAM.
            dk (int): Dimensionality of attention keys in SAM.
        """
        super(DescriptorGraphDataset, self).__init__()
        self.path = path
        self.mode = mode
        self.max_num_nodes = max_num_nodes
        self.feature_dim = feature_dim
        self.eta = eta
        self.dk = dk

        # Get list of image files
        self.files = sorted(
            [
                os.path.join(path, x)
                for x in os.listdir(path)
                if x.endswith(".jpg") or x.endswith(".png")
            ]
        )

        if len(self.files) == 0:
            raise FileNotFoundError(f"No image files found in {path}.")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]

        # Extract graph data and adjacency matrix
        data = extract_descriptors_and_build_graph_with_andm(
            fname,
            max_num_nodes=self.max_num_nodes,
            feature_dim=self.feature_dim,
            eta=self.eta,
            dk=self.dk,
        )

        # Unpack the data
        x, edge_index, pos, edge_attr, adjacency_matrix = data

        # Determine the label from the filename
        try:
            file_parts = os.path.basename(fname).split("_")
            label_str = file_parts[1].split(".")[0]
            label = int(label_str)
        except:
            label = -1  # Use -1 for missing or invalid labels

        # Convert adjacency matrix to PyTorch Geometric edge index and edge attributes
        if adjacency_matrix.shape[0] > 1:  # Ensure adjacency matrix is not empty
            adj_indices = torch.nonzero(
                torch.tensor(adjacency_matrix), as_tuple=False
            ).t()
            adj_weights = torch.tensor(
                adjacency_matrix[adj_indices[0], adj_indices[1]], dtype=torch.float
            )
        else:
            adj_indices = torch.empty((2, 0), dtype=torch.long)
            adj_weights = torch.empty((0,), dtype=torch.float)

        # Create PyTorch Geometric data object
        graph_data = Data(
            x=x,
            edge_index=adj_indices,
            edge_attr=adj_weights,
            y=torch.tensor([label], dtype=torch.float),
            pos=pos,
        )

        return graph_data
