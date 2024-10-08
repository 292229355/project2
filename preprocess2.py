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
import cv2
import dlib
from PIL import Image
from sklearn.preprocessing import StandardScaler
from transformers import AutoImageProcessor, SuperPointForKeypointDetection
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
processor = AutoImageProcessor.from_pretrained("magic-leap-community/superpoint")
superpoint_model = SuperPointForKeypointDetection.from_pretrained(
    "magic-leap-community/superpoint"
)
superpoint_model.eval()
superpoint_model.to(device)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")


def extract_descriptors_and_build_graph(
    image_pth, max_num_nodes=500, distance_threshold=100, feature_dim=128
):
    img = cv2.imread(image_pth)
    if img is None:
        print(f"Image at path {image_pth} not found.")
        # 返回一个默认的零向量图
        x = torch.zeros((1, feature_dim), dtype=torch.float)
        edge_index = torch.empty((2, 0), dtype=torch.long)
        y = torch.zeros((1, 2), dtype=torch.float)
        edge_attr = torch.empty((0, 1), dtype=torch.float)
        return x, edge_index, y, edge_attr

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = detector(gray)
    if len(faces) == 0:
        print(f"No faces detected in image {image_pth}.")
        # 返回一个默认的零向量图
        x = torch.zeros((1, feature_dim), dtype=torch.float)
        edge_index = torch.empty((2, 0), dtype=torch.long)
        y = torch.zeros((1, 2), dtype=torch.float)
        edge_attr = torch.empty((0, 1), dtype=torch.float)
        return x, edge_index, y, edge_attr

    # 检测人脸
    face = faces[0]
    landmarks = predictor(gray, face)
    points = np.array(
        [(landmarks.part(n).x, landmarks.part(n).y) for n in range(68)],
        dtype=np.int32,
    )

    # 创建人脸蒙版
    mask = np.zeros_like(gray)
    cv2.fillConvexPoly(mask, points, 255)
    face_img = cv2.bitwise_and(gray, gray, mask=mask)

    # 使用 SIFT 检测关键点和描述符
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(face_img, None)

    if descriptors is None or len(descriptors) == 0:
        print(f"No descriptors extracted from image {image_pth}.")
        # 返回一个默认的零向量图
        x = torch.zeros((1, feature_dim), dtype=torch.float)
        edge_index = torch.empty((2, 0), dtype=torch.long)
        y = torch.zeros((1, 2), dtype=torch.float)
        edge_attr = torch.empty((0, 1), dtype=torch.float)
        return x, edge_index, y, edge_attr

    keypoint_coords = np.array([kp.pt for kp in keypoints], dtype=np.float32)

    # 标准化描述符
    scaler = StandardScaler()
    descriptors = scaler.fit_transform(descriptors)

    # 限制节点数量（max_num_nodes）
    if descriptors.shape[0] > max_num_nodes:
        indices = np.random.choice(descriptors.shape[0], max_num_nodes, replace=False)
        descriptors = descriptors[indices]
        keypoint_coords = keypoint_coords[indices]

    num_nodes = descriptors.shape[0]

    # 构建边索引
    edge_index = []
    edge_attr = []  # 存储边的特征（如距离）
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            distance = np.linalg.norm(keypoint_coords[i] - keypoint_coords[j])
            if distance < distance_threshold:
                edge_index.append([i, j])
                edge_index.append([j, i])  # 无向图
                edge_attr.append([1 / (distance + 1e-5)])  # 边的权重，距离越近权重越大
                edge_attr.append([1 / (distance + 1e-5)])

    if len(edge_index) == 0:
        print(f"No edges formed in graph for image {image_pth}.")
        # 返回一个默认的零向量图
        x = torch.zeros((1, feature_dim), dtype=torch.float)
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 1), dtype=torch.float)
        y = torch.zeros((1, 2), dtype=torch.float)
        return x, edge_index, y, edge_attr

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)

    x = torch.from_numpy(descriptors).float()
    y = torch.from_numpy(keypoint_coords).float()  # 关键点坐标

    return (
        x,
        edge_index,
        y,
        edge_attr,
    )  # 返回节点特征 x，边索引 edge_index，关键点坐标 y，边特征 edge_attr


def extract_descriptors_and_build_graph2(
    img_pth,
    max_num_nodes=500,  # 限制節點數
    distance_threshold=10,  # 節點之間連接邊的距離
    feature_dim=256,  # SuperPoint 的descriptor維度256
):
    img = cv2.imread(img_pth)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = detector(gray)
    # 已經篩選過資料(人臉偵測)，不需要再返回0向量
    """''
    if len(faces) == 0:
        # print(f"No faces detected in image {image_pth}.")
        # 偵測不到人臉就返回零向量
        x = torch.zeros((1, feature_dim), dtype=torch.float)
        edge_index = torch.empty((2, 0), dtype=torch.long)
        y = torch.zeros((1, 2), dtype=torch.float)
        edge_attr = torch.empty((0, 1), dtype=torch.float)
        return x, edge_index, y, edge_attr
    """ ""

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


# Initialize dlib's face detector and shape predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(
    "shape_predictor_68_face_landmarks.dat"
)  # Ensure the path is correct


def extract_descriptors_and_build_graph3(
    image_pth,
    max_num_nodes=500,
    distance_threshold=10,
    feature_dim=256,  # SuperPoint descriptor dimension
    processor=None,  # Initialize your SuperPoint processor
    superpoint_model=None,  # Initialize your SuperPoint model
    device="cpu",  # Ensure correct device
):
    img = cv2.imread(image_pth)
    if img is None:
        print(f"Image at path {image_pth} not found.")
        # Return zero vectors
        x = torch.zeros((1, feature_dim), dtype=torch.float)
        edge_index = torch.empty((2, 0), dtype=torch.long)
        y = torch.zeros((1, 2), dtype=torch.float)
        edge_attr = torch.empty((0, 1), dtype=torch.float)
        return x, edge_index, y, edge_attr

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = detector(gray)
    if len(faces) == 0:
        # No faces detected, return zero vectors
        x = torch.zeros((1, feature_dim), dtype=torch.float)
        edge_index = torch.empty((2, 0), dtype=torch.long)
        y = torch.zeros((1, 2), dtype=torch.float)
        edge_attr = torch.empty((0, 1), dtype=torch.float)
        return x, edge_index, y, edge_attr

    face = faces[0]
    landmarks = predictor(gray, face)
    points = np.array(
        [(landmarks.part(n).x, landmarks.part(n).y) for n in range(68)],
        dtype=np.int32,
    )

    # Define regions based on landmarks
    regions = {
        "left_eye": points[36:42],
        "right_eye": points[42:48],
        "nose": points[27:36],
        "mouth": points[48:68],
        "left_eyebrow": points[17:22],
        "right_eyebrow": points[22:27],
        "left_cheek": points[0:17],  # Approximation
        "right_cheek": points[17:27],  # Approximation
    }

    descriptors_list = []
    keypoints_list = []
    region_tags = []

    for region_name, region_points in regions.items():
        # Create mask for the region
        mask = np.zeros_like(gray)
        cv2.fillConvexPoly(mask, region_points, 255)
        region_img = cv2.bitwise_and(gray, gray, mask=mask)

        # Convert to RGB for SuperPoint
        region_img_pil = Image.fromarray(region_img).convert("RGB")

        # Perform SuperPoint feature extraction
        inputs = processor(region_img_pil, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = superpoint_model(**inputs)

        image_mask = outputs.mask[0]
        image_indices = torch.nonzero(image_mask).squeeze()
        if image_indices.numel() == 0:
            continue  # No keypoints in this region

        # Get keypoints and descriptors
        keypoints = outputs.keypoints[0][image_indices]
        descriptors = outputs.descriptors[0][image_indices]

        keypoint_coords = keypoints.cpu().numpy()
        descriptors = descriptors.cpu().numpy()

        # Adjust keypoint coordinates relative to the whole face
        # Since we're processing regions separately, we need to map them back
        # to the original image coordinates. This step assumes that the
        # processor doesn't alter the image size. Adjust if necessary.
        # If processor resizes images, you'll need to scale keypoints accordingly.

        # Normalize descriptors
        scaler = StandardScaler()
        descriptors = scaler.fit_transform(descriptors)

        descriptors_list.append(descriptors)
        keypoints_list.append(keypoint_coords)
        # Tag each descriptor with its region (optional)
        region_tags.extend([region_name] * descriptors.shape[0])

    if not descriptors_list:
        print(f"No keypoints detected in any region for image {image_pth}.")
        # Return zero vectors
        x = torch.zeros((1, feature_dim), dtype=torch.float)
        edge_index = torch.empty((2, 0), dtype=torch.long)
        y = torch.zeros((1, 2), dtype=torch.float)
        edge_attr = torch.empty((0, 1), dtype=torch.float)
        return x, edge_index, y, edge_attr

    # Concatenate all regions' descriptors and keypoints
    descriptors = np.vstack(descriptors_list)
    keypoint_coords = np.vstack(keypoints_list)

    # Limit the number of nodes
    if descriptors.shape[0] > max_num_nodes:
        indices = np.random.choice(descriptors.shape[0], max_num_nodes, replace=False)
        descriptors = descriptors[indices]
        keypoint_coords = keypoint_coords[indices]
        if region_tags:
            region_tags = [region_tags[i] for i in indices]

    num_nodes = descriptors.shape[0]

    # Build edge index based on distance threshold
    edge_index = []
    edge_attr = []  # Edge features (e.g., inverse distance)
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            distance = np.linalg.norm(keypoint_coords[i] - keypoint_coords[j])
            if distance < distance_threshold:
                edge_index.append([i, j])
                edge_index.append([j, i])
                edge_attr.append([1 / (distance + 1e-5)])
                edge_attr.append([1 / (distance + 1e-5)])

    if len(edge_index) == 0:
        print(f"No edges formed in graph for image {image_pth}.")
        # Return a default zero vector graph
        x = torch.zeros((1, feature_dim), dtype=torch.float)
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 1), dtype=torch.float)
        y = torch.zeros((1, 2), dtype=torch.float)
        return x, edge_index, y, edge_attr

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)

    x = torch.from_numpy(descriptors).float()
    y = torch.from_numpy(keypoint_coords).float()  # Keypoint coordinates

    return (
        x,
        edge_index,
        y,
        edge_attr,
    )  # Returns node features x, edge indices edge_index, keypoint coordinates y, edge features edge_attr


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
        x, edge_index, pos, edge_attr = data

        try:
            file_parts = os.path.basename(fname).split("_")
            label_str = file_parts[1].split(".")[0]
            label = int(label_str)
        except:
            label = -1

        # 创建PyTorch Geometric的資料格式
        graph_data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=torch.tensor([label], dtype=torch.float),
            pos=pos,
        )

        return graph_data
