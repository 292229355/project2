from sklearn.model_selection import train_test_split
import os
import shutil
import dlib
import cv2

# dlib 的人臉偵測器和 shape_predictor
detector = dlib.get_frontal_face_detector()
# predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")


# 偵測圖片中是否包含人臉的函數
def detect_face(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return False
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    return len(faces) > 0  # 若偵測到人臉，返回 True


# 真實和生成眼睛圖片的目錄路徑
real_eyes_dir = "dataset/Real"
fake_eyes_dir = "dataset/ADM"

# 列出目錄中的所有檔案
all_files = os.listdir(fake_eyes_dir)
all_files2 = os.listdir(real_eyes_dir)

# 建立包含偵測到人臉的圖片路徑
fake_images = [
    os.path.join(fake_eyes_dir, file)
    for file in all_files
    if detect_face(os.path.join(fake_eyes_dir, file))
]
real_images = [
    os.path.join(real_eyes_dir, file)
    for file in all_files2
    if detect_face(os.path.join(real_eyes_dir, file))
]

# 為圖片標記標籤，真實圖片標記為0，生成圖片標記為1
real_labels = [0] * len(real_images)
fake_labels = [1] * len(fake_images)

# 合併圖片路徑與標籤
all_images = real_images + fake_images
all_labels = real_labels + fake_labels

# 將數據集切分為訓練集、測試集與驗證集
train_images, test_val_images, train_labels, test_val_labels = train_test_split(
    all_images, all_labels, test_size=0.2, random_state=42
)
val_images, test_images, val_labels, test_labels = train_test_split(
    test_val_images, test_val_labels, test_size=0.5, random_state=42
)

# 定義主目錄路徑，將所有數據集存入這個目錄中
main_dir = "ADM_dataset"  # 主目錄名稱
train_dir = os.path.join(main_dir, "train")
val_dir = os.path.join(main_dir, "valid")
test_dir = os.path.join(main_dir, "test")

# 如果主目錄和子目錄不存在則創建
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)


# 將圖片移動到對應的目錄
def move_images_to_dir(image_paths, labels, destination_dir):
    for i, (image_path, label) in enumerate(zip(image_paths, labels)):
        image_name = os.path.basename(image_path)
        new_image_name = (
            f"{i}_{label}{os.path.splitext(image_name)[1]}"  # 根據標籤和索引重命名
        )
        destination = os.path.join(destination_dir, new_image_name)
        shutil.copy(image_path, destination)  # 使用shutil.move進行移動，也可以選擇複製


# 移動訓練集圖片
move_images_to_dir(train_images, train_labels, train_dir)

# 移動驗證集圖片
move_images_to_dir(val_images, val_labels, val_dir)

# 移動測試集圖片
move_images_to_dir(test_images, test_labels, test_dir)

# 列印訓練集、驗證集和測試集的大小
print(f"訓練集大小: {len(train_images)}")
print(f"驗證集大小: {len(val_images)}")
print(f"測試集大小: {len(test_images)}")
