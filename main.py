import numpy as np
import re
import pandas as pd
from io import StringIO
from PIL import Image, ImageOps
from torchvision import transforms, datasets
from torch.utils.data import Dataset
import os
from torch.utils.data import DataLoader
import torch
from sklearn.preprocessing import MultiLabelBinarizer

def load_data(path):
    with open(path) as file:
        lines = [re.sub(r'([^,])"(\s*[^\n])', r'\1/"\2', line) for line in file]
    df = pd.read_csv(StringIO(''.join(lines)), escapechar="/")
    # print(df)
    return df

def find_max_image_size(folder_path):
    max_width = 0
    max_height = 0

    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            image_path = os.path.join(folder_path, filename)
            try:
                with Image.open(image_path) as img:
                    width, height = img.size
                    if width > max_width:
                        max_width = width
                    if height > max_height:
                        max_height = height
            except IOError:
                print(f"Cannot open {filename}")

    return max_width, max_height

def pad_image_to_square(image, target_size, fill_color='black'):
    # 计算填充量
    width, height = image.size
    max_side = max(width, height)
    left_padding = (max_side - width) // 2
    right_padding = max_side - width - left_padding
    top_padding = (max_side - height) // 2
    bottom_padding = max_side - height - top_padding

    # 创建填充
    padded_image = ImageOps.expand(image, border=(left_padding, top_padding, right_padding, bottom_padding), fill=fill_color)
    # 缩放到目标尺寸
    return padded_image.resize((target_size, target_size))

def process_dataset(input_folder, output_folder, target_size=300):
    # 确保输出文件夹存在
    os.makedirs(output_folder, exist_ok=True)

    # 遍历输入文件夹中的所有文件
    for image_file in os.listdir(input_folder):
        # 构造完整的文件路径
        input_path = os.path.join(input_folder, image_file)
        output_path = os.path.join(output_folder, image_file)

        # 只处理图像文件
        if os.path.isfile(input_path) and input_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            with Image.open(input_path) as img:
                # 填充并缩放图像
                processed_image = pad_image_to_square(img, target_size)
                # 保存处理后的图像
                processed_image.save(output_path)

    # 调用函数，指定输入和输出文件夹路径
    process_dataset('path_to_your_dataset_folder', 'path_to_output_folder')

def calculate_mean_std(loader):
    mean = 0.0
    std = 0.0
    total_images_count = 0

    for images, _ in loader:
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        total_images_count += batch_samples

    mean /= total_images_count
    std /= total_images_count

    return mean, std

def str_to_list(s):
    return list(map(int, s.split()))

class CustomImageDataset(Dataset):
    def __init__(self, img_dir, labels, transform=None):
        """
        Args:
            img_dir (str): 图片存放的目录。
            labels (list of lists): 每个图片对应的标签列表。
            transform (callable, optional): 需要应用到每张图片的变换。
        """
        self.img_dir = img_dir
        self.labels = labels
        self.transform = transform
        self.image_filenames = [f for f in os.listdir(img_dir) if os.path.isfile(os.path.join(img_dir, f))]

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.image_filenames[idx])
        image = Image.open(img_path).convert('RGB')  # 确保为 RGB 格式
        label = torch.tensor(self.labels[idx])  # 假设标签已经是适合的格式
        if self.transform:
            image = self.transform(image)
        return image, label

if __name__ == "__main__":
    TRAIN_FILENAME = 'train.csv'
    TEST_FILENAME = 'test.csv'
    data_path = "fixed_data"
    train_df = load_data(TRAIN_FILENAME)
    test_df = load_data(TEST_FILENAME)

    train_df = train_df[["ImageID", "Labels"]]
    test_df = test_df[["ImageID"]]

    train_df["Labels_list"] = train_df["Labels"].apply(str_to_list)
    # print(train_df.head(5))

    # Get all label names
    unique_numbers = set()
    for num_list in train_df["Labels_list"]:
        unique_numbers.update(num_list)
    labels_list = list(unique_numbers)
    # print(labels_list)

    ## Dataset insights
    # print(f"Train null in Data:\n{train_df.isna().sum()}")
    # print(f"Test null in Data:\n{test_df.isna().sum()}")
    # print(f"Train and test data shape:\n{train_df.shape}, {test_df.shape}")

    # # Resize images and get new dataset
    # width, height = find_max_image_size("data")
    # print("max width:", width, " max height:", height)
    # process_dataset('data', 'fixed_data', target_size=320)

    ## Image process
    # target_size = 256
    # transform_pipeline = transforms.Compose([
    #     transforms.Resize((224, 224)),  # 调整大小到 224x224
    #     transforms.RandomHorizontalFlip(),  # 随机水平翻转
    #     transforms.RandomRotation(15),  # 随机旋转 15 度
    #     transforms.ColorJitter(brightness=0.2, contrast=0.2),  # 随机改变亮度和对比度
    #     transforms.ToTensor(),  # 转换为 PyTorch 张量
    #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 标准化
    # ])

    # Turn to multi-hot coding
    mlb = MultiLabelBinarizer()
    labels_encoded = mlb.fit_transform(train_df['Labels_list'])
    encoded_train_df = pd.DataFrame(labels_encoded, columns=mlb.classes_)
    encoded_train_df['ImageID'] = train_df['ImageID']
    # print(encoded_train_df.head(10))

    # Initialize Dataloader

