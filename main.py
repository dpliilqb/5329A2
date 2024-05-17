from collections import Counter

import numpy as np
import re
import pandas as pd
from io import StringIO
from PIL import Image, ImageOps
from torchvision import transforms, datasets
from torch.utils.data import Dataset
import os
import random
from torch.utils.data import DataLoader
import torch
from sklearn.preprocessing import MultiLabelBinarizer
from Model import Net
import torchmetrics
import copy
from torchmetrics import F1Score
from time import time
import csv

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
    width, height = image.size
    max_side = max(width, height)
    left_padding = (max_side - width) // 2
    right_padding = max_side - width - left_padding
    top_padding = (max_side - height) // 2
    bottom_padding = max_side - height - top_padding

    padded_image = ImageOps.expand(image, border=(left_padding, top_padding, right_padding, bottom_padding), fill=fill_color)
    return padded_image.resize((target_size, target_size))

def process_dataset(input_folder, output_folder, target_size=300):
    os.makedirs(output_folder, exist_ok=True)

    for image_file in os.listdir(input_folder):

        input_path = os.path.join(input_folder, image_file)
        output_path = os.path.join(output_folder, image_file)

        if os.path.isfile(input_path) and input_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            with Image.open(input_path) as img:
                processed_image = pad_image_to_square(img, target_size)
                processed_image.save(output_path)

    # 调用函数，指定输入和输出文件夹路径

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

class ImageTagsDataset(Dataset):
    def __init__(self, dataframe, image_folder, target_size=256):
        self.labels = dataframe.iloc[:, :-1].values
        self.image_ids = dataframe['ImageID'].values
        self.image_folder = image_folder
        self.target_size = target_size
        self.transform = transforms.Compose([
            # transforms.Resize((self.target_size, self.target_size)),
            transforms.RandomResizedCrop(224),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image_path = f"{self.image_folder}/{image_id}"
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)

        labels = torch.tensor(self.labels[idx], dtype=torch.float32)
        return image, labels

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=25, device='cpu'):
    model = model.to(device)
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = float('inf')

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        # 每个 epoch 都有训练和验证阶段
        for phase in ['train', 'val']:
            if phase == 'train':
                # model.train()  # 训练模式
                dataloader = train_loader
            else:
                # model.eval()  # 验证模式
                dataloader = val_loader

            running_loss = 0.0
            all_labels = []
            all_preds = []

            # 遍历数据
            for inputs, labels in dataloader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # 每次迭代都要将梯度归零
                optimizer.zero_grad()

                # 前向传播
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    # 反向传播 + 优化（仅在训练阶段）
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # 统计损失
                running_loss += loss.item() * inputs.size(0)

                # 保存预测结果和真实标签
                preds = torch.sigmoid(outputs) > 0.5
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())

            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_f1 = f1_score(np.array(all_labels), np.array(all_preds), average='micro')

            print(f'{phase} Loss: {epoch_loss:.4f} F1: {epoch_f1:.4f}')

            # 复制模型的最佳权重
            if phase == 'val' and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())

    print('Training complete')
    print(f'Best val loss: {best_loss:4f}')

    return model, best_model_wts
if __name__ == "__main__":
    TRAIN_FILENAME = 'train.csv'
    TEST_FILENAME = 'test.csv'
    data_path = "fixed_data"
    train_df = load_data(TRAIN_FILENAME)
    test_df = load_data(TEST_FILENAME)

    train_df = train_df[["ImageID", "Labels"]]
    test_df = test_df[["ImageID"]]

    train_df["Labels_list"] = train_df["Labels"].apply(str_to_list)
    # print(train_df["Labels_list"])
    # print(train_df.head(5))

    # 输出类别的统计
    # all_categories = [category for sublist in train_df['Labels_list'] for category in sublist]
    # category_counts = Counter(all_categories)
    #
    # # 将结果转换为dataframe并显示
    # counts_df = pd.DataFrame(category_counts.items(), columns=['Category', 'Count'])
    # print(counts_df)

    all_categories = [category for sublist in train_df['Labels_list'] for category in sublist]
    # 统计每个类别出现的次数
    category_counts = pd.Series(all_categories).value_counts()
    # 找到占比最多的类别
    max_category = category_counts.idxmax()
    max_count = category_counts.max()
    # 计算需要保留的样本数量（这里取所有类别的平均值）
    mean_count = int(category_counts.mean())
    # 找到包含最多类别的行
    max_category_rows = train_df[train_df['Labels_list'].apply(lambda x: max_category in x)]
    # 随机选择需要保留的样本
    random_indices = random.sample(list(max_category_rows.index), 2*mean_count)
    undersampled_max_category_rows = train_df.loc[random_indices]
    # 找到其余的行
    other_rows = train_df[~train_df.index.isin(max_category_rows.index)]
    # 合并平衡后的dataframe
    train_df = pd.concat([undersampled_max_category_rows, other_rows])
    train_df = train_df.reset_index(drop=True)
    # print(train_df.head(3))
    all_categories = [category for sublist in train_df['Labels_list'] for category in sublist]
    category_counts = Counter(all_categories)

    # 将结果转换为dataframe并显示
    counts_df = pd.DataFrame(category_counts.items(), columns=['Category', 'Count'])
    print(counts_df)


    # Get all label names
    unique_numbers = set()
    for num_list in train_df["Labels_list"]:
        unique_numbers.update(num_list)
    labels_list = list(unique_numbers)
    # print(labels_list)

    # Labels insights
    # all_categories = set()
    # train_df["Labels_list"].apply(lambda x: all_categories.update(x))
    # all_categories = sorted(list(all_categories))
    # print("All Classes:", all_categories)
    # [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19]

    ## Dataset insights
    # print(f"Train null in Data:\n{train_df.isna().sum()}")
    # print(f"Test null in Data:\n{test_df.isna().sum()}")
    # print(f"Train and test data shape:\n{train_df.shape}, {test_df.shape}")

    # # Resize images and get new dataset
    # width, height = find_max_image_size("data")
    # print("max width:", width, " max height:", height)
    # process_dataset('data', 'fixed_data', target_size=320)


    # Turn to multi-hot coding
    mlb = MultiLabelBinarizer()
    labels_encoded = mlb.fit_transform(train_df['Labels_list'])
    encoded_train_df = pd.DataFrame(labels_encoded, columns=mlb.classes_)
    encoded_train_df['ImageID'] = train_df['ImageID']
    # print(encoded_train_df.head(10))

    # Initialize Dataloader
    dataset = ImageTagsDataset(encoded_train_df, "fixed_data")

    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)
    # print(dataset.__getitem__(1))
    # print(torch.cuda.is_available() )
    num_classes = 18
    threshold = 0.8
    model = Net(in_channels=3, num_classes=num_classes, threshold=threshold, learning_rate=0.0001, optimizer='Adam')
    model.to(model.device)
    max_F1 = 0.0
    model.init_weights()
    epochs = 50
    f1_score = F1Score(task="multiclass", num_classes=num_classes, average='macro')
    start_time = time()
    for epoch in range(epochs):
        epoch_start_time = time()
        best_batch_loss = 0.0
        epoch_F1 = 0.0
        all_labels = []
        all_preds = []
        running_loss = 0.0
        for image, label in dataloader:
            model.optimizer.zero_grad()

            image, label = image.to(model.device), label.to(model.device)
            outputs = model.forward(image)
            batch_loss = model.loss(outputs, label)
            batch_loss.backward()
            model.optimizer.step()
            # preds = outputs.to(model.device)
            # print("Training pred", preds)
            # print("Training label:", label)
            running_loss += batch_loss.item() * image.size(0)
            # running_loss += batch_loss.item()
            preds = model.sigmoid(outputs)
            preds = preds > threshold
            # print("train pred:", preds.cpu().numpy())
            all_labels.append(label.cpu())
            all_preds.append(preds.cpu())
            # print("Batch loss: ", batch_loss.item())
        epoch_end_time = time()
        all_labels = torch.cat(all_labels, dim=0)
        all_preds = torch.cat(all_preds, dim=0)
        epoch_loss = running_loss / len(dataloader.dataset)
        # print("Epoch loss: ", epoch_loss)
        f1_score.update(all_preds, all_labels)
        epoch_f1 = f1_score.compute()
        # print("Epoch F1: ", epoch_f1)

        # label = label.to(model.device)
        print(f'Epoch: {epoch+1} Train Loss: {epoch_loss:.4f} F1: {epoch_f1.item():.4f} Cost Time: {epoch_end_time - epoch_start_time:.4f}')
    end_time = time()
    print(f'Total Time: {end_time - start_time:.4f}')
    torch.save(model.state_dict(), 'Models/model_1.pth')

