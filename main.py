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
from Model import Net
import torchmetrics

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

class ImageTagsDataset(Dataset):
    def __init__(self, dataframe, image_folder, target_size = 256):
        self.labels = dataframe.iloc[:, :-1].values
        self.image_ids = dataframe['ImageID'].values
        self.image_folder = image_folder
        self.target_size = target_size
        self.transform = transforms.Compose([
            transforms.Resize((self.target_size, self.target_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
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

    ## Image process
    # target_size = 256
    # transform_pipeline = transforms.Compose([
    #     transforms.Resize((224, 224)),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.RandomRotation(15),
    #     transforms.ColorJitter(brightness=0.2, contrast=0.2),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    # ])
    # train_df["Image_data"] = transform_pipeline(train_df["ImageID"])
    # print(train_df.head(3))

    # Turn to multi-hot coding
    mlb = MultiLabelBinarizer()
    labels_encoded = mlb.fit_transform(train_df['Labels_list'])
    encoded_train_df = pd.DataFrame(labels_encoded, columns=mlb.classes_)
    encoded_train_df['ImageID'] = train_df['ImageID']
    # print(encoded_train_df.head(10))

    # Initialize Dataloader
    dataset = ImageTagsDataset(encoded_train_df, "fixed_data")
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=2)
    # print(dataset.__getitem__(1))
    print(torch.cuda.is_available() )
    model = Net()
    model.to(model.device)
    max_F1 = 0.0
    # model.init_weights()
    epochs = 1
    f1_score = torchmetrics.F1Score(task="multiclass" , num_classes=18, average='macro').to(model.device)

    for epoch in range(epochs):
        for image, label in dataloader:
            image, label = image.to(model.device), label.to(model.device)
            loss, pred = model.train_step(image, label)

            pred = pred.to(model.device)
            label = label.to(model.device)
            f1 = f1_score(pred, label)
            if f1 > max_F1:
                max_F1 = f1
            print(f"loss: {loss.item():.4f}; Batch F1 Score: {f1:.4f}; Highest F1 Score: {max_F1:.4f}")

    idx_to_class = {0: '1', 1: '2', 2: '3', 3: '4', 4: '5', 5: '6', 6: '7', 7: '8', 8: '9', 9: '10', 10: '11', 11: '13', 12: '14', 13: '15', 14: '16', 15: '17', 16: '18', 17: '19'}
    test_set = ImageTagsDataset(test_df, "fixed_data")
    test_dataloader = DataLoader(test_set, batch_size=32, shuffle=True, num_workers=2)
    for image, label in test_dataloader:
        image, label = image.to(model.device), label.to(model.device)
        loss, pred = model.test_step(image, label)
        predicted_labels = pred.argmax(dim=1)
        predicted_class_names = [idx_to_class[label.item()] for label in predicted_labels]
        print(predicted_class_names)
