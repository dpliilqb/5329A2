import re
from io import StringIO

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MultiLabelBinarizer
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms, models
from torchvision.models import resnet18, ResNet18_Weights, ResNet50_Weights
from PIL import Image
import pandas as pd
import pickle
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.nn.utils.rnn import pad_sequence
from time import time
from torchmetrics.classification import MultilabelAccuracy, MultilabelPrecision, MultilabelRecall, MultilabelF1Score
from sklearn.model_selection import train_test_split
import random
from collections import Counter
import torch.nn.functional as F
from transformers import BertTokenizer, BertModel

class CustomCNN(nn.Module):
    def __init__(self):
        super(CustomCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(64 * 56 * 56, 512)  # 假设输入图像大小为224x224

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 56 * 56)
        x = F.relu(self.fc1(x))
        return x

class MultiLabelDataset(Dataset):
    def __init__(self, dataframe, vocab, transform=None, max_length = 50):
        self.dataframe = dataframe
        self.vocab = vocab
        self.transform = transform
        self.tokenizer = get_tokenizer("basic_english")
        self.max_length = max_length
        self.is_test = 0

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_path = "fixed_data/" + self.dataframe.iloc[idx]['ImageID']
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        if "Labels" in self.dataframe.columns:
            labels = torch.tensor(self.dataframe.iloc[idx]['Labels'], dtype=torch.float32)
        else:
            self.is_test = 1
        # print("Labels: ", labels)
        description = self.dataframe.iloc[idx]['Caption']
        tokens = self.tokenizer(description)
        text_indices = [self.vocab[token] for token in tokens]

        if len(text_indices) > self.max_length:
            text_indices = text_indices[:self.max_length]
        else:
            text_indices = text_indices + [self.vocab['<pad>']] * (self.max_length - len(text_indices))
        text_indices = torch.tensor(text_indices, dtype=torch.long)

        if self.is_test:
            return image, text_indices
        else:
            return image, labels, text_indices

class TextEmbeddingModel(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super(TextEmbeddingModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, embed_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(embed_dim * 2, embed_dim)

    def forward(self, x):
        x = self.embedding(x)
        _, (hidden, _) = self.lstm(x)
        hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)  # 双向LSTM的输出拼接
        x = self.fc(hidden)
        return x

class MultiLabelModel(nn.Module):
    def __init__(self, num_classes, vocab_size, embed_dim):
        super(MultiLabelModel, self).__init__()
        self.cnn = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        # self.cnn = CustomCNN()
        self.cnn.fc = nn.Identity()  # 去掉最后一层全连接层
        self.text_model = TextEmbeddingModel(vocab_size, embed_dim)
        self.fc1 = nn.Linear(512 + embed_dim, 256)  # 512 是 ResNet18 的输出维度, 50是2048
        self.fc2 = nn.Linear(256, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, image, text):
        image_features = self.cnn(image)
        text_features = self.text_model(text)
        combined_features = torch.cat((image_features, text_features), dim=1)
        x = self.dropout(combined_features)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

def yield_tokens(data_iter):
    tokenizer = get_tokenizer("basic_english")
    for text in data_iter['Caption']:
        yield tokenizer(text)

def load_data(path):
    with open(path) as file:
        lines = [re.sub(r'([^,])"(\s*[^\n])', r'\1/"\2', line) for line in file]
    df = pd.read_csv(StringIO(''.join(lines)), escapechar="/")
    # print(df)
    return df

def str_to_list(s):
    return list(map(int, s.split()))

# def collate_fn(batch):
#     images, labels, descriptions = zip(*batch)
#     descriptions = pad_sequence(descriptions, batch_first=True, padding_value=0)
#     return torch.stack(images), torch.stack(labels), descriptions

if __name__ == '__main__':
    # 数据准备
    raw_dataframe = load_data("train.csv")
    test_data = load_data("test.csv")
    raw_dataframe["Labels"] = raw_dataframe["Labels"].apply(str_to_list)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tokenizer = get_tokenizer("basic_english")
    # 计算每个描述文本的标记序列长度
    train_length = raw_dataframe['Caption'].apply(lambda desc: len(tokenizer(desc)))
    test_length = test_data['Caption'].apply(lambda desc: len(tokenizer(desc)))
    max_length = max(train_length.max(), test_length.max())
    print("Max Length:", max_length)
    print(raw_dataframe.head(3))
    # 创建词汇表
    text_colum = pd.concat([raw_dataframe['Caption'], test_data['Caption']]).drop_duplicates().reset_index(drop=True)
    text_colum = pd.DataFrame(text_colum, columns=['Caption'])
    vocab = build_vocab_from_iterator(yield_tokens(text_colum), specials=["<unk>"])
    # print("Vocabulary Size:", len(vocab))
    vocab.set_default_index(vocab["<unk>"])
    with open('vocab.pkl', 'wb') as f:
        pickle.dump(vocab, f)

    all_categories = [category for sublist in raw_dataframe['Labels'] for category in sublist]
    # 统计每个类别出现的次数
    category_counts = pd.Series(all_categories).value_counts()
    # 找到占比最多的类别
    max_category = category_counts.idxmax()
    max_count = category_counts.max()
    # 计算需要保留的样本数量（这里取所有类别的平均值）
    mean_count = int(category_counts.mean())
    # 找到包含最多类别的行
    max_category_rows = raw_dataframe[raw_dataframe['Labels'].apply(lambda x: max_category in x)]
    # 随机选择需要保留的样本
    random_indices = random.sample(list(max_category_rows.index), 2*mean_count)
    undersampled_max_category_rows = raw_dataframe.loc[random_indices]
    # 找到其余的行
    other_rows = raw_dataframe[~raw_dataframe.index.isin(max_category_rows.index)]
    # 合并平衡后的dataframe
    dataframe = pd.concat([undersampled_max_category_rows, other_rows])
    dataframe = dataframe.reset_index(drop=True)
    all_categories = [category for sublist in dataframe['Labels'] for category in sublist]
    category_counts = Counter(all_categories)
    # 将结果转换为dataframe并显示
    counts_df = pd.DataFrame(category_counts.items(), columns=['Category', 'Count'])
    print("Catogories count:\n", counts_df)

    mlb = MultiLabelBinarizer()
    labels_encoded = mlb.fit_transform(raw_dataframe['Labels'])
    encoded_dataframe = pd.DataFrame(labels_encoded, columns=mlb.classes_)
    raw_dataframe['Labels'] = encoded_dataframe.apply(lambda row: row.tolist(), axis=1)

    # 超参数
    # num_classes = dataframe['Labels'].apply(len).max()  # 假设标签长度是固定的
    num_classes = 18
    vocab_size = len(vocab)  # 根据词汇表大小设置词汇表维度
    embed_dim = 256
    batch_size = 32
    learning_rate = 0.001
    num_epochs = 30
    threshold = 0.5

    # 数据变换
    # transform = transforms.Compose([
    #     transforms.Resize((224, 224)),
    #     transforms.ToTensor(),
    # ])
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224),  # 随机裁剪并调整大小
        transforms.RandomHorizontalFlip(),  # 随机水平翻转
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),  # 颜色抖动
        transforms.ToTensor(),  # 转换为张量
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # 归一化
    ])

    val_transforms = transforms.Compose([
        transforms.Resize(256),  # 调整大小
        transforms.CenterCrop(224),  # 中心裁剪
        transforms.ToTensor(),  # 转换为张量
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # 归一化
    ])

    train_df, val_df = train_test_split(raw_dataframe, test_size=0.2, random_state=42)

    # All labels
    encoded_train_labels = np.array(train_df['Labels'].tolist())
    # Number of each class
    class_sample_counts = encoded_train_labels.sum(axis=0)
    # Total amount of data
    class_counts = class_sample_counts.sum(axis=0)
    # The weight for each class
    class_weights = class_sample_counts / (class_counts + 1e-5)
    class_weights = torch.from_numpy(class_weights).to(device)
    # Sample weight
    weights = 1.0 / class_sample_counts
    samples_weight = encoded_train_labels.dot(weights)
    # 创建WeightedRandomSampler
    sampler = WeightedRandomSampler(samples_weight, num_samples=len(samples_weight), replacement=True)

    train_dataset = MultiLabelDataset(train_df, vocab, train_transforms, max_length=max_length)
    val_dataset = MultiLabelDataset(val_df, vocab, val_transforms, max_length=max_length)
    # train_loader = DataLoader(train_dataset, sampler=sampler, batch_size=batch_size, num_workers=4)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # 数据集和数据加载器
    # dataset = MultiLabelDataset(dataframe, vocab, transform, max_length)
    # data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    # 模型、损失函数和优化器
    model = MultiLabelModel(num_classes, vocab_size, embed_dim).to(device)
    criterion = nn.BCEWithLogitsLoss(weight = class_weights)  # 多标签分类使用 BCEWithLogitsLoss
    # optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

    print("------------------Train Start!------------------")
    # 训练
    accuracy_metric = MultilabelAccuracy(num_labels=num_classes).to(device)
    precision_metric = MultilabelPrecision(num_labels=num_classes, average='macro').to(device)
    recall_metric = MultilabelRecall(num_labels=num_classes, average='macro').to(device)
    f1_metric = MultilabelF1Score(num_labels=num_classes, average='macro').to(device)
    train_start_time = time()
    best_val_f1 = 0.0
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        accuracy_metric.reset()
        precision_metric.reset()
        recall_metric.reset()
        f1_metric.reset()
        epoch_start_time = time()

        for images, labels, descriptions in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            descriptions = descriptions.to(device)
            optimizer.zero_grad()
            outputs = model(images, descriptions)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds = torch.sigmoid(outputs) > threshold  # 将输出转为二进制标签
            accuracy_metric.update(preds, labels)
            precision_metric.update(preds, labels)
            recall_metric.update(preds, labels)
            f1_metric.update(preds, labels)

        train_accuracy = accuracy_metric.compute()
        train_precision = precision_metric.compute()
        train_recall = recall_metric.compute()
        train_f1 = f1_metric.compute()

        model.eval()  # 确保模型处于验证模式
        with torch.no_grad():
            accuracy_metric.reset()
            precision_metric.reset()
            recall_metric.reset()
            f1_metric.reset()

            for images, labels, descriptions in val_loader:
                # 将数据转移到GPU
                images = images.to(device)
                labels = labels.to(device)
                descriptions = descriptions.to(device)
                outputs = model(images, descriptions)
                # 更新验证集评价指标
                preds = torch.sigmoid(outputs) > threshold
                accuracy_metric.update(preds, labels)
                precision_metric.update(preds, labels)
                recall_metric.update(preds, labels)
                f1_metric.update(preds, labels)
            # 计算验证集评价指标
            val_accuracy = accuracy_metric.compute()
            val_precision = precision_metric.compute()
            val_recall = recall_metric.compute()
            val_f1 = f1_metric.compute()
        avg_loss = total_loss / len(train_loader)

        # scheduler.step()
        epoch_end_time = time()

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), "saved_models/model_7.pth")
            print(f'Saved Best Model with Val F1: {best_val_f1:.4f}')

        print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {avg_loss:.4f}, "
              f"Time: {epoch_end_time-epoch_start_time:.2f}s, Val F1: {val_f1:.4f}, Train F1: {train_f1:.4f}, "
              f"Train precision: {train_precision:.2f}, Train Recall: {train_recall:.2f}, "
              f"Val precision: {val_precision:.2f}, Val recall: {val_recall:.2f}, "
              f"Val accuracy: {val_accuracy:.2f}, Train accuracy: {train_accuracy:.2f}")

    train_end_time = time()
    print(f"Total training Time: {train_end_time - train_start_time} s")
    # 训练完成后保存模型
    # torch.save(model.state_dict(), 'saved_models/model_5.pth')
    # print(f"Model saved!")
    print("------------------Training complete!------------------")
