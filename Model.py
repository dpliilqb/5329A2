import copy

import torch
import torch.nn as nn
import torch.optim as optim


# class Net(nn.Module):
#     def __init__(self, threshold = 0.5):
#         super(Net, self).__init__(),
#         self.conv1 = nn.Conv2d(3, 10, kernel_size=4, padding=1, stride=2)
#         self.conv2 = nn.Conv2d(10, 20, kernel_size=4, padding=1, stride=2)
#         self.conv2_drop = nn.Dropout2d()
#         self.fc1 = nn.Linear(20*64*64, 64)
#         self.fc2 = nn.Linear(64, 32)
#         self.fc3 = nn.Linear(32, 18)
#         self.relu = nn.ReLU()
#         self.softmax = nn.Softmax(dim=1)
#         self.sigmoid = nn.Sigmoid()
#         self.init_weights()
#         self.loss = nn.CrossEntropyLoss()
#         self.optimizer = torch.optim.Adam(self.parameters(), lr=0.1)
#         self.threshold = threshold
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
#     def init_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out')
#                 if m.bias is not None:
#                     nn.init.constant_(m.bias, 0)
#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.relu(x)
#         x = self.conv2(x)
#         x = self.relu(x)
#         x = self.conv2_drop(x)
#         x = x.view(-1, 20*64*64)
#         x = self.fc1(x)
#         x = self.relu(x)
#         x = self.fc2(x)
#         x = self.relu(x)
#         x = self.fc3(x)
#         x = self.sigmoid(x)
#         return x
#
#     def predict(self, x):
#         x = self.conv1(x)
#         x = self.relu(x)
#         x = self.conv2(x)
#         x = self.relu(x)
#         # x = self.conv2_drop(x)
#         # print("Shape after conv1", x.shape)
#         x = x.view(x.size(0), -1)
#         x = self.fc1(x)
#         x = self.relu(x)
#         x = self.fc2(x)
#         x = self.relu(x)
#         x = self.fc3(x)
#         x = self.sigmoid(x)
#
#         return x
#
#     def loss_function(self, x, y):
#         return self.loss(x, y)
#
#     def accuracy(self, x, y):
#         pred = self.predict(x)
#         return self.accuracy(pred, y)
#
#     def evaluate(self, x, y):
#         pred = self.predict(x)
#         return self.accuracy(pred, y)
#
#     def train_step(self, x, y):
#         x, y = x.to(self.device), y.to(self.device)
#         # self.optimizer.zero_grad()
#         pred = self.forward(x)
#         loss = self.loss(pred, y)
#         loss.backward()
#         self.optimizer.step()
#         return loss, pred
#
#     def test_step(self, x):
#         pred = self.forward(x)
#         threshold = self.threshold
#         predictions = (pred > threshold).float()
#         # print("Pred success")
#         # print("Pre shape", pred.shape)
#         # print("Y shape", y.shape)
#         # loss = self.loss(pred, y)
#
#         return predictions
class Net(nn.Module):
    def __init__(self, in_channels=3, num_classes=18, threshold=0.5, learning_rate=0.0001, optimizer='Adam',
                 **optimizer_params):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1, stride=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=1)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1, stride=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=1)
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2_drop = nn.Dropout2d()

        # Raw image with size 256*256
        self.fc1 = nn.Linear(256 * 28 * 28, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, num_classes)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        self.init_weights()

        self.loss = nn.BCEWithLogitsLoss()  # 使用适合多标签分类的损失函数
        self.threshold = threshold
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 初始化优化器
        optimizer_class = getattr(optim, optimizer)
        self.optimizer = optimizer_class(self.parameters(), lr=learning_rate, **optimizer_params)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool1(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.conv4(x)
        x = self.relu(x)
        x = self.pool2(x)
        x = self.conv5(x)
        x = self.relu(x)
        x = self.conv6(x)
        x = self.relu(x)
        x = self.pool3(x)
        x = self.conv2_drop(x)
        x = x.view(x.size(0), -1)
        # x = x.view(-1, 20 * 16 * 16)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        # print("Training: ", x)
        return x

    def predict(self, x):
        x = self.forward(x)
        x = self.sigmoid(x)
        return (x > self.threshold).float()
        # return x
    # def loss_function(self, x, y):
    #     loss = self.loss(x, y)
    #     return loss

    def train_model(self, train_loader, val_loader, num_epochs=25):
        best_model_wts = copy.deepcopy(self.state_dict())
        best_loss = float('inf')

        for epoch in range(num_epochs):
            print(f'Epoch {epoch + 1}/{num_epochs}')
            print('-' * 10)

            # 每个 epoch 都有训练和验证阶段
            for phase in ['train', 'val']:
                if phase == 'train':
                    self.train()  # 训练模式
                    dataloader = train_loader
                else:
                    self.eval()  # 验证模式
                    dataloader = val_loader

                running_loss = 0.0
                all_labels = []
                all_preds = []

                # 遍历数据
                for inputs, labels in dataloader:
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)

                    # 每次迭代都要将梯度归零
                    self.optimizer.zero_grad()

                    # 前向传播
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = self(inputs)
                        loss = self.loss(outputs, labels)

                        # 反向传播 + 优化（仅在训练阶段）
                        if phase == 'train':
                            loss.backward()
                            self.optimizer.step()

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
                    best_model_wts = copy.deepcopy(self.state_dict())

        print('Training complete')
        print(f'Best val loss: {best_loss:4f}')

        # 加载最佳模型权重
        self.load_state_dict(best_model_wts)
        return self