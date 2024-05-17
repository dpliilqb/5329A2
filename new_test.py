import numpy as np
import torch.nn as nn
from new_train import MultiLabelDataset, load_data, MultiLabelModel
from torch.utils.data import DataLoader
import torch
import pandas as pd
from torchvision import transforms
import pickle

if __name__ == '__main__':
    TEST_FILENAME = 'test.csv'
    data_path = "fixed_data"
    batch_size = 32
    num_classes = 18
    test_df = load_data(TEST_FILENAME)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    with open('vocab.pkl', 'rb') as f:
        vocab = pickle.load(f)
    vocab.set_default_index(vocab["<unk>"])
    vocab_size = len(vocab)
    # print(f"Vocab size: {vocab_size}")
    embed_dim = 256

    # transform = transforms.Compose([
    #     transforms.Resize((224, 224)),
    #     transforms.ToTensor(),
    # ])

    test_transforms = transforms.Compose([
        transforms.Resize(256),  # 调整大小
        transforms.CenterCrop(224),  # 中心裁剪
        transforms.ToTensor(),  # 转换为张量
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # 归一化
    ])

    # 数据集和数据加载器
    test_dataset = MultiLabelDataset(test_df, vocab, test_transforms, max_length=56)
    test_data_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    idx_to_class = {0: '1', 1: '2', 2: '3', 3: '4', 4: '5', 5: '6', 6: '7', 7: '8', 8: '9', 9: '10', 10: '11', 11: '13',
                    12: '14', 13: '15', 14: '16', 15: '17', 16: '18', 17: '19'}

    threshold = 0.5
    model = MultiLabelModel(num_classes, vocab_size, embed_dim)
    model.load_state_dict(torch.load("saved_models/model_7.pth"))

    # checkpoint = torch.load('saved_models/model_1.pth')
    # print(checkpoint.keys())
    # checkpoint = torch.load('path_to_checkpoint.pth')
    # 处理参数形状不匹配问题
    # if 'text_model.embedding.weight' in checkpoint:
    #     old_embedding_weight = checkpoint['text_model.embedding.weight']
    #     num_embeddings_to_copy = min(old_embedding_weight.size(0), model.text_model.embedding.weight.size(0))
    #     model.text_model.embedding.weight.data[:num_embeddings_to_copy] = old_embedding_weight[:num_embeddings_to_copy]
    #     # 移除嵌入层权重以避免重复加载
    #     del checkpoint['text_model.embedding.weight']
    # # 加载其余的状态字典
    # model.load_state_dict(checkpoint, strict=False)

    model.to(device)
    model.eval()

    all_preds = []
    final_labels = []
    with torch.no_grad():
        for images, descriptions in test_data_loader:
            images = images.to(device)
            descriptions = descriptions.to(device)

            outputs = model.forward(images, descriptions)
            preds = torch.sigmoid(outputs).cpu().numpy() > 0.5
            pred_list = np.array(preds)

            for sublist in pred_list:
                predicted_class_names = []
                for idx, value in enumerate(sublist):
                    if value:  # 如果值为True
                        predicted_class_names.append(idx_to_class[idx])
                final_labels.append(predicted_class_names)


    # predictions = np.array(all_preds)

    result_dict = {"ImageID": [], "Labels": []}
    for i in range(len(final_labels)):
        str_list = [str(j) for j in final_labels[i]]
        label_text = " ".join(str_list)
        image_text = str(test_df["ImageID"][i])
        result_dict["ImageID"].append(image_text)
        result_dict["Labels"].append(label_text)

    result_df = pd.DataFrame(result_dict, columns=["ImageID", "Labels"])
    print(result_df.head(3))
    with open('output.csv', 'w') as file:
        pass
    result_df.to_csv('output.csv', index=False)
    print("Written to output.csv")
