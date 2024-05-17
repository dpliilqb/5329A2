from main import ImageTagsDataset, load_data
from torch.utils.data import DataLoader
from Model import Net
import torch
import pandas as pd

if __name__ == '__main__':
    TEST_FILENAME = 'test.csv'
    data_path = "fixed_data"
    test_df = load_data(TEST_FILENAME)
    test_df = test_df[["ImageID"]]

    idx_to_class = {0: '1', 1: '2', 2: '3', 3: '4', 4: '5', 5: '6', 6: '7', 7: '8', 8: '9', 9: '10', 10: '11', 11: '13',
                    12: '14', 13: '15', 14: '16', 15: '17', 16: '18', 17: '19'}
    test_set = ImageTagsDataset(test_df, "fixed_data")
    test_dataloader = DataLoader(test_set, batch_size=32, shuffle=True, num_workers=4)
    threshold = 0.15
    model = Net()
    model.load_state_dict(torch.load("Models/model_1.pth"))
    model.to(model.device)
    final_labels = []
    for image, label in test_dataloader:
        image, label = image.to(model.device), label.to(model.device)
        pred = model.forward(image)
        pred = model.sigmoid(pred) > threshold
        # print("Predicted: ", pred)
        pred_list = [tensor.tolist() for tensor in pred]
        # print("Pred_list:", pred)
        # predicted_labels = pred.argmax(dim=1)
        # print("Predicted labels: ", predicted_labels)
        for sublist in pred_list:
            predicted_class_names = []
            for idx, value in enumerate(sublist):
                if value:  # 如果值为True
                    predicted_class_names.append(idx_to_class[idx])
            final_labels.append(predicted_class_names)
    # print("Labels:", final_labels)

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