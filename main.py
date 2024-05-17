from PIL import Image, ImageOps
import os
import re
from io import StringIO
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MultiLabelBinarizer
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms, models
from torchvision.models import ResNet18_Weights, ResNet50_Weights
from PIL import Image
import pandas as pd
import pickle
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from time import time
from torchmetrics.classification import MultilabelAccuracy, MultilabelPrecision, MultilabelRecall, MultilabelF1Score
from sklearn.model_selection import train_test_split
import random
from collections import Counter
import torch.nn.functional as F

def find_max_image_size(folder_path):
    """
    Finds the maximum width and height among all images in the specified folder.

    Args:
        folder_path (str): The path to the folder containing images.

    Returns:
        tuple: A tuple containing the maximum width and maximum height found among the images.
    """
    max_width = 0  # Initialize the maximum width
    max_height = 0  # Initialize the maximum height

    # Iterate over all files in the specified folder
    for filename in os.listdir(folder_path):
        # Check if the file is an image based on its extension
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            image_path = os.path.join(folder_path, filename)  # Construct the full image path
            try:
                with Image.open(image_path) as img:  # Open the image
                    width, height = img.size  # Get the width and height of the image
                    if width > max_width:  # Update max_width if the current width is greater
                        max_width = width
                    if height > max_height:  # Update max_height if the current height is greater
                        max_height = height
            except IOError:
                print(f"Cannot open {filename}")  # Print an error message if the image cannot be opened

    return max_width, max_height  # Return the maximum width and height

def pad_image_to_square(image, target_size, fill_color='black'):
    """
    Pads an image to make it square and resizes it to the target size.

    Args:
        image (PIL.Image.Image): The input image to be padded and resized.
        target_size (int): The size of the square side after resizing.
        fill_color (str or tuple): The color used to fill the padding area. Default is 'black'.

    Returns:
        PIL.Image.Image: The padded and resized image.
    """
    width, height = image.size  # Get the original width and height of the image
    max_side = max(width, height)  # Determine the size of the square side (the maximum dimension)

    # Calculate the padding needed to make the image square
    left_padding = (max_side - width) // 2
    right_padding = max_side - width - left_padding
    top_padding = (max_side - height) // 2
    bottom_padding = max_side - height - top_padding

    # Pad the image to make it square
    padded_image = ImageOps.expand(image, border=(left_padding, top_padding, right_padding, bottom_padding), fill=fill_color)

    # Resize the padded image to the target size
    return padded_image.resize((target_size, target_size))

def process_dataset(input_folder, output_folder, target_size=300):
    """
    Processes a dataset of images by padding them to be square and resizing to the target size.

    Args:
        input_folder (str): The folder containing the original images.
        output_folder (str): The folder where the processed images will be saved.
        target_size (int, optional): The size of the sides of the square image after resizing. Default is 300.

    """
    os.makedirs(output_folder, exist_ok=True)  # Create the output folder if it doesn't exist

    # Iterate over all files in the input folder
    for image_file in os.listdir(input_folder):
        input_path = os.path.join(input_folder, image_file)  # Construct the full input path
        output_path = os.path.join(output_folder, image_file)  # Construct the full output path

        # Check if the file is a valid image file
        if os.path.isfile(input_path) and input_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            with Image.open(input_path) as img:  # Open the image
                # Process the image by padding it to be square and resizing to the target size
                processed_image = pad_image_to_square(img, target_size)
                processed_image.save(output_path)  # Save the processed image to the output path

class CustomCNN(nn.Module):
    def __init__(self):
        super(CustomCNN, self).__init__()
        # Define the first convolutional layer with 3 input channels, 32 output channels,
        # a kernel size of 3, stride of 1, and padding of 1.
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)

        # Define the second convolutional layer with 32 input channels, 64 output channels,
        # a kernel size of 3, stride of 1, and padding of 1.
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)

        # Define a max pooling layer with a kernel size of 2, stride of 2, and no padding.
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # Define the first fully connected layer.
        # Assuming the input image size is 224x224, and after convolutions and pooling,
        # the feature map size is 56x56 with 64 channels. Thus the input features to fc1 are 64 * 56 * 56.
        self.fc1 = nn.Linear(64 * 56 * 56, 512)

    def forward(self, x):
        # Apply the first convolutional layer followed by ReLU activation and max pooling.
        x = self.pool(F.relu(self.conv1(x)))

        # Apply the second convolutional layer followed by ReLU activation and max pooling.
        x = self.pool(F.relu(self.conv2(x)))

        # Flatten the output from the convolutional layers to make it suitable for the fully connected layer.
        x = x.view(-1, 64 * 56 * 56)

        # Apply the first fully connected layer with ReLU activation.
        x = F.relu(self.fc1(x))

        return x

class MultiLabelDataset(Dataset):
    def __init__(self, dataframe, vocab, transform=None, max_length=50):
        """
        Initializes the MultiLabelDataset object.

        Args:
            dataframe (pd.DataFrame): A pandas DataFrame containing the data.
            vocab (dict): A dictionary mapping tokens to integer indices.
            transform (callable, optional): A function/transform that takes in a PIL image
                and returns a transformed version. Default is None.
            max_length (int): The maximum length of the tokenized text.

        Attributes:
            dataframe (pd.DataFrame): Stores the input dataframe.
            vocab (dict): Stores the vocabulary used for token to index mapping.
            transform (callable): Optional transformation to apply to the images.
            tokenizer (function): Tokenizer function to convert text to tokens. Here it's set to "basic_english".
            max_length (int): Stores the maximum length for text sequences.
            is_test (int): Flag to indicate if the dataset is used for testing. If 1, no labels are expected.
        """
        self.dataframe = dataframe
        self.vocab = vocab
        self.transform = transform
        self.tokenizer = get_tokenizer("basic_english")  # Use the basic English tokenizer from torchtext.
        self.max_length = max_length
        self.is_test = 0  # Initially assume it's not a test dataset (i.e., labels are present).

    def __len__(self):
        """
        Returns the total number of items in the dataset.
        """
        return len(self.dataframe)

    def __getitem__(self, idx):
        """
        Retrieves the item at the specified index `idx` from the dataset.

        Args:
            idx (int): Index of the item to retrieve.

        Returns:
            tuple: Depending on whether labels are present, it returns either
            (image, text_indices) or (image, labels, text_indices).
        """
        # Construct the path to the image file and open the image.
        img_path = "fixed_data/" + self.dataframe.iloc[idx]['ImageID']
        image = Image.open(img_path).convert('RGB')  # Ensure the image is in RGB format.

        # Apply the transform to the image if a transform is specified.
        if self.transform:
            image = self.transform(image)

        # Check if the 'Labels' column exists in the dataframe; if it does, process the labels.
        if "Labels" in self.dataframe.columns:
            labels = torch.tensor(self.dataframe.iloc[idx]['Labels'], dtype=torch.float32)
        else:
            self.is_test = 1  # Set as a test dataset if no labels column, changing behavior of dataset.

        # Retrieve and tokenize the caption associated with the image.
        description = self.dataframe.iloc[idx]['Caption']
        tokens = self.tokenizer(description)

        # Convert tokens into indices using the vocabulary. Pad or truncate to `max_length`.
        text_indices = [self.vocab.get(token, self.vocab['<unk>']) for token in tokens]
        if len(text_indices) > self.max_length:
            text_indices = text_indices[:self.max_length]
        else:
            text_indices.extend([self.vocab['<pad>']] * (self.max_length - len(text_indices)))
        text_indices = torch.tensor(text_indices, dtype=torch.long)

        # Return the appropriate data depending on whether it is a test set or not.
        if self.is_test:
            return image, text_indices
        else:
            return image, labels, text_indices

class TextEmbeddingModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, dropout_rate=0.5):
        """
        Initializes the TextEmbeddingModel object.

        Args:
            vocab_size (int): The size of the vocabulary.
            embed_dim (int): The dimension of the embedding space.
            dropout_rate (float, optional): The dropout rate used after the LSTM layers. Default is 0.5.

        Attributes:
            embedding (nn.Embedding): An embedding layer that converts input tokens into embeddings.
            lstm (nn.LSTM): A bidirectional LSTM layer that processes the sequence of embeddings.
            dropout (nn.Dropout): A dropout layer to prevent overfitting by randomly setting input units to 0.
            fc (nn.Linear): A linear layer that maps from the LSTM output space to the embedding dimension.
        """
        super(TextEmbeddingModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)  # Embedding layer with padding index set to 0.
        self.lstm = nn.LSTM(embed_dim, embed_dim, batch_first=True, bidirectional=True)  # Bidirectional LSTM for capturing dependencies from both directions.
        self.dropout = nn.Dropout(dropout_rate)  # Dropout layer for regularization.
        self.fc = nn.Linear(embed_dim * 2, embed_dim)  # Fully connected layer, output size is the embedding dimension.

        self._init_weights()  # Initialize weights after setting up the model.

    def _init_weights(self):
        """
        Initializes weights of the model with custom rules: Xavier uniform for weights and zeros for biases.
        """
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)  # Xavier initialization for weights.
            elif 'bias' in name:
                nn.init.constant_(param, 0)  # Initializing biases to zero.

    def forward(self, x):
        """
        Defines the forward pass of the model.

        Args:
            x (Tensor): The input tensor containing padded sequences of token indices.

        Returns:
            Tensor: The output tensor after processing the input through embedding, LSTM, dropout, and a fully connected layer.
        """
        x = self.embedding(x)  # Convert token indices to embeddings.
        x, (hidden, _) = self.lstm(x)  # Process embeddings through the LSTM.
        hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)  # Concatenate the outputs from the last two states (bidirectional).
        x = self.dropout(hidden)  # Apply dropout to the concatenated outputs.
        x = self.fc(x)  # Pass through the fully connected layer.
        return x

class TransformerEmbeddingModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers, max_seq_length):
        """
        Initializes the TransformerEmbeddingModel object.

        Args:
            vocab_size (int): The size of the vocabulary.
            embed_dim (int): The dimension of the embedding space.
            num_heads (int): The number of attention heads in the transformer.
            num_layers (int): The number of transformer layers.
            max_seq_length (int): The maximum sequence length for positional encoding.

        Attributes:
            embedding (nn.Embedding): An embedding layer that converts input tokens into embeddings.
            positional_encoding (nn.Parameter): A parameter representing positional encodings for the sequence.
            transformer (nn.Transformer): A transformer model to process the sequence of embeddings.
            fc (nn.Linear): A linear layer that maps the transformer output to the embedding dimension.
        """
        super(TransformerEmbeddingModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)  # Embedding layer for token indices.
        # Positional encoding to add positional information to embeddings.
        self.positional_encoding = nn.Parameter(torch.zeros(1, max_seq_length, embed_dim))
        # Transformer model with specified embedding dimension, number of heads, and layers.
        self.transformer = nn.Transformer(embed_dim, num_heads, num_layers, num_layers)
        self.fc = nn.Linear(embed_dim, embed_dim)  # Fully connected layer for output transformation.

    def forward(self, x):
        """
        Defines the forward pass of the model.

        Args:
            x (Tensor): The input tensor containing sequences of token indices.

        Returns:
            Tensor: The output tensor after processing the input through embedding, transformer, and a fully connected layer.
        """
        # Combine token embeddings with positional encodings.
        x = self.embedding(x) + self.positional_encoding[:, :x.size(1), :]
        # Permute to match transformer input shape (sequence length, batch size, embedding dimension).
        x = x.permute(1, 0, 2)
        # Pass through the transformer model.
        x = self.transformer(x, x)
        # Permute back to (batch size, sequence length, embedding dimension).
        x = x.permute(1, 0, 2)
        # Average pooling over the sequence length to get a fixed-size representation.
        x = x.mean(dim=1)
        # Pass through the fully connected layer.
        x = self.fc(x)
        return x

class MultimodalAttention(nn.Module):
    def __init__(self, d_I, d_T, d):
        """
        Initializes the MultimodalAttention object.

        Args:
            d_I (int): The dimension of the image embeddings.
            d_T (int): The dimension of the text embeddings.
            d (int): The common dimension for the linear transformations.

        Attributes:
            W_I (nn.Linear): Linear layer to project image embeddings to a common dimension d.
            W_T (nn.Linear): Linear layer to project text embeddings to a common dimension d.
            W_Q (nn.Linear): Linear layer to project the image features to query vectors.
            W_K (nn.Linear): Linear layer to project the text features to key vectors.
            W_V (nn.Linear): Linear layer to project the text features to value vectors.
        """
        super(MultimodalAttention, self).__init__()
        self.W_I = nn.Linear(d_I, d)  # Linear transformation for image embeddings.
        self.W_T = nn.Linear(d_T, d)  # Linear transformation for text embeddings.
        self.W_Q = nn.Linear(d, d)  # Linear transformation to create query vectors.
        self.W_K = nn.Linear(d, d)  # Linear transformation to create key vectors.
        self.W_V = nn.Linear(d, d)  # Linear transformation to create value vectors.

    def forward(self, img_embed, text_embed):
        """
        Defines the forward pass of the model.

        Args:
            img_embed (Tensor): The image embeddings.
            text_embed (Tensor): The text embeddings.

        Returns:
            Tensor: The output tensor after applying multimodal attention.
        """
        # Project the image and text embeddings to a common dimension.
        I_prime = self.W_I(img_embed)
        T_prime = self.W_T(text_embed)

        # Generate query vectors from the image embeddings.
        Q_I = self.W_Q(I_prime)
        # Generate key and value vectors from the text embeddings.
        K_T = self.W_K(T_prime)
        V_T = self.W_V(T_prime)

        # Compute the attention weights using the dot product of queries and keys,
        # scaled by the square root of the key dimension.
        attention_weights = F.softmax(Q_I @ K_T.T / torch.sqrt(torch.tensor(K_T.shape[-1], dtype=torch.float32)),
                                      dim=-1)
        # Compute the attended text features by weighting the value vectors with the attention weights.
        H = attention_weights @ V_T

        return H

class MultiLabelModel(nn.Module):
    def __init__(self, num_classes, vocab_size, embed_dim):
        """
        Initializes the MultiLabelModel object.

        Args:
            num_classes (int): The number of output classes for the multi-label classification.
            vocab_size (int): The size of the vocabulary.
            embed_dim (int): The dimension of the text embeddings.

        Attributes:
            cnn (nn.Module): A pre-trained ResNet50 model with the final fully connected layer removed.
            text_model (TextEmbeddingModel): A text embedding model for processing text inputs.
            fc1 (nn.Linear): A linear layer to combine image and text features.
            fc2 (nn.Linear): A linear layer to reduce dimensionality.
            fc3 (nn.Linear): A linear layer to produce the final output logits for each class.
            relu (nn.ReLU): ReLU activation function.
            dropout (nn.Dropout): Dropout layer for regularization to prevent overfitting.
        """
        super(MultiLabelModel, self).__init__()
        # Load a pre-trained ResNet50 model and remove the final fully connected layer.
        self.cnn = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        # If you want to use a custom CNN, you can uncomment the following line and comment out the previous line.
        # self.cnn = CustomCNN()
        self.cnn.fc = nn.Identity()  # Remove the final fully connected layer of ResNet50.

        # Initialize the text embedding model.
        self.text_model = TextEmbeddingModel(vocab_size, embed_dim)
        # If you want to use a transformer-based text embedding model, you can uncomment the following line.
        # self.transformer = TransformerEmbeddingModel(vocab_size, embed_dim, num_heads=4, num_layers=2, max_seq_length=60)

        # Define fully connected layers for combining image and text features.
        self.fc1 = nn.Linear(2048 + embed_dim, 128)  # For ResNet50, the output dimension is 2048.
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)

        # Define the ReLU activation function.
        self.relu = nn.ReLU()

        # If you want to use a multimodal attention mechanism, you can uncomment the following line.
        # self.attention = MultimodalAttention(2048, embed_dim, 512)

        # Define a dropout layer with a dropout rate of 0.2 for regularization.
        self.dropout = nn.Dropout(0.2)

    def forward(self, image, text):
        """
        Defines the forward pass of the model.

        Args:
            image (Tensor): The input tensor containing image data.
            text (Tensor): The input tensor containing text data.

        Returns:
            Tensor: The output tensor containing logits for each class.
        """
        # Extract features from the image using the CNN.
        image_features = self.cnn(image)

        # Extract features from the text using the text embedding model.
        text_features = self.text_model(text)

        # Concatenate image and text features along the feature dimension.
        x = torch.cat((image_features, text_features), dim=1)

        # If using multimodal attention, uncomment the following line.
        # x = self.attention(image_features, text_features)

        # Apply dropout to the concatenated features.
        x = self.dropout(x)

        # Pass through the first fully connected layer and apply ReLU activation.
        x = self.fc1(x)
        x = self.relu(x)

        # Apply dropout.
        x = self.dropout(x)

        # Pass through the second fully connected layer and apply ReLU activation.
        x = self.fc2(x)
        x = self.relu(x)

        # Apply dropout.
        x = self.dropout(x)

        # Pass through the final fully connected layer to get the output logits.
        x = self.fc3(x)

        return x

def yield_tokens(data_iter):
    """
    Tokenizes the captions from the input data iterator.

    Args:
        data_iter (iterable): An iterable containing the data with 'Caption' as one of the keys.

    Yields:
        list: A list of tokens for each caption in the data.
    """
    # Initialize the tokenizer with basic English tokenization.
    tokenizer = get_tokenizer("basic_english")

    # Iterate over the 'Caption' column in the input data iterator.
    for text in data_iter['Caption']:
        # Tokenize the current caption and yield the list of tokens.
        yield tokenizer(text)

def load_data(path):
    """
    Loads and preprocesses data from a CSV file.

    Args:
        path (str): The file path to the CSV file.

    Returns:
        pd.DataFrame: A pandas DataFrame containing the loaded data.
    """
    # Open the file at the specified path.
    with open(path) as file:
        # Read lines from the file and preprocess them.
        # This line substitutes problematic quotes within fields to prevent errors in CSV parsing.
        lines = [re.sub(r'([^,])"(\s*[^\n])', r'\1/"\2', line) for line in file]

    # Combine the preprocessed lines into a single string and read it into a DataFrame.
    df = pd.read_csv(StringIO(''.join(lines)), escapechar="/")

    # Print the DataFrame (uncomment the next line for debugging purposes).
    # print(df)

    return df

def str_to_list(s):
    """
    Converts a string of space-separated integers into a list of integers.

    Args:
        s (str): A string containing space-separated integers.

    Returns:
        list: A list of integers parsed from the input string.
    """
    # Split the string by spaces, map each substring to an integer, and return the resulting list.
    return list(map(int, s.split()))

if __name__ == "__main__":
    # Load the training and testing data
    raw_dataframe = load_data("train.csv")
    test_data = load_data("test.csv")

    # Convert the string representation of labels to a list of integers
    raw_dataframe["Labels"] = raw_dataframe["Labels"].apply(str_to_list)

    # Determine the device to be used for training (GPU if available, otherwise CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize the tokenizer for basic English tokenization
    tokenizer = get_tokenizer("basic_english")

    # Calculate the length of each caption in the training and testing datasets
    train_length = raw_dataframe['Caption'].apply(lambda desc: len(tokenizer(desc)))
    test_length = test_data['Caption'].apply(lambda desc: len(tokenizer(desc)))

    # Determine the maximum length of captions in both datasets
    max_length = max(train_length.max(), test_length.max())
    print("Max Length:", max_length)

    # Combine the captions from training and testing datasets to build the vocabulary
    text_column = pd.concat([raw_dataframe['Caption'], test_data['Caption']]).drop_duplicates().reset_index(drop=True)
    text_column = pd.DataFrame(text_column, columns=['Caption'])

    # Build the vocabulary from the captions using the tokenizer
    vocab = build_vocab_from_iterator(yield_tokens(text_column), specials=["<unk>"])
    vocab.set_default_index(vocab["<unk>"])

    # Save the vocabulary to a file for future use
    with open('vocab.pkl', 'wb') as f:
        pickle.dump(vocab, f)

    # Flatten the list of labels to find the most frequent category
    all_categories = [category for sublist in raw_dataframe['Labels'] for category in sublist]
    category_counts = pd.Series(all_categories).value_counts()

    # Identify the category with the maximum count
    max_category = category_counts.idxmax()
    max_count = category_counts.max()
    mean_count = int(category_counts.mean())

    # Undersample the rows with the most frequent category to balance the dataset
    max_category_rows = raw_dataframe[raw_dataframe['Labels'].apply(lambda x: max_category in x)]
    random_indices = random.sample(list(max_category_rows.index), 2 * mean_count)
    undersampled_max_category_rows = raw_dataframe.loc[random_indices]

    # Combine the undersampled rows with the rest of the dataset
    other_rows = raw_dataframe[~raw_dataframe.index.isin(max_category_rows.index)]
    dataframe = pd.concat([undersampled_max_category_rows, other_rows]).reset_index(drop=True)

    # Recalculate the category counts after undersampling
    all_categories = [category for sublist in dataframe['Labels'] for category in sublist]
    category_counts = Counter(all_categories)
    counts_df = pd.DataFrame(category_counts.items(), columns=['Category', 'Count'])
    print("Categories count:\n", counts_df)

    # Binarize the labels using MultiLabelBinarizer
    mlb = MultiLabelBinarizer()
    labels_encoded = mlb.fit_transform(raw_dataframe['Labels'])

    # Create a DataFrame with the encoded labels
    encoded_dataframe = pd.DataFrame(labels_encoded, columns=mlb.classes_)
    raw_dataframe['Labels'] = encoded_dataframe.apply(lambda row: row.tolist(), axis=1)

    # Define hyperparameters and other configurations
    num_classes = 18  # The number of output classes for the multi-label classification task
    vocab_size = len(vocab)  # The size of the vocabulary
    embed_dim = 256  # The dimension of the text embeddings
    batch_size = 32  # The number of samples per batch
    learning_rate = 0.001  # The learning rate for the optimizer
    num_epochs = 30  # The number of epochs to train the model
    threshold = 0.9  # Threshold for determining class membership in multi-label classification

    # Define data augmentation and preprocessing transformations for training and validation datasets
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224),  # Randomly resize and crop the image to 224x224 pixels
        transforms.RandomHorizontalFlip(),  # Randomly flip the image horizontally
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
        # Randomly change image brightness, contrast, saturation, and hue
        transforms.ToTensor(),  # Convert the image to a PyTorch tensor
    ])

    val_transforms = transforms.Compose([
        transforms.Resize(256),  # Resize the image to 256x256 pixels
        transforms.CenterCrop(224),  # Center crop the image to 224x224 pixels
        transforms.ToTensor(),  # Convert the image to a PyTorch tensor
    ])

    # Split the data into training and validation sets
    train_df, val_df = train_test_split(raw_dataframe, test_size=0.2, random_state=33)

    # Encode the training labels into a numpy array
    encoded_train_labels = np.array(train_df['Labels'].tolist())

    # Calculate the number of samples for each class
    class_sample_counts = encoded_train_labels.sum(axis=0)

    # Calculate the total number of samples
    class_counts = class_sample_counts.sum(axis=0)

    # Calculate the weight for each class to handle class imbalance
    class_weights = class_sample_counts / (class_counts + 1e-5)
    class_weights = torch.from_numpy(class_weights).to(device)

    # Calculate the sample weights to handle class imbalance
    weights = 1.0 / class_sample_counts
    samples_weight = encoded_train_labels.dot(weights)

    # Create a weighted random sampler to ensure balanced sampling of classes during training
    sampler = WeightedRandomSampler(samples_weight, num_samples=len(samples_weight), replacement=True)

    # Create the training and validation datasets
    train_dataset = MultiLabelDataset(train_df, vocab, train_transforms, max_length=max_length)
    val_dataset = MultiLabelDataset(val_df, vocab, val_transforms, max_length=max_length)

    # Create the data loaders for training and validation
    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=4)  # sampler=sampler,
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Initialize the model and move it to the appropriate device (GPU or CPU)
    model = MultiLabelModel(num_classes, vocab_size, embed_dim).to(device)

    # Define the loss function
    criterion = nn.BCEWithLogitsLoss()

    # Define the optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    print("------------------Train Start!------------------")

    # Initialize evaluation metrics for multi-label classification
    accuracy_metric = MultilabelAccuracy(num_labels=num_classes).to(device)
    precision_metric = MultilabelPrecision(num_labels=num_classes, average='macro').to(device)
    recall_metric = MultilabelRecall(num_labels=num_classes, average='macro').to(device)
    f1_metric = MultilabelF1Score(num_labels=num_classes, average='macro').to(device)

    # Record the start time of the training process
    train_start_time = time()
    best_val_f1 = 0.0  # Initialize the best validation F1 score

    # Training loop
    for epoch in range(num_epochs):
        model.train()  # Set the model to training mode
        total_loss = 0  # Initialize the total loss for the epoch
        accuracy_metric.reset()  # Reset metrics for the new epoch
        precision_metric.reset()
        recall_metric.reset()
        f1_metric.reset()
        epoch_start_time = time()  # Record the start time of the epoch

        # Training loop for each batch
        for images, labels, descriptions in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            descriptions = descriptions.to(device)

            optimizer.zero_grad()  # Zero the parameter gradients
            outputs = model(images, descriptions)  # Forward pass
            loss = criterion(outputs, labels)  # Compute the loss
            loss.backward()  # Backward pass
            optimizer.step()  # Update model parameters

            total_loss += loss.item()  # Accumulate the total loss
            preds = torch.sigmoid(outputs) > threshold  # Apply threshold to get binary predictions

            # Update metrics with predictions and labels
            accuracy_metric.update(preds, labels)
            precision_metric.update(preds, labels)
            recall_metric.update(preds, labels)
            f1_metric.update(preds, labels)

        # Compute training metrics for the epoch
        train_accuracy = accuracy_metric.compute()
        train_precision = precision_metric.compute()
        train_recall = recall_metric.compute()
        train_f1 = f1_metric.compute()

        # Validation loop
        model.eval()  # Set the model to evaluation mode
        with torch.no_grad():  # Disable gradient calculation
            accuracy_metric.reset()  # Reset metrics for the new validation pass
            precision_metric.reset()
            recall_metric.reset()
            f1_metric.reset()

            for images, labels, descriptions in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                descriptions = descriptions.to(device)

                outputs = model(images, descriptions)  # Forward pass
                preds = torch.sigmoid(outputs) > threshold  # Apply threshold to get binary predictions

                # Update metrics with predictions and labels
                accuracy_metric.update(preds, labels)
                precision_metric.update(preds, labels)
                recall_metric.update(preds, labels)
                f1_metric.update(preds, labels)

            # Compute validation metrics for the epoch
            val_accuracy = accuracy_metric.compute()
            val_precision = precision_metric.compute()
            val_recall = recall_metric.compute()
            val_f1 = f1_metric.compute()

        avg_loss = total_loss / len(train_loader)  # Compute average loss for the epoch

        # Record the end time of the epoch
        epoch_end_time = time()

        # Save the best model based on validation F1 score
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), "saved_models/model_10.pth")
            print(f'Saved Best Model with Val F1: {best_val_f1:.4f}')

        # Print training and validation metrics for the epoch
        print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {avg_loss:.4f}, "
              f"Time: {epoch_end_time - epoch_start_time:.2f}s, Val F1: {val_f1:.4f}, Train F1: {train_f1:.4f}, "
              f"Train precision: {train_precision:.2f}, Train Recall: {train_recall:.2f}, "
              f"Val precision: {val_precision:.2f}, Val recall: {val_recall:.2f}, "
              f"Val accuracy: {val_accuracy:.2f}, Train accuracy: {train_accuracy:.2f}")

    # Record the end time of the training process
    train_end_time = time()
    print(f"Total training Time: {train_end_time - train_start_time} s")
    print("------------------Training complete!------------------")