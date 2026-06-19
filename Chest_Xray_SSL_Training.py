# Karthikeyan Kumaravel Krishnan
# Improving Chest X-ray Classification with Semi-Supervised Learning

# To plot graphs
import numpy as numPy
import matplotlib.pyplot as plotMaker

# To randomly sample images from each class folder (This prevents data bias)
import random

# To work with file paths in computer
from pathlib import Path

# Allows SSL Model to reuse unlabeled data
from itertools import cycle

# To build and train neural networks
import torch
from torch import nn as neuralNet
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

# To generate evaluation metrics
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix, roc_curve, auc
from PIL import Image
from sklearn.preprocessing import label_binarize

# Model Inputs / Configuration
# Checks if computer has a GPU to run the modeling, otherwise it resorts to using the CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Sets a seed so that the same outputs can be regenerated
random_generated_seed = 5804

# Number of training rounds/epochs each model would train
num_of_supervised_epochs = 10
num_of_semi_supervised_epochs = 10

# Location of the dataset (Reads from the same folder the code is in)
root_directory_name = "COVID-19_Radiography_Dataset"

# Model configuration variables
image_size = 128
max_images_per_folder = 100
batch_size = 8
learning_rate = 3e-4
unsupervised_weight = 0.5

# Ensures the same seed is being used
random.seed(random_generated_seed)
numPy.random.seed(random_generated_seed)
torch.manual_seed(random_generated_seed)

# Used to normalize our training data (Pixels of image)
# Each image pixel would need to be normalized per channel when using values for CNNs
CNN_image_pixel_mean = [0.485, 0.456, 0.406]
CNN_image_pixel_standard_deviation = [0.229, 0.224, 0.225]

# Resizes all images in data set to a fixed square shape to ensure fixed-size inputs and consistent dimensions
def resize_training_images(use_augmentation=False):

    # Makes changes to each image by changing its pixel values and normalizes its RGB values
    return transforms.Compose([
        transforms.Resize((image_size,image_size)),
        *( [transforms.ColorJitter(0.1,0.1)] if use_augmentation else [] ),
        transforms.ToTensor(),
        transforms.Normalize(CNN_image_pixel_mean, CNN_image_pixel_standard_deviation)
    ])

# Created a class that would load datasets and configure them for model use
class Dataset_Loader(Dataset):

    # Constructor methods
    def __init__(self, samples, transform=None, unlabeled=False):
        self.samples = samples
        self.transform = transform
        self.unlabeled = unlabeled

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, label = self.samples[index]
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image if self.unlabeled else (image, label)

    # A python dictionary is used to convert the class name into a numeric label since when training models, the labels must be numbers not strings
    # Explicit mapping of class names to labels
    # For example, inside the "COVID-19_Radiography_Dataset" folder there are 4 folders: "COVID", "NORMAL", "PNEUMONIA","LUNG_OPACITY"
    class_label_map = {
        "COVID":0,
        "NORMAL":1,
        "VIRAL PNEUMONIA":2,
        "LUNG_OPACITY":3
    }

# Creates a small convolutional neural network (CNN) for Chest X-ray classification
def create_CNN_classifier(number_of_categories):
    return neuralNet.Sequential(
        neuralNet.Conv2d(3,16,3,padding=1), neuralNet.ReLU(), neuralNet.MaxPool2d(2),
        neuralNet.Conv2d(16,32,3,padding=1), neuralNet.ReLU(), neuralNet.MaxPool2d(2),
        neuralNet.Flatten(),
        neuralNet.Linear(32*(image_size//4)**2,64), neuralNet.ReLU(),
        neuralNet.Linear(64,number_of_categories)
    )

# Helper method that collects the model's raw data output (logits)
# Would be used later to graph ROC curves
def get_logits_and_labels(model, dataset):
    model.eval()
    logits, labels = [], []
    with torch.no_grad():
        for image, label in dataset:
            output = model(image.unsqueeze(0).to(device))
            logits.append(output.cpu()[0])
            labels.append(label)
    return numPy.array(logits), numPy.array(labels)

# Helper method that calculates the class weights of a model during training
# This ensures that there is no class imbalance
def compute_class_weights(labels, num_classes):
    counts = numPy.bincount(labels, minlength=num_classes).clip(1)
    weights = (1 / counts) * num_classes / numPy.sum(1 / counts)
    return torch.tensor(weights, dtype=torch.float)

# Used to generate and plot confusion matrices
def create_confusion_matrix(title, true_labels, predicted_labels, class_names):
    matrix = confusion_matrix(true_labels, predicted_labels)
    plotMaker.figure(figsize=(8, 6))
    plotMaker.title(title)
    plotMaker.imshow(matrix, cmap="Blues")
    plotMaker.xticks(range(len(class_names)), class_names, rotation=45)
    plotMaker.yticks(range(len(class_names)), class_names)

    # Loops over each row in the confusion matrix
    for row_index, row in enumerate(matrix):
        # Loops over each column in the current row
        for column_index, value in enumerate(row):
            plotMaker.text(column_index,row_index,value,ha="center",va="center")
    plotMaker.tight_layout()
    plotMaker.show()

# Used to generate and plot ROC curves
def create_ROC_Curve(title, true_labels, logits, class_names):
    probabilities = torch.softmax(torch.tensor(logits),1).numpy()
    true_binary = label_binarize(true_labels, classes=range(len(class_names)))

    # Loops over each class name and its corresponding column
    for index,name in enumerate(class_names):

        # Checks if the class has positive binary values
        if true_binary[:,index].sum():
            fpr,tpr,_ = roc_curve(true_binary[:,index], probabilities[:,index])
            plotMaker.plot(fpr,tpr,label=f"{name} (AUC={auc(fpr,tpr):.2f})")

    plotMaker.plot([0,1],[0,1],'k--')
    plotMaker.title(title)
    plotMaker.legend()
    plotMaker.show()

# Prepares the model with the tools/functions needed for training
def initialize_model_training(model, class_weights):

    # Sets the learning rate for the Adam optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Creates a function that would be used to find how wrong the model's predictions are
    loss_function = neuralNet.CrossEntropyLoss(weight=class_weights.to(device) if class_weights is not None else None)

    return optimizer, loss_function

# Trains the model using ONLY labeled data
def train_supervised_model(model, labeled_loader, epochs, class_weights=None):

    # Sets up the optimizer and loss function for model training and learning
    optimizer, loss_function = initialize_model_training(model, class_weights)

    # Loops through epochs/training rounds to train the model based on images from dataset
    for training_attempt in range(epochs):
        model.train()

        # Loops through labeled images of the dataset
        for images, labels in labeled_loader:
            images, labels = images.to(device), labels.to(device)

            # Calculates supervised loss
            loss = loss_function(model(images), labels)

            # Clears previously stored gradients, computes new gradients, and updates model with those gradients
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


# Trains the model using labeled + unlabeled data with 80% confidence threshold
def train_semi_supervised_model(model,labeled_loader,unlabeled_loader,epochs,class_weights=None, confidence_threshold=0.8):

    # Sets up the optimizer and loss function for model training and learning
    optimizer, loss_function = initialize_model_training(model, class_weights)

    unlabeled_image_iterator = cycle(unlabeled_loader)

    for training_attempt in range(epochs):
        model.train()

        # Loops through labeled images of the dataset
        for images, labels in labeled_loader:
            images, labels = images.to(device), labels.to(device)

            # Calculates supervised loss
            loss = loss_function(model(images), labels)

            # Retrieves the next batch of unlabeled images
            unlabeled_images = next(unlabeled_image_iterator).to(device)

            # Generates pseudo-labels
            with torch.no_grad():
                pseudo_logits = model(unlabeled_images)
                pseudo_probs = torch.softmax(pseudo_logits, dim=1)
                pseudo_labels = torch.argmax(pseudo_probs, 1)
                max_probs, _ = torch.max(pseudo_probs, dim=1)

            # Filters out low-confidence pseudo-labels
            mask = max_probs >= confidence_threshold
            if mask.sum() > 0:
                # Computes semi-supervised loss using the pseudo-labels as if they are real labels
                unsupervised_loss = loss_function(model(unlabeled_images), pseudo_labels)
                loss += unsupervised_weight * unsupervised_loss

            # Clears previously stored gradients, computes new gradients, and updates model with those gradients
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

def main():

    # Reads all image files from the dataset directory and parses them correctly
    all_samples=[]
    for folder, label in Dataset_Loader.class_label_map.items():
        image_directory = Path(root_directory_name)/folder/"images"
        if image_directory.exists():
            images = random.sample(list(image_directory.iterdir()),
                                   min(max_images_per_folder,len(list(image_directory.iterdir()))))
            all_samples += [(str(img),label) for img in images]

    # Shuffles the images from the dataset in a random order to avoid data bias
    random.shuffle(all_samples)

    # Test-train split where 70% of images are used for training and 30% for evaluation
    test_train_split = int(0.7 * len(all_samples))
    training_images, evaluation_images = all_samples[:test_train_split], all_samples[test_train_split:]

    # Ensures training data is evenly/fairly split between supervised and semi-supervised model training
    num_labeled = len(training_images) // 2
    labeled_training_images = training_images[:num_labeled]
    unlabeled_training_images = training_images[num_labeled:]

    # Dataset used separately for testing/validation to generate metrics
    evaluation_dataset = Dataset_Loader(evaluation_images, resize_training_images())

    class_names = list(Dataset_Loader.class_label_map.keys())
    class_weights = compute_class_weights([l for _,l in training_images], len(class_names))

    print("--------------------- SUPERVISED MODEL (Epochs = " + str(num_of_supervised_epochs) + ") ---------------------");

    supervised_dataset = Dataset_Loader(labeled_training_images, resize_training_images())
    supervised_loader = DataLoader(supervised_dataset,batch_size,shuffle=True, generator=torch.Generator().manual_seed(random_generated_seed))
    supervised_model = create_CNN_classifier(len(class_names)).to(device)

    # Trains the supervised model
    train_supervised_model(
        supervised_model,
        supervised_loader,
        num_of_supervised_epochs,
        class_weights
    )

    # Retrieves data from the trained supervised model
    logits, true_labels = get_logits_and_labels(supervised_model, evaluation_dataset)
    predictions = logits.argmax(1)

    # Displays metrics for supervised model
    print(classification_report(true_labels, predictions, target_names=class_names))
    print("Supervised Accuracy:", accuracy_score(true_labels, predictions))
    print("Supervised Macro F1:", f1_score(true_labels, predictions, average="macro"))
    print("\n")

    create_confusion_matrix("Supervised Confusion Matrix", true_labels, predictions, class_names)
    create_ROC_Curve("Supervised ROC", true_labels, logits, class_names)

    print("--------------------- SEMI-SUPERVISED MODEL (Epochs = " + str(num_of_supervised_epochs) + ") ---------------------");

    unlabeled_dataset = Dataset_Loader(unlabeled_training_images, resize_training_images(True), unlabeled=True)
    unlabeled_loader = DataLoader(unlabeled_dataset,batch_size,shuffle=True, generator=torch.Generator().manual_seed(random_generated_seed))
    semi_supervised_model = create_CNN_classifier(len(class_names)).to(device)
    semi_supervised_model.load_state_dict(supervised_model.state_dict())

    # Trains the semi-supervised model
    train_semi_supervised_model(
        semi_supervised_model,
        supervised_loader,
        unlabeled_loader,
        num_of_semi_supervised_epochs,
        class_weights
    )

    # Retrieves data from the trained semi-supervised model
    logits, true_labels = get_logits_and_labels(semi_supervised_model, evaluation_dataset)
    predictions = logits.argmax(1)

    # Displays metrics for semi-supervised model
    print(classification_report(true_labels, predictions, target_names=class_names))
    create_confusion_matrix("Semi-Supervised Confusion Matrix", true_labels, predictions, class_names)
    create_ROC_Curve("Semi-Supervised ROC", true_labels, logits, class_names)

    print("Semi-Supervised Accuracy:", accuracy_score(true_labels, predictions))
    print("Semi-Supervised Macro F1:", f1_score(true_labels, predictions, average="macro"))

if __name__=="__main__":
    main()
