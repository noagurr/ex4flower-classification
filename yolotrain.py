import os
import scipy.io
import shutil
import random
import tarfile
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# ===========================
# STEP 1: Extract Dataset
# ===========================
dataset_path = "/Users/noagu/Documents/noa/bgu/שנה ג/lemida_hishuvit/ex 4/102flowers.tgz"
extract_path = "/Users/noagu/Documents/noa/bgu/שנה ג/lemida_hishuvit/ex 4/flowers"

if not os.path.exists(extract_path):
    print("Extracting dataset...")
    with tarfile.open(dataset_path, "r:gz") as tar:
        tar.extractall(path=extract_path)
    print(" Dataset extracted successfully!")

# ===========================
# STEP 2: Data Preprocessing and Splitting
# ===========================

# Define paths
base_dir = "/Users/noagu/Documents/noa/bgu/שנה ג/lemida_hishuvit/ex 4"
image_dir = os.path.join(base_dir, "flowers/jpg")  # Extracted images
labels_path = os.path.join(base_dir, "imagelabels.mat")

# Load labels
labels_mat = scipy.io.loadmat(labels_path)
labels = labels_mat['labels'][0] - 1  # Convert to zero-based indexing

# Define dataset directories
output_dirs = {
    "train": os.path.join(base_dir, "train"),
    "val": os.path.join(base_dir, "val"),
    "test": os.path.join(base_dir, "test")
}


# Function to reset directories (ensuring a fresh split every run)
def reset_dirs():
    for key in output_dirs:
        shutil.rmtree(output_dirs[key], ignore_errors=True)  # Delete old directories
        os.makedirs(output_dirs[key], exist_ok=True)


# Get all images
image_files = sorted(os.listdir(image_dir))  # Ensure order matches labels


# Function to split and copy images
def split_and_copy(seed):
    print(f"===== Performing Split with seed {seed} =====")
    random.seed(seed)
    num_images = len(image_files)
    indices = list(range(num_images))
    random.shuffle(indices)

    train_split = int(0.5 * num_images)
    val_split = int(0.75 * num_images)

    train_indices = indices[:train_split]
    val_indices = indices[train_split:val_split]
    test_indices = indices[val_split:]

    # Copy files
    def copy_files(indices, subset):
        for idx in indices:
            img_name = image_files[idx]
            label = labels[idx]
            class_dir = os.path.join(output_dirs[subset], str(label))
            os.makedirs(class_dir, exist_ok=True)
            shutil.copy(os.path.join(image_dir, img_name), os.path.join(class_dir, img_name))

    copy_files(train_indices, "train")
    copy_files(val_indices, "val")
    copy_files(test_indices, "test")
    print(f"Data split complete for seed {seed}!")


# Perform new random splits every time the script runs
reset_dirs()
split_and_copy(seed=42)
split_and_copy(seed=99)

# ===========================
# STEP 3: Model Training (YOLOv5)
# ===========================

# Check if GPU (MPS) is available
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# Define transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


# Function to load data
def load_data():
    train_dataset = ImageFolder(root=output_dirs["train"], transform=transform)
    val_dataset = ImageFolder(root=output_dirs["val"], transform=transform)
    test_dataset = ImageFolder(root=output_dirs["test"], transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    return train_loader, val_loader, test_loader


# Load YOLOv5 model
def load_yolo():
    model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5s-cls.pt', force_reload=True)
    model = model.model  # Extract model
    model.model[-1].linear = nn.Linear(model.model[-1].linear.in_features, 102)  # Adjust output layer for 102 classes
    return model.to(device)


# Training function
def train_model(model, train_loader, val_loader, test_loader, num_epochs=8):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    train_losses, val_losses, test_losses = [], [], []
    train_accuracies, val_accuracies, test_accuracies = [], [], []

    for epoch in range(num_epochs):
        model.train()
        train_loss, correct_train, total_train = 0.0, 0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        train_losses.append(train_loss / len(train_loader))
        train_accuracies.append(100 * correct_train / total_train)

        # Validation
        model.eval()
        val_loss, correct_val, total_val = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

        val_losses.append(val_loss / len(val_loader))
        val_accuracies.append(100 * correct_val / total_val)

        # Test
        test_loss, correct_test, total_test = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                test_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total_test += labels.size(0)
                correct_test += (predicted == labels).sum().item()

        test_losses.append(test_loss / len(test_loader))
        test_accuracies.append(100 * correct_test / total_test)

        print(f"Epoch [{epoch + 1}/{num_epochs}], Train Acc: {train_accuracies[-1]:.2f}%, "
              f"Val Acc: {val_accuracies[-1]:.2f}%, Test Acc: {test_accuracies[-1]:.2f}%")

    print(f" Final Test Accuracy: {test_accuracies[-1]:.2f}%\n")
    return train_losses, val_losses, test_losses, train_accuracies, val_accuracies, test_accuracies


# Train and plot results for two splits
for i in range(2):
    train_loader, val_loader, test_loader = load_data()
    model = load_yolo()
    results = train_model(model, train_loader, val_loader, test_loader)

       plt.figure(figsize=(12, 5))

    # Accuracy Plot
    plt.subplot(1, 2, 1)
    plt.plot(results[3], label="Train Acc", marker="o")
    plt.plot(results[4], label="Val Acc", marker="o")
    plt.plot(results[5], label="Test Acc", marker="o")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.title(f"Accuracy Over Epochs - Split {i + 1}")

    # Loss Plot
    plt.subplot(1, 2, 2)
    plt.plot(results[0], label="Train Loss", marker="o")
    plt.plot(results[1], label="Val Loss", marker="o")
    plt.plot(results[2], label="Test Loss", marker="o")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.title(f"Loss Over Epochs - Split {i + 1}")

    plt.savefig(f"training_results_split_{i + 1}.png")
    plt.show()
