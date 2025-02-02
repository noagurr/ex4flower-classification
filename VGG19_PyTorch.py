import os
import numpy as np
import scipy.io
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
import matplotlib.pyplot as plt
from torchvision.models import VGG19_Weights

########## Load Dataset ##########
dataset_dir = "/home/maayan/Maayan/Machine_learning/work4/102flowers/jpg"
labels_file = "/home/maayan/Maayan/Machine_learning/work4/imagelabels.mat"


mat = scipy.io.loadmat(labels_file)
labels = mat["labels"][0] - 1  

image_files = sorted(os.listdir(dataset_dir))

class FlowersDataset(Dataset):
    def __init__(self, img_dir, img_labels, img_names, transform=None):
        self.img_dir = img_dir
        self.img_labels = img_labels
        self.img_names = img_names
        self.transform = transform

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_names[idx])
        image = Image.open(img_path).convert("RGB")
        label = self.img_labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label


transform = transforms.Compose([
    transforms.Resize((224, 224)),  
    transforms.ToTensor(), 
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  
])


full_dataset = FlowersDataset(dataset_dir, labels, image_files, transform=transform)

########## Split Dataset ##########
def split_dataset_twice(dataset, seed1=42, seed2=99):
    torch.manual_seed(seed1)
    train_size = int(0.5 * len(dataset))
    val_size = int(0.25 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_set_1, val_set_1, test_set_1 = random_split(dataset, [train_size, val_size, test_size])

    torch.manual_seed(seed2)
    train_set_2, val_set_2, test_set_2 = random_split(dataset, [train_size, val_size, test_size])

    return (train_set_1, val_set_1, test_set_1), (train_set_2, val_set_2, test_set_2)


(split1_train, split1_val, split1_test), (split2_train, split2_val, split2_test) = split_dataset_twice(full_dataset)


batch_size = 32

train_loader_1 = DataLoader(split1_train, batch_size=batch_size, shuffle=True)
val_loader_1 = DataLoader(split1_val, batch_size=batch_size, shuffle=False)
test_loader_1 = DataLoader(split1_test, batch_size=batch_size, shuffle=False)

train_loader_2 = DataLoader(split2_train, batch_size=batch_size, shuffle=True)
val_loader_2 = DataLoader(split2_val, batch_size=batch_size, shuffle=False)
test_loader_2 = DataLoader(split2_test, batch_size=batch_size, shuffle=False)

########## Load Pretrained VGG19 Model ##########
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.vgg19(weights=VGG19_Weights.DEFAULT)


for param in model.features.parameters():
    param.requires_grad = False


model.classifier[6] = nn.Linear(4096, 102) 
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=0.0001)

########## Train Function ##########
def train_model(model, train_loader, val_loader, test_loader, criterion, optimizer, epochs=8):
    train_losses, val_losses, test_losses = [], [], []
    train_accs, val_accs, test_accs = [], [], []

    for epoch in range(epochs):
        model.train()
        train_loss, train_correct = 0.0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_correct += (outputs.argmax(1) == labels).sum().item()

        train_losses.append(train_loss / len(train_loader))
        train_accs.append(train_correct / len(train_loader.dataset))

        model.eval()
        val_loss, val_correct = 0.0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                val_correct += (outputs.argmax(1) == labels).sum().item()

        val_losses.append(val_loss / len(val_loader))
        val_accs.append(val_correct / len(val_loader.dataset))

        test_loss, test_correct, total = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                test_loss += loss.item()
                test_correct += (outputs.argmax(1) == labels).sum().item()
                total += labels.size(0)

        test_losses.append(test_loss / len(test_loader))
        test_accs.append(test_correct / total)

        print(f"Epoch {epoch+1}/{epochs} | Train Acc: {train_accs[-1]:.4f} | Val Acc: {val_accs[-1]:.4f} | Test Acc: {test_accs[-1]:.4f}")

    return train_losses, val_losses, test_losses, train_accs, val_accs, test_accs

########## Train Model on Two Splits ##########
train_losses_1, val_losses_1, test_losses_1, train_accs_1, val_accs_1, test_accs_1 = train_model(
    model, train_loader_1, val_loader_1, test_loader_1, criterion, optimizer, epochs=8
)

train_losses_2, val_losses_2, test_losses_2, train_accs_2, val_accs_2, test_accs_2 = train_model(
    model, train_loader_2, val_loader_2, test_loader_2, criterion, optimizer, epochs=8
)

########## Plot Results ##########
def plot_metrics(train_accs_1, val_accs_1, test_accs_1, train_accs_2, val_accs_2, test_accs_2,
                 train_losses_1, val_losses_1, test_losses_1, train_losses_2, val_losses_2, test_losses_2):
    epochs = range(1, len(train_accs_1) + 1)

    plt.figure(figsize=(12, 5))

    # Accuracy
    plt.subplot(1, 2, 1)
#    plt.plot(epochs, train_accs_1, label="Train Accuracy - Split 1")
#    plt.plot(epochs, val_accs_1, label="Validation Accuracy - Split 1")
#    plt.plot(epochs, test_accs_1, label="Test Accuracy - Split 1", linestyle="--")
    plt.plot(epochs, train_accs_2, label="Train Accuracy")
    plt.plot(epochs, val_accs_2, label="Validation Accuracy")
    plt.plot(epochs, test_accs_2, label="Test Accuracy", linestyle="--")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title("Training, Validation, and Test Accuracy")

    # Cross-Entropy Loss
    plt.subplot(1, 2, 2)
#    plt.plot(epochs, train_losses_1, label="Train Loss - Split 1")
#    plt.plot(epochs, val_losses_1, label="Validation Loss - Split 1")
#    plt.plot(epochs, test_losses_1, label="Test Loss - Split 1", linestyle="--")
    plt.plot(epochs, train_losses_2, label="Train Loss")
    plt.plot(epochs, val_losses_2, label="Validation Loss")
    plt.plot(epochs, test_losses_2, label="Test Loss", linestyle="--")
    plt.xlabel("Epochs")
    plt.ylabel("Cross-Entropy Loss")
    plt.legend()
    plt.title("Training, Validation, and Test Loss")

    plt.show()


plot_metrics(train_accs_1, val_accs_1, test_accs_1,
             train_accs_2, val_accs_2, test_accs_2,
             train_losses_1, val_losses_1, test_losses_1,
             train_losses_2, val_losses_2, test_losses_2)
             
             
########## Display Final Test Set Accuracy ##########
#print(f"Final Test Accuracy (Split 1): {test_accs_1[-1]:.2f}")
print(f"Final Test Accuracy: {test_accs_2[-1] * 100:.2f}%")



