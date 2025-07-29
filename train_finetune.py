import os
import zipfile
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision import models
from torch.utils.data import DataLoader
from tqdm import tqdm
import gdown

DATASET_URL = "https://drive.google.com/uc?id=YOUR_FILE_ID"  # 🔁 Replace with actual ID
DATA_ZIP = "dataset.zip"
DATA_DIR = "dataset"
BATCH_SIZE = 32
NUM_EPOCHS = 5
NUM_CLASSES = 2
MODEL_PATH = "fine_tuned_ai_vs_human.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def download_and_extract_dataset():
    if not os.path.exists(DATA_DIR):
        print("📥 Downloading dataset from Google Drive...")
        gdown.download(DATASET_URL, DATA_ZIP, quiet=False)

        print("📂 Extracting dataset...")
        with zipfile.ZipFile(DATA_ZIP, 'r') as zip_ref:
            zip_ref.extractall(DATA_DIR)

        print("✅ Dataset ready.")
    else:
        print("✅ Dataset already exists.")


def main():
    print("🚀 Starting fine-tuning")

    download_and_extract_dataset()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    train_ds = datasets.ImageFolder(os.path.join(DATA_DIR, "train"), transform=transform)
    val_ds = datasets.ImageFolder(os.path.join(DATA_DIR, "validate"), transform=transform)
    test_ds = datasets.ImageFolder(os.path.join(DATA_DIR, "test"), transform=transform)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    for epoch in range(NUM_EPOCHS):
        model.train()
        total, correct = 0, 0
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")
        for images, labels in loop:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            loop.set_postfix(loss=loss.item(), acc=100.0 * correct / total)

    torch.save(model.state_dict(), MODEL_PATH)
    print(f"\n✅ Model saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()
