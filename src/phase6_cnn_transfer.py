"""
Phase 6: CNN / Transfer Learning Model
--------------------------------------
- Builds Custom CNN and PyTorch ResNet18 Transfer Learning Architecture.
- Trains the deep learning network on dataset/splits/train with validation tuning.
- Saves trained weights to outputs/models/resnet_model.pth.
- Generates Training vs Validation Loss & Accuracy curves to outputs/figures/phase6_learning_curves.png.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, models, datasets
import matplotlib.pyplot as plt
from config import SPLIT_DATA_DIR, MODELS_DIR, FIGURES_DIR, NUM_CLASSES, BATCH_SIZE, EPOCHS, LEARNING_RATE, RANDOM_SEED

# Set seed
torch.manual_seed(RANDOM_SEED)

class CustomMedicalCNN(nn.Module):
    """
    Custom 3-block CNN for viva architecture comparison.
    """
    def __init__(self, num_classes=NUM_CLASSES):
        super(CustomMedicalCNN, self).__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def get_transfer_learning_model(num_classes=NUM_CLASSES):
    """
    ResNet18 Transfer Learning backbone pretrained on ImageNet.
    """
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    # Fine-tune final fully-connected layer
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, num_classes)
    )
    return model


def get_data_loaders():
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_ds = datasets.ImageFolder(SPLIT_DATA_DIR / "train", transform=train_transform)
    val_ds = datasets.ImageFolder(SPLIT_DATA_DIR / "val", transform=val_test_transform)
    test_ds = datasets.ImageFolder(SPLIT_DATA_DIR / "test", transform=val_test_transform)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    return train_loader, val_loader, test_loader


def run_phase6(model_type="resnet18"):
    print("\n" + "="*70)
    print(f" 🤖 PHASE 6: CNN / TRANSFER LEARNING MODEL ({model_type.upper()})")
    print("="*70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"   Computing Device: {device}")

    train_loader, val_loader, test_loader = get_data_loaders()

    if model_type == "custom":
        model = CustomMedicalCNN(num_classes=NUM_CLASSES)
    else:
        model = get_transfer_learning_model(num_classes=NUM_CLASSES)

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    best_val_loss = float("inf")
    save_path = MODELS_DIR / "resnet_model.pth"

    print("\n🚀 Training Deep Learning Model...")
    for epoch in range(EPOCHS):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        epoch_train_loss = running_loss / total
        epoch_train_acc = correct / total

        # Validation phase
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        epoch_val_loss = val_loss / val_total
        epoch_val_acc = val_correct / val_total

        history["train_loss"].append(epoch_train_loss)
        history["val_loss"].append(epoch_val_loss)
        history["train_acc"].append(epoch_train_acc)
        history["val_acc"].append(epoch_val_acc)

        print(f"   Epoch [{epoch+1}/{EPOCHS}] -> Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc*100:.1f}% | Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc*100:.1f}%")

        # Save best model checkpoint
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), save_path)
            print(f"   💾 Saved Best Model Checkpoint (Val Loss: {best_val_loss:.4f})")

    # Plot Training & Validation Learning Curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    epochs_range = range(1, EPOCHS + 1)
    
    ax1.plot(epochs_range, history["train_loss"], 'o-', label="Train Loss", color='crimson')
    ax1.plot(epochs_range, history["val_loss"], 'o--', label="Val Loss", color='orange')
    ax1.set_title("Loss Curves")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("CrossEntropy Loss")
    ax1.legend()

    ax2.plot(epochs_range, history["train_acc"], 's-', label="Train Accuracy", color='teal')
    ax2.plot(epochs_range, history["val_acc"], 's--', label="Val Accuracy", color='green')
    ax2.set_title("Accuracy Curves")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.legend()

    plt.suptitle("Phase 6: Transfer Learning Training Dynamics", fontsize=14, fontweight="bold")
    out_fig = FIGURES_DIR / "phase6_learning_curves.png"
    plt.savefig(out_fig, dpi=300)
    plt.close()
    print(f"✅ Saved Learning Curves Chart: {out_fig}")
    print("✅ Phase 6 Complete!\n")
    return model, history

if __name__ == "__main__":
    run_phase6()
