import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from config import DEVICE, HYPERPARAMS, MODEL_DIR, BEST_MODEL_FILENAME
from data.dataloader import get_mnist_loaders
from models.mlp import MLPNet
from models.cnn import SimpleCNN
from utils import train_one_epoch, validate

def main():
    parser = argparse.ArgumentParser(description="Train MNIST digit recognizer")
    parser.add_argument("--model", choices=["mlp", "cnn"], default="mlp",
                        help="Which model to train: 'mlp' or 'cnn'")
    parser.add_argument("--epochs", type=int, default=HYPERPARAMS["num_epochs"],
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=HYPERPARAMS["batch_size"],
                        help="Batch size for training and validation")
    parser.add_argument("--lr", type=float, default=HYPERPARAMS["lr"],
                        help="Learning rate for optimizer")
    args = parser.parse_args()

    os.makedirs(MODEL_DIR, exist_ok=True)
    device = DEVICE

    train_loader, test_loader = get_mnist_loaders(batch_size=args.batch_size)

  
    if args.model == "mlp":
        model = MLPNet().to(device)
    else:
        model = SimpleCNN().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_val_acc = 0.0
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, device, train_loader, optimizer, criterion)
        val_loss, val_acc     = validate(model, device, test_loader,    criterion)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_path = os.path.join(MODEL_DIR, BEST_MODEL_FILENAME)
            torch.save(model.state_dict(), save_path)

        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        print(f"Epoch {epoch:02d}/{args.epochs} | "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%  | "
              f"Val   Loss: {val_loss:.4f}, Val   Acc: {val_acc:.2f}%")

    print(f"\nBest Val Accuracy = {best_val_acc:.2f}%")

    final_loss, final_acc = validate(model, device, test_loader, criterion)
    print(f"Final Test Accuracy: {final_acc:.2f}%")

    epochs = list(range(1, args.epochs + 1))
    plt.figure(figsize=(6,4))
    plt.plot(epochs, train_losses, marker='o', label="Train Loss")
    plt.plot(epochs, val_losses,   marker='o', label="Val Loss")
    plt.title("Loss vs. Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig("loss_plot.png")
    plt.close()

    plt.figure(figsize=(6,4))
    plt.plot(epochs, train_accuracies, marker='o', label="Train Acc")
    plt.plot(epochs, val_accuracies,   marker='o', label="Val Acc")
    plt.title("Accuracy vs. Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.grid(True)
    plt.savefig("accuracy_plot.png")
    plt.close()

    print("Saved plots: loss_plot.png, accuracy_plot.png")

if __name__ == "__main__":
    main()
