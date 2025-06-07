import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 64
LEARNING_RATE = 0.001
NUM_EPOCHS = 10

MODEL_DIR = "./models_checkpoints"
BEST_MODEL_FILENAME = "best_mnist_model.pth"

HYPERPARAMS = {
    "batch_size": BATCH_SIZE,
    "lr": LEARNING_RATE,
    "num_epochs": NUM_EPOCHS
}
