import torchvision
import torchvision.transforms as transforms
from PIL import Image


transform = transforms.ToPILImage()
mnist_test = torchvision.datasets.MNIST(root="./data", train=False, download=True)

img, label = mnist_test[5]
img.save(f"digit_{label}.png")
print(f"Saved digit_{label}.png (label = {label})")
