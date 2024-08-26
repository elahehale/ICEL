import torch
from torchvision import models, datasets, transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os
import random
from pytorch_grad_cam import GradCAM, HiResCAM
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image

# Create output directory if it doesn't exist
output_dir = './outputs'
os.makedirs(output_dir, exist_ok=True)

# Load the pretrained ResNet18 model
model = models.resnet18(pretrained=True)
model.eval()

# Define the target layers
target_layers = [model.layer4[-1]]

# Define the transform to apply to the images
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load the Tiny ImageNet training set
dataset = datasets.ImageFolder(root='../GC-loss/tiny-imagenet/tiny-imagenet-200/train', transform=transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)

# Randomly select 6 images from the dataset
images = []
image_names = []
for i, (img, label) in enumerate(dataloader):
    if len(images) < 6:
        images.append(img)
        image_names.append(dataset.samples[i][0].split('/')[-1])  # Store image filenames
    else:
        break

# Create HiResCAM object
cam = HiResCAM(model=model, target_layers=target_layers)

# Generate and save heatmaps
for idx, img_tensor in enumerate(images):
    img_tensor = img_tensor.to('cuda' if torch.cuda.is_available() else 'cpu')

    # Generate HiResGradCAM heatmap
    grayscale_cam = cam(input_tensor=img_tensor, targets=None)[0]

    # Prepare the image for visualization
    img_np = img_tensor.squeeze().cpu().numpy().transpose(1, 2, 0)
    img_np = np.clip((img_np * [0.229, 0.224, 0.225]) + [0.485, 0.456, 0.406], 0, 1)
    visualization = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)

    # Save the visualization
    output_path = os.path.join(output_dir, f"HiResGradCAM_{image_names[idx]}")
    plt.imsave(output_path, visualization)
    print(f"Saved HiResGradCAM for image {image_names[idx]} at {output_path}")
