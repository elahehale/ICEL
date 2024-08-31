import torch
from torchvision import models, datasets, transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os
import random
from pytorch_grad_cam import GradCAM, HiResCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from torchvision.models import resnet50, ResNet50_Weights
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity

# Create output directory if it doesn't exist
output_dir = './outputs50'
os.makedirs(output_dir, exist_ok=True)

# Choose device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the pretrained ResNet18 model
model = resnet50(weights=ResNet50_Weights.DEFAULT)

model.to(device)  # Move model to the chosen device
model.eval()

# Define the target layers after moving the model to the device
target_layers = [model.layer4[-1].conv3]


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
    if i < 200 :
        images.append(img)
        image_names.append(dataset.samples[i][0].split('/')[-1])  # Store image filenames
    else:
        break

# Create HiResCAM object
cam = HiResCAM(model=model, target_layers=target_layers)

cum_simmilarity = 0
# Generate and save heatmaps
for idx, img_tensor in enumerate(images):
    img_tensor = img_tensor.to(device)  # Move image tensor to the same device as the model
    # Get the model's predictions
    output = model(img_tensor)
    probabilities = F.softmax(output, dim=1)

    # Get top-1 and top-2 classes
    top1_class = torch.argmax(probabilities)
    top2_class = torch.topk(probabilities, 2).indices[0, 1]

    # Create ClassifierOutputTarget instances for top-1 and top-2 classes
    targets_top1 = [ClassifierOutputTarget(top1_class.item())]
    targets_top2 = [ClassifierOutputTarget(top2_class.item())]

    # Generate HiResGradCAM heatmap for top-1 class
    grayscale_cam_top1 = cam(input_tensor=img_tensor, targets=targets_top1)[0]

    # Generate HiResGradCAM heatmap for top-2 class
    grayscale_cam_top2 = cam(input_tensor=img_tensor, targets=targets_top2)[0]

    # Calculate cosine similarity between the two heatmaps
    grayscale_cam_top1_flat = grayscale_cam_top1.flatten()
    grayscale_cam_top2_flat = grayscale_cam_top2.flatten()
    similarity = cosine_similarity(
        [grayscale_cam_top1_flat],
        [grayscale_cam_top2_flat]
    )[0][0]
    cum_simmilarity = cum_simmilarity + similarity

    # Print cosine similarity
    print(f"Cosine similarity between top-1 and top-2 HiResGradCAM heatmaps for image {image_names[idx]}: {similarity}")

    # Prepare the image for visualization
    img_np = img_tensor.squeeze().cpu().numpy().transpose(1, 2, 0)
    img_np = np.clip((img_np * [0.229, 0.224, 0.225]) + [0.485, 0.456, 0.406], 0, 1)

    # Generate visualizations for both top-1 and top-2 classes
    visualization_top1 = show_cam_on_image(img_np, grayscale_cam_top1, use_rgb=True)
    visualization_top2 = show_cam_on_image(img_np, grayscale_cam_top2, use_rgb=True)
    # Plot and save the 2x1 grid visualization
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(visualization_top1)
    axes[0].set_title('Top-1 Class')
    axes[0].axis('off')

    axes[1].imshow(visualization_top2)
    axes[1].set_title('Top-2 Class')
    axes[1].axis('off')

    # Save the grid visualization
    output_path = os.path.join(output_dir, f"HiResGradCAM_{image_names[idx]}")
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"Saved HiResGradCAM grid for image {image_names[idx]} at {output_path}")

print( cum_simmilarity / len(images))