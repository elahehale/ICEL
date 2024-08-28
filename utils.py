import torch
import numpy as np
import random
import torch.backends.cudnn as cudnn
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Train ResNet18 on Tiny ImageNet')
    parser.add_argument('--batch-size', default=256, type=int, help='Batch size for training')
    parser.add_argument('--l-param', default=0.5, type=float, help='Lambda parameter for regularization term')
    parser.add_argument('--epochs', default=90, type=int, help='Number of epochs to train')
    parser.add_argument('--learning-rate', default=0.001, type=float, help='Learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, help='Momentum for SGD')
    parser.add_argument('--lr-step-size', default=30, type=int, help='Learning rate reduction checkpoints')
    parser.add_argument('--weight-decay', default=5e-4, type=float, help='Weight decay for SGD')
    parser.add_argument('--data-dir', default='./tiny-imagenet-200', type=str, help='Directory of Tiny ImageNet')
    parser.add_argument('--save-dir', default='./checkpoints', type=str, help='Directory to save checkpoints')
    parser.add_argument('--resume', action='store_true', help='Resume training from checkpoint')
    parser.add_argument('--checkpoint', default='./checkpoints/ckpt.pth', type=str, help='Path to checkpoint')
    parser.add_argument('--validate', action='store_true', help='Only validate the model using the saved checkpoint')
    parser.add_argument('--num-workers', default=8, type=int, help='Number of data loader workers')
    parser.add_argument('--seed', default=42, type=int, help='Random seed')
    return parser.parse_args()


def seed_everything(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def save_images_from_dataloader(dataloader, file_path, num_images=16):
    dataiter = iter(dataloader)
    images, _ = dataiter.next()
    images = images[:num_images]
    img_grid = vutils.make_grid(images, nrow=4, normalize=True)
    plt.figure(figsize=(8, 8))
    plt.imshow(img_grid.permute(1, 2, 0))
    plt.axis('off')
    plt.savefig(file_path)
    plt.close()
