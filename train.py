import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from PIL import Image
from torch.utils.data import Dataset
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.models import resnet18
import torch.nn.functional as F
from pytorch_grad_cam import HiResCAM, GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import os
import time
import matplotlib.pyplot as plt
import torchvision.utils as vutils
from torch.optim.lr_scheduler import StepLR
import random
import numpy as np
from torch.utils.tensorboard import SummaryWriter


def save_images_from_dataloader(dataloader, file_path, num_images=16):
    # Get a batch of images
    dataiter = iter(dataloader)
    images, _ = dataiter.next()

    # Select only the specified number of images
    images = images[:num_images]

    # Make a grid of images
    img_grid = vutils.make_grid(images, nrow=4, normalize=True)

    # Plot the grid
    plt.figure(figsize=(8, 8))
    plt.imshow(img_grid.permute(1, 2, 0))  # Convert from Tensor format to (H, W, C) format
    plt.axis('off')  # Turn off axis labels

    # Save the plot as an image file
    plt.savefig(file_path)
    plt.close()


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


def validate(net, valloader, criterion, device):
    net.eval()  # Set the model to evaluation mode
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():  # Disable gradient calculation
        for batch_idx, (inputs, targets) in enumerate(valloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)

            loss = criterion(outputs, targets)
            total_loss += loss.item()
            _, predicted = outputs.max(1)

            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            if batch_idx % 10 == 0:
                print(predicted)
                print('-------------')
                print(targets)
                print(f'[{batch_idx}/{len(valloader)}] '
                      f'XE Loss {loss.item():.4e} '
                      f'Acc {100. * correct / total:.2f}%')
    acc = 100. * correct / total
    avg_loss = total_loss / len(valloader)
    print(f'Validation Loss: {avg_loss:.4f}, Accuracy: {acc:.2f}%')

    return acc, avg_loss


def icel_calculation(images, hi_res_cam, top2):
    targets_top1 = [ClassifierOutputTarget(top2[i, 0].item()) for i in range(images.size(0))]
    targets_top2 = [ClassifierOutputTarget(top2[i, 1].item()) for i in range(images.size(0))]

    grayscale_cams_top1 = hi_res_cam(input_tensor=images, targets=targets_top1)
    grayscale_cams_top2 = hi_res_cam(input_tensor=images, targets=targets_top2)

    cosine_similarities = []

    for idx in range(images.size(0)):
        cam_top1 = grayscale_cams_top1[idx]
        cam_top2 = grayscale_cams_top2[idx]

        cam_top1_flat = torch.tensor(cam_top1.flatten(), dtype=torch.float32)
        cam_top2_flat = torch.tensor(cam_top2.flatten(), dtype=torch.float32)

        cosine_similarity = F.cosine_similarity(cam_top1_flat.unsqueeze(0), cam_top2_flat.unsqueeze(0), dim=1)
        cosine_similarities.append(cosine_similarity)

    return torch.mean(torch.stack(cosine_similarities))


def train(epoch, net, trainloader, optimizer, criterion, device, lambda_parameter, hi_res_cam, writer):
    net.train()
    total_loss = 0
    correct = 0
    total = 0

    start_time = time.time()

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        batch_start_time = time.time()

        inputs, targets = inputs.to(device), targets.to(device)
        outputs = net(inputs)

        # if torch.isnan(outputs).any() or torch.isinf(outputs).any():
        #     print("Found NaN or Inf in the outputs during training!")

        _, top2 = outputs.topk(2, dim=1)

        icel_loss = icel_calculation(inputs, hi_res_cam, top2)
        # icel_loss =0
        loss = criterion(outputs, targets) + lambda_parameter * icel_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        batch_time = time.time() - batch_start_time
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        if batch_idx % 10 == 0:
            print(f'Epoch: [{epoch}][{batch_idx}/{len(trainloader)}] {predicted.eq(targets).sum().item()} '
                  f'Time: {batch_time:.3f} ({time.time() - start_time:.3f})  '
                  f'XE Loss {loss.item():.4e} '
                  f'ICEL Loss { lambda_parameter * icel_loss.item():.4e} '
                  f'Acc {100. * correct / total:.2f}%')

            # Log to TensorBoard
            global_step = epoch * len(trainloader) + batch_idx
            writer.add_scalar('Loss/train', loss.item(), global_step)
            writer.add_scalar('Accuracy/train', 100. * correct / total, global_step)

    print(f'End of Epoch {epoch}, Total Time: {time.time() - start_time:.3f} seconds')


def test(epoch, net, testloader, criterion, device, writer):
    net.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    acc = 100. * correct / total
    avg_loss = total_loss / len(testloader)

    print(f'Test Epoch {epoch}, Loss: {avg_loss:.4f}, Accuracy: {acc:.2f}%')

    # Log to TensorBoard
    writer.add_scalar('Loss/test', avg_loss, epoch)
    writer.add_scalar('Accuracy/test', acc, epoch)

    return acc


def main():
    args = parse_args()
    writer = SummaryWriter(log_dir='./logs/tiny200-icel-test')

    seed_everything(args.seed)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)
    best_acc = 0
    start_epoch = 0

    # Transforms for training
    transform_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    # Transforms for testing
    transform_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    trainset = torchvision.datasets.ImageFolder(root=f'{args.data_dir}/train', transform=transform_train)
    trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                             pin_memory=True, worker_init_fn=seed_worker)

    save_images_from_dataloader(trainloader, 'sample_train_images.png')
    #
    # testset = torchvision.datasets.ImageFolder(root=f'{args.data_dir}/val', transform=transform_test)
    # testloader = DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
    #                         pin_memory=True, worker_init_fn=seed_worker)

    wnid_to_label = {wnid: i for i, wnid in enumerate(trainset.classes)}

    # Load validation data using custom dataset
    val_annotations_file = os.path.join(args.data_dir, 'val', 'val_annotations.txt')
    val_img_dir = os.path.join(args.data_dir, 'val', 'images')
    testset = TinyImageNetValDataset(val_annotations_file, val_img_dir, wnid_to_label, transform=transform_test)
    testloader = DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                            pin_memory=True, worker_init_fn=seed_worker)

    net = resnet18(pretrained=False, num_classes=200)
    net = net.to(device)

    if args.resume or args.validate:
        print('Loading model from checkpoint...')
        checkpoint = torch.load(args.checkpoint)
        net.load_state_dict(checkpoint['net'])
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch']

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.learning_rate, momentum=args.momentum,
                          weight_decay=args.weight_decay)

    scheduler = StepLR(optimizer, step_size=args.lr_step_size, gamma=0.1)
    lambda_parameter = args.l_param

    target_layers = [net.layer4[-1]]
    hi_res_cam = HiResCAM(model=net, target_layers=target_layers)

    if args.validate:
        validate(net, testloader, criterion, device)
        return  # Exit after validation

    for epoch in range(start_epoch, start_epoch + args.epochs):
        train(epoch, net, trainloader, optimizer, criterion, device, lambda_parameter, hi_res_cam, writer)
        acc = test(epoch, net, testloader, criterion, device, writer)
        print(f'Saving checkpoint for epoch {epoch + 1}...')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        os.makedirs(args.save_dir, exist_ok=True)
        checkpoint_path = f'{args.save_dir}/ckpt_epoch_{epoch + 1}.pth'
        torch.save(state, checkpoint_path)
        if acc > best_acc:
            best_acc = acc
        scheduler.step()


class TinyImageNetValDataset(Dataset):
    def __init__(self, annotations_file, img_dir, wnid_to_label, transform=None):
        self.img_labels = []
        with open(annotations_file, 'r') as f:
            for line in f:
                img_file, wnid = line.split('\t')[:2]
                label = wnid_to_label[wnid]
                self.img_labels.append((img_file, label))

        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels[idx][0])
        image = Image.open(img_path).convert("RGB")
        label = self.img_labels[idx][1]
        if self.transform:
            image = self.transform(image)
        return image, label


if __name__ == '__main__':
    main()
