import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.models import resnet18

def parse_args():
    parser = argparse.ArgumentParser(description='Train ResNet18 on Tiny ImageNet')
    parser.add_argument('--batch-size', default=128, type=int, help='Batch size for training')
    parser.add_argument('--epochs', default=50, type=int, help='Number of epochs to train')
    parser.add_argument('--learning-rate', default=0.01, type=float, help='Learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, help='Momentum for SGD')
    parser.add_argument('--weight-decay', default=5e-4, type=float, help='Weight decay for SGD')
    parser.add_argument('--data-dir', default='./tiny-imagenet-200', type=str, help='Directory of Tiny ImageNet')
    parser.add_argument('--save-dir', default='./checkpoints', type=str, help='Directory to save checkpoints')
    parser.add_argument('--resume', action='store_true', help='Resume training from checkpoint')
    parser.add_argument('--checkpoint', default='./checkpoints/ckpt.pth', type=str, help='Path to checkpoint')
    parser.add_argument('--num-workers', default=4, type=int, help='Number of data loader workers')
    return parser.parse_args()


def train(epoch, net, trainloader, optimizer, criterion, device):
    net.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        if batch_idx % 10 == 0:
            print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item()}, Accuracy: {100.*correct/total}%')

def test(epoch, net, testloader, criterion, device):
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

    print(f'Test Epoch {epoch}, Loss: {total_loss/len(testloader)}, Accuracy: {100.*correct/total}%')

def main():
    args = parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    best_acc = 0
    start_epoch = 0

    # Data preparation
    transform_train = transforms.Compose([
        transforms.RandomCrop(64, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    print(args)
    trainset = torchvision.datasets.ImageFolder(root=f'{args.data_dir}/train', transform=transform_train)
    trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    testset = torchvision.datasets.ImageFolder(root=f'{args.data_dir}/val', transform=transform_test)
    testloader = DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # Model
    net = resnet18(pretrained=False, num_classes=200)
    net = net.to(device)

    if args.resume:
        print('Resuming from checkpoint...')
        checkpoint = torch.load(args.checkpoint)
        net.load_state_dict(checkpoint['net'])
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch']

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)

    for epoch in range(start_epoch, start_epoch + args.epochs):
        train(epoch, net, trainloader, optimizer, criterion, device)
        test(epoch, net, testloader, criterion, device)

        # Save checkpoint
        acc = 100.*correct / total
        if acc > best_acc:
            print('Saving checkpoint...')
            state = {
                'net': net.state_dict(),
                'acc': acc,
                'epoch': epoch,
            }
            if not os.path.isdir(args.save_dir):
                os.mkdir(args.save_dir)
            torch.save(state, f'{args.save_dir}/ckpt.pth')
            best_acc = acc

if __name__ == '__main__':
    main()
