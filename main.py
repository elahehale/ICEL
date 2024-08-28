import os
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter
import torchvision

from model import get_model, load_checkpoint
from dataset import TinyImageNetValDataset
from training import train, validate, test
from utils import parse_args, seed_everything, seed_worker, save_images_from_dataloader


def main():
    args = parse_args()
    writer = SummaryWriter(log_dir='./logs/tiny200-icel-test-structure')
    seed_everything(args.seed)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)
    best_acc = 0
    start_epoch = 0

    transform_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    transform_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    trainset = torchvision.datasets.ImageFolder(root=f'{args.data_dir}/train', transform=transform_train)
    trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                             pin_memory=True, worker_init_fn=seed_worker)

    save_images_from_dataloader(trainloader, 'sample_train_images.png')

    wnid_to_label = {wnid: i for i, wnid in enumerate(trainset.classes)}

    val_annotations_file = os.path.join(args.data_dir, 'val', 'val_annotations.txt')
    val_img_dir = os.path.join(args.data_dir, 'val', 'images')
    testset = TinyImageNetValDataset(val_annotations_file, val_img_dir, wnid_to_label, transform=transform_test)
    testloader = DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                            pin_memory=True, worker_init_fn=seed_worker)

    net = get_model(num_classes=200)
    net = net.to(device)

    if args.resume or args.validate:
        print('Loading model from checkpoint...')
        best_acc, start_epoch = load_checkpoint(net, args.checkpoint)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = SGD(net.parameters(), lr=args.learning_rate, momentum=args.momentum,
                    weight_decay=args.weight_decay)
    scheduler = StepLR(optimizer, step_size=args.lr_step_size, gamma=0.1)
    lambda_parameter = args.l_param

    from pytorch_grad_cam import HiResCAM
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


if __name__ == '__main__':
    main()
