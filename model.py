import torch
from torchvision.models import resnet18


def get_model(num_classes=200):
    net = resnet18(pretrained=False, num_classes=num_classes)
    return net


def load_checkpoint(net, checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    net.load_state_dict(checkpoint['net'])
    return checkpoint['acc'], checkpoint['epoch']
