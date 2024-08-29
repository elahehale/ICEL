import torch
import torch.nn.functional as F
import time
from pytorch_grad_cam import HiResCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


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
        _, top2 = outputs.topk(2, dim=1)
        icel_loss = icel_calculation(inputs, hi_res_cam, top2)
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
            print(f'Epoch: [{epoch}][{batch_idx}/{len(trainloader)}] '
                  f'XE Loss {loss.item():.4e} '
                  f'ICEL Loss {lambda_parameter * icel_loss.item():.4e} '
                  f'Acc {100. * correct / total:.2f}%')

            global_step = epoch * len(trainloader) + batch_idx
            writer.add_scalar('Loss/train', loss.item(), global_step)
            writer.add_scalar('Accuracy/train', 100. * correct / total, global_step)
            writer.add_scalar('ICELoss/train', icel_loss.item(), global_step)

    print(f'End of Epoch {epoch}, Total Time: {time.time() - start_time:.3f} seconds')


def validate(net, valloader, criterion, device):
    net.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(valloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            if batch_idx % 10 == 0:
                print(f'[{batch_idx}/{len(valloader)}] '
                      f'XE Loss {loss.item():.4e} '
                      f'Acc {100. * correct / total:.2f}%')

    acc = 100. * correct / total
    avg_loss = total_loss / len(valloader)
    print(f'Validation Loss: {avg_loss:.4f}, Accuracy: {acc:.2f}%')
    return acc, avg_loss


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
    writer.add_scalar('Loss/test', avg_loss, epoch)
    writer.add_scalar('Accuracy/test', acc, epoch)

    return acc
