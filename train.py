import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from model import build_se_resnet50
from utils import cutmix_data, SoftLabelCrossEntropyLoss


def get_dataloaders(batch_size=128):
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])

    train_dataset = torchvision.datasets.CIFAR100(
        root='./data', train=True, download=True, transform=train_transform
    )
    test_dataset = torchvision.datasets.CIFAR100(
        root='./data', train=False, download=True, transform=test_transform
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    return train_loader, test_loader


def train_one_epoch(model, train_loader, optimizer, device, cutmix_prob=0.5, alpha=1.0, max_grad_norm=1.0):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    criterion = nn.CrossEntropyLoss()
    soft_criterion = SoftLabelCrossEntropyLoss()

    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()

        r = np.random.rand(1)
        if r < cutmix_prob:
            inputs, targets_a, targets_b, lam = cutmix_data(inputs, targets, alpha)
            outputs = model(inputs)
            loss = soft_criterion(outputs, targets_a, targets_b, lam)
        else:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            targets_a = targets
            targets_b = targets
            lam = 1.0

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += (lam * predicted.eq(targets_a).sum().float()
                    + (1 - lam) * predicted.eq(targets_b).sum().float()).item()

    epoch_loss = running_loss / total
    epoch_acc = 100.0 * correct / total
    return epoch_loss, epoch_acc


def evaluate(model, test_loader, device):
    model.eval()
    running_loss = 0.0
    correct_top1 = 0
    correct_top5 = 0
    total = 0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct_top1 += predicted.eq(targets).sum().item()

            _, top5_pred = outputs.topk(5, dim=1, largest=True, sorted=True)
            correct_top5 += top5_pred.eq(targets.view(-1, 1).expand_as(top5_pred)).sum().item()

    epoch_loss = running_loss / total
    top1_acc = 100.0 * correct_top1 / total
    top5_acc = 100.0 * correct_top5 / total
    return epoch_loss, top1_acc, top5_acc


def plot_results(train_losses, val_losses, train_accs, val_accs):
    epochs = range(1, len(train_losses) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(epochs, train_losses, 'b-', label='Train Loss')
    ax1.plot(epochs, val_losses, 'r-', label='Val Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)

    ax2.plot(epochs, train_accs, 'b-', label='Train Acc')
    ax2.plot(epochs, val_accs, 'r-', label='Val Acc')
    ax2.set_title('Training and Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig('training_curves.png', dpi=150, bbox_inches='tight')
    plt.close()


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    num_epochs = 50
    batch_size = 128
    learning_rate = 0.1
    cutmix_prob = 0.5
    cutmix_alpha = 1.0

    train_loader, test_loader = get_dataloaders(batch_size)

    model = build_se_resnet50(num_classes=100).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)
    warmup_epochs = 5
    warmup_scheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs)
    cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs - warmup_epochs)
    scheduler = optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_epochs]
    )

    train_losses, val_losses = [], []
    train_accs, val_accs, val_top5_accs = [], [], []
    best_acc = 0.0

    for epoch in range(1, num_epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, device, cutmix_prob, cutmix_alpha
        )
        val_loss, val_acc, val_top5 = evaluate(model, test_loader, device)
        scheduler.step()

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        val_top5_accs.append(val_top5)

        print(f"Epoch [{epoch}/{num_epochs}] "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f} | Val Acc(Top-1): {val_acc:.2f}% | Val Acc(Top-5): {val_top5:.2f}%")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')

    print(f"\nBest Validation Accuracy (Top-1): {best_acc:.2f}%")

    plot_results(train_losses, val_losses, train_accs, val_accs)
    print("Training curves saved to training_curves.png")


if __name__ == '__main__':
    main()
