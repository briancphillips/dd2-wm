import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from collections import Counter

from src.data.datasets import (
    get_cifar100_dataloaders,
    get_gtsrb_dataloaders,
    get_vggface_dataloaders,
    get_chexpert_dataloaders
)
from src.models.resnet import ResNet18


# -----------------------------
# Metrics helpers
# -----------------------------
@torch.no_grad()
def multilabel_metrics_from_logits(logits, targets, threshold=0.5):
    """
    logits:  [B, C]
    targets: [B, C] float {0,1}
    Returns raw counts for epoch-level aggregation:
      correct_labels, total_labels, exact_matches, tp, fp, fn
    """
    probs = torch.sigmoid(logits)
    preds = (probs >= threshold).float()

    correct_labels = (preds == targets).float().sum().item()
    total_labels = targets.numel()

    exact_matches = preds.eq(targets).all(dim=1).float().sum().item()

    tp = (preds * targets).sum(dim=0)
    fp = (preds * (1 - targets)).sum(dim=0)
    fn = ((1 - preds) * targets).sum(dim=0)

    return correct_labels, total_labels, exact_matches, tp, fp, fn


def inspect_split_singlelabel(name, loader, max_batches=10):
    ys = []
    seen = 0
    for i, (_, y) in enumerate(loader):
        ys.append(y)
        seen += y.numel()
        if i + 1 >= max_batches:
            break
    ys = torch.cat(ys).cpu()
    c = Counter(ys.tolist())
    print(f"\n== {name} (first {seen} labels from {max_batches} batches) ==")
    print("unique labels:", ys.unique().numel())
    print("label min/max:", int(ys.min()), int(ys.max()))
    print("most common:", c.most_common(5))


# -----------------------------
# Train/Eval
# -----------------------------
def train_epoch_singlelabel(model, loader, criterion, optimizer, device, epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch} [Train]")
    for inputs, targets in pbar:
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        pbar.set_postfix({'loss': loss.item(), 'acc': 100.0 * correct / total})

    epoch_loss = running_loss / total
    epoch_acc = 100.0 * correct / total
    return epoch_loss, epoch_acc


@torch.no_grad()
def evaluate_singlelabel(model, loader, criterion, device, epoch, mode="Val"):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch} [{mode}]")
    for inputs, targets in pbar:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        pbar.set_postfix({'loss': loss.item(), 'acc': 100.0 * correct / total})

    epoch_loss = running_loss / total
    epoch_acc = 100.0 * correct / total
    return epoch_loss, epoch_acc


def train_epoch_multilabel(model, loader, criterion, optimizer, device, epoch):
    model.train()
    running_loss = 0.0
    total_samples = 0

    # running metrics
    total_correct_labels = 0
    total_labels = 0
    total_exact_matches = 0
    tp_sum = None
    fp_sum = None
    fn_sum = None

    pbar = tqdm(loader, desc=f"Epoch {epoch} [Train]")
    for inputs, targets in pbar:
        inputs = inputs.to(device)
        targets = targets.to(device)  # float [B, C]

        optimizer.zero_grad()
        logits = model(inputs)        # [B, C]
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        bs = inputs.size(0)
        running_loss += loss.item() * bs
        total_samples += bs

        correct_labels, batch_total_labels, exact_matches, tp, fp, fn = multilabel_metrics_from_logits(logits.detach(), targets.detach())
        total_correct_labels += correct_labels
        total_labels += batch_total_labels
        total_exact_matches += exact_matches

        if tp_sum is None:
            tp_sum = tp
            fp_sum = fp
            fn_sum = fn
        else:
            tp_sum += tp
            fp_sum += fp
            fn_sum += fn

        cur_label_acc = 100.0 * (total_correct_labels / total_labels) if total_labels > 0 else 0.0
        
        eps = 1e-8
        precision = tp_sum / (tp_sum + fp_sum + eps)
        recall = tp_sum / (tp_sum + fn_sum + eps)
        f1s = (2 * precision * recall) / (precision + recall + eps)
        cur_macro_f1 = f1s.mean().item()

        pbar.set_postfix({
            'loss': loss.item(),
            'label_acc': f"{cur_label_acc:.1f}%",
            'macro_f1': f"{cur_macro_f1:.3f}"
        })

    epoch_loss = running_loss / total_samples if total_samples > 0 else 0.0
    epoch_label_acc = 100.0 * (total_correct_labels / total_labels) if total_labels > 0 else 0.0
    epoch_example_acc = 100.0 * (total_exact_matches / total_samples) if total_samples > 0 else 0.0
    
    eps = 1e-8
    precision = tp_sum / (tp_sum + fp_sum + eps)
    recall = tp_sum / (tp_sum + fn_sum + eps)
    f1s = (2 * precision * recall) / (precision + recall + eps)
    epoch_macro_f1 = f1s.mean().item()

    return epoch_loss, epoch_label_acc, epoch_example_acc, epoch_macro_f1


@torch.no_grad()
def evaluate_multilabel(model, loader, criterion, device, epoch, mode="Val"):
    model.eval()
    running_loss = 0.0
    total_samples = 0

    total_correct_labels = 0
    total_labels = 0
    total_exact_matches = 0
    tp_sum = None
    fp_sum = None
    fn_sum = None

    pbar = tqdm(loader, desc=f"Epoch {epoch} [{mode}]")
    for inputs, targets in pbar:
        inputs = inputs.to(device)
        targets = targets.to(device)

        logits = model(inputs)
        loss = criterion(logits, targets)

        bs = inputs.size(0)
        running_loss += loss.item() * bs
        total_samples += bs

        correct_labels, batch_total_labels, exact_matches, tp, fp, fn = multilabel_metrics_from_logits(logits, targets)
        total_correct_labels += correct_labels
        total_labels += batch_total_labels
        total_exact_matches += exact_matches

        if tp_sum is None:
            tp_sum = tp
            fp_sum = fp
            fn_sum = fn
        else:
            tp_sum += tp
            fp_sum += fp
            fn_sum += fn

        cur_label_acc = 100.0 * (total_correct_labels / total_labels) if total_labels > 0 else 0.0
        
        eps = 1e-8
        precision = tp_sum / (tp_sum + fp_sum + eps)
        recall = tp_sum / (tp_sum + fn_sum + eps)
        f1s = (2 * precision * recall) / (precision + recall + eps)
        cur_macro_f1 = f1s.mean().item()

        pbar.set_postfix({
            'loss': loss.item(),
            'label_acc': f"{cur_label_acc:.1f}%",
            'macro_f1': f"{cur_macro_f1:.3f}"
        })

    epoch_loss = running_loss / total_samples if total_samples > 0 else 0.0
    epoch_label_acc = 100.0 * (total_correct_labels / total_labels) if total_labels > 0 else 0.0
    epoch_example_acc = 100.0 * (total_exact_matches / total_samples) if total_samples > 0 else 0.0
    
    eps = 1e-8
    precision = tp_sum / (tp_sum + fp_sum + eps)
    recall = tp_sum / (tp_sum + fn_sum + eps)
    f1s = (2 * precision * recall) / (precision + recall + eps)
    epoch_macro_f1 = f1s.mean().item()

    return epoch_loss, epoch_label_acc, epoch_example_acc, epoch_macro_f1


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="Baseline Model Training")
    parser.add_argument('--dataset', type=str, default='cifar100', choices=['cifar100', 'gtsrb', 'vggface', 'chexpert'])
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--data-dir', type=str, default='./data')
    parser.add_argument('--out-dir', type=str, default='./checkpoints')
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--diagnose', action='store_true', help="Print quick dataset diagnostics and continue.")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    is_32x32 = True
    is_multilabel = False

    # Load Data
    if args.dataset == 'cifar100':
        train_loader, val_loader, test_loader, _ = get_cifar100_dataloaders(
            data_dir=args.data_dir, batch_size=args.batch_size, num_workers=args.num_workers
        )
        num_classes = 100

    elif args.dataset == 'gtsrb':
        train_loader, val_loader, test_loader, _ = get_gtsrb_dataloaders(
            data_dir=args.data_dir, batch_size=args.batch_size, num_workers=args.num_workers
        )
        num_classes = 43

    elif args.dataset == 'vggface':
        train_loader, val_loader, test_loader, _ = get_vggface_dataloaders(
            data_dir=os.path.join(args.data_dir, 'VGGFace2'),
            batch_size=args.batch_size,
            num_workers=args.num_workers
        )
        is_32x32 = False

        # derive num_classes dynamically
        train_ds = getattr(train_loader.dataset, "dataset", train_loader.dataset)
        num_classes = len(train_ds.classes)

        if args.diagnose:
            inspect_split_singlelabel("train", train_loader)
            inspect_split_singlelabel("val", val_loader)
            val_ds = getattr(val_loader.dataset, "dataset", val_loader.dataset)
            if hasattr(train_ds, "classes") and hasattr(val_ds, "classes"):
                overlap = set(train_ds.classes) & set(val_ds.classes)
                print("\ntrain num classes:", len(train_ds.classes))
                print("val   num classes:", len(val_ds.classes))
                print("identity overlap train∩val:", len(overlap))

    elif args.dataset == 'chexpert':
        # CheXpert is multi-label
        train_loader, val_loader, test_loader, _ = get_chexpert_dataloaders(
            data_dir=os.path.join(args.data_dir, 'CheXpert'),
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            print_sizes=args.diagnose
        )
        num_classes = 5
        is_32x32 = False
        is_multilabel = True

    # Initialize Model
    model = ResNet18(num_classes=num_classes, is_32x32=is_32x32).to(device)

    # Loss/Optim
    if is_multilabel:
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    writer = SummaryWriter(log_dir=os.path.join(args.out_dir, f'logs_{args.dataset}'))

    best_metric = -1e9
    best_epoch = -1
    best_path = os.path.join(args.out_dir, f'best_{args.dataset}_resnet18.pth')

    for epoch in range(1, args.epochs + 1):
        if is_multilabel:
            train_loss, train_label_acc, train_ex_acc, train_macro_f1 = train_epoch_multilabel(
                model, train_loader, criterion, optimizer, device, epoch
            )
            val_loss, val_label_acc, val_ex_acc, val_macro_f1 = evaluate_multilabel(
                model, val_loader, criterion, device, epoch, mode="Val"
            )

            scheduler.step()

            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalar('Loss/val', val_loss, epoch)
            writer.add_scalar('LabelAcc/train', train_label_acc, epoch)
            writer.add_scalar('LabelAcc/val', val_label_acc, epoch)
            writer.add_scalar('ExampleAcc/train', train_ex_acc, epoch)
            writer.add_scalar('ExampleAcc/val', val_ex_acc, epoch)
            writer.add_scalar('MacroF1/train', train_macro_f1, epoch)
            writer.add_scalar('MacroF1/val', val_macro_f1, epoch)
            writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)

            print(
                f"Epoch {epoch}/{args.epochs} - "
                f"Train Loss: {train_loss:.4f} LabelAcc: {train_label_acc:.2f}% MacroF1: {train_macro_f1:.3f} | "
                f"Val Loss: {val_loss:.4f} LabelAcc: {val_label_acc:.2f}% MacroF1: {val_macro_f1:.3f}"
            )

            # Save best by val_label_acc (or swap to macro_f1 if you prefer)
            current_metric = val_label_acc

        else:
            train_loss, train_acc = train_epoch_singlelabel(model, train_loader, criterion, optimizer, device, epoch)
            val_loss, val_acc = evaluate_singlelabel(model, val_loader, criterion, device, epoch, mode="Val")

            scheduler.step()

            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalar('Accuracy/train', train_acc, epoch)
            writer.add_scalar('Loss/val', val_loss, epoch)
            writer.add_scalar('Accuracy/val', val_acc, epoch)
            writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)

            print(
                f"Epoch {epoch}/{args.epochs} - "
                f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | "
                f"Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}%"
            )

            current_metric = val_acc

        # Save best model
        if current_metric > best_metric:
            print(f"Validation metric improved from {best_metric:.4f} to {current_metric:.4f}. Saving model...")
            best_metric = current_metric
            best_epoch = epoch
            torch.save(model.state_dict(), best_path)

    # Final Test
    print(f"Loading best model (epoch {best_epoch}, best_metric={best_metric:.4f}) for final testing...")
    model.load_state_dict(torch.load(best_path, map_location=device))

    if is_multilabel:
        test_loss, test_label_acc, test_ex_acc, test_macro_f1 = evaluate_multilabel(
            model, test_loader, criterion, device, epoch="Final", mode="Test"
        )
        print(
            f"Final Test - Loss: {test_loss:.4f} "
            f"LabelAcc: {test_label_acc:.2f}% "
            f"ExampleAcc: {test_ex_acc:.2f}% "
            f"MacroF1: {test_macro_f1:.3f}"
        )
    else:
        test_loss, test_acc = evaluate_singlelabel(model, test_loader, criterion, device, epoch="Final", mode="Test")
        print(f"Final Test Accuracy: {test_acc:.2f}%")

    writer.close()

if __name__ == '__main__':
    main()
