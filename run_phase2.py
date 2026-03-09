import os
import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader, Subset

from src.data.datasets import get_cifar100_dataloaders, get_gtsrb_dataloaders, get_vggface_dataloaders, get_chexpert_dataloaders
from src.models.resnet import ResNet18
from src.attacks.witches_brew import WitchesBrewPoisoner
from src.detector.dynadetect import DynaDetectAnomalyScorer

def main():
    parser = argparse.ArgumentParser(description="Phase 2 Pipeline: Poison & Detect")
    parser.add_argument('--dataset', type=str, default='cifar100', choices=['cifar100', 'gtsrb', 'vggface', 'chexpert'])
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--num-poisons', type=int, default=100)
    parser.add_argument('--target-class', type=int, default=0)
    parser.add_argument('--poison-class', type=int, default=1)
    parser.add_argument('--data-dir', type=str, default='./data')
    parser.add_argument('--model-path', type=str, required=True, help="Path to pre-trained baseline model")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load Dataset
    print("Loading Dataset...")
    is_32x32 = True
    is_multilabel = False
    if args.dataset == 'cifar100':
        _, _, _, full_train_dataset = get_cifar100_dataloaders(data_dir=args.data_dir, batch_size=args.batch_size)
        num_classes = 100
    elif args.dataset == 'gtsrb':
        _, _, _, full_train_dataset = get_gtsrb_dataloaders(data_dir=args.data_dir, batch_size=args.batch_size)
        num_classes = 43
    elif args.dataset == 'vggface':
        _, _, _, full_train_dataset = get_vggface_dataloaders(data_dir=os.path.join(args.data_dir, 'VGGFace2'), batch_size=args.batch_size)
        num_classes = len(full_train_dataset.classes)
        is_32x32 = False
    elif args.dataset == 'chexpert':
        _, _, _, full_train_dataset = get_chexpert_dataloaders(data_dir=os.path.join(args.data_dir, 'CheXpert'), batch_size=args.batch_size)
        num_classes = 5
        is_32x32 = False
        is_multilabel = True
    
    # Load Model
    print(f"Loading Model from {args.model_path}...")
    model = ResNet18(num_classes=num_classes, is_32x32=is_32x32).to(device)
    model.load_state_dict(torch.load(args.model_path))
    model.eval()

    # Step 1: Generate Poisoned Data
    print("\n--- Starting Witches' Brew Attack Simulation ---")
    target_img = None
    target_label = None
    poison_indices = []
    clean_images = []
    poison_labels = []

    # Find target and poison base images
    for idx, (img, label) in enumerate(full_train_dataset):
        if is_multilabel:
            is_target = (label[args.target_class].item() == 1.0)
            is_poison = (label[args.poison_class].item() == 1.0 and label[args.target_class].item() == 0.0)
        else:
            is_target = (label == args.target_class)
            is_poison = (label == args.poison_class)

        if is_target and target_img is None:
            target_img = img
            target_label = label
        elif is_poison and len(poison_indices) < args.num_poisons:
            poison_indices.append(idx)
            clean_images.append(img)
            poison_labels.append(label)
            
        if target_img is not None and len(poison_indices) == args.num_poisons:
            break

    if target_img is None:
        raise ValueError(f"Could not find a target image for class {args.target_class}")
    if len(poison_indices) < args.num_poisons:
        raise ValueError(f"Could not find {args.num_poisons} images for class {args.poison_class}")

    clean_images_tensor = torch.stack(clean_images)
    if is_multilabel:
        poison_labels_tensor = torch.stack(poison_labels)
    else:
        poison_labels_tensor = torch.tensor(poison_labels)

    criterion = nn.BCEWithLogitsLoss() if is_multilabel else nn.CrossEntropyLoss()
    poisoner = WitchesBrewPoisoner(model=model, epsilon=16/255, steps=50, criterion=criterion)
    
    poisoned_images = poisoner.generate_poisons(
        poison_images=clean_images_tensor,
        poison_labels=poison_labels_tensor,
        target_img=target_img,
        target_label=target_label,
        device=device
    )

    print(f"Successfully generated {args.num_poisons} poisoned images.")

    # Create a mixed dataset for detection (Replace original images with poisoned ones)
    print("\n--- Injecting Poisons into Dataset ---")
    # For simplicity in this script, we'll create a new dataset class wrapper or just manipulate tensors.
    # We will use the original train_dataset but overwrite the targeted indices.
    
    # Normally we would create a custom Dataset wrapper here, but for quick testing:
    # We will just evaluate DynaDetect on the poisoned images directly to see their scores.
    
    # Step 2: DynaDetect2.0 Scoring
    print("\n--- Starting DynaDetect2.0 Anomaly Detection ---")
    detector = DynaDetectAnomalyScorer(model=model, num_classes=num_classes)
    
    # Use a subset of clean data to build the Mahalanobis distribution
    clean_indices = [i for i in range(5000) if i not in poison_indices and i < len(full_train_dataset)]
    clean_subset = Subset(full_train_dataset, clean_indices)
    clean_loader = DataLoader(clean_subset, batch_size=args.batch_size, shuffle=False)
    
    detector.fit_distributions(clean_loader, device)

    # Score the clean data
    clean_distances, _, _ = detector.score_samples(clean_loader, device)
    
    # Score the poisoned data
    # Create a quick dataloader for the poisoned tensors
    poison_dataset = torch.utils.data.TensorDataset(poisoned_images, poison_labels_tensor)
    poison_loader = DataLoader(poison_dataset, batch_size=args.batch_size, shuffle=False)
    
    poison_distances, _, _ = detector.score_samples(poison_loader, device)

    # Output Results
    print("\n--- Detection Results ---")
    clean_mean = np.mean(clean_distances)
    poison_mean = np.mean(poison_distances)
    
    print(f"Average Mahalanobis Distance (Clean): {clean_mean:.4f}")
    print(f"Average Mahalanobis Distance (Poisoned): {poison_mean:.4f}")
    
    threshold = np.percentile(clean_distances, 95)
    print(f"Detection Threshold (95th percentile of clean): {threshold:.4f}")
    
    detected_poisons = np.sum(poison_distances > threshold)
    detection_rate = (detected_poisons / args.num_poisons) * 100
    print(f"Poison Detection Rate: {detection_rate:.2f}% ({detected_poisons}/{args.num_poisons})")
    
    print("\nPhase 2 Complete. We now have flagged poisoned samples ready to be repurposed as watermarks.")

if __name__ == '__main__':
    main()
