import os  
import torch  
import pandas as pd  
from PIL import Image  
from torchvision import datasets, transforms  
from torch.utils.data import DataLoader, random_split, Dataset  
  
  
# -----------------------------  
# CIFAR-100  
# -----------------------------  
def get_cifar100_dataloaders(data_dir='./data', batch_size=128, val_split=0.1, num_workers=4):  
    mean = (0.5071, 0.4865, 0.4409)  
    std = (0.2673, 0.2564, 0.2762)  
  
    train_transform = transforms.Compose([  
        transforms.RandomCrop(32, padding=4),  
        transforms.RandomHorizontalFlip(),  
        transforms.RandomRotation(15),  
        transforms.ToTensor(),  
        transforms.Normalize(mean, std)  
    ])  
  
    test_transform = transforms.Compose([  
        transforms.ToTensor(),  
        transforms.Normalize(mean, std)  
    ])  
  
    full_train_dataset = datasets.CIFAR100(root=data_dir, train=True, download=True, transform=train_transform)  
    test_dataset = datasets.CIFAR100(root=data_dir, train=False, download=True, transform=test_transform)  
  
    val_size = int(len(full_train_dataset) * val_split)  
    train_size = len(full_train_dataset) - val_size  
  
    train_dataset, val_dataset = random_split(  
        full_train_dataset,  
        [train_size, val_size],  
        generator=torch.Generator().manual_seed(42)  
    )  
  
    # clean val view  
    val_dataset_clean = datasets.CIFAR100(root=data_dir, train=True, download=False, transform=test_transform)  
    val_dataset.dataset = val_dataset_clean  
  
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)  
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)  
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)  
  
    return train_loader, val_loader, test_loader, full_train_dataset  
  
  
# -----------------------------  
# GTSRB  
# -----------------------------  
def get_gtsrb_dataloaders(data_dir='./data', batch_size=128, val_split=0.1, num_workers=4):  
    mean = (0.3337, 0.3064, 0.3171)  
    std = (0.2672, 0.2564, 0.2629)  
  
    train_transform = transforms.Compose([  
        transforms.Resize((32, 32)),  
        transforms.RandomCrop(32, padding=4),  
        transforms.RandomRotation(15),  
        transforms.ToTensor(),  
        transforms.Normalize(mean, std)  
    ])  
  
    test_transform = transforms.Compose([  
        transforms.Resize((32, 32)),  
        transforms.ToTensor(),  
        transforms.Normalize(mean, std)  
    ])  
  
    full_train_dataset = datasets.GTSRB(root=data_dir, split='train', download=True, transform=train_transform)  
    test_dataset = datasets.GTSRB(root=data_dir, split='test', download=True, transform=test_transform)  
  
    val_size = int(len(full_train_dataset) * val_split)  
    train_size = len(full_train_dataset) - val_size  
  
    train_dataset, val_dataset = random_split(  
        full_train_dataset,  
        [train_size, val_size],  
        generator=torch.Generator().manual_seed(42)  
    )  
  
    val_dataset_clean = datasets.GTSRB(root=data_dir, split='train', download=False, transform=test_transform)  
    val_dataset.dataset = val_dataset_clean  
  
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)  
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)  
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)  
  
    return train_loader, val_loader, test_loader, full_train_dataset  
  
  
# -----------------------------  
# VGGFace2 (fixed)  
# -----------------------------  
def get_vggface_dataloaders(  
    data_dir='./data/VGGFace2',  
    batch_size=64,  
    val_split=0.1,  
    num_workers=4  
):  
    """  
    VGGFace2 classification baseline:  
    - Uses ONLY the train identities folder and splits it into train/val  
    - Ensures shared class_to_idx mapping (ImageFolder created from same root)  
    """  
  
    mean = (0.485, 0.456, 0.406)  
    std = (0.229, 0.224, 0.225)  
  
    train_transform = transforms.Compose([  
        transforms.Resize(256),  
        transforms.RandomResizedCrop(224),  
        transforms.RandomHorizontalFlip(),  
        transforms.ToTensor(),  
        transforms.Normalize(mean, std)  
    ])  
  
    val_transform = transforms.Compose([  
        transforms.Resize(256),  
        transforms.CenterCrop(224),  
        transforms.ToTensor(),  
        transforms.Normalize(mean, std)  
    ])  
  
    train_dir = os.path.join(data_dir, 'train')  
    if not os.path.exists(train_dir):  
        raise FileNotFoundError(  
            f"VGGFace2 train directory not found: {train_dir}. "  
            f"Expected structure: {data_dir}/train/<identity>/*.jpg"  
        )  
  
    # One shared mapping  
    full_train = datasets.ImageFolder(root=train_dir, transform=train_transform)  
  
    val_size = int(len(full_train) * val_split)  
    train_size = len(full_train) - val_size  
  
    g = torch.Generator().manual_seed(42)  
    train_subset, val_subset = random_split(full_train, [train_size, val_size], generator=g)  
  
    # Make deterministic val view with SAME mapping  
    full_train_valview = datasets.ImageFolder(root=train_dir, transform=val_transform)  
    val_subset.dataset = full_train_valview  
  
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)  
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)  
  
    # Use val as test for baseline (you can build a 2nd split later)  
    test_loader = val_loader  
  
    return train_loader, val_loader, test_loader, full_train  
  
  
# -----------------------------  
# CheXpert (multi-label)  
# -----------------------------  
class CheXpertMultiLabelDataset(Dataset):  
    """  
    Multi-label CheXpert dataset (5 labels).  
    Returns:  
      image: Tensor [3, 224, 224]  
      target: FloatTensor [5] with 0/1 values  
    """  
  
    def __init__(self, csv_file, root_dir, transform=None, drop_uncertain=True):  
        self.root_dir = root_dir  
        self.transform = transform  
  
        df = pd.read_csv(csv_file)  
  
        # Standard 5 labels (same as you used)  
        self.pathologies = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Pleural Effusion']  
  
        # Handle uncertainty and NaNs  
        df = df.fillna(0.0)  
  
        # CheXpert uses -1 for uncertain.  
        # For baseline we map uncertain -> 0 (or you could treat -1 as 1 depending on protocol).  
        if drop_uncertain:  
            df[self.pathologies] = df[self.pathologies].replace(-1.0, 0.0)  
        else:  
            # If you want to keep -1, you'd need a mask and custom loss.  
            df[self.pathologies] = df[self.pathologies].replace(-1.0, 0.0)  
  
        # Build paths (CSV often contains 'CheXpert-v1.0-small/...')  
        self.image_paths = df['Path'].apply(lambda x: x.replace('CheXpert-v1.0-small/', '')).tolist()  
  
        # Targets as 0/1 float matrix  
        targets = df[self.pathologies].values.astype('float32')  
        targets = (targets > 0.5).astype('float32')  # enforce 0/1  
        self.targets = torch.from_numpy(targets)  
  
    def __len__(self):  
        return len(self.image_paths)  
  
    def __getitem__(self, idx):  
        img_path = os.path.join(self.root_dir, self.image_paths[idx])  
        image = Image.open(img_path).convert('RGB')  
  
        if self.transform:  
            image = self.transform(image)  
  
        target = self.targets[idx]  # FloatTensor [5]  
        return image, target  
  
  
def get_chexpert_dataloaders(data_dir='./data/CheXpert', batch_size=64, val_split=0.1, num_workers=4, print_sizes=False):  
    train_csv = os.path.join(data_dir, 'train.csv')  
    valid_csv = os.path.join(data_dir, 'valid.csv')  
  
    if not os.path.exists(train_csv) or not os.path.exists(valid_csv):  
        raise FileNotFoundError(  
            f"CheXpert CSVs not found in {data_dir}. Expected train.csv and valid.csv"  
        )  
  
    mean = (0.485, 0.456, 0.406)  
    std = (0.229, 0.224, 0.225)  
  
    train_transform = transforms.Compose([  
        transforms.Resize(256),  
        transforms.RandomResizedCrop(224),  
        transforms.RandomHorizontalFlip(),  
        transforms.ToTensor(),  
        transforms.Normalize(mean, std)  
    ])  
  
    val_transform = transforms.Compose([  
        transforms.Resize(256),  
        transforms.CenterCrop(224),  
        transforms.ToTensor(),  
        transforms.Normalize(mean, std)  
    ])  
  
    full_train = CheXpertMultiLabelDataset(csv_file=train_csv, root_dir=data_dir, transform=train_transform)  
    test_dataset = CheXpertMultiLabelDataset(csv_file=valid_csv, root_dir=data_dir, transform=val_transform)  
  
    val_size = int(len(full_train) * val_split)  
    train_size = len(full_train) - val_size  
  
    train_subset, val_subset = random_split(  
        full_train,  
        [train_size, val_size],  
        generator=torch.Generator().manual_seed(42)  
    )  
  
    # Deterministic val view with same underlying CSV rows  
    full_train_valview = CheXpertMultiLabelDataset(csv_file=train_csv, root_dir=data_dir, transform=val_transform)  
    val_subset.dataset = full_train_valview  
  
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)  
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)  
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)  
  
    if print_sizes:  
        print("CheXpert sizes:",  
              "train_full =", len(full_train),  
              "train_split =", len(train_subset),  
              "val_split =", len(val_subset),  
              "test_valid =", len(test_dataset))  
  
    return train_loader, val_loader, test_loader, full_train