import os
from PIL import Image
import torch
from torchvision import transforms, datasets

import os
from PIL import Image
import torch
from torchvision import transforms, datasets

def load_dataset(root, transform=None):
    """Load dataset from given root directory."""
    dataset = datasets.ImageFolder(root=root, transform=transform)
    data = []
    targets = []
    for path, target in dataset.samples:
        try:
            with open(path, 'rb') as f:
                img = Image.open(f)
                if transform is not None:
                    img = transform(img)
                data.append(img)
                targets.append(target)
        except OSError:
            print(f"Skipping {path} as it is corrupted or incomplete.")
    return torch.stack(data), torch.tensor(targets)

def create_data_loaders(data_dir, batch_size, device):
    data_transform = transforms.Compose([
        transforms.Resize((100, 100)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_data, train_targets = load_dataset(os.path.join(data_dir, 'train'), transform=data_transform)
    val_data, val_targets = load_dataset(os.path.join(data_dir, 'val'), transform=data_transform)
    test_data, test_targets = load_dataset(os.path.join(data_dir, 'test'), transform=data_transform)

    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(train_data.to(device), train_targets.to(device)),
        batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(val_data.to(device), val_targets.to(device)),
        batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(test_data.to(device), test_targets.to(device)),
        batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader

