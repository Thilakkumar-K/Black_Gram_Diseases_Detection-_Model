import os
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms, models
import torchvision.transforms.functional as TF
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from tqdm import tqdm
import json

# Try to import albumentations, fallback to torchvision transforms
try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2

    ALBUMENTATIONS_AVAILABLE = False  # Temporarily disable
    print("Albumentations temporarily disabled, using torchvision transforms")
except ImportError:
    ALBUMENTATIONS_AVAILABLE = False
    print("Albumentations not available, using torchvision transforms")


def set_seed(seed=42):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class CustomCNN(nn.Module):
    """Custom CNN architecture for plant leaf classification"""

    def __init__(self, num_classes=5):
        super(CustomCNN, self).__init__()

        # Feature extraction layers
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),

            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),

            # Block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),

            # Block 4
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((7, 7)),
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class AlbumentationsTransform:
    """Wrapper for Albumentations transforms"""

    def __init__(self, transform):
        self.transform = transform

    def __call__(self, image):
        # Convert PIL Image to numpy array
        if hasattr(image, 'convert'):
            image = np.array(image.convert('RGB'))

        # Apply albumentations transform
        augmented = self.transform(image=image)
        return augmented['image']


def get_transforms(use_augmentation=True):
    """Get data transforms for training and validation"""

    if ALBUMENTATIONS_AVAILABLE and use_augmentation:
        # Training transforms with Albumentations
        train_transform = AlbumentationsTransform(A.Compose([
            A.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.0)),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.Rotate(limit=25, p=0.5),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
            A.CoarseDropout(max_holes=8, max_height=32, max_width=32, p=0.5),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ]))

        # Validation transforms
        val_transform = AlbumentationsTransform(A.Compose([
            A.Resize(size=(224, 224)),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ]))

    else:
        # Fallback to torchvision transforms
        if use_augmentation:
            train_transform = transforms.Compose([
                transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.3),
                transforms.RandomRotation(25),
                transforms.ToTensor(),  # Convert to tensor first
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.RandomErasing(p=0.5, scale=(0.02, 0.15)),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            train_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

        val_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    return train_transform, val_transform


def create_model(model_type='resnet50', num_classes=5):
    """Create model - either custom CNN or pretrained ResNet50"""

    if model_type == 'custom':
        print("Using Custom CNN architecture")
        model = CustomCNN(num_classes=num_classes)

    elif model_type == 'resnet50':
        print("Using ResNet50 with transfer learning")
        model = models.resnet50(pretrained=True)

        # Freeze early layers (optional - comment out to fine-tune all layers)
        for param in model.parameters():
            param.requires_grad = False

        # Replace final layer
        num_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

        # Unfreeze last few layers for fine-tuning
        for param in model.layer4.parameters():
            param.requires_grad = True
        for param in model.fc.parameters():
            param.requires_grad = True

    else:
        raise ValueError(f"Unknown model type: {model_type}")

    return model


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train model for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(dataloader, desc="Training")
    for inputs, targets in pbar:
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        pbar.set_postfix({
            'Loss': f'{running_loss / len(pbar):.4f}',
            'Acc': f'{100. * correct / total:.2f}%'
        })

    epoch_loss = running_loss / len(dataloader)
    epoch_acc = correct / total

    return epoch_loss, epoch_acc


def validate(model, dataloader, criterion, device):
    """Validate model"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Validation")
        for inputs, targets in pbar:
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            pbar.set_postfix({
                'Loss': f'{running_loss / len(pbar):.4f}',
                'Acc': f'{100. * correct / total:.2f}%'
            })

    epoch_loss = running_loss / len(dataloader)
    epoch_acc = correct / total

    return epoch_loss, epoch_acc


# Configuration class for easy PyCharm usage
class TrainConfig:
    def __init__(self):
        # Dataset settings
        self.data_path = 'dataset'
        self.model_type = 'resnet50'  # 'custom' or 'resnet50'

        # Training hyperparameters
        self.batch_size = 32
        self.lr = 1e-4
        self.epochs = 30
        self.patience = 8

        # System settings
        self.output_dir = 'outputs'
        self.seed = 42
        self.num_workers = 4  # Reduce to 0 if having issues on Windows
        self.force_cpu = False  # Set to True to force CPU usage

        # Advanced settings
        self.weight_decay = 1e-4
        self.save_every_n_epochs = 5  # Save checkpoint every N epochs


def main(config=None):
    """
    Main training function
    Args:
        config: TrainConfig object or None (uses argparse if None)
    """

    # Handle both PyCharm direct execution and command line
    if config is None:
        parser = argparse.ArgumentParser(description='Train Plant Leaf Classifier')
        parser.add_argument('--data_path', type=str, default='dataset',
                            help='Path to dataset directory')
        parser.add_argument('--model_type', type=str, default='resnet50',
                            choices=['custom', 'resnet50'],
                            help='Model architecture to use')
        parser.add_argument('--batch_size', type=int, default=32,
                            help='Batch size for training')
        parser.add_argument('--lr', type=float, default=1e-4,
                            help='Learning rate')
        parser.add_argument('--epochs', type=int, default=30,
                            help='Number of epochs to train')
        parser.add_argument('--patience', type=int, default=8,
                            help='Early stopping patience')
        parser.add_argument('--output_dir', type=str, default='outputs',
                            help='Directory to save outputs')
        parser.add_argument('--seed', type=int, default=42,
                            help='Random seed')
        parser.add_argument('--force_cpu', action='store_true',
                            help='Force CPU usage even if GPU available')

        args = parser.parse_args()

        # Convert argparse to config object
        config = TrainConfig()
        config.data_path = args.data_path
        config.model_type = args.model_type
        config.batch_size = args.batch_size
        config.lr = args.lr
        config.epochs = args.epochs
        config.patience = args.patience
        config.output_dir = args.output_dir
        config.seed = args.seed
        config.force_cpu = getattr(args, 'force_cpu', False)

    # Set random seed
    set_seed(config.seed)

    # Create output directory
    os.makedirs(config.output_dir, exist_ok=True)

    # Device configuration with better error handling
    if config.force_cpu:
        device = torch.device('cpu')
        print(f'Forcing CPU usage as requested')
    else:
        if torch.cuda.is_available():
            device = torch.device('cuda')
            print(f'Using GPU: {torch.cuda.get_device_name()}')
            print(f'GPU Memory: {torch.cuda.get_device_properties(device).total_memory / 1024 ** 3:.1f} GB')
        else:
            device = torch.device('cpu')
            print(f'CUDA not available, using CPU')

    # Adjust settings based on device
    if device.type == 'cpu':
        # Reduce batch size and workers for CPU
        config.batch_size = min(config.batch_size, 16)
        config.num_workers = 0  # Avoid multiprocessing issues on CPU
        print(f"Adjusted batch size to {config.batch_size} for CPU usage")

    print(f'Final device: {device}')
    print(f'Batch size: {config.batch_size}')
    print(f'Number of workers: {config.num_workers}')

    # Data transforms
    train_transform, val_transform = get_transforms(use_augmentation=True)

    # Check if dataset exists
    train_path = os.path.join(config.data_path, 'train')
    val_path = os.path.join(config.data_path, 'val')

    if not os.path.exists(train_path):
        raise ValueError(f"Training directory not found: {train_path}")
    if not os.path.exists(val_path):
        raise ValueError(f"Validation directory not found: {val_path}")

    # Datasets
    try:
        train_dataset = datasets.ImageFolder(
            root=train_path,
            transform=train_transform
        )

        val_dataset = datasets.ImageFolder(
            root=val_path,
            transform=val_transform
        )
    except Exception as e:
        print(f"Error loading datasets: {e}")
        print("Please ensure your dataset has the correct structure:")
        print("dataset/")
        print("  train/")
        print("    class1/")
        print("    class2/")
        print("  val/")
        print("    class1/")
        print("    class2/")
        raise

    # Data loaders with error handling
    try:
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            pin_memory=(device.type == 'cuda'),
            persistent_workers=(config.num_workers > 0)
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=(device.type == 'cuda'),
            persistent_workers=(config.num_workers > 0)
        )
    except Exception as e:
        print(f"Error creating data loaders: {e}")
        print("Trying with num_workers=0...")
        config.num_workers = 0

        train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=(device.type == 'cuda')
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=(device.type == 'cuda')
        )

    # Get number of classes and class names
    num_classes = len(train_dataset.classes)
    class_names = train_dataset.classes
    print(f'Number of classes: {num_classes}')
    print(f'Classes: {class_names}')

    # Save class names
    with open(os.path.join(config.output_dir, 'class_names.json'), 'w') as f:
        json.dump(class_names, f)

    # Model
    model = create_model(config.model_type, num_classes)
    model = model.to(device)

    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)

    # TensorBoard writer
    writer = SummaryWriter(os.path.join(config.output_dir, 'tensorboard'))

    # Training loop
    best_val_acc = 0.0
    patience_counter = 0
    train_losses, train_accs = [], []
    val_losses, val_accs = [], []

    print(f'\nStarting training for {config.epochs} epochs...')
    print(f'Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}')

    for epoch in range(config.epochs):
        print(f'\nEpoch {epoch + 1}/{config.epochs}')
        print('-' * 50)

        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)

        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        # Scheduler step
        scheduler.step()

        # Save metrics
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        # TensorBoard logging
        writer.add_scalar('Loss/Train', train_loss, epoch)
        writer.add_scalar('Loss/Val', val_loss, epoch)
        writer.add_scalar('Accuracy/Train', train_acc, epoch)
        writer.add_scalar('Accuracy/Val', val_acc, epoch)
        writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], epoch)

        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')

        # Early stopping and model saving
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0

            # Save best model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_acc': best_val_acc,
                'model_type': config.model_type,
                'num_classes': num_classes,
                'class_names': class_names,
                'config': config.__dict__  # Save config for reproducibility
            }, os.path.join(config.output_dir, 'model_best.pth'))

            print(f'New best validation accuracy: {best_val_acc:.4f}')
        else:
            patience_counter += 1

        # Save checkpoint every N epochs
        if (epoch + 1) % config.save_every_n_epochs == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'model_type': config.model_type,
                'num_classes': num_classes,
                'class_names': class_names
            }, os.path.join(config.output_dir, f'checkpoint_epoch_{epoch + 1}.pth'))

        if patience_counter >= config.patience:
            print(f'\nEarly stopping triggered after {config.patience} epochs without improvement')
            break

    # Close TensorBoard writer
    writer.close()

    # Save training history
    history = {
        'train_losses': train_losses,
        'train_accs': train_accs,
        'val_losses': val_losses,
        'val_accs': val_accs,
        'config': config.__dict__
    }

    with open(os.path.join(config.output_dir, 'training_history.json'), 'w') as f:
        json.dump(history, f)

    # Plot training history
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Acc')
    plt.plot(val_accs, label='Val Acc')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(config.output_dir, 'training_history.png'), dpi=300, bbox_inches='tight')
    plt.show()  # Show plot in PyCharm

    print(f'\n{"=" * 60}')
    print('TRAINING COMPLETED')
    print(f'{"=" * 60}')
    print(f'Best validation accuracy: {best_val_acc:.4f}')
    print(f'Final training accuracy: {train_accs[-1]:.4f}')
    print(f'Model saved to: {os.path.join(config.output_dir, "model_best.pth")}')
    print(f'Training history saved to: {config.output_dir}')

    return model, best_val_acc, history


# PyCharm-friendly execution functions
def train_with_defaults():
    """Quick training with default settings - perfect for PyCharm"""
    config = TrainConfig()
    return main(config)


def train_resnet_cpu():
    """Train ResNet50 on CPU"""
    config = TrainConfig()
    config.force_cpu = True
    config.batch_size = 16  # Smaller batch for CPU
    config.epochs = 20
    return main(config)


def train_custom_cnn():
    """Train custom CNN"""
    config = TrainConfig()
    config.model_type = 'custom'
    config.lr = 5e-4  # Higher learning rate for custom model
    return main(config)


def quick_test_run():
    """Quick test run with few epochs"""
    config = TrainConfig()
    config.epochs = 3
    config.batch_size = 8
    config.patience = 2
    return main(config)


if __name__ == '__main__':
    # Check if running from PyCharm or command line
    import sys

    if len(sys.argv) == 1:
        # Running from PyCharm without arguments
        print("=" * 60)
        print("PYCHARM MODE - Choose your training option:")
        print("=" * 60)
        print("1. train_with_defaults() - Standard ResNet50 training")
        print("2. train_resnet_cpu() - CPU-only ResNet50 training")
        print("3. train_custom_cnn() - Custom CNN architecture")
        print("4. quick_test_run() - Quick test with few epochs")
        print("=" * 60)
        print("Execute one of these functions in PyCharm console or uncomment below:")
        print()

        # Uncomment the line below for the training mode you want:
        # train_with_defaults()
        # train_resnet_cpu()
        # train_custom_cnn()
        quick_test_run()

        print("Or run from command line with arguments for full control.")

    else:
        # Command line execution
        main()