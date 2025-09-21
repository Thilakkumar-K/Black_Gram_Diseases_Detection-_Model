import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import json
from tqdm import tqdm

# Try to import albumentations
try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2

    ALBUMENTATIONS_AVAILABLE = True
except ImportError:
    ALBUMENTATIONS_AVAILABLE = False


class CustomCNN(nn.Module):
    """Custom CNN architecture (same as in train.py)"""

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


def get_test_transform():
    """Get test data transform (same as validation transform)"""

    if ALBUMENTATIONS_AVAILABLE:
        transform = AlbumentationsTransform(A.Compose([
            A.Resize(224, 224),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ]))
    else:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    return transform


def load_model(model_path, device):
    """Load trained model from checkpoint"""

    print(f"Loading model from {model_path}")
    checkpoint = torch.load(model_path, map_location=device)

    # Get model parameters
    model_type = checkpoint.get('model_type', 'resnet50')
    num_classes = checkpoint.get('num_classes', 5)
    class_names = checkpoint.get('class_names', [f'class_{i}' for i in range(num_classes)])

    # Create model
    if model_type == 'custom':
        model = CustomCNN(num_classes=num_classes)
    elif model_type == 'resnet50':
        model = models.resnet50(pretrained=False)
        num_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Load state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    print(f"Model loaded successfully!")
    print(f"Model type: {model_type}")
    print(f"Number of classes: {num_classes}")
    print(f"Class names: {class_names}")
    best_val_acc = checkpoint.get('best_val_acc', 'N/A')
    if isinstance(best_val_acc, (int, float)):
        print(f"Best validation accuracy: {best_val_acc:.4f}")
    else:
        print(f"Best validation accuracy: {best_val_acc}")

    return model, class_names


def evaluate_model(model, test_loader, device, class_names):
    """Evaluate model on test set"""

    model.eval()
    all_predictions = []
    all_targets = []
    all_probs = []

    print("\nEvaluating model on test set...")

    with torch.no_grad():
        for inputs, targets in tqdm(test_loader, desc="Testing"):
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            probabilities = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)

            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            all_probs.extend(probabilities.cpu().numpy())

    # Calculate accuracy
    accuracy = accuracy_score(all_targets, all_predictions)

    # Generate classification report
    report = classification_report(
        all_targets,
        all_predictions,
        target_names=class_names,
        digits=4
    )

    # Generate confusion matrix
    cm = confusion_matrix(all_targets, all_predictions)

    return accuracy, report, cm, all_predictions, all_targets, all_probs


def plot_confusion_matrix(cm, class_names, save_path=None):
    """Plot and save confusion matrix"""

    plt.figure(figsize=(10, 8))

    # Calculate percentages
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

    # Create annotation labels
    annot = np.empty_like(cm).astype(str)
    nrows, ncols = cm.shape
    for i in range(nrows):
        for j in range(ncols):
            c = cm[i, j]
            p = cm_percent[i, j]
            if c == 0:
                annot[i, j] = ''
            else:
                annot[i, j] = f'{c}\n({p:.1f}%)'

    # Plot heatmap
    sns.heatmap(cm,
                annot=annot,
                fmt='',
                cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names,
                cbar_kws={'label': 'Count'})

    plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")

    plt.show()
    return plt.gcf()


def predict_single_image(model, image_path, class_names, device, transform):
    """Predict class for a single image"""

    try:
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')

        # Apply transform
        if ALBUMENTATIONS_AVAILABLE and hasattr(transform, 'transform'):
            # For AlbumentationsTransform
            image_np = np.array(image)
            transformed = transform.transform(image=image_np)
            input_tensor = transformed['image']
        else:
            # For torchvision transforms
            input_tensor = transform(image)

        # Add batch dimension
        input_tensor = input_tensor.unsqueeze(0).to(device)

        # Make prediction
        model.eval()
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted = probabilities.max(1)

            predicted_class = class_names[predicted.item()]
            confidence_score = confidence.item()

        # Get top-k predictions
        top_probs, top_classes = probabilities.topk(min(3, len(class_names)), dim=1)

        print(f"\n{'=' * 50}")
        print(f"PREDICTION FOR: {os.path.basename(image_path)}")
        print(f"{'=' * 50}")
        print(f"Predicted Class: {predicted_class}")
        print(f"Confidence: {confidence_score:.4f} ({confidence_score * 100:.2f}%)")
        print(f"\nTop predictions:")

        for i in range(top_probs.size(1)):
            class_name = class_names[top_classes[0][i].item()]
            prob = top_probs[0][i].item()
            print(f"  {i + 1}. {class_name}: {prob:.4f} ({prob * 100:.2f}%)")

        return predicted_class, confidence_score

    except Exception as e:
        print(f"Error predicting image {image_path}: {str(e)}")
        return None, None


# Configuration class for easy PyCharm usage
class TestConfig:
    def __init__(self):
        # Dataset settings
        self.data_path = 'dataset'
        self.model_path = 'outputs/model_best.pth'

        # Testing settings
        self.batch_size = 32
        self.output_dir = 'outputs'
        self.single_image = None

        # System settings
        self.num_workers = 4  # Reduce to 0 if having issues on Windows
        self.force_cpu = False  # Set to True to force CPU usage


def main(config=None):
    """
    Main testing function
    Args:
        config: TestConfig object or None (uses argparse if None)
    """

    # Handle both PyCharm direct execution and command line
    if config is None:
        parser = argparse.ArgumentParser(description='Test Plant Leaf Classifier')
        parser.add_argument('--model_path', type=str, default='outputs/model_best.pth',
                            help='Path to trained model')
        parser.add_argument('--data_path', type=str, default='dataset',
                            help='Path to dataset directory')
        parser.add_argument('--batch_size', type=int, default=32,
                            help='Batch size for testing')
        parser.add_argument('--output_dir', type=str, default='outputs',
                            help='Directory to save outputs')
        parser.add_argument('--single_image', type=str, default=None,
                            help='Path to single image for prediction')
        parser.add_argument('--force_cpu', action='store_true',
                            help='Force CPU usage even if GPU available')

        args = parser.parse_args()

        # Convert argparse to config object
        config = TestConfig()
        config.model_path = args.model_path
        config.data_path = args.data_path
        config.batch_size = args.batch_size
        config.output_dir = args.output_dir
        config.single_image = args.single_image
        config.force_cpu = getattr(args, 'force_cpu', False)

    # Device configuration with better error handling
    if config.force_cpu:
        device = torch.device('cpu')
        print(f'Forcing CPU usage as requested')
    else:
        if torch.cuda.is_available():
            device = torch.device('cuda')
            print(f'Using GPU: {torch.cuda.get_device_name()}')
        else:
            device = torch.device('cpu')
            print(f'CUDA not available, using CPU')

    # Adjust settings based on device
    if device.type == 'cpu':
        config.batch_size = min(config.batch_size, 16)
        config.num_workers = 0
        print(f"Adjusted batch size to {config.batch_size} for CPU usage")

    print(f'Final device: {device}')

    # Create output directory
    os.makedirs(config.output_dir, exist_ok=True)

    # Check if model exists
    if not os.path.exists(config.model_path):
        raise FileNotFoundError(f"Model file not found: {config.model_path}")

    # Load model and class names
    model, class_names = load_model(config.model_path, device)

    # Get transform
    transform = get_test_transform()

    # If single image prediction is requested
    if config.single_image:
        if os.path.exists(config.single_image):
            predict_single_image(model, config.single_image, class_names, device, transform)
        else:
            print(f"Error: Image file {config.single_image} not found!")
        return

    # Load test dataset
    test_path = os.path.join(config.data_path, 'test')

    if not os.path.exists(test_path):
        print(f"Error: Test directory {test_path} not found!")
        print("Please ensure your dataset has the following structure:")
        print("dataset/")
        print("  train/")
        print("    class1/")
        print("    class2/")
        print("    ...")
        print("  val/")
        print("    class1/")
        print("    class2/")
        print("    ...")
        print("  test/")
        print("    class1/")
        print("    class2/")
        print("    ...")
        return

    try:
        test_dataset = datasets.ImageFolder(
            root=test_path,
            transform=transform
        )
    except Exception as e:
        print(f"Error loading test dataset: {e}")
        return

    # Create test loader with error handling
    try:
        test_loader = DataLoader(
            test_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=(device.type == 'cuda'),
            persistent_workers=(config.num_workers > 0)
        )
    except Exception as e:
        print(f"Error creating data loader: {e}")
        print("Trying with num_workers=0...")
        test_loader = DataLoader(
            test_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=(device.type == 'cuda')
        )

    print(f"\nTest dataset loaded:")
    print(f"Number of test samples: {len(test_dataset)}")
    print(f"Test classes found: {test_dataset.classes}")

    # Verify class names match
    if test_dataset.classes != class_names:
        print(f"\nWarning: Test dataset classes {test_dataset.classes} don't match")
        print(f"trained model classes {class_names}")
        print("Using model's class names for evaluation...")

    # Evaluate model
    accuracy, report, cm, predictions, targets, probabilities = evaluate_model(
        model, test_loader, device, class_names
    )

    # Print results
    print(f"\n{'=' * 60}")
    print('TEST RESULTS')
    print(f"{'=' * 60}")
    print(f"Test Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print(f"\nClassification Report:")
    print(report)

    # Save results
    results = {
        'test_accuracy': float(accuracy),
        'classification_report': report,
        'confusion_matrix': cm.tolist(),
        'class_names': class_names,
        'predictions': [int(p) for p in predictions],
        'targets': [int(t) for t in targets],
        'config': config.__dict__
    }

    with open(os.path.join(config.output_dir, 'test_results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    # Plot and save confusion matrix
    cm_path = os.path.join(config.output_dir, 'confusion_matrix.png')
    plot_confusion_matrix(cm, class_names, save_path=cm_path)

    # Calculate per-class accuracy
    print(f"\n{'=' * 40}")
    print("PER-CLASS ACCURACY")
    print(f"{'=' * 40}")

    class_accuracies = []
    for i, class_name in enumerate(class_names):
        class_correct = cm[i, i]
        class_total = cm[i, :].sum()
        class_acc = class_correct / class_total if class_total > 0 else 0
        class_accuracies.append(class_acc)
        print(f"{class_name}: {class_acc:.4f} ({class_acc * 100:.2f}%) [{class_correct}/{class_total}]")

    # Additional statistics
    print(f"\n{'=' * 40}")
    print("ADDITIONAL STATISTICS")
    print(f"{'=' * 40}")
    print(f"Number of test samples: {len(targets)}")
    print(f"Number of classes: {len(class_names)}")
    print(f"Overall accuracy: {accuracy:.4f}")
    print(f"Average per-class accuracy: {np.mean(class_accuracies):.4f}")
    print(f"Standard deviation of class accuracies: {np.std(class_accuracies):.4f}")

    # Show class distribution in test set
    print(f"\nTest set class distribution:")
    unique, counts = np.unique(targets, return_counts=True)
    for class_idx, count in zip(unique, counts):
        percentage = (count / len(targets)) * 100
        print(f"  {class_names[class_idx]}: {count} samples ({percentage:.1f}%)")

    print(f"\nResults saved to {config.output_dir}")
    print(f"  - test_results.json: Detailed results")
    print(f"  - confusion_matrix.png: Confusion matrix plot")

    return accuracy, report, cm


def interactive_prediction():
    """Interactive mode for single image predictions"""

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Default model path
    model_path = 'outputs/model_best.pth'

    if not os.path.exists(model_path):
        model_path = input("Enter model path: ").strip()
        if not os.path.exists(model_path):
            print(f"Model file not found: {model_path}")
            return

    # Load model
    model, class_names = load_model(model_path, device)
    transform = get_test_transform()

    print(f"\n{'=' * 60}")
    print("INTERACTIVE PLANT LEAF CLASSIFIER")
    print(f"{'=' * 60}")
    print("Enter image paths to get predictions (type 'quit' to exit)")

    while True:
        image_path = input("\nEnter image path: ").strip()

        if image_path.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break

        if not image_path:
            continue

        if os.path.exists(image_path):
            predict_single_image(model, image_path, class_names, device, transform)
        else:
            print(f"Error: File '{image_path}' not found!")


# PyCharm-friendly execution functions
def test_with_defaults():
    """Quick testing with default settings - perfect for PyCharm"""
    config = TestConfig()
    return main(config)


def test_on_cpu():
    """Test model on CPU"""
    config = TestConfig()
    config.force_cpu = True
    config.batch_size = 16
    return main(config)


def predict_image(image_path):
    """Predict a single image - convenient for PyCharm"""
    config = TestConfig()
    config.single_image = image_path
    return main(config)


def test_custom_model():
    """Test with custom model path"""
    config = TestConfig()
    config.model_path = 'outputs/checkpoint_epoch_10.pth'  # Example checkpoint
    return main(config)


if __name__ == '__main__':
    # Check if running from PyCharm or command line
    import sys

    if len(sys.argv) == 1:
        # Running from PyCharm without arguments
        print("=" * 60)
        print("PYCHARM MODE - Choose your testing option:")
        print("=" * 60)
        print("1. test_with_defaults() - Standard testing")
        print("2. test_on_cpu() - CPU-only testing")
        print("3. predict_image('path/to/image.jpg') - Single image prediction")
        print("4. test_custom_model() - Test with custom model checkpoint")
        print("5. interactive_prediction() - Interactive mode")
        print("=" * 60)
        print("Execute one of these functions in PyCharm console or uncomment below:")
        print()

        # Uncomment the line below for the testing mode you want:
        # test_with_defaults()
        # test_on_cpu()
        predict_image('D:/crop_disease_identification/Black_Gram_Modle/dataset/test/Healthy/20241104_110916.jpg')
        # test_custom_model()
        # interactive_prediction()

        print("Or run from command line with arguments for full control.")

    else:
        # Command line execution
        main()