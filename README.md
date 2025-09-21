# 🌱 Plant Disease Classification System

*Empowering farmers and gardeners with AI-powered plant health diagnosis*

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)

> **Revolutionizing agriculture through computer vision** - Instantly identify plant diseases from leaf images with state-of-the-art deep learning models.

---

## 🚀 What Makes This Special?

This isn't just another image classifier. It's a comprehensive plant health diagnostic tool designed with real-world agricultural challenges in mind:

- **🎯 Production-Ready**: Built for deployment in farming environments
- **⚡ Lightning Fast**: Get results in milliseconds, not minutes
- **🧠 Dual Architecture**: Choose between custom CNN or transfer learning with ResNet50
- **📊 Rich Analytics**: Detailed confidence scores and per-class performance metrics
- **🔧 Developer Friendly**: PyCharm-optimized with multiple execution modes
- **📈 Visual Insights**: Comprehensive confusion matrices and training visualizations

## 🌟 Key Features

### 🎨 Flexible Model Architecture
```python
# Quick start with different models
train_with_defaults()     # ResNet50 transfer learning
train_custom_cnn()        # Custom CNN architecture
train_resnet_cpu()        # CPU-optimized training
```

### 📸 Smart Image Processing
- **Advanced Augmentation**: Rotation, flips, color jittering, and cutout
- **Robust Preprocessing**: Handles various image formats and sizes
- **Normalization**: ImageNet statistics for optimal transfer learning

### 🎯 Production-Grade Testing
- **Interactive Prediction Mode**: Real-time image classification
- **Batch Evaluation**: Comprehensive test set analysis
- **Performance Metrics**: Accuracy, precision, recall, and F1-scores
- **Visual Reports**: Confusion matrices and classification reports

---

## 🛠 Installation & Setup

### Prerequisites
```bash
# Ensure you have Python 3.8+ installed
python --version
```

### Quick Install
```bash
# Clone the repository
git clone https://github.com/yourusername/plant-disease-classification.git
cd plant-disease-classification

# Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install scikit-learn matplotlib seaborn tqdm pillow tensorboard

# Optional: Install albumentations for advanced augmentations
pip install albumentations
```

### 📁 Dataset Structure
Organize your dataset following this structure:
```
dataset/
├── train/
│   ├── healthy/
│   ├── disease_1/
│   ├── disease_2/
│   └── ...
├── val/
│   ├── healthy/
│   ├── disease_1/
│   ├── disease_2/
│   └── ...
└── test/
    ├── healthy/
    ├── disease_1/
    ├── disease_2/
    └── ...
```

---

## 🎮 Usage Guide

### 🏃‍♂️ Quick Start

**Training a Model:**
```python
# For PyCharm users - just run one of these:
train_with_defaults()    # Standard training
train_custom_cnn()       # Custom architecture
quick_test_run()         # Fast testing (3 epochs)

# Command line usage:
python train.py --model_type resnet50 --epochs 30 --batch_size 32
```

**Testing Your Model:**
```python
# Single image prediction
predict_image('path/to/your/leaf_image.jpg')

# Full test set evaluation
test_with_defaults()

# Interactive mode
interactive_prediction()

# Command line:
python test.py --model_path outputs/model_best.pth --single_image leaf.jpg
```

### 🎛 Advanced Configuration

```python
# Customize training parameters
config = TrainConfig()
config.model_type = 'resnet50'     # or 'custom'
config.batch_size = 64
config.lr = 1e-4
config.epochs = 50
config.patience = 10

# Run with custom config
main(config)
```

---

## 🧠 Model Architectures

### 📐 System Architecture Overview

```mermaid
graph TB
    subgraph "Input Pipeline"
        A[Plant Leaf Image<br/>224x224x3] --> B[Data Augmentation<br/>Rotation, Flip, ColorJitter]
        B --> C[Normalization<br/>ImageNet Stats]
    end
    
    subgraph "Model Selection"
        C --> D{Model Type?}
        D -->|Custom| E[Custom CNN<br/>4 Conv Blocks]
        D -->|Transfer Learning| F[ResNet50<br/>Pre-trained]
    end
    
    subgraph "Custom CNN Architecture"
        E --> G[Block 1: 3→64<br/>Conv+BN+ReLU+Pool]
        G --> H[Block 2: 64→128<br/>Conv+BN+ReLU+Pool]
        H --> I[Block 3: 128→256<br/>Conv+BN+ReLU+Pool]
        I --> J[Block 4: 256→512<br/>Conv+BN+ReLU+Pool]
        J --> K[Adaptive AvgPool<br/>7x7]
        K --> L[FC Classifier<br/>25088→1024→512→Classes]
    end
    
    subgraph "ResNet50 Transfer Learning"
        F --> M[Frozen Backbone<br/>Feature Extraction]
        M --> N[Fine-tuned Layer4<br/>Progressive Unfreezing]
        N --> O[Custom Classifier<br/>2048→512→Classes]
    end
    
    subgraph "Output & Training"
        L --> P[Softmax Probabilities]
        O --> P
        P --> Q[Disease Prediction<br/>+ Confidence Score]
        P --> R[Cross Entropy Loss]
        R --> S[AdamW Optimizer<br/>Cosine Annealing]
        S --> T[Early Stopping<br/>Patience=8]
    end
    
    subgraph "Evaluation"
        Q --> U[Confusion Matrix]
        Q --> V[Classification Report]
        Q --> W[Per-class Accuracy]
    end

    style A fill:#e1f5fe
    style Q fill:#e8f5e8
    style E fill:#fff3e0
    style F fill:#fce4ec
    style P fill:#f3e5f5
```

### 🔬 Custom CNN Architecture Details

```mermaid
graph LR
    subgraph "Input Layer"
        A[224×224×3<br/>RGB Image]
    end
    
    subgraph "Block 1"
        A --> B[Conv2d 3→64<br/>3×3, padding=1]
        B --> C[BatchNorm2d]
        C --> D[ReLU]
        D --> E[Conv2d 64→64<br/>3×3, padding=1]
        E --> F[BatchNorm2d]
        F --> G[ReLU]
        G --> H[MaxPool2d<br/>2×2]
        H --> I[Dropout2d<br/>p=0.25]
    end
    
    subgraph "Block 2"
        I --> J[Conv2d 64→128<br/>3×3, padding=1]
        J --> K[BatchNorm2d]
        K --> L[ReLU]
        L --> M[Conv2d 128→128<br/>3×3, padding=1]
        M --> N[BatchNorm2d]
        N --> O[ReLU]
        O --> P[MaxPool2d<br/>2×2]
        P --> Q[Dropout2d<br/>p=0.25]
    end
    
    subgraph "Block 3"
        Q --> R[Conv2d 128→256<br/>3×3, padding=1]
        R --> S[BatchNorm2d]
        S --> T[ReLU]
        T --> U[Conv2d 256→256<br/>3×3, padding=1]
        U --> V[BatchNorm2d]
        V --> W[ReLU]
        W --> X[MaxPool2d<br/>2×2]
        X --> Y[Dropout2d<br/>p=0.25]
    end
    
    subgraph "Block 4"
        Y --> Z[Conv2d 256→512<br/>3×3, padding=1]
        Z --> AA[BatchNorm2d]
        AA --> BB[ReLU]
        BB --> CC[Conv2d 512→512<br/>3×3, padding=1]
        CC --> DD[BatchNorm2d]
        DD --> EE[ReLU]
        EE --> FF[MaxPool2d<br/>2×2]
        FF --> GG[Dropout2d<br/>p=0.25]
    end
    
    subgraph "Classifier"
        GG --> HH[AdaptiveAvgPool2d<br/>7×7]
        HH --> II[Flatten<br/>25088 features]
        II --> JJ[Linear 25088→1024]
        JJ --> KK[BatchNorm1d]
        KK --> LL[ReLU + Dropout 0.5]
        LL --> MM[Linear 1024→512]
        MM --> NN[BatchNorm1d]
        NN --> OO[ReLU + Dropout 0.5]
        OO --> PP[Linear 512→Classes]
    end

    style A fill:#e3f2fd
    style PP fill:#e8f5e8
```

### 🎯 ResNet50 Transfer Learning Strategy

```mermaid
graph TB
    subgraph "ImageNet Pre-trained ResNet50"
        A[Input 224×224×3] --> B[Conv1 + BN + ReLU]
        B --> C[MaxPool]
        C --> D[Layer1: 4×Bottleneck<br/>64→256 channels]
        D --> E[Layer2: 4×Bottleneck<br/>256→512 channels]
        E --> F[Layer3: 6×Bottleneck<br/>512→1024 channels]
        F --> G[Layer4: 3×Bottleneck<br/>1024→2048 channels]
    end
    
    subgraph "Fine-tuning Strategy"
        H[Frozen Layers<br/>Conv1 → Layer3] 
        I[Trainable Layers<br/>Layer4 Only]
        J[Custom Classifier<br/>Fully Trainable]
    end
    
    subgraph "Custom Classification Head"
        G --> K[Global Average Pool<br/>2048 features]
        K --> L[Dropout p=0.5]
        L --> M[Linear 2048→512]
        M --> N[ReLU Activation]
        N --> O[Dropout p=0.3]
        O --> P[Linear 512→num_classes]
        P --> Q[Softmax Output]
    end
    
    B -.-> H
    C -.-> H
    D -.-> H
    E -.-> H
    F -.-> H
    G --> I
    K --> J

    style H fill:#ffebee
    style I fill:#fff8e1
    style J fill:#e8f5e8
    style Q fill:#e8f5e8
```

### Architecture Comparison

| Feature | Custom CNN | ResNet50 Transfer |
|---------|------------|-------------------|
| **Parameters** | ~25M | ~23M (trainable: ~2M) |
| **Training Time** | Longer | Faster convergence |
| **Memory Usage** | Moderate | Higher |
| **Performance** | Good for specific datasets | Better generalization |
| **Interpretability** | Higher | Lower |

---

## 📊 Performance Metrics

The system provides comprehensive evaluation metrics:

- **Overall Accuracy**: Model performance across all classes
- **Per-Class Metrics**: Precision, recall, F1-score for each disease
- **Confusion Matrix**: Visual representation of classification results
- **Confidence Scores**: Prediction certainty for each classification
- **Training Curves**: Loss and accuracy progression over epochs

### Sample Results
```
Test Accuracy: 0.9450 (94.50%)

Classification Report:
                precision    recall  f1-score   support
       Healthy      0.9567    0.9456    0.9511       183
   Black_Spot      0.9234    0.9567    0.9398       162
   Leaf_Rust      0.9456    0.9123    0.9286       171
   Powdery_M      0.9678    0.9234    0.9451       154
```

---

## 🎨 Visualization Features

### 📈 Training Monitoring
- **Real-time Progress Bars** with loss and accuracy updates
- **TensorBoard Integration** for detailed training analysis
- **Automatic Plot Generation** for training curves
- **Model Architecture Visualization**

### 📊 Test Results
- **Interactive Confusion Matrices** with percentage annotations
- **Per-Class Performance Charts**
- **Prediction Confidence Histograms**
- **Sample Predictions Gallery**

---

## 🔧 Development Features

### 👨‍💻 PyCharm Optimization
- **Pre-configured Execution Functions** for common tasks
- **Intelligent Error Handling** with helpful suggestions
- **Automatic Environment Detection** (CPU/GPU)
- **Debug-Friendly Code Structure**

### 🛠 Robust Error Handling
```python
# Automatic fallbacks for common issues
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
    config.batch_size = min(config.batch_size, 16)  # Auto-adjust
```

### 📁 Smart File Management
- **Automatic Directory Creation**
- **Checkpoint Saving** every N epochs
- **Model Versioning** with metadata
- **Training History Persistence**

---

## 🌐 Real-World Applications

### 🚜 Agricultural Use Cases
- **Field Diagnostics**: Mobile deployment for on-site disease detection
- **Crop Monitoring**: Integration with IoT systems and drones
- **Early Warning Systems**: Preventive disease management
- **Quality Control**: Post-harvest assessment

### 📱 Integration Possibilities
- **Mobile Apps**: React Native or Flutter frontends
- **Web Services**: REST API deployment
- **Edge Devices**: Raspberry Pi or NVIDIA Jetson
- **Cloud Platforms**: AWS, GCP, or Azure deployment

---

## 🤝 Contributing

We welcome contributions from the agricultural AI community!

### 🌟 How to Contribute
1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit** your changes (`git commit -m 'Add amazing feature'`)
4. **Push** to the branch (`git push origin feature/amazing-feature`)
5. **Open** a Pull Request

### 📝 Development Guidelines
- Follow PEP 8 style guidelines
- Add comprehensive docstrings
- Include unit tests for new features
- Update documentation for API changes

---

## 📋 Requirements

### Core Dependencies
```txt
torch>=2.0.0
torchvision>=0.15.0
scikit-learn>=1.0.0
matplotlib>=3.5.0
seaborn>=0.11.0
tqdm>=4.64.0
pillow>=9.0.0
tensorboard>=2.10.0
```

### Optional Enhancements
```txt
albumentations>=1.3.0  # Advanced augmentations
opencv-python>=4.7.0   # Additional image processing
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **PyTorch Team** for the excellent deep learning framework
- **Agricultural Research Community** for domain expertise
- **Open Source Contributors** who make projects like this possible
- **Farmers and Gardeners** who inspire us to build better tools

---

## 📞 Support & Community

- **Issues**: [GitHub Issues](https://github.com/yourusername/plant-disease-classification/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/plant-disease-classification/discussions)
- **Email**: your.email@domain.com

---

<div align="center">

**🌱 Built with ❤️ for sustainable agriculture**

*If this project helps your agricultural research or farming operations, please consider giving it a ⭐*

</div>
