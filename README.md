# Brain Tumor MRI Classifier 🧠

A deep learning application for automated brain tumor detection and classification using transfer learning with EfficientNetV2. This system analyzes MRI brain scans and classifies tumors with high accuracy using a simple Tkinter-based GUI.

![Brain Tumor Classifier Demo](Capture%20d'écran%202025-11-22%20140314.png)

## 🎯 Features

- **Transfer Learning**: Leverages EfficientNetV2B0 pre-trained on ImageNet
- **Data Augmentation**: Random flips, rotations, and zoom for robust training
- **Automatic Class Detection**: Dynamically identifies tumor types from dataset structure
- **Real-time Predictions**: Instant classification with confidence scores
- **User-Friendly GUI**: Simple Tkinter interface for uploading and analyzing MRI scans
- **Model Checkpointing**: Automatically saves the best performing model during training

## 📋 Requirements

### Prerequisites

- Python 3.8 or higher
- NVIDIA GPU (recommended for training)
- 4GB+ RAM

### Dependencies

```
tensorflow>=2.10.0
numpy>=1.21.0
opencv-python>=4.6.0
pillow>=9.0.0
```

Install all dependencies:
```bash
pip install tensorflow numpy opencv-python pillow
```

## 🚀 Quick Start

### 1. Prepare Your Dataset

Organize your MRI images in the following structure:

```
data/
├── glioma/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── meningioma/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
└── pituitary/
    ├── image1.jpg
    ├── image2.jpg
    └── ...
```
Dataset from : https://www.kaggle.com/datasets/fernando2rad/brain-tumor-mri-images-44c
Each subfolder name becomes a tumor class automatically.

### 2. Train the Model

```bash
python train.py
```

**Training Parameters:**
- Image Size: 224x224 pixels
- Batch Size: 32
- Learning Rate: 0.001
- Max Epochs: 50
- Validation Split: 20%

The best model will be saved as `best_mri_classifier.h5`.

### 3. Run the Classifier

```bash
python chatboot.py
```

Click "Select MRI Image" to upload a scan and view the prediction with confidence score.

## 🏗️ Project Structure

```
Brain-Tumor-Classifier-CNN/
├── train.py              # Model training script
├── chatboot.py           # GUI application
├── best_mri_classifier.h5  # Trained model (generated)
├── data/                 # Dataset directory
│   ├── glioma/
│   ├── meningioma/
│   └── pituitary/...
└── README.md            # Documentation
```

## 🔬 Model Architecture

### Base Model
- **EfficientNetV2B0** (pre-trained on ImageNet)
- Input shape: 224×224×3
- Frozen base layers for transfer learning

### Custom Layers
1. Data Augmentation (Random Flip, Rotation, Zoom)
2. EfficientNetV2 Preprocessing
3. Global Average Pooling 2D
4. Dropout (0.3)
5. Dense Output Layer (Softmax activation)

### Training Features
- **Loss Function**: Sparse Categorical Crossentropy
- **Optimizer**: Adam (lr=0.001)
- **Callbacks**:
  - Model Checkpoint (saves best model)
  - Early Stopping (patience=10)

## 📊 Data Augmentation

The model includes built-in augmentation to improve generalization:

- Horizontal and vertical flipping
- Random rotation (±20%)
- Random zoom (±20%)

## 🎮 Using the Application

### GUI Interface

1. **Launch Application**: Run `python chatboot.py`
2. **Select Image**: Click "Select MRI Image" button
3. **View Results**: See prediction and confidence percentage
4. **Try Another**: Select a new image to classify

### Supported Image Formats
- JPEG (.jpg, .jpeg)
- PNG (.png)

## 📈 Training Output

During training, you'll see:
```
Found classes: ['glioma' 'meningioma' 'pituitary' ...]
Epoch 1/50
45/45 [==============================] - 12s 268ms/step - loss: 0.4521 - accuracy: 0.8234 - val_loss: 0.2145 - val_accuracy: 0.9123
...
✅ Training finished. Model saved as 'best_mri_classifier.h5'
```

## ⚙️ Configuration

Edit constants in `train.py` to customize training:

```python
IMAGE_SIZE = (224, 224)  # Input image dimensions
BATCH_SIZE = 32          # Samples per training batch
DATA_DIR = 'data/'       # Dataset location
LEARNING_RATE = 0.001    # Optimizer learning rate
EPOCHS = 50              # Maximum training epochs
```

## 🔧 Troubleshooting

### Common Issues

**Issue**: "Image not found or invalid path"
- Ensure image file exists and path is correct
- Check image format (must be .jpg, .jpeg, or .png)

**Issue**: Model file not found
- Run `train.py` first to generate the model
- Verify `best_mri_classifier.h5` exists in project directory

**Issue**: Low accuracy
- Increase dataset size (min. 100 images per class)
- Train for more epochs
- Adjust data augmentation parameters

## 📝 Code Explanation

### train.py
- Loads dataset from folder structure
- Applies data augmentation
- Builds transfer learning model
- Trains with validation split
- Saves best model

### chatboot.py
- Loads trained model
- Provides GUI for image selection
- Preprocesses input images
- Displays predictions with confidence

## ⚠️ Important Disclaimer

**Medical Use Warning**: This application is designed for educational and research purposes only. It is **NOT** intended for clinical diagnosis or medical decision-making. Always consult qualified healthcare professionals for medical advice and diagnosis.

## 🤝 Contributing

Contributions are welcome! To contribute:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit changes (`git commit -am 'Add feature'`)
4. Push to branch (`git push origin feature/improvement`)
5. Open a Pull Request



## 🙏 Acknowledgments

- **EfficientNetV2** by Google Research
- TensorFlow and Keras communities
- Public brain tumor MRI datasets
- Medical imaging researchers

## 🔮 Future Enhancements

- [ ] Add visualization of model attention (Grad-CAM)
- [ ] Export detailed prediction reports (PDF)
- [ ] Support for DICOM medical image format
- [ ] Multi-model ensemble for improved accuracy
- [ ] Web-based interface using Flask
- [ ] Confidence threshold alerts
- [ ] Batch prediction mode
- [ ] Performance metrics dashboard

## 📧 Contact

For questions, suggestions, or collaborations:

- **GitHub**: https://github.com/AjahoudOmayma
- **Email**: omaymaajahoud@gmail.com


---

**⭐ If you find this project helpful, please star the repository!**

## 📚 Additional Resources

- [EfficientNetV2 Paper](https://arxiv.org/abs/2104.00298)
- [TensorFlow Documentation](https://www.tensorflow.org/)
- [Medical Image Analysis Guide](https://www.ncbi.nlm.nih.gov/pmc/articles/)

**Built with ❤️ for medical AI research**
