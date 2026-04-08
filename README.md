# Brain Tumor Classifier — CNN Image Classifier

A multi-class Convolutional Neural Network (CNN) to classify brain 
MRI scans into 4 tumor categories.

## Dataset
- **Classes**: Glioma · Meningioma · Pituitary Tumor · No Tumor
- **Training images**: 5,712
- **Testing images**: 1,311
- **Input size**: 150×150 grayscale

## Model Architecture
Conv2D(64, 3×3, ReLU)
→ MaxPooling2D
→ Flatten
→ Dense(256, ReLU)
→ Dropout(0.5)
→ Dense(128, ReLU)
→ Dropout(0.5)
→ Dense(4, Softmax)

## Key Features
- Data caching and prefetching pipelines for efficient GPU training
- Dropout layers to reduce overfitting
- Categorical cross-entropy loss with Adam optimizer
- GPU-accelerated training (NVIDIA RTX 3060)

## Tech Stack
- Python
- TensorFlow / Keras
- NumPy
- Matplotlib
