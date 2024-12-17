# Gunshot Detection System

A deep learning-based gunshot detection system with real-time 360-degree coverage and gun type classification capabilities.

## Features
- Real-time gunshot detection with >98% accuracy target
- Gun type classification for multiple weapon types
- Shot angle detection using triangulation
- 360-degree coverage using multiple microphones
- Shoulder-mounted device design

## Dataset Structure
The system expects audio data organized in the following structure:
```
dataset/
    AK-47/
        *.wav files
    AK-12/
        *.wav files
    M16/
        *.wav files
    M249/
        *.wav files
    MP5/
        *.wav files
    MG-42/
        *.wav files
    IMI Desert Eagle/
        *.wav files
    Zastava M92/
        *.wav files
```

## Setup for Google Colab
1. Clone this repository:
```bash
!git clone [your-repo-url]
cd GunShotDetection
```

2. Install required packages:
```bash
!pip install -r requirements.txt
```

3. Run training:
```python
!python train_model.py
```

## Files Description
- `gunshot_detection.py`: Main detection class with feature extraction and model architecture
- `train_model.py`: Training script with data loading and augmentation
- `requirements.txt`: Required Python packages

## Model Architecture
- CNN-based architecture with multiple convolutional blocks
- Feature extraction using MFCCs, spectral features, and gunshot-specific characteristics
- Real-time processing capabilities with multiple microphone inputs

## Training
The model uses various data augmentation techniques:
- Random time shifts
- Pitch shifting
- Speed changes
- Noise injection

## Results
Training metrics and model performance will be saved as:
- `training_history.png`: Loss and accuracy curves
- `confusion_matrix.png`: Classification performance visualization
- `training_log.csv`: Detailed training metrics
- `best_model.h5`: Best performing model weights 