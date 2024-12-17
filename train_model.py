import os
import numpy as np
import librosa
import soundfile as sf
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from gunshot_detection import GunShotDetector
import tensorflow as tf
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import random
from tqdm import tqdm

def augment_audio(audio, sample_rate):
    """Apply random augmentation to audio"""
    try:
        augmented = audio.copy()
        
        # Random time shift
        shift = int(random.uniform(-0.1, 0.1) * len(audio))
        if shift > 0:
            augmented = np.pad(audio, (shift, 0), mode='constant')[:-shift]
        else:
            augmented = np.pad(audio, (0, -shift), mode='constant')[-shift:]
        
        # Random pitch shift
        n_steps = random.uniform(-3, 3)
        augmented = librosa.effects.pitch_shift(augmented, sr=sample_rate, n_steps=n_steps)
        
        # Random speed change
        speed_factor = random.uniform(0.9, 1.1)
        augmented = librosa.effects.time_stretch(augmented, rate=speed_factor)
        
        # Random noise injection
        noise_factor = random.uniform(0.001, 0.005)
        noise = np.random.normal(0, 1, len(augmented))
        augmented = augmented + noise_factor * noise
        
        return augmented
    except Exception as e:
        print(f"Error in augmentation: {str(e)}")
        return audio

def load_dataset(dataset_path, sample_rate=44100, augment=True, augment_factor=2):
    """Load and preprocess the gunshot dataset with augmentation"""
    X = []
    y = []
    
    detector = GunShotDetector(sample_rate=sample_rate)
    
    # Get all gun type folders
    gun_types = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]
    
    for gun_type in tqdm(gun_types, desc="Processing gun types"):
        gun_dir = os.path.join(dataset_path, gun_type)
        print(f"\nProcessing {gun_type}...")
        
        audio_files = [f for f in os.listdir(gun_dir) if f.endswith('.wav')]
        
        for audio_file in tqdm(audio_files, desc=f"Processing {gun_type} files"):
            file_path = os.path.join(gun_dir, audio_file)
            try:
                # Load audio file
                audio_data, sr = librosa.load(file_path, sr=sample_rate)
                
                # Original audio features
                features = detector.extract_features(audio_data)
                if features is not None and not np.any(np.isnan(features)):
                    X.append(features)
                    y.append(gun_type)
                
                # Augmented versions
                if augment:
                    for _ in range(augment_factor):
                        aug_audio = augment_audio(audio_data, sr)
                        aug_features = detector.extract_features(aug_audio)
                        if aug_features is not None and not np.any(np.isnan(aug_features)):
                            X.append(aug_features)
                            y.append(gun_type)
                        
            except Exception as e:
                print(f"Error processing {file_path}: {str(e)}")
                continue
    
    if len(X) == 0:
        raise ValueError("No valid features extracted from the dataset")
    
    return np.array(X), np.array(y)

def plot_training_history(history):
    """Plot training history"""
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.close()

def plot_confusion_matrix(y_true, y_pred, classes):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.close()

def main():
    # Parameters
    dataset_path = "dataset"
    sample_rate = 44100
    test_size = 0.2
    validation_size = 0.2
    random_state = 42
    augment = True
    augment_factor = 2
    
    print("Loading and augmenting dataset...")
    try:
        X, y = load_dataset(dataset_path, sample_rate, augment, augment_factor)
    except Exception as e:
        print(f"Error loading dataset: {str(e)}")
        return
    
    print(f"\nDataset loaded successfully:")
    print(f"Number of samples: {len(X)}")
    print(f"Feature vector size: {X.shape[1]}")
    print(f"Number of classes: {len(np.unique(y))}")
    
    # Initialize detector and encode labels
    detector = GunShotDetector(sample_rate=sample_rate)
    detector.label_encoder.fit(y)
    y_encoded = tf.keras.utils.to_categorical(detector.label_encoder.transform(y))
    
    # Split dataset
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y_encoded, test_size=test_size, random_state=random_state, stratify=y
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=validation_size, 
        random_state=random_state, stratify=y_trainval
    )
    
    # Scale features
    detector.scaler.fit(X_train)
    X_train_scaled = detector.scaler.transform(X_train)
    X_val_scaled = detector.scaler.transform(X_val)
    X_test_scaled = detector.scaler.transform(X_test)
    
    # Build and train model
    print("\nTraining model...")
    detector.model = detector.build_model()
    
    # Add callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=10,
            restore_best_weights=True,
            min_delta=0.001
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=0.00001,
            min_delta=0.001
        ),
        tf.keras.callbacks.ModelCheckpoint(
            'best_model.h5',
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        tf.keras.callbacks.CSVLogger('training_log.csv')
    ]
    
    # Train with class weights to handle imbalance
    class_counts = np.sum(y_train, axis=0)
    total = np.sum(class_counts)
    class_weights = {i: total / (len(class_counts) * count) for i, count in enumerate(class_counts)}
    
    history = detector.model.fit(
        X_train_scaled, y_train,
        validation_data=(X_val_scaled, y_val),
        epochs=100,
        batch_size=32,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1
    )
    
    # Plot training history
    plot_training_history(history)
    
    # Load best model
    detector.model.load_weights('best_model.h5')
    
    # Evaluate model
    print("\nEvaluating model...")
    y_pred = np.argmax(detector.model.predict(X_test_scaled), axis=1)
    y_test_decoded = np.argmax(y_test, axis=1)
    
    accuracy = accuracy_score(y_test_decoded, y_pred)
    print(f"\nTest Accuracy: {accuracy * 100:.2f}%")
    
    print("\nClassification Report:")
    class_names = detector.label_encoder.classes_
    print(classification_report(y_test_decoded, y_pred, target_names=class_names))
    
    # Plot confusion matrix
    plot_confusion_matrix(y_test_decoded, y_pred, class_names)
    
    # Save model and preprocessing objects
    print("\nSaving model and preprocessing objects...")
    detector.save_model('gunshot_model.h5')
    
    # Save scaler and label encoder
    import pickle
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(detector.scaler, f)
    with open('label_encoder.pkl', 'wb') as f:
        pickle.dump(detector.label_encoder, f)
    
    print("Training complete! Model and preprocessing objects saved.")

if __name__ == "__main__":
    main() 