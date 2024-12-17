import numpy as np
import librosa
import soundfile as sf
from sklearn.preprocessing import LabelEncoder, StandardScaler
import tensorflow as tf
from tensorflow.keras import layers, models
import pickle
import threading
import queue
import pyaudio
import math
import scipy.signal
from scipy.signal import hilbert
import scipy.stats
from datetime import datetime

class GunShotDetector:
    def __init__(self, model_path=None, sample_rate=44100, n_mics=4):
        self.sample_rate = sample_rate
        self.n_mics = n_mics
        
        # Microphone array positions (can be customized based on shoulder mount)
        self.mic_positions = np.array([
            [0, 0, 0],      # Front
            [0, 10, 0],     # Right
            [-10, 0, 0],    # Back
            [-10, 10, 0]    # Left
        ])  # Microphone positions in cm
        
        # Audio processing parameters
        self.n_mfcc = 40
        self.hop_length = 512  # Increased for better stability
        self.n_fft = 2048     # Increased for better frequency resolution
        self.window_size = int(0.025 * sample_rate)  # 25ms window
        
        # Initialize models and encoders
        self.model = self._load_model(model_path) if model_path else None
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        
        # Real-time processing
        self.audio_queues = [queue.Queue() for _ in range(n_mics)]
        self.is_running = False
        self.detection_threshold = 0.98  # High confidence threshold
        
        # Initialize audio streams
        self.audio = pyaudio.PyAudio()
        self.streams = []

    def extract_features(self, audio_data):
        """Extract comprehensive audio features optimized for gunshot detection"""
        try:
            # Ensure audio is the right length
            target_length = 2 * self.sample_rate  # 2 seconds
            if len(audio_data) > target_length:
                audio_data = audio_data[:target_length]
            elif len(audio_data) < target_length:
                audio_data = np.pad(audio_data, (0, target_length - len(audio_data)))
            
            # Normalize audio
            audio_data = librosa.util.normalize(audio_data)
            
            features = []
            
            # 1. MFCC Features (fixed length)
            mfccs = librosa.feature.mfcc(y=audio_data, sr=self.sample_rate, 
                                       n_mfcc=self.n_mfcc, n_fft=self.n_fft,
                                       hop_length=self.hop_length)
            mfcc_stats = np.hstack([
                np.mean(mfccs, axis=1),
                np.std(mfccs, axis=1),
                np.max(mfccs, axis=1),
                np.min(mfccs, axis=1)
            ])
            features.extend(mfcc_stats)

            # 2. Spectral Features (fixed length)
            stft = np.abs(librosa.stft(audio_data, n_fft=self.n_fft, hop_length=self.hop_length))
            
            # Spectral centroid
            spec_cent = librosa.feature.spectral_centroid(S=stft, sr=self.sample_rate)[0]
            spec_cent_stats = [np.mean(spec_cent), np.std(spec_cent)]
            features.extend(spec_cent_stats)

            # Spectral rolloff
            spec_roll = librosa.feature.spectral_rolloff(S=stft, sr=self.sample_rate)[0]
            spec_roll_stats = [np.mean(spec_roll), np.std(spec_roll)]
            features.extend(spec_roll_stats)

            # Zero crossing rate (fixed windows)
            zcr = librosa.feature.zero_crossing_rate(audio_data, frame_length=self.n_fft, 
                                                   hop_length=self.hop_length)[0]
            zcr_stats = [np.mean(zcr), np.std(zcr)]
            features.extend(zcr_stats)

            # 3. Gunshot-specific features
            # Energy ratio in high frequencies
            freqs = librosa.fft_frequencies(sr=self.sample_rate, n_fft=self.n_fft)
            high_freq_mask = freqs >= 2000
            high_freq_energy = np.sum(np.abs(stft[high_freq_mask, :]))
            total_energy = np.sum(np.abs(stft))
            energy_ratio = high_freq_energy / (total_energy + 1e-10)
            features.append(energy_ratio)

            # Attack and decay
            envelope = np.abs(hilbert(audio_data))
            peak_idx = np.argmax(envelope)
            attack_time = peak_idx / self.sample_rate
            decay_time = (len(envelope) - peak_idx) / self.sample_rate
            features.extend([attack_time, decay_time])

            # 4. Statistical features
            features.extend([
                np.mean(audio_data),
                np.std(audio_data),
                scipy.stats.skew(audio_data),
                scipy.stats.kurtosis(audio_data),
                np.max(np.abs(audio_data)),
                np.sum(np.abs(audio_data))
            ])

            # Ensure all features are finite
            features = np.array(features, dtype=np.float32)
            features[~np.isfinite(features)] = 0
            
            return features

        except Exception as e:
            print(f"Error in feature extraction: {str(e)}")
            # Return zero vector of expected length in case of error
            return np.zeros(self.n_mfcc * 4 + 14, dtype=np.float32)

    def build_model(self):
        """Build CNN model optimized for gunshot detection"""
        input_shape = len(self.extract_features(np.zeros(self.sample_rate)))
        
        model = models.Sequential([
            layers.Input(shape=(input_shape,)),
            layers.Reshape((-1, 1)),
            
            # First Convolutional Block
            layers.Conv1D(64, 3, activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv1D(64, 3, activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling1D(2),
            layers.Dropout(0.2),
            
            # Second Convolutional Block
            layers.Conv1D(128, 3, activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv1D(128, 3, activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling1D(2),
            layers.Dropout(0.3),
            
            # Third Convolutional Block
            layers.Conv1D(256, 3, activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv1D(256, 3, activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.GlobalAveragePooling1D(),
            layers.Dropout(0.4),
            
            # Dense Layers
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(len(self.label_encoder.classes_), activation='softmax')
        ])
        
        # Compile with Adam optimizer and learning rate schedule
        initial_learning_rate = 0.001
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate,
            decay_steps=1000,
            decay_rate=0.9,
            staircase=True)
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        return model

    def calculate_direction(self, time_delays):
        """Calculate direction of gunshot using time delays between microphones"""
        if not time_delays:
            return None
            
        c = 343.2  # Speed of sound in m/s
        angles = []
        
        # Calculate angles using time differences
        for i in range(self.n_mics):
            for j in range(i + 1, self.n_mics):
                if time_delays[i][j] != 0:
                    # Convert positions to meters
                    pos_i = self.mic_positions[i] / 100
                    pos_j = self.mic_positions[j] / 100
                    
                    # Calculate distance between microphones
                    d = np.linalg.norm(pos_i - pos_j)
                    
                    # Calculate angle using time difference of arrival
                    theta = math.acos((c * time_delays[i][j]) / d)
                    
                    # Convert to spherical coordinates
                    azimuth = math.atan2(pos_j[1] - pos_i[1], pos_j[0] - pos_i[0])
                    elevation = math.asin((pos_j[2] - pos_i[2]) / d)
                    
                    angles.append((theta, azimuth, elevation))
        
        if not angles:
            return None
            
        # Average the angles
        mean_theta = np.mean([a[0] for a in angles])
        mean_azimuth = np.mean([a[1] for a in angles])
        mean_elevation = np.mean([a[2] for a in angles])
        
        # Convert to degrees
        return {
            'azimuth': np.degrees(mean_azimuth) % 360,
            'elevation': np.degrees(mean_elevation),
            'confidence': self._calculate_angle_confidence(angles)
        }

    def _calculate_angle_confidence(self, angles):
        """Calculate confidence level of angle detection"""
        if not angles:
            return 0.0
            
        # Calculate standard deviation of angles
        theta_std = np.std([a[0] for a in angles])
        azimuth_std = np.std([a[1] for a in angles])
        elevation_std = np.std([a[2] for a in angles])
        
        # Convert standard deviations to confidence scores (inverse relationship)
        max_std = math.pi / 4  # 45 degrees
        confidence = 1.0 - min(1.0, (theta_std + azimuth_std + elevation_std) / (3 * max_std))
        return confidence

    def start_monitoring(self):
        """Start real-time monitoring with all microphones"""
        self.is_running = True
        
        # Initialize streams for each microphone
        for mic_idx in range(self.n_mics):
            stream = self.audio.open(
                format=pyaudio.paFloat32,
                channels=1,
                rate=self.sample_rate,
                input=True,
                input_device_index=mic_idx,
                stream_callback=lambda in_data, frame_count, time_info, status, q=self.audio_queues[mic_idx]:
                    self._audio_callback(in_data, frame_count, time_info, status, q)
            )
            self.streams.append(stream)
            stream.start_stream()
        
        # Start processing thread
        self.processing_thread = threading.Thread(target=self._process_audio)
        self.processing_thread.start()

    def _audio_callback(self, in_data, frame_count, time_info, status, queue):
        """Callback for audio stream"""
        if self.is_running:
            queue.put(in_data)
        return (None, pyaudio.paContinue)

    def _process_audio(self):
        """Process audio from all microphones"""
        while self.is_running:
            # Get audio from all microphones
            audio_frames = []
            timestamps = []
            
            for q in self.audio_queues:
                if not q.empty():
                    audio_frames.append(np.frombuffer(q.get(), dtype=np.float32))
                    timestamps.append(datetime.now())
            
            if len(audio_frames) == self.n_mics:
                # Process each audio frame
                detections = []
                for frame in audio_frames:
                    gun_type, confidence = self._detect_gunshot(frame)
                    if gun_type and confidence > self.detection_threshold:
                        detections.append((gun_type, confidence))
                
                # If we have detections from multiple microphones
                if len(detections) >= 2:
                    # Calculate time delays between microphones
                    time_delays = self._calculate_time_delays(audio_frames, timestamps)
                    
                    # Get direction
                    direction = self.calculate_direction(time_delays)
                    
                    if direction:
                        # Create detection event
                        event = {
                            'timestamp': timestamps[0].isoformat(),
                            'gun_type': detections[0][0],
                            'confidence': float(detections[0][1]),
                            'direction': direction,
                            'num_detections': len(detections)
                        }
                        
                        # Handle detection
                        self._handle_detection(event)

    def _detect_gunshot(self, audio_frame):
        """Detect gunshot in audio frame"""
        features = self.extract_features(audio_frame)
        features_scaled = self.scaler.transform(features.reshape(1, -1))
        
        if self.model:
            predictions = self.model.predict(features_scaled, verbose=0)
            gun_type = self.label_encoder.inverse_transform([np.argmax(predictions)])[0]
            confidence = float(np.max(predictions))
            return gun_type, confidence
        
        return None, 0.0

    def _calculate_time_delays(self, audio_frames, timestamps):
        """Calculate time delays between microphone pairs"""
        delays = [[0 for _ in range(self.n_mics)] for _ in range(self.n_mics)]
        
        for i in range(self.n_mics):
            for j in range(i + 1, self.n_mics):
                # Calculate cross-correlation
                correlation = scipy.signal.correlate(audio_frames[i], audio_frames[j])
                max_idx = np.argmax(correlation)
                
                # Convert to time delay
                delay = (max_idx - len(audio_frames[i])) / self.sample_rate
                
                # Add timestamp difference
                delay += (timestamps[j] - timestamps[i]).total_seconds()
                
                delays[i][j] = delay
                delays[j][i] = -delay
        
        return delays

    def _handle_detection(self, event):
        """Handle gunshot detection event"""
        print(f"Gunshot detected!")
        print(f"Type: {event['gun_type']}")
        print(f"Confidence: {event['confidence']:.2f}")
        print(f"Direction: Azimuth {event['direction']['azimuth']:.1f}°, "
              f"Elevation {event['direction']['elevation']:.1f}°")
        print(f"Direction Confidence: {event['direction']['confidence']:.2f}")
        print("---")

    def stop_monitoring(self):
        """Stop real-time monitoring"""
        self.is_running = False
        
        # Stop and close all streams
        for stream in self.streams:
            stream.stop_stream()
            stream.close()
        
        self.audio.terminate()
        self.processing_thread.join()

    def _load_model(self, model_path):
        """Load trained model from file"""
        return tf.keras.models.load_model(model_path)

    def save_model(self, model_path):
        """Save trained model to file"""
        if self.model:
            self.model.save(model_path)

if __name__ == "__main__":
    # Example usage
    detector = GunShotDetector()
    
    try:
        detector.start_monitoring()
    except KeyboardInterrupt:
        detector.stop_monitoring() 