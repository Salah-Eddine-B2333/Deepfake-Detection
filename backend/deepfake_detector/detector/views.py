import cv2
import os
import numpy as np
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
import tensorflow as tf
from PIL import Image
import base64
from io import BytesIO
import tensorflow_hub as hub
import logging
import tempfile
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DeepfakeDetector:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        model_path = os.path.join('detector', 'ml_models', 'deepfake_model_ResNet50NewOne.h5')
        feature_extractor_path = os.path.join('detector', 'ml_models', 'cnn_face_classifier_weights.h5')
        lstm_model_path = os.path.join('detector', 'ml_models', 'cpu_gpu_lstm_model.h5')
        
        # Load ESRGAN model from TensorFlow Hub
        esrgan_path = "https://tfhub.dev/captain-pool/esrgan-tf2/1"
        
        try:
            # Load CNN model for single image classification
            self.model = tf.keras.models.load_model(model_path)
            
            # Load feature extractor for video frames
            self.feature_extractor = self.build_feature_extractor(feature_extractor_path)
            
            # Load LSTM model for video sequence analysis
            self.lstm_model = tf.keras.models.load_model(lstm_model_path)
            
            # Load super-resolution model
            self.upscale_model = hub.load(esrgan_path)
            
            logger.info("All models loaded successfully")
        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")
            self.image_model = None
            self.feature_extractor = None
            self.lstm_model = None
    
    def build_cnn(self):
        """Recreate the CNN architecture for feature extraction"""
        model = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(64, (5, 5), activation='relu', padding='same', input_shape=(256, 256, 3)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Dropout(0.2),
            
            tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Dropout(0.2),
            
            tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Dropout(0.2),
            
            tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D(2, 2),
            
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        return model
    
    def build_feature_extractor(self, weights_path):
        """Build the feature extractor from CNN model"""
        try:
            # Build the CNN model first
            feature_extractor = self.build_cnn()
            feature_extractor.load_weights(weights_path)
            
            # Create feature extractor by removing classification head
            feature_extractor_model = tf.keras.Model(
                inputs=feature_extractor.inputs,
                outputs=feature_extractor.layers[-3].output  # Output before Flatten
            )
            
            logger.info("Feature extractor built successfully")
            return feature_extractor_model
        except Exception as e:
            logger.error(f"Error building feature extractor: {str(e)}")
            return None
    
    
            
    def detect_faces(self, image):
        cv2_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2GRAY)
        # Adjust scaleFactor and minNeighbors for better face detection
        faces = self.face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1,  # Slightly increased 
            minNeighbors=6,   # More strict neighbor requirements
            minSize=(60, 60)  # Minimum face size to detect
        )
        return faces, cv2_image

    #Super Resolution Scale model
    def preprocessing(self, img):
        """Make this match your training preprocessing exactly"""
        if img.shape[0] < 4 or img.shape[1] < 4:
            raise ValueError("Image dimensions must be at least 4x4 pixels.")
        
        image_size = (tf.convert_to_tensor(img.shape[:-1]) // 4) * 4
        cropped_image = tf.image.crop_to_bounding_box(
            img, 0, 0, image_size[0], image_size[1])
        
        # Normalize to [0, 1] range expected by ESRGAN - match training exactly
        preprocessed_image = tf.cast(cropped_image, tf.float32) / 255.0
        return tf.expand_dims(preprocessed_image, 0)
    
        
    def srmodel(self, img):
        preprocessed_image = self.preprocessing(img)  
        new_image = self.upscale_model(preprocessed_image)  
        return tf.squeeze(new_image) 
    
    #Predict the face
    def predict_face(self, face_image, w, h):
        try:
            dimension = (256, 256)
            
            # Use the same logic for super-resolution as in training
            if w >= 256 and h >= 256:
                resized_face = cv2.resize(face_image, dimension, interpolation=cv2.INTER_AREA)
            else:
                face_tensor = tf.convert_to_tensor(face_image, dtype=tf.float32)
                sr_face = self.srmodel(face_tensor)
                sr_face = (sr_face.numpy() * 255).astype(np.uint8)
                resized_face = cv2.resize(sr_face, dimension, interpolation=cv2.INTER_AREA)
            
            # Normalize exactly as in training
            normalized_face = resized_face / 255.0
            
            # Add batch dimension
            face_batch = np.expand_dims(normalized_face, axis=0)
            
            # Get prediction
            prediction = self.model.predict(face_batch, verbose=0)
            
            # FIXED: Invert the class interpretation
            # If your model training had class 0 as "Fake" and class 1 as "Real"
            class_idx = np.argmax(prediction[0])
            is_deepfake = (class_idx == 0)  # Changed from 1 to 0 - assuming 0 is "Fake" , 1 is "Real"
            confidence = float(prediction[0][class_idx])
            
            logger.info(f"Raw prediction: {prediction[0]}, argmax: {class_idx}")
            logger.info(f"Classes from training: {self.model.output_names}")
            return is_deepfake, confidence
        
        except Exception as e:
            logger.error(f"Error in predict_face: {str(e)}")
            raise
        
        
    def extract_face_features(self, face_image):
        try:
            # Resize to expected dimensions
            face_resized = cv2.resize(face_image, (256, 256), interpolation=cv2.INTER_AREA)
            
            # Convert to RGB and normalize
            face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
            face_normalized = face_rgb.astype(np.float32) / 255.0
            
            # Add batch dimension
            face_batch = np.expand_dims(face_normalized, axis=0)
            
            # Extract features
            features = self.feature_extractor.predict(face_batch, verbose=0)
            
            return features[0]  # Remove batch dimension
        except Exception as e:
            logger.error(f"Error extracting face features: {str(e)}")
            return None

# *Image API View-------------------------------------------------------------------------------------------------------
class DetectDeepfakeView(APIView):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.detector = DeepfakeDetector()
        
        # Warm up the model with a dummy prediction
        if self.detector.model is not None:  # Added check
            dummy_input = np.zeros((1, 256, 256, 3), dtype=np.float32)
            _ = self.detector.model.predict(dummy_input)
            logger.info("Model warmed up successfully")

    def post(self, request):
        try:
            if self.detector.model is None:
                return Response({'error': 'Deepfake detection model not loaded'}, 
                                status=status.HTTP_500_INTERNAL_SERVER_ERROR)
            
            image_data = request.data.get('image', '').split(',')[1]
            image_bytes = base64.b64decode(image_data)
            image = Image.open(BytesIO(image_bytes))
            
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
                
            faces, cv2_image = self.detector.detect_faces(image)
            
            if len(faces) == 0:
                return Response({
                    'message': 'No faces detected',
                    'faces_detected': 0
                }, status=status.HTTP_200_OK)

            results = []
            face_images = []  # To store face images for visualization
            
            for (x, y, w, h) in faces:
                face_img = cv2_image[y:y+h, x:x+w]
                face_img_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
                
                # Get prediction (now returns tuple)
                is_deepfake, confidence = self.detector.predict_face(face_img_rgb, w, h)
                
                # Use the returned values directly
                color = (0, 0, 255) if is_deepfake else (0, 255, 0)
                cv2.rectangle(cv2_image, (x, y), (x+w, y+h), color, 2)
                label = f'{"Deepfake" if is_deepfake else "Real"} {confidence:.2%}'
                cv2.putText(cv2_image, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                
                # Store face image for visualization
                face_images.append(face_img)
                
                # Add to results  # Fixed: moved inside the loop
                results.append({
                    'face_coordinates': [int(x), int(y), int(w), int(h)],
                    'is_deepfake': is_deepfake,
                    'confidence': confidence
                })
            
            # Encode the processed main image
            _, buffer = cv2.imencode('.jpg', cv2_image)
            processed_image = base64.b64encode(buffer).decode('utf-8')
            
            # Don't try to encode the 'faces' coordinates array
            # Instead, create a montage of extracted face images if needed
            processed_faces = ""
            if face_images:
                # Create a simple montage or just use the first face
                face_montage = face_images[0]  # For simplicity, just using first face
                _, face_buffer = cv2.imencode('.jpg', face_montage)
                processed_faces = base64.b64encode(face_buffer).decode('utf-8')
            
            return Response({
                'processed_image': f'data:image/jpeg;base64,{processed_image}',
                'faces_detected': len(faces),
                'results': results,
                'faces': f'data:image/jpeg;base64,{processed_faces}' if face_images else ""
            }, status=status.HTTP_200_OK)

        except Exception as e:
            logger.error(f"Error details: {str(e)}")
            return Response({'error': str(e)}, status=status.HTTP_400_BAD_REQUEST)

# *Video API View-------------------------------------------------------------------------------------------------------------
class VideoDeepfakeDetectionView(APIView):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.detector = DeepfakeDetector()
        
        # Warm up the model with a dummy prediction
        if self.detector.model is not None:  # Added check
            dummy_input = np.zeros((1, 256, 256, 3), dtype=np.float32)
            _ = self.detector.model.predict(dummy_input)
            logger.info("Video detection model warmed up successfully")
        
        # Warm up LSTM model
        if self.detector.lstm_model:
            dummy_seq_input = np.zeros((1, 64, 16, 16, 256), dtype=np.float32)
            _ = self.detector.lstm_model.predict(dummy_seq_input)  # Fixed: was dummy_input
        
        logger.info("Video detection models warmed up successfully")
    
    def post(self, request):
        try:
            if self.detector.model is None:
                return Response({'error': 'Deepfake detection model not loaded'}, 
                                status=status.HTTP_500_INTERNAL_SERVER_ERROR)
            
            # Get the video file from the request
            video_file = request.FILES.get('video')
            if not video_file:
                return Response({'error': 'No video file provided'}, 
                                status=status.HTTP_400_BAD_REQUEST)
            
            # Save the video file temporarily
            temp_path = default_storage.save(f'temp_videos/{video_file.name}', ContentFile(video_file.read()))
            video_path = default_storage.path(temp_path)
            
            # Process the video
            results = self.process_video(video_path)
            
            # Clean up
            default_storage.delete(temp_path)
            
            return Response(results, status=status.HTTP_200_OK)
            
        except Exception as e:
            logger.error(f"Error in video processing: {str(e)}")
            return Response({'error': str(e)}, status=status.HTTP_400_BAD_REQUEST)
        
    
    
    
    def detect_faces_for_frames(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Adjust scaleFactor and minNeighbors for better face detection
        faces = self.detector.face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1,  # Slightly increased 
            minNeighbors=6,   # More strict neighbor requirements
            minSize=(50, 50)  # Minimum face size to detect
        )
        return faces
# *Process Video----------------------------------------------------------------------------------------------------------------------------------------------    
    def process_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise Exception("Error opening video file")
        
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = frame_count / fps if fps > 0 else 0

        # Sample frames at regular intervals
        sample_interval = max(1, int(fps / 5)) 
        max_frames = min(frame_count, int(fps * 150)) # Limit to 150 seconds max

        frames_processed = 0
        frames_with_faces = 0
        deepfake_frames = 0

        frame_results = []
        confidence_scores = []
        time_series_data = []

        while frames_processed < max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            # Only process every nth frame
            if frames_processed % sample_interval == 0:
                faces = self.detect_faces_for_frames(frame)

                if faces is not None and len(faces) > 0:
                    frames_with_faces += 1
                    frame_deepfakes = 0
                    frame_conf_scores = []

                    for (x, y, w, h) in faces:
                        face_img = frame[y:y+h, x:x+w]
                        face_img_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
                        is_deepfake, confidence = self.detector.predict_face(face_img_rgb, w, h)

                        adj_confidence = confidence if is_deepfake else -confidence
                        frame_conf_scores.append(adj_confidence)
                        confidence_scores.append(adj_confidence)

                        if is_deepfake:
                            frame_deepfakes += 1
                            deepfake_frames += 1

                    avg_frame_confidence = sum(frame_conf_scores) / len(frame_conf_scores) if frame_conf_scores else 0
                    timestamp = (frames_processed * sample_interval) / fps if fps > 0 else 0
                    frame_data = {
                        'frame_number': frames_processed,
                        'timestamp': timestamp,
                        'faces_detected': len(faces),
                        'deepfakes_detected': frame_deepfakes,
                        'confidence_score': avg_frame_confidence
                    }
                    frame_results.append(frame_data)
                    time_series_data.append([timestamp, avg_frame_confidence])

            frames_processed += 1

        cap.release()

        lstm_results = self.process_video_with_lstm(video_path)
        is_deepfake_lstm = lstm_results['detection_results']['final_decision']['is_deepfake']
        confidence_lstm = lstm_results['detection_results']['final_decision']['confidence']

        result = self.apply_classification_methods(
            confidence_scores, 
            time_series_data, 
            frame_results, 
            frames_with_faces, 
            deepfake_frames,
            is_deepfake_lstm,  
            confidence_lstm    
        )

        processed_video = self.process_and_save_video_lstm(video_path, lstm_results)

        final_result = {
            'video_info': {
                'total_frames': frame_count,
                'processed_frames': frames_processed,
                'duration_seconds': duration,
                'fps': fps
            },
            'detection_results': result,
            'frame_by_frame': frame_results,
            'processed_video': f'data:video/mp4;base64,{processed_video}'
        }

        return final_result
    
#*Process Video With LSTM--------------------------------------------------------------------------------------------------------------------------------------------    
    def process_video_with_lstm(self, video_path):
        """Process video using feature extraction + LSTM approach"""
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise Exception("Error opening video file")
        
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = frame_count / fps if fps > 0 else 0
        
        # Prepare for frame extraction
        max_frames = 64  # Match LSTM sequence length from training
        frame_interval = max(1, int(frame_count / max_frames))
        
        # Store frames and features
        frames_processed = 0
        faces_detected = 0
        
        # List to store feature maps
        feature_sequence = []
        frame_results = []  # To store frame-by-frame results for visualization
        
        current_frame = 0
        
        while len(feature_sequence) < max_frames and current_frame < frame_count:
            # Set frame position
            cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
            ret, frame = cap.read()
            
            if not ret:
                break
            
            # Detect faces
            faces = self.detect_faces_for_frames(frame)
            
            # If faces detected in this frame
            if len(faces) > 0:
                # Take the largest face
                face = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)[0]
                x, y, w, h = face
                
                # Extract face
                face_img = frame[y:y+h, x:x+w]
                
                # Extract features from this face
                face_features = self.detector.extract_face_features(face_img)
                
                if face_features is not None:
                    # Add to sequence
                    feature_sequence.append(face_features)
                    faces_detected += 1
                    
                    # Store frame info
                    frame_results.append({
                        'frame_number': current_frame,
                        'timestamp': current_frame / fps,
                        'faces_detected': len(faces),
                        'selected_face': [int(x), int(y), int(w), int(h)]
                    })
            
            # Move to next interval
            current_frame += frame_interval
            frames_processed += 1
        
        cap.release()
        
        # Check if we have enough frames with faces for prediction
        if len(feature_sequence) < 5:  # Minimum threshold for reliable prediction
            return {
                'video_info': {
                    'total_frames': frame_count,
                    'processed_frames': frames_processed,
                    'duration_seconds': duration,
                    'fps': fps
                },
                'detection_results': {
                    'final_decision': {
                        'is_deepfake': False,
                        'confidence': 0.0,
                        'message': 'Not enough face frames detected for reliable prediction'
                    }
                },
                'frame_by_frame': frame_results
            }
        
        # Pad sequence if needed
        if len(feature_sequence) < max_frames:
            # Use the last feature to pad (better than zeros)
            last_feature = feature_sequence[-1]
            padding_needed = max_frames - len(feature_sequence)
            feature_sequence.extend([last_feature] * padding_needed)
        
        # Convert to numpy array and make batch
        feature_sequence = np.array(feature_sequence)  # Shape: (64, 16, 16, 256)
        
        # Transpose to match LSTM model input shape if needed (sequence first)
        feature_sequence = np.transpose(feature_sequence, (0, 2, 1, 3))  # Shape: (64, 16, 16, 256)
        
        # Add batch dimension
        feature_sequence_batch = np.expand_dims(feature_sequence, axis=0)  # Shape: (1, 64, 16, 16, 256)
        
        # Get prediction from LSTM model
        predictions = self.detector.lstm_model.predict(feature_sequence_batch)
        
        # Process prediction
        prediction_class = np.argmax(predictions[0])
        confidence = float(predictions[0][prediction_class])
        is_deepfake = bool(prediction_class == 1)  # Assuming class 1 is "Fake" and 0 is "Real"
        
        # Create detailed results
        lstm_results = {
            'final_decision': {
                'is_deepfake': is_deepfake, 
                'confidence': confidence,
                'method': 'LSTM sequence analysis'
            }
        }
        
        # Generate final result
        final_result = {
            'video_info': {
                'total_frames': frame_count,
                'processed_frames': frames_processed,
                'frames_with_faces': faces_detected,
                'duration_seconds': duration,
                'fps': fps
            },
            'detection_results': lstm_results,
            'frame_by_frame': frame_results
        }
        
        return final_result
    
#*Classification Methodes Functions----------------------------------------------------------------------------------------------------------------------------    
    def apply_classification_methods(self, confidence_scores, time_series_data, frame_results, frames_with_faces, deepfake_frames, Is_deepfake , Confidence_LSTM):
        """Apply multiple methods to classify the video and return the most reliable classification"""
        result = {}
        
        # 1. Simple percentage method (your original approach)
        if frames_with_faces > 0:
            deepfake_percentage = (deepfake_frames / frames_with_faces) * 100
        else:
            deepfake_percentage = 0
        
        result['percentage_method'] = {
            'frames_with_faces': frames_with_faces,
            'frames_with_deepfakes': deepfake_frames,
            'deepfake_percentage': deepfake_percentage,
            'is_deepfake': deepfake_percentage > 40,  # Original threshold
        }
        
        # 2. Statistical analysis of confidence scores
        if confidence_scores:
            mean_confidence = sum(confidence_scores) / len(confidence_scores)
            # Calculate variance
            variance = sum((x - mean_confidence) ** 2 for x in confidence_scores) / len(confidence_scores)
            std_dev = variance ** 0.5
            
            # Weighted confidence (gives more importance to higher confidence scores)
            weighted_conf = sum(abs(c) * c for c in confidence_scores) / sum(abs(c) for c in confidence_scores) if confidence_scores else 0
            
            result['statistical_method'] = {
                'mean_confidence': mean_confidence,
                'std_deviation': std_dev,
                'weighted_confidence': weighted_conf,
                'is_deepfake': mean_confidence > 0 or weighted_conf > 0
            }
        
        # 3. Temporal analysis - check for consistency across time
        if time_series_data:
            # Convert to numpy for easier processing
            ts_data = np.array(time_series_data)
            
            # Check for temporal consistency (abrupt changes might indicate editing)
            if len(ts_data) > 1:
                diffs = np.abs(np.diff(ts_data[:, 1]))
                max_change = np.max(diffs) if len(diffs) > 0 else 0
                avg_change = np.mean(diffs) if len(diffs) > 0 else 0
                
                # Calculate moving average to detect sustained patterns
                window_size = min(5, len(ts_data))
                if window_size > 1:
                    moving_avgs = np.convolve(ts_data[:, 1], np.ones(window_size)/window_size, mode='valid')
                    sustained_pattern = np.mean(moving_avgs) if len(moving_avgs) > 0 else 0
                else:
                    sustained_pattern = 0
                
                result['temporal_method'] = {
                    'max_confidence_change': float(max_change),
                    'avg_confidence_change': float(avg_change),
                    'sustained_pattern': float(sustained_pattern),
                    'is_deepfake': sustained_pattern > 0.25 or (avg_change > 0.5)
                }
        
        # 4. Clustering-based approach - use KMeans to cluster frame confidences
        if len(confidence_scores) >= 5:  # Need sufficient samples
            try:
                from sklearn.cluster import KMeans
                
                # Reshape for KMeans
                X = np.array(confidence_scores).reshape(-1, 1)
                
                # Use 2 clusters (real vs fake)
                kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
                clusters = kmeans.fit_predict(X)
                
                # Calculate means of each cluster
                cluster_means = [np.mean(X[clusters == i]) for i in range(2)]
                
                # Determine which cluster is more likely to be deepfake (higher mean)
                fake_cluster = np.argmax(cluster_means)
                fake_percentage = np.sum(clusters == fake_cluster) / len(clusters) * 100
                
                result['clustering_method'] = {
                    'cluster_means': [float(cm) for cm in cluster_means],
                    'fake_cluster_percentage': float(fake_percentage),
                    'is_deepfake': fake_percentage > 40 and cluster_means[fake_cluster] > 0.3
                }
            except Exception as e:
                logger.warning(f"Clustering analysis failed: {str(e)}")
        # 5. LSTM model classification
        if Is_deepfake != None:
            result['LSTM_method'] = {
            'is_deepfake': Is_deepfake,
            'Confidence_LSTM': Confidence_LSTM
        }
        
        
        # 6. Make final decision based on ensemble of methods
        methods_results = []
        
        # Collect results from each method
        if 'percentage_method' in result:
            methods_results.append(result['percentage_method']['is_deepfake'])
            
        if 'statistical_method' in result:
            methods_results.append(result['statistical_method']['is_deepfake'])
            
        if 'temporal_method' in result:
            methods_results.append(result['temporal_method']['is_deepfake'])
            
        if 'clustering_method' in result:
            methods_results.append(result['clustering_method']['is_deepfake'])
        if 'LSTM_method' in result:
            methods_results.append(result['LSTM_method']['is_deepfake'])
        
        # Final decision by majority voting
        is_deepfake = sum(methods_results) > len(methods_results) / 2
        
        # Calculate confidence based on agreement between methods
        agreement_percentage = sum(methods_results) / len(methods_results) * 100 if methods_results else 50
        
        # Add ensemble decision to results
        result['final_decision'] = {
            'is_deepfake': is_deepfake,
            'confidence': min(100, agreement_percentage + 20 if is_deepfake else 100 - agreement_percentage),
            'method_agreement': f"{sum(methods_results)}/{len(methods_results)} methods"
        }
        
        return result
#* for LSTM verison----------------------------------------------------------------------------------------------------------------------------------------------    
    def process_and_save_video_lstm(self, video_path, results):
        """Process video and add visual indicators based on detection results"""
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise Exception("Error opening video file")
        
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Create a temporary file for the output video
        output_path = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False).name
        
        # Define the codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Get the prediction results
        final_decision = results['detection_results']['final_decision']
        is_deepfake = final_decision['is_deepfake']
        confidence = final_decision['confidence']
        
        # Get frame-by-frame results
        frame_results = {fr['frame_number']: fr for fr in results.get('frame_by_frame', [])}
        
        # Process video and add indicators
        current_frame = 0
        
        # To prevent processing all frames, set a maximum
        max_frames = min(frame_count, int(fps * 120))  # 120 seconds max
        
        # Create color based on detection (red for deepfake, green for real)
        result_color = (0, 0, 255) if is_deepfake else (0, 255, 0)
        
        while current_frame < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Check if this frame has detection data
            if current_frame in frame_results:
                frame_data = frame_results[current_frame]
                
                # Draw rectangle around detected face
                if 'selected_face' in frame_data:
                    x, y, w, h = frame_data['selected_face']
                    cv2.rectangle(frame, (x, y), (x+w, y+h), result_color, 2)
            
            # Add overall result as text overlay
            result_text = f"{'DEEPFAKE' if is_deepfake else 'REAL'} - {confidence:.2%}"
            cv2.putText(
                frame, 
                result_text, 
                (30, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.9, 
                result_color, 
                2
            )
            
            # Write frame to output
            out.write(frame)
            current_frame += 1
            
        # Release resources
        cap.release()
        out.release()
        
        # Read the output video and encode it
        with open(output_path, 'rb') as video_file:
            video_data = video_file.read()
        
        # Clean up
        os.remove(output_path)
        
        # Encode the video to base64
        encoded_video = base64.b64encode(video_data).decode('utf-8')
        
        return encoded_video