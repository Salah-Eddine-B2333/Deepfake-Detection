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
        model_path = os.path.join('detector', 'ml_models', 'deepfake_model_MobileNetV2.h5')
        # Load ESRGAN model from TensorFlow Hub
        esrgan_path = "https://tfhub.dev/captain-pool/esrgan-tf2/1"
        
        try:
            self.model = tf.keras.models.load_model(model_path)
            self.upscal_model = hub.load(esrgan_path)
            logger.info("Models loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            self.model = None
            
    def detect_faces(self, image):
        cv2_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2GRAY)
        # Adjust scaleFactor and minNeighbors for better face detection
        faces = self.face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1,  # Slightly increased 
            minNeighbors=5,   # More strict neighbor requirements
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
        new_image = self.upscal_model(preprocessed_image)  
        return tf.squeeze(new_image) 
    
    #Predict the face
    def predict_face(self, face_image, w, h):
        try:
            dimension = (256, 256)
            
            # Apply exact same preprocessing as training
            face_rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
            
            # Use the same logic for super-resolution as in training
            if w >= 256 and h >= 256:
                resized_face = cv2.resize(face_rgb, dimension, interpolation=cv2.INTER_AREA)
            else:
                face_tensor = tf.convert_to_tensor(face_rgb, dtype=tf.float32)
                sr_face = self.srmodel(face_tensor)
                sr_face = (sr_face.numpy() * 255).astype(np.uint8)
                resized_face = cv2.resize(sr_face, dimension, interpolation=cv2.INTER_CUBIC)
            
            # Normalize exactly as in training
            normalized_face = resized_face / 255.0
            
            # Add batch dimension
            face_batch = np.expand_dims(normalized_face, axis=0)
            
            # Get prediction
            prediction = self.model.predict(face_batch, verbose=0)
            
            # FIXED: Invert the class interpretation
            # If your model training had class 0 as "Fake" and class 1 as "Real"
            class_idx = np.argmax(prediction[0])
            is_deepfake = (class_idx == 0)  # Changed from 1 to 0 - assuming 0 is "Fake"
            confidence = float(prediction[0][class_idx])
            
            logger.info(f"Raw prediction: {prediction[0]}, argmax: {class_idx}")
            logger.info(f"Classes from training: {self.model.output_names}")
            return is_deepfake, confidence
        
        except Exception as e:
            logger.error(f"Error in predict_face: {str(e)}")
            raise

# Image API View
class DetectDeepfakeView(APIView):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.detector = DeepfakeDetector()
        
        # Warm up the model with a dummy prediction
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
                
                results.append({
                    'bbox': [int(x), int(y), int(w), int(h)],
                    'is_deepfake': is_deepfake,
                    'confidence': confidence
                })

            # Encode the processed main image
            _, buffer = cv2.imencode('.jpg', cv2_image)
            processed_image = base64.b64encode(buffer).decode('utf-8')
            
            # Don't try to encode the 'faces' coordinates array
            # Instead, create a montage of extracted face images if needed
            face_montage = None
            if face_images:
                # Create a simple montage or just use the first face
                face_montage = face_images[0]  # For simplicity, just using first face
                _, face_buffer = cv2.imencode('.jpg', face_montage)
                processed_faces = base64.b64encode(face_buffer).decode('utf-8')
            else:
                processed_faces = ""

            return Response({
                'processed_image': f'data:image/jpeg;base64,{processed_image}',
                'faces_detected': len(faces),
                'results': results,
                'faces': f'data:image/jpeg;base64,{processed_faces}' if face_images else ""
            }, status=status.HTTP_200_OK)

        except Exception as e:
            logger.error(f"Error details: {str(e)}")
            return Response({'error': str(e)}, status=status.HTTP_400_BAD_REQUEST)

# Video API View
class VideoDeepfakeDetectionView(APIView):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.detector = DeepfakeDetector()
        
        # Warm up the model with a dummy prediction
        dummy_input = np.zeros((1, 256, 256, 3), dtype=np.float32)
        _ = self.detector.model.predict(dummy_input)
        logger.info("Video detection model warmed up successfully")
    
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
            
            # Process and save the video with annotations
            processed_video = self.process_and_save_video(video_path)
            results['processed_video'] = f'data:video/mp4;base64,{processed_video}'
            
            # Clean up
            default_storage.delete(temp_path)
            
            return Response(results, status=status.HTTP_200_OK)
            
        except Exception as e:
            logger.error(f"Error in video processing: {str(e)}")
            return Response({'error': str(e)}, status=status.HTTP_400_BAD_REQUEST)
    
    def process_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise Exception("Error opening video file")
        
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = frame_count / fps if fps > 0 else 0
        
        # We don't want to process every frame - sample frames at regular intervals
        sample_interval = max(1, int(fps / 2))  # Process 2 frames per second
        
        frames_processed = 0
        frames_with_faces = 0
        deepfake_frames = 0
        frame_results = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Only process every nth frame
            if frames_processed % sample_interval == 0:
                # Convert frame to PIL Image for face detection
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(frame_rgb)
                
                # Detect faces
                faces, _ = self.detector.detect_faces(pil_image)
                
                if len(faces) > 0:
                    frames_with_faces += 1
                    frame_deepfakes = 0
                    
                    for (x, y, w, h) in faces:
                        face_img = frame[y:y+h, x:x+w]
                        face_img_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
                        
                        # Get prediction
                        is_deepfake, confidence = self.detector.predict_face(face_img_rgb, w, h)
                        
                        if is_deepfake:
                            frame_deepfakes += 1
                            deepfake_frames += 1
                    
                    # Record results for this frame
                    frame_results.append({
                        'frame_number': frames_processed,
                        'timestamp': frames_processed / fps if fps > 0 else 0,
                        'faces_detected': len(faces),
                        'deepfakes_detected': frame_deepfakes
                    })
            
            frames_processed += 1
        
        cap.release()
        
        # Calculate overall statistics
        if frames_with_faces > 0:
            deepfake_percentage = (deepfake_frames / frames_with_faces) * 100
        else:
            deepfake_percentage = 0
        
        # Generate final result
        final_result = {
            'video_info': {
                'total_frames': frame_count,
                'processed_frames': frames_processed,
                'duration_seconds': duration,
                'fps': fps
            },
            'detection_results': {
                'frames_with_faces': frames_with_faces,
                'frames_with_deepfakes': deepfake_frames,
                'deepfake_percentage': deepfake_percentage,
                'is_likely_deepfake': deepfake_percentage > 30,  # Threshold can be adjusted
                'confidence': min(100, deepfake_percentage * 1.5)  # Simple confidence calculation
            },
            'frame_by_frame': frame_results
        }
        
        return final_result
    
    def process_and_save_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise Exception("Error opening video file")
        
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Create a temporary file for the output video
        output_path = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False).name
        
        # Define the codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        sample_interval = max(1, int(fps / 2))  # Process 2 frames per second
        
        # To prevent processing all frames, set a maximum (e.g., 2 minutes of video)
        max_frames = min(int(frame_count), int(fps * 120))  # 120 seconds
        
        while frame_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
        
            # Create a copy of frame to draw on
            annotated_frame = frame.copy()
        
            # Only process every nth frame
            if frame_count % sample_interval == 0:
                # Convert frame to PIL Image for face detection
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(frame_rgb)
            
                # Detect faces
                faces, _ = self.detector.detect_faces(pil_image)
            
                for (x, y, w, h) in faces:
                    face_img = frame[y:y+h, x:x+w]
                    face_img_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
                
                    # Get prediction
                    is_deepfake, confidence = self.detector.predict_face(face_img_rgb, w, h)
                
                    # Draw rectangle and label
                    color = (0, 0, 255) if is_deepfake else (0, 255, 0)
                    cv2.rectangle(annotated_frame, (x, y), (x+w, y+h), color, 2)
                    label = f'{"Deepfake" if is_deepfake else "Real"} {confidence:.2%}'
                    cv2.putText(annotated_frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        
            # Write the frame
            out.write(annotated_frame)
            frame_count += 1
    
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