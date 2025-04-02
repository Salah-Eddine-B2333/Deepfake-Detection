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
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info("Face image processed successfully.")

class DeepfakeDetector:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        model_path = os.path.join('detector', 'ml_models', 'deepfake_model_ResNet50.h5')
        # Load ESRGAN model from TensorFlow Hub
        esrgan_path = "https://tfhub.dev/captain-pool/esrgan-tf2/1"
        
        try:
            self.model = tf.keras.models.load_model(model_path)  
            self.upscal_model = hub.load(esrgan_path)
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            self.model = None
    def detect_faces(self, image):
        cv2_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2GRAY)
        # Adjust scaleFactor and minNeighbors for better face detection
        faces = self.face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.3,  # Slightly increased 
            minNeighbors=5,   # More strict neighbor requirements
            minSize=(30, 30)  # Minimum face size to detect
        )
        return faces, cv2_image

#Super Resolution Scale model------------------------------------------------------------------------
    

    def preprocessing(self , img):
        # Ensure image has valid dimensions
        if img.shape[0] < 4 or img.shape[1] < 4:
            raise ValueError("Image dimensions must be at least 4x4 pixels.")
        
        imageSize = (tf.convert_to_tensor(img.shape[:-1]) // 4) * 4
        cropped_image = tf.image.crop_to_bounding_box(
            img, 0, 0, imageSize[0], imageSize[1])
        preprocessed_image = tf.cast(cropped_image, tf.float32)
        return tf.expand_dims(preprocessed_image, 0)

    def srmodel(self , img):
        preprocessed_image = self.preprocessing(img)  
        new_image = self.upscal_model(preprocessed_image)  
        return tf.squeeze(new_image) 
    
#Predict the face--------------------------------------------------------------------------------------
    def predict_face(self, face_image):
        try:
            if face_image is None or face_image.size == 0:
                raise ValueError("Invalid input image.")
            
            dimension = (256, 256)
            
            # Resize the face image to the expected input size
            face_image = cv2.resize(face_image, dimension)
            
            # Convert the image to RGB (if it's in BGR format)
            face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
            
            # Normalize the image (scale pixel values to [0, 1])
            face_image = face_image / 255.0
            
            # Add the batch dimension
            face_image = np.expand_dims(face_image, axis=0)
            
            # Get prediction (outputs 2 classes)
            prediction = self.model.predict(face_image, verbose=0)
            
            # Return probability of deepfake (second class)
            return float(prediction[0][1])  # Ensure float conversion
        
        except Exception as e:
            logger.error(f"Error in predict_face: {str(e)}")
            raise

class DetectDeepfakeView(APIView):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.detector = DeepfakeDetector()

    def post(self, request):
        try:
            if self.detector.model is None:
                return Response({'error': 'Deepfake detection model not loaded'}, 
                                status=status.HTTP_500_INTERNAL_SERVER_ERROR)
            
            image_data = request.data.get('image', '').split(',')[1]
            image_bytes = base64.b64decode(image_data)
            image = Image.open(BytesIO(image_bytes))
            
            faces, cv2_image = self.detector.detect_faces(image)
            
            if len(faces) == 0:
                return Response({
                    'message': 'No faces detected',
                    'faces_detected': 0
                }, status=status.HTTP_200_OK)

            results = []
            for (x, y, w, h) in faces:
                face_img = cv2_image[y:y+h, x:x+w]
                face_img = cv2.cvtColor(np.array(face_img), cv2.COLOR_BGR2RGB)
                prediction = self.detector.predict_face(face_img)
                
                is_deepfake = prediction > 0.5
                confidence = float(prediction if is_deepfake else 1 - prediction)
                
                color = (0, 0, 255) if is_deepfake else (0, 255, 0)
                cv2.rectangle(cv2_image, (x, y), (x+w, y+h), color, 2)
                label = f'{"Deepfake" if is_deepfake else "Real"} {confidence:.2%}'
                cv2.putText(cv2_image, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                
                results.append({
                    'bbox': [int(x), int(y), int(w), int(h)],
                    'is_deepfake': is_deepfake,
                    'confidence': confidence
                    
                })

            _, buffer = cv2.imencode('.jpg', cv2_image)
            processed_image = base64.b64encode(buffer).decode('utf-8')
            """this is for the faces images"""
            _, buffer = cv2.imencode('.jpg', faces)
            processed_faces = base64.b64encode(buffer).decode('utf-8')

            return Response({
                'processed_image': f'data:image/jpeg;base64,{processed_image}',
                'faces_detected': len(faces),
                'results': results,
                'faces': f'data:image/jpeg;base64,{processed_faces}'
            }, status=status.HTTP_200_OK)

        except Exception as e:
            print(f"Error details: {str(e)}")  # Print specific error
            return Response({'error': str(e)}, status=status.HTTP_400_BAD_REQUEST)