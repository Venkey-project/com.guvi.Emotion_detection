import streamlit as st
import torch.nn as nn
from PIL import Image
import torch
import torchvision.transforms as transforms
import mediapipe as mp
import numpy as np

class CNN_FER(nn.Module):
    def __init__(self, dropout_rate=0.3):
        super(CNN_FER, self).__init__()
        
        # Feature extraction blocks with residual connections
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout_rate)
        )
        
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout_rate)
        )
        
        self.block3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout_rate)
        )
        
        # Additional layer for more capacity
        self.block4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout_rate)
        )
        
        # Global average pooling to reduce parameters
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 7)
        )
        
    def forward(self,x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        
        # Global average pooling
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        return x

# load Trained model
model = CNN_FER()
model.load_state_dict(torch.load(r"..\Data\CNN_FER2013.pth"), strict=False)
model.eval()

# Image Transformation
transformation = transforms.Compose([
    transforms.Resize((48,48)),
    transforms.ToTensor(),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Emotion labels
labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Initialize Mediapipe Face Detection
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

def emotion_prediction(image):
    # converting image to rgb for mediapipe
    image_rgb = np.array(image.convert("RGB"))

    # Use Mediapipe to detect the face
    with mp_face_detection.FaceDetection(min_detection_confidence=0.2) as face_detection:
        results = face_detection.process(image_rgb)
        
        if results.detections:
            # Get the bounding box for the face
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = image_rgb.shape
                x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
                
                # Crop the image to the detected face
                face_crop = image.crop((x, y, x + w, y + h))
                
                # Perform prediction on the cropped face image
                face_crop = transformation(face_crop).unsqueeze(0)
                output = model(face_crop)
                _, predicted = torch.max(output, 1)
                
                # Return the predicted emotion
                return labels[predicted.item()]
        else:
            return "No face detected"
def main():
    st.title("Emotion Detection from Image or Webcam")

    st.write("Choose an option to detect the emotion from image or webcam.")

    
    choice = st.radio("Select input type", ("Upload Image", "Use Webcam"))

    if choice == "Upload Image":
        uploaded_image = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
        
        if uploaded_image is not None:
           
            image = Image.open(uploaded_image)
            st.image(image, caption="Uploaded Image", use_container_width=True)
            
           
            emotion = emotion_prediction(image)
            st.write(f"Predicted Emotion: {emotion}")
        else:
            st.write("No image uploaded yet.")
    
    elif choice == "Use Webcam":
        
        captured_image = st.camera_input("Capture Image")
        
        if captured_image is not None:
            
            image = Image.open(captured_image)
            st.image(image, caption="Captured Image", use_container_width=True)
            
            
            emotion = emotion_prediction(image)
            st.write(f"Predicted Emotion: {emotion}")
        else:
            st.write("Waiting for camera input...")
if __name__ == '__main__':
    main()