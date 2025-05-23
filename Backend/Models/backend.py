import random
import cv2
from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import librosa
import torch
from typing import List, Tuple
from PIL import Image
import torchvision
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Set this at the beginning to avoid thread issues
import matplotlib.pyplot as plt
from torchvision import transforms
from tensorflow.keras.models import load_model
from transformers import (
    AutoImageProcessor, 
    AutoModelForImageClassification,
    AutoImageProcessor as VideoProcessor,
    AutoModelForVideoClassification
)
import pandas as pd
from werkzeug.security import generate_password_hash, check_password_hash

# Initialize the models at startup
device = torch.device('cpu')
# Image deepfake detection model initialization - UPDATED
vit_processor = AutoImageProcessor.from_pretrained("dima806/deepfake_vs_real_image_detection")
vit_model = AutoModelForImageClassification.from_pretrained("dima806/deepfake_vs_real_image_detection")
vit_model.to(device)

# Video deepfake detection model initialization
video_processor = VideoProcessor.from_pretrained("Ammar2k/videomae-base-finetuned-deepfake-subset")
video_model = AutoModelForVideoClassification.from_pretrained("Ammar2k/videomae-base-finetuned-deepfake-subset")
video_model.to(device)

# Path to Excel file for user authentication
EXCEL_FILE_PATH = 'users.xlsx'

# Create Excel file if it doesn't exist
def init_excel_file():
    if not os.path.exists(EXCEL_FILE_PATH):
        df = pd.DataFrame(columns=['name', 'username', 'email', 'password'])
        df.to_excel(EXCEL_FILE_PATH, index=False)
        print('Excel file created successfully')

def format_frames(frame, output_size):
    """
    Pad and resize an image from a video.

    Args:
        frame: Image that needs to resized and padded. 
        output_size: Pixel size of the output frame image.

    Return:
        Formatted frame with padding of specified output size.
    """
    frame = cv2.resize(frame, output_size) 
    return frame

def frames_from_video_file(video_path, n_frames, output_size=(224, 224), frame_step=15):
    """
    Creates frames from each video file present for each category.

    Args:
        video_path: File path to the video.
        n_frames: Number of frames to be created per video file.
        output_size: Pixel size of the output frame image.

    Return:
        An NumPy array of frames in the shape of (n_frames, height, width, channels).
    """
    # Read each video frame by frame
    result = []
    src = cv2.VideoCapture(str(video_path))  

    video_length = src.get(cv2.CAP_PROP_FRAME_COUNT)

    need_length = 1 + (n_frames - 1) * frame_step

    if need_length > video_length:
        start = 0
    else:
        max_start = video_length - need_length
        start = random.randint(0, max_start + 1)

    src.set(cv2.CAP_PROP_POS_FRAMES, start)
    ret, frame = src.read()
    if ret:
        result.append(format_frames(frame, output_size))

        for _ in range(n_frames - 1):
            for _ in range(frame_step):
                ret, frame = src.read()
            if ret:
                frame = format_frames(frame, output_size)
                result.append(frame)
            else:
                result.append(np.zeros_like(result[0]))
    src.release()
    result = np.array(result)

    return result

def extract_video_frames_for_videomae(video_path, num_frames=16):
    """
    Extract frames from a video file in the format expected by VideoMAE.
    
    Args:
        video_path: Path to the video file
        num_frames: Number of frames to extract (VideoMAE typically uses 16)
        
    Returns:
        List of PIL Image objects
    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames <= num_frames:
        # If video has fewer frames than required, duplicate some frames
        indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    else:
        # Sample frames evenly from the video
        indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    
    for i in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if ret:
            # Convert BGR (OpenCV format) to RGB (PIL format)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(frame_rgb)
            frames.append(pil_img)
        else:
            # If frame extraction fails, add a black frame
            if frames:
                frames.append(Image.fromarray(np.zeros_like(np.array(frames[0]))))
            else:
                frames.append(Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8)))
    
    cap.release()
    return frames

def predict_video_deepfake(video_path):
    """
    Predict whether a video is real or fake using the VideoMAE model
    
    Args:
        video_path: Path to the video file
        
    Returns:
        Tuple of (prediction label, confidence score)
    """
    # Extract frames from video
    frames = extract_video_frames_for_videomae(video_path)
    
    # Prepare inputs for the model
    inputs = video_processor(images=frames, return_tensors="pt").to(device)
    
    # Make prediction
    with torch.no_grad():
        outputs = video_model(**inputs)
    
    logits = outputs.logits
    probabilities = torch.softmax(logits, dim=1)
    
    # Get prediction
    pred_class_idx = torch.argmax(probabilities, dim=1).item()
    confidence = probabilities.max().item()
    
    # Map to appropriate label (assuming model's labels match our needs)
    # Check model's id2label mapping
    if hasattr(video_model.config, 'id2label'):
        pred_label = video_model.config.id2label[pred_class_idx].lower()
        # Make sure the label is either 'fake' or 'real'
        if 'fake' in pred_label:
            return ("fake", confidence)
        else:
            return ("real", confidence)
    else:
        # Fallback if id2label is not available
        class_names = ["fake", "real"]
        return (class_names[pred_class_idx % len(class_names)], confidence)

def pred_with_vit(image_path):
    """
    Predict using the specialized deepfake detection model from Hugging Face
    """
    img = Image.open(image_path)
    inputs = vit_processor(images=img, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = vit_model(**inputs)
    
    logits = outputs.logits
    probabilities = torch.softmax(logits, dim=1)
    
    # Get the predicted class
    pred_class_idx = torch.argmax(probabilities, dim=1).item()
    confidence = probabilities.max().item()
    
    # Get the label from the model's config
    if hasattr(vit_model.config, 'id2label'):
        pred_label = vit_model.config.id2label[pred_class_idx].lower()
        # Normalize to either 'fake' or 'real'
        if 'fake' in pred_label or 'deepfake' in pred_label:
            return ("fake", confidence)
        else:
            return ("real", confidence)
    else:
        # Fallback if id2label is not available
        class_names = ["fake", "real"]
        return (class_names[pred_class_idx % len(class_names)], confidence)

def predictFake(path):
    m,_=librosa.load(path,sr=16000)
    max_length=500
    mfccs = librosa.feature.mfcc(y=m, sr=16000, n_mfcc=40)

    if mfccs.shape[1] < max_length:
        mfccs = np.pad(mfccs, ((0, 0), (0, max_length - mfccs.shape[1])), mode='constant')
    else:
        mfccs = mfccs[:, :max_length]
    
    model=load_model('C:\\Users\\smdar\\Desktop\\Anokha project\\AudioModel.h5')
    output=model.predict(mfccs.reshape(-1,40,500))
    if output[0][0]>0.5:
        return "fake"
    else:
        return "real"

def save_images(path):
    paths = []
    video_frames = frames_from_video_file(path, 3)
    
    if len(video_frames) == 0:
        return paths
    
    for i in range(min(3, len(video_frames))):
        image_3d = video_frames[i]
        if image_3d.shape[2] == 4:  # Convert RGBA to RGB if needed
            image_3d = image_3d[:, :, :3]
        
        # Create a new figure for each image
        plt.figure(figsize=(1, 1))
        plt.imshow(image_3d)
        plt.axis('off')
        
        # Save the figure
        save_path = f"uploads/image{i}.jpg"
        paths.append(save_path)
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
        plt.close()
        
    return paths

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "http://localhost:5173"}})  # Updated CORS to allow all routes

UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
from collections import Counter

def find_mode(arr):
    if not arr:
        return "unknown"
    counts = Counter(arr)
    max_count = max(counts.values())
    mode = next(key for key, value in counts.items() if value == max_count)
    return mode

# Initialize the Excel file for user data
init_excel_file()

@app.route('/api/signup', methods=['POST'])
def signup():
    try:
        data = request.json
        name = data.get('name')
        username = data.get('username')
        email = data.get('email')
        password = data.get('password')
        
        # Hash password for security
        hashed_password = generate_password_hash(password)
        
        # Load Excel file
        if os.path.exists(EXCEL_FILE_PATH):
            df = pd.read_excel(EXCEL_FILE_PATH)
        else:
            df = pd.DataFrame(columns=['name', 'username', 'email', 'password'])
        
        # Check if username or email already exists
        if username in df['username'].values or email in df['email'].values:
            return jsonify({'message': 'Username or email already exists'}), 400
        
        # Add new user
        new_user = pd.DataFrame({
            'name': [name],
            'username': [username],
            'email': [email],
            'password': [hashed_password]
        })
        
        df = pd.concat([df, new_user], ignore_index=True)
        
        # Save Excel file
        df.to_excel(EXCEL_FILE_PATH, index=False)
        
        return jsonify({'message': 'User registered successfully'}), 201
    
    except Exception as e:
        print('Error registering user:', e)
        return jsonify({'message': 'Server error'}), 500

@app.route('/api/signin', methods=['POST'])
def signin():
    try:
        data = request.json
        username = data.get('username')
        password = data.get('password')
        
        # Load Excel file
        if not os.path.exists(EXCEL_FILE_PATH):
            return jsonify({'message': 'Invalid credentials'}), 401
        
        df = pd.read_excel(EXCEL_FILE_PATH)
        
        # Check if user exists
        user_row = df[df['username'] == username]
        if user_row.empty:
            return jsonify({'message': 'Invalid credentials'}), 401
        
        stored_password = user_row['password'].values[0]
        
        # Check password
        if check_password_hash(stored_password, password):
            user = {
                'name': user_row['name'].values[0],
                'username': user_row['username'].values[0],
                'email': user_row['email'].values[0]
            }
            return jsonify({
                'message': 'Login successful',
                'user': user
            }), 200
        else:
            return jsonify({'message': 'Invalid credentials'}), 401
    
    except Exception as e:
        print('Error during login:', e)
        return jsonify({'message': 'Server error'}), 500

@app.route('/upload', methods=['POST'])
def upload_file():
    print(request.files)
    if 'image' in request.files:
        file = request.files['image']
        filename = file.filename
        if file.filename == '':
            return jsonify({'error': 'No selected file'})
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        # Use the specialized deepfake detection model
        a = pred_with_vit(image_path=file_path)
        
        os.remove(file_path)
        return jsonify([{'message': 'File uploaded successfully', 'file_path': file_path}, a[0]])
    
    if 'audio' in request.files:
        file = request.files['audio']
        filename = file.filename
        if file.filename == '':
            return jsonify({'error': 'No selected file'})
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        ans = predictFake(file_path)
        print(ans)
        return jsonify([{'message': 'File uploaded successfully'}, ans])
    
    if 'video' in request.files:
        file = request.files['video']
        filename = file.filename
        if file.filename == '':
            return jsonify({'error': 'No selected file'})
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Use the VideoMAE model for video deepfake detection
        pred_result, confidence = predict_video_deepfake(file_path)
        print(f"Video prediction: {pred_result} with confidence {confidence:.4f}")
        
        os.remove(file_path)
        return jsonify([{'message': 'File uploaded successfully', 'file_path': file_path}, pred_result])

if __name__ == '__main__': 
    app.run(debug=True, port=5000)
