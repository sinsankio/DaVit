import os
import uuid
import base64
from datetime import datetime

import cv2
import numpy as np
import pandas as pd
import torch
from moviepy import ImageSequenceClip
from transformers import ViTFeatureExtractor, ViTModel


MODEL_ID = 'google/vit-base-patch16-224'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = None
feature_extractor = None

# data preprocessing functions
def decode_base64_image(base64_string):
    try:
        if 'base64' in base64_string:
            base64_string = base64_string.split(',')[1]

        img_bytes = base64.b64decode(base64_string)
        np_arr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        
        return img
    except Exception as e:
        return None

def preprocess_image(image):
    if image is None:
        return None
    
    image = image[60:135, :, :]
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224))

    return image

# model utility functions
def load_inference_engine(
        chkpt_path: str = os.getcwd() + '/app/chkpts/vit_v1.pth'
    ):
    global model, feature_extractor

    if model and feature_extractor:
        return model

    class ViTForDave2(torch.nn.Module):
        def __init__(self, vit_model):
            super(ViTForDave2, self).__init__()
            self.vit = vit_model
            
            self.fc1 = torch.nn.Linear(self.vit.config.hidden_size, 128)
            self.do1 = torch.nn.Dropout(0.25)
            self.ac1 = torch.nn.ELU()

            self.fc2 = torch.nn.Linear(128, 64)
            self.do2 = torch.nn.Dropout(0.25)
            self.ac2 = torch.nn.ELU()

            self.fc3 = torch.nn.Linear(64, 32)
            self.do3 = torch.nn.Dropout(0.25)
            self.ac3 = torch.nn.ELU()

            self.regressor = torch.nn.Linear(32, 1)
            self.tanh = torch.nn.Tanh()
            
        def forward(self, x):
            outputs = self.vit(pixel_values=x)
            pooled_output = outputs.pooler_output
            x = self.fc1(pooled_output)
            x = self.do1(x)
            x = self.ac1(x)

            x = self.fc2(x)
            x = self.do2(x)
            x = self.ac2(x)

            x = self.fc3(x)
            x = self.do3(x)
            x = self.ac3(x)

            return self.regressor(x).squeeze(-1)
    
    if not model:
        vit = ViTModel.from_pretrained(MODEL_ID, device_map='auto')
        model = ViTForDave2(vit)
        state_dict = torch.load(chkpt_path)
        model.load_state_dict(state_dict)
        model = model.to(DEVICE)
    if not feature_extractor:
        feature_extractor = ViTFeatureExtractor.from_pretrained(MODEL_ID)

    return model, feature_extractor

def make_inference(image):
    global model, feature_extractor

    if not model or not feature_extractor:
        model, feature_extractor = load_inference_engine()
    
    model.eval()
    with torch.no_grad():
        inputs = feature_extractor(images=image, return_tensors="pt").to(DEVICE)
        prediction = model(inputs['pixel_values'])
        steering_angle = prediction.squeeze().item()
    
    steering_angle = max(-1.0, min(1.0, steering_angle))

    return steering_angle  


# post-processing functions
def calculate_throttle(steering_angle, current_speed, speed_limit, throttle_factor = 0.8):
    try:
        throttle = throttle_factor - abs(steering_angle) * 0.5 - (current_speed / speed_limit) ** 2
        throttle = max(0.0, min(1.0, throttle))

        return throttle
    except Exception as e:
        return 0.5 # safe default 

def smooth_steering(steering_angle, previous_steering_angle, smoothing_factor=0.8):
    try:
        if previous_steering_angle is None:
            return steering_angle
        
        smoothed_angle = smoothing_factor * previous_steering_angle + (1 - smoothing_factor) * steering_angle
    
        return smoothed_angle
    except Exception as e:  
        return steering_angle


# post saving for further verification and validation
def save_telemetry(telemetry, save_dir, img_id):
    if os.path.exists(save_dir) is False:
        os.makedirs(save_dir)
    if os.path.exists(save_dir + '/IMG') is False:
        os.makedirs(save_dir + '/IMG')
    
    file_save_path = save_dir + f'/IMG/{img_id}.jpg'
    cv2.imwrite(file_save_path, telemetry['image'])

    return file_save_path

def record_telemetry_results(results, save_dir = os.getcwd() + '/dev/data/track-01-results/'):
    save_dir = save_dir + f'phase-{str(uuid.uuid4())[:8]}'
    if os.path.exists(save_dir) is False:
        os.makedirs(save_dir)

    for idx, record in enumerate(results):
        telemetry_file_path = save_telemetry(record['telemetry'], save_dir, img_id=str('snap-' + str(idx).zfill(5)))
        record = {
            'file_path': telemetry_file_path,
            'timestamp': record['telemetry']['timestamp'],
            'steering_angle': record['steering_angle'],
            'throttle': record['throttle'],
            'speed': record['speed']
        }
        
        csv_path = save_dir + '/driving_log.csv'
        if os.path.exists(csv_path) is False:
            df = pd.DataFrame(data=[record])
        else:
            df = pd.read_csv(csv_path)
        df = pd.concat([df, pd.DataFrame([record])], ignore_index=True)
        df.to_csv(csv_path, index=False)    
    
    return save_dir

def create_driving_video(save_dir = os.getcwd() + '/dev/data/track-01-results/phase-01', fps=30):
    image_dir = save_dir + '/IMG/'

    # convert file folder into list filtered for image file types
    image_list = sorted([os.path.join(image_dir, image_file)
                        for image_file in os.listdir(image_dir)])

    image_list = [image_file for image_file in image_list if os.path.splitext(image_file)[1][1:].lower() in ['jpeg', 'gif', 'png', 'jpg']]

    # two methods of naming output video to handle varying environments
    video_path = save_dir + '/video/'
    if os.path.exists(video_path) is False:
        os.makedirs(video_path)

    video_path = video_path + f'output_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.mp4'
    clip = ImageSequenceClip(image_list, fps=fps)
    clip.write_videofile(video_path)
