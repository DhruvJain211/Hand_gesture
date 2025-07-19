import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

IMG_SIZE = 150
GESTURES = ['01_palm', '02_L', '03_fist', '04_fist_moved', '05_thumb', '06_index',
            '07_ok', '08_palm_moved', '09_c', '10_down']

def load_data(data_dir = r'C:\Users\dhruv\.cache\kagglehub\datasets\gti-upm\leapgestrecog\versions\1\leapGestRecog'):
    data = []
    labels = []
    
    # Loop through users: 00, 01, ..., 09
    for user_folder in os.listdir(data_dir):
        user_path = os.path.join(data_dir, user_folder)
        if not os.path.isdir(user_path):
            continue  # Skip files
        for gesture_id, gesture_name in enumerate(GESTURES):
            gesture_path = os.path.join(user_path, gesture_name)
            if not os.path.exists(gesture_path):
                print(f"⚠️ Missing: {gesture_path}")
                continue
            for img_name in os.listdir(gesture_path):
                img_path = os.path.join(gesture_path, img_name)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                data.append(img)
                labels.append(gesture_id)

    print(f"✅ Loaded {len(data)} images.")
    data = np.array(data).reshape(-1, IMG_SIZE, IMG_SIZE, 1) / 255.0
    labels = to_categorical(labels, num_classes=len(GESTURES))
    return train_test_split(data, labels, test_size=0.2, random_state=42)
