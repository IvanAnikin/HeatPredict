import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import array_to_img

def preprocess_image(image_path, target_size):
    OUTPUT_DIR = "test_images"
    img = tf.io.read_file(image_path)
    img = tf.image.decode_png(img, channels=3)
    img = tf.image.resize(img, target_size)
    img = img / 255.0

    pred_img = array_to_img(img)
    pred_img.save(os.path.join(OUTPUT_DIR, f"predicted_image.png"))

if __name__ == "__main__":
    
    DATA_DIR = "data/Prague/train"
    IMAGE_X = 64      
    IMAGE_Y = 128    
    SEQUENCE_LEN = 3
    target_size=(IMAGE_X, IMAGE_Y)

    lst_dir = os.path.join(DATA_DIR, "LST")
    lst_paths = sorted([os.path.join(lst_dir, img) for img in os.listdir(lst_dir)])
    preprocess_image(lst_paths[0], target_size)
        

