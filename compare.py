import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import os
import json
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import array_to_img

from utils.data_loader import load_data, preprocess_image, apply_colormap


def predict_and_display(model_path, data_dir, sequence_length=3, sequence_step=1, future_step=1, target_size=(128, 128)):
    model = load_model(model_path)
    
    train_data, _ = load_data(data_dir, batch_size=1, sequence_length=sequence_length, sequence_step=sequence_step, future_step=future_step, target_size=target_size)
    
    for inputs, actual_output in train_data:
        predicted_output = model.predict(inputs)

        pred_img = array_to_img(predicted_output[0])

        predicted_output_colored = apply_colormap(pred_img)
        actual_output_colored = apply_colormap(array_to_img(actual_output[0]))

        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(predicted_output_colored)
        axes[0].set_title('Predicted Image')
        axes[0].axis('off')

        axes[1].imshow(actual_output_colored)
        axes[1].set_title('Actual Image')
        axes[1].axis('off')

        plt.show()

if __name__ == "__main__":
    with open('config.json', 'r') as file:
        config = json.load(file)

    MODEL_PATH = config['MODEL_PATH']
    DATA_DIR = config['DATA_DIR']
    IMAGE_Y = config['IMAGE_Y']
    IMAGE_X = config['IMAGE_X']
    SEQUENCE_LEN = config['SEQUENCE_LEN']
    SEQUENCE_STEP = config['SEQUENCE_STEP']
    FUTURE_STEP = config['FUTURE_STEP']
    
    target_size = (IMAGE_X, IMAGE_Y)

    predict_and_display(MODEL_PATH, DATA_DIR, sequence_length=SEQUENCE_LEN, sequence_step=SEQUENCE_STEP, future_step=FUTURE_STEP, target_size=target_size)
