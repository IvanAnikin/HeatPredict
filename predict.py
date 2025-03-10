import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import array_to_img

from utils.data_loader import preprocess_image, apply_colormap



def prepare_input_sequences(data_dir, sequence_length=3, sequence_step=1, target_size=(128, 128)):
    def get_sorted_image_paths(indicator_dir):
        png_dir = os.path.join(indicator_dir, "png")
        year_dirs = sorted(os.listdir(png_dir))
        image_paths = []
        for year in year_dirs:
            year_path = os.path.join(png_dir, year)
            if os.path.isdir(year_path):
                year_images = sorted(
                    [os.path.join(year_path, img) for img in os.listdir(year_path) if img.endswith(('.png', '.jpg'))]
                )
                image_paths.extend(year_images)
        return image_paths

    evi_dir = os.path.join(data_dir, "evi")
    ndwi_dir = os.path.join(data_dir, "ndwi")
    lst_dir = os.path.join(data_dir, "lst")

    evi_paths = get_sorted_image_paths(evi_dir)
    ndwi_paths = get_sorted_image_paths(ndwi_dir)
    lst_paths = get_sorted_image_paths(lst_dir)

    min_length = min(len(evi_paths), len(ndwi_paths), len(lst_paths))
    total_samples = min_length - (sequence_length - 1) * sequence_step

    if total_samples < 1:
        raise ValueError(
            f"Not enough images to form sequences. Required at least {sequence_length * sequence_step}, found: {min_length}"
        )

    input_sequences = []

    for i in range(total_samples):
        evi_sequence = [preprocess_image(evi_paths[j], target_size) for j in range(i, i + sequence_length * sequence_step, sequence_step)]
        ndwi_sequence = [preprocess_image(ndwi_paths[j], target_size) for j in range(i, i + sequence_length * sequence_step, sequence_step)]
        lst_sequence = [preprocess_image(lst_paths[j], target_size) for j in range(i, i + sequence_length * sequence_step, sequence_step)]

        evi_stack = tf.concat(evi_sequence, axis=-1)  
        ndwi_stack = tf.concat(ndwi_sequence, axis=-1)
        lst_stack = tf.concat(lst_sequence, axis=-1)

        input_tensor = tf.concat([evi_stack, ndwi_stack, lst_stack], axis=-1) 
        input_sequences.append(input_tensor)

    return np.array(input_sequences)


def predict_and_save(model_path, data_dir, output_dir, sequence_length=3, sequence_step=1, target_size=(128, 128)):
    model = load_model(model_path)
    input_sequences = prepare_input_sequences(data_dir, sequence_length, sequence_step, target_size)
    
    predictions = model.predict(input_sequences)
    
    os.makedirs(output_dir, exist_ok=True)
    for i, pred in enumerate(predictions):
        pred_img = array_to_img(pred)
        pred_img = apply_colormap(pred_img)
        pred_img = array_to_img(pred_img)
        pred_img.save(os.path.join(output_dir, f"predicted_image_{i + 1}.png"))

if __name__ == "__main__":
    MODEL_PATH = "models/final/heat_map_model.keras" 
    DATA_DIR = r"C:\Users\ivana\Downloads\Bakalarka\anime\urban_resilience\abudhabi\T39RZH\indicators"
    OUTPUT_DIR = "predicted_images"
    IMAGE_Y = 128
    IMAGE_X = 128
    SEQUENCE_LEN = 4
    SEQUENCE_STEP = 2
    target_size = (IMAGE_X, IMAGE_Y)

    predict_and_save(MODEL_PATH, DATA_DIR, OUTPUT_DIR, sequence_length=SEQUENCE_LEN, sequence_step=SEQUENCE_STEP, target_size=target_size)
