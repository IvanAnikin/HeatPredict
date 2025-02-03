import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import array_to_img

def preprocess_image(image_path, target_size=(128, 64)):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_png(img, channels=3)
    img = tf.image.resize(img, target_size)
    img = img / 255.0
    return img


def prepare_input_sequences(data_dir, sequence_length=3, target_size=(128, 64)):
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
    if min_length < sequence_length + 1:
        raise ValueError(
            f"Not enough images to form sequences. Minimum required: {sequence_length + 1}, but found: {min_length}"
        )

    inputs_evi, inputs_ndwi, inputs_lst = [], [], []
    for i in range(min_length - sequence_length):
        evi_sequence = [preprocess_image(evi_paths[j], target_size) for j in range(i, i + sequence_length)]
        ndwi_sequence = [preprocess_image(ndwi_paths[j], target_size) for j in range(i, i + sequence_length)]
        lst_sequence = [preprocess_image(lst_paths[j], target_size) for j in range(i, i + sequence_length)]

        inputs_evi.append(tf.stack(evi_sequence, axis=-1))
        inputs_ndwi.append(tf.stack(ndwi_sequence, axis=-1))
        inputs_lst.append(tf.stack(lst_sequence, axis=-1))

    return np.array(inputs_evi), np.array(inputs_ndwi), np.array(inputs_lst)


def predict_and_save(model_path, data_dir, output_dir, sequence_length=3, target_size=(128, 64)):
    model = load_model(model_path)
    inputs_evi, inputs_ndwi, inputs_lst = prepare_input_sequences(data_dir, sequence_length, target_size)
    predictions = model.predict([inputs_evi, inputs_ndwi, inputs_lst])
    os.makedirs(output_dir, exist_ok=True)
    for i, pred in enumerate(predictions):
        pred_img = array_to_img(pred)
        pred_img.save(os.path.join(output_dir, f"predicted_image_{i + 1}.png"))

if __name__ == "__main__":
    MODEL_PATH = "models/final/heat_map_model.keras" # .h5
    DATA_DIR = r"C:\Users\ivana\Downloads\Bakalarka\anime\urban_resilience\abudhabi\T39RZH\indicators"
    OUTPUT_DIR = "predicted_images"
    IMAGE_Y = 64      
    IMAGE_X = 64    
    SEQUENCE_LEN = 3
    target_size=(IMAGE_X, IMAGE_Y)

    predict_and_save(MODEL_PATH, DATA_DIR, OUTPUT_DIR, sequence_length=SEQUENCE_LEN, target_size=target_size)
