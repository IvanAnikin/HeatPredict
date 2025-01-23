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
    evi_dir = os.path.join(data_dir, "EVI")
    ndwi_dir = os.path.join(data_dir, "NDWI")
    lst_dir = os.path.join(data_dir, "LST")

    evi_paths = sorted([os.path.join(evi_dir, img) for img in os.listdir(evi_dir)])
    ndwi_paths = sorted([os.path.join(ndwi_dir, img) for img in os.listdir(ndwi_dir)])
    lst_paths = sorted([os.path.join(lst_dir, img) for img in os.listdir(lst_dir)])

    inputs_evi, inputs_ndwi, inputs_lst = [], [], []
    for i in range(len(evi_paths) - sequence_length):
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
    MODEL_PATH = "models/final/heat_map_model.keras"
    DATA_DIR = "data/Prague/train"
    OUTPUT_DIR = "predicted_images"
    predict_and_save(MODEL_PATH, DATA_DIR, OUTPUT_DIR, sequence_length=3, target_size=(128, 64))
