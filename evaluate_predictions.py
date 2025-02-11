import tensorflow as tf
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import array_to_img

# Load model
def load_trained_model(model_path):
    return load_model(model_path)

# Load test dataset
def load_test_data(data_dir, sequence_length=3, target_size=(64, 64)):
    def preprocess_image(image_path):
        img = tf.io.read_file(image_path)
        img = tf.image.decode_png(img, channels=3)
        img = tf.image.resize(img, target_size)
        img = img / 255.0
        return img

    def get_sorted_image_paths(indicator_dir):
        png_dir = os.path.join(indicator_dir, "png")
        if not os.path.exists(png_dir):
            return []
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
    out_dir = os.path.join(data_dir, "lst")  # Output should match the target

    evi_paths = get_sorted_image_paths(evi_dir)
    ndwi_paths = get_sorted_image_paths(ndwi_dir)
    lst_paths = get_sorted_image_paths(lst_dir)
    output_paths = get_sorted_image_paths(out_dir)

    min_length = min(len(evi_paths), len(ndwi_paths), len(lst_paths))
    if min_length < sequence_length + 1:
        raise ValueError(f"Not enough images to form sequences. Found: {min_length}, Required: {sequence_length + 1}")

    inputs_evi, inputs_ndwi, inputs_lst, outputs = [], [], [], []
    for i in range(min_length - sequence_length):
        evi_sequence = [preprocess_image(evi_paths[j]) for j in range(i, i + sequence_length)]
        ndwi_sequence = [preprocess_image(ndwi_paths[j]) for j in range(i, i + sequence_length)]
        lst_sequence = [preprocess_image(lst_paths[j]) for j in range(i, i + sequence_length)]
        output_tensor = preprocess_image(output_paths[i + sequence_length])  # Future frame

        inputs_evi.append(tf.stack(evi_sequence, axis=-1))
        inputs_ndwi.append(tf.stack(ndwi_sequence, axis=-1))
        inputs_lst.append(tf.stack(lst_sequence, axis=-1))
        outputs.append(output_tensor)

    return np.array(inputs_evi), np.array(inputs_ndwi), np.array(inputs_lst), np.array(outputs)


# Evaluate the model and compute accuracy metrics
def evaluate_model(model_path, data_dir, sequence_length=3, target_size=(64, 64)):
    print("Loading model...")
    model = load_trained_model(model_path)

    print("Loading test data...")
    inputs_evi, inputs_ndwi, inputs_lst, actual_outputs = load_test_data(data_dir, sequence_length, target_size)

    print("Making predictions...")
    predictions = model.predict([inputs_evi, inputs_ndwi, inputs_lst])

    # Calculate evaluation metrics
    mse = np.mean((predictions - actual_outputs) ** 2)
    mae = np.mean(np.abs(predictions - actual_outputs))

    print(f"Model Evaluation Results:")
    print(f"Mean Squared Error (MSE): {mse:.6f}")
    print(f"Mean Absolute Error (MAE): {mae:.6f}")

    return predictions, actual_outputs, mse, mae


# Compare predicted and actual images visually
def plot_predictions(predictions, actual_outputs, num_images=5, save_path="prediction_comparison.png"):
    fig, axes = plt.subplots(num_images, 2, figsize=(10, num_images * 3))

    for i in range(num_images):
        pred_img = array_to_img(predictions[i])
        actual_img = array_to_img(actual_outputs[i])

        axes[i, 0].imshow(pred_img)
        axes[i, 0].set_title(f"Predicted Image {i+1}")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(actual_img)
        axes[i, 1].set_title(f"Actual Image {i+1}")
        axes[i, 1].axis("off")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()
    print(f"Comparison images saved as {save_path}")


if __name__ == "__main__":
    MODEL_PATH = "models/final/heat_map_model.keras"
    DATA_DIR = r"C:\Users\ivana\Downloads\Bakalarka\anime\urban_resilience\abudhabi\T39RZH\indicators"
    IMAGE_Y = 64
    IMAGE_X = 64
    SEQUENCE_LEN = 3
    TARGET_SIZE = (IMAGE_X, IMAGE_Y)

    predictions, actual_outputs, mse, mae = evaluate_model(MODEL_PATH, DATA_DIR, sequence_length=SEQUENCE_LEN, target_size=TARGET_SIZE)
    plot_predictions(predictions, actual_outputs, num_images=5)