import tensorflow as tf
import os
from utils.data_loader import preprocess_image  

model = tf.keras.models.load_model("models/final/heat_map_model.h5")

input_data_path = "data/Prague/val"
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)


def load_input_images(input_path):
    input_images = []
    for img_name in sorted(os.listdir(input_path)):
        img_path = os.path.join(input_path, img_name)
        img = preprocess_image(img_path) 
        input_images.append(img)
    return tf.stack(input_images)

input_data = load_input_images(input_data_path)

predictions = model.predict(input_data)

for i, pred in enumerate(predictions):
    output_path = os.path.join(output_dir, f"heat_map_prediction_{i}.png")
    tf.keras.preprocessing.image.save_img(output_path, pred)
    print(f"Saved: {output_path}")
