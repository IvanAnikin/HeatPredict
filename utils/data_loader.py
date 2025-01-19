import tensorflow as tf
import os

def preprocess_image(image_path, target_size=(1299, 636)):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_png(img, channels=3)
    img = tf.image.resize(img, target_size)  
    img = img / 255.0  
    return img

def load_data(train_dir, val_dir, batch_size=32, sequence_length=3):
    def load_dataset(data_dir):
        evi_dir = os.path.join(data_dir, "EVI")
        ndwi_dir = os.path.join(data_dir, "NDWI")
        lst_dir = os.path.join(data_dir, "LST")
        out_dir = os.path.join(data_dir, "LST")

        evi_paths = sorted([os.path.join(evi_dir, img) for img in os.listdir(evi_dir)])
        ndwi_paths = sorted([os.path.join(ndwi_dir, img) for img in os.listdir(ndwi_dir)])
        lst_paths = sorted([os.path.join(lst_dir, img) for img in os.listdir(lst_dir)])
        out_paths = sorted([os.path.join(out_dir, img) for img in os.listdir(out_dir)])

        inputs, outputs = [], []
        for i in range(len(evi_paths) - sequence_length):
            evi_sequence = [preprocess_image(evi_paths[j]) for j in range(i, i + sequence_length)]
            ndwi_sequence = [preprocess_image(ndwi_paths[j]) for j in range(i, i + sequence_length)]
            lst_sequence = [preprocess_image(lst_paths[j]) for j in range(i, i + sequence_length)]

            input_tensor = tf.concat(evi_sequence + ndwi_sequence + lst_sequence, axis=-1)
            inputs.append(input_tensor)

            output_tensor = preprocess_image(out_paths[i + sequence_length])
            outputs.append(output_tensor)

        dataset = tf.data.Dataset.from_tensor_slices((inputs, outputs))
        dataset = dataset.batch(batch_size)
        return dataset

    train_ds = load_dataset(train_dir)
    val_ds = load_dataset(val_dir)
    return train_ds, val_ds
