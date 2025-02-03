import tensorflow as tf

import os

def preprocess_image(image_path, target_size=(1299, 636)):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_png(img, channels=3)
    img = tf.image.resize(img, target_size)  
    img = img / 255.0  
    return img

def load_data1(train_dir, val_dir, batch_size=32, sequence_length=3):
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



def load_data(train_dir, val_dir, batch_size=32, sequence_length=3, target_size=(64, 64)):
    def load_dataset(data_dir):
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
        out_dir = os.path.join(data_dir, "lst")

        evi_paths = get_sorted_image_paths(evi_dir)
        ndwi_paths = get_sorted_image_paths(ndwi_dir)
        lst_paths = get_sorted_image_paths(lst_dir)
        out_paths = get_sorted_image_paths(out_dir)

        min_length = min(len(evi_paths), len(ndwi_paths), len(lst_paths))
        if min_length < sequence_length + 1:
            raise ValueError(
                f"Not enough images to form sequences. Minimum required: {sequence_length + 1}, but found: {min_length}"
            )

        evi_inputs, ndwi_inputs, lst_inputs, outputs = [], [], [], []
        for i in range(min_length - sequence_length):

            evi_sequence = [preprocess_image(evi_paths[j], target_size) for j in range(i, i + sequence_length)]
            ndwi_sequence = [preprocess_image(ndwi_paths[j], target_size) for j in range(i, i + sequence_length)]
            lst_sequence = [preprocess_image(lst_paths[j], target_size) for j in range(i, i + sequence_length)]

            evi_inputs.append(tf.stack(evi_sequence, axis=-1))
            ndwi_inputs.append(tf.stack(ndwi_sequence, axis=-1))
            lst_inputs.append(tf.stack(lst_sequence, axis=-1))

            output_tensor = preprocess_image(out_paths[i + sequence_length], target_size)
            outputs.append(output_tensor)

        dataset = tf.data.Dataset.from_tensor_slices(((evi_inputs, ndwi_inputs, lst_inputs), outputs))
        dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        return dataset

    train_ds = load_dataset(train_dir)
    val_ds = load_dataset(val_dir)
    return train_ds, val_ds


