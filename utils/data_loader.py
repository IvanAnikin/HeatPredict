import tensorflow as tf

import os

def preprocess_image(image_path, target_size=(1299, 636)):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_png(img, channels=3)
    img = tf.image.resize(img, target_size)  
    img = img / 255.0  
    return img


def load_data(root_dir, batch_size=32, sequence_length=3, sequence_step=1, future_step=1, target_size=(64, 64)):
    def load_dataset(city_dir):
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

        indicators_path = os.path.join(city_dir, "indicators")
        evi_paths = get_sorted_image_paths(os.path.join(indicators_path, "evi"))
        ndwi_paths = get_sorted_image_paths(os.path.join(indicators_path, "ndwi"))
        lst_paths = get_sorted_image_paths(os.path.join(indicators_path, "lst"))
        out_paths = get_sorted_image_paths(os.path.join(indicators_path, "lst"))

        min_length = min(len(evi_paths), len(ndwi_paths), len(lst_paths))
        if min_length < sequence_length + future_step:
            return None, None

        total_samples = min_length - future_step - (sequence_length - 1) * sequence_step
        split_index = int(0.8 * total_samples)

        evi_inputs_train, ndwi_inputs_train, lst_inputs_train, outputs_train = [], [], [], []
        evi_inputs_val, ndwi_inputs_val, lst_inputs_val, outputs_val = [], [], [], []

        for i in range(total_samples): 
            evi_sequence = [preprocess_image(evi_paths[j], target_size) for j in range(i, i + sequence_length * sequence_step, sequence_step)]
            ndwi_sequence = [preprocess_image(ndwi_paths[j], target_size) for j in range(i, i + sequence_length * sequence_step, sequence_step)]
            lst_sequence = [preprocess_image(lst_paths[j], target_size) for j in range(i, i + sequence_length * sequence_step, sequence_step)]

            output_tensor = preprocess_image(out_paths[i + (sequence_length - 1) * sequence_step + future_step], target_size)

            if i < split_index:
                evi_inputs_train.append(tf.stack(evi_sequence, axis=-1))
                ndwi_inputs_train.append(tf.stack(ndwi_sequence, axis=-1))
                lst_inputs_train.append(tf.stack(lst_sequence, axis=-1))
                outputs_train.append(output_tensor)
            else:
                evi_inputs_val.append(tf.stack(evi_sequence, axis=-1))
                ndwi_inputs_val.append(tf.stack(ndwi_sequence, axis=-1))
                lst_inputs_val.append(tf.stack(lst_sequence, axis=-1))
                outputs_val.append(output_tensor)

        train_dataset = tf.data.Dataset.from_tensor_slices(((evi_inputs_train, ndwi_inputs_train, lst_inputs_train), outputs_train))
        val_dataset = tf.data.Dataset.from_tensor_slices(((evi_inputs_val, ndwi_inputs_val, lst_inputs_val), outputs_val))

        return train_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE), val_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    train_datasets, val_datasets = [], []
    for city in os.listdir(root_dir):
        city_path = os.path.join(root_dir, city)
        if os.path.isdir(city_path):
            subfolders = [f for f in os.listdir(city_path) if os.path.isdir(os.path.join(city_path, f))]
            if subfolders:
                train_ds, val_ds = load_dataset(os.path.join(city_path, subfolders[2]))
                if train_ds and val_ds:
                    train_datasets.append(train_ds)
                    val_datasets.append(val_ds)

    train_ds = train_datasets[0] if train_datasets else None
    val_ds = val_datasets[0] if val_datasets else None
    for ds in train_datasets[1:]:
        train_ds = train_ds.concatenate(ds)
    for ds in val_datasets[1:]:
        val_ds = val_ds.concatenate(ds)

    return train_ds, val_ds


