import tensorflow as tf

import os

def preprocess_image(image_path, target_size=(1299, 636)):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_png(img, channels=3)
    img = tf.image.resize(img, target_size)  
    img = img / 255.0  
    return img

def load_data1(train_dir, val_dir, batch_size=32, sequence_length=3, target_size=(64, 64)):
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



def load_data(root_dir, batch_size=32, sequence_length=3, target_size=(64, 64)):
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
        if min_length < sequence_length + 1:
            return None

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
        return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    train_datasets, val_datasets = [], []
    for city in os.listdir(root_dir):
        city_path = os.path.join(root_dir, city)
        if os.path.isdir(city_path):
            subfolders = [f for f in os.listdir(city_path) if os.path.isdir(os.path.join(city_path, f))]
            if subfolders:
                city_dataset = load_dataset(os.path.join(city_path, subfolders[2]))
                if city_dataset:
                    train_datasets.append(city_dataset)
                    val_datasets.append(city_dataset)

    train_ds = train_datasets[0] if train_datasets else None
    val_ds = val_datasets[0] if val_datasets else None
    for ds in train_datasets[1:]:
        train_ds = train_ds.concatenate(ds)
    for ds in val_datasets[1:]:
        val_ds = val_ds.concatenate(ds)

    return train_ds, val_ds



# def get_sorted_image_paths(indicator_dir):
#     png_dir = os.path.join(indicator_dir, "png")
#     if not os.path.exists(png_dir):
#         return []
    
#     year_dirs = sorted(os.listdir(png_dir))
#     image_paths = []
#     for year in year_dirs:
#         year_path = os.path.join(png_dir, year)
#         if os.path.isdir(year_path):
#             year_images = sorted(
#                 [os.path.join(year_path, img) for img in os.listdir(year_path) if img.endswith(('.png', '.jpg'))]
#             )
#             image_paths.extend(year_images)
#     return image_paths

# def load_dataset(data_dir, sequence_length=3, target_size=(64, 64), batch_size=32):
#     evi_dir = os.path.join(data_dir, "evi")
#     ndwi_dir = os.path.join(data_dir, "ndwi")
#     lst_dir = os.path.join(data_dir, "lst")
#     out_dir = os.path.join(data_dir, "lst")

#     evi_paths = get_sorted_image_paths(evi_dir)
#     ndwi_paths = get_sorted_image_paths(ndwi_dir)
#     lst_paths = get_sorted_image_paths(lst_dir)
#     out_paths = get_sorted_image_paths(out_dir)

#     min_length = min(len(evi_paths), len(ndwi_paths), len(lst_paths))
#     if min_length < sequence_length + 1:
#         raise ValueError(
#             f"Not enough images to form sequences. Minimum required: {sequence_length + 1}, but found: {min_length}"
#         )

#     evi_inputs, ndwi_inputs, lst_inputs, outputs = [], [], [], []
#     for i in range(min_length - sequence_length):
#         evi_sequence = [preprocess_image(evi_paths[j], target_size) for j in range(i, i + sequence_length)]
#         ndwi_sequence = [preprocess_image(ndwi_paths[j], target_size) for j in range(i, i + sequence_length)]
#         lst_sequence = [preprocess_image(lst_paths[j], target_size) for j in range(i, i + sequence_length)]

#         evi_inputs.append(tf.stack(evi_sequence, axis=-1))
#         ndwi_inputs.append(tf.stack(ndwi_sequence, axis=-1))
#         lst_inputs.append(tf.stack(lst_sequence, axis=-1))

#         output_tensor = preprocess_image(out_paths[i + sequence_length], target_size)
#         outputs.append(output_tensor)

#     dataset = tf.data.Dataset.from_tensor_slices(((evi_inputs, ndwi_inputs, lst_inputs), outputs))
#     dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
#     return dataset

# def load_all_cities(base_dir, batch_size=8, sequence_length=3, target_size=(64, 64)):
#     city_dirs = []
#     for city in os.listdir(base_dir):
#         city_path = os.path.join(base_dir, city)
#         if os.path.isdir(city_path):
#             scene_dirs = [d for d in os.listdir(city_path) if os.path.isdir(os.path.join(city_path, d))]
#             if scene_dirs:
#                 indicators_path = os.path.join(city_path, scene_dirs[0], "indicators")
#                 city_dirs.append(indicators_path)
    
#     datasets = []
#     for city_dir in city_dirs:
#         try:
#             dataset = load_dataset(city_dir, sequence_length, target_size, batch_size)
#             datasets.append(dataset)
#             print(f"Loaded dataset from {city_dir}")
#         except ValueError as e:
#             print(f"Skipping {city_dir}: {e}")
    
#     return datasets
