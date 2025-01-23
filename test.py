import tensorflow as tf
from tensorflow.keras import layers, models, Input
from tensorflow.keras.utils import plot_model
from tensorflow.keras.mixed_precision import set_global_policy

import os

def preprocess_image(image_path, target_size=(1299, 636)):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_png(img, channels=3)
    img = tf.image.resize(img, target_size)
    img = img / 255.0
    return img

def load_data(train_dir, val_dir, batch_size=32, sequence_length=3, target_size=(1299, 636)):
    def load_dataset(data_dir):
        evi_dir = os.path.join(data_dir, "EVI")
        ndwi_dir = os.path.join(data_dir, "NDWI")
        lst_dir = os.path.join(data_dir, "LST")
        out_dir = os.path.join(data_dir, "LST")

        evi_paths = sorted([os.path.join(evi_dir, img) for img in os.listdir(evi_dir)])
        ndwi_paths = sorted([os.path.join(ndwi_dir, img) for img in os.listdir(ndwi_dir)])
        lst_paths = sorted([os.path.join(lst_dir, img) for img in os.listdir(lst_dir)])
        out_paths = sorted([os.path.join(out_dir, img) for img in os.listdir(out_dir)])

        inputs_evi, inputs_ndwi, inputs_lst, outputs = [], [], [], []
        for i in range(len(evi_paths) - sequence_length):
            evi_sequence = [preprocess_image(evi_paths[j], target_size) for j in range(i, i + sequence_length)]
            ndwi_sequence = [preprocess_image(ndwi_paths[j], target_size) for j in range(i, i + sequence_length)]
            lst_sequence = [preprocess_image(lst_paths[j], target_size) for j in range(i, i + sequence_length)]

            inputs_evi.append(tf.stack(evi_sequence, axis=-1))
            inputs_ndwi.append(tf.stack(ndwi_sequence, axis=-1))
            inputs_lst.append(tf.stack(lst_sequence, axis=-1))

            output_tensor = preprocess_image(out_paths[i + sequence_length], target_size)
            outputs.append(output_tensor)

        dataset = tf.data.Dataset.from_tensor_slices(((inputs_evi, inputs_ndwi, inputs_lst), outputs))
        dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        return dataset

    train_ds = load_dataset(train_dir)
    val_ds = load_dataset(val_dir)
    return train_ds, val_ds


def build_model(input_shape=(128, 64, 3, 3)):
    def branch(input_shape):
        branch_input = Input(shape=input_shape)
        x = layers.Conv3D(16, (3, 3, 3), activation='relu', padding='same')(branch_input)
        x = layers.MaxPooling3D((2, 2, 1))(x)  
        x = layers.Conv3D(32, (3, 3, 3), activation='relu', padding='same')(x)
        x = layers.MaxPooling3D((2, 2, 1))(x)
        x = layers.Conv3D(64, (3, 3, 3), activation='relu', padding='same')(x)
        x = layers.Flatten()(x) 
        return models.Model(inputs=branch_input, outputs=x)

    evi_branch = branch(input_shape)
    ndwi_branch = branch(input_shape)
    lst_branch = branch(input_shape)

    combined = layers.concatenate([evi_branch.output, ndwi_branch.output, lst_branch.output])

    x = layers.Dense(128, activation='relu')(combined)
    x = layers.Dense(IMAGE_X // 4 * IMAGE_Y // 4 * 64, activation='relu')(x)

    x = layers.Reshape((IMAGE_X // 4, IMAGE_Y // 4, 64))(x)  

    x = layers.Conv2DTranspose(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.UpSampling2D((2, 2))(x) 
    x = layers.Conv2DTranspose(32, (3, 3), activation='relu', padding='same')(x)
    x = layers.UpSampling2D((2, 2))(x) 

    x = layers.Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

    model = models.Model(inputs=[evi_branch.input, ndwi_branch.input, lst_branch.input], outputs=x)
    return model


if __name__ == "__main__":


    IMAGE_X = 64  # 128   # 325 (1/4)   
    IMAGE_Y = 128 # 64    # 159 (1/4)
    SEQUENCE_LEN = 3
    INDICATORS_COUNT = 3
    BATCH_SIZE = 8
    EPOCHS = 10

    target_size=(IMAGE_X, IMAGE_Y)
    train_ds, val_ds = load_data("data/Prague/train", "data/Prague/val", batch_size=BATCH_SIZE, 
                                    sequence_length=SEQUENCE_LEN, target_size=target_size)

    model = build_model(input_shape=(IMAGE_X, IMAGE_Y, SEQUENCE_LEN, INDICATORS_COUNT))
    #plot_model(model, to_file="model_structure3.png", show_shapes=True, show_layer_names=True)

    model.compile(optimizer='adam', loss='mse', metrics=['mae'])

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath="models/checkpoints/model_checkpoint.h5",
            save_best_only=True,
            monitor="val_loss",
            mode="min"
        ),
        tf.keras.callbacks.TensorBoard(log_dir="logs")
    ]

    set_global_policy('mixed_float16')

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks
    )

    os.makedirs("models/final", exist_ok=True)
    #model.save("models/final/heat_map_model.h5")
    model.save("models/final/heat_map_model.keras")

