

import os
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import json

import tensorflow as tf
from tensorflow.keras.utils import plot_model
from tensorflow.keras.mixed_precision import set_global_policy

from utils.data_loader import load_data
from utils.models import build_model


if __name__ == "__main__":


    with open('config.json', 'r') as file:
        config = json.load(file)
    
    MODEL_PATH = config['MODEL_PATH']
    DATA_DIR = config['DATA_DIR']
    IMAGE_Y = config['IMAGE_Y']
    IMAGE_X = config['IMAGE_X']
    SEQUENCE_LEN = config['SEQUENCE_LEN']
    SEQUENCE_STEP = config['SEQUENCE_STEP']
    FUTURE_STEP = config['FUTURE_STEP']
    INDICATORS_COUNT = config['INDICATORS_COUNT']
    BATCH_SIZE = config['BATCH_SIZE']
    EPOCHS = config['EPOCHS']

    target_size=(IMAGE_X, IMAGE_Y)

    train_ds, val_ds = load_data(DATA_DIR, 
                    batch_size=BATCH_SIZE, sequence_length=SEQUENCE_LEN, sequence_step=SEQUENCE_STEP, 
                    future_step=FUTURE_STEP, target_size=target_size)

    model = build_model(input_shape=(IMAGE_X, IMAGE_Y, INDICATORS_COUNT * SEQUENCE_LEN))
    #plot_model(model, to_file="model_structure_unet.png", show_shapes=True, show_layer_names=True)

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
    model.save(MODEL_PATH)

    history_df = pd.DataFrame(history.history)
    history_df.to_csv("training_log.csv", index=False)

    # Save history as a pickle file
    with open("training_history.pkl", "wb") as f:
        pickle.dump(history.history, f)

