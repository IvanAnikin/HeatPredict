

import os
import pickle
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.utils import plot_model
from tensorflow.keras.mixed_precision import set_global_policy

from utils.data_loader import load_data
from utils.models import build_model


if __name__ == "__main__":


    IMAGE_X = 128 
    IMAGE_Y = 128
    SEQUENCE_LEN = 4
    SEQUENCE_STEP = 2
    FUTURE_STEP = 2
    INDICATORS_COUNT = 3
    BATCH_SIZE = 16
    EPOCHS = 10

    target_size=(IMAGE_X, IMAGE_Y)
    train_ds, val_ds = load_data(r"C:\Users\ivana\Downloads\Bakalarka\anime\urban_resilience", 
                    batch_size=BATCH_SIZE, sequence_length=SEQUENCE_LEN, sequence_step=SEQUENCE_STEP, 
                    future_step=FUTURE_STEP, target_size=target_size)

    model = build_model(IMAGE_X=IMAGE_X, IMAGE_Y=IMAGE_Y, SEQUENCE_LEN=SEQUENCE_LEN, INDICATORS_COUNT=INDICATORS_COUNT)
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
    model.save("models/final/heat_map_model.keras")

    history_df = pd.DataFrame(history.history)
    history_df.to_csv("training_log.csv", index=False)

    # Save history as a pickle file
    with open("training_history.pkl", "wb") as f:
        pickle.dump(history.history, f)

