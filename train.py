from utils.data_loader import load_data
from models.utils import build_model
import tensorflow as tf
import os

# Train: data/Prague/train/LST ? 
train_ds, val_ds = load_data("data/Prague/train", "data/Prague/val", batch_size=32)

print("Data loaded")

model = build_model(input_shape=(1299, 636, 27))
print("Model built") 
model.compile(optimizer='adam', loss='mse', metrics=['mae'])
print("Model compied")

callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        filepath="models/checkpoints/model_checkpoint.h5",
        save_best_only=True,
        monitor="val_loss",
        mode="min"
    ),
    tf.keras.callbacks.TensorBoard(log_dir="logs")
]

history = model.fit(train_ds, validation_data=val_ds, epochs=50, callbacks=callbacks)
print("Model fit complete")

os.makedirs("models/final", exist_ok=True)
model.save("models/final/heat_map_model.h5")
print("Model saved")
