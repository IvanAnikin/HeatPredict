from tensorflow.keras import layers, models

def build_model(input_shape):
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same', strides=2),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same', strides=2),
        layers.Conv2D(128, (3, 3), activation='relu', padding='same', strides=2),
        layers.Conv2DTranspose(128, (3, 3), activation='relu', padding='same', strides=2),
        layers.Conv2DTranspose(64, (3, 3), activation='relu', padding='same', strides=2),
        layers.Conv2DTranspose(32, (3, 3), activation='relu', padding='same', strides=2),
        layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')  
    ])
    return model
