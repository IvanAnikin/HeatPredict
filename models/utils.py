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



def build_model2(input_shape=(1299, 636, 3)):
    def branch(input_shape):
        branch_input = Input(shape=input_shape)
        x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(branch_input)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = layers.Flatten()(x)
        return models.Model(inputs=branch_input, outputs=x)

    evi_branch = branch(input_shape)
    ndwi_branch = branch(input_shape)
    lst_branch = branch(input_shape)

    combined = layers.concatenate([evi_branch.output, ndwi_branch.output, lst_branch.output])

    '''
    x = layers.Dense(256, activation='relu')(combined)
    x = layers.Dense(128, activation='relu')(x)
    # x = layers.Reshape((33, 66, 64))(x)  # Adjust dimensions based on input shape and output target
    x = layers.Reshape((8, 4, 4))(x)
    '''

    flatten_dim = combined.shape[-1] 
    reshape_height = IMAGE_X // (2 ** 2) 
    reshape_width = IMAGE_Y // (2 ** 2)   
    reshape_channels = flatten_dim // (reshape_height * reshape_width)
    assert reshape_channels * reshape_height * reshape_width == flatten_dim, \
        "Cannot reshape to the target dimensions. Adjust pooling levels or layer structure."
    x = layers.Dense(256, activation='relu')(combined)
    x = layers.Dense(flatten_dim, activation='relu')(x)  
    x = layers.Reshape((reshape_height, reshape_width, reshape_channels))(x)



    x = layers.Conv2DTranspose(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.Conv2DTranspose(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

    model = models.Model(inputs=[evi_branch.input, ndwi_branch.input, lst_branch.input], outputs=x)
    return model


# TOO BIG
def build_model3(input_shape=(1299, 636, 3)):
    def branch(input_shape):
        branch_input = Input(shape=input_shape)
        x = layers.Conv3D(32, (3, 3, 3), activation='relu', padding='same')(branch_input)
        x = layers.MaxPooling3D((2, 2, 1))(x)
        x = layers.Conv3D(64, (3, 3, 3), activation='relu', padding='same')(x)
        x = layers.MaxPooling3D((2, 2, 1))(x)
        x = layers.Conv3D(128, (3, 3, 3), activation='relu', padding='same')(x)
        x = layers.Flatten()(x)
        return models.Model(inputs=branch_input, outputs=x)

    evi_branch = branch(input_shape)
    ndwi_branch = branch(input_shape)
    lst_branch = branch(input_shape)
    combined = layers.concatenate([evi_branch.output, ndwi_branch.output, lst_branch.output])


    flatten_dim = combined.shape[-1] 
    reshape_height = IMAGE_X // (2 ** 2) 
    reshape_width = IMAGE_Y // (2 ** 2)   
    reshape_channels = flatten_dim // (reshape_height * reshape_width)
    assert reshape_channels * reshape_height * reshape_width == flatten_dim, \
        "Cannot reshape to the target dimensions. Adjust pooling levels or layer structure."
    x = layers.Dense(256, activation='relu')(combined)
    x = layers.Dense(flatten_dim, activation='relu')(x)  
    x = layers.Reshape((reshape_height, reshape_width, reshape_channels))(x)


    x = layers.Conv2DTranspose(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.Conv2DTranspose(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

    model = models.Model(inputs=[evi_branch.input, ndwi_branch.input, lst_branch.input], outputs=x)
    return model


# wrong output shape
def build_model5(input_shape=(128, 64, 3, 3)):
    def branch(input_shape):
        branch_input = Input(shape=input_shape)
        x = layers.Conv3D(16, (3, 3, 3), activation='relu', padding='same')(branch_input)
        x = layers.MaxPooling3D((2, 2, 1))(x)  
        x = layers.Conv3D(32, (3, 3, 3), activation='relu', padding='same')(x)
        x = layers.MaxPooling3D((2, 2, 1))(x)  
        x = layers.Conv3D(64, (3, 3, 3), activation='relu', padding='same')(x)
        x = layers.GlobalAveragePooling3D()(x)  
        return models.Model(inputs=branch_input, outputs=x)

    evi_branch = branch(input_shape)
    ndwi_branch = branch(input_shape)
    lst_branch = branch(input_shape)

    combined = layers.concatenate([evi_branch.output, ndwi_branch.output, lst_branch.output])

    x = layers.Dense(128, activation='relu')(combined)
    x = layers.Dense(64, activation='relu')(x)

    #x = layers.Reshape((IMAGE_X // (2 ** 2), IMAGE_Y // (2 ** 2), 4))(x)  
    reshape_dim = int(x.shape[-1])  
    reshape_height = int(reshape_dim ** 0.5) 
    reshape_width = reshape_dim // reshape_height
    reshape_channels = 1
    x = layers.Reshape((reshape_height, reshape_width, reshape_channels))(x)

    x = layers.Conv2DTranspose(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)  

    model = models.Model(inputs=[evi_branch.input, ndwi_branch.input, lst_branch.input], outputs=x)
    return model