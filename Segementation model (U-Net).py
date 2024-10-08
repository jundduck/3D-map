!pip install tensorflow
!pip install opencv-python
!pip install matplotlib
!pip install numpy


# U-Net Model Definition
import tensorflow as tf
from tensorflow.keras import layers, models

def unet_model(input_size=(256, 256, 3)):
    inputs = layers.Input(input_size)

    # incode
    c1 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    c1 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(c1)
    p1 = layers.MaxPooling2D((2, 2))(c1)

    c2 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(p1)
    c2 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c2)
    p2 = layers.MaxPooling2D((2, 2))(c2)

    # decode
    u1 = layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c2)
    u1 = layers.concatenate([u1, c1])
    c3 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(u1)
    c3 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(c3)

    # output
    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(c3)

    model = models.Model(inputs=[inputs], outputs=[outputs])
    return model




# loading and preprocessing
import cv2
import numpy as np
import os

def load_data(image_dir, mask_dir):
    images = []
    masks = []

    for img_name in os.listdir(image_dir):
        img = cv2.imread(os.path.join(image_dir, img_name))
        img = cv2.resize(img, (256, 256))
        img = img / 255.0  # Normalize
        images.append(img)

        mask = cv2.imread(os.path.join(mask_dir, img_name), cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (256, 256))
        mask = (mask > 0).astype(np.uint8) 
        masks.append(mask)

    return np.array(images), np.array(masks)



# Training
image_dir = 'path/to/images'
mask_dir = 'path/to/masks'

from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping

x, y = load_data(image_dir, mask_dir)

model = unet_model()

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Early Stopping set
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=100, callbacks=[early_stopping])






# Visualization
def predict_and_visualize(model, image):
    pred = model.predict(np.expand_dims(image, axis=0))
    pred_mask = (pred.squeeze() > 0.5).astype(np.uint8) * 255

    cv2.imshow("Original Image", image)
    cv2.imshow("Predicted Mask", pred_mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


test_image = cv2.imread('path/to/test/image.jpg')
test_image = cv2.resize(test_image, (256, 256))
test_image = test_image / 255.0  # Normalize

predict_and_visualize(model, test_image)
