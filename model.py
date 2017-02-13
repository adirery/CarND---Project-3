import numpy as np
import pandas as pd
import os, json

from skimage.exposure import adjust_gamma
from keras.models import Sequential
from keras.layers import Input, Dropout, Flatten, Activation, Convolution2D, MaxPooling2D, Dense
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint, EarlyStopping

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy import ndimage
from scipy.misc import imresize

### Create steering angle labels
data = pd.read_csv('data/driving_log.csv', header=None)
data.columns = ('center image', 'left image', 'right image', 'steering angle', 'throttle', 'break', 'speed')

angles = np.array(data['steering angle'])

print("Loading of steering angle labels complete. ", len(angles), " labels loaded")

# create np arrays for center / left / right images
images = np.asarray(os.listdir("data/IMG"))
center = np.ndarray(shape=(len(angles), 20, 64, 3))
left = np.ndarray(shape=(len(angles), 20, 64, 3))
right = np.ndarray(shape=(len(angles), 20, 64, 3))

print("ndarrays created for images, left & right")
# Load images
# Resize images to 32 x 64 to increase training / validation speed
# Crop image as top 12px not useful for learning --> input img dimension 20 x 64 x 3
count = 0

for image in images:
    image_file = os.path.join("data/IMG", image)
    if image.startswith("center"):
        image_data = ndimage.imread(image_file).astype(np.float32)
        center[count % len(angles)] = imresize(image_data, (32,64,3))[12:,:,:]
    elif image.startswith("left"):
        image_data = ndimage.imread(image_file).astype(np.float32)
        left[count % len(angles)] = imresize(image_data, (32,64,3))[12:,:,:]
    elif image.startswith("right"):
        image_data = ndimage.imread(image_file).astype(np.float32)
        right[count % len(angles)] = imresize(image_data, (32,64,3))[12:,:,:]
    count += 1
    if count%100==0:
        print("Went throught ", count , "of overall ", len(images))

print("Loading of images complete. ", count, " images loaded")

# Create training set
X_train = np.concatenate((center, left, right), axis=0)
y_train = np.concatenate((angles, (angles - 0.1), (angles + 0.1)), axis=0)

print("Training set created")

# Adjust gamma in images to increase contrast
X_train = adjust_gamma(X_train)

print("Training set gamma adjusted")

# Mirror images & labels to adjust for left steering bias (issue of training set)
mirror = np.ndarray(shape=(X_train.shape))

count = 0
for i in range(len(X_train)):
    mirror[count] = np.fliplr(X_train[i])
    count += 1

mirror_angles = y_train * -1

X_train = np.concatenate((X_train, mirror), axis=0)
y_train = np.concatenate((y_train, mirror_angles), axis=0)

print("Images mirrored. Full training set created")

# Split training / validation set
X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size = 0.1)

print("Training set split --> training (90%) & validation (10%)")

# Create Keras Model Architecture
model = Sequential()

model.add(BatchNormalization(axis=1, input_shape=(20,64,3)))
model.add(Convolution2D(16, 3, 3, border_mode="valid", subsample=(2,2), activation="relu"))
model.add(Convolution2D(24, 3, 3, border_mode="valid", subsample=(1,2), activation="relu"))
model.add(Convolution2D(36, 3, 3, border_mode="valid", activation="relu"))
model.add(Convolution2D(48, 3, 3, border_mode="valid", activation="relu"))
model.add(Convolution2D(48, 3, 3, border_mode="valid", activation="relu"))
model.add(Flatten())
model.add(Dense(512))
model.add(Dropout(0.5))
model.add(Activation("relu"))
model.add(Dense(10))
model.add(Activation("relu"))
model.add(Dense(1))

model.summary()

# Optimize model with Adam / learning rate to 0.0001, loss function is mean-squared-error
adam = Adam(lr=0.0001)
model.compile(loss="mse", optimizer= adam)

# Save model when loss improves
checkpoint = ModelCheckpoint(filepath = 'model.h5', verbose = 1, save_best_only = True, monitor="val_loss")

# Stop training when loss stops to decrease
callback = EarlyStopping(monitor="val_loss", patience=2, verbose=1)

# Train model
model.fit(X_train,
        y_train,
        nb_epoch=20,
        verbose=1,
        batch_size=64,
        shuffle=True,
        validation_data=(X_validation, y_validation),
        callbacks=[checkpoint, callback])

json_string = model.to_json()
with open("model.json", "w") as savefile:
    json.dump(json_string, savefile)

model.save("model.h5")

print("Model saved")