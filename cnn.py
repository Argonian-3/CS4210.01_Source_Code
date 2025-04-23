#conda activate tf
# python <filename>
#-------------------------------------------------------------------------
# AUTHOR: your name
# FILENAME: title of the source file
# SPECIFICATION: description of the program
# FOR: CS 4210 - Assignment #4
# TIME SPENT: how long it took you to complete the assignment
#-------------------------------------------------------------------------

# Importing Python libraries
import os
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras import layers, models
import operator

# Function to load dataset
def load_digit_images_from_folder(folder_path, image_size=(32, 32)):
    X = []
    y = []
    for filename in os.listdir(folder_path):
        # Getting the label of the image (it's the first number in the filename)
        # --> add your Python code here
        label = int(filename[0:operator.indexOf(filename,'_')])

        img = Image.open(os.path.join(folder_path, filename)).convert('L').resize(image_size)

        X.append(np.array(img))
        y.append(label)
    return np.array(X), np.array(y)

# Set your own paths here (relative to your project folder)
train_path = os.path.join("digit_dataset", "train")
test_path = os.path.join("digit_dataset", "test")

X_train, Y_train = load_digit_images_from_folder(train_path)
X_test, Y_test = load_digit_images_from_folder(test_path)

# Normalizing the data: convert pixel values from range [0, 255] to [0, 1]. Hint: divide them by 255
# --> add your Python code here
X_train = X_train / 255.0
X_test = X_test / 255.0

# Reshaping the input images to include the channel dimension: (num_images, height, width, channels)
# --> add your Python code here
X_train = X_train.reshape(len(X_train),32,32,1)
X_test = X_test.reshape(len(X_test),32,32,1)

model = models.Sequential([
    layers.Conv2D(32,3,1,"valid",activation='relu'),

    layers.MaxPool2D(2),

    layers.Flatten(),

    layers.Dense(64,activation='relu'),

    layers.Dense(10,activation='softmax',name='outputs')
])
print(X_train.dtype)
print(Y_train.dtype)
model.compile(optimizer='sgd', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(X_train, Y_train, batch_size=32, epochs=10, validation_data=(X_test, Y_test))

loss, acc = model.evaluate(X_test, Y_test)

print(acc)