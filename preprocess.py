from sklearn.model_selection import train_test_split
import random
import time
import datetime
from pathlib import Path
import cv2
import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import StratifiedKFold
import tensorflow_similarity as tfsim
import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow_datasets as tfds
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

parser = argparse.ArgumentParser(
    description='get SSL parameters')

parser.add_argument('-i','--img_dir', dest='DATA',
                    help='data directory')
parser.add_argument('-n','--num_aug', dest='AUG', default= 1, type=int,
                    help='number of augmentations')
parser.add_argument('-s','--im_size', dest='SIZE', default= 64, type=int,
                    help='size of images')
args = vars(parser.parse_args())

im_size = args['SIZE']
builder = tfds.ImageFolder(args['DATA'])
ds_info = builder.info # num examples, labels... are automatically calculated
# assuming that data split was or will be specified, we DO NOT use 'split' parameter
ds = builder.as_dataset(shuffle_files=True, as_supervised=True) 

class_names = builder.info.features['label'].names
NUM_CLASSES= len(class_names)

# Create an empty list to store augmented images and labels
augmented_images = []
augmented_labels = []
x_raw, y_raw = [], []

# unifying all images to one size
ds = ds.take(-1)
for image, label in tfds.as_numpy(ds):
    # ori = image
    image = cv2.resize(image, dsize=(im_size, im_size), interpolation=cv2.INTER_CUBIC)

    x_raw.append(image)
    y_raw.append(label)


# Set the directory path where your data is located
data_dir = args['DATA']
# Set the batch size and number of augmentations
num_augmentations = args['AUG']

# Initialize the ImageDataGenerator with desired augmentation parameters
datagen = ImageDataGenerator(
    # Specify desired augmentation techniques and parameters
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Use the ImageDataGenerator.flow_from_directory() method to load the data and perform augmentation
data_generator = datagen.flow_from_directory(
    directory=data_dir,
    target_size=(im_size, im_size),  # Specify the desired image size
    batch_size=32,
    class_mode='categorical',
    shuffle=True
)

# Iterate through the augmented batches
for k in range(num_augmentations):
    for i in range(len(data_generator)):
        # Generate augmented images and labels batch
        batch_images, batch_labels = data_generator.next()
        for j in range(batch_images.shape[0]):

        # Append the augmented images and labels to the lists
            augmented_images.append(batch_images[j])
            augmented_labels.append(np.argmax(batch_labels[j]))

# Convert the lists into NumPy arrays
augmented_images = np.array(augmented_images)
augmented_labels = np.array(augmented_labels)

# merge and shuffle augmented with original images while keeping respective labels  
merged_imgs = np.concatenate((x_raw,np.array(augmented_images)),axis=0)
merged_labels = np.concatenate((y_raw,np.array(augmented_labels)),axis=0)


indices = np.arange(merged_imgs.shape[0])
np.random.shuffle(indices)

# Use these indices to shuffle both the images and labels
x_merged = merged_imgs[indices]
y_merged = merged_labels[indices]

X_train, X_test, y_train, y_test = train_test_split(x_raw, y_raw, test_size=0.1, random_state=42, stratify=y_raw)

# save the processed test and each xval sets as np images for later use
if not os.path.exists('np_imgs'):
    os.mkdir('np_imgs')
np.save('np_imgs/x_test', X_test)
np.save('np_imgs/y_test', y_test)
#### create  sets for cross validation
skf = StratifiedKFold(n_splits=5) # 5 is the number of folds
f_num = 1
for train_index, val_index in skf.split(X_train, y_train):   
    train_images, val_images = X_train[train_index], X_train[val_index]
    train_labels, val_labels = y_train[train_index], y_train[val_index]
    np.save(f'np_imgs/x_Xtrain{f_num}', train_images)
    np.save(f'np_imgs/y_Xtrain{f_num}', train_labels)
    np.save(f'np_imgs/x_Xval{f_num}', val_images)
    np.save(f'np_imgs/y_Xval{f_num}', val_labels)
    f_num += 1
