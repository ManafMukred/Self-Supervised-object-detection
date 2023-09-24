from sklearn.model_selection import train_test_split
import cv2
import argparse
import numpy as np
import tensorflow_similarity as tfsim
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

parser = argparse.ArgumentParser(
    description='get SSL parameters')

parser.add_argument('-i','--img_dir', dest='DATA',
                    help='data directory')
parser.add_argument('-n','--num_aug', dest='AUG', default= 0, type=int,
                    help='number of augmentations')
parser.add_argument('-s','--im_size', dest='SIZE', default= 64, type=int,
                    help='size of images')
args = vars(parser.parse_args())

def data_process(data_dir= args['DATA'], num_aug= args['AUG'],im_size = args['SIZE'] ):
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

    if num_aug != 0:
    # Iterate through the augmented batches
        for k in range(num_aug):
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

    else:
        merged_imgs = x_raw
        merged_labels = y_raw

    indices = np.arange(merged_imgs.shape[0])
    np.random.shuffle(indices)

    # Use these indices to shuffle both the images and labels
    x_merged = merged_imgs[indices]
    y_merged = merged_labels[indices]

    X_train, X_test, y_train, y_test = train_test_split(x_merged, y_merged, test_size=0.1, random_state=42, stratify=y_raw)

    return X_train, X_test, y_train, y_test


def img_scaling(img):
    return tf.keras.applications.imagenet_utils.preprocess_input(
        img, 
        data_format=None, 
        mode='torch')

@tf.function
def simsclr_augmentor(img):
    return simsiam_augmentor(img, blur=True, area_range=(0.33, 1.0), h2=1.0, h3=1.0)
@tf.function
def simsiam_augmentor(img, blur=True, area_range=(0.2, 1.0), h2=0.4, h3=0.4):

    # random resize and crop. Increase the size before we crop.
    img = tfsim.augmenters.augmentation_utils.cropping.crop_and_resize(
        img, args['SIZE'], args['SIZE'], area_range=area_range
    )
    
    # The following transforms expect the data to be [0, 1]
    img /= 255.
    
    # random color jitter
    def _jitter_transform(x):
        return tfsim.augmenters.augmentation_utils.color_jitter.color_jitter_rand(
            x,
            np.random.uniform(0.0, 0.4),
            np.random.uniform(0.0, h2),
            np.random.uniform(0.0, h3),
            np.random.uniform(0.0, 0.1),
            "multiplicative",
        )

    img = tfsim.augmenters.augmentation_utils.random_apply.random_apply(_jitter_transform, p=0.8, x=img)

    # # random grayscale
    def _grascayle_transform(x):
        return tfsim.augmenters.augmentation_utils.color_jitter.to_grayscale(x)

    img = tfsim.augmenters.augmentation_utils.random_apply.random_apply(_grascayle_transform, p=0.2, x=img)

    # optional random gaussian blur
    if blur:
        img = tfsim.augmenters.augmentation_utils.blur.random_blur(img, p=0.5, height=args['SIZE'], width=args['SIZE'])

    # random horizontal flip
    img = tf.image.random_flip_left_right(img)
    
    # scale the data back to [0, 255]
    img = img * 255.
    img = tf.clip_by_value(img, 0., 255.)

    return img


def get_projector(input_dim, dim, activation="relu", num_layers: int = 2):
    inputs = tf.keras.layers.Input((input_dim,), name="projector_input")
    x = inputs

    for i in range(num_layers - 1):
        x = tf.keras.layers.Dense(
            dim,
            use_bias=False,
            kernel_initializer=tf.keras.initializers.LecunUniform(),
            name=f"projector_layer_{i}",
        )(x)
        x = tf.keras.layers.BatchNormalization(epsilon=1.001e-5, name=f"batch_normalization_{i}")(x)
        x = tf.keras.layers.Activation(activation, name=f"{activation}_activation_{i}")(x)
    x = tf.keras.layers.Dense(
        dim,
        use_bias=False,
        kernel_initializer=tf.keras.initializers.LecunUniform(),
        name="projector_output",
    )(x)
    x = tf.keras.layers.BatchNormalization(
        epsilon=1.001e-5,
        center=False,  # Page:5, Paragraph:2 of SimSiam paper
        scale=False,  # Page:5, Paragraph:2 of SimSiam paper
        name=f"batch_normalization_ouput",
    )(x)
    # Metric Logging layer. Monitors the std of the layer activations.
    # Degnerate solutions colapse to 0 while valid solutions will move
    # towards something like 0.0220. The actual number will depend on the layer size.
    o = tfsim.layers.ActivationStdLoggingLayer(name="proj_std")(x)
    projector = tf.keras.Model(inputs, o, name="projector")
    return projector

def get_predictor(input_dim, hidden_dim=512, activation="relu"):
    inputs = tf.keras.layers.Input(shape=(input_dim,), name="predictor_input")
    x = inputs

    x = tf.keras.layers.Dense(
        hidden_dim,
        use_bias=False,
        kernel_initializer=tf.keras.initializers.LecunUniform(),
        name="predictor_layer_0",
    )(x)
    x = tf.keras.layers.BatchNormalization(epsilon=1.001e-5, name="batch_normalization_0")(x)
    x = tf.keras.layers.Activation(activation, name=f"{activation}_activation_0")(x)

    x = tf.keras.layers.Dense(
        input_dim,
        kernel_initializer=tf.keras.initializers.LecunUniform(),
        name="predictor_output",
    )(x)
    # Metric Logging layer. Monitors the std of the layer activations.
    # Degnerate solutions colapse to 0 while valid solutions will move
    # towards something like 0.0220. The actual number will depend on the layer size.
    o = tfsim.layers.ActivationStdLoggingLayer(name="pred_std")(x)
    predictor = tf.keras.Model(inputs, o, name="predictor")
    return predictor
 
@tf.function
def eval_augmenter(img):
    # random resize and crop. Increase the size before we crop.
    img = tfsim.augmenters.augmentation_utils.cropping.crop_and_resize(
        img, args['SIZE'], args['SIZE'], area_range=(0.2, 1.0)
    )
    # random horizontal flip
    img = tf.image.random_flip_left_right(img)
    img = tf.clip_by_value(img, 0., 255.)
    return img