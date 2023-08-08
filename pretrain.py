from  models.official.modeling.optimization.lars_optimizer import LARS
import argparse
import os
import random
import time
import datetime
from pathlib import Path
import cv2
import pickle
import matplotlib.pyplot as plt
import numpy as np
import tensorflow_similarity as tfsim
import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow_datasets as tfds
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# create model dir
DATA_PATH = Path("my_models")
if not DATA_PATH.exists():
    DATA_PATH.mkdir(parents=True)
# extend the GPU memory region needed by the TensorFlow process.
gpu_devices = tf.config.experimental.list_physical_devices("GPU") 
for device in gpu_devices: 
    tf.config.experimental.set_memory_growth(device, True)

parser = argparse.ArgumentParser(
    description='get SSL parameters')
parser.add_argument('-a','--algorithm', dest='ALGORITHM', default= 'simclr',
                    help='name of self supervised algorithm')
parser.add_argument('-i','--img_dir', dest='DATA',
                    help='name of self supervised algorithm')
parser.add_argument('-n','--num_aug', dest='AUG', default= 1, type=int,
                    help='name of self supervised algorithm')
args = vars(parser.parse_args())


with tf.device("CPU"):

    im_size = 64
    builder = tfds.ImageFolder(args['DATA'])
    ds_info = builder.info # num examples, labels... are automatically calculated
    # assuming that data split was or will be specified, we DO NOT use 'split' parameter
    ds = builder.as_dataset(shuffle_files=True, as_supervised=True) 

class_names = builder.info.features['label'].names
NUM_CLASSES= len(class_names)

# Set the directory path where your data is located
data_dir = args['DATA']

num_augmentations = args['AUG']

# Create an empty list to store augmented images and labels
augmented_images = []
augmented_labels = []

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

#### setting hyperparameters 

ALGORITHM = args['ALGORITHM']  # @param ["barlow", "simsiam", "simclr", "vicreg"]

# setting up hyperparameters
IMG_SIZE = 64
BATCH_SIZE = 256
PRE_TRAIN_EPOCHS = 100
PRE_TRAIN_STEPS_PER_EPOCH = 18720 // BATCH_SIZE   #################################33
VAL_STEPS_PER_EPOCH = 4680// BATCH_SIZE   #20
WEIGHT_DECAY = 1e-4
DIM = 2048  # The layer size for the projector and predictor models.
WARMUP_LR = 0.0
WARMUP_STEPS = 0
TEMPERATURE = None


# time
now = datetime.datetime.now() 
TIME = now.strftime("%H%M.%d%m%Y")
DATA_NAME = f'{IMG_SIZE}image{NUM_CLASSES}'

if ALGORITHM == "simsiam":
    INIT_LR = 5e-2 * int(BATCH_SIZE / 256)  # try optimal 0.5
    WEIGHT_DECAY = 5e-4  # optimal 1e-5
    N_LAYER = 2
    DIM = 2048  # The layer size for the projector and predictor models.

elif ALGORITHM == "barlow":
    # INIT_LR = 1e-3  # Initial LR for the learning rate schedule.
    INIT_LR = 0.2 * int(BATCH_SIZE / 256) # for LARS optimizer
    WEIGHT_DECAY = 1e-6 * 1.5
    WARMUP_STEPS = 1000
    DIM = 2048  # The layer size for the projector


elif ALGORITHM == "simclr":
    INIT_LR = 0.3 * int(BATCH_SIZE / 256) # for LARS optimizer
    # INIT_LR = 1e-3  # Initial LR for the learning rate schedule, see section B.1 in the paper.
    TEMPERATURE = 0.1  # Tuned for CIFAR10, see section B.9 in the paper.
    WEIGHT_DECAY = 1e-6 
    N_LAYER = 2
    DIM = 128  # The layer size for the projector and predictor models.

from augmentation import augment_image_pretraining, augment_image_finetuning

def img_scaling(img):
    return tf.keras.applications.imagenet_utils.preprocess_input(
        img, 
        data_format=None, 
        mode='torch')

@tf.function
def simsclr_augmenter(img):
    return simsiam_augmenter(img, blur=True, area_range=(0.33, 1.0), h2=1.0, h3=1.0)
@tf.function
def simsiam_augmenter(img, blur=True, area_range=(0.2, 1.0), h2=0.4, h3=0.4):

    # random resize and crop. Increase the size before we crop.
    img = tfsim.augmenters.augmentation_utils.cropping.crop_and_resize(
        img, IMG_SIZE, IMG_SIZE, area_range=area_range
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
        img = tfsim.augmenters.augmentation_utils.blur.random_blur(img, p=0.5, height=im_size, width=im_size)

    # random horizontal flip
    img = tf.image.random_flip_left_right(img)
    
    # scale the data back to [0, 255]
    img = img * 255.
    img = tf.clip_by_value(img, 0., 255.)

    return img


@tf.function()
def process(img):
    if ALGORITHM == 'simsiam':
        view1 = simsiam_augmenter(img)
        view1 = img_scaling(view1)
        view2 = simsiam_augmenter(img)
        view2 = img_scaling(view2)
        return (view1, view2)
    else:
        view1 = simsclr_augmenter(img)
        view1 = img_scaling(view1)
        view2 = simsclr_augmenter(img)
        view2 = img_scaling(view2)
        return (view1, view2)

## backbone function
def get_backbone(img_size, activation="relu", preproc_mode="torch"):
    input_shape = (img_size, img_size, 3)
    backbone = tfsim.architectures.ResNet50Sim(
        input_shape,
        include_top=False,  # Take the pooling layer as the output.
        pooling="avg",
    )
    return backbone

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

@tf.function
def eval_augmenter(img):
    # random resize and crop. Increase the size before we crop.
    img = tfsim.augmenters.augmentation_utils.cropping.crop_and_resize(
        img, IMG_SIZE, IMG_SIZE, area_range=(0.2, 1.0)
    )
    # random horizontal flip
    img = tf.image.random_flip_left_right(img)
    img = tf.clip_by_value(img, 0., 255.)
    return img

def get_eval_model(img_size, backbone, total_steps, trainable=True,algo = None, lr=0.1):
        if algo == "simsiam":
            lr = 0.1
            decay = 1e-4
            # lr =5.0
            cosine_decayed_lr = tf.keras.experimental.CosineDecay(initial_learning_rate=lr, decay_steps=total_steps)
            # opt = tf.keras.optimizers.SGD(cosine_decayed_lr, momentum=0.9, decay=decay)
            opt = LARS(learning_rate= 0.02, weight_decay_rate=0)
        elif algo == "barlow":
            lr = 0.03
            decay= 1e-4
            cosine_decayed_lr = tf.keras.experimental.CosineDecay(initial_learning_rate=lr, decay_steps=total_steps)
            opt = tf.keras.optimizers.SGD(cosine_decayed_lr, momentum=0.9, decay= decay)
        elif algo == "simclr":
            lr = 0.026
            decay = 1e-6
            cosine_decayed_lr = tf.keras.experimental.CosineDecay(initial_learning_rate=lr, decay_steps=total_steps)
            opt = tf.keras.optimizers.SGD(cosine_decayed_lr, momentum=0.9, decay= decay, nesterov= True)
        else:
            decay = 1e-6
            lr = 0.026
            cosine_decayed_lr = tf.keras.experimental.CosineDecay(initial_learning_rate=lr, decay_steps=total_steps)
            opt = tf.keras.optimizers.SGD(cosine_decayed_lr, momentum=0.9,decay=decay)
            # opt = tf.keras.optimizers.SGD()
            
        backbone.trainable = trainable
        inputs = tf.keras.layers.Input((img_size, img_size, 3), name="eval_input")
        x = backbone(inputs, training=trainable)
        #####################
        # flatten = Flatten()(x)
        # dense1 = Dense(512, activation='relu')(flatten)
        # dropout1 = Dropout(0.5)(dense1)
        # dense2 = Dense(128, activation='relu')(dropout1)
        # x = Dropout(0.5)(dense2)
        ###########
        o = tf.keras.layers.Dense(NUM_CLASSES, activation="softmax")(x)
        model = tf.keras.Model(inputs, o)
        model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=["acc"])
        return model


EXP_NAME = f"Xval5_{ALGORITHM}_{DATA_NAME}_b{BATCH_SIZE}_e{PRE_TRAIN_EPOCHS}_{TIME}"
Path(DATA_PATH / EXP_NAME).mkdir()
Path(DATA_PATH / EXP_NAME / "cls_eval").mkdir()
# setting logs dir
log_dir = DATA_PATH / EXP_NAME / "logs" 
chkpt_dir = DATA_PATH / EXP_NAME / "checkpoint" 

x_test = np.load('np_images/xraw5carsaug64_Xtest.npy')
y_test = np.load('np_images/yraw5carsaug64_Xtest.npy')
x_test  =  tf.convert_to_tensor(x_test)
y_test  =  tf.convert_to_tensor(y_test)

########################################## main train for Xval ##########################################
Histories = {}
pt_Histories = {}
no_pt_Histories = {}
ptpct_Histories = {}
no_res = []
pt_res = []
pctpt_res = []
num_trials = 5
for i in range(num_trials): 

    with tf.device("CPU"):  
        x_train = np.load(f'np_images/xraw5carsaug64_Xt{i+1}.npy')
        y_train = np.load(f'np_images/yraw5carsaug64_Xt{i+1}.npy')
        print(f"Cross validation {i+1} training of {x_train.shape[0]} images")
        x_train =  tf.convert_to_tensor(x_train)
        y_train =  tf.convert_to_tensor(y_train)
        x_val = np.load(f'np_images/xraw5carsaug64_Xv{i+1}.npy')
        y_val = np.load(f'np_images/yraw5carsaug64_Xv{i+1}.npy')
        print(f"Cross validation {i+1} validation of {x_val.shape[0]} images")
        x_val =  tf.convert_to_tensor(x_val)
        y_val =  tf.convert_to_tensor(y_val)

        train_ds = tf.data.Dataset.from_tensor_slices(x_train)
        train_ds = train_ds.repeat()
        train_ds = train_ds.shuffle(1024)
        train_ds = train_ds.map(process, num_parallel_calls=tf.data.AUTOTUNE)
        train_ds = train_ds.batch(BATCH_SIZE)
        train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
        val_ds = tf.data.Dataset.from_tensor_slices(x_val)
        val_ds = val_ds.repeat()
        val_ds = val_ds.shuffle(1024)
        val_ds = val_ds.map(process, num_parallel_calls=tf.data.AUTOTUNE)
        val_ds = val_ds.batch(BATCH_SIZE)
        val_ds = val_ds.prefetch(tf.data.AUTOTUNE)


        backbone = get_backbone(IMG_SIZE)


        predictor = None # Passing None will automatically build the default predictor.
        projector = get_projector(input_dim=backbone.output.shape[-1], dim=DIM, num_layers= N_LAYER)

        model = tfsim.models.create_contrastive_model(
            backbone=backbone,
            projector=projector,
            predictor=predictor,
            algorithm=ALGORITHM,
            name=ALGORITHM,
        )

    if ALGORITHM == "simsiam":
        loss = tfsim.losses.SimSiamLoss(projection_type="cosine_distance", name=ALGORITHM)
        lr_decayed_fn = tf.keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=INIT_LR,
            decay_steps=PRE_TRAIN_EPOCHS * PRE_TRAIN_STEPS_PER_EPOCH,
        )
        wd_decayed_fn = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=WEIGHT_DECAY,
        decay_steps=PRE_TRAIN_EPOCHS * PRE_TRAIN_STEPS_PER_EPOCH,
    )
        optimizer = tfa.optimizers.SGDW(learning_rate=lr_decayed_fn, weight_decay=wd_decayed_fn, momentum=0.9)
    elif ALGORITHM == "barlow":
        loss = tfsim.losses.Barlow(name=ALGORITHM)
        # optimizer = tfa.optimizers.LAMB(learning_rate=INIT_LR)
        optimizer = LARS(learning_rate=INIT_LR, weight_decay_rate= 0.000015)
    elif ALGORITHM == "simclr":
        loss = tfsim.losses.SimCLRLoss(name=ALGORITHM, temperature=TEMPERATURE)
        # optimizer = tfa.optimizers.LAMB(learning_rate=INIT_LR)
        optimizer = LARS(learning_rate=INIT_LR)
    elif ALGORITHM == "vicreg":
        loss = tfsim.losses.VicReg(name=ALGORITHM)
        # optimizer = tfa.optimizers.LAMB(learning_rate=INIT_LR)
        optimizer = LARS(learning_rate=INIT_LR)

    else:
        raise ValueError(f"{ALGORITHM} is not supported.")

    model.compile(
        optimizer=optimizer,
        loss=loss,
    )

    tbc = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir,
        histogram_freq=1,
        update_freq=100,
    )
    mcp = tf.keras.callbacks.ModelCheckpoint(
        filepath=chkpt_dir,
        monitor="val_loss",
        mode="min",
        save_best_only=True,
        save_weights_only=True,
    )

    history = model.fit(
        train_ds,
        epochs=PRE_TRAIN_EPOCHS,
        steps_per_epoch=PRE_TRAIN_STEPS_PER_EPOCH,
        validation_data=val_ds,
        validation_steps=VAL_STEPS_PER_EPOCH,
        callbacks=[tbc, mcp],
        verbose=2,
        
    )

#################### linear evaluation ##############################

    BATCH_SIZE = 256
    TEST_EPOCHS = 90
    TEST_STEPS_PER_EPOCH = len(x_train) // BATCH_SIZE

    eval_train_ds = tf.data.Dataset.from_tensor_slices((x_train, tf.keras.utils.to_categorical(y_train, NUM_CLASSES)))
    eval_train_ds = eval_train_ds.repeat()
    eval_train_ds = eval_train_ds.shuffle(1024)
    eval_train_ds = eval_train_ds.map(lambda x, y: (eval_augmenter(x), y), tf.data.AUTOTUNE)
    eval_train_ds = eval_train_ds.map(lambda x, y: (img_scaling(x), y), tf.data.AUTOTUNE)
    eval_train_ds = eval_train_ds.batch(BATCH_SIZE)
    eval_train_ds = eval_train_ds.prefetch(tf.data.AUTOTUNE)

    eval_val_ds = tf.data.Dataset.from_tensor_slices((x_val, tf.keras.utils.to_categorical(y_val, NUM_CLASSES)))
    eval_val_ds = eval_val_ds.repeat()
    eval_val_ds = eval_val_ds.shuffle(1024)
    eval_val_ds = eval_val_ds.map(lambda x, y: (img_scaling(tf.cast(x, dtype=tf.float32)), y), tf.data.AUTOTUNE)
    eval_val_ds = eval_val_ds.batch(BATCH_SIZE)
    eval_val_ds = eval_val_ds.prefetch(tf.data.AUTOTUNE)

    eval_test_ds = tf.data.Dataset.from_tensor_slices((x_test, tf.keras.utils.to_categorical(y_test, NUM_CLASSES)))
    eval_test_ds = eval_test_ds.map(lambda x, y: (img_scaling(tf.cast(x, dtype=tf.float32)), y), tf.data.AUTOTUNE)
    eval_test_ds = eval_test_ds.batch(BATCH_SIZE)
    eval_test_ds = eval_test_ds.prefetch(tf.data.AUTOTUNE)

    no_pt_eval_model = get_eval_model(
        img_size=IMG_SIZE,
        backbone=get_backbone(IMG_SIZE, DIM),
        total_steps=TEST_EPOCHS * TEST_STEPS_PER_EPOCH,
        trainable=False,

    )

    no_pt_history = no_pt_eval_model.fit(
        eval_train_ds,
        batch_size=BATCH_SIZE,
        epochs=TEST_EPOCHS,                                 
        steps_per_epoch=TEST_STEPS_PER_EPOCH,
        validation_data=eval_val_ds,
        validation_steps=VAL_STEPS_PER_EPOCH,
        verbose=2,
    )



    pt_eval_model = get_eval_model(
        img_size=IMG_SIZE,
        backbone=model.backbone,
        total_steps=TEST_EPOCHS * TEST_STEPS_PER_EPOCH,
        trainable=False,
        algo = ALGORITHM
    )
    # pt_eval_model.summary()
    pt_history = pt_eval_model.fit(
        eval_train_ds,
        batch_size=BATCH_SIZE,
        epochs=TEST_EPOCHS,
        steps_per_epoch=TEST_STEPS_PER_EPOCH,
        validation_data=eval_val_ds,
        validation_steps=VAL_STEPS_PER_EPOCH,
        verbose=2,
    )

    percent = 10 # value in %

    ########3 prepare eval data with 25%
    percent = 10 # value in %
    div = 100 // percent
    x_train_pct = x_train[:int(x_train.shape[0])//div]
    y_train_pct = y_train[:x_train.shape[0]//div]
    x_val_pct = x_val[:x_val.shape[0]//div]
    y_val_pct = y_val[:x_val.shape[0]//div]

    TEST_EPOCHS = 90
    TEST_STEPS_PER_EPOCH = len(x_train_pct) // BATCH_SIZE

    eval_train_ds_pct = tf.data.Dataset.from_tensor_slices((x_train_pct, tf.keras.utils.to_categorical(y_train_pct, NUM_CLASSES)))
    eval_train_ds_pct = eval_train_ds_pct.repeat()
    eval_train_ds_pct = eval_train_ds_pct.shuffle(1024)
    eval_train_ds_pct = eval_train_ds_pct.map(lambda x, y: (eval_augmenter(x), y), tf.data.AUTOTUNE)
    eval_train_ds_pct = eval_train_ds_pct.map(lambda x, y: (img_scaling(x), y), tf.data.AUTOTUNE)
    eval_train_ds_pct = eval_train_ds_pct.batch(BATCH_SIZE)
    eval_train_ds_pct = eval_train_ds_pct.prefetch(tf.data.AUTOTUNE)

    eval_val_ds_pct = tf.data.Dataset.from_tensor_slices((x_val_pct, tf.keras.utils.to_categorical(y_val_pct, NUM_CLASSES)))
    eval_val_ds_pct = eval_val_ds_pct.repeat()
    eval_val_ds_pct = eval_val_ds_pct.shuffle(1024)
    eval_val_ds_pct = eval_val_ds_pct.map(lambda x, y: (img_scaling(tf.cast(x, dtype=tf.float32)), y), tf.data.AUTOTUNE)
    eval_val_ds_pct = eval_val_ds_pct.batch(BATCH_SIZE)
    eval_val_ds_pct = eval_val_ds_pct.prefetch(tf.data.AUTOTUNE)


    # ### Pretrained 25%
    ptpct_eval_model = get_eval_model(
        img_size=IMG_SIZE,
        backbone=model.backbone,
        total_steps=TEST_EPOCHS * TEST_STEPS_PER_EPOCH,
        trainable=False,
        algo= ALGORITHM,

    )
    ptpct_history = ptpct_eval_model.fit(
        eval_train_ds_pct,
        batch_size=BATCH_SIZE,
        epochs=TEST_EPOCHS,
        steps_per_epoch=TEST_STEPS_PER_EPOCH,
        validation_data=eval_val_ds_pct,
        validation_steps=VAL_STEPS_PER_EPOCH,
        verbose=2,
    )



    #### testing 
    no_res.append(no_pt_eval_model.evaluate(eval_test_ds))
    pt_res.append(pt_eval_model.evaluate(eval_test_ds))
    pctpt_res.append(ptpct_eval_model.evaluate(eval_test_ds))

    print("no pretrain", no_res[i])
    print("pretrained", pt_res[i])
    print("10pretrained", pctpt_res[i])

    Histories[f"history_X{i+1}"] = history.history
    pt_Histories[f"history_X{i+1}"] = pt_history.history
    no_pt_Histories[f"history_X{i+1}"] = no_pt_history.history
    ptpct_Histories[f"history_X{i+1}"] = ptpct_history.history
    
    ############# saving best model ################
    if i == 0:
        pt_eval_model.save(DATA_PATH / EXP_NAME / "trained_model")
        # no_pt_eval_model.save(DATA_PATH / EXP_NAME / "trained_model")

    
    #check if this round's accuracy performs better to save it as our best model    
    elif pt_res[i][1] > pt_res[i-1][1]:
    
        pt_eval_model.save(DATA_PATH / EXP_NAME / "trained_model")
        
    #check if this round's accuracy performs better to save it as our best model    
    elif pt_res[i][1] > pt_res[i-1][1]:
    
        no_pt_eval_model.save(DATA_PATH / EXP_NAME / "trained_model")

# saving models and their history data
print("FINAL no pretrain", no_res)
print("FINAL pretrained", pt_res)
print("FINAL 10pretrained", pctpt_res)
#### save base pretrined 
# model.save(DATA_PATH / EXP_NAME / "trained_model")   ####################################################

with open(DATA_PATH / EXP_NAME/ "Histories", 'wb') as file_pi:      ###############################################3
        pickle.dump(Histories, file_pi)


no_pt_eval_model.save(DATA_PATH / EXP_NAME / "cls_eval" / f"NPt_{TEST_EPOCHS}")

with open(DATA_PATH / EXP_NAME / "cls_eval" / f"NPt_e{TEST_EPOCHS}_Histories", 'wb') as file_pi:
    pickle.dump(no_pt_Histories, file_pi)

pt_eval_model.save(DATA_PATH / EXP_NAME / "cls_eval" / f"Pt_{TEST_EPOCHS}")

with open(DATA_PATH / EXP_NAME / "cls_eval" / f"Pt_e{TEST_EPOCHS}_Histories", 'wb') as file_pi:
    pickle.dump(pt_Histories, file_pi)
    
ptpct_eval_model.save(DATA_PATH / EXP_NAME / "cls_eval" / f"Pt{percent}pct_e{TEST_EPOCHS}")

with open(DATA_PATH / EXP_NAME / "cls_eval" / f"Pt{percent}pct_e{TEST_EPOCHS}_Histories", 'wb') as file_pi:
    pickle.dump(ptpct_Histories, file_pi)


# clear any occupied VRAM in the GPU after training
from numba import cuda
device = cuda.get_current_device()
device.reset()




