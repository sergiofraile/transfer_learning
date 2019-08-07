import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]="3" 
import constants as ct
import plotter as pt
import tensorflow as tf
# tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
from loading_pre_trained_model import prepareBaseModel
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras import layers
from tensorflow.keras import Model

# Training and validation directories
train_dir = os.path.join( ct.BASE_DIR, 'train')
validation_dir = os.path.join( ct.BASE_DIR, 'validation')

train_cats_dir = os.path.join(train_dir, 'cats') # Directory with our training cat pictures
train_dogs_dir = os.path.join(train_dir, 'dogs') # Directory with our training dog pictures
validation_cats_dir = os.path.join(validation_dir, 'cats') # Directory with our validation cat pictures
validation_dogs_dir = os.path.join(validation_dir, 'dogs')# Directory with our validation dog pictures

train_cat_fnames = os.listdir(train_cats_dir)
train_dog_fnames = os.listdir(train_dogs_dir)

# Pre trained model
pre_trained_model = prepareBaseModel()

# Last layer and output of the pre trained model
last_layer = pre_trained_model.get_layer('mixed7')
# print('last layer output shape: ', last_layer.output_shape)
last_output = last_layer.output

# Flatten the output layer to 1 dimension
x = layers.Flatten()(last_output)
# Add a fully connected layer with 1,024 hidden units and ReLU activation
x = layers.Dense(1024, activation='relu')(x)
# Add a dropout rate of 0.2
x = layers.Dropout(0.2)(x)                  
# Add a final sigmoid layer for classification
x = layers.Dense  (1, activation='sigmoid')(x)           

model = Model( pre_trained_model.input, x) 

model.compile(optimizer = RMSprop(lr=ct.LEARNING_RATE), 
              loss = 'binary_crossentropy', 
              metrics = ['acc'])

# Add our data-augmentation parameters to ImageDataGenerator
train_datagen = ImageDataGenerator(rescale = 1./255.,
                                   rotation_range = ct.ROTATION_RANE,
                                   width_shift_range = ct.WIDTH_SHIFT_RANGE,
                                   height_shift_range = ct.HEIGHT_SHIFT_RANGE,
                                   shear_range = ct.SHEAR_RANGE,
                                   zoom_range = ct.ZOOM_RANGE,
                                   horizontal_flip = ct.HORIZONTAL_FLIP)

# Note that the validation data should not be augmented!
test_datagen = ImageDataGenerator( rescale = 1.0/255. )

# Flow training images in batches of 20 using train_datagen generator
train_generator = train_datagen.flow_from_directory(train_dir,
                                                    batch_size = ct.BATCH_SIZE,
                                                    class_mode = 'binary', 
                                                    target_size = (ct.INPUT_SHAPE, ct.INPUT_SHAPE))     

# Flow validation images in batches of 20 using test_datagen generator
validation_generator =  test_datagen.flow_from_directory( validation_dir,
                                                          batch_size  = ct.BATCH_SIZE,
                                                          class_mode  = 'binary', 
                                                          target_size = (ct.INPUT_SHAPE, ct.INPUT_SHAPE))

history = model.fit_generator(
  train_generator,
  validation_data = validation_generator,
  steps_per_epoch = ct.STEPS_PER_EPOCH,
  epochs = ct.EPOCHS,
  validation_steps = ct.VALIDATION_STEPS,
  verbose = 2)                                                        

pt.plotHistory(history)
