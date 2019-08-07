import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]="3" 
import constants as ct
from tensorflow.keras.applications.inception_v3 import InceptionV3

def prepareBaseModel():
  # Load the file with the network weights
  local_weights_file = ct.INCEPTION_V3_WEIGHTS
  # Create a network (Inception_v3)
  pre_trained_model = InceptionV3(
    input_shape = (ct.INPUT_SHAPE, ct.INPUT_SHAPE, 3), 
    include_top = False, 
    weights = None)
  # Load the weights into the network
  pre_trained_model.load_weights(local_weights_file)
  # Lock the layers for training
  for layer in pre_trained_model.layers:
    layer.trainable = False
  pre_trained_model.summary()

  return pre_trained_model

