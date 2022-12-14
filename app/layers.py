# Custom L1 Distance layer module
# Import dependencies
import tensorflow as tf
from tensorflow.keras.layers import Layer

# Custom L1 distance layer 
# Siamese L1 Distance class its needed to load the custom model
class L1Dist(Layer):
  # Init method - inheritance
  def __init__(self, **kwargs):
    super().__init__()
  
  # Magic happens here - similarity calculation
  def call(self, input_embedding, validation_embedding):
    return tf.math.abs(input_embedding - validation_embedding)