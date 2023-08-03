import tensorflow as tf
import numpy as np
import os, sys
import tensorrt
assert tensorrt.Builder(tensorrt.Logger())
from tensorflow.python.compiler.tensorrt import trt_convert as trt

# Navigate to correct position in filesystem
script_directory = os.path.dirname(os.path.abspath(sys.argv[0]))
os.chdir(script_directory)

# Set up the model
INPUT_IMG_SIZE = 224
INPUT_IMG_SHAPE = (224, 224, 3)
model = tf.keras.applications.MobileNetV2(
  input_shape=INPUT_IMG_SHAPE
)

SAVED_MODEL_DIR="./original_models"
model.save(SAVED_MODEL_DIR)

converter = trt.TrtGraphConverterV2(
   input_saved_model_dir=SAVED_MODEL_DIR,
   precision_mode=trt.TrtPrecisionMode.FP16
)

trt_func = converter.convert()
converter.summary()

data_float32 = np.zeros((1, INPUT_IMG_SIZE, INPUT_IMG_SIZE, 3), dtype=np.float32)
def input_fn():
    yield [data_float32]

converter.build(input_fn=input_fn)
     
OUTPUT_SAVED_MODEL_DIR="./optimized_models"
converter.save(output_saved_model_dir=OUTPUT_SAVED_MODEL_DIR)
