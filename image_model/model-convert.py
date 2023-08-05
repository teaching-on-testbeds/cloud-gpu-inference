import tensorflow as tf
import numpy as np
import os, sys
import tensorrt
assert tensorrt.Builder(tensorrt.Logger())
from tensorflow.python.compiler.tensorrt import trt_convert as trt
from keras.models import load_model

# Navigate to correct position in filesystem
script_directory = os.path.dirname(os.path.abspath(sys.argv[0]))
os.chdir(script_directory)

# Set up the model
def model_convert(model_path)
  INPUT_IMG_SIZE = 224
  INPUT_IMG_SHAPE = (224, 224, 3)

  if model_path == "mobilenet":
    model = tf.keras.applications.MobileNetV2(
  input_shape=INPUT_IMG_SHAPE
  )
  else:
    model =load_model(model_path)
  
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
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="To parse model name")
    parser.add_argument("model_name", type=str, default="mobilenet", help="Name of model file.")

    
    args = parser.parse_args()
    model_convert(args.model_name)
