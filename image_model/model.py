import tensorflow as tf
from PIL import Image
import numpy as np
import os
import sys
import time
import numpy as np
from keras.models import load_model
import argparse

# Navigate to correct position in filesystem
script_directory = os.path.dirname(os.path.abspath(sys.argv[0]))
os.chdir(script_directory)

# Set up the model
def predict(model_path, label_file, image):
  INPUT_IMG_SIZE = 224
  INPUT_IMG_SHAPE = (224, 224, 3)
  if model_path == "mobilenet":
    model = tf.keras.applications.MobileNetV2(
  input_shape=INPUT_IMG_SHAPE
  )
  else:
    model =load_model(model_path)
  @tf.function
  def serve(x):
    return model(x, training=False)


  # Prepare and pass the input image
  image_path = image 
  img = Image.open(image_path).convert('RGB')
  img = img.resize((INPUT_IMG_SIZE, INPUT_IMG_SIZE), Image.BICUBIC)
  input_data = np.array(img)/255.0
  input_data = input_data.reshape(1, INPUT_IMG_SIZE, INPUT_IMG_SIZE, 3)

  # First prediction is slow, we won't count it
  output = serve(input_data)

  # Now, start counting!
  start_time = time.time()

  # Make a prediction!
  output = serve(input_data)

  # Get and print the result
  inf_time =  time.time() - start_time 
  print(f"time: {inf_time}s" )

  top_3 = np.argsort(output.numpy().squeeze())[-3:][::-1]
  if label_file == "mobilenet" :
    
        url = tf.keras.utils.get_file(
            'ImageNetLabels.txt',
            'https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt')
        imagenet_labels = np.array(open(url).read().splitlines())[1:]
  else:
        with open(label_file, 'r') as f:
            imagenet_labels = [line.strip() for line in f.readlines()]

  for i in top_3:
      print('{:.6f}'.format(output.numpy()[0, i]), ':',  imagenet_labels[i])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="To parse model name, labelfile and image")
    parser.add_argument("--model_name",dest="model_name", type=str, default="mobilenet", help="Name of model file.")
    parser.add_argument("--labelfile_name", dest="labelfile_name",type=str, default="mobilenet", help="Name of label file.")
    parser.add_argument("--image_name",dest="image_name", type=str, default="parrot.jpg", help="Name of image file.")

    
    args = parser.parse_args()
    predict(args.model_name, args.labelfile_name, args.image_name)
