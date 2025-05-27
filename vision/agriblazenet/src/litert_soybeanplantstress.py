################################################################################
# LiteRT: AgriBlazeNet
################################################################################
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
tf.keras.backend.clear_session()

import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import time, glob, gc
from sisutils.utils import augment_dataset, plot_acc_loss, evaluate_ds, plot_cm, plot_roc
from sisutils.sis_model import sis_imgcls_model

SEED_VAL=np.random.randint(100)
print("TensorFlow Version and seed: ", tf.__version__, " and ", SEED_VAL)

CONFIG = dict (
    run = 1,
    quant_method = 'dynamic_range',
    num_labels = 9, # No. of Classes
    img_width = 128, # img: 128 x 128
    img_height = 128,
    channels = 3,
    batch_size = 4,
    learning_rate = 0.001,
    img_mean = [0.257, 0.321, 0.23], # Mean: 128- [0.257, 0.321, 0.23]
    img_std = [0.115, 0.149, 0.087], # Std: 128- [0.115, 0.149, 0.087]
    label_mode = 'int',
    model_arch_type_list = ['blz'], # Model Name: ['blz'], ['bcbam'], ['mvit_h4b4'], ['cbmvit_h4b4']
    model_arch_module = 'imgcls_blaze', # Arch. Module Name: imgcls_blaze_cbwssam, imgcls_blaze_cbam, imgcls_blaze
    test_dir_path = '../../../share/data/plantscience/imgcls/plantstress/test/',
    modelweight_dir = 'weight/keras/ips128/blz/',
    tfl_model_dir = 'weight/litert/liteips128/',
    tfl_ext = '.tflite',
)

tf.keras.utils.set_random_seed(SEED_VAL)
tf.config.experimental.enable_op_determinism()
AUTOTUNE = tf.data.experimental.AUTOTUNE

def get_test_dataset(test_dir, image_size, label_mode, batch_size):
  test_ds = tf.keras.utils.image_dataset_from_directory(directory=test_dir, image_size=image_size, label_mode=label_mode,
                                                                batch_size=batch_size, crop_to_aspect_ratio=True)
  print("test_ds: ", len(test_ds))
  test_labels = test_ds.class_names
  print("Labels from Test datasets: ", test_labels)
  no_test_images = tf.data.experimental.cardinality(test_ds).numpy()
  print("No. of Testing Images: " + str(no_test_images))
  test_ds = test_ds.cache()
  test_ds = test_ds.prefetch(buffer_size=AUTOTUNE)
  return test_ds

def get_test_images_labels(test_ds):
  test_images = []
  test_labels = []
  # Loop through the test dataset using test_ds.take(len(test_ds)) and unbatch it to extract individual images and labels.
  for image, label in test_ds.take(len(test_ds)).unbatch():
    test_images.append(image)
    test_labels.append(label)
  print("test_images, test_labels: ", len(test_images), len(test_labels))
  return test_images, test_labels

# A helper function to evaluate the TF Lite model using "test" dataset.
def evaluate_model(interpreter, test_images):
  input_index = interpreter.get_input_details()[0]["index"]
  output_index = interpreter.get_output_details()[0]["index"]
  # Run predictions on every image in the "test" dataset.
  prediction_classes = []
  for test_image in test_images:
    # Pre-processing: add batch dimension and convert to float32 to match with the model's input data format.
    test_image = np.expand_dims(test_image, axis=0).astype(np.float32)
    interpreter.set_tensor(input_index, test_image)
    # Run inference.
    interpreter.invoke()
    # Post-processing: remove batch dimension and find the digit with highest probability.
    output = interpreter.tensor(output_index)
    digit = np.argmax(output()[0])
    prediction_classes.append(digit)
  # Compare prediction results with ground truth labels to calculate accuracy.
  accurate_count = 0
  for index in range(len(prediction_classes)):
    if prediction_classes[index] == test_labels[index]:
      accurate_count += 1
  accuracy = accurate_count * 1.0 / len(prediction_classes)
  return accuracy

if __name__ == '__main__':

  tf_modelweight_list = glob.glob(CONFIG['modelweight_dir']+"/*.keras")
  tfl_model_dir = Path(CONFIG['tfl_model_dir']).mkdir(parents=True, exist_ok=True)
  test_ds = get_test_dataset(test_dir=CONFIG['test_dir_path'], image_size=(CONFIG['img_height'], CONFIG['img_width']),
                             label_mode = CONFIG['label_mode'], batch_size=CONFIG['batch_size'])
  test_images, test_labels = get_test_images_labels(test_ds)

  for model_architecture in CONFIG['model_arch_type_list']:
    filtered_tf_modelweight_list = [Path(weight_filename) for weight_filename in tf_modelweight_list if str(model_architecture) in str(weight_filename)]

  print(model_architecture, " - LiteRT list: ", filtered_tf_modelweight_list)
  keras_filename_list = [weight_file.stem for weight_file in filtered_tf_modelweight_list]
  keras_filename_list.sort()
  keras_weightfile_list = [weight_file for weight_file in filtered_tf_modelweight_list]
  keras_weightfile_list.sort()
  print("Keras Weight filename List: ", keras_weightfile_list)
  tflite_filename_list = [Path(CONFIG['tfl_model_dir'] + elem + CONFIG['tfl_ext']) for elem in keras_filename_list]
  tflite_filename_list.sort()
  print("LiteRT Weight Filename List: ", tflite_filename_list)

  for keras_filename, tfl_filename in zip(keras_weightfile_list, tflite_filename_list):
     # load model
     model = sis_imgcls_model(model_arch_module=CONFIG['model_arch_module'], input_shape=(CONFIG['img_height'], CONFIG['img_width'], CONFIG['channels']),
                              no_classes=CONFIG['num_labels'], img_mean=CONFIG['img_mean'], img_std=CONFIG['img_std'])

     model = tf.keras.models.load_model(keras_filename)
     model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=CONFIG['learning_rate']), loss='sparse_categorical_crossentropy', metrics=['sparse_categorical_accuracy'])

     # Evaluating the model on the test dataset.
     model_loss, model_accuracy = model.evaluate(test_ds, batch_size=CONFIG['batch_size'], verbose=0)
     print('Test Loss: ', keras_filename.stem,' = ', np.around(model_loss, decimals=2))
     print('Test Accuracy: ', keras_filename.stem,' = ', np.around(model_accuracy, decimals=2))

     # select quantization technique: fp16, dynamic_range, integer
     if CONFIG['quant_method'] == 'dynamic_range':
       # Convert to a TensorFlow Lite model and quantize the model on export
       converter = tf.lite.TFLiteConverter.from_keras_model(model)
       converter.optimizations = [tf.lite.Optimize.DEFAULT]
       tflite_quant_model = converter.convert()

       # Save TF-Lite weights
       open(tfl_filename, "wb").write(tflite_quant_model)
       print("Save Dynamic Range Quantization (DRQ) weight for Keras filename: ", keras_filename, " as LiteRT filename ", tfl_filename)
       # Run the TFLite models
       dr_interpreter = tf.lite.Interpreter(model_path=str(tfl_filename))
       dr_interpreter.allocate_tensors()

       # Evaluating the model on the test images.
       dr_test_accuracy = evaluate_model(dr_interpreter, test_images)
       # Printing the test accuracy for the Dynamically Quantized TFLite model and the baseline Keras model.
       print('Post-training DRQ Test Accuracy: ', np.around(dr_test_accuracy, decimals=2))
       time.sleep(3)
       del model, converter, tflite_quant_model, dr_interpreter
       time.sleep(2)
     else:
       print("Verify quantization method! ")

  print("---------------------------------------------------------------")
  print("Quantization done for the referred model !")
  _ = gc.collect()
