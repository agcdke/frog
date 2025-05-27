################################################################################
# Evaluation on Test dataset
################################################################################
import os
import tensorflow as tf
tf.keras.backend.clear_session()
from tensorflow.keras.callbacks import (CSVLogger, TensorBoard, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau)
from tensorflow.keras import layers
from tensorflow.keras.utils import plot_model
from pathlib import Path
import numpy as np
from sklearn.metrics import (confusion_matrix, f1_score, precision_score, recall_score, classification_report)
import datetime, gc, time
import pandas as pd
from sisutils.utils import augment_dataset, plot_acc_loss, evaluate_ds, plot_cm, plot_roc
from sisutils.sis_model import sis_imgcls_model

SEED_VAL=np.random.randint(100)
print("TensorFlow Version and seed: ", tf.__version__, " and ", SEED_VAL)

# Dir. Structure - "Train-Val-Test"
gpus = tf.config.list_physical_devices('GPU')
if gpus:
  try:
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
  except RuntimeError as e:
    print(e)

# Memory: ImgSize (Mean, Std), BatchSize, Transformer heads & blocks
CONFIG = dict (
    run = 1,
    model_arch_module = 'imgcls_blaze_cbwssam', # model: imgcls_blaze_cbwssam, imgcls_blaze_cbam, imgcls_blaze
    model_arch_type = 'mvit_h4b4', # blz, bcbam, (T1)mvit_h4b4, (T2)cbmvit_h4b4
    num_labels = 9, # No. of Classes
    img_width = 128, # img: 128 x 128
    img_height = 128,
    channels = 3,
    batch_size = 32, # batch: 64 x 2 GPU = 128
    epochs = 50,
    learning_rate = 0.001,
    min_delta = 0.0001,
    reduce_lr_factor = 0.2,
    img_mean = [0.257, 0.321, 0.23], # Mean: 128- [0.257, 0.321, 0.23]
    img_std = [0.115, 0.149, 0.087], # Std: 128- [0.115, 0.149, 0.087]
    color_mode = 'rgb',
    label_mode = 'int',
    data_dirpath = '../../../share/data/plantscience/imgcls/plantstress/',
    modelweight_parentdir = 'weight/imgcls/mgpu_ips128/',
    kerasweight_parentdir = 'weight/keras/ips128/',
    model_logdir = 'metricinfo/mgpu_iowaplantstress/',
)

tf.keras.utils.set_random_seed(SEED_VAL)
tf.config.experimental.enable_op_determinism()
AUTOTUNE = tf.data.experimental.AUTOTUNE

if __name__ == '__main__':

    exp_name='msps_' + str(CONFIG['model_arch_module']) + "_" + str(CONFIG['model_arch_type'])  + "_img_" + str(CONFIG['img_width']) + "_batch_"  + str(CONFIG['batch_size']) + "_run_" + str(CONFIG['run'])
    print("Experiment Name: ", exp_name)
    cm_plot_labels = ['BacterialBlight', 'SeptoriaBrownSpot', 'FrogeyeLeafSpot', 'Healthy', 'HerbicideInjury', 'IronDeficiencyChlorosis', 'PotassiumDeficiency',
                      'BacterialPustule', 'SuddenDeathSyndrome']

    # Datasets
    test_dir = Path(CONFIG['data_dirpath']+'test/')
    test_ds = tf.keras.utils.image_dataset_from_directory(test_dir, image_size=(CONFIG['img_height'], CONFIG['img_width']), label_mode = CONFIG['label_mode'],
                                                          batch_size=CONFIG['batch_size'], crop_to_aspect_ratio=True)

    # Basic Config Information
    print("model_architecture, img_size, batch_size, mean & stddev: ", CONFIG['model_arch_module'], " , img=(", CONFIG['img_width'] , " X ", CONFIG['img_height'], "), bs=",
          CONFIG['batch_size'], " , mean & std=", CONFIG['img_mean'], " & ", CONFIG['img_std'])

    # Model Artefacts
    model_artefact_dir = str(CONFIG['model_logdir'])
    model_log_dir = Path(model_artefact_dir).mkdir(parents=True, exist_ok=True)
    logfile = model_artefact_dir + "Log_{}.csv".format(exp_name)
    log_dir = model_artefact_dir + "logs/{}".format(exp_name)

    # Model weight in keras format
    modelweight_dirpath = str(CONFIG['modelweight_parentdir'])
    modelweight_dir = Path(modelweight_dirpath).mkdir(parents=True, exist_ok=True)
    weight_filepath = modelweight_dirpath + "{}.keras".format(exp_name)

    kerasweight_dirpath = str(CONFIG['kerasweight_parentdir'])
    kerasweight_dir = Path(kerasweight_dirpath).mkdir(parents=True, exist_ok=True)
    kerasweight_filepath = kerasweight_dirpath + "{}.keras".format(exp_name)

    # Load Model weight
    model = tf.keras.models.load_model(filepath=weight_filepath)

    test_score = model.evaluate(test_ds, batch_size=CONFIG['batch_size'], verbose="auto", return_dict=True)
    print("Testset Evaluation Score: ", test_score)
    del model, test_ds
    _ = gc.collect()
