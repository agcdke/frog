################################################################################
# Evaluation on Test dataset
################################################################################
import os
# os.environ["SM_FRAMEWORK"] = "tf.keras"
import random, gc, time
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import tensorflow as tf
tf.keras.backend.clear_session()
from tensorflow.keras.callbacks import (CSVLogger, TensorBoard, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau)
from tensorflow.data.experimental import (cardinality)
from sisutils.classdict_cwfid_dataset import CwfidDataset
from sisutils.sis_model import sis_unet_model
from sisutils.seg_class_loss_metrics import *

# Seed and GPUs
SEED_VAL=np.random.randint(100)
print("TensorFlow Version and seed: ", tf.__version__, " and ", SEED_VAL)

gpus = tf.config.list_physical_devices('GPU')
if gpus:
  try:
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    #tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
  except RuntimeError as e:
    print(e)

config_params = {
        'run': 1,
        'model_arch_module': 'unet_blaze_cbwssam', # model: unet_blaze, unet_blaze_cbam, unet_blaze_cbwssam
        'model_arch_type': 'T1H8B8', # unet_blaze: {blz}/ unet_blaze_cbam: {bcbam}/ unet_blaze_cbwssam: {(T1H8B8)-mvit_h8b8, (T2H8B8)-cbmvit_h8b8}
        'cbam_block': 'mvit_h8b8',
        'attn_block': 'cbwssam',
        'loss_fn': 'unet3p_hybrid', # unet3p_hybrid_loss, basnet_hybrid_loss
        'data_dir': f'../../../share/data/plantscience/semseg/cwfid/',
        'yaml_file': f'cwfid_train_test_split.yaml',
        'class_dict_file': f'cwfid_class_dict.csv',
        'img_h': 128,
        'img_w': 128,
        'batch_size': 8,
        'num_classes': 3, # black = background, red = weed, green = crop
        'n_channels': 3,
        'shuffle': False,
        'buffer_size': 30,
        'repeat_size': 10,
        'es_patience': 25,
        'rdlr_patience': 10,
        'epochs': 200,
        'learning_rate': 1e-3,
        'min_delta': 1e-4,
        'reduce_lr_factor': 0.5,
        'min_lr': 1e-7,
        'tversky_alpha': 0.7,
        'tversky_beta': 0.7,
        'modelweight_parentdir': f'weights/semseg/cwfid_classwise_256/model/',
        'kerasweight_parentdir': f'weights/semseg/cwfid_classwise_256/lastmodel/',
        'model_logdir': f'metricinfo/semseg/cwfid_classwise_256/',
    }
# TF Deterministic Approach
tf.keras.utils.set_random_seed(SEED_VAL)
tf.config.experimental.enable_op_determinism()
AUTOTUNE = tf.data.experimental.AUTOTUNE

if __name__ == '__main__':

    exp_name='cwfid_' + str(config_params['model_arch_module']) + "_" + str(config_params['model_arch_type']) + "_loss_" + str(config_params['loss_fn']) + "_img_" + str(config_params['img_w']) + "_batch_"  + str(config_params['batch_size']) + "_run_" + str(config_params['run'])
    print("Experiment Name: ", exp_name)
    # TensorBoard, and Model weight in keras format
    model_logdirpath = str(config_params['model_logdir'])
    model_logdir = Path(model_logdirpath).mkdir(parents=True, exist_ok=True)
    log_dir = model_logdirpath + "logs/{}".format(exp_name)

    modelweight_dirpath = str(config_params['modelweight_parentdir'])
    weight_filepath = modelweight_dirpath + "{}.keras".format(exp_name)

    kerasweight_dirpath = str(config_params['kerasweight_parentdir'])
    kerasweight_filepath = kerasweight_dirpath + "{}.keras".format(exp_name)

    # Dataset
    cwfid_dataset = CwfidDataset(config_params)
    train_dataset, val_dataset, test_dataset = cwfid_dataset.get_cwfid_dataset(sanity_checker=False)
    print("Cardinality(train, Val, test): ", cardinality(train_dataset).numpy(),
                                            cardinality(val_dataset).numpy(),
                                            cardinality(test_dataset).numpy())

    final_train_dataset, final_val_dataset, final_test_dataset = cwfid_dataset.get_cwfid_dataset_with_imgaug()
    print("Aug-Cardinality(train, Val, test): ", cardinality(final_train_dataset).numpy(),
                                            cardinality(final_val_dataset).numpy(),
                                            cardinality(final_test_dataset).numpy())

    # Model
 
    model = tf.keras.models.load_model(filepath=weight_filepath)
    # custom_objects={'num_classes': config_params['num_classes'], 'attn_block': config_params['attn_block'], 'cbam_block':config_params['cbam_block']})
    

    test_preds = model.predict(final_test_dataset, batch_size=config_params['batch_size'])
    print("Testset Prediction Info: ", len(test_preds), test_preds[0].shape) # 20, (128,128,3)
    test_evals = model.evaluate(final_test_dataset, batch_size=config_params['batch_size'], return_dict=True)
    print("Testset Evaluation Score: ", test_evals)

    # Calculate Inference Time
    N_warmup_run = 5
    N_run = 100
    elapsed_time = []
    for i in range(N_warmup_run):
        preds = model.predict(final_test_dataset, batch_size=config_params['batch_size'])
    for i in range(N_run):
        start_time = time.time()
        preds = model.predict(final_test_dataset, batch_size=config_params['batch_size'])
        end_time = time.time()
        elapsed_time = np.append(elapsed_time, end_time - start_time)
        if i % 10 == 0:
            print('Step {}: {:5.1f}ms'.format(i, (elapsed_time[-10:].mean()) * 1000))
    print('Throughput: {:.0f} images in {:7.1f} sec'.format(N_run * config_params['batch_size'], elapsed_time.sum()))
    print('Throughput: {:.0f} images/s'.format(N_run * config_params['batch_size'] / elapsed_time.sum()))
    
    del model, final_train_dataset, final_val_dataset, final_test_dataset, train_dataset, val_dataset, test_dataset
    _ = gc.collect()
