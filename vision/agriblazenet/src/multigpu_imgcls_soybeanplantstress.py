################################################################################
# AgriBlazeNet Model Training
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
    # tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
  except RuntimeError as e:
    print(e)

# Memory: ImgSize (Mean, Std), BatchSize, Transformer heads & blocks
CONFIG = dict (
    run = 1,
    model_arch_module = 'imgcls_blaze', # model: imgcls_blaze_cbwssam, imgcls_blaze_cbam, imgcls_blaze
    model_arch_type = 'blz', # blz, bcbam, (T1)mvit_h4b4, (T2)cbmvit_h4b4
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

    exp_name='isps_' + str(CONFIG['model_arch_module']) + "_" + str(CONFIG['model_arch_type'])  + "_img_" + str(CONFIG['img_width']) + "_batch_"  + str(CONFIG['batch_size']) + "_run_" + str(CONFIG['run'])
    print("Experiment Name: ", exp_name)
    cm_plot_labels = ['BacterialBlight', 'SeptoriaBrownSpot', 'FrogeyeLeafSpot', 'Healthy', 'HerbicideInjury', 'IronDeficiencyChlorosis', 'PotassiumDeficiency',
                      'BacterialPustule', 'SuddenDeathSyndrome']
    train_dir = Path(CONFIG['data_dirpath']+'train/')
    val_dir = Path(CONFIG['data_dirpath']+'val/')
    test_dir = Path(CONFIG['data_dirpath']+'test/')

    # Basic Config Information
    print("model_architecture, img_size, batch_size, mean & stddev: ", CONFIG['model_arch_module'], " , img=(", CONFIG['img_width'] , " X ", CONFIG['img_height'], "), bs=", CONFIG['batch_size'], " , mean & std=", CONFIG['img_mean'], " & ", CONFIG['img_std'])

    # Image Augmentation
    train_data_augmentation = tf.keras.Sequential([layers.RandomFlip("horizontal_and_vertical"),
                                            layers.RandomRotation(factor=0.5),
                                            layers.RandomContrast(factor=0.2),
                                            layers.RandomBrightness(factor=0.2),])

    # Model Artefacts
    model_artefact_dir = str(CONFIG['model_logdir'])
    model_log_dir = Path(model_artefact_dir).mkdir(parents=True, exist_ok=True)
    plot_filename = model_artefact_dir + "Acc_Loss_{}.png".format(exp_name)
    logfile = model_artefact_dir + "Log_{}.csv".format(exp_name)
    log_dir = model_artefact_dir + "logs/{}".format(exp_name)
    plot_valcm_filename = model_artefact_dir + "Val_CM_{}.png".format(exp_name)
    plot_valroc_filename = model_artefact_dir + "Val_ROC_{}.png".format(exp_name)
    plot_valroconly_filename = model_artefact_dir + "Val_ROC_only_{}.png".format(exp_name)
    plot_testcm_filename = model_artefact_dir + "Test_CM_{}.png".format(exp_name)
    plot_testroc_filename = model_artefact_dir + "Test_ROC_{}.png".format(exp_name)
    plot_testroconly_filename = model_artefact_dir + "Test_ROC_only_{}.png".format(exp_name)

    # Model weight in keras format
    modelweight_dirpath = str(CONFIG['modelweight_parentdir'])
    modelweight_dir = Path(modelweight_dirpath).mkdir(parents=True, exist_ok=True)
    weight_filepath = modelweight_dirpath + "{}.keras".format(exp_name)

    kerasweight_dirpath = str(CONFIG['kerasweight_parentdir'])
    kerasweight_dir = Path(kerasweight_dirpath).mkdir(parents=True, exist_ok=True)
    kerasweight_filepath = kerasweight_dirpath + "{}.keras".format(exp_name)

    # Logging
    csv_logger = CSVLogger(logfile, append=True, separator=';')
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

    # Multi-GPU Strategy
    strategy = tf.distribute.MirroredStrategy()
    print('Number of GPUs in workstation: {}'.format(strategy.num_replicas_in_sync))
    mgpu_batch_size = int(CONFIG['batch_size']) * strategy.num_replicas_in_sync
    print("Multi-GPU Batch Size: ", mgpu_batch_size)

    # Datasets
    train_ds = tf.keras.utils.image_dataset_from_directory(train_dir, image_size=(CONFIG['img_height'], CONFIG['img_width']), label_mode = CONFIG['label_mode'],
                                                           batch_size=mgpu_batch_size, crop_to_aspect_ratio=True)
    val_ds = tf.keras.utils.image_dataset_from_directory(val_dir, image_size=(CONFIG['img_height'], CONFIG['img_width']), label_mode = CONFIG['label_mode'],
                                                         batch_size=mgpu_batch_size, crop_to_aspect_ratio=True)
    test_ds = tf.keras.utils.image_dataset_from_directory(test_dir, image_size=(CONFIG['img_height'], CONFIG['img_width']), label_mode = CONFIG['label_mode'],
                                                          batch_size=mgpu_batch_size, crop_to_aspect_ratio=True)
    labels = train_ds.class_names
    test_labels = test_ds.class_names
    print("Labels from Train datasets: ", labels)
    print("Labels from Test datasets: ", test_labels)

    # Image augmentation
    train_ds = augment_dataset(ds=train_ds, data_augmentation=train_data_augmentation, augment=True)
    val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)
    test_ds = test_ds.prefetch(buffer_size=AUTOTUNE)
    print("Dataset Cardinality(train, val, test): ", train_ds.cardinality(), " , ", val_ds.cardinality(), " , ", test_ds.cardinality())

    # Other Params
    model_save_checkpoint = ModelCheckpoint(weight_filepath, save_best_only=True, save_weights_only=False, verbose=1)
    early_stop = EarlyStopping(monitor='val_loss', patience=10, mode='min', min_delta=CONFIG['min_delta'], verbose=1, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=CONFIG['reduce_lr_factor'], patience=2, min_delta=CONFIG['min_delta'], mode='min', verbose=1)

    # Model MGPU-Strategy Scope
    with strategy.scope():
        model = sis_imgcls_model(model_arch_module=CONFIG['model_arch_module'], input_shape=(CONFIG['img_height'], CONFIG['img_width'], CONFIG['channels']),
                             no_classes=CONFIG['num_labels'], img_mean=CONFIG['img_mean'], img_std=CONFIG['img_std'])

        print("Model Summary:")
        model.build(input_shape=(None,CONFIG['img_height'], CONFIG['img_width'], CONFIG['channels']))
        print(model.summary())
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=CONFIG['learning_rate']), loss='sparse_categorical_crossentropy', metrics=['sparse_categorical_accuracy'])

    # Model Fit
    history = model.fit(train_ds, validation_data = val_ds, epochs = CONFIG['epochs'], verbose = 1, batch_size = mgpu_batch_size,
                        callbacks = [model_save_checkpoint, early_stop, reduce_lr, csv_logger, tensorboard_callback])

    # Model save Keras weight
    model.save(kerasweight_filepath)

    # Plot Metrics
    plot_acc_loss(history, plot_filename = plot_filename, title_type = 'trainval')

    val_score = model.evaluate(val_ds, batch_size=mgpu_batch_size, verbose="auto")
    print(f'Val loss: {val_score[0]} , Val accuracy: {val_score[1]}')
    for name, value in zip(model.metrics_names, val_score):
        print('Val dataset evaluation: ', name, ': ', value)

    test_score = model.evaluate(test_ds, batch_size=mgpu_batch_size, verbose="auto", return_dict=True)
    print("Testset Evaluation Score: ", test_score)
    '''
    print(f'Test loss: {test_score[0]} , Test accuracy: {test_score[1]}')
    for name, value in zip(model.metrics_names, test_score):
        print('Test dataset evaluation: ', name, ': ', value)
    '''
    print('\n-------------------Evaluate on Test Dataset-------------------\n')
    true_labels_test, raw_preds_test, thres_preds_test = evaluate_ds(test_ds, model, batch_size=mgpu_batch_size)

    print(" Get Confusion Matrix on Test dataset ! ")
    cm_test = confusion_matrix(y_true=true_labels_test, y_pred=thres_preds_test)
    print(" Save Confusion Matrix on Test dataset ! ")
    plot_cm(cm=cm_test, classes=cm_plot_labels, title='Test CM', filename=plot_testcm_filename)

    # Test dataset
    print('\n-------------------Test dataset-------------------\n')
    print("Micro on Test dataset:", )
    print("Precision score: ", precision_score(y_true=true_labels_test, y_pred=thres_preds_test, average='micro', zero_division=0))
    print("Recall score: ", recall_score(y_true=true_labels_test, y_pred=thres_preds_test, average='micro', zero_division=0))
    print("F1 score: ", f1_score(y_true=true_labels_test, y_pred=thres_preds_test, average='micro', zero_division=0))
    print("------------------------------------------------------------------------")
    print("Weighted on Test dataset:", )
    print("Precision score: ", precision_score(y_true=true_labels_test, y_pred=thres_preds_test, average='weighted', zero_division=0))
    print("Recall score: ", recall_score(y_true=true_labels_test, y_pred=thres_preds_test, average='weighted', zero_division=0))
    print("F1 score: ", f1_score(y_true=true_labels_test, y_pred=thres_preds_test, average='weighted', zero_division=0))
    print("------------------------------------------------------------------------")
    print("Macro on Test dataset:", )
    print("Precision score: ", precision_score(y_true=true_labels_test, y_pred=thres_preds_test, average='macro', zero_division=0))
    print("Recall score: ", recall_score(y_true=true_labels_test, y_pred=thres_preds_test, average='macro', zero_division=0))
    print("F1 score: ", f1_score(y_true=true_labels_test, y_pred=thres_preds_test, average='macro', zero_division=0))
    print("------------------------------------------------------------------------")
    print("Test dataset Classification Report:\n")
    print(classification_report(y_true=true_labels_test, y_pred=thres_preds_test, target_names=test_labels, zero_division=0))
    print("\n")
    print(" Save ROC plot on Test dataset ! ")
    plot_roc(true_labels_test, thres_preds_test, filename=plot_testroc_filename, roconly_filename = plot_testroconly_filename,
             n_class=CONFIG['num_labels'], cm_plot_labels = cm_plot_labels, dataset='test')
    print("------------------------------------------------------------------------")
    # Calculate Inference Time
    N_warmup_run = 5
    N_run = 100
    elapsed_time = []
    for i in range(N_warmup_run):
        preds = model.predict(test_ds, batch_size=mgpu_batch_size)
    for i in range(N_run):
        start_time = time.time()
        preds = model.predict(test_ds, batch_size=mgpu_batch_size)
        end_time = time.time()
        elapsed_time = np.append(elapsed_time, end_time - start_time)
        if i % 10 == 0:
            print('Step {}: {:5.1f}ms'.format(i, (elapsed_time[-10:].mean()) * 1000))
    print('Throughput: {:.0f} images in {:7.1f} sec'.format(N_run * mgpu_batch_size, elapsed_time.sum()))
    print('Throughput: {:.0f} images/s'.format(N_run * mgpu_batch_size / elapsed_time.sum()))
    del model, train_ds, val_ds, test_ds
    _ = gc.collect()
