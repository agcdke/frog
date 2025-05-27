################################################################################
# Utility Function
# Acknowledgement: wandb.ai for evaluation and plotting metrics info utility functions.
################################################################################

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import sklearn, itertools, time
from sklearn.metrics import roc_curve, auc
from itertools import cycle
from tensorflow.keras.applications import (MobileNetV2)
from tensorflow.keras.layers import (Input, GlobalAveragePooling2D, GlobalMaxPooling2D, Reshape, Dense, Multiply, Flatten, MaxPooling2D, AveragePooling2D, Dropout, BatchNormalization,
                                     Permute, Concatenate, Conv2D, Add, Activation, Lambda, Normalization)
from tensorflow.keras import Model
from sklearn.preprocessing import (label_binarize, LabelBinarizer, LabelEncoder, OneHotEncoder)
import sys
sys.path.append("..")

'''-------------------------------FLOPS---------------------------------------------'''
@tf.keras.saving.register_keras_serializable()
def get_flops(model_h5_path):
    session = tf.compat.v1.Session()
    graph = tf.compat.v1.get_default_graph()
    with graph.as_default():
        with session.as_default():
            model = tf.keras.models.load_model(model_h5_path)
            run_meta = tf.compat.v1.RunMetadata()
            opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
            # We use the Keras session graph in the call to the profiler.
            flops = tf.compat.v1.profiler.profile(graph=graph, run_meta=run_meta, cmd='op', options=opts)
            return flops.total_float_ops

@tf.keras.saving.register_keras_serializable()
def plot_acc_loss(history, plot_filename, title_type='trainval'):
    pd.DataFrame(history.history).plot(figsize=(8, 5))
    plt.grid(True)
    plt.gca().set_ylim(0, 1) # set the vertical range to [0-1]
    if title_type == 'trainval':
     plt.title('Training and Validation Loss & Accuracy')
    else:
     plt.title('Training Loss & Accuracy')
    plt.savefig(fname=plot_filename)
    plt.clf()
    plt.close()

'''----------------------------Augment Dataset---------------------------------'''
@tf.keras.saving.register_keras_serializable()
def augment_dataset(ds, data_augmentation, augment=None, AUTOTUNE=tf.data.AUTOTUNE):
  # Use image augmentation only on the training set.
  if augment == True:
    ds = ds.map(lambda x, y: (data_augmentation(x, training=True), y), num_parallel_calls=AUTOTUNE)
    # Use buffered prefetching on datasets.
    return ds.prefetch(buffer_size=AUTOTUNE)

  elif augment == False:
    # Use buffered prefetching on datasets.
    return ds.prefetch(buffer_size=AUTOTUNE)

'''----------------------------Evaluate Dataset---------------------------------'''
@tf.keras.saving.register_keras_serializable()
def evaluate_ds(test_dataloader, model, batch_size):
  true_labels = []
  raw_preds = []
  thres_preds = []
  for imgs, labels in iter(test_dataloader):
    preds = model.predict(imgs, batch_size=batch_size)
    true_labels.extend(labels)
    raw_preds.extend(preds.copy())
    preds = np.argmax(preds, axis=1)
    thres_preds.extend(preds)
  return np.array(true_labels), np.array(raw_preds), np.array(thres_preds)

'''----------------------------Plot---------------------------------'''
@tf.keras.saving.register_keras_serializable()
def plot_cm(cm, classes, filename, normalize=False, title='', cmap=plt.cm.Blues):
  plt.imshow(cm, interpolation='nearest', cmap=cmap)
  plt.title(title)
  #plt.colorbar()
  tick_marks = np.arange(len(classes))
  plt.xticks(tick_marks, classes, rotation=45)
  plt.yticks(tick_marks, classes)
  if normalize:
      cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
      print("Normalized confusion matrix")
  else:
      print('Confusion matrix, without normalization')
  print(cm)

  thresh = cm.max() / 2.
  for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
      plt.text(j, i, cm[i, j], horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")
  plt.tight_layout()
  plt.ylabel('True label')
  plt.xlabel('Predicted label')
  plt.savefig(filename)
  plt.clf()
  plt.close()

@tf.keras.saving.register_keras_serializable()
def plot_roc_binary(test_labels, predictions, filename, roconly_filename, n_class, cm_plot_labels, dataset='test'):

  print("cm_plot_labels: ", cm_plot_labels)

  if n_class == 2:
    label_encoder = LabelEncoder()
    onehot_encoder = OneHotEncoder(sparse=False)
    y_test_int_encoded = label_encoder.fit_transform(test_labels)
    y_pred_int_encoded = label_encoder.fit_transform(predictions)

    y_test_int_encoded = y_test_int_encoded.reshape(len(y_test_int_encoded), 1)
    y_pred_int_encoded = y_pred_int_encoded.reshape(len(y_pred_int_encoded), 1)

    y_test = onehot_encoder.fit_transform(y_test_int_encoded)
    y_pred = onehot_encoder.fit_transform(y_pred_int_encoded)

  # Compute ROC curve and ROC area for each class
  fpr = dict()
  tpr = dict()
  roc_auc = dict()
  thresholds = dict()

  for i in range(n_class):
    fpr[i], tpr[i], thresholds[i] = roc_curve(y_test[:, i], y_pred[:, i], pos_label=1)
    roc_auc[i] = auc(fpr[i], tpr[i])
    print("For class-",i,": ", fpr[i], " and ", tpr[i], " and also ", roc_auc[i])

  # Compute micro-average ROC curve and ROC area
  fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_pred.ravel(), pos_label=1)
  roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
  print(f"Micro-averaged One-vs-Rest ROC AUC score: {roc_auc['micro']:.2f}")
  # Compute macro-average ROC curve and ROC area First aggregate all false positive rates
  all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_class)]))

  # Then interpolate all ROC curves at this points
  mean_tpr = np.zeros_like(all_fpr)
  for i in range(n_class):
    mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

  # Finally average it and compute AUC
  mean_tpr /= n_class

  fpr["macro"] = all_fpr
  tpr["macro"] = mean_tpr
  roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
  print(f"Macro-averaged One-vs-Rest ROC AUC score:  {roc_auc['macro']:.2f}")

  # Plot linewidth.
  lw = 2
  # Plot all ROC curves
  plt.figure(1)
  plt.plot(fpr["micro"], tpr["micro"], label='micro-average ROC curve (area = {0:0.2f})'.format(roc_auc["micro"]), color='deeppink', linestyle=':', linewidth=4)
  plt.plot(fpr["macro"], tpr["macro"], label='macro-average ROC curve (area = {0:0.2f})'.format(roc_auc["macro"]), color='navy', linestyle=':', linewidth=4)
  colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])

  plt.plot([0, 1], [0, 1], 'k--', lw=lw)
  plt.xlim([0.0, 1.0])
  plt.ylim([0.0, 1.05])
  plt.xlabel('False Positive Rate')
  plt.ylabel('True Positive Rate')
  plt.title('Receiver operating characteristic to multi-class')
  plt.legend(loc="lower right")
  plt.savefig(roconly_filename)

  for i, color in zip(range(n_class), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw, label='ROC curve of class {0} (area = {1:0.2f})'.format(i, roc_auc[i]))

  plt.plot([0, 1], [0, 1], 'k--', lw=lw)
  plt.xlim([0.0, 1.0])
  plt.ylim([0.0, 1.05])
  plt.xlabel('False Positive Rate')
  plt.ylabel('True Positive Rate')
  if dataset == 'train':
    plt.title('ROC Curve to multi-class Train dataset')
  elif dataset == 'val':
    plt.title('ROC Curve to multi-class Val dataset')
  else:
    plt.title('ROC Curve to multi-class Test dataset')
  plt.legend(loc="lower right")
  plt.savefig(filename)
  plt.clf()
  plt.close()

@tf.keras.saving.register_keras_serializable()
def plot_roc(test_labels, predictions, filename, roconly_filename, n_class, cm_plot_labels, dataset='test'):

  print("cm_plot_labels: ", cm_plot_labels)

  if n_class == 2:
    label_encoder = LabelEncoder()
    onehot_encoder = OneHotEncoder(sparse=False, dtype=int)
    y_test_int_encoded = label_encoder.fit_transform(test_labels)
    y_pred_int_encoded = label_encoder.fit_transform(predictions)

    y_test_int_encoded = y_test_int_encoded.reshape(len(y_test_int_encoded), 1)
    y_pred_int_encoded = y_pred_int_encoded.reshape(len(y_pred_int_encoded), 1)
    y_test = onehot_encoder.fit_transform(y_test_int_encoded)
    y_pred = onehot_encoder.fit_transform(y_pred_int_encoded)

  if n_class > 2:
    print("\n----------++++++++++----------++++++++++----------\n")
    print("label_binarize: ", label_binarize(y=cm_plot_labels, classes=cm_plot_labels))
    print("\n----------++++++++++----------++++++++++----------\n")
    y_test = label_binarize(test_labels, classes=np.arange(n_class))
    y_pred = label_binarize(predictions, classes=np.arange(n_class))


  # Compute ROC curve and ROC area for each class
  fpr = dict()
  tpr = dict()
  roc_auc = dict()
  thresholds = dict()

  for i in range(n_class):
    fpr[i], tpr[i], thresholds[i] = roc_curve(y_test[:, i], y_pred[:, i],)
    roc_auc[i] = auc(fpr[i], tpr[i])
    print("For class-",i,": ", fpr[i], " and ", tpr[i], " and also ", roc_auc[i])

  # Compute micro-average ROC curve and ROC area
  fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_pred.ravel())
  roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
  print(f"Micro-averaged One-vs-Rest ROC AUC score: {roc_auc['micro']:.2f}")
  # Compute macro-average ROC curve and ROC area First aggregate all false positive rates
  all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_class)]))

  # Then interpolate all ROC curves at this points
  mean_tpr = np.zeros_like(all_fpr)
  for i in range(n_class):
    mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

  # Finally average it and compute AUC
  mean_tpr /= n_class

  fpr["macro"] = all_fpr
  tpr["macro"] = mean_tpr
  roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
  print(f"Macro-averaged One-vs-Rest ROC AUC score:  {roc_auc['macro']:.2f}")

  # Plot linewidth.
  lw = 2
  # Plot all ROC curves
  plt.figure(1)
  plt.plot(fpr["micro"], tpr["micro"], label='micro-average ROC curve (area = {0:0.2f})'.format(roc_auc["micro"]), color='deeppink', linestyle=':', linewidth=4)
  plt.plot(fpr["macro"], tpr["macro"], label='macro-average ROC curve (area = {0:0.2f})'.format(roc_auc["macro"]), color='navy', linestyle=':', linewidth=4)
  colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])

  plt.plot([0, 1], [0, 1], 'k--', lw=lw)
  plt.xlim([0.0, 1.0])
  plt.ylim([0.0, 1.05])
  plt.xlabel('False Positive Rate')
  plt.ylabel('True Positive Rate')
  plt.title('Receiver operating characteristic to multi-class')
  plt.legend(loc="lower right")
  plt.savefig(roconly_filename)

  for i, color in zip(range(n_class), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw, label='ROC curve of class {0} (area = {1:0.2f})'.format(i, roc_auc[i]))

  plt.plot([0, 1], [0, 1], 'k--', lw=lw)
  plt.xlim([0.0, 1.0])
  plt.ylim([0.0, 1.05])
  plt.xlabel('False Positive Rate')
  plt.ylabel('True Positive Rate')
  if dataset == 'train':
    plt.title('ROC Curve to multi-class Train dataset')
  elif dataset == 'val':
    plt.title('ROC Curve to multi-class Val dataset')
  else:
    plt.title('ROC Curve to multi-class Test dataset')
  plt.legend(loc="lower right")
  plt.savefig(filename)
  plt.clf()
  plt.close()
'''----------------------------End of Utility---------------------------------'''
