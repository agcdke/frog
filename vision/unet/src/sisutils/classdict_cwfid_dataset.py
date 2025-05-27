'''
Dataset: https://github.com/cwfid/dataset
'''
import tensorflow as tf
tf.keras.backend.clear_session()

import os, yaml, random
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import numpy as np
from tensorflow.keras.utils import (load_img, img_to_array, to_categorical)
from tensorflow.data.experimental import (cardinality)
import matplotlib.pyplot as plt

AUTOTUNE = tf.data.AUTOTUNE

class CwfidDataset:
    def __init__(self, config_params):
        self.config_params = config_params
        with open(self.config_params['data_dir'] + config_params['yaml_file'], 'r') as f:
            self.data = yaml.safe_load(f)

    def get_true_mask(self, train_mode=False):
        img = []

        if train_mode == False:
            for number in self.data['test']:
                number = "{0:03d}".format(number)
                x_imgpath = os.path.join(self.config_params['data_dir'], 'annotations', str.format(number)) + '_annotation.png'
                x_img = load_img(x_imgpath,
                                target_size=(self.config_params['img_h'], self.config_params['img_w']),
                                keep_aspect_ratio=True)
                x_img = img_to_array(x_img, dtype='float32') / 255
                img.append(x_img)

        else:
            for number in self.data['train']:
                number = "{0:03d}".format(number)
                x_imgpath = os.path.join(self.config_params['data_dir'], 'annotations', str.format(number)) + '_annotation.png'
                x_img = load_img(x_imgpath,
                                target_size=(self.config_params['img_h'], self.config_params['img_w']),
                                keep_aspect_ratio=True)
                x_img = img_to_array(x_img, dtype='float32') / 255
                img.append(x_img)

        return np.array(img)

    def prepare_rgb_img_dataset(self, train_mode=False):
        img = []

        if train_mode == False:
            for number in self.data['test']:
                number = "{0:03d}".format(number)
                x_imgpath = os.path.join(self.config_params['data_dir'], 'images', str.format(number)) + '_image.png'
                x_img = load_img(x_imgpath,
                                target_size=(self.config_params['img_h'], self.config_params['img_w']),
                                keep_aspect_ratio=True)
                x_img = img_to_array(x_img, dtype='float32') / 255
                img.append(x_img)

        else:
            for number in self.data['train']:
                number = "{0:03d}".format(number)
                x_imgpath = os.path.join(self.config_params['data_dir'], 'images', str.format(number)) + '_image.png'
                x_img = load_img(x_imgpath,
                                target_size=(self.config_params['img_h'], self.config_params['img_w']),
                                keep_aspect_ratio=True)
                x_img = img_to_array(x_img, dtype='float32') / 255
                img.append(x_img)

        return np.array(img)

    def rgb_to_labels(self, img):
            mask_labels = pd.read_csv(self.config_params['data_dir'] + self.config_params['class_dict_file'])
            label_seg = np.zeros(img.shape,dtype=np.uint8)
            img = img*255 # from normalized image to range (0-255)

            for i in range(mask_labels.shape[0]):
                # Pixel color matching based on CSV file (0:class_name, 1:r, 2:g, 3:b)
                label_seg[np.all(img == list(mask_labels.iloc[i, [1,2,3]]), axis=-1)] = i
            label_seg = label_seg[:,:,0]  # Take first channel
            return label_seg

    def prepare_rgb_mask_dataset(self, train_mode=False):
        labels = []

        if train_mode == False:
            for number in self.data['test']:
                number = "{0:03d}".format(number)
                y_imgpath = os.path.join(self.config_params['data_dir'], 'annotations', str.format(number)) + '_annotation.png'
                y_img = load_img(y_imgpath,
                                target_size=(self.config_params['img_h'], self.config_params['img_w']),
                                keep_aspect_ratio=True, color_mode='rgb')
                y_img = img_to_array(y_img, dtype='float32') / 255
                y_img = self.rgb_to_labels(y_img) # transform label to categorical_classes
                labels.append(y_img)

        else:
            for number in self.data['train']:
                number = "{0:03d}".format(number)
                y_imgpath = os.path.join(self.config_params['data_dir'], 'annotations', str.format(number)) + '_annotation.png'
                y_img = load_img(y_imgpath,
                                target_size=(self.config_params['img_h'], self.config_params['img_w']),
                                keep_aspect_ratio=True, color_mode='rgb')
                y_img = img_to_array(y_img, dtype='float32') / 255
                y_img = self.rgb_to_labels(y_img) # transform label to categorical_classes
                labels.append(y_img)


        labels = np.array(labels)
        labels = np.expand_dims(labels, axis=3)
        print("labels.shape: ", labels.shape)
        unique_labels = np.unique(labels)

        if train_mode==False:
          return labels
        else:
          return labels, unique_labels

    def get_cwfid_dataset(self, val_percent=0.2, sanity_checker=False):
        '''
        Crop-Weed Field Dataset
        '''
        train_x = self.prepare_rgb_img_dataset(train_mode=True)
        train_y, unique_labels = self.prepare_rgb_mask_dataset(train_mode=True)
        train_true_mask = self.get_true_mask(train_mode=True)

        print("unique_labels: ", unique_labels)
        labels_cat = to_categorical(unique_labels, num_classes=self.config_params['num_classes'])
        print("labels_categorical:\n", labels_cat)

        test_x = self.prepare_rgb_img_dataset(train_mode=False)
        test_y = self.prepare_rgb_mask_dataset(train_mode=False)

        if sanity_checker == True:
            self.sanity_check_img_mask_category(train_x, train_y, train_true_mask)

        # Convert Mask To_Categorical of Number_of_Classes
        train_masks_cat = to_categorical(train_y, num_classes=self.config_params['num_classes'])
        train_y_cat = train_masks_cat.reshape((train_y.shape[0], train_y.shape[1], train_y.shape[2], self.config_params['num_classes']))


        test_masks_cat = to_categorical(test_y, num_classes=self.config_params['num_classes'])
        test_y_cat = test_masks_cat.reshape((test_y.shape[0], test_y.shape[1], test_y.shape[2], self.config_params['num_classes']))

        print("Shape(train) & to_cat: ", train_x.shape, train_y.shape, train_y_cat.shape)
        print("Shape(test) & to_cat: ", test_x.shape, test_y.shape, test_y_cat.shape)

        # Create TF Dataset combining Input images and masks
        train_val_dataset = tf.data.Dataset.from_tensor_slices((train_x, train_y_cat))
        test_dataset = tf.data.Dataset.from_tensor_slices((test_x, test_y_cat))

        val_size = int(val_percent * len(train_x)) # Take val_percent % as val dataset from train_val dataset
        #train_val_dataset = train_val_dataset.shuffle(self.config_params['buffer_size'])
        val_dataset = train_val_dataset.take(val_size)
        train_dataset = train_val_dataset.skip(val_size)

        return train_dataset, val_dataset, test_dataset

    def sanity_check_img_mask_category(self, train_x, train_y, train_true_mask):
        img_rand_no = random.randint(0, len(train_x))
        print("img_rand_no: ", img_rand_no)
        self.visualize_img_mask(train_x, train_y, train_true_mask, img_rand_no)

    def visualize_img_mask(self, train_x, train_y, train_true_mask, img_rand_no):
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
            ax1.imshow(train_x[img_rand_no])
            ax1.set_title("Image")
            ax2.imshow(train_true_mask[img_rand_no])
            ax2.set_title("True Annotation")
            ax3.imshow(train_y[img_rand_no, :, :, 0])
            ax3.set_title("Categorical")
            plt.show()

    def brightness(self, img, mask):
        img = tf.image.adjust_brightness(img, 0.1)
        return img, mask

    def contrast(self, img, mask):
        img = tf.image.adjust_contrast(img, 0.1)
        return img, mask

    def heu(self, img, mask):
        img = tf.image.adjust_hue(img, -0.1)
        return img, mask

    def saturation(self, img, mask):
        img = tf.image.adjust_saturation(img, 0.1)
        return img, mask

    def gamma(self, img, mask):
        img = tf.image.adjust_gamma(img, 0.1)
        return img, mask

    def flip_horizontal(self, img, mask):
        img = tf.image.flip_left_right(img)
        mask = tf.image.flip_left_right(mask)
        return img, mask

    def flip_vertical(self, img, mask):
        img = tf.image.flip_up_down(img)
        mask = tf.image.flip_up_down(mask)
        return img, mask

    def rotate(self, img, mask):
        img = tf.image.rot90(img)
        mask = tf.image.rot90(mask)
        return img, mask

    def get_cwfid_dataset_with_imgaug(self, shuffle=False):
        train_dataset, val_dataset, test_dataset = self.get_cwfid_dataset()
        cardinality_train = int(cardinality(train_dataset).numpy())
        cardinality_val = int(cardinality(val_dataset).numpy())

        train_brightness = train_dataset.map(self.brightness)
        train_contrast = train_dataset.map(self.contrast)
        train_hue = train_dataset.map(self.heu)
        train_saturation = train_dataset.map(self.saturation)
        train_gamma = train_dataset.map(self.gamma)
        train_flip_hor = train_dataset.map(self.flip_horizontal)
        train_flip_ver = train_dataset.map(self.flip_vertical)
        train_rotate = train_dataset.map(self.rotate)

        train_dataset = train_dataset.concatenate(train_brightness)
        train_dataset = train_dataset.concatenate(train_contrast)
        train_dataset = train_dataset.concatenate(train_hue)
        train_dataset = train_dataset.concatenate(train_saturation)
        train_dataset = train_dataset.concatenate(train_gamma)
        train_dataset = train_dataset.concatenate(train_flip_hor)
        train_dataset = train_dataset.concatenate(train_flip_ver)
        train_dataset = train_dataset.concatenate(train_rotate)

        # Batch Size: Distributed Strategy for 2 GPUs, so batch_size=(2 * batch_size)
        if shuffle == True:
            train_dataset = train_dataset.shuffle(self.config_params['buffer_size']).take(self.config_params['buffer_size']).cache().batch(2*self.config_params['batch_size']).repeat(self.config_params['repeat_size']).prefetch(buffer_size=AUTOTUNE)
        else:
            train_dataset = train_dataset.take(self.config_params['buffer_size']).cache().batch(self.config_params['batch_size']).repeat(self.config_params['repeat_size']).prefetch(buffer_size=AUTOTUNE)

        val_dataset = val_dataset.take(self.config_params['buffer_size']).cache().batch(self.config_params['batch_size']).repeat(self.config_params['repeat_size']).prefetch(buffer_size=AUTOTUNE)
        test_dataset = test_dataset.take(self.config_params['buffer_size']).cache().batch(self.config_params['batch_size']).repeat(self.config_params['repeat_size']).prefetch(buffer_size=AUTOTUNE)
        return train_dataset, val_dataset, test_dataset


    def visualize_augimg_mask(self, image, augimg_mask, img_rand_no):
        print("augimg_mask[img_rand_no, :, :] ", augimg_mask[img_rand_no, :, :].shape)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        ax1.imshow(image[img_rand_no, :, :])
        ax1.set_title("Aug. Image")
        ax2.imshow(augimg_mask[img_rand_no, :, :])
        ax2.set_title("Aug. Annotation")
        plt.show()
