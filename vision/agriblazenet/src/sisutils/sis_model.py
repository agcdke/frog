################################################################################
# AgriBlazeNet: Different Architecture Selection
################################################################################
import tensorflow as tf
import sys
sys.path.append("..")
from .sis_agriblazenet import (BlazeImgClsModel)

'''--------------------------------------------------------------------Image Classification Model--------------------------------------------------------------------'''
@tf.keras.saving.register_keras_serializable()
def sis_imgcls_model(model_arch_module, no_classes, input_shape=None, img_mean = None, img_std =None):
    if model_arch_module == "imgcls_blaze":
       model = BlazeImgClsModel(no_classes=no_classes, img_mean=img_mean, img_std=img_std)
       return model

    elif model_arch_module == "imgcls_blaze_cbam":
       model = BlazeImgClsModel(no_classes=no_classes, img_mean=img_mean, img_std=img_std, attn_block='cbam')
       return model

    elif model_arch_module == "imgcls_blaze_cbwssam":
       model = BlazeImgClsModel(no_classes=no_classes, img_mean=img_mean, img_std=img_std, attn_block='cbwssam')
       return model

'''-------------------------------------------------------------------------------------------------------------------------------------------------------------------'''
