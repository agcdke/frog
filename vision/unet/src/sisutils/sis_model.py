################################################################################
# AgriBlazeNet: Different Architecture Selection
# imgcls_blaze_cbwssam (for T1H4B4, T2H4B4), imgcls_blaze_cbam(for BFCBAM), imgcls_blaze(for BF)
################################################################################
import tensorflow as tf
import sys
sys.path.append("..")
from .sis_agriblazenet_unet import (BlazeUnet)

'''--------------------------------------------------------------------Semantic Segmentation U-Net Model--------------------------------------------------------------------'''
@tf.keras.utils.register_keras_serializable()
def sis_unet_model(model_arch_module, num_classes, cbam_block=None):
   if model_arch_module == "unet_blaze":
       print("Model: unet_blaze")
       model = BlazeUnet(num_classes=num_classes)
       return model

   elif model_arch_module == "unet_blaze_cbam":
       print("Model: unet_blaze_cbam")
       model = BlazeUnet(num_classes=num_classes, attn_block='cbam')
       return model

   elif model_arch_module == "unet_blaze_cbwssam":
       print("Model: unet_blaze_cbwssam")
       if cbam_block == 'mvit_h4b4':
         print("CBwSSAM Type-1: T1H4B4")
         model = BlazeUnet(num_classes=num_classes, attn_block='cbwssam', cbam_block='mvit_h4b4')
       elif cbam_block == 'cbmvit_h4b4':
         print("CBwSSAM Type-2: T2H4B4")
         model = BlazeUnet(num_classes=num_classes, attn_block='cbwssam', cbam_block='cbmvit_h4b4')
       elif cbam_block == 'mvit_h8b8':
         print("CBwSSAM Type-1: T1H8B8")
         model = BlazeUnet(num_classes=num_classes, attn_block='cbwssam', cbam_block='mvit_h8b8')
       elif cbam_block == 'cbmvit_h8b8':
         print("CBwSSAM Type-2: T2H8B8")
         model = BlazeUnet(num_classes=num_classes, attn_block='cbwssam', cbam_block='cbmvit_h8b8')

       return model
'''-------------------------------------------------------------------------------------------------------------------------------------------------------------------'''
