################################################################################
# SIS AgriBlazeNet
################################################################################

import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import (Input, GlobalAveragePooling2D, GlobalMaxPooling2D, Reshape, Dense, Multiply, Flatten, MaxPooling2D,
                                     AveragePooling2D, Dropout, BatchNormalization, LayerNormalization, DepthwiseConv2D, Permute, Layer,
                                     Concatenate, Conv2D, Conv1D, Add, Activation, Lambda, Rescaling, Normalization, MultiHeadAttention,
                                     Embedding, ReLU)
from tensorflow.keras import activations
from tensorflow.keras import Model, Sequential
from tensorflow.keras.utils import plot_model
import sys, math
sys.path.append("..")

'''--------------------------------------------------------------------CBAM--------------------------------------------------------------------'''
@tf.keras.saving.register_keras_serializable()
class ChannelAttention2D(Layer):
    def __init__(self, ratio=8, **kwargs):
        super().__init__(**kwargs)
        self.ratio = ratio
        self.gap = GlobalAveragePooling2D()
        self.gmp = GlobalMaxPooling2D()
        self.add = Add()
        self.activation_sigmoid = Activation('sigmoid')
        self.multiply = Multiply()

    def build(self, input_feature_shape):
        self.channel = input_feature_shape[-1]
        self.shared_layer_1 = Dense(units = self.channel//self.ratio, activation="relu", kernel_initializer='he_normal', use_bias=True, bias_initializer='zeros')
        self.shared_layer_2 = Dense(units = self.channel, kernel_initializer='he_normal', use_bias=True, bias_initializer='zeros')
        self.reshape = Reshape((1,1,self.channel))

    def __call__(self, input_feature):
        if not self.built:
            self.build(input_feature.shape)
            self.built = True
        return self.call(input_feature)

    def call(self, input_feature):
        x_gap = self.gap(input_feature)
        x_gap = self.reshape(x_gap)
        assert x_gap.shape[1:] == (1,1,self.channel)
        x_gap = self.shared_layer_1(x_gap)
        assert x_gap.shape[1:] == (1,1,self.channel//self.ratio)
        x_gap = self.shared_layer_2(x_gap)
        assert x_gap.shape[1:] == (1,1,self.channel)

        x_gmp = self.gmp(input_feature)
        x_gmp = self.reshape(x_gmp)
        assert x_gmp.shape[1:] == (1,1,self.channel)
        x_gmp = self.shared_layer_1(x_gmp)
        assert x_gmp.shape[1:] == (1,1,self.channel//self.ratio)
        x_gmp = self.shared_layer_2(x_gmp)
        assert x_gmp.shape[1:] == (1,1,self.channel)

        channel_attn_feature = self.add([x_gap,x_gmp])
        channel_attn_feature = self.activation_sigmoid(channel_attn_feature)
        return self.multiply([input_feature, channel_attn_feature])

@tf.keras.saving.register_keras_serializable()
class ChannelwiseAvgPool2D(Layer):
    def __init__(self, axis=3, **kwargs):
        super().__init__(**kwargs)
        self.axis = axis

    def call(self, inputs):
        return tf.math.reduce_mean(inputs, self.axis, keepdims=True)

@tf.keras.saving.register_keras_serializable()
class ChannelwiseMaxPool2D(Layer):
    def __init__(self, axis=3, **kwargs):
        super().__init__(**kwargs)
        self.axis = axis

    def call(self, inputs):
        return tf.math.reduce_max(inputs, self.axis, keepdims=True)

@tf.keras.saving.register_keras_serializable()
class SpatialAttention2D(Layer):
    def __init__(self, kernel_size=7, **kwargs):
        super().__init__(**kwargs)
        self.kernel_size = kernel_size
        self.channelwise_avg_pool = ChannelwiseAvgPool2D(axis=3)
        self.channelwise_max_pool = ChannelwiseMaxPool2D(axis=3)
        self.channelwise_concat = Concatenate(axis=3)
        self.conv_1 = Conv2D(filters = 1, kernel_size=self.kernel_size, strides=1, padding='same', activation='sigmoid', kernel_initializer='he_normal', use_bias=False)
        self.multiply = Multiply()

    def build(self, input_feature_shape):
            self.channel = input_feature_shape[-1]

    def __call__(self, input_feature):
        if not self.built:
            self.build(input_feature.shape)
            self.built = True
        return self.call(input_feature)

    def call(self, input_feature):
        self.channel_attn_feature = input_feature
        x_avgpool = self.channelwise_avg_pool(self.channel_attn_feature)
        assert x_avgpool.shape[-1] == 1
        x_maxpool = self.channelwise_max_pool(self.channel_attn_feature)
        assert x_maxpool.shape[-1] == 1

        concat = self.channelwise_concat([x_avgpool, x_maxpool])
        assert concat.shape[-1] == 2
        spatial_attn_feature = self.conv_1(concat)
        assert spatial_attn_feature.shape[-1] == 1
        return self.multiply([input_feature, spatial_attn_feature])

@tf.keras.saving.register_keras_serializable()
class Cbam2D(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.channel_attn_feature = ChannelAttention2D(ratio=8)
        self.cbam_feature = SpatialAttention2D(kernel_size=7)

    def call(self, input_feature):
        x = self.channel_attn_feature(input_feature)
        x = self.cbam_feature(x)
        return x

'''---------------------------------------------------SIS-Mobile-ViT-------------------------------------------------------------------------------------'''
@tf.keras.saving.register_keras_serializable()
class TransformerBlock(Layer):
    def __init__(self, transformer_layers, projection_dim, num_heads, ff_dim, dropout_rate):
        super().__init__()
        self.transformer_layers = transformer_layers
        self.mh_attn = MultiHeadAttention(num_heads=num_heads, key_dim=projection_dim, dropout=dropout_rate)
        self.ffn = Sequential([Dense(ff_dim, activation="relu"), Dense(projection_dim)])
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(dropout_rate)
        self.dropout2 = Dropout(dropout_rate)

    def call(self, inputs):
        x = inputs
        for _ in range(self.transformer_layers):
          attn_output = self.mh_attn(inputs, inputs)
          out1 = self.layernorm1(inputs + attn_output)
          ffn_output = self.ffn(out1)
          ffn_output = self.dropout2(ffn_output)
          x = self.layernorm2(out1 + ffn_output)
        return x

@tf.keras.saving.register_keras_serializable()
class SisMobileViT(Layer):
    def __init__(self, transformer_layers, projection_dim, num_heads, ff_dim, dropout_rate, per_row_no_patch=4, strides=1, **kwargs):
        super().__init__(**kwargs)
        self.per_row_no_patch = per_row_no_patch
        self.conv_local_feature = Conv2D(filters=projection_dim, kernel_size = (1, 1), padding="same", strides=strides)
        self.transformer_block = TransformerBlock(transformer_layers=transformer_layers, projection_dim=projection_dim, num_heads=num_heads, ff_dim=ff_dim, dropout_rate=dropout_rate)
        self.conv_folded_feature_map = Conv2D(filters=projection_dim, kernel_size = (1, 1), padding="same", strides=strides, activation=tf.keras.activations.swish)
        self.conv_local_global_features = Conv2D(filters=projection_dim, kernel_size=(3, 3), padding="same", strides=strides, activation=tf.keras.activations.swish)
        self.concat = Concatenate(axis=-1)

    def build(self, input_feature_shape):
        self.channel = input_feature_shape[-1]
        self.feature_spatial_dim = input_feature_shape[1]
        print("self.channel, self.feature_spatial_dim: ", self.channel, " , ", self.feature_spatial_dim)
        self.patch_size = int(math.pow(self.feature_spatial_dim/self.per_row_no_patch,2))

    def __call__(self, input_feature):
        if not self.built:
            self.build(input_feature.shape)
            self.built = True
        return self.call(input_feature)

    def call(self, input_feature):
        x = input_feature
        # Local projection
        local_features = self.conv_local_feature(x)
        # Unfold and Transformer
        num_patches = int((local_features.shape[1] * local_features.shape[2]) / self.patch_size)
        non_overlapping_patches = Reshape((self.patch_size, num_patches, self.channel))(local_features)
        global_features = self.transformer_block(non_overlapping_patches)
        # Fold into Conv-datacube
        folded_feature_map = Reshape((*local_features.shape[1:-1], self.channel))(global_features)
        # Point-wise Conv and Concat
        folded_feature_map = self.conv_folded_feature_map(folded_feature_map)
        local_global_features = self.concat([x, folded_feature_map]) # experiment
        # Fuse local and global features
        local_global_features = self.conv_local_global_features(local_global_features)
        return local_global_features

'''---------------------------------------------------CBwSSAM-------------------------------------------------------------------------------------'''
# CBwSSAM (MobileViT on Channel Attention)
@tf.keras.saving.register_keras_serializable()
class SpatialMobileViT2D(Layer):
    def __init__(self, projection_dim, ff_dim, num_heads, num_blocks, **kwargs):
        super().__init__(**kwargs)
        self.sis_mvit = SisMobileViT(transformer_layers=num_blocks, projection_dim=projection_dim, ff_dim=ff_dim, num_heads=num_heads, dropout_rate=0.1)

    def build(self, input_feature_shape):
        self.channel = input_feature_shape[-1]

    def __call__(self, input_feature):
        if not self.built:
            self.build(input_feature.shape)
            self.built = True
        return self.call(input_feature)

    def call(self, input_feature):
        input_spatial_attn_feature = input_feature
        spatial_selfattn_feature = self.sis_mvit(input_spatial_attn_feature)
        assert spatial_selfattn_feature.shape[-1] == self.channel
        return spatial_selfattn_feature

# CBwSSAM (enhanced CBAM's Spatial Attention)
@tf.keras.saving.register_keras_serializable()
class ConvBlockSelfAttention2D(Layer):
    def __init__(self, projection_dim, ff_dim, num_heads, num_blocks, kernel_size=7, **kwargs):
        super().__init__(**kwargs)
        self.channelwise_avg_pool = ChannelwiseAvgPool2D(axis=3)
        self.channelwise_max_pool = ChannelwiseMaxPool2D(axis=3)
        self.channelwise_concat = Concatenate(axis=3)
        self.conv_1 = Conv2D(filters = 1, kernel_size=kernel_size, strides=1, padding='same', activation='sigmoid', kernel_initializer='he_normal', use_bias=False)
        self.sis_mvit = SisMobileViT(transformer_layers=num_blocks, projection_dim=projection_dim, ff_dim=ff_dim, num_heads=num_heads, dropout_rate=0.1)
        self.multiply = Multiply()

    def build(self, input_feature_shape):
        self.channel = input_feature_shape[-1]

    def __call__(self, input_feature):
        if not self.built:
            self.build(input_feature.shape)
            self.built = True
        return self.call(input_feature)

    def call(self, input_feature):
        # MViT on input_feature (Channel Attention Feature)
        channel_and_selfattn_feature = self.sis_mvit(input_feature)
        assert channel_and_selfattn_feature.shape[-1] == self.channel
        # CBAM Spatial Attention
        input_spatial_attn_feature = input_feature
        x_avgpool = self.channelwise_avg_pool(input_spatial_attn_feature)
        assert x_avgpool.shape[-1] == 1
        x_maxpool = self.channelwise_max_pool(input_spatial_attn_feature)
        assert x_maxpool.shape[-1] == 1

        concat = self.channelwise_concat([x_avgpool, x_maxpool])
        assert concat.shape[-1] == 2
        spatial_attn_feature = self.conv_1(concat)
        assert spatial_attn_feature.shape[-1] == 1
        # CBAM Feature
        cbam_feature = self.multiply([channel_and_selfattn_feature, spatial_attn_feature])
        assert cbam_feature.shape[-1] == self.channel
        return cbam_feature

'''------------------------------------------------------CBwSSAM Layer--------------------------------------------------------------------------------'''
# Choose 'cbam' parameter Manually, e.g., cbam='mvit_h4b4' or cbam='mvit_h8b8'
@tf.keras.saving.register_keras_serializable()
class Cbwssam2D(Layer):
    def __init__(self, projection_dim, ff_dim, cbam='mvit_h4b4',**kwargs):
        super().__init__(**kwargs)
        self.channel_attn_feature = ChannelAttention2D(ratio=8)
        self.cbam_info = cbam
        if cbam == 'cbmvit_h4b4':
          self.cbam_feature = ConvBlockSelfAttention2D(projection_dim=projection_dim, ff_dim=ff_dim, kernel_size=7, num_heads=4, num_blocks=4)

        elif cbam == 'cbmvit_h8b8':
          self.cbam_feature = ConvBlockSelfAttention2D(projection_dim=projection_dim, ff_dim=ff_dim, kernel_size=7, num_heads=8, num_blocks=8)

        elif cbam == 'mvit_h4b4':
          self.cbam_feature = SpatialMobileViT2D(projection_dim=projection_dim, ff_dim=ff_dim, num_heads=4, num_blocks=4)

        elif cbam == 'mvit_h8b8':
          self.cbam_feature = SpatialMobileViT2D(projection_dim=projection_dim, ff_dim=ff_dim, num_heads=8, num_blocks=8)

    def call(self, input_feature):
        print("cbam_info: ", self.cbam_info)
        x = self.channel_attn_feature(input_feature)
        x = self.cbam_feature(x)
        return x


'''--------------------------------------------------------Blaze Block--------------------------------------------------------------------------------'''
@tf.keras.saving.register_keras_serializable()
class SingleBlazeBlock(Layer):
  def __init__(self, filters, strides=1, **kwargs):
    super(SingleBlazeBlock, self).__init__()
    self.strides = strides
    self.filters = filters
    if strides == 2:
      self.pool = MaxPooling2D()

    self.dw_conv_1 = DepthwiseConv2D(kernel_size=(5, 5), strides=strides, padding="same")
    self.proj_conv_1 = Conv2D(filters, kernel_size=(1, 1), strides=(1, 1), padding="same")
    self.norm_1 = BatchNormalization()
    self.activation_1 = ReLU()

  def __call__(self, input_feature):
    if not self.built:
      self.build(input_feature.shape)
      self.built = True
    return self.call(input_feature)

  def call(self, x_input):
    x = self.dw_conv_1(x_input)
    x = self.proj_conv_1(x)
    if self.strides == 2:
      x_input = self.pool(x_input)


    padding = self.filters - x_input.shape[-1]
    if padding != 0:
       padding_values = [[0, 0], [0, 0], [0, 0], [0, padding]]
       x_input = tf.pad(x_input, paddings=padding_values)

    x = x + x_input
    x = self.norm_1(x)
    x = self.activation_1(x)
    return x

@tf.keras.saving.register_keras_serializable()
class DoubleBlazeBlock(Layer):
  def __init__(self, proj_filters, expand_filters, strides=1, **kwargs):
    super(DoubleBlazeBlock, self).__init__()
    self.strides = strides
    self.proj_filters = proj_filters
    self.expand_filters = expand_filters
    if strides == 2:
      self.pool = MaxPooling2D()

    # Project
    self.dw_conv_1 = DepthwiseConv2D(kernel_size=(5, 5), strides=strides, padding="same")
    self.proj_conv_1 = Conv2D(proj_filters, kernel_size=(1, 1), strides=(1, 1), padding="same")
    self.norm_1 = BatchNormalization()
    self.activation_1 = ReLU()

    # Expand (always strides=1)
    self.dw_conv_2 = DepthwiseConv2D(kernel_size=(5, 5), strides=1, padding="same")
    self.proj_conv_2 = Conv2D(expand_filters, kernel_size=(1, 1), strides=(1, 1), padding="same")
    self.norm_2 = BatchNormalization()
    self.activation_2 = ReLU()

  def __call__(self, input_feature):
    if not self.built:
      self.build(input_feature.shape)
      self.built = True
    return self.call(input_feature)

  def call(self, x_input):
    x = self.dw_conv_1(x_input)
    x = self.proj_conv_1(x)
    x = self.activation_1(x)
    if self.strides == 2:
      x_input = self.pool(x_input)

    x = self.dw_conv_2(x)
    x = self.proj_conv_2(x)


    padding = self.expand_filters - x_input.shape[-1]
    if padding != 0:
       padding_values = [[0, 0], [0, 0], [0, 0], [0, padding]]
       x_input = tf.pad(x_input, paddings=padding_values)

    x = x + x_input
    x = self.norm_2(x)
    x = self.activation_2(x)
    return x

'''--------------------------------------------------------Blaze Image Classification Model--------------------------------------------------------------------------------'''
@tf.keras.saving.register_keras_serializable()
class BlazeImgClsModel(Model):
  def __init__(self, no_classes, img_mean, img_std, attn_block=None, **kwargs):
    super(BlazeImgClsModel, self).__init__(**kwargs)
    self.attn_block = attn_block
    self.no_classes = no_classes
    self.img_mean = img_mean
    self.img_std = img_std
    self.feature_norm = Normalization(mean=img_mean, variance=img_std)
    self.conv_1 = Conv2D(filters=24, kernel_size=(5, 5), strides=2, padding="same")
    self.activation_1 = ReLU()
    self.single_block_1 = SingleBlazeBlock(filters=24)
    self.single_block_2 = SingleBlazeBlock(filters=24)
    self.single_block_3 = SingleBlazeBlock(filters=48, strides=2)

    if self.attn_block == 'cbam':
      self.cbam_1 = Cbam2D()
    elif self.attn_block == 'cbwssam':
      self.cbwssam_1 = Cbwssam2D(projection_dim=48, ff_dim=48)

    self.single_block_4 = SingleBlazeBlock(filters=48)
    self.single_block_5 = SingleBlazeBlock(filters=48, strides=2)
    self.double_block_1 = DoubleBlazeBlock(proj_filters=24, expand_filters=96, strides=2)
    if self.attn_block == 'cbam':
      self.cbam_2 = Cbam2D()
    elif self.attn_block == 'cbwssam':
      self.cbwssam_2 = Cbwssam2D(projection_dim=96, ff_dim=96)

    self.double_block_2 = DoubleBlazeBlock(proj_filters=24, expand_filters=96)
    self.double_block_3 = DoubleBlazeBlock(proj_filters=24, expand_filters=96)
    self.double_block_4 = DoubleBlazeBlock(proj_filters=24, expand_filters=96, strides=2)
    if self.attn_block == 'cbam':
      self.cbam_3 = Cbam2D()
    elif self.attn_block == 'cbwssam':
      self.cbwssam_3 = Cbwssam2D(projection_dim=96, ff_dim=96)

    self.double_block_5 = DoubleBlazeBlock(proj_filters=24, expand_filters=96)
    self.double_block_6 = DoubleBlazeBlock(proj_filters=24, expand_filters=96)
    self.gap = GlobalAveragePooling2D()
    self.dense = Dense(units=no_classes, activation = 'softmax')

  def get_config(self):
    config = super().get_config()
    config_params = {
      'no_classes': self.no_classes,
      'attn_block': self.attn_block,
      'img_mean': self.img_mean,
      'img_std': self.img_std,
    }
    config.update(config_params)
    return config

  def call(self, x):
    x_batch, x_height, x_width, x_channel = x.shape
    x = self.feature_norm(x)
    x = self.conv_1(x)
    x = self.activation_1(x)
    x = self.single_block_1(x)
    x = self.single_block_2(x)
    x = self.single_block_3(x)

    if self.attn_block == 'cbam':
       x = self.cbam_1(x)
    elif self.attn_block == 'cbwssam':
       x = self.cbwssam_1(x)

    x = self.single_block_4(x)
    x = self.single_block_5(x)
    x = self.double_block_1(x)

    if self.attn_block == 'cbam':
       x = self.cbam_2(x)
    elif self.attn_block == 'cbwssam':
       x = self.cbwssam_2(x)

    x = self.double_block_2(x)
    x = self.double_block_3(x)
    x = self.double_block_4(x)

    if self.attn_block == 'cbam':
       x = self.cbam_3(x)
    elif self.attn_block == 'cbwssam':
       x = self.cbwssam_3(x)

    x = self.double_block_5(x)
    x = self.double_block_6(x)
    x = self.gap(x)
    x = self.dense(x)
    return x

'''--------------------------------------------------------End of AgriBlazeNet Model--------------------------------------------------------------------------------'''
