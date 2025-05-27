################################################################################
# AgriBlazeU-Net
################################################################################

import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import (Input, GlobalAveragePooling2D, GlobalMaxPooling2D, Reshape, Dense, Multiply, Flatten, MaxPooling2D,
                                     AveragePooling2D, Dropout, BatchNormalization, LayerNormalization, DepthwiseConv2D, Permute, Layer,
                                     Concatenate, Conv2D, Conv1D, Add, Activation, Lambda, Rescaling, Normalization, MultiHeadAttention,
                                     Embedding, ReLU, LeakyReLU, Conv2DTranspose)
from tensorflow.keras import activations
from tensorflow.keras import Model, Sequential
from tensorflow.keras.utils import plot_model
import sys, math
sys.path.append("..")

'''--------------------------------------------------------------------CBAM--------------------------------------------------------------------'''
@tf.keras.utils.register_keras_serializable()
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

@tf.keras.utils.register_keras_serializable()
class ChannelwiseAvgPool2D(Layer):
    def __init__(self, axis=3, **kwargs):
        super().__init__(**kwargs)
        self.axis = axis

    def call(self, inputs):
        return tf.math.reduce_mean(inputs, self.axis, keepdims=True)

@tf.keras.utils.register_keras_serializable()
class ChannelwiseMaxPool2D(Layer):
    def __init__(self, axis=3, **kwargs):
        super().__init__(**kwargs)
        self.axis = axis

    def call(self, inputs):
        return tf.math.reduce_max(inputs, self.axis, keepdims=True)

@tf.keras.utils.register_keras_serializable()
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

@tf.keras.utils.register_keras_serializable()
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
@tf.keras.utils.register_keras_serializable()
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

@tf.keras.utils.register_keras_serializable()
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
@tf.keras.utils.register_keras_serializable()
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
@tf.keras.utils.register_keras_serializable()
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
'''
Please select appropriate option MANUALLY for 'cbam' parameter, .e.g, cbam='mvit_h4b4'.
When invoke 'cbwssam: T1H4B4' module, set cbam='mvit_h4b4'. Otherwise, when invoke 'cbwssam: T2H4B4' module, set cbam='cbmvit_h4b4'.
Options mvit_h8b8, cbmvit_h8b8 used for 8 stacks of ViT encoder and 8 heads as multi-head attention. But "h8b8" is not part of the conference paper.
'''
@tf.keras.utils.register_keras_serializable()
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
@tf.keras.utils.register_keras_serializable()
class SingleBlazeBlock(Layer):
  def __init__(self, filters, strides=1, act_name='relu', **kwargs):
    super(SingleBlazeBlock, self).__init__()
    self.act_name = act_name
    self.strides = strides
    self.filters = filters
    if strides == 2:
      self.pool = MaxPooling2D()

    self.dw_conv_1 = DepthwiseConv2D(kernel_size=(5, 5), strides=strides, padding="same")
    self.proj_conv_1 = Conv2D(filters, kernel_size=(1, 1), strides=(1, 1), padding="same")
    self.norm_1 = BatchNormalization()

    if self.act_name == 'relu':
      self.activation_1 = ReLU()
    if self.act_name== 'lrelu':
      self.activation_1 = LeakyReLU()

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

    if self.act_name == 'relu':
      x = self.activation_1(x)
    if self.act_name== 'lrelu':
      x = self.activation_1(x)

    return x

@tf.keras.utils.register_keras_serializable()
class DoubleBlazeBlock(Layer):
  def __init__(self, proj_filters, expand_filters, strides=1, act_name = 'relu', **kwargs):
    super(DoubleBlazeBlock, self).__init__()
    self.act_name = act_name
    self.strides = strides
    self.proj_filters = proj_filters
    self.expand_filters = expand_filters
    if strides == 2:
      self.pool = MaxPooling2D()

    # Project
    self.dw_conv_1 = DepthwiseConv2D(kernel_size=(5, 5), strides=strides, padding="same")
    self.proj_conv_1 = Conv2D(proj_filters, kernel_size=(1, 1), strides=(1, 1), padding="same")
    self.norm_1 = BatchNormalization()

    if self.act_name == 'relu':
      self.activation_1 = ReLU()
    if self.act_name== 'lrelu':
      self.activation_1 = LeakyReLU()

    # Expand (always strides=1)
    self.dw_conv_2 = DepthwiseConv2D(kernel_size=(5, 5), strides=1, padding="same")
    self.proj_conv_2 = Conv2D(expand_filters, kernel_size=(1, 1), strides=(1, 1), padding="same")
    self.norm_2 = BatchNormalization()

    if self.act_name == 'relu':
      self.activation_2 = ReLU()
    if self.act_name== 'lrelu':
      self.activation_2 = LeakyReLU()

  def __call__(self, input_feature):
    if not self.built:
      self.build(input_feature.shape)
      self.built = True
    return self.call(input_feature)

  def call(self, x_input):
    x = self.dw_conv_1(x_input)
    x = self.proj_conv_1(x)

    if self.act_name == 'relu':
      x = self.activation_1(x)
    if self.act_name== 'lrelu':
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

    if self.act_name == 'relu':
      x = self.activation_2(x)
    if self.act_name== 'lrelu':
      x = self.activation_2(x)

    return x

'''--------------------------------------------------------Blaze Image Classification Encoder for Barlow Twins--------------------------------------------------------------------------------'''
@tf.keras.utils.register_keras_serializable()
class BlazeBTEncoder(Model):
  def __init__(self, no_classes, img_mean, img_std, attn_block=None, cbam_block=None, **kwargs):
    super(BlazeImgClsModel, self).__init__(**kwargs)
    self.attn_block = attn_block
    self.cbam_block = cbam_block
    self.feature_norm = Normalization(mean=img_mean, variance=img_std)
    self.conv_1 = Conv2D(filters=24, kernel_size=(5, 5), strides=2, padding="same")
    self.activation_1 = ReLU()
    self.single_block_1 = SingleBlazeBlock(filters=24)
    self.single_block_2 = SingleBlazeBlock(filters=24)
    self.single_block_3 = SingleBlazeBlock(filters=48, strides=2)

    if self.attn_block == 'cbam':
      self.cbam_1 = Cbam2D()
    elif self.attn_block == 'cbwssam':
      if self.cbam_block == 'mvit_h4b4':
        self.cbwssam_1 = Cbwssam2D(projection_dim=48, ff_dim=48, cbam='mvit_h4b4')
      elif self.cbam_block == 'cbmvit_h4b4':
        self.cbwssam_1 = Cbwssam2D(projection_dim=48, ff_dim=48, cbam='cbmvit_h4b4')

    self.single_block_4 = SingleBlazeBlock(filters=48)
    self.single_block_5 = SingleBlazeBlock(filters=48, strides=2)
    self.double_block_1 = DoubleBlazeBlock(proj_filters=24, expand_filters=96, strides=2)

    if self.attn_block == 'cbam':
      self.cbam_2 = Cbam2D()
    elif self.attn_block == 'cbwssam':
      if self.cbam_block == 'mvit_h4b4':
        self.cbwssam_2 = Cbwssam2D(projection_dim=96, ff_dim=96, cbam='mvit_h4b4')
      elif self.cbam_block == 'cbmvit_h4b4':
        self.cbwssam_2 = Cbwssam2D(projection_dim=96, ff_dim=96, cbam='cbmvit_h4b4')

    self.double_block_2 = DoubleBlazeBlock(proj_filters=24, expand_filters=96)
    self.double_block_3 = DoubleBlazeBlock(proj_filters=24, expand_filters=96)
    self.double_block_4 = DoubleBlazeBlock(proj_filters=24, expand_filters=96, strides=2)

    if self.attn_block == 'cbam':
      self.cbam_3 = Cbam2D()
    elif self.attn_block == 'cbwssam':
      if self.cbam_block == 'mvit_h4b4':
        self.cbwssam_3 = Cbwssam2D(projection_dim=96, ff_dim=96, cbam='mvit_h4b4')
      elif self.cbam_block == 'cbmvit_h4b4':
        self.cbwssam_3 = Cbwssam2D(projection_dim=96, ff_dim=96, cbam='cbmvit_h4b4')

    self.double_block_5 = DoubleBlazeBlock(proj_filters=24, expand_filters=96)
    self.double_block_6 = DoubleBlazeBlock(proj_filters=24, expand_filters=96)
    self.gap = GlobalAveragePooling2D()
    self.dense = Dense(units=no_classes, activation = 'softmax')

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

'''--------------------------------------------------------U-Net Semantic Segmentation Model-----------------------------------------------------------------------'''
@tf.keras.utils.register_keras_serializable()
class BridgeBlock(Layer):
  def __init__(self, filters, act_name='relu', **kwargs):
    super(BridgeBlock, self).__init__()
    self.act_name = act_name
    self.dw_conv_1 = DepthwiseConv2D(kernel_size=(5, 5), strides=(1,1), padding="same")
    self.proj_conv_1 = Conv2D(filters, kernel_size=(1, 1), strides=(1, 1), padding="same")
    self.norm_1 = BatchNormalization()

    if self.act_name == 'relu':
      self.activation_1 = ReLU()
    if self.act_name== 'lrelu':
      self.activation_1 = LeakyReLU()

  def __call__(self, input_feature):
    if not self.built:
      self.build(input_feature.shape)
      self.built = True
    return self.call(input_feature)

  def call(self, x_input):
    x = self.dw_conv_1(x_input)
    x = self.proj_conv_1(x)
    x = self.norm_1(x)

    if self.act_name == 'relu':
      x = self.activation_1(x)
    if self.act_name== 'lrelu':
      x = self.activation_1(x)

    return x

@tf.keras.utils.register_keras_serializable()
class ExpandConvBlock(Layer):
  def __init__(self, filters, act_name='relu', **kwargs):
    super(ExpandConvBlock, self).__init__()
    self.act_name = act_name
    self.dw_conv_1 = DepthwiseConv2D(kernel_size=(5, 5), strides=(1,1), padding="same")
    self.proj_conv_1 = Conv2D(filters, kernel_size=(1, 1), strides=(1, 1), padding="same") # strides=(2, 2), when ConvT padding="same". strides=(1, 1),when ConvT padding="valid"
    self.norm_1 = BatchNormalization()

    if self.act_name == 'relu':
      self.activation_1 = ReLU()
    if self.act_name== 'lrelu':
      self.activation_1 = LeakyReLU()

    self.dw_conv_2 = DepthwiseConv2D(kernel_size=(5, 5), strides=(1, 1), padding="same")
    self.proj_conv_2 = Conv2D(filters, kernel_size=(1, 1), strides=(1, 1), padding="same")
    self.norm_2 = BatchNormalization()

    if self.act_name == 'relu':
      self.activation_2 = ReLU()
    if self.act_name== 'lrelu':
      self.activation_2 = LeakyReLU()

  def __call__(self, input_feature):
    if not self.built:
      self.build(input_feature.shape)
      self.built = True
    return self.call(input_feature)

  def call(self, x_input):
    x = self.dw_conv_1(x_input)
    x = self.proj_conv_1(x)
    x = self.norm_1(x)

    if self.act_name=='relu':
      x = self.activation_1(x)
    if self.act_name== 'lrelu':
      x = self.activation_1(x)

    x = self.dw_conv_2(x)
    x = self.proj_conv_2(x)
    x = self.norm_2(x)

    if self.act_name == 'relu':
      x = self.activation_2(x)
    if self.act_name== 'lrelu':
      x = self.activation_2(x)

    return x

# act_output: sigmoid(for binary-class), softmax(for multi-class)
@tf.keras.utils.register_keras_serializable()
class BlazeUnet(Model):
  def __init__(self, num_classes, drouput_val=0.2, act_output='softmax', attn_block=None, cbam_block=None, **kwargs):
    super(BlazeUnet, self).__init__(**kwargs)
    self.num_classes = num_classes
    self.drouput_val = drouput_val
    self.act_output = act_output
    self.attn_block = attn_block
    self.cbam_block = cbam_block
    # Contracting Block-1 (256x256)
    self.single_cblock_1_1 = SingleBlazeBlock(filters=24, act_name = 'lrelu')
    self.single_cblock_1_2 = SingleBlazeBlock(filters=24, act_name = 'lrelu') # Skip-1
    self.dropout_cblock_1 = Dropout(drouput_val)

    # Contracting Block-2 (128x128)
    self.single_cblock_2_1 = SingleBlazeBlock(filters=24, strides=2, act_name = 'lrelu') # Downsample-1
    if self.attn_block == 'cbam':
      self.cbam_cblock_2 = Cbam2D()
    elif self.attn_block == 'cbwssam':
      if self.cbam_block == 'mvit_h4b4':
        self.cbwssam_cblock_2 = Cbwssam2D(projection_dim=24, ff_dim=24, cbam='mvit_h4b4')
      elif self.cbam_block == 'cbmvit_h4b4':
        self.cbwssam_cblock_2 = Cbwssam2D(projection_dim=24, ff_dim=24, cbam='cbmvit_h4b4')
      elif self.cbam_block == 'mvit_h8b8':
        self.cbwssam_cblock_2 = Cbwssam2D(projection_dim=24, ff_dim=24, cbam='mvit_h8b8')
      elif self.cbam_block == 'cbmvit_h8b8':
        self.cbwssam_cblock_2 = Cbwssam2D(projection_dim=24, ff_dim=24, cbam='cbmvit_h8b8')

    self.single_cblock_2_2 = SingleBlazeBlock(filters=48, act_name = 'lrelu')
    self.single_cblock_2_3 = SingleBlazeBlock(filters=48, act_name = 'lrelu') # Skip-2
    self.dropout_cblock_2 = Dropout(drouput_val)

    # Contracting Block-3 (64x64)
    self.single_cblock_3_1 = SingleBlazeBlock(filters=48, strides=2, act_name = 'lrelu') # Downsample-2
    if self.attn_block == 'cbam':
      self.cbam_cblock_3 = Cbam2D()
    elif self.attn_block == 'cbwssam':
      if self.cbam_block == 'mvit_h4b4':
        self.cbwssam_cblock_3 = Cbwssam2D(projection_dim=48, ff_dim=48, cbam='mvit_h4b4')
      elif self.cbam_block == 'cbmvit_h4b4':
        self.cbwssam_cblock_3 = Cbwssam2D(projection_dim=48, ff_dim=48, cbam='cbmvit_h4b4')
      elif self.cbam_block == 'mvit_h8b8':
        self.cbwssam_cblock_3 = Cbwssam2D(projection_dim=48, ff_dim=48, cbam='mvit_h8b8')
      elif self.cbam_block == 'cbmvit_h8b8':
        self.cbwssam_cblock_3 = Cbwssam2D(projection_dim=48, ff_dim=48, cbam='cbmvit_h8b8')

    self.double_cblock_3_2 = DoubleBlazeBlock(proj_filters=24, expand_filters=96, act_name = 'lrelu')
    self.double_cblock_3_3 = DoubleBlazeBlock(proj_filters=24, expand_filters=96, act_name = 'lrelu') # Skip-3
    self.dropout_cblock_3 = Dropout(drouput_val)

    # Contracting Block-4(32x32)
    self.double_cblock_4_1 = DoubleBlazeBlock(proj_filters=24, expand_filters=96, strides=2, act_name = 'lrelu') # Downsample-3
    if self.attn_block == 'cbam':
      self.cbam_cblock_4 = Cbam2D()
    elif self.attn_block == 'cbwssam':
      if self.cbam_block == 'mvit_h4b4':
        self.cbwssam_cblock_4 = Cbwssam2D(projection_dim=96, ff_dim=96, cbam='mvit_h4b4')
      elif self.cbam_block == 'cbmvit_h4b4':
        self.cbwssam_cblock_4 = Cbwssam2D(projection_dim=96, ff_dim=96, cbam='cbmvit_h4b4')
      elif self.cbam_block == 'mvit_h8b8':
        self.cbwssam_cblock_4 = Cbwssam2D(projection_dim=96, ff_dim=96, cbam='mvit_h8b8')
      elif self.cbam_block == 'cbmvit_h8b8':
        self.cbwssam_cblock_4 = Cbwssam2D(projection_dim=96, ff_dim=96, cbam='cbmvit_h8b8')

    self.double_block_4_2 = DoubleBlazeBlock(proj_filters=48, expand_filters=192, act_name = 'lrelu')
    self.double_block_4_3 = DoubleBlazeBlock(proj_filters=48, expand_filters=192, act_name = 'lrelu') # Skip-4
    self.dropout_cblock_4 = Dropout(drouput_val)

    # Bridge Block - filters: 192 (16x16)
    self.double_bblock_1 = DoubleBlazeBlock(proj_filters=48, expand_filters=192, strides=2, act_name = 'lrelu') # Downsample-4
    if self.attn_block == 'cbam':
      self.cbam_bblock_1 = Cbam2D()
    elif self.attn_block == 'cbwssam':
      if self.cbam_block == 'mvit_h4b4':
        self.cbwssam_bblock_1 = Cbwssam2D(projection_dim=192, ff_dim=192, cbam='mvit_h4b4')
      elif self.cbam_block == 'cbmvit_h4b4':
        self.cbwssam_bblock_1 = Cbwssam2D(projection_dim=192, ff_dim=192, cbam='cbmvit_h4b4')
      elif self.cbam_block == 'mvit_h8b8':
        self.cbwssam_bblock_1 = Cbwssam2D(projection_dim=192, ff_dim=192, cbam='mvit_h8b8')
      elif self.cbam_block == 'cbmvit_h8b8':
        self.cbwssam_bblock_1 = Cbwssam2D(projection_dim=192, ff_dim=192, cbam='cbmvit_h8b8')

    self.mid_bblock_2 = BridgeBlock(filters=192, act_name='lrelu') # middle of Bridge layer
    self.double_bblock_3 = DoubleBlazeBlock(proj_filters=48, expand_filters=192, act_name = 'lrelu') # entry point for Expand layer

    # Expanding Block-4(32x32xSkip-4:192)
    self.convT_exblock_4_1 = Conv2DTranspose(filters=192, kernel_size=(2,2), strides=(2,2), padding="valid")
    self.concat_exblock_4_2 = Concatenate(axis=-1) # Skip-4
    self.expandconv_exblock_4_3 = ExpandConvBlock(filters=192, act_name='lrelu')

    # Expanding Block-3(64x64xSkip-3:96)
    self.convT_exblock_3_1 = Conv2DTranspose(filters=96, kernel_size=(2,2), strides=(2,2), padding="valid")
    self.concat_exblock_3_2 = Concatenate(axis=-1) # Skip-3
    self.expandconv_exblock_3_3 = ExpandConvBlock(filters=96, act_name='lrelu')

    # Expanding Block-2(128x128xSkip-2:48)
    self.convT_exblock_2_1 = Conv2DTranspose(filters=48, kernel_size=(2,2), strides=(2,2), padding="valid")
    self.concat_exblock_2_2 = Concatenate(axis=-1) # Skip-2
    self.expandconv_exblock_2_3 = ExpandConvBlock(filters=48, act_name='lrelu')

    # Expanding Block-1(256x256xSkip-1:24)
    self.convT_exblock_1_1 = Conv2DTranspose(filters=24, kernel_size=(2,2), strides=(2,2), padding="valid")
    self.concat_exblock_1_2 = Concatenate(axis=-1) # Skip-1
    self.expandconv_exblock_1_3 = ExpandConvBlock(filters=24, act_name='lrelu')

    # Output layer(256x256)
    self.output_layer = Conv2D(filters=num_classes, kernel_size=(1, 1), padding="same", activation= act_output)

  def get_config(self):
    config = super().get_config()
    config_params = {
      'num_classes': self.num_classes,
      'attn_block': self.attn_block,
      'cbam_block': self.cbam_block,
      'act_output':self.act_output,
      'drouput_val': self.drouput_val,
    }
    config.update(config_params)
    return config
  '''
  @classmethod
  def from_config(cls, config):
    config["num_classes"] = tf.keras.layers.deserialize(config["num_classes"])
    config["attn_block"] = tf.keras.layers.deserialize(config["attn_block"])
    config["cbam_block"] = tf.keras.layers.deserialize(config["cbam_block"])
    return cls(**config)
  '''
  # def build(self, input_shape):
    # super(BlazeUnet, self).build(input_shape)
    # Create a dummy input tensor to build the model
    # self.build(input_shape)
    # self.built=True # # Custom attribute denoting that the model has been built

  def call(self, x):
    x_batch, x_height, x_width, x_channel = x.shape
    # Contracting Block-1
    x = self.single_cblock_1_1(x)
    x = self.single_cblock_1_2(x)
    skip_1 = x
    x = self.dropout_cblock_1(x)
    # Contracting Block-2
    x = self.single_cblock_2_1(x)
    if self.attn_block == 'cbam':
       x = self.cbam_cblock_2(x)
    elif self.attn_block == 'cbwssam':
       x = self.cbwssam_cblock_2(x)

    x = self.single_cblock_2_2(x)
    x = self.single_cblock_2_3(x)
    skip_2 = x
    x = self.dropout_cblock_2(x)

    # Contracting Block-3
    x = self.single_cblock_3_1(x)
    if self.attn_block == 'cbam':
       x = self.cbam_cblock_3(x)
    elif self.attn_block == 'cbwssam':
       x = self.cbwssam_cblock_3(x)

    x = self.double_cblock_3_2(x)
    x = self.double_cblock_3_3(x)
    skip_3 = x
    x = self.dropout_cblock_3(x)

    # Contracting Block-4
    x = self.double_cblock_4_1(x)
    if self.attn_block == 'cbam':
       x = self.cbam_cblock_4(x)
    elif self.attn_block == 'cbwssam':
       x = self.cbwssam_cblock_4(x)

    x = self.double_block_4_2(x)
    x = self.double_block_4_3(x)
    skip_4 = x
    x = self.dropout_cblock_4(x)

    # Bridge Block
    x = self.double_bblock_1(x)
    if self.attn_block == 'cbam':
       x = self.cbam_bblock_1(x)
    elif self.attn_block == 'cbwssam':
       x = self.cbwssam_bblock_1(x)

    x = self.mid_bblock_2(x)
    x = self.double_bblock_3(x)

    # Expanding Block-4
    x = self.convT_exblock_4_1(x)
    x = self.concat_exblock_4_2([x,skip_4])
    x = self.expandconv_exblock_4_3(x)

    # Expanding Block-3
    x = self.convT_exblock_3_1(x)
    x = self.concat_exblock_3_2([x,skip_3])
    x = self.expandconv_exblock_3_3(x)

    # Expanding Block-2
    x = self.convT_exblock_2_1(x)
    x = self.concat_exblock_2_2([x,skip_2])
    x = self.expandconv_exblock_2_3(x)

    # Expanding Block-1
    x = self.convT_exblock_1_1(x)
    x = self.concat_exblock_1_2([x,skip_1])
    x = self.expandconv_exblock_1_3(x)
    # Output layer
    x = self.output_layer(x)
    return x
'''--------------------------------------------------------End of Blaze architecture--------------------------------------------------------------------------------'''
