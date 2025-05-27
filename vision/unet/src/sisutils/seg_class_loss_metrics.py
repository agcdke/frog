'''
Acknowledgement: https://github.com/shruti-jadon/Semantic-Segmentation-Loss-Functions
'''
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.losses import binary_crossentropy, BinaryCrossentropy

# beta = 0.25, alpha = 0.25, gamma = 2, epsilon = 1e-5, smooth = 1

@tf.keras.saving.register_keras_serializable()
class DiceCoef(tf.keras.losses.Loss):
    def __init__(self, reduction=tf.keras.losses.Reduction.AUTO, name='dice_coef'):
        print("Dice coefficient initialized")
        super(DiceCoef, self).__init__(reduction=reduction, name=name)

    def get_config(self):
        config = super().get_config()
        return config

    def call(self, y_true, y_pred):
        # Added by Arnab: https://github.com/qubvel/segmentation_models/issues/592
        y_true = tf.cast(y_true, y_pred.dtype)
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        intersection = K.sum(y_true_f * y_pred_f)
        return (2. * intersection + K.epsilon()) / (K.sum(y_true_f) + K.sum(y_pred_f) + K.epsilon())

@tf.keras.saving.register_keras_serializable()
class Sensitivity(tf.keras.losses.Loss):
    def __init__(self, reduction=tf.keras.losses.Reduction.AUTO, name='sensitivity'):
        print("Sensitivity initialized")
        super(Sensitivity, self).__init__(reduction=reduction, name=name)

    def get_config(self):
        config = super().get_config()
        return config

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, y_pred.dtype) # Added
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        return true_positives / (possible_positives + K.epsilon())

@tf.keras.saving.register_keras_serializable()
class Specificity(tf.keras.losses.Loss):
    def __init__(self, reduction=tf.keras.losses.Reduction.AUTO, name='specificity'):
        print("Specificity initialized")
        super(Specificity, self).__init__(reduction=reduction, name=name)

    def get_config(self):
        config = super().get_config()
        return config

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, y_pred.dtype) # Added
        true_negatives = K.sum(K.round(K.clip((1 - y_true) * (1 - y_pred), 0, 1)))
        possible_negatives = K.sum(K.round(K.clip(1 - y_true, 0, 1)))
        return true_negatives / (possible_negatives + K.epsilon())

@tf.keras.saving.register_keras_serializable()
class JacardSimilarity(tf.keras.losses.Loss):
    def __init__(self, reduction=tf.keras.losses.Reduction.AUTO, name='jacard_similarity'):
        print("Jacard similarity initialized")
        super(JacardSimilarity, self).__init__(reduction=reduction, name=name)

    def get_config(self):
        config = super().get_config()
        return config

    def call(self, y_true, y_pred):
        """
         Intersection-Over-Union (IoU), also known as the Jaccard Index
        """
        y_true = tf.cast(y_true, y_pred.dtype) # Added
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)

        intersection = K.sum(y_true_f * y_pred_f)
        union = K.sum((y_true_f + y_pred_f) - (y_true_f * y_pred_f))
        return intersection / union

@tf.keras.saving.register_keras_serializable()
class WeightedCrossEntropyLoss(tf.keras.losses.Loss):
    def __init__(self, beta=0.25, reduction=tf.keras.losses.Reduction.AUTO, name='weighted_ce_loss'):
        print("Weighted cross-entropy loss initialized")
        super(WeightedCrossEntropyLoss, self).__init__(reduction=reduction, name=name)
        self.beta = beta

    def get_config(self):
        config = super().get_config()
        return config

    def convert_to_logits(self, y_pred):
        y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(),
                                  1 - tf.keras.backend.epsilon())
        return tf.math.log(y_pred / (1 - y_pred))


    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, y_pred.dtype) # Added
        y_pred = self.convert_to_logits(y_pred)
        pos_weight = self.beta / (1 - self.beta)
        loss = tf.nn.weighted_cross_entropy_with_logits(logits=y_pred,
                                                        targets=y_true,
                                                        pos_weight=pos_weight)
        return tf.reduce_mean(loss)


@tf.keras.saving.register_keras_serializable()
class DepthSoftmax(tf.keras.losses.Loss):
    def __init__(self, reduction=tf.keras.losses.Reduction.AUTO, name='depth_softmax_loss'):
        print("Depth softmax initialized")
        super(DepthSoftmax, self).__init__(reduction=reduction, name=name)

    def get_config(self):
        config = super().get_config()
        return config

    def call(self, matrix):
        sigmoid = lambda x: 1 / (1 + K.exp(-x))
        sigmoided_matrix = sigmoid(matrix)
        softmax_matrix = sigmoided_matrix / K.sum(sigmoided_matrix, axis=0)
        return softmax_matrix

@tf.keras.saving.register_keras_serializable()
class DiceLoss(tf.keras.losses.Loss):
    def __init__(self, smooth = 1., reduction=tf.keras.losses.Reduction.AUTO, name='dice_loss'):
        print("Dice loss initialized")
        super(DiceLoss, self).__init__(reduction=reduction, name=name)
        self.smooth=smooth

    def get_config(self):
        config = super().get_config()
        return config

    def generalized_dice_coefficient(self, y_true, y_pred):
        y_true = tf.cast(y_true, y_pred.dtype) # Added
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        intersection = K.sum(y_true_f * y_pred_f)
        score = (2. * intersection + self.smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + self.smooth)
        return score

    def call(self, y_true, y_pred):
        loss = 1 - self.generalized_dice_coefficient(y_true, y_pred)
        return loss


@tf.keras.saving.register_keras_serializable()
class TverskyLoss(tf.keras.losses.Loss):
    def __init__(self, smooth = 1., alpha = 0.7, reduction=tf.keras.losses.Reduction.AUTO, name='tversky_loss'):
        print("Tversky loss initialized")
        super(TverskyLoss, self).__init__(reduction=reduction, name=name)
        self.smooth=smooth
        self.alpha = alpha

    def get_config(self):
        config = super().get_config()
        return config

    def tversky_index(self, y_true, y_pred):
        y_true = tf.cast(y_true, y_pred.dtype) # Added
        y_true_pos = K.flatten(y_true)
        y_pred_pos = K.flatten(y_pred)
        true_pos = K.sum(y_true_pos * y_pred_pos)
        false_neg = K.sum(y_true_pos * (1 - y_pred_pos))
        false_pos = K.sum((1 - y_true_pos) * y_pred_pos)

        return (true_pos + self.smooth) / (true_pos + alpha * false_neg + (
                1 - self.alpha) * false_pos + self.smooth)

    def call(self, y_true, y_pred):
        return 1 - self.tversky_index(y_true, y_pred)

'''
class FocalTverskyLoss(TverskyLoss):
    def __init__(self, gamma = 0.75):
        print("Focal Tversky loss initialized")
        self.gamma=gamma
        
    def get_config(self):
        config = super().get_config()
        return config 
        
    def call(self, y_true, y_pred):
        pt_1 = self.tversky_index(y_true, y_pred)
        
        return K.pow((1 - pt_1), gamma)

@tf.keras.saving.register_keras_serializable()
class LogCoshDiceLoss(DiceLoss):
    def __init__(self):
        print("Log cosh dice loss initialized")
        
    def get_config(self):
        config = super().get_config()
        return config
        
    def call(self, y_true, y_pred):
        x = self.dice_loss(y_true, y_pred)
        return tf.math.log((tf.exp(x) + tf.exp(-x)) / 2.0)
'''

@tf.keras.saving.register_keras_serializable()
class Unet3pHybridLoss(tf.keras.losses.Loss):
    def __init__(self, alpha = 0.25, gamma = 2, reduction=tf.keras.losses.Reduction.AUTO, name='unet3p_hybrid_loss'):
        print("Unet3p hybrid loss initialized")
        self.alpha=alpha
        self.gamma = gamma
        super(Unet3pHybridLoss, self).__init__(reduction=reduction, name=name)

    def get_config(self):
        config = super().get_config()
        return config

    def jacard_similarity(self, y_true, y_pred):
        """
         Intersection-Over-Union (IoU), also known as the Jaccard Index
        """
        y_true = tf.cast(y_true, y_pred.dtype) # Added
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)

        intersection = K.sum(y_true_f * y_pred_f)
        union = K.sum((y_true_f + y_pred_f) - (y_true_f * y_pred_f))
        return intersection / union

    def jacard_loss(self, y_true, y_pred):
        """
         Intersection-Over-Union (IoU), also known as the Jaccard loss
        """
        return 1 - self.jacard_similarity(y_true, y_pred)

    def ssim_loss(self, y_true, y_pred):
        """
        Structural Similarity Index (SSIM) loss
        """
        y_true = tf.cast(y_true, y_pred.dtype) # Added
        return 1 - tf.image.ssim(y_true, y_pred, max_val=1)

    def focal_loss_with_logits(self, logits, targets, y_pred):
        weight_a = self.alpha * (1 - y_pred) ** self.gamma * targets
        weight_b = (1 - self.alpha) * y_pred ** self.gamma * (1 - targets)

        return (tf.math.log1p(tf.exp(-tf.abs(logits))) + tf.nn.relu(-logits)) * (weight_a + weight_b) + logits * weight_b

    def focal_loss(self, y_true, y_pred):
        y_true = tf.cast(y_true, y_pred.dtype) # Added
        y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(),
                                  1 - tf.keras.backend.epsilon())
        logits = tf.math.log(y_pred / (1 - y_pred))

        loss = self.focal_loss_with_logits(logits=logits, targets=y_true, y_pred=y_pred)

        return tf.reduce_mean(loss)

    def call(self, y_true, y_pred):
        """
        Hybrid loss proposed in UNET 3+ (https://arxiv.org/ftp/arxiv/papers/2004/2004.08790.pdf)
        Hybrid loss for segmentation in three-level hierarchy – pixel, patch and map-level,
        which is able to capture both large-scale and fine structures with clear boundaries.
        """
        y_true = tf.cast(y_true, y_pred.dtype) # Added
        focal_loss = self.focal_loss(y_true, y_pred)
        ms_ssim_loss = self.ssim_loss(y_true, y_pred)
        jacard_loss = self.jacard_loss(y_true, y_pred)

        return focal_loss + ms_ssim_loss + jacard_loss
