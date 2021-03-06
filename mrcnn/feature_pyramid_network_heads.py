import os
import datetime
import re
import math
from collections import OrderedDict
import multiprocessing
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.backend as backend
import tensorflow.keras.layers as layers
#import tensorflow.keras.utils as utils         #Avoid double use of utils  ==> use keras.utils
from tensorflow.python.eager import context
import tensorflow.keras.models as models
from mrcnn.roialign_layer import PyramidROIAlign

#import utils
############################################################
#  Feature Pyramid Network Heads
############################################################

def fpn_classifier_graph(rois, feature_maps, image_meta,
                         pool_size, num_classes, train_bn=True,
                         fc_layers_size=1024):
    """Builds the computation graph of the feature pyramid network classifier
    and regressor heads.

    rois: [batch, num_rois, (y1, x1, y2, x2)] Proposal boxes in normalized
          coordinates.
    feature_maps: List of feature maps from different layers of the pyramid,
                  [P2, P3, P4, P5]. Each has a different resolution.
    image_meta: [batch, (meta data)] Image details. See compose_image_meta()
    pool_size: The width of the square feature map generated from ROI Pooling.
    num_classes: number of classes, which determines the depth of the results
    train_bn: Boolean. Train or freeze Batch Norm layers
    fc_layers_size: Size of the 2 FC layers

    Returns:
        logits: [batch, num_rois, NUM_CLASSES] classifier logits (before softmax)
        probs: [batch, num_rois, NUM_CLASSES] classifier probabilities
        bbox_deltas: [batch, num_rois, NUM_CLASSES, (dy, dx, log(dh), log(dw))] Deltas to apply to
                     proposal boxes
    """
    # Delete all x = layers.TimeDistributed(BatchNorm(), ..) since we don't use BatchNormalization
    # ROI Pooling
    # Shape: [batch, num_rois, POOL_SIZE, POOL_SIZE, channels]
    x = PyramidROIAlign([pool_size, pool_size],
                        name="roi_align_classifier")([rois, image_meta] + feature_maps)
    # Two 1024 FC layers (implemented with Conv2D for consistency)
    x = layers.TimeDistributed(layers.Conv2D(fc_layers_size, (pool_size, pool_size), padding="valid"),
                               name="mrcnn_class_conv1")(x)
    x = layers.Activation('relu')(x)
    x = layers.TimeDistributed(layers.Conv2D(fc_layers_size, (1, 1)),
                               name="mrcnn_class_conv2")(x)
    x = layers.Activation('relu')(x)

    shared = layers.Lambda(lambda x: backend.squeeze(backend.squeeze(x, 3), 2),
                           name="pool_squeeze")(x)

    # Classifier head
    mrcnn_class_logits = layers.TimeDistributed(layers.Dense(num_classes),
                                                name='mrcnn_class_logits')(shared)
    mrcnn_probs = layers.TimeDistributed(layers.Activation("softmax"),
                                         name="mrcnn_class")(mrcnn_class_logits)

    # BBox head
    # [batch, num_rois, NUM_CLASSES * (dy, dx, log(dh), log(dw))]
    x = layers.TimeDistributed(layers.Dense(num_classes * 4, activation='linear'),
                               name='mrcnn_bbox_fc')(shared)
    # Reshape to [batch, num_rois, NUM_CLASSES, (dy, dx, log(dh), log(dw))]
    s = backend.int_shape(x)
    if s[1] is None:
        mrcnn_bbox = layers.Reshape((-1, num_classes, 4), name="mrcnn_bbox")(x)
    else:
        mrcnn_bbox = layers.Reshape((s[1], num_classes, 4), name="mrcnn_bbox")(x)

    return mrcnn_class_logits, mrcnn_probs, mrcnn_bbox


def build_fpn_mask_graph(rois, feature_maps, image_meta,
                         pool_size, num_classes, train_bn=True):
    """Builds the computation graph of the mask head of Feature Pyramid Network.

    rois: [batch, num_rois, (y1, x1, y2, x2)] Proposal boxes in normalized
          coordinates.
    feature_maps: List of feature maps from different layers of the pyramid,
                  [P2, P3, P4, P5]. Each has a different resolution.
    image_meta: [batch, (meta data)] Image details. See compose_image_meta()
    pool_size: The width of the square feature map generated from ROI Pooling.
    num_classes: number of classes, which determines the depth of the results
    train_bn: Boolean. Train or freeze Batch Norm layers

    Returns: Masks [batch, num_rois, MASK_POOL_SIZE, MASK_POOL_SIZE, NUM_CLASSES]
    """
    # Delete all x = layers.TimeDistributed(BatchNorm(), ..) since we don't use BatchNormalization
    # ROI Pooling
    # Shape: [batch, num_rois, MASK_POOL_SIZE, MASK_POOL_SIZE, channels]
    x = PyramidROIAlign([pool_size, pool_size],
                        name="roi_align_mask")([rois, image_meta] + feature_maps)

    # Conv layers
    x = layers.TimeDistributed(layers.Conv2D(256, (3, 3), padding="same"),
                               name="mrcnn_mask_conv1")(x)

    x = layers.Activation('relu')(x)

    x = layers.TimeDistributed(layers.Conv2D(256, (3, 3), padding="same"),
                               name="mrcnn_mask_conv2")(x)

    x = layers.Activation('relu')(x)

    x = layers.TimeDistributed(layers.Conv2D(256, (3, 3), padding="same"),
                               name="mrcnn_mask_conv3")(x)

    x = layers.Activation('relu')(x)

    x = layers.TimeDistributed(layers.Conv2D(256, (3, 3), padding="same"),
                               name="mrcnn_mask_conv4")(x)

    x = layers.Activation('relu')(x)

    x = layers.TimeDistributed(layers.Conv2DTranspose(256, (2, 2), strides=2, activation="relu"),
                               name="mrcnn_mask_deconv")(x)
    x = layers.TimeDistributed(layers.Conv2D(num_classes, (1, 1), strides=1, activation="sigmoid"),
                               name="mrcnn_mask")(x)
    return x



