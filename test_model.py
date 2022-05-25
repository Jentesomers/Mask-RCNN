
import os
import random
import sys
import json
import datetime
import numpy as np
import skimage.draw
import cv2
from mrcnn.visualize import display_instances
from mrcnn import visualize
import matplotlib.pyplot as plt
from tensorflow.python.keras.callbacks import TensorBoard
from time import time
import tensorflow as tf
#from mrcnn.data_generator_and_formatting import load_image_gt, load_image


# Root directory of the project
ROOT_DIR = r"/host/Mask-RCNN"

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs/test")


model_path = os.path.join(ROOT_DIR, "logs/object20220525T1444/mask_rcnn_object_0003.h5")

# Directory of images to run detection on
IMAGE_DIR = os.path.join(ROOT_DIR, "images")

class CustomConfig(Config):
    """Configuration for training on the custom  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "object"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 3  # Background + beer, foam and foam_beer

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 10

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.

config = CustomConfig()

# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config, s=0, checkp_path='test1')

# Load trained weights
print("Loading weights from ", model_path)
model.load_weights(model_path, by_name=True)


# Run object detection

# Load a random image from the images folder
file_names = next(os.walk(IMAGE_DIR))[2]
image = skimage.io.imread(os.path.join(IMAGE_DIR, random.choice(file_names)))
#filename = os.path.join(IMAGE_DIR, 'bier2.jpg')
#image = skimage.io.imread(filename)

# Run detection
results = model.detect([image], verbose=1)

class_names = {"beer", "foam", "foam_beer"}

# Visualize results
r = results[0]
visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
                            class_names, r['scores'])







# # Evaluation on test set
#     class InferenceConfig(CustomConfig):        # Opportunity to overrite configurations
#         GPU_COUNT = 1
#         IMAGES_PER_GPU = 1
#         USE_MINI_MASK  = False
#
#
#
#     inference_config = InferenceConfig()
#
#     # Recreate the model in inference mode
#     model = modellib.MaskRCNN(mode="inference",
#                               config=inference_config,
#                               model_dir=DEFAULT_LOGS_DIR, s=0)  #Set standard deviation for gaussian noise to zero so no augmention will be performed when calling the function load_img_gt
#
#     # Get path to saved weights
#     # Either set a specific path or find last trained weights
#     # model_path = os.path.join(ROOT_DIR, ".h5 file name here")
#     model_path = model.find_last()
#
#     # Load trained weights
#     #print("Loading weights from ", model_path)
#     model.load_weights(model_path, by_name=True)
#
#     # Compute VOC-Style mAP @ IoU=0.5
#     # Running on test set
#
#     #load test set
#     # Validation dataset
#     dataset_test = CustomDataset()
#     dataset_test.load_custom(dataset_dir, "test")        #Still need to put test set in dataset (and export annotations as json from VIA)
#     dataset_test.prepare()
#
#     test_image_ids = dataset_test.image_ids
#
#     # AP (Average precision) is a popular metric in measuring the accuracy of object detectors. IoU (Intersection over union) measures the overlap between 2 boundaries
#     APs = []
#     for image_id in test_image_ids:
#         # Load image and ground truth data
#         image, image_meta, gt_class_id, gt_bbox, gt_mask = \
#             load_image_gt(dataset_test, inference_config,
#                                    image_id)  # , use_mini_mask=False)
#         molded_images = np.expand_dims(mold_image(image, inference_config), 0)
#         # Run object detection
#         results = model.detect([image], verbose=0)
#         r = results[0]
#         # Compute AP
#         AP, precisions, recalls, overlaps = \
#             utils.compute_ap(gt_bbox, gt_class_id, gt_mask,
#                              r["rois"], r["class_ids"], r["scores"], r['masks'])
#         APs.append(AP)
#
#     meanAP = np.mean(APs)
#
#     print("mAP: ", meanAP)
