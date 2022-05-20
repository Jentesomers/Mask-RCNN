import os
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


# Root directory of the project
ROOT_DIR = r"C:\Users\jente\OneDrive\Documenten\GitHub\Mask-RCNN"

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils


# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")



class CustomConfig(Config):
    """Configuration for training on the custom  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "object"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 2

    # Number of classes (including background)
    NUM_CLASSES = 1 + 3  # Background + beer, foam and foam_beer

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 10

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9

############################################################
#  Dataset
############################################################

class CustomDataset(utils.Dataset):

    def load_custom(self, dataset_dir, subset):
        """Load a subset of the Dog-Cat dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        # Add classes. We have only 3 classes to add.
        self.add_class("object", 1, "beer")
        self.add_class("object", 2, "foam")
        self.add_class("object", 3, "foam_beer")

        # Train or validation dataset?
        assert subset in ["train", "val"]
        dataset_dir = os.path.join(dataset_dir, subset)

        # We mostly care about the x and y coordinates of each region
        annotations = json.load(open(os.path.join(dataset_dir,
                                 'via_region_data.json')))
        #print(annotations)
        annotations = list(annotations.values())  # don't need the dict keys
        #print(annotations)


        # The VIA tool saves images in the JSON even if they don't have any
        # annotations. Skip unannotated images.
        annotations = [a for a in annotations if a['regions']]
        
        # Add images
        for a in annotations:
            # print(a)
            # Get the x, y coordinates of points of the polygons that make up
            # the outline of each object instance. There are stores in the
            # shape_attributes (see json format above)
            polygons = [r['shape_attributes'] for r in a['regions']] 
            objects = [s['region_attributes']['names'] for s in a['regions']]
            #print('Printing objects ..')
            #print("objects:", objects)
            name_dict = {"beer": 1, "foam": 2, "foam_beer": 3}

            # key = tuple(name_dict)
            #num_ids = []
            num_ids = [name_dict[a] for a in objects]
            # for n in objects:
            #
            #     try:
            #         if n['object'] == 'beer':
            #             num_ids.append(1)
            #         elif n['object'] == 'foam':
            #             num_ids.append(2)
            #         elif n['object'] == 'foam_beer':
            #             num_ids.append(3)
            #     except:
            #         pass
     
            # num_ids = [int(n['Event']) for n in objects]
            # load_mask() needs the image size to convert polygons to masks.
            # Unfortunately, VIA doesn't include it in JSON, so we must read
            # the image. This is only managable since the dataset is tiny.
            #print("numids", num_ids)
            image_path = os.path.join(dataset_dir, a['filename'])
            #image = skimage.io.imread(image_path)
            #height, width = image.shape[:2]
            height, width = 1147, 600   #No need to read all the images to get the shape, they're all the same size
            self.add_image(
                "object",  ## for a single class just add the name here
                image_id=a['filename'],  # use file name as a unique image id
                path=image_path,
                width=width, height=height,
                polygons=polygons,
                num_ids=num_ids
                )
        print('Dataset created!')

    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a Dog-Cat dataset image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "object":
            return super(self.__class__, self).load_mask(image_id)

        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]
        info = self.image_info[image_id]
        if info["source"] != "object":
            return super(self.__class__, self).load_mask(image_id)
        num_ids = info['num_ids']
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])],
                        dtype=np.uint8)
        for i, p in enumerate(info["polygons"]):
            # Get indexes of pixels inside the polygon and set them to 1
        	rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])

        	mask[rr, cc, i] = 1

        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID only, we return an array of 1s
        # Map class names to class IDs.
        num_ids = np.array(num_ids, dtype=np.int32)
        return mask, num_ids #np.ones([mask.shape[-1]], dtype=np.int32)

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "object":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)





#Optimize with optuna:
import optuna

def objective(trial):
    config = CustomConfig()
    config.display()  # Display model configuration

    standard_deviation = trial.suggest_float('standard_deviation', 1e-5, 0.5,
                                             log=True)  # Create upper limit for standard deviation of the Gaussian noise added to the images so it can be optimized with optuna, added it as argument to MaskRCNN class (see also model.py)
    model = modellib.MaskRCNN(mode="training", config=config,
                              model_dir=DEFAULT_LOGS_DIR, s=standard_deviation)

    tensorboard = TensorBoard(log_dir="logs/train".format(
        time()))  # open Anaconda Prompt and write tensorboard --logdir=logs/train to visualize
    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)

    weights_path = COCO_WEIGHTS_PATH
    # Download weights file
    if not os.path.exists(weights_path):
        utils.download_trained_weights(weights_path)

    model.load_weights(weights_path, by_name=True, exclude=[
        "mrcnn_class_logits", "mrcnn_bbox_fc",
        "mrcnn_bbox", "mrcnn_mask"])
    print('weights loaded, beginning to train')


    """Train the model."""
    # Training dataset.
    dataset_train = CustomDataset()
    dataset_train.load_custom(r"C:\Users\jente\OneDrive\Documenten\GitHub\Mask-RCNN\Dataset", "train")
    dataset_train.prepare()

    # Validation dataset
    dataset_val = CustomDataset()
    dataset_val.load_custom(r"C:\Users\jente\OneDrive\Documenten\GitHub\Mask-RCNN\Dataset", "val")
    dataset_val.prepare()

    print('Both datasets have been prepared')

    # Load and display random samples                           Better place would be after train(model), but training fails due to memory
    image_ids = np.random.choice(dataset_train.image_ids, 4)
    for image_id in image_ids:
        image = dataset_train.load_image(image_id)
        mask, class_ids = dataset_train.load_mask(image_id)
        visualize.display_top_masks(image, mask, class_ids, dataset_train.class_names)

    # Do training in two stages, first only train weights of top layers, then do fine tuning by training all layers with a lower learning rate
    print("Training network heads..")


    # make parameter for the hyper parameters to optimize with optuna
    epoch = trial.suggest_int('epoch', 5, 35, log=True)
    lr = trial.suggest_float('lr', 1e-5, 1e-1, log=True)
    model.train(dataset_train, dataset_val,
                learning_rate=lr,
                # Set lr as a parameter to be optimized with optuna (no longer taken from config.py)
                custom_callbacks=[tensorboard, callback],  # Add custom callback (in def train() in model.py) for tensorboard and early stopping
                epochs=epoch,
                layers='heads')

    # Optional, fine tune the network
    print("Fine tuning network..")
    model.train(dataset_train, dataset_val,
                learning_rate=lr / 10,
                custom_callbacks=[tensorboard, callback],  # Add custom callback (in def train() in model.py) for tensorboard and early stopping
                epochs=epoch,
                layers="all")



    # Evaluation on test set
    class InferenceConfig(ShapesConfig):        # Opportunity to overrite configurations
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1

    inference_config = InferenceConfig()

    # Recreate the model in inference mode
    model = modellib.MaskRCNN(mode="inference",
                              config=inference_config,
                              model_dir=DEFAULT_LOGS_DIR, s=0)  #Set standard deviation for gaussian noise to zero so no augmention will be performed when calling the function load_img_gt

    # Get path to saved weights
    # Either set a specific path or find last trained weights
    # model_path = os.path.join(ROOT_DIR, ".h5 file name here")
    model_path = model.find_last()

    # Load trained weights
    print("Loading weights from ", model_path)
    model.load_weights(model_path, by_name=True)

    # Compute VOC-Style mAP @ IoU=0.5
    # Running on test set

    #load test set
    # Validation dataset
    dataset_test = CustomDataset()
    dataset_test.load_custom(r"C:\Users\jente\OneDrive\Documenten\GitHub\Mask-RCNN\Dataset", "test")        #Still need to put test set in dataset (and export annotations as json from VIA)
    dataset_test.prepare()

    test_image_ids = dataset_test.image_ids

    # AP (Average precision) is a popular metric in measuring the accuracy of object detectors. IoU (Intersection over union) measures the overlap between 2 boundaries
    APs = []
    for image_id in test_image_ids:
        # Load image and ground truth data
        image, image_meta, gt_class_id, gt_bbox, gt_mask = \
            modellib.load_image_gt(dataset_test, inference_config,
                                   image_id)  # , use_mini_mask=False)
        molded_images = np.expand_dims(modellib.mold_image(image, inference_config), 0)
        # Run object detection
        results = model.detect([image], verbose=0)
        r = results[0]
        # Compute AP
        AP, precisions, recalls, overlaps = \
            utils.compute_ap(gt_bbox, gt_class_id, gt_mask,
                             r["rois"], r["class_ids"], r["scores"], r['masks'])
        APs.append(AP)

    meanAP = np.mean(APs)

    print("mAP: ", meanAP)
    return meanAP


# 3. Create a study object and optimize the objective function.
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=10)