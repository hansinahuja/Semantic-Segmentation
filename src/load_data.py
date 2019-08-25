# Import required libraries
import cv2
import os
import numpy as np

# File paths
DATA_PATH = "../dataset/"
TRAIN_IMAGES_PATH = DATA_PATH + "train/train_images/"
TRAIN_SEGMENTATIONS_PATH = DATA_PATH + "train/train_image_segmentations/"
TEST_IMAGES_PATH = DATA_PATH + "test/test_images/"
TEST_SEGMENTATIONS_PATH = DATA_PATH + "test/test_image_segmentations/"

# Preprocessing parameters
height = 224
width = 224
num_classes = 12


def load(mode):

    # Load training data for training the model
    if(mode == "train"):
        X_train = []
        Y_train = []
        FILE_NAMES = os.listdir(TRAIN_IMAGES_PATH)
        for FILE_NAME in FILE_NAMES:

            # Read and preprocess input images
            input_image = cv2.imread(TRAIN_IMAGES_PATH + FILE_NAME, 1)
            input_image = cv2.resize(input_image, (height, width))
            input_image = np.float32(input_image)
            input_image = input_image / 127.5 - 1
            X_train.append(input_image)

            # Read and preprocess input image segmentations
            segmented_image_in_filters = np.zeros((height, width, num_classes))
            segmented_image = cv2.imread(TRAIN_SEGMENTATIONS_PATH + FILE_NAME, 1)
            segmented_image = cv2.resize(segmented_image, (height, width))
            segmented_image = segmented_image[:, :, 0]
            for segment_number in range(num_classes):
                segmented_image_in_filters[:, :, segment_number] = (segmented_image == segment_number).astype(int)
            Y_train.append(segmented_image_in_filters)

        X_train = np.array(X_train)
        Y_train = np.array(Y_train)
        return X_train, Y_train

    # Load testing data to make predictions
    if(mode == "test"):
        X_test = []
        Y_test = []
        FILE_NAMES = os.listdir(TEST_IMAGES_PATH)

        for FILE_NAME in FILE_NAMES:

            # Read and preprocess input images
            input_image = cv2.imread(TEST_IMAGES_PATH + FILE_NAME, 1)
            input_image = cv2.resize(input_image, (height, width))
            input_image = np.float32(input_image)
            input_image = input_image / 127.5 - 1
            X_test.append(input_image)

            # Read and preprocess input image segmentations
            segmented_image_in_filters = np.zeros((height, width, num_classes))
            segmented_image = cv2.imread(TEST_SEGMENTATIONS_PATH + FILE_NAME, 1)
            segmented_image = cv2.resize(segmented_image, (height, width))
            segmented_image = segmented_image[:, :, 0]
            for segment_number in range(num_classes):
                segmented_image_in_filters[:, :, segment_number] = (segmented_image == segment_number).astype(int)
            Y_test.append(segmented_image_in_filters)

        X_test = np.array(X_test)
        Y_test = np.array(Y_test)
        return X_test, Y_test
