# Import required libraries
import tensorflow as tf
import numpy as np

# Import local modules
import load_data as ld

# File paths
CHECKPOINT_PATH = '../checkpoints'
MODEL_FILE_NAME = '/model.h5'

# Load testing data
print("Loading testing data...")
X_test, Y_test = ld.load(mode="test")

# Load trained model
model = tf.keras.models.load_model(CHECKPOINT_PATH + MODEL_FILE_NAME)

# Predict segmentations
predictions = model.predict(X_test)

# Calculate accuracy
predictions = np.argmax(predictions, axis=3)
true_segments = np.argmax(Y_test, axis=3)
matching_predictions = np.array((predictions == true_segments))
accuracy = np.sum(matching_predictions) / (Y_test.shape[0] * Y_test.shape[1] * Y_test.shape[2])
print("Accuracy = ", accuracy)
