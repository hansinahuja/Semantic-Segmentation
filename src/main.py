# Import local modules
import create_model as cm
import load_data as ld

# File paths
CHECKPOINT_PATH = "../checkpoints"
WEIGHTS_FILE_NAME = '/weights'
MODEL_FILE_NAME = '/model.h5'

# Training parameters
epochs = 200                # Number of epochs
batch_size = 32             # Training batch size
validation_split = 0.1      # Fraction of training data for validation
verbose = 1                 # Show progress bar

# Load training data
print("Loading training data...")
X_train, Y_train = ld.load(mode="train")


# Create model
print("Creating model...")
model = cm.create()

# Train model
print("Training model...")
model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size, validation_split=validation_split, verbose=verbose)

# Save model and model weights
print("Saving model and model weights...")
model.save(CHECKPOINT_PATH + MODEL_FILE_NAME)
model.save_weights(CHECKPOINT_PATH + WEIGHTS_FILE_NAME)
