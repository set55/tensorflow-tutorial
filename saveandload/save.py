import os

import tensorflow as tf
from tensorflow import keras

print(tf.version.VERSION)

# Define a simple sequential model
def create_model():
  model = tf.keras.Sequential([
    keras.layers.Dense(512, activation='relu', input_shape=(784,)),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(10)
  ])

  model.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

  return model



(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

train_labels = train_labels[:1000]
test_labels = test_labels[:1000]

print('train_images.shape: ', train_images.shape)

train_images = train_images[:1000].reshape(-1, 28 * 28) / 255.0
test_images = test_images[:1000].reshape(-1, 28 * 28) / 255.0

print('train_images.shape: ', train_images.shape)

# Create a basic model instance
model = create_model()

# Display the model's architecture
model.summary()

checkpoint_path = "training_1/cp.weights.h5"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create a callback that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)

# Train the model with the new callback
model.fit(train_images, 
          train_labels,  
          epochs=10,
          validation_data=(test_images, test_labels),
          callbacks=[cp_callback])  # Pass callback to training

# This may generate warnings related to saving the state of the optimizer.
# These warnings (and similar warnings throughout this notebook)
# are in place to discourage outdated usage, and can be ignored.


# Recreate a basic model instance
remodel = create_model()

# Evaluate the model
loss, acc = remodel.evaluate(test_images, test_labels, verbose=2)
print("Untrained model, accuracy: {:5.2f}%".format(100 * acc))

# Loads the weights
remodel.load_weights(checkpoint_path)

# Re-evaluate the model
loss, acc = remodel.evaluate(test_images, test_labels, verbose=2)
print("Restored model, accuracy: {:5.2f}%".format(100 * acc))





# Include the epoch in the file name (uses `str.format`)
checkpoint_path = "training_2/cp-{epoch:04d}.weights.h5"
checkpoint_dir = os.path.dirname(checkpoint_path)

batch_size = 32

# Calculate the number of batches per epoch
import math
import glob
n_batches = len(train_images) / batch_size
n_batches = math.ceil(n_batches)    # round up the number of batches to the nearest whole integer

# Create a callback that saves the model's weights every 5 epochs
cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path, 
    verbose=1, 
    save_weights_only=True,
    save_freq=5*n_batches)

# Create a new model instance
batchmodel = create_model()

# Save the weights using the `checkpoint_path` format
# Create a file for checkpoint_path.format(epoch=0)
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)
open(checkpoint_path.format(epoch=0), 'a').close()
batchmodel.save_weights(checkpoint_path.format(epoch=0))


# Train the model with the new callback
batchmodel.fit(train_images, 
          train_labels,
          epochs=50, 
          batch_size=batch_size, 
          callbacks=[cp_callback],
          validation_data=(test_images, test_labels),
          verbose=1)

batchmodel.save('batchmodel.keras')
batchmodel.save('batchmodel.h5')
# batchmodel.save('mymodel')


# List all the checkpoint files in the directory
checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "cp-*.weights.h5"))

# Print the path of each checkpoint file
for checkpoint_file in checkpoint_files:
    print(checkpoint_file)
    tmpModel = create_model()
    tmpModel.load_weights(checkpoint_file)
    tmpModel.summary()
    loss, acc = tmpModel.evaluate(test_images, test_labels, verbose=2)
    print("Restored model from {}, accuracy: {:5.2f}%".format(checkpoint_file, 100 * acc))



