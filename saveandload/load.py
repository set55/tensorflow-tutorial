import tensorflow as tf

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

test_images = test_images.reshape(-1, 28 * 28) / 255.0

kerasmodel = tf.keras.models.load_model('batchmodel.keras')
kerasmodel.summary()

# Evaluate the model
loss, acc = kerasmodel.evaluate(test_images, test_labels, verbose=2)
print("Restored model, accuracy: {:5.2f}%".format(100 * acc))

h5model = tf.keras.models.load_model('batchmodel.h5')
h5model.summary()

# Evaluate the model
loss, acc = h5model.evaluate(test_images, test_labels, verbose=2)
print("Restored model, accuracy: {:5.2f}%".format(100 * acc))

