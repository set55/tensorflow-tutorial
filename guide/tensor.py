import tensorflow as tf
import numpy as np

# rank 0 tensor
rank_0_tensor = tf.constant(5, dtype=tf.float64)
print(rank_0_tensor, rank_0_tensor.shape)

# rank 1 tensor
rank_1_tensor = tf.constant([0, 1, 1, 2, 3, 5, 8, 13, 21, 34], dtype=tf.float32)
print(rank_1_tensor, rank_1_tensor.shape)

# rank 2 tensor
rank_2_tensor = tf.constant([[1, 2], [3, 4], [5, 6]], dtype=tf.int32)
print(rank_2_tensor, rank_2_tensor.shape)

# rank 3 tensor
rank_3_tensor = tf.constant([[[1], [2], [3]], [[4], [5], [6]]], dtype=tf.int64)
print(rank_3_tensor, rank_3_tensor.shape)

print(np.array(rank_3_tensor))

print(rank_3_tensor.numpy())

a = tf.constant([[1, 2], [3, 4]])
b = tf.constant([[5, 6], [7, 8]])

print(tf.add(a, b))
print(tf.multiply(a, b))
print(tf.matmul(a, b))

print(a + b)
print(a * b)
print(a @ b)


c = tf.constant([[4.0, 5.0], [10.0, 1.0]])
print(tf.reduce_max(c))
print(tf.math.argmax(c))
print(tf.nn.softmax(c))


rank_4_tensor = tf.zeros([3, 2, 4, 5])
print("Type of every element:", rank_4_tensor.dtype)
print("Number of axes:", rank_4_tensor.ndim)
print("Shape of tensor:", rank_4_tensor.shape)
print("Elements along axis 0 of tensor:", rank_4_tensor.shape[0])
print("Elements along the last axis of tensor:", rank_4_tensor.shape[-1])
print("Total number of elements (3*2*4*5): ", tf.size(rank_4_tensor).numpy())

print(tf.rank(rank_4_tensor)) # if you want get ndim with tensorflow object, use rank
print(tf.shape(rank_4_tensor)) # if you want get shape with tensorflow object, use shape

print(rank_1_tensor.numpy())
print('first:', rank_1_tensor[0].numpy())
print('last:', rank_1_tensor[-1].numpy())
print('2nd and 3rd:', rank_1_tensor[1:3].numpy())

print(rank_2_tensor.numpy())
print('corrdinate[0, 1] element: ', rank_2_tensor[0, 1].numpy())


x = tf.constant([[1], [2], [3]])
print(x.shape, x.numpy())

reshaped = tf.reshape(x, [1, 3])

print(reshaped.shape, reshaped.numpy())


print(tf.reshape(rank_3_tensor, [2, 3]).numpy())
print(tf.reshape(rank_3_tensor, [-1]).numpy())


# conver dtype

the_f64_tensor = tf.constant([2.2, 3.3, 4.4], dtype=tf.float64)
the_f16_tensor = tf.cast(the_f64_tensor, dtype=tf.float16)
print(the_f16_tensor)
the_uint8_tensor = tf.cast(the_f16_tensor, dtype=tf.uint8)
print(the_uint8_tensor)

ragged_list = [[0, 1, 2, 3], [4, 5], [6, 7, 8], [9]]
ragged_tensor = tf.ragged.constant(ragged_list)
print(ragged_tensor, 'shape: ', ragged_tensor.shape)


scalar_string_tensor = tf.constant("Gray wolf")
print(scalar_string_tensor, scalar_string_tensor.shape, scalar_string_tensor.numpy())

tensor_of_strings = tf.constant(["Gray wolf", "Quick brown fox", "Lazy dog"])
print(tensor_of_strings, tensor_of_strings.shape, tensor_of_strings.numpy())


print(tf.strings.split(scalar_string_tensor, sep=" "))
print(tf.strings.split(tensor_of_strings, sep=" "))


sparse_tensor = tf.sparse.SparseTensor(indices=[[0, 0], [1, 2]], values=[1, 2], dense_shape=[3, 4])

print(sparse_tensor)

print(tf.sparse.to_dense(sparse_tensor).numpy())