import tensorflow as tf

# Uncomment to see where your variables get placed (see below)
# tf.debugging.set_log_device_placement(True)

my_tensor = tf.constant([[1.0, 2.0], [3.0, 4.0]])
my_variable = tf.Variable(my_tensor)


bool_variable = tf.Variable([False, False, False, True])
complex_variable = tf.Variable([5 + 4j, 6 + 1j])

print("Shape: ", my_variable.shape, bool_variable.shape, complex_variable.shape)
print("DType: ", my_variable.dtype, bool_variable.dtype, complex_variable.dtype)
print("As NumPy: ", my_variable.numpy(), bool_variable.numpy(), complex_variable.numpy())

print("my_variable:", my_variable)
print("my_variable tensor: ", tf.convert_to_tensor(my_variable))
print("highest my_variable: ", tf.math.argmax(my_variable))

# if variable reshape will convert to tensor
print("reshape my_variable: ", tf.reshape(my_variable, [1, 4]))

a = tf.Variable([2.0, 3.0])
print("a:", a)
a.assign([1, 2])
print("a, after assign:", a)


# try:
#     a.assign([1.0, 2.0, 3.0])
# except Exception as e:
#     print(f"{type(e).__name__}: {e}")

b = tf.Variable(a)

a.assign([5, 6])

print(f"a: {a}, b: {b}")

print(f"a assign add 2, -1: {a.assign_add([2, -1])}, b: {b}")
print(f"a assign sub 7,9, a: {a.assign_sub([7, 9])}, b: {b}")