import sys
print("Python:", sys.version)
print("Python path:", sys.executable)

try:
    import numpy
    print("✓ Numpy:", numpy.__version__)
except Exception as e:
    print("✗ Numpy error:", e)

try:
    import scipy
    print("✓ Scipy:", scipy.__version__)
except Exception as e:
    print("✗ Scipy error:", e)

try:
    import sklearn
    print("✓ Sklearn:", sklearn.__version__)
except Exception as e:
    print("✗ Sklearn error:", e)

try:
    import tensorflow as tf
    print("✓ TensorFlow:", tf.__version__)
    print("✓ GPU:", tf.config.list_physical_devices('GPU'))
except Exception as e:
    print("✗ TensorFlow error:", e)