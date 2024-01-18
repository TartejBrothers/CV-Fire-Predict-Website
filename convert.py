import tf2onnx
from keras.models import load_model

# Load your Keras model
keras_model = load_model("model/model.h5")

# Convert Keras model to ONNX
onnx_model, _ = tf2onnx.convert.from_keras(keras_model)

# Save the ONNX model to a file
with open("model.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())
