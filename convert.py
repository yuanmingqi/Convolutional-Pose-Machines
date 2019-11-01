import tensorflow as tf

global model
saved_model_dir = './snapshots/model_stage_6_batch_2.h5'
model = tf.keras.models.load_model(saved_model_dir)

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
open("./saved_model/converted_model.tflite", "wb").write(tflite_model)
