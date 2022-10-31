# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

"""Simple test for signal library usage."""

import os

from absl import app
from absl import logging
import numpy as np
import tensorflow as tf

from tflite_micro.python.tflite_micro.signal.ops import window_op
#from tflite_micro.tensorflow.lite.micro.python.interpreter.src import tflm_runtime

_window_length = 320
_shift = 14
_window_weights = window_op.hann_window_weights(_window_length, _shift)
_input_signal = []
for _ in range(1):
  # rand_audio_in.append(np.random.rand(1, 320))
  _input_signal.append(np.random.randint(-32768, 32767, (1, 320), np.int16))
#_input_signal = tf.cast(tf.random.uniform(shape=[1,320], minval=-32768, maxval=32767, dtype=np.int32, seed=123), np.int16)
print("Input Signal:\n", _input_signal)

class WindowLayer(tf.keras.layers.Layer):

  def __init__(self, frame_size, window_weights, shift, name='WINDOW'):
    super().__init__(name=name)
    self.frame_size = frame_size
    self.shift = shift
    self.window_weights = window_weights

  def call(self, inputs: tf.Tensor) -> tf.Tensor:
    return tf.cast(window_op.window(inputs, self.window_weights, self.shift), tf.float32)


class CustomCallback(tf.keras.callbacks.Callback):

  def on_epoch_end(self, epoch, logs=None):
    keys = list(logs.keys())
    print('End epoch {} of training; got log keys: {}'.format(epoch, keys))


def create_train_model(window_result):
  input_layer = tf.keras.Input(shape=(320), dtype=np.int16)
  window_layer_out = WindowLayer(_window_length, _window_weights, _shift)(input_layer)
  model = tf.keras.models.Model(inputs=input_layer, outputs=window_layer_out)
  model.compile(
      optimizer='adam',
      loss=tf.keras.losses.MeanSquaredError(),
      metrics=['mean_absolute_error'])
  
  # Nothing to really train
  model.fit(
      _input_signal,
      window_result,
      epochs=4,
      verbose=1,
      callbacks=[CustomCallback()])
  return model

def convert_to_tflite(model):
  converter = tf.lite.TFLiteConverter.from_keras_model(model)
  converter.allow_custom_ops = True
  return converter.convert()

def save_model(model, filename, save_dir):
  if not os.path.exists(save_dir):
    os.makedirs(save_dir)
  with open(save_dir + "/" + filename, "wb") as f:
    f.write(model)
  logging.info("Model saved to %s", save_dir)

def main(_):
  #TODO Manual calculation of window function and test/compare

  # static use of window op
  window_result = window_op.window(_input_signal, _window_weights, _shift)
  print("Static Window Op Results:\n" + window_result)

  # tensorflow in-graph use + training
  print("Using Window Op in-graph for Tensorflow training\n")
  tf_model = create_train_model(window_result)

  # TFLM python inference use
  print("Converting and saving as TFLite model\n")
  tflite_model = convert_to_tflite(tf_model)
  save_model(tflite_model, "window_example.tflite", "/tmp/window_example")
  interpreter = tflm_runtime.Interpreter(tflite_model,
                                         10000)
  interpreter.set_input(_input_signal, 0)
  interpreter.invoke()

if __name__ == "__main__":
  app.run(main)
