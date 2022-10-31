/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <climits>

#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/examples/signal/window_example_model_data.h"

/*
 * Simple C++ example code showing TFLM signal library window op usage.
 */

namespace {

constexpr int kTensorArenaSize = 10000;
alignas(16) uint8_t tensor_arena[kTensorArenaSize];

}  // namespace

void input_random_data(tflite::MicroInterpreter& interpreter) {
  std::srand(123);
  TfLiteTensor* input = interpreter.input(0);

  // Pre-populate input tensor with random values.
  int input_length = input->bytes / sizeof(int16_t);
  int16_t* input_values = tflite::GetTensorData<int16_t>(input);
  for (int i = 0; i < input_length; i++) {
    // Pre-populate input tensor with a random value based on a constant seed.
    input_values[i] = static_cast<int16_t>(
        std::rand() % (std::numeric_limits<int16_t>::max() -
                        std::numeric_limits<int16_t>::min() + 1));
}
}

int main(int argc, char* argv[]) {
  tflite::AllOpsResolver resolver;
  const tflite::Model* model = tflite::GetModel(g_window_example_model_data);
  tflite::MicroInterpreter interpreter(
      model, resolver, tensor_arena, kTensorArenaSize);
  
  input_random_data(interpreter);
  interpreter.Invoke();
  MicroPrintf("");

}
