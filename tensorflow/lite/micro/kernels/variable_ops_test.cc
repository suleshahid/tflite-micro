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
#include <stdint.h>

#include <memory>
#include <numeric>

#include "flatbuffers/flexbuffers.h"
#include "tensorflow/lite/micro/test_helpers.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/testing/micro_test.h"
#include "tensorflow/lite/schema/schema_generated.h"


constexpr int kTensorArenaSize = 3000000;
alignas(16) uint8_t tensor_arena[kTensorArenaSize];

namespace {
// This is the order we declare the operators in each model, it is the same for
// all models in this test.
constexpr uint32_t VAR_HANDLE = 0;
constexpr uint32_t READ_VARIABLE = 1;
constexpr uint32_t ASSIGN_VARIABLE = 2;
constexpr uint32_t CALL_ONCE = 3;
}  // namespace


namespace tflite {

class VariableOpsTester {
public:
  void TestAssignThenRead() {
    const std::vector<char> model = CreateModelAssignThenRead();
    TestResourceVariables(model);
  }
  void TestResourceVariables(const std::vector<char>& buffer);
  std::vector<char> CreateModelAssignThenRead() const;
  const std::vector<int32_t> &Shape() const { return shape_; }
  const std::vector<int32_t> &ResourceShape() const { return resource_shape_; }

private:
  std::vector<int32_t> shape_ = {1, 2, 2, 3};
  std::vector<int32_t> resource_shape_ = {1};
};

std::vector<char> VariableOpsTester::CreateModelAssignThenRead() const {
flatbuffers::FlatBufferBuilder builder;

const std::vector<flatbuffers::Offset<OperatorCode>> operator_codes = {
    CreateOperatorCode(builder, BuiltinOperator_VAR_HANDLE),
    CreateOperatorCode(builder, BuiltinOperator_READ_VARIABLE),
    CreateOperatorCode(builder, BuiltinOperator_ASSIGN_VARIABLE),
};

const std::vector<flatbuffers::Offset<Buffer>> buffers{{
    CreateBuffer(builder, builder.CreateVector({})),
}};

// tensor 0 is graph input
// tensor 1 is VAR_HANDLE output
// tensor 2 is graph output

const std::vector<flatbuffers::Offset<Tensor>> tensors{{
    CreateTensor(
        builder,
        builder.CreateVector<int32_t>(Shape().data(), Shape().size()),
        TensorType_FLOAT32),
    CreateTensor(builder,
                  builder.CreateVector<int32_t>(ResourceShape().data(),
                                                ResourceShape().size()),
                  TensorType_RESOURCE),
    CreateTensor(
        builder,
        builder.CreateVector<int32_t>(Shape().data(), Shape().size()),
        TensorType_FLOAT32),
}};

const flatbuffers::Offset<Operator> var_handle_op = CreateOperator(
    builder, VAR_HANDLE, builder.CreateVector<int32_t>({}),
    builder.CreateVector<int32_t>({1}),
    tflite::BuiltinOptions_VarHandleOptions,
    CreateVarHandleOptions(builder, builder.CreateString("container"),
                            builder.CreateString("shared_name"))
        .Union());

const flatbuffers::Offset<Operator> assign_op = CreateOperator(
    builder, ASSIGN_VARIABLE, builder.CreateVector<int32_t>({1, 0}),
    builder.CreateVector<int32_t>({}));

const flatbuffers::Offset<Operator> read_op =
    CreateOperator(builder, READ_VARIABLE, builder.CreateVector<int32_t>({1}),
                    builder.CreateVector<int32_t>({2}));

const flatbuffers::Offset<SubGraph> subgraph = CreateSubGraph(
    builder, builder.CreateVector(tensors.data(), tensors.size()),
    builder.CreateVector<int32_t>({0}), builder.CreateVector<int32_t>({2}),
    builder.CreateVector({var_handle_op, assign_op, read_op}));
MicroPrintf("codes: %d%d", operator_codes.data()->IsNull(), false);
const flatbuffers::Offset<Model> model_buffer = CreateModel(
    builder, TFLITE_SCHEMA_VERSION,
    builder.CreateVector(operator_codes.data(), operator_codes.size()),
    builder.CreateVector(&subgraph, 1),
    builder.CreateString("ReadVariable model"),
    builder.CreateVector(buffers.data(), buffers.size()));

builder.Finish(model_buffer);

std::vector<char> finished_buffer = std::vector<char>(builder.GetBufferPointer(),
                           builder.GetBufferPointer() + builder.GetSize());

return finished_buffer;

}

void VariableOpsTester::TestResourceVariables(const std::vector<char>& buffer) {
  const Model* model = GetModel(buffer.data());
  TFLITE_DCHECK(model);
  MicroAllocator* allocator = MicroAllocator::Create(tensor_arena, kTensorArenaSize);
  TFLITE_DCHECK(allocator);
  MicroResourceVariables* resource_variables = MicroResourceVariables::Create(allocator, model);
  TFLITE_DCHECK(resource_variables);

  MicroInterpreter interpreter(model, testing::GetOpResolver(), allocator, resource_variables);

  interpreter.AllocateTensors();

  //TF_LITE_MICRO_EXPECT_EQ(interpreter.AllocateTensors(), kTfLiteOk);

  TfLiteTensor* input_data_index = interpreter.input(0);
  GetTensorData<float>(input_data_index)[0] = 1717;

  interpreter.Invoke();

  //TF_LITE_MICRO_EXPECT_EQ(interpreter.Invoke(), kTfLiteOk);

  // Verify output.
  TfLiteTensor* output = interpreter.output(0);
  //TF_LITE_MICRO_EXPECT_EQ(output->dims->size, 0);
  if (GetTensorData<float>(output)[0] == 1717) {
    MicroPrintf("Success");
  }
  //TF_LITE_MICRO_EXPECT_EQ(GetTensorData<float>(output)[0], 1717);
}
}

int main(int args, char** argv) {

}

// TF_LITE_MICRO_TESTS_BEGIN

// TF_LITE_MICRO_TEST(VariableOpsTest) { tflite::VariableOpsTester::TestAssignThenRead(); }

// TF_LITE_MICRO_TESTS_END
