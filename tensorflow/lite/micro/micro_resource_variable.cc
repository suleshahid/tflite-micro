/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/lite/micro/micro_resource_variable.h"

#include <cstring>

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/compatibility.h"
#include "tensorflow/lite/micro/memory_helpers.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_utils.h"
#include "tensorflow/lite/schema/schema_utils.h"

namespace tflite {

namespace {}  // namespace

MicroResourceVariables* MicroResourceVariables::Create(
    MicroAllocator* allocator, const Model* model) {
  TFLITE_DCHECK(allocator != nullptr);
  
  unsigned int max_num_variables = GetNumResourceVariables(model);
  if (max_num_variables == 0) {
    return nullptr;
  }
  //just call below Create from here to avoid duplicate?

  uint8_t* allocator_buffer = static_cast<uint8_t*>(
      allocator->AllocatePersistentBuffer(sizeof(MicroResourceVariables)));
  MicroResourceVariable* variable_array =
      static_cast<MicroResourceVariable*>(allocator->AllocatePersistentBuffer(
          sizeof(MicroResourceVariable) * max_num_variables));
  MicroResourceVariables* variables = new (allocator_buffer)
      MicroResourceVariables(variable_array, max_num_variables);
  return variables;
}

MicroResourceVariables* MicroResourceVariables::Create(
    MicroAllocator* allocator, int max_num_variables) {
  TFLITE_DCHECK(allocator != nullptr);

  uint8_t* allocator_buffer = static_cast<uint8_t*>(
      allocator->AllocatePersistentBuffer(sizeof(MicroResourceVariables)));
  MicroResourceVariable* variable_array =
      static_cast<MicroResourceVariable*>(allocator->AllocatePersistentBuffer(
          sizeof(MicroResourceVariable) * max_num_variables));
  MicroResourceVariables* variables = new (allocator_buffer)
      MicroResourceVariables(variable_array, max_num_variables);
  return variables;
}

int MicroResourceVariables::CreateIdIfNoneFound(const char* container,
                                                const char* shared_name) {
  int resource_id = FindId(container, shared_name);
  if (resource_id >= 0) {
    return resource_id;
  }

  // no existing variable found for the given container and shared name pair.
  if (num_resource_variables_ >= max_variable_count_) {
    MicroPrintf(
        "Failed to allocate resource variable. Maximum resource variable count "
        "(%d) "
        "reached.",
        max_variable_count_);
    return -1;
  }

  resource_id = num_resource_variables_++;
  resource_variables_[resource_id].container = container;
  resource_variables_[resource_id].shared_name = shared_name;
  resource_variables_[resource_id].resource_buffer = nullptr;
  resource_variables_[resource_id].bytes = 0;
  return resource_id;
}

TfLiteStatus MicroResourceVariables::Read(int id,
                                          const TfLiteEvalTensor* tensor) {
  if (id < 0 || id >= num_resource_variables_) {
    MicroPrintf("Attempting to read non-existent resource variable %d", id);
    return kTfLiteError;
  }
  MicroResourceVariable variable = resource_variables_[id];
  TFLITE_DCHECK(EvalTensorBytes(tensor) == variable.bytes);
  TFLITE_DCHECK(variable.resource_buffer != nullptr);
  memcpy(tensor->data.raw, variable.resource_buffer, variable.bytes);
  return kTfLiteOk;
}

TfLiteStatus MicroResourceVariables::Allocate(int id, TfLiteContext* context,
                                              const TfLiteTensor* tensor) {
  if (id < 0 || id >= num_resource_variables_) {
    MicroPrintf("Attempting to read non-existent resource variable %d", id);
    return kTfLiteError;
  }

  MicroResourceVariable& variable = resource_variables_[id];

  if (variable.resource_buffer == nullptr) {
    variable.bytes = tensor->bytes;
    variable.resource_buffer =
        context->AllocatePersistentBuffer(context, tensor->bytes);
    if (variable.resource_buffer == nullptr) {
      MicroPrintf("Failed to allocate resource buffer.");
      return kTfLiteError;
    }
    // Zero out resource buffers by deafult. Buffers can be initialized to
    // nonzero values using ASSIGN_VARIABLE.
    memset(variable.resource_buffer, 0, variable.bytes);
  }

  return kTfLiteOk;
}

TfLiteStatus MicroResourceVariables::Assign(int id,
                                            const TfLiteEvalTensor* tensor) {
  if (id < 0 || id >= num_resource_variables_) {
    MicroPrintf("Attempting to read non-existent resource variable %d", id);
    return kTfLiteError;
  }
  MicroResourceVariable variable = resource_variables_[id];

  if (variable.resource_buffer == nullptr) {
    MicroPrintf(
        "Attempting to assign from a TfLiteEvalTensor before the resource "
        "buffer has been allocated. Make sure to call AssignResourceVariable "
        "with a TfLiteTensor first.");
    return kTfLiteError;
  }
  TFLITE_DCHECK(EvalTensorBytes(tensor) == variable.bytes);
  memcpy(variable.resource_buffer, tensor->data.raw, variable.bytes);
  return kTfLiteOk;
}

TfLiteStatus MicroResourceVariables::ResetAll() {
  for (int i = 0; i < num_resource_variables_; i++) {
    MicroResourceVariable variable = resource_variables_[i];
    memset(variable.resource_buffer, 0, variable.bytes);
  }
  return kTfLiteOk;
}

int MicroResourceVariables::FindId(const char* container,
                                   const char* shared_name) {
  for (int i = 0; i < num_resource_variables_; i++) {
    // Some TFLite flatbuffers contain null container names to save space.
    if ((container == nullptr ||
         !strcmp(container, resource_variables_[i].container)) &&
        !strcmp(shared_name, resource_variables_[i].shared_name)) {
      return i;
    }
  }
  return -1;
}

unsigned int MicroResourceVariables::GetNumResourceVariables(const Model* model) {
  unsigned int num_resource_variables = 0;
  int num_subgraphs = model->subgraphs()->size();
  for (int subgraph_idx = 0; subgraph_idx < num_subgraphs;
      subgraph_idx++) {
    const SubGraph* subgraph = model->subgraphs()->Get(subgraph_idx);
    TFLITE_DCHECK(subgraph != nullptr);

    auto* opcodes = model->operator_codes();
    for (size_t i = 0; i < NumSubgraphOperators(subgraph); ++i) {
      const auto* op = subgraph->operators()->Get(i);
      const size_t index = op->opcode_index();
      if (index >= opcodes->size()) {
        MicroPrintf("Missing registration for opcode_index %d\n", index);
        return kTfLiteError;
      }
      const auto* opcode = opcodes->Get(index);
      auto builtin_code = opcode->builtin_code();
      MicroPrintf("bc: %d", builtin_code);
      if (builtin_code <= BuiltinOperator_PLACEHOLDER_FOR_GREATER_OP_CODES) {
        MicroPrintf("less than 127");
        builtin_code = GetBuiltinCode(opcode);
      }
      //MicroPrintf("builtinCode = %d", EnumNameBuiltinOperator(builtin_code));
      if (builtin_code == BuiltinOperator_ASSIGN_VARIABLE){
        num_resource_variables++;
      }
    }
  }
  return num_resource_variables;
}

}  // namespace tflite
