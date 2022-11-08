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

#include "tensorflow/lite/schema/schema_utils.h"
#include "tensorflow/lite/micro/flatbuffer_utils.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"

namespace tflite {

unsigned int GetNumResourceVariables(const Model* model) {
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
      const auto builtin_code = GetBuiltinCode(opcode);
      if (builtin_code == BuiltinOperator_ASSIGN_VARIABLE){
        num_resource_variables++;
      }
    }
  }
  return num_resource_variables;
}

unsigned int GetNumUniqueOps(const Model* model) {
    // just model->operator_codes()->size()?
    return model->operator_codes()->size();
//   unsigned int num_ops = 0;
//   int num_subgraphs = model->subgraphs()->size();
//   for (int subgraph_idx = 0; subgraph_idx < num_subgraphs;
//       subgraph_idx++) {
//     const SubGraph* subgraph = model->subgraphs()->Get(subgraph_idx);
//     TFLITE_DCHECK(subgraph != nullptr);
//     auto* opcodes = model->operator_codes();
//     for (size_t i = 0; i < NumSubgraphOperators(subgraph); ++i) {
//         MicroPrintf("SGOPS: %d", opcodes->size());
//       num_ops++;
//     }
//   }
//   return num_ops;
} 



}  // namespace tflite
