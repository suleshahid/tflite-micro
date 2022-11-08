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

#ifndef TENSORFLOW_LITE_MICRO_MICRO_UTILS_H_
#define TENSORFLOW_LITE_MICRO_MICRO_UTILS_H_

#include "tensorflow/lite/schema/schema_utils.h"
#include "tensorflow/lite/micro/flatbuffer_utils.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"

namespace tflite {

unsigned int GetNumResourceVariables(const Model* model);

unsigned int GetNumUniqueOps(const Model* model);

template<unsigned int tOpCount>
TfLiteStatus RegisterRequiredOps(MicroMutableOpResolver<tOpCount> &resolver, const Model* model) {
  AllOpsResolver all_ops_resolver = AllOpsResolver();
  auto* opcodes = model->operator_codes();

  for (size_t i = 0; i < opcodes->size(); ++i) {
    const auto* opcode = opcodes->Get(i);
    const auto builtin_code = GetBuiltinCode(opcode);

    if (builtin_code == BuiltinOperator_CUSTOM) {
      const char* name = opcode->custom_code()->c_str();
      const auto* registration = all_ops_resolver.FindOp(name);
      if (registration == nullptr) {
        MicroPrintf("Registration is null for custom op %s\n", name);
        return kTfLiteError;
      }
      resolver.AddCustom(name, registration);
    } else {
      const auto* registration = all_ops_resolver.FindOp(builtin_code);
      if (registration == nullptr) {
        MicroPrintf("Registration is null for built-in op %s\n", EnumNameBuiltinOperator(builtin_code));
        return kTfLiteError;
      }

      MicroOpResolver::BuiltinParseFunction parser =
          all_ops_resolver.GetOpDataParser(builtin_code);
      if (parser == nullptr) {
        MicroPrintf("Did not find a parser for %s",
                    EnumNameBuiltinOperator(builtin_code));

        return kTfLiteError;
      }
      resolver.AddBuiltinPublic(builtin_code, *registration, parser);
    }
  }
  return kTfLiteOk;
}

}  // namespace tflite

#endif  // TENSORFLOW_LITE_MICRO_MICRO_UTILS_H_