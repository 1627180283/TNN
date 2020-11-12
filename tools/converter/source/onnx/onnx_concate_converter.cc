// Tencent is pleased to support the open source community by making TNN available.
//
// Copyright (C) 2020 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#include "onnx_base_converter.h"
#include "onnx_utils.h"

namespace TNN_CONVERTER {

DECLARE_OP_CONVERTER(Concat);

std::string OnnxConcatConverter::TNNOpType(const onnx::NodeProto &node, bool quantized_model) {
    return "Concat";
}

TNN_NS::ActivationType OnnxConcatConverter::ActivationType(const onnx::NodeProto &node) {
    return TNN_NS::ActivationType_None;
}

TNN_NS::Status OnnxConcatConverter::exec(tnn::NetStructure &net_structure, tnn::NetResource &net_resource,
                                         const onnx::NodeProto &node,
                                         std::map<std::string, const onnx::TensorProto *>& proxy_initializers_map,
                                         std::map<std::string, std::shared_ptr<OnnxProxyNode>>& proxy_nodes,
                                         bool &quantized_model) {
    auto param       = new TNN_NS::ConcatLayerParam;
    auto cur_layer   = net_structure.layers.back();
    cur_layer->param = std::shared_ptr<TNN_NS::LayerParam>(param);
    param->type      = cur_layer->type_str;
    param->name      = cur_layer->name;
    param->quantized = false;
    param->axis      = GetAttributeInt(node, "axis", 1);
    ASSERT(node.input_size() == 2);
    for (int i = 0; i < node.input_size(); ++i) {
        const auto& input = node.input(i);
        if (proxy_initializers_map.find(input) != proxy_initializers_map.end()) {
            const auto extra_tensor = proxy_initializers_map[input];
            float* extra_data_ptr = (float*)GetDataFromTensor(*extra_tensor, onnx::TensorProto_DataType_FLOAT);
            auto extra_data_size = GetTensorProtoDataSize(*extra_tensor);
            for (int j = 0; j < extra_data_size; ++j) {
                param->extra_data.push_back(extra_data_ptr[j]);
            }
            cur_layer->inputs.resize(1);
            cur_layer->inputs[0] = node.input(1 -i);
        }
    }

    return TNN_NS::TNN_CONVERT_OK;
}

REGISTER_CONVERTER(Concat, Concat);

}  // namespace TNN_CONVERTER
