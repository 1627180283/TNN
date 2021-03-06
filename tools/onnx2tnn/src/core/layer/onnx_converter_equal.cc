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

#include "onnx_op_converter.h"
#include "onnx_utility.h"

DECLARE_OP_CONVERTER_WITH_FUNC(Equal,
                               virtual std::vector<std::string> GetValidInputNames(NodeProto &node, OnnxNetInfo &net_info););
string OnnxOpConverterEqual::TNNOpType(NodeProto &node, OnnxNetInfo &net_info) {
    return "Equal";
}

std::vector<std::string> OnnxOpConverterEqual::GetValidInputNames(NodeProto &node, OnnxNetInfo &net_info) {
    return GetAllInputNames(node, net_info);
}

string OnnxOpConverterEqual::TNNLayerParam(NodeProto &node, OnnxNetInfo &net_info) {
    return "";
}

bool OnnxOpConverterEqual::HasLayerResource(NodeProto &node, OnnxNetInfo &net_info) {
    return false;
}

int OnnxOpConverterEqual::WriteTNNModel(Serializer *net_writer, NodeProto &node, OnnxNetInfo &net_info) {
    return 0;
}

REGISTER_OP_CONVERTER(Equal, Equal);


