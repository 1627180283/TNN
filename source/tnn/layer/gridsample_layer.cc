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

#include "tnn/layer/elementwise_layer.h"

namespace TNN_NS {

DECLARE_LAYER(GridSample, LAYER_GRIDSAMPLE);

Status GridSampleLayer::InferOutputDataType() {
    return BaseLayer::InferOutputDataType();
}

Status GridSampleLayer::InferOutputShape(bool ignore_error) {
    BaseLayer::InferOutputShape(ignore_error);
    
    Blob* input_blob  = input_blobs_[0];
    Blob* grid_blob  = input_blobs_[1];
    Blob* output_blob = output_blobs_[0];
     
    auto input_dims = input_blob->GetBlobDesc().dims;
    auto grid_dims = grid_blob->GetBlobDesc().dims;
    
    auto output_dims = input_dims;
    for (int i=2,j=1; i<output_dims.size() && j<grid_dims.size(); i++,j++) {
        output_dims[i] = grid_dims[j];
    }

    output_blob->GetBlobDesc().dims = output_dims;
    return TNN_OK;
}

REGISTER_LAYER(GridSample, LAYER_GRIDSAMPLE);

}  // namespace TNN_NS
