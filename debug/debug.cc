#include <fstream>
#include <memory>
#include <string>

#include "tnn/core/blob.h"
#include "tnn/core/common.h"
#include "tnn/core/instance.h"
#include "tnn/core/macro.h"
#include "tnn/core/status.h"
#include "tnn/core/tnn.h"
#include "tnn/utils/blob_converter.h"

using namespace std;


int TNNNetInit_(string modelContent, string protoContent, shared_ptr<TNN_NS::TNN> &tnn,
                shared_ptr<TNN_NS::Instance> &instance) {
    TNN_NS::ModelConfig modelConfig;
    modelConfig.model_type = TNN_NS::MODEL_TYPE_TNN;
    modelConfig.params = {protoContent, modelContent};

    tnn = std::make_shared<TNN_NS::TNN>();
    TNN_NS::Status status = tnn->Init(modelConfig);

    printf("init status = %s\n", status.description().c_str());

//    MOTION_LOGD("TNNNetInit init status %s.\n", status.description().c_str());
    RETURN_ON_NEQ(status, TNN_NS::TNN_OK);

    /* Create TNN instance */
    TNN_NS::NetworkConfig networkConfig;
    networkConfig.device_type = TNN_NS::DEVICE_OPENCL;
    networkConfig.precision = TNN_NS::PRECISION_AUTO;
    instance = tnn->CreateInst(networkConfig, status);
    printf("instance status = %s\n", status.description().c_str());
    if (status != TNN_NS::TNN_OK) {
//        MOTION_LOGE("tnn create instance for GPU failed! error: %s, Fallback to ARM CPU.\n",
//                    status.description().c_str());
        networkConfig.device_type = TNN_NS::DEVICE_ARM;
        networkConfig.precision = TNN_NS::PRECISION_AUTO;
        instance = tnn->CreateInst(networkConfig, status);
        RETURN_ON_NEQ(status, TNN_NS::TNN_OK);
//        MOTION_LOGE("tnn create instance for ARM CPU successfully\n");
    } else {
//        MOTION_LOGD("tnn create instance for GPU successfully\n");
    }

    return 0;
}

std::string read_all_content(std::string path)
{
    std::ifstream in(path);
    std::istreambuf_iterator<char> begin(in);
    std::istreambuf_iterator<char> end;
    std::string str_cont(begin, end);
    return str_cont;
}

std::string read_all_content_byte(std::string path)
{
    std::fstream t(path, std::ios::in | std::ios::binary);
    std::istreambuf_iterator<char> begin(t);
    std::istreambuf_iterator<char> end;
    std::string str_cont(begin, end);
    return str_cont;
}

int main()
{
    std::string proto_path = "/Users/ealinli/workspace/Tencent/model/onnx/junfu/tnn/coarse.opt.tnnproto";
    std::string model_path = "/Users/ealinli/workspace/Tencent/model/onnx/junfu/tnn/coarse.opt.tnnmodel";

    shared_ptr<TNN_NS::TNN>tnn_;
    shared_ptr<TNN_NS::Instance>instance_;


    string protoContent = read_all_content_byte(proto_path);
    string modelContent = read_all_content(model_path);

    int ret = TNNNetInit_(modelContent, protoContent, tnn_, instance_);

    return 0;
}