#include "paddle/fluid/framework/block_desc.h"
#include "paddle/fluid/framework/op_desc.h"
#include "paddle/fluid/framework/program_desc.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/inference/utils/singleton.h"

#include "paddle/fluid/inference/tests/api/tester_helper.h"

namespace paddle {
namespace inference {

void test_zerocopy_tensor(){
    AnalysisConfig config;
    config.SetModel(FLAGS_infer_model + "/__model__",
                FLAGS_infer_model + "/__params__");
    config.SwitchUseFeedFetchOps(false);
     
    auto predictor = CreatePaddlePredictor(config);
    int batch_size = 1;
    int channels = 1;
    int height = 48;
    int width = 512;
    int nums = batch_size * channels * height * width;

    float* input = new float[nums];
    for (int i = 0; i < nums; ++i) input[i] = 0;
    auto input_names = predictor->GetInputNames();
    PaddlePlace p= PaddlePlace::kCPU;
    PaddlePlace * place = &p;
    int *size =NULL;

    auto input_t = predictor->GetInputTensor(input_names[0]);
    input_t->Reshape({batch_size , channels , height , width});
    input_t->copy_from_cpu<float>(input);
    input_t->data<float>(place,size);
    input_t->mutable_data<float>(p);

    predictor->ZeroCopyRun();

    std::vector<float> out_data;
    auto output_names = predictor->GetOutputNames();
    auto output_t = predictor->GetOutputTensor(output_names[0]);
    std::vector<int> output_shape = output_t->shape();
    int out_num = std::accumulate(output_shape.begin(), output_shape.end(), 1, std::multiplies<int>());
    out_data.resize(out_num);
    output_t->copy_to_cpu<float>(out_data.data());

}
TEST(test_zerocopy_tensor, zerocopy_tensor) { test_zerocopy_tensor(); }



}
}