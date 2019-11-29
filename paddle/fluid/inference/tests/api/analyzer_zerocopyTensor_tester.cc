#include "paddle/fluid/framework/block_desc.h"
#include "paddle/fluid/framework/op_desc.h"
#include "paddle/fluid/framework/program_desc.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/inference/utils/singleton.h"

#include "paddle/fluid/inference/tests/api/tester_helper.h"

namespace paddle {
namespace inference {
  using framework::proto::VarType;
void AddTensorToBlockDesc(framework::proto::BlockDesc* block,
                          const std::string& name,
                          const std::vector<int64_t>& shape,framework::proto::VarType::Type type,
                          bool persistable = false) {

  auto* var = block->add_vars();
  framework::VarDesc desc(name);
  desc.SetType(VarType::LOD_TENSOR);
  desc.SetDataType(type);
  desc.SetShape(shape);
  desc.SetPersistable(persistable);
  *var = *desc.Proto();
}

void serialize_params(std::string* str, framework::Scope* scope,
                      const std::vector<std::string>& params) {
  std::ostringstream os;
#ifdef PADDLE_WITH_CUDA
  platform::CUDAPlace place;
  platform::CUDADeviceContext ctx(place);
#else
  platform::CPUDeviceContext ctx;
#endif
  for (const auto& param : params) {
    PADDLE_ENFORCE_NOT_NULL(scope->FindVar(param),
                            "Block should already have a '%s' variable", param);
    auto* tensor = scope->FindVar(param)->GetMutable<framework::LoDTensor>();
    framework::SerializeToStream(os, *tensor, ctx);
  }
  *str = os.str();
}

/*
 * Get a random float value between [low, high]
 */
template <typename T>
T random(float low, float high) {
  // static std::random_device rd;
  T temp=1;
  return temp;
}

template <typename T>
void RandomizeTensor(framework::LoDTensor* tensor,
                     const platform::Place& place) {
  auto dims = tensor->dims();
  size_t num_elements = analysis::AccuDims(dims, dims.size());
  PADDLE_ENFORCE_GT(num_elements, 0);

  platform::CPUPlace cpu_place;
  framework::LoDTensor temp_tensor;
  temp_tensor.Resize(dims);
  auto* temp_data = temp_tensor.mutable_data<float>(cpu_place);

  for (size_t i = 0; i < num_elements; i++) {
    *(temp_data + i) = random<T>(0., 1.);
    LOG(INFO) << "weights: " << *(temp_data + i);
  }

  TensorCopySync(temp_tensor, place, tensor);
}
template <typename T>
void CreateTensor(framework::Scope* scope, const std::string& name,
                  const std::vector<int64_t>& shape) {
  auto* var = scope->Var(name);
  auto* tensor = var->GetMutable<framework::LoDTensor>();
  auto dims = framework::make_ddim(shape);
  tensor->Resize(dims);
#ifdef PADDLE_WITH_CUDA
  platform::CUDAPlace place;
#else
  platform::CPUPlace place;
#endif
  RandomizeTensor<T>(tensor, place);
}


void make_modle(std::string* model, std::string* param){

  framework::ProgramDesc program;
  LOG(INFO) << "program.block size is " << program.Size();
  auto* block_ = program.Proto()->mutable_blocks(0);
  LOG(INFO) << "create block desc";
  framework::BlockDesc block_desc(&program, block_);

  auto* feed1 = block_desc.AppendOp();
  feed1->SetType("feed");
  feed1->SetInput("X", {"feed"});
  feed1->SetOutput("Out", {"var_1"});
  feed1->SetAttr("col", 0);

  auto* feed2 = block_desc.AppendOp();
  feed2->SetType("feed");
  feed2->SetInput("X", {"feed"});
  feed2->SetOutput("Out", {"var_2"});
  feed2->SetAttr("col", 1);

  auto* feed3 = block_desc.AppendOp();
  feed3->SetType("feed");
  feed3->SetInput("X", {"feed"});
  feed3->SetOutput("Out", {"var_3"});
  feed3->SetAttr("col", 2);

  auto* feed4 = block_desc.AppendOp();
  feed4->SetType("feed");
  feed4->SetInput("X", {"feed"});
  feed4->SetOutput("Out", {"var_4"});
  feed4->SetAttr("col", 3);

  auto* scale1 = block_desc.AppendOp();
  scale1->SetType("scale");
  scale1->SetInput("X", {"var_1"});
  scale1->SetOutput("Out", {"out_1"});

  auto* scale2 = block_desc.AppendOp();
  scale2->SetType("scale");
  scale2->SetInput("X", {"var_2"});
  scale2->SetOutput("Out", {"out_2"});

  auto* scale3 = block_desc.AppendOp();
  scale3->SetType("scale");
  scale3->SetInput("X", {"var_3"});
  scale3->SetOutput("Out", {"out_3"});

  auto* scale4 = block_desc.AppendOp();
  scale4->SetType("scale");
  scale4->SetInput("X", {"var_4"});
  scale4->SetOutput("Out", {"out_4"});

  auto* fetch1 = block_desc.AppendOp();
  fetch1->SetType("fetch");
  fetch1->SetInput("X", std::vector<std::string>({"out_1"}));
  fetch1->SetOutput("Out", std::vector<std::string>({"out1"}));
  fetch1->SetAttr("col", 0);

  auto* fetch2 = block_desc.AppendOp();
  fetch2->SetType("fetch");
  fetch2->SetInput("X", std::vector<std::string>({"out_2"}));
  fetch2->SetOutput("Out", std::vector<std::string>({"out2"}));
  fetch2->SetAttr("col", 1);

  auto* fetch3 = block_desc.AppendOp();
  fetch3->SetType("fetch");
  fetch3->SetInput("X", std::vector<std::string>({"out_3"}));
  fetch3->SetOutput("Out", std::vector<std::string>({"out3"}));
  fetch3->SetAttr("col", 2);

  auto* fetch4 = block_desc.AppendOp();
  fetch4->SetType("fetch");
  fetch4->SetInput("X", std::vector<std::string>({"out_4"}));
  fetch4->SetOutput("Out", std::vector<std::string>({"out4"}));
  fetch4->SetAttr("col", 3);


    // Set inputs' variable shape in BlockDesc
  AddTensorToBlockDesc(block_, "var_1", std::vector<int64_t>({2}), VarType::INT32, true);
  AddTensorToBlockDesc(block_, "var_2", std::vector<int64_t>({2}), VarType::INT64,true);
  AddTensorToBlockDesc(block_, "var_3", std::vector<int64_t>({2}), VarType::FP32,true);
  AddTensorToBlockDesc(block_, "var_4", std::vector<int64_t>({2}), VarType::UINT8, true);

  // Set outputs' variable shape in BlockDesc
  AddTensorToBlockDesc(block_, "out_1", std::vector<int64_t>({2}), VarType::INT32,false);
  AddTensorToBlockDesc(block_, "out_2", std::vector<int64_t>({2}), VarType::INT64,false);
  AddTensorToBlockDesc(block_, "out_3", std::vector<int64_t>({2}), VarType::FP32,false);
  AddTensorToBlockDesc(block_, "out_4", std::vector<int64_t>({2}), VarType::UINT8,false);

  AddTensorToBlockDesc(block_, "out1", std::vector<int64_t>({2}), VarType::INT32,false);
  AddTensorToBlockDesc(block_, "out2", std::vector<int64_t>({2}), VarType::INT64,false);
  AddTensorToBlockDesc(block_, "out3", std::vector<int64_t>({2}), VarType::FP32,false);
  AddTensorToBlockDesc(block_, "out4", std::vector<int64_t>({2}), VarType::UINT8,false);



  *block_->add_ops() = *feed1->Proto();
  *block_->add_ops() = *feed2->Proto();
  *block_->add_ops() = *feed3->Proto();
  *block_->add_ops() = *feed4->Proto();

  *block_->add_ops() = *scale1->Proto();
  *block_->add_ops() = *scale2->Proto();
  *block_->add_ops() = *scale3->Proto();
  *block_->add_ops() = *scale4->Proto();

  *block_->add_ops() = *fetch1->Proto();
  *block_->add_ops() = *fetch2->Proto();
  *block_->add_ops() = *fetch3->Proto();
  *block_->add_ops() = *fetch4->Proto();

    framework::Scope scope;
#ifdef PADDLE_WITH_CUDA
  platform::CUDAPlace place;
  platform::CUDADeviceContext ctx(place);
#else
  platform::CPUPlace place;
  platform::CPUDeviceContext ctx(place);
#endif  // Prepare variables.
  std::vector<std::string> repetitive_params{"var_1", "var_2","var_3","var_4"};
  CreateTensor<int>(&scope, "var_1", std::vector<int64_t>({2}));
  CreateTensor<int64_t>(&scope, "var_2", std::vector<int64_t>({2}));
  CreateTensor<float>(&scope, "var_3", std::vector<int64_t>({2}));
  CreateTensor<uint8_t>(&scope, "var_4", std::vector<int64_t>({2}));
  //ASSERT_EQ(block_->ops_size(), 4);
  *model = program.Proto()->SerializeAsString();
  serialize_params(param, &scope, repetitive_params);
}

void test_zerocopy_tensor(){
    AnalysisConfig config;
    std::string mod;
    std::string par;
    std::cout<<"----------Q----------1"<<std::endl;
    make_modle(&mod,&par);
    std::cout<<"----------Q----------2"<<std::endl;
    config.SetModelBuffer(&mod[0],mod.size(),&par[0],par.size());
    std::cout<<"----------Q----------3"<<std::endl;
    std::cout<<"---------------mod---------------"<<std::endl;
    std::cout<<mod<<std::endl;
    std::cout<<"---------------par---------------"<<std::endl;
    std::cout<<par<<std::endl;
    std::cout<<"----------------end--------------"<<std::endl;
    config.SwitchUseFeedFetchOps(false);
     
    auto predictor = CreatePaddlePredictor(config);
    std::cout<<"----------Q----------4"<<std::endl;
    int shape = 2;
    int input_1[2]={1,2};
    int64_t input_2[2]={3,4};
    float input_3[1][2]={{5,6}};
    float *inputs=new float[2];
    inputs[0] =5; 
    inputs[1] =6; 
    uint8_t input_4[2]={7,8};
    std::cout<<"----------Q----------5"<<std::endl;
    auto input_names = predictor->GetInputNames();
    std::cout<<"----size of input_names---"<<input_names.size()<<std::endl;
    std::cout<<"----------Q----------6"<<std::endl;
    PaddlePlace p= PaddlePlace::kCPU;
    PaddlePlace * place = &p;
    int *size =NULL;
    for(auto it=input_names.begin(); it !=input_names.end();++it)
       std::cout<<*it<<std::endl;	
    std::cout<<"----------Q----------7"<<std::endl;		
    auto input_t_1 = predictor->GetInputTensor(input_names[0]);
    std::cout<<"----------Q----------8"<<std::endl;		
    input_t_1->Reshape({shape});
    input_t_1->copy_from_cpu<int>(input_1);

    input_t_1->data<int>(place,size);
    input_t_1->mutable_data<int>(p);

    auto input_t_2 = predictor->GetInputTensor(input_names[1]);
    input_t_2->Reshape({shape});
    input_t_2->copy_from_cpu<int64_t>(input_2);
    input_t_2->data<int64_t>(place,size);
    input_t_2->mutable_data<int64_t>(p);

    auto input_t_3 = predictor->GetInputTensor(input_names[2]);
    input_t_3->Reshape({1,2});
    input_t_3->copy_from_cpu<float>(inputs);
    input_t_3->data<float>(place,size);
    input_t_3->mutable_data<float>(p);

    auto input_t_4 = predictor->GetInputTensor(input_names[3]);
    input_t_4->Reshape({shape});
    input_t_4->copy_from_cpu<uint8_t>(input_4);
    input_t_4->data<uint8_t>(place,size);
    input_t_4->mutable_data<uint8_t>(p);

    std::cout<<"----------Q----------80"<<std::endl;
    predictor->ZeroCopyRun();
    std::cout<<"----------Q----------90"<<std::endl;

    std::vector<int> out_data_1;
    std::vector<int64_t> out_data_2;
    std::vector<float> out_data_3;
    std::vector<uint8_t> out_data_4;
    auto output_names = predictor->GetOutputNames();

    for(auto it=output_names.begin(); it !=output_names.end();++it)
       std::cout<<*it<<std::endl;	

    auto output_t_1 = predictor->GetOutputTensor(output_names[0]);
    std::vector<int> output_shape_1 = output_t_1->shape();
    int out_num_1 = std::accumulate(output_shape_1.begin(), output_shape_1.end(), 1, std::multiplies<int>());
    out_data_1.resize(out_num_1);
    output_t_1->copy_to_cpu<int>(out_data_1.data());

    auto output_t_2 = predictor->GetOutputTensor(output_names[1]);
    std::vector<int> output_shape_2 = output_t_2->shape();
    int out_num_2 = std::accumulate(output_shape_2.begin(), output_shape_2.end(), 1, std::multiplies<int>());
    out_data_2.resize(out_num_2);
    output_t_2->copy_to_cpu<int64_t>(out_data_2.data());

    auto output_t_3 = predictor->GetOutputTensor(output_names[2]);
    std::vector<int> output_shape_3 = output_t_3->shape();
    int out_num_3 = std::accumulate(output_shape_3.begin(), output_shape_3.end(), 1, std::multiplies<int>());
    out_data_3.resize(out_num_3);
    output_t_3->copy_to_cpu<float>(out_data_3.data());

    auto output_t_4 = predictor->GetOutputTensor(output_names[3]);
    std::vector<int> output_shape_4  = output_t_4 ->shape();
    int out_num_4  = std::accumulate(output_shape_4 .begin(), output_shape_4 .end(), 1, std::multiplies<int>());
    out_data_4 .resize(out_num_4);
    output_t_4 ->copy_to_cpu<uint8_t>(out_data_4 .data());

    std::cout<<"----------1----------"<<std::endl;
    for(auto it =out_data_1.begin();it != out_data_1.end();++it)
        std::cout<<*it<<std::endl;
    std::cout<<"----------2----------"<<std::endl;
    for(auto it =out_data_2.begin();it != out_data_2.end();++it)
        std::cout<<*it<<std::endl;
    std::cout<<"----------3----------"<<std::endl;
    for(auto it =out_data_3.begin();it != out_data_3.end();++it)
        std::cout<<*it<<std::endl;
    std::cout<<"----------4----------"<<std::endl;
    for(auto it =out_data_4.begin();it != out_data_4.end();++it)
        std::cout<<*it<<std::endl;

}
TEST(test_zerocopy_tensor, zerocopy_tensor) { test_zerocopy_tensor(); }



}
}