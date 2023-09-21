#include "/root/paddlejob/workspace/env_run/zkk/TensorRT-8.2.4.2/include//NvInfer.h"
#include <vector>
#include <iostream>
// g++ A.cc -lnvinfer -lcudart  -L /usr/local/tensorrt/lib/ -L/usr/local/cuda/lib64 -I /usr/local/cuda/include
// export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/tensorrt/lib

#define TRT_VERSION                                    \
  NV_TENSORRT_MAJOR * 1000 + NV_TENSORRT_MINOR * 100 + \
      NV_TENSORRT_PATCH * 10 + NV_TENSORRT_BUILD

#include "/usr/local/cuda/include/cuda_runtime.h"
class TensorrtLogger : public nvinfer1::ILogger {
  nvinfer1::ILogger::Severity verbosity_;

 public:
  TensorrtLogger(Severity verbosity = Severity::kWARNING)
      : verbosity_(verbosity) {}
  void log(Severity severity, const char* msg) noexcept override {
    if (severity <= verbosity_) {
        printf("%s\n", msg);
  }
  }
};

int main()
{
    std::cout << TRT_VERSION  << std::endl;
    static TensorrtLogger trt_logger(nvinfer1::ILogger::Severity::kWARNING);
    auto trt_builder = nvinfer1::createInferBuilder(trt_logger);
    const auto explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    auto trt_network = trt_builder->createNetworkV2(explicitBatch);
    auto trt_config = trt_builder->createBuilderConfig();
    trt_config->setFlag(nvinfer1::BuilderFlag::kINT8);
    trt_config->setMaxWorkspaceSize(1<<30);
    trt_config->setProfilingVerbosity(nvinfer1::ProfilingVerbosity::kDETAILED);

    nvinfer1::Dims a;
    a.nbDims = 2;
    a.d[0] = -1;
    a.d[1] = -1;
    trt_network->addInput("foo0", nvinfer1::DataType::kFLOAT, a);

    nvinfer1::Dims b;
    b.nbDims = 4;
    b.d[0] = -1;
    b.d[1] = 128;
    b.d[2] = 14;
    b.d[3] = 14;


    int in_b_size = 1;
    int num_in = b.d[1];
    int num_output = 12;
    float* tmp1 = new float[num_output * 3 * 3 * num_in];
    float* tmp2 = new float[num_output];

  nvinfer1::Weights kernel_weight;
  kernel_weight.type = nvinfer1::DataType::kFLOAT;
  kernel_weight.values = static_cast<void*>(tmp1);
  kernel_weight.count = num_output * 3 * 3 * num_in;
  nvinfer1::Weights bias_weight;
  bias_weight.type = nvinfer1::DataType::kFLOAT;
  bias_weight.values = static_cast<void*>(tmp2);
  bias_weight.count = num_output;

nvinfer1::IOptimizationProfile* profile1 = trt_builder->createOptimizationProfile();
profile1->setDimensions("foo0", nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims2(1,128 * 14 * 14));
profile1->setDimensions("foo0", nvinfer1::OptProfileSelector::kOPT, nvinfer1::Dims2(3,128 * 14 * 14));
profile1->setDimensions("foo0", nvinfer1::OptProfileSelector::kMAX, nvinfer1::Dims2(128,128 * 14 * 14));
trt_config->addOptimizationProfile(profile1);


    auto* x = trt_network->getInput(0);
    x->setDynamicRange(-1, 1);
    auto* reshape_layer = trt_network->addShuffle(*x);
    reshape_layer->setReshapeDimensions(b);
    x = reshape_layer->getOutput(0);
   x->setDynamicRange(-1,1);

    auto* conv_layer = trt_network->addConvolutionNd(*x, num_output, nvinfer1::Dims2(3,3), kernel_weight, bias_weight);
    trt_network->markOutput(*conv_layer->getOutput(0));
   conv_layer->getOutput(0)->setDynamicRange(-1, 1);


    for (int i = 0 ; i < trt_network->getInput(0)->getDimensions().nbDims; i++) {
        std::cout << trt_network->getInput(0)->getDimensions().d[i] << std::endl;
    }

    std::cout << "构建engine完毕，输出 shape：" << std::endl;
    auto engine = trt_builder->buildEngineWithConfig(*trt_network, *trt_config);
    auto engine_out_dims = engine->getBindingDimensions(0);
    for (int i = 0 ; i < engine_out_dims.nbDims; i++) {
        std::cout << engine_out_dims.d[i] << std::endl;
    }


    // 下面是运行时候拉！
    auto execution_context = engine->createExecutionContext();

    auto inspector = engine->createEngineInspector();
    inspector->setExecutionContext(execution_context);
    std::cout << inspector->getEngineInformation(nvinfer1::LayerInformationFormat::kJSON) << std::endl;

    a.d[0] = 1;
    a.d[1] = 128 * 14 * 14;
    execution_context->setBindingDimensions(0, a);
    int in_a_size = 1;
    for (int i = 0; i < a.nbDims; i++) in_a_size *= a.d[i];

    auto result_dims = execution_context->getBindingDimensions(1);

    int result_size = 1;
    std::cout << "execution_context时的shape：" << std::endl;
    for (int i = 0 ; i < result_dims.nbDims; i++) {
        std::cout << result_dims.d[i] << std::endl;
        result_size *= result_dims.d[i];
    }

    std::vector<void*> device_ptrs(2, nullptr);
    float* tmp = new float[result_size];

    for (int i = 0;i < in_a_size; i++) tmp[i] = 100;

    cudaMalloc( (void**)&device_ptrs[0], in_a_size * sizeof(float) );
    cudaMalloc( (void**)&device_ptrs[1], result_size * sizeof(float) );
    cudaMemcpy(device_ptrs[1], tmp, in_a_size * sizeof(float), cudaMemcpyHostToDevice );

    cudaStream_t stream;
    cudaStreamCreate(&stream);    
    execution_context->enqueueV2(device_ptrs.data(), stream, nullptr);
    cudaDeviceSynchronize();
    std::cout << "in_a_size" <<  in_a_size <<  std::endl;
    std::cout << "result_size " << result_size  << std::endl;
    cudaMemcpy(tmp, device_ptrs[1], result_size * sizeof(float), cudaMemcpyDeviceToHost );
}

