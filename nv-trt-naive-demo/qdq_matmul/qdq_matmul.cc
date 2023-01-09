#include "/usr/local/tensorrt/include/NvInfer.h"
#include <vector>
#include <iostream>
#include <stdio.h>
#include <cstring>

// g++ qdq_matmul.cc -lnvinfer -lcudart  -L /usr/local/tensorrt/lib/ -L/usr/local/cuda/lib64 -I /usr/local/cuda/include
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

    nvinfer1::Dims input0_dims;
    input0_dims.nbDims = 2;
    input0_dims.d[0] = -1;
    input0_dims.d[1] = 224;


    trt_network->addInput("foo0", nvinfer1::DataType::kFLOAT, input0_dims);

    int m = 224;
    int n = 224;
    float* weight_ptr = new float[m * n];
    memset(weight_ptr, 0, sizeof(float) * m * n);

  nvinfer1::Weights kernel_weight;
  kernel_weight.type = nvinfer1::DataType::kFLOAT;
  kernel_weight.values = static_cast<void*>(weight_ptr);
  kernel_weight.count = m * n;
  nvinfer1::Dims kernel_dims;
  kernel_dims.nbDims = 2;
  kernel_dims.d[0] = m;
  kernel_dims.d[1] = n;
  auto kernel_tensor = trt_network->addConstant(kernel_dims, kernel_weight)->getOutput(0);
  float kernel_scale[n] = {0};

  nvinfer1::Weights kernel_scale_weight;
  kernel_scale_weight.type = nvinfer1::DataType::kFLOAT;
  kernel_scale_weight.values = static_cast<void*>(kernel_scale);
  kernel_scale_weight.count = n;
  nvinfer1::Dims kernel_scale_dims;
  kernel_scale_dims.nbDims = 1;
  kernel_scale_dims.d[0] = n;
  auto kernel_scale_tensor = trt_network->addConstant(kernel_scale_dims, kernel_scale_weight)->getOutput(0);
  auto* kernel_quant_layer = trt_network->addQuantize(*kernel_tensor, *kernel_scale_tensor);
  kernel_quant_layer->setAxis(1);
  auto kernel_quant_tensor =  kernel_quant_layer->getOutput(0);
  auto* kernel_dequant_layer = trt_network->addDequantize(*kernel_quant_tensor, *kernel_scale_tensor);
  kernel_dequant_layer->setAxis(1);
  auto kernel_dequant_tensor =  kernel_dequant_layer->getOutput(0);


nvinfer1::IOptimizationProfile* profile1 = trt_builder->createOptimizationProfile();
profile1->setDimensions("foo0", nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims2(1,224));
profile1->setDimensions("foo0", nvinfer1::OptProfileSelector::kOPT, nvinfer1::Dims2(3,224));
profile1->setDimensions("foo0", nvinfer1::OptProfileSelector::kMAX, nvinfer1::Dims2(128,224));
trt_config->addOptimizationProfile(profile1);


  nvinfer1::Weights scale_weight;
  float scale_ptr[1] = {1.2756160497665 / 127.f};
  scale_weight.type = nvinfer1::DataType::kFLOAT;
  scale_weight.values = static_cast<void*>(scale_ptr);
  scale_weight.count = 1;
  nvinfer1::Dims scale_dims;
  scale_dims.nbDims = 1;
  scale_dims.d[0] = 1;
  auto* scale_tensor = trt_network->addConstant(scale_dims, scale_weight)->getOutput(0);



    auto* x = trt_network->getInput(0);
    auto* quant_layer = trt_network->addQuantize(*x, *scale_tensor);
    quant_layer->setAxis(1);
    x = quant_layer->getOutput(0);
    auto* dequant_layer = trt_network->addDequantize(*x, *scale_tensor);
    dequant_layer->setAxis(1);
    x = dequant_layer->getOutput(0);

   // 必须将 kernel_weight设置为空！
    auto* conv_layer = trt_network->addMatrixMultiply(*x, nvinfer1::MatrixOperation::kNONE,
                                                      *kernel_dequant_tensor, nvinfer1::MatrixOperation::kNONE);
    trt_network->markOutput(*conv_layer->getOutput(0));

    for (int i = 0 ; i < trt_network->getInput(0)->getDimensions().nbDims; i++) {
        std::cout << trt_network->getInput(0)->getDimensions().d[i] << std::endl;
    }

    std::cout << "构建engine完毕，输出 shape：" << std::endl;
    auto engine = trt_builder->buildEngineWithConfig(*trt_network, *trt_config);
    auto engine_out_dims = engine->getBindingDimensions(0);
    for (int i = 0 ; i < engine_out_dims.nbDims; i++)
    {
        std::cout << engine_out_dims.d[i] << std::endl;
    }
    
    auto execution_context = engine->createExecutionContext();





    nvinfer1::Dims runtime_input0_dims;
    runtime_input0_dims.nbDims = 2;
    runtime_input0_dims.d[0] = 1;
    runtime_input0_dims.d[1] = 224;
    execution_context->setBindingDimensions(0, runtime_input0_dims);
    int runtime_input0_size = 1;
    for (int i = 0; i < runtime_input0_dims.nbDims; i++) runtime_input0_size *= runtime_input0_dims.d[i];

    auto runtime_result_dims = execution_context->getBindingDimensions(1);

    int runtime_result_size = 1;
    std::cout << "execution_context时的shape：" << std::endl;
    for (int i = 0 ; i < runtime_result_dims.nbDims; i++) {
        std::cout << runtime_result_dims.d[i] << std::endl;
        runtime_result_size *= runtime_result_dims.d[i];
    }

    std::vector<void*> device_ptrs(2, nullptr);
    float* runtime_input0_ptr = new float[runtime_input0_size];
    memset(runtime_input0_ptr, 0, sizeof(float) * runtime_input0_size);
    float* runtime_result_ptr = new float[runtime_result_size];

    for (int i = 0;i < runtime_input0_size; i++) runtime_input0_ptr[i] = 1.0;

    cudaMalloc( (void**)&device_ptrs[0], runtime_input0_size * sizeof(float) );
    cudaMalloc( (void**)&device_ptrs[1], runtime_result_size * sizeof(float) );
    cudaMemcpy(device_ptrs[0], runtime_input0_ptr, runtime_input0_size * sizeof(float), cudaMemcpyHostToDevice );

    cudaStream_t stream;
    cudaStreamCreate(&stream);    
    execution_context->enqueueV2(device_ptrs.data(), stream, nullptr);
    cudaDeviceSynchronize();
    std::cout << "in_a_size" <<  runtime_input0_size <<  std::endl;
    std::cout << "result_size " << runtime_result_size   << std::endl;
    cudaMemcpy(runtime_result_ptr, device_ptrs[1], runtime_result_size  * sizeof(float), cudaMemcpyDeviceToHost );

    free(runtime_input0_ptr);
    free(runtime_result_ptr);

    cudaFree(device_ptrs[0]);
    cudaFree(device_ptrs[1]);
}
