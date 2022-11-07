import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np

logger = trt.Logger(trt.Logger.WARNING)
builder = trt.Builder(logger)
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
parser = trt.OnnxParser(network, logger)

model_path = "/zhoukangkang/old_tipc-opt/ait_examples/bert/model_folded.onnx"

success = parser.parse_from_file(model_path)
config = builder.create_builder_config()
config.set_flag(trt.BuilderFlag.FP16)
config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30) # 1 MiB
profile = builder.create_optimization_profile()


batch = 1
seq_len = 128
input_shape = [batch, seq_len]

profile.set_shape("input.1", input_shape, input_shape, input_shape)
profile.set_shape("attention_mask", input_shape, input_shape, input_shape)
profile.set_shape("input.3", input_shape, input_shape, input_shape)

config.add_optimization_profile(profile)
serialized_engine = builder.build_serialized_network(network, config)

# with open("sample.engine", "wb") as f:
#     f.write(serialized_engine)

runtime = trt.Runtime(logger)
engine = runtime.deserialize_cuda_engine(serialized_engine)
context = engine.create_execution_context()

context.set_binding_shape(0, input_shape)
context.set_binding_shape(1, input_shape)
context.set_binding_shape(2, input_shape)

input_name = "input.1"
output_name = "1816"
input_idx = engine[input_name]
output_idx = engine[output_name]

h_input0 = cuda.pagelocked_empty(trt.volume(engine.get_binding_shape(0)),dtype=np.int64)
h_input1 = cuda.pagelocked_empty(trt.volume(engine.get_binding_shape(1)),dtype=np.int64)
h_input2 = cuda.pagelocked_empty(trt.volume(engine.get_binding_shape(2)),dtype=np.int64)
h_output =cuda.pagelocked_empty(trt.volume(engine.get_binding_shape(3)),dtype=np.float16)

d_input0 = cuda.mem_alloc(h_input0.nbytes)
d_input1 = cuda.mem_alloc(h_input1.nbytes)
d_input2 = cuda.mem_alloc(h_input2.nbytes)
d_output = cuda.mem_alloc(h_output.nbytes)

print(h_input0.nbytes)
print(h_output.nbytes)

#创建一个流，在其中复制输入/输出并运行推断
stream = cuda.Stream()
cuda.memcpy_htod_async(d_input0, h_input0, stream)
cuda.memcpy_htod_async(d_input0, h_input0, stream)
cuda.memcpy_htod_async(d_input0, h_input0, stream)
stream.synchronize()

import datetime
import time

for i in range(100):
    starttime = datetime.datetime.now()
    context.execute_async(bindings=[int(d_input0), int(d_input1), int(d_input2), int(d_output)], stream_handle=stream.handle)
    stream.synchronize()
    endtime = datetime.datetime.now()
    duringtime = endtime - starttime
    print (duringtime.seconds * 1000 + duringtime.microseconds / 1000.0)# 单位是毫秒
