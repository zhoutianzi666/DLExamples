import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np

# input_shape = [1, 64, 576, 576]
# h_input0 = cuda.pagelocked_empty(trt.volume(input_shape),dtype=np.float32)

# npzfile = np.load('/root/paddlejob/workspace/env_run/zkk/outfile.npz')
# h_input0 = npzfile['input_tensor']
# print(np.std(h_input0), np.mean(h_input0))
# exit(0)

logger = trt.Logger(trt.Logger.WARNING)
builder = trt.Builder(logger)
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
parser = trt.OnnxParser(network, logger)

model_path = "/root/paddlejob/workspace/env_run/zkk/model.onnx"

success = parser.parse_from_file(model_path)
config = builder.create_builder_config()
# config.set_flag(trt.BuilderFlag.F32)
config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 35)
profile = builder.create_optimization_profile()


input_shape = [1, 64, 576, 576]

profile.set_shape("pts_feat", input_shape, input_shape, input_shape)

config.add_optimization_profile(profile)
config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED



serialized_engine = builder.build_serialized_network(network, config)
runtime = trt.Runtime(logger)
engine = runtime.deserialize_cuda_engine(serialized_engine)

print("save engine for later use.")
with open("engine_file_path", "wb") as f:
    f.write(engine.serialize())

# with open("engine_file_path", "rb") as f, trt.Runtime(logger) as runtime:
#     engine = runtime.deserialize_cuda_engine(f.read())


inspector = engine.create_engine_inspector()
a = inspector.get_engine_information(trt.LayerInformationFormat.JSON)
print(a)


context = engine.create_execution_context()

context.set_binding_shape(0, input_shape)

input_name = "pts_feat"
output_name = "conv2d_103.tmp_1"
input_idx = engine[input_name]
output_idx = engine[output_name]

h_input0 = cuda.pagelocked_empty(trt.volume(engine.get_binding_shape(0)),dtype=np.float32)
h_output =cuda.pagelocked_empty(trt.volume(engine.get_binding_shape(1)),dtype=np.float32)

npzfile = np.load('/root/paddlejob/workspace/env_run/zkk/outfile.npz')
h_input0 = npzfile['input_tensor']

d_input0 = cuda.mem_alloc(h_input0.nbytes)
d_output = cuda.mem_alloc(h_output.nbytes)

#创建一个流，在其中复制输入/输出并运行推断
stream = cuda.Stream()
cuda.memcpy_htod_async(d_input0, h_input0, stream)
stream.synchronize()

import datetime
import time

for i in range(10):
    cuda.memcpy_htod_async(d_input0, h_input0, stream)
    starttime = datetime.datetime.now()
    context.execute_async(bindings=[int(d_input0), int(d_output)], stream_handle=stream.handle)
    stream.synchronize()
    endtime = datetime.datetime.now()
    duringtime = endtime - starttime
    cuda.memcpy_dtoh_async(h_output, d_output, stream)
    print (duringtime.seconds * 1000 + duringtime.microseconds / 1000.0)# 单位是毫

cuda.memcpy_dtoh_async(h_output, d_output, stream)
print(np.std(h_output), np.mean(h_output))
