import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np

logger = trt.Logger(trt.Logger.WARNING)
builder = trt.Builder(logger)
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
parser = trt.OnnxParser(network, logger)

model_path = "./prune_model.onnx"
precision = "bf16"

success = parser.parse_from_file(model_path)
config = builder.create_builder_config()
if precision == "fp16":
   config.set_flag(trt.BuilderFlag.FP16)
elif precision == "bf16":
    config.set_flag(trt.BuilderFlag.BF16)
config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 35)
profile = builder.create_optimization_profile()


input_shape = [6, 465, 720, 3]

profile.set_shape("stack_0.tmp_0", input_shape, input_shape, input_shape)

config.add_optimization_profile(profile)
config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED

engine_file_path = "engine_file_path_" + precision
if os.path.exists(engine_file_path):
    with open(engine_file_path, "rb") as f, trt.Runtime(logger) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
else:
    serialized_engine = builder.build_serialized_network(network, config)
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(serialized_engine)
    print("save engine for later use.")
    with open(engine_file_path, "wb") as f:
        f.write(engine.serialize())

context = engine.create_execution_context()
context.set_binding_shape(0, input_shape)

h_input0 = cuda.pagelocked_empty(trt.volume(engine.get_binding_shape(0)),dtype=np.float32)
h_input0 = np.zeros(h_input0.shape).astype(np.float32)
h_output =cuda.pagelocked_empty(trt.volume(engine.get_binding_shape(1)),dtype=np.float32)

d_input0 = cuda.mem_alloc(h_input0.nbytes)
d_output = cuda.mem_alloc(h_output.nbytes)

stream = cuda.Stream()
for i in range(10):
    cuda.memcpy_htod_async(d_input0, h_input0, stream)
    context.execute_async(bindings=[int(d_input0), int(d_output)], stream_handle=stream.handle)
    cuda.memcpy_dtoh_async(h_output, d_output, stream)
stream.synchronize()


import datetime
import time

stream.synchronize()
starttime = datetime.datetime.now()

for i in range(10):
    cuda.memcpy_htod_async(d_input0, h_input0, stream)
    context.execute_async(bindings=[int(d_input0), int(d_output)], stream_handle=stream.handle)
    cuda.memcpy_dtoh_async(h_output, d_output, stream)

stream.synchronize()
endtime = datetime.datetime.now()
duringtime = endtime - starttime
print (duringtime.seconds * 1000 + duringtime.microseconds / 1000.0)# 单位是毫

# fp32 is : 41.957417 -35.9053,  81.156 ms
# bf16 is : 41.957417 -35.9053,  83.132 ms
# fp16 is : 41.98418 -35.900158, 53.892 ms
print(np.std(h_output), np.mean(h_output))
