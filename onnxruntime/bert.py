import onnxruntime as ort
import numpy as np
import datetime
import time

sess_opt = ort.SessionOptions()
trt_providers = [('TensorrtExecutionProvider', {
    'device_id': 0,
    'trt_max_workspace_size': 1073741824,
    'trt_fp16_enable': True,
    'trt_int8_enable': False
}), ('CUDAExecutionProvider', {
    'device_id': 0,
    'arena_extend_strategy': 'kNextPowerOfTwo',
    'gpu_mem_limit': 2 * 1024 * 1024 * 1024,
    'cudnn_conv_algo_search': 'EXHAUSTIVE',
    'do_copy_in_default_stream': True,
})]
model_file = "/zhoukangkang/old_tipc-opt/ait_examples/bert/model.onnx"
#sess_opt.optimized_model_filepath = './model_optim.onnx'
sess = ort.InferenceSession(model_file, sess_options=sess_opt, providers=trt_providers)
input_data = {}

batch = 16
seq_len = 128
real_input = np.zeros([batch, seq_len], dtype=np.int64)
name = "input.1"
input_data[name] = real_input
name = "attention_mask"
input_data[name] = real_input
name = "input.3"
input_data[name] = real_input


X_ortvalue = ort.OrtValue.ortvalue_from_numpy(real_input, 'cuda', 0)
io_binding = sess.io_binding()
io_binding.bind_input(name='input.1', device_type=X_ortvalue.device_name(), device_id=0, element_type=np.int64, shape=X_ortvalue.shape(), buffer_ptr=X_ortvalue.data_ptr())
io_binding.bind_input(name='attention_mask', device_type=X_ortvalue.device_name(), device_id=0, element_type=np.int64, shape=X_ortvalue.shape(), buffer_ptr=X_ortvalue.data_ptr())
io_binding.bind_input(name='input.3', device_type=X_ortvalue.device_name(), device_id=0, element_type=np.int64, shape=X_ortvalue.shape(), buffer_ptr=X_ortvalue.data_ptr())
Y_ortvalue = ort.OrtValue.ortvalue_from_shape_and_type([batch, seq_len, 30522], np.float16, 'cuda', 0)  # Change the shape to the actual shape of the output being bound
io_binding.bind_output(name='1816', device_type=Y_ortvalue.device_name(), device_id=0, element_type=np.float16, shape=Y_ortvalue.shape(), buffer_ptr=Y_ortvalue.data_ptr())

for i in range(100):
    starttime = datetime.datetime.now()
    sess.run_with_iobinding(io_binding)
    endtime = datetime.datetime.now()
    duringtime = endtime - starttime
    print (duringtime.seconds * 1000 + duringtime.microseconds / 1000.0)# 单位是毫秒

# for i in range(100):
#     starttime = datetime.datetime.now()
#     output = sess.run(None, input_data)
#     print(output[0].shape)
#     endtime = datetime.datetime.now()
#     duringtime = endtime - starttime
#     print (duringtime.seconds * 1000 + duringtime.microseconds / 1000.0)# 单位是毫秒
