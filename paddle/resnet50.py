import paddle
import paddle.inference as paddle_infer
import numpy as np
import datetime
import time

model_file="/zhoukangkang/old_tipc-opt/ait_examples/resnet/pd_model/model.pdmodel"
model_params="/zhoukangkang/old_tipc-opt/ait_examples/resnet/pd_model/model.pdiparams"
shape_range_file = "resnet50/shape.txt"

config = paddle_infer.Config(model_file, model_params)
#config.collect_shape_range_info(shape_range_file)
config.enable_tuned_tensorrt_dynamic_shape(shape_range_file, True)
config.enable_memory_optim()
config.switch_ir_optim(True)
#config.switch_ir_debug(True)
precision_mode = paddle_infer.PrecisionType.Half
config.enable_use_gpu(256, 1)
config.enable_tensorrt_engine(
    workspace_size=1073741824,
    precision_mode=precision_mode,
    max_batch_size=10,
    min_subgraph_size=3,
    use_calib_mode=False)
predictor = paddle_infer.create_predictor(config)
batch = 16


for i in range(100):

    name = "x0"
    input_tensor = predictor.get_input_handle(name)
    fake_input = np.zeros([batch,3,224,224], dtype=np.float32)
    input_tensor.copy_from_cpu(fake_input)

    #paddle.device.cuda.synchronize()
    starttime = datetime.datetime.now()
    predictor.run()
    #paddle.device.cuda.synchronize()
    endtime = datetime.datetime.now()
    duringtime = endtime - starttime
    print (duringtime.seconds * 1000 + duringtime.microseconds / 1000.0)# 单位是毫秒

    name = "linear_1.tmp_1"
    output_tensor = predictor.get_output_handle(name)
    output_data = output_tensor.copy_to_cpu()
    #print(output_data)


