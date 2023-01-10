


(Unnamed Layer* 0) [Constant] + (Unnamed Layer* 2) [Quantize] + transpose_before_(Unnamed Layer* 7) [Matrix Multiply] + (Unnamed Layer* 7) [Matrix Multiply]
Begins: 18.3144s
Ends: 18.3144s (+53.810 μs)
Thread: 117566



- 还真跑了int8 的 kernel！
- 权重输入也必须是qdq哦！

```cpp
sm80_xmma_fprop_implicit_gemm_interleaved_i8f32_i8i32_f32_nchw_vect_c_32kcrs_vect_c_32_nchw_vect_c_32_tilesize64x32x64_stage6_warpsize2x1x1_g1_tensor16x8x32_simple_t1r1s1_execute_kernel_trt
Begins: 18.3144s
Ends: 18.3144s (+12.384 μs)
grid:  <<<40, 1, 1>>>
block: <<<64, 1, 1>>>
Launch Type: Regular
Static Shared Memory: 0 bytes
Dynamic Shared Memory: 36,864 bytes
Registers Per Thread: 124
Local Memory Per Thread: 0 bytes
Local Memory Total: 244,187,136 bytes
Shared Memory executed: 167,936 bytes
Shared Memory Bank Size: 4 B
Theoretical occupancy: 12.5 %
Launched from thread: 117566
Latency: ←20.559 μs
Correlation ID: 39902
Stream: Stream 34
```

- 我要是在下面加上个qdq+激活，结果变成了下面这个kernel了！

```cpp
sm80_xmma_fprop_implicit_gemm_interleaved_i8i8_i8i32_f32_nchw_vect_c_32kcrs_vect_c_32_nchw_vect_c_32_tilesize32x32x64_stage6_warpsize2x1x1_g1_tensor16x8x32_t1r1s1_linkable_execute_kernel_trt
```

- 上面的fp32是后处理的来干的东西哦！
- [ElementComputeEpilogue](https://github.com/NVIDIA/cutlass/blob/a1046d49c18465ae8f25187c4c4f3db9ea1278f2/examples/09_turing_tensorop_conv2dfprop/turing_tensorop_conv2dfprop.cu#L187)

```cpp
Begins: 23.2454s
Ends: 23.2454s (+12.032 μs)
grid:  <<<40, 1, 1>>>
block: <<<64, 1, 1>>>
Launch Type: Regular
Static Shared Memory: 0 bytes
Dynamic Shared Memory: 24,576 bytes
Registers Per Thread: 86
Local Memory Per Thread: 0 bytes
Local Memory Total: 244,187,136 bytes
Shared Memory executed: 167,936 bytes
Shared Memory Bank Size: 4 B
Theoretical occupancy: 18.75 %
Launched from thread: 121093
Latency: ←19.250 μs
Correlation ID: 75802
Stream: Stream 34
```
