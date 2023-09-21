




- 这个demo只是想说明，我如果不给conv的输出设置scale，那么conv就很有可能走sm80_xmma_fprop_implicit_gemm_interleaved_i8f32_i8i32_f32_n这样的kernel！




{"Layers": [{
  "Name": "(Unnamed Layer* 0) [Shuffle]",
  "LayerType": "NoOp",
  "Inputs": [
  {
    "Name": "foo0",
    "Location": "Device",
    "Dimensions": [-1,25088],
    "Format/Datatype": "N/A due to dynamic shapes"
  }],
  "Outputs": [
  {
    "Name": "(Unnamed Layer* 0) [Shuffle]_output",
    "Location": "Device",
    "Dimensions": [-1,128,14,14],
    "Format/Datatype": "N/A due to dynamic shapes"
  }],
  "TacticValue": "0x0000000000000000"
},{
  "Name": "Reformatting CopyNode for Input Tensor 0 to (Unnamed Layer* 1) [Convolution]",
  "LayerType": "Reformat",
  "Inputs": [
  {
    "Name": "(Unnamed Layer* 0) [Shuffle]_output",
    "Location": "Device",
    "Dimensions": [-1,128,14,14],
    "Format/Datatype": "N/A due to dynamic shapes"
  }],
  "Outputs": [
  {
    "Name": "Reformatted Input Tensor 0 to (Unnamed Layer* 1) [Convolution]",
    "Location": "Device",
    "Dimensions": [-1,128,14,14],
    "Format/Datatype": "N/A due to dynamic shapes"
  }],
  "ParameterType": "Reformat",
  "Origin": "REFORMAT",
  "TacticValue": "0x00000000000003e8"
},{
  "Name": "(Unnamed Layer* 1) [Convolution]",
  "LayerType": "CaskConvolution",
  "Inputs": [
  {
    "Name": "Reformatted Input Tensor 0 to (Unnamed Layer* 1) [Convolution]",
    "Location": "Device",
    "Dimensions": [-1,128,14,14],
    "Format/Datatype": "N/A due to dynamic shapes"
  }],
  "Outputs": [
  {
    "Name": "(Unnamed Layer* 1) [Convolution]_output",
    "Location": "Device",
    "Dimensions": [-1,12,12,12],
    "Format/Datatype": "N/A due to dynamic shapes"
  }],
  "ParameterType": "Convolution",
  "Kernel": [3,3],
  "PaddingMode": "kEXPLICIT_ROUND_DOWN",
  "PrePadding": [0,0],
  "PostPadding": [0,0],
  "Stride": [1,1],
  "Dilation": [1,1],
  "OutMaps": 12,
  "Groups": 1,
  "Weights": {"Type": "Int8", "Count": 13824},
  "Bias": {"Type": "Float", "Count": 12},
  "HasSparseWeights": 0,
  "Activation": "NONE",
  "HasBias": 1,
  "HasReLU": 0,
  "TacticName": "sm80_xmma_fprop_implicit_gemm_interleaved_i8f32_i8i32_f32_nchw_vect_c_32kcrs_vect_c_32_nchw_tilesize64x32x64_stage6_warpsize2x1x1_g1_tensor16x8x32_t1r3s3_alignc4_nchw",
  "TacticValue": "0xff4e814e86dc840c"
}],
"Bindings": ["foo0"
,"(Unnamed Layer* 1) [Convolution]_output"
]}




