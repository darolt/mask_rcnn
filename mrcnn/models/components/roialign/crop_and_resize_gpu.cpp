
#include <torch/extension.h>

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

#include <THC/THC.h>
#include "crop_and_resize_kernel.h"

extern THCState *state;


void crop_and_resize_gpu_forward(
    at::Tensor image,
    at::Tensor boxes,           // [y1, x1, y2, x2]
    at::Tensor box_index,    // range in [0, batch_size)
    const float extrapolation_value,
    const int crop_height,
    const int crop_width,
    at::Tensor crops) {

  const int batch_size = image.size(0);
  const int depth = image.size(1);
  const int image_height = image.size(2);
  const int image_width = image.size(3);

  const int num_boxes = boxes.size(0);

  // init output space
  crops.resize_({num_boxes, depth, crop_height, crop_width});

  cudaStream_t stream = THCState_getCurrentStream(state);
  CropAndResizeLaucher(
    image.data<float>(),
    boxes.data<float>(),
    box_index.data<int>(),
    num_boxes, batch_size, image_height, image_width,
    crop_height, crop_width, depth, extrapolation_value,
    crops.data<float>(),
    stream
  );
}


void crop_and_resize_gpu_backward(
    THCudaTensor * grads,
    THCudaTensor * boxes,      // [y1, x1, y2, x2]
    THCudaIntTensor * box_index,    // range in [0, batch_size)
    THCudaTensor * grads_image // resize to [bsize, c, hc, wc]
) {
    // shape
    const int batch_size = THCudaTensor_size(state, grads_image, 0);
    const int depth = THCudaTensor_size(state, grads_image, 1);
    const int image_height = THCudaTensor_size(state, grads_image, 2);
    const int image_width = THCudaTensor_size(state, grads_image, 3);

    const int num_boxes = THCudaTensor_size(state, grads, 0);
    const int crop_height = THCudaTensor_size(state, grads, 2);
    const int crop_width = THCudaTensor_size(state, grads, 3);

    // init output space
    THCudaTensor_zero(state, grads_image);

    cudaStream_t stream = THCState_getCurrentStream(state);
    CropAndResizeBackpropImageLaucher(
        THCudaTensor_data(state, grads),
        THCudaTensor_data(state, boxes),
        THCudaIntTensor_data(state, box_index),
        num_boxes, batch_size, image_height, image_width,
        crop_height, crop_width, depth,
        THCudaTensor_data(state, grads_image),
        stream
    );
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("crop_and_resize_gpu_forward", &crop_and_resize_gpu_forward, "Crop and Resize (CUDA)");
  m.def("crop_and_resize_gpu_backward", &crop_and_resize_gpu_backward, "Crop and Resize (CUDA)");
}