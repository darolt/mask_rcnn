// ------------------------------------------------------------------
// Non-Maximum Suppression
// Licensed under The MIT License
// Written by Jean Da Rolt
// ------------------------------------------------------------------
#include <torch/extension.h>

#include <vector>

#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_INPUT(x) CHECK_CUDA(x);


// CUDA forward declarations
at::Tensor nms_cuda(const at::Tensor& boxes,
                    const float threhsold,
                    const int nb_max_proposals);


// CUDA proxy
at::Tensor nms_wrapper(at::Tensor boxes,
                       const at::Tensor scores,
                       const float threshold,
                       const int proposal_count) {
  CHECK_INPUT(boxes);
  assert((boxes.dim() == 3) &&
         "NMS input must have shape (batch_size, nb_boxes, 5)");
  assert((threshold > 0 && threshold <= 1) &&
         "NMS Threshold should be in (0,1] range.");

  // order boxes by score
  auto order = std::get<1>(scores.sort(1, true));
  order = order.unsqueeze(2).expand({-1, -1, 4});
  auto boxes_sorted = boxes.gather(1, order).contiguous();

  auto keep = nms_cuda(boxes_sorted, threshold, proposal_count);

  // filter original boxes using NMS results
  keep = keep.unsqueeze(2).expand({-1, -1, 4});
  boxes = boxes.gather(1, keep);

  return boxes;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("nms_wrapper", &nms_wrapper, "NMS (CUDA)");
}