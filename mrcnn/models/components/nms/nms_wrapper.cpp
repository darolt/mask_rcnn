// ------------------------------------------------------------------
// Non-Maximum Suppression
// Licensed under The MIT License
// Written by Jean Da Rolt
// ------------------------------------------------------------------
#include <torch/extension.h>

#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_INPUT(x) CHECK_CUDA(x);


// CUDA forward declarations
at::Tensor nms_cuda(const at::Tensor& boxes,
                    const float threhsold,
                    const int nb_max_proposals);


void validate_inputs(at::Tensor boxes,
                     const at::Tensor scores,
                     const float threshold,
                     const int proposal_count) {
  CHECK_INPUT(boxes);
  assert((boxes.dim() == 3 && boxes.sizes(2) == 4) &&
         "Boxes must have shape (batch_size, nb_boxes, 4)");
  CHECK_INPUT(scores);
  assert((scores.dim() == 2) &&
         "Scores must have shape (batch_size, nb_boxes)");
  assert((scores.sizes(0) == boxes.sizes(0) &&
          scores.sizes(1) == boxes.sizes(1) ) &&
         "NMS Threshold should be in (0,1] range.");
  assert((threshold > 0 && threshold <= 1) &&
         "NMS Threshold should be in (0,1] range.");
  assert((proposal_count > 0) &&
         "Number of maximum proposals should be greater than 1.");
  assert((proposal_count <= boxes.sizes(1)) &&
         "Number of maximum proposals should be less than nb of boxes.");
  return;
}


at::Tensor nms_indexes(at::Tensor boxes,
                       const at::Tensor scores,
                       const float threshold,
                       const int proposal_count) {
  validate_inputs(boxes, scores, threshold, proposal_count);

  // order boxes by score
  auto order = std::get<1>(scores.sort(1, true));
  order = order.unsqueeze(2).expand({-1, -1, 4});
  auto boxes_sorted = boxes.gather(1, order).contiguous();

  return nms_cuda(boxes_sorted, threshold, proposal_count);
}

// CUDA proxy
at::Tensor nms_wrapper(at::Tensor boxes,
                       const at::Tensor scores,
                       const float threshold,
                       const int proposal_count) {

  auto keep = nms_indexes(boxes, scores, threshold, proposal_count);

  // filter original boxes using NMS results
  keep = keep.unsqueeze(2).expand({-1, -1, 4});
  return boxes.gather(1, keep).contiguous();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("nms_wrapper", &nms_wrapper, "NMS (CUDA)");
  m.def("nms_indexes", &nms_indexes, "NMS (CUDA)");
}