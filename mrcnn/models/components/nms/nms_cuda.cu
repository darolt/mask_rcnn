// ------------------------------------------------------------------
// Non-Maximum Suppression
// Licensed under The MIT License
// Written by Jean Da Rolt
// ------------------------------------------------------------------
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

#include <THC/THC.h>
#include <THC/THCDeviceUtils.cuh>

#include <vector>
#include <iostream>


int const block_size = sizeof(unsigned long long) * 8;


/**
 * Computes IoU of two boxes.
 */
__device__ inline float compute_iou(float const * const box1,
                                    float const * const box2) {
  float left = max(box1[0], box2[0]);
  float right = min(box1[2], box2[2]);
  float top = max(box1[1], box2[1]);
  float bottom = min(box1[3], box2[3]);
  float width = max(right - left + 1, 0.f);
  float height = max(bottom - top + 1, 0.f);
  float intersection = width * height;
  float area1 = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1);
  float area2 = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1);
  float union_ = area1 + area2 - intersection;
  return intersection/union_;
}

/**
 * Computes a matrix of NxN IoUs, with all combinations of 2
 * boxes.
 */
__global__ void compute_iou_kernel(
    const int nb_boxes,
    const float threshold,
    const float* const __restrict__ dev_boxes,
    unsigned long long* __restrict__ iou_matrix,
    const int c_row,
    const int c_col,
    const int nb_col_blocks,
    const int nb_elements_tri) {

  int row_idx = blockIdx.x / nb_col_blocks;
  int col_idx = blockIdx.x % nb_col_blocks;
  const int img_idx = blockIdx.y;
  const int thread_idx = threadIdx.x;
  if (col_idx < row_idx) {
    row_idx = c_row - row_idx;
    col_idx = c_col - col_idx;
  }

  const int row_size = min(nb_boxes - row_idx*block_size, block_size);
  const int col_size = min(nb_boxes - col_idx*block_size, block_size);
  int img_addr = img_idx*nb_boxes;

  __shared__ float block_boxes[block_size*4];
  if (thread_idx < col_size) {
    int block_addr = col_idx*block_size;
    int box_addr = (img_addr + block_addr + thread_idx)*4;
    for (int i=0; i<4; i++)
      block_boxes[thread_idx*4 + i] = dev_boxes[box_addr + i];
  }
  __syncthreads();

  if ((thread_idx < row_size)) {
    int block_addr = img_addr + row_idx*block_size + thread_idx;
    const float *cur_box = &dev_boxes[block_addr*4];
    unsigned long long t = 0;
    int start = (row_idx == col_idx) ? thread_idx + 1 : 0;
    for (int i = start; i < col_size; i++) {
      const float *box2 = &block_boxes[i*4];
      if (compute_iou(cur_box, box2) > threshold) {
        t |= 1ULL << i;
      }
    }
    int new_img_idx = img_idx*nb_elements_tri*block_size;
    int new_idx = thread_idx*nb_elements_tri + blockIdx.x;
    iou_matrix[new_img_idx + new_idx] = t;
  }
}

/**
 * Given the IoU matrix, computes NMS.
 */
inline at::Tensor compute_nms(
  const int nb_col_blocks,
  const int batch_size,
  const int nb_max_proposals,
  const int nb_elements_tri,
  const int nb_boxes,
  std::vector<unsigned long long> iou_matrix_host,
  const at::Tensor boxes) {

  std::vector<unsigned long long> suppress(nb_col_blocks);

  at::Tensor keep = at::empty({batch_size, nb_max_proposals},
                              boxes.options().dtype(at::kLong)
                                             .device(at::kCPU)
                                             .requires_grad(false));

  // TODO use threads
  for (int img_idx = 0; img_idx < batch_size; img_idx++) {
    std::fill(suppress.begin(), suppress.end(), 0);
    int num_to_keep = 0;
    for (int i = 0; i < nb_boxes; i++) {
      int nblock = i / block_size;
      int inblock = i % block_size;

      if (!(suppress[nblock] & (1ULL << inblock))) {
        keep[img_idx][num_to_keep++] = i;
        unsigned long long *p = &iou_matrix_host[0] +
                                img_idx*nb_elements_tri*block_size +
                                inblock*nb_elements_tri;
        for (int j = nblock; j < nb_col_blocks; j++) {
          int linear_idx = nblock*nb_col_blocks + j%nb_col_blocks;
          suppress[j] |= p[linear_idx];
        }
        if (num_to_keep == nb_max_proposals)
          break;
      }
    }
  }
  return keep;
}

at::Tensor nms_cuda(const at::Tensor boxes,
                    const float threshold,
                    const int nb_max_proposals) {
  int batch_size = boxes.size(0);
  int nb_boxes = boxes.size(1);

  const int nb_col_blocks = THCCeilDiv(nb_boxes, block_size);

  THCState *state = at::globalContext().lazyInitCUDA();

  // compute triangular number
  int nb_elements_tri = 0;
  for (int i=0; i < nb_col_blocks; i++)
    nb_elements_tri += i;
  int pivot_row = nb_elements_tri/nb_col_blocks;
  int pivot_col = nb_elements_tri % nb_col_blocks;
  int is_even = nb_col_blocks % 2;
  int c_row = 2*pivot_row + is_even;
  int c_col = 2*pivot_col + is_even + 1;

  unsigned int sizeofULL = sizeof(unsigned long long);
  unsigned int iou_matrix_size = batch_size*nb_elements_tri*block_size;
  unsigned int iou_matrix_size_bytes = iou_matrix_size*sizeofULL;
  auto iou_matrix = (unsigned long long*) THCudaMalloc(state, iou_matrix_size_bytes);

  dim3 grid_dim(nb_elements_tri, batch_size);
  dim3 block_dim(block_size);
  compute_iou_kernel<<<grid_dim, block_dim>>>(
    nb_boxes, threshold, boxes.data<float>(), iou_matrix, c_row,
    c_col, nb_col_blocks, nb_elements_tri);

  std::vector<unsigned long long> iou_matrix_host(iou_matrix_size);
  THCudaCheck(cudaMemcpy(&iou_matrix_host[0], iou_matrix,
                         iou_matrix_size_bytes, cudaMemcpyDeviceToHost));

  auto keep = compute_nms(nb_col_blocks, batch_size, nb_max_proposals,
                          nb_elements_tri, nb_boxes, iou_matrix_host,
                          boxes);

  THCudaFree(state, iou_matrix);
  return keep.to(boxes.device());
}