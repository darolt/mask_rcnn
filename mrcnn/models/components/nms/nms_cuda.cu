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
#include <thread>

unsigned int sizeofULL = sizeof(unsigned long long);
int const block_size = sizeof(unsigned long long) * 8;
extern THCState *state;

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

  // map linear grid to rectangular grid
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

  // copy to block shared memory
  __shared__ float block_boxes[block_size*4];
  if (thread_idx < col_size) {
    int block_addr = col_idx*block_size;
    int box_addr = (img_addr + block_addr + thread_idx)*4;
    for (int i=0; i<4; i++)
      block_boxes[thread_idx*4 + i] = dev_boxes[box_addr + i];
  }
  __syncthreads();

  if ((thread_idx < row_size)) {
    // compute iou
    int block_addr = img_addr + row_idx*block_size + thread_idx;
    const float *cur_box = &dev_boxes[block_addr*4];
    unsigned long long t = 0;
    int start = (row_idx == col_idx) ? thread_idx + 1 : 0;
    for (int i = start; i < col_size; i++) {
      const float *box2 = &block_boxes[i*4];
      if (compute_iou(cur_box, box2) > threshold)
        t |= 1ULL << i;
    }
    // map rectangular grid back to linear grid
    int new_img_idx = img_idx*nb_elements_tri*block_size;
    int new_idx = thread_idx*nb_elements_tri + blockIdx.x;
    iou_matrix[new_img_idx + new_idx] = t;
  }
}

/**
 * Given the IoU matrix, computes NMS.
 */
void compute_nms(
    const int thread_idx,
    at::Tensor& keep,
    const int nb_col_blocks,
    const int nb_max_proposals,
    const int nb_elements_tri,
    const int nb_boxes,
    std::vector<unsigned long long>& iou_matrix_host) {

  std::vector<unsigned long long> suppress(nb_col_blocks);
  std::fill(suppress.begin(), suppress.end(), 0);

  int num_to_keep = 0, last_index = 0;
  for (int i = 0; i < nb_boxes; i++) {
    int nblock = i / block_size;
    int inblock = i % block_size;

    if (!(suppress[nblock] & (1ULL << inblock))) {
      keep[thread_idx][num_to_keep++] = i;
      last_index = 1;
      unsigned long long *p = &iou_matrix_host[0] +
                              thread_idx*nb_elements_tri*block_size +
                              inblock*nb_elements_tri;
      for (int j = nblock; j < nb_col_blocks; j++) {
        int linear_idx = nblock*nb_col_blocks + j%nb_col_blocks;
        suppress[j] |= p[linear_idx];
      }
      if (num_to_keep == nb_max_proposals)
        break;
    }

  }
  // nb_max_proposals was not reached, must fill other values
  while (num_to_keep < nb_max_proposals) {
    keep[thread_idx][num_to_keep++] = last_index;
  }

  return;
}

/**
 * Compute NMS.
 *
 * In order to reduce time and memory it works only with the upper
 * triangle of a matrix (it is not necessary to compute IoU twice).
 * Matrix is reduced in the following way:
 *   0   1  2  3  4  5       0  1  2  3  4  5      0  1  2  3  4  5
 *   6   7  8  9 10 11  p1      7  8  9 10 11 p2      7  8  9 10 11
 *   12 13 14 15 16 17  -->       14 15 16 17 -->    14 15 16 17
 *   18 19 20               18 19 20              20 19 18
 *                          12 13                    13 12
 *                          6                         6
 *
 *   0  1  2  3  4  5
 *      7  8  9 10 11
 *        14 15 16 17
 *           20 19 18
 *              13 12
 *                  6
 * new_row = p1 = rx + rx - row + n%2
 * new_col = p2 = cx + cx - col + n%2 + 1
 */
at::Tensor nms_cuda(const at::Tensor& boxes,
                    const float threshold,
                    const int nb_max_proposals) {
  int batch_size = boxes.size(0);
  int nb_boxes = boxes.size(1);

  const int nb_col_blocks = THCCeilDiv(nb_boxes, block_size);

  // compute triangular number
  int nb_elements_tri = 0;
  for (int i=1; i <= nb_col_blocks; i++)
    nb_elements_tri += i;
  int pivot_row = nb_elements_tri/nb_col_blocks;
  int pivot_col = nb_elements_tri % nb_col_blocks;
  int is_even = nb_col_blocks % 2;
  int c_row = 2*pivot_row + is_even;
  int c_col = 2*pivot_col + is_even + 1;

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

  THCudaFree(state, iou_matrix);
  at::Tensor keep = at::empty({batch_size, nb_max_proposals},
                              boxes.options().dtype(at::kLong)
                                             .device(at::kCPU)
                                             .requires_grad(false));

  std::thread threads[batch_size];
  for (int img_idx=0; img_idx < batch_size; img_idx++) {
    threads[img_idx] = std::thread(
      compute_nms,
      img_idx,
      std::ref(keep),
      nb_col_blocks,
      nb_max_proposals,
      nb_elements_tri,
      nb_boxes,
      std::ref(iou_matrix_host));
  }

  for (auto& thread: threads) {
    thread.join();
  }
  return keep.to(boxes.device());
}