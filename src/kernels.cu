#include <vector>

#include "../tester/utils.h"

template <typename T>
__global__ void countRanks(T *d_input, size_t n, int *d_count) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i >= n)
    return;

  T cur = d_input[i];
  int cnt = 0;
  for (int j = 0; j < n; j++) {
    if (d_input[j] > cur) {
      cnt++;
    }
  }
  d_count[i] = cnt;
}

// clang-format off
/**
 * @brief Find the k-th largest element in a vector using CUDA.
 * 
 * @tparam T Type of elements in the input vector (should support `int` and `float`).
 * @param h_input Host-side input vector.
 * @param k 1-based index of the element to find (e.g., `k=1` returns the largest element).
 * @return T The k-th largest element in `h_input`.

 * @note Must use CUDA kernels for all compute-intensive steps; no significant CPU allowed.
 * @note Library functions that can directly complete a significant part of the work are NOT allowed. 
 * @note For invalid cases, return T(-100).
 * @note Handles device memory management (allocate/copy/free) internally. Errors should be thrown.
 */
// clang-format on
template <typename T> T kthLargest(const std::vector<T> &h_input, size_t k) {
  const size_t n = h_input.size();
  if (h_input.empty() || k == 0 || k > n) {
    return T(-100);
  }

  T *d_input = nullptr;
  int *d_count = nullptr;
  cudaMalloc(&d_input, n * sizeof(T));
  cudaMalloc(&d_count, n * sizeof(int));
  cudaMemcpy(d_input, h_input.data(), n * sizeof(T), cudaMemcpyHostToDevice);

  dim3 blockSize(256);
  dim3 gridSize((n + blockSize.x - 1) / blockSize.x);
  countRanks<<<gridSize, blockSize>>>(d_input, n, d_count);
  cudaDeviceSynchronize();

  std::vector<int> h_count(n);
  cudaMemcpy(h_count.data(), d_count, n * sizeof(int), cudaMemcpyDeviceToHost);

  T result;
  for (int i = 0; i < n; i++) {
    if (h_count[i] == k - 1) {
      result = h_input[i];
      break;
    }
  }

  cudaFree(d_input);
  cudaFree(d_count);

  return result;
}

// each therad idx computes O[b, t, q, h]
// template <typename T>
// __global__ void flashAttentionKernel(T *Q, T *K, T *V, T *O, int batch_size,
//                                      int target_seq_len, int src_seq_len,
//                                      int query_heads, int kv_heads,
//                                      int head_dim, bool is_causal, size_t n)
//                                      {
//   size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
//   if (idx >= n)
//     return;

//   // O[b, t, q, h]
//   int h = idx % head_dim;
//   int q = (idx / head_dim) % query_heads;
//   int t = (idx / (head_dim * query_heads)) % target_seq_len;
//   int b = idx / (head_dim * query_heads * target_seq_len);

//   int group_size = query_heads / kv_heads;
//   int kv_h = q / group_size; // GQA

//   T max_logit = -INFINITY;
//   T denom = 0.0f;
//   T o = 0.0f;
//   for (int s = 0; s < src_seq_len; s++) {
//     if (is_causal && s > t)
//       continue;

//     T cur_logit = 0;
//     for (int hd = 0; hd < head_dim; hd++) {
//       cur_logit +=
//           Q[((b * target_seq_len + t) * query_heads + q) * head_dim + hd] *
//           K[((b * src_seq_len + s) * kv_heads + kv_h) * head_dim + hd];
//     }
//     cur_logit /= sqrtf(head_dim);
//     T new_max_logit = cur_logit > max_logit ? cur_logit : max_logit;
//     T x = exp(max_logit - new_max_logit);
//     T y = exp(cur_logit - new_max_logit);
//     T new_denom = denom * x + y;
//     o = o * x * (denom / new_denom) +
//         y / new_denom *
//             V[((b * src_seq_len + s) * kv_heads + kv_h) * head_dim + h];
//     max_logit = new_max_logit;
//     denom = new_denom;
//   }

//   O[idx] = o;
// }

template <typename T>
__global__ void flashAttentionKernel(T *Q, T *K, T *V, T *O, int batch_size,
                                     int target_seq_len, int src_seq_len,
                                     int query_heads, int kv_heads,
                                     int head_dim, bool is_causal, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n)
    return;

  // O[b, t, q, h]
  int h = idx % head_dim;
  int q = (idx / head_dim) % query_heads;
  int t = (idx / (head_dim * query_heads)) % target_seq_len;
  int b = idx / (head_dim * query_heads * target_seq_len);

  int kv_h = q * kv_heads / query_heads; // GQA

  // -------------------------------
  // 找到 logit 最大值（softmax 稳定）
  // -------------------------------
  T max_logit = -1e20;
  for (int s = 0; s < src_seq_len; s++) {
    // if (is_causal && s > t)
    //   continue;

    T dot = 0;
    for (int hd = 0; hd < head_dim; hd++) {
      dot += Q[((b * target_seq_len + t) * query_heads + q) * head_dim + hd] *
             K[((b * src_seq_len + s) * kv_heads + kv_h) * head_dim + hd];
    }
    if (dot > max_logit)
      max_logit = dot;
  }

  // -------------------------------
  // softmax 分母
  // -------------------------------
  T denom = 0;
  for (int s = 0; s < src_seq_len; s++) {
    if (is_causal && s > t)
      continue;

    T dot = 0;
    for (int hd = 0; hd < head_dim; hd++) {
      dot += Q[((b * target_seq_len + t) * query_heads + q) * head_dim + hd] *
             K[((b * src_seq_len + s) * kv_heads + kv_h) * head_dim + hd];
    }
    denom += exp(dot - max_logit);
  }

  // -------------------------------
  // 加权 V 得到输出
  // -------------------------------
  T out_val = 0;
  for (int s = 0; s < src_seq_len; s++) {
    if (is_causal && s > t)
      continue;

    T dot = 0;
    for (int hd = 0; hd < head_dim; hd++) {
      dot += Q[((b * target_seq_len + t) * query_heads + q) * head_dim + hd] *
             K[((b * src_seq_len + s) * kv_heads + kv_h) * head_dim + hd];
    }
    T weight = exp(dot - max_logit) / denom;
    out_val +=
        weight * V[((b * src_seq_len + s) * kv_heads + kv_h) * head_dim + h];
  }

  O[idx] = out_val;
}

// clang-format off
/**
 * @brief Computes flash attention for given query, key, and value tensors.
 * 
 * @tparam T Data type (float) for input/output tensors
 * @param[in] h_q Query tensor of shape [batch_size, tgt_seq_len, query_heads, head_dim]
 * @param[in] h_k Key tensor of shape [batch_size, src_seq_len, kv_heads, head_dim]
 * @param[in] h_v Value tensor of shape [batch_size, src_seq_len, kv_heads, head_dim]
 * @param[out] h_o Output attention tensor of shape [batch_size, tgt_seq_len, query_heads, head_dim]
 * @param[in] batch_size Batch dimension size
 * @param[in] target_seq_len Target sequence length
 * @param[in] src_seq_len Source sequence length  
 * @param[in] query_heads Number of query attention heads
 * @param[in] kv_heads Number of key/value heads (supports grouped query attention)
 * @param[in] head_dim Dimension size of each attention head
 * @param[in] is_causal Whether to apply causal masking
 */
// clang-format on
template <typename T>
void flashAttention(const std::vector<T> &h_q, const std::vector<T> &h_k,
                    const std::vector<T> &h_v, std::vector<T> &h_o,
                    int batch_size, int target_seq_len, int src_seq_len,
                    int query_heads, int kv_heads, int head_dim,
                    bool is_causal) {
  T *d_q = nullptr;
  T *d_k = nullptr;
  T *d_v = nullptr;
  T *d_o = nullptr;

  cudaMalloc(&d_q, h_q.size() * sizeof(T));
  cudaMalloc(&d_k, h_k.size() * sizeof(T));
  cudaMalloc(&d_v, h_v.size() * sizeof(T));
  cudaMalloc(&d_o, h_o.size() * sizeof(T));

  cudaMemcpy(d_q, h_q.data(), h_q.size() * sizeof(T), cudaMemcpyHostToDevice);
  cudaMemcpy(d_k, h_k.data(), h_k.size() * sizeof(T), cudaMemcpyHostToDevice);
  cudaMemcpy(d_v, h_v.data(), h_v.size() * sizeof(T), cudaMemcpyHostToDevice);

  size_t n = batch_size * target_seq_len * query_heads * head_dim;
  int blockSize = 256;
  int gridSize = (n + blockSize - 1) / blockSize;

  flashAttentionKernel<<<gridSize, blockSize>>>(
      d_q, d_k, d_v, d_o, batch_size, target_seq_len, src_seq_len, query_heads,
      kv_heads, head_dim, is_causal, n);

  cudaMemcpy(h_o.data(), d_o, h_o.size() * sizeof(T), cudaMemcpyDeviceToHost);

  cudaFree(d_q);
  cudaFree(d_k);
  cudaFree(d_v);
  cudaFree(d_o);
}

// clang-format off
// *********************************************************************
// Explicit Template Instantiations (REQUIRED FOR LINKING WITH TESTER.O)
// DO NOT MODIFY THIS SECTION
// *********************************************************************
template int kthLargest<int>(const std::vector<int>&, size_t);
template float kthLargest<float>(const std::vector<float>&, size_t);
template void flashAttention<float>(const std::vector<float>&, const std::vector<float>&,
  const std::vector<float>&, std::vector<float>&,
  int, int, int, int, int, int, bool);
// clang-format on
