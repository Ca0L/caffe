#include <vector>

#include "caffe/layers/conv_layer.hpp"

template <typename Dtype>
void kmeans_cluster(vector<int> &label, vector<Dtype> &centroid, Dtype *weight, int n_weight, vector<int> &mask, int n_cluster, int max_iter)
{
  Dtype max_weight = numeric_limits<Dtype>::lowest();
  Dtype min_weight = numeric_limits<Dtype>::max();
  for (int i = 0; i < n_weight; ++i)
  {
    if (mask[i])
    {
      if (weight[i] > max_weight)
        max_weight = weight[i];
      if (weight[i] < min_weight)
        min_weight = weight[i];
    }
  }

  // linearly initialize centroids
  for (int i = 0; i < n_cluster; ++i)
    centroid[i] = min_weight + (max_weight - min_weight) * i / (n_cluster - 1);
  
  fill(label.begin(), label.begin() + n_weight, -1);

  vector<Dtype> cluster_dist(n_weight);
  vector<Dtype> cluster_sum(n_cluster, 0);
  vector<int> cluster_sz(n_cluster, 0);

  int iter = 0;
  double pre_dist = numeric_limits<double>::max();
  double cur_dist = 0.0;

  while (iter < max_iter)
  {
    if (fabs(pre_dist - cur_dist) / pre_dist < 0.01) 
      break;
    pre_dist = cur_dist;
    cur_dist = 0.0;

    // find closest cluster for each weight
    for (int i = 0; i < n_weight; ++i)
    {
      if (mask[i])
      {
        Dtype dist;
        Dtype min_dist = numeric_limits<Dtype>::max();
        int closest = -1;
        for (int j = 0; j < n_cluster; ++j)
        {
          dist = fabs(weight[i] - centroid[i]);
          if (dist < min_dist)
          {
            min_dist = dist;
            closest = j;
          }
        }
        label[i] = closest;
        cluster_dist[i] = min_dist;
      }
    }

    // calc new dist
    for (int i = 0; i < n_weight; ++i)
    {
      if (mask[i])
        cur_dist += cluster_dist[i];
    }

    // gen new centroids
    for (int i = 0; i < n_cluster; ++i)
    {
      cluster_sum[i] = 0;
      cluster_sz[i] = 0;
    }

    for (int i = 0; i < n_weight; ++i)
    {
      if (mask[i])
      {
        cluster_sum[label[i]] += weight[i];
        cluster_sz[label[i]] += 1;
      }
    }

    for (int i = 0; i < n_cluster; ++i)
    {
      centroid[i] = cluster_sum[i] / cluster_sz[i];
    }

    ++iter;
  }
}

namespace caffe {

template <typename Dtype>
void ConvolutionLayer<Dtype>::computeBlobMask(float ratio)
{
  LOG(INFO) << "conv compute blob mask" << endl;
  int count = this->blobs_[0]->count();
  this->mask_.resize(count);

  this->indices_.resize(count);
  this->centroids_.resize(CONV_QUNUM);

  const Dtype *weight = this->blobs_[0]->cpu_data();
  vector<Dtype> sorted_weight(count);

  transform(weight, weight + count, sorted_weight.begin(), fabs);
  sort(sorted_weight.begin(), sorted_weight.end());

  int index = int(count * ratio);
  Dtype *mu_weight = this->blobs_[0]->mutable_cpu_data();
  int rat = 0;

  if (index > 0)
  {
    Dtype thr = sorted_weight[index - 1];
    LOG(INFO) << "CONV THR: " << thr << " " << ratio << endl;

    for (int i = 0; i < count; ++i)
    {
      this->mask_[i] = (weight[i] < -thr || weight[i] >= thr ? 1 : 0);
      mu_weight[i] *= this->mask_[i];
      rat += (1 - this->mask_[i]);
    }
  }
  else
  {
    for (int i = 0; i < count; ++i)
    {
      this->mask_[i] = (weight[i] != 0.f ? 1 : 0);
      rat += (1 - this->mask_[i]);
    }
  }

  LOG(INFO) << "sparsity: " << 1.f * rat / count << endl;
  int n_centroid = CONV_QUNUM;
  kmeans_cluster(this->indices_, this->centroids_, mu_weight, count, this->masks_, n_centroid, 1000);
}

template <typename Dtype>
void ConvolutionLayer<Dtype>::compute_output_shape() {
  const int* kernel_shape_data = this->kernel_shape_.cpu_data();
  const int* stride_data = this->stride_.cpu_data();
  const int* pad_data = this->pad_.cpu_data();
  const int* dilation_data = this->dilation_.cpu_data();
  this->output_shape_.clear();
  for (int i = 0; i < this->num_spatial_axes_; ++i) {
    // i + 1 to skip channel axis
    const int input_dim = this->input_shape(i + 1);
    const int kernel_extent = dilation_data[i] * (kernel_shape_data[i] - 1) + 1;
    const int output_dim = (input_dim + 2 * pad_data[i] - kernel_extent)
        / stride_data[i] + 1;
    this->output_shape_.push_back(output_dim);
  }
}

template <typename Dtype>
void ConvolutionLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  Dtype *mu_weight = this->blobs_[0]->mutable_cpu_data();
  int count = this->blobs_[0]->count();
  for (int i = 0; i < count; ++i)
  {
    if (this->masks_[i])
      mu_weight = this->centroids_[this->indices_[i]];
  }
  const Dtype* weight = this->blobs_[0]->cpu_data();
  for (int i = 0; i < bottom.size(); ++i) {
    const Dtype* bottom_data = bottom[i]->cpu_data();
    Dtype* top_data = top[i]->mutable_cpu_data();
    for (int n = 0; n < this->num_; ++n) {
      this->forward_cpu_gemm(bottom_data + n * this->bottom_dim_, weight,
          top_data + n * this->top_dim_);
      if (this->bias_term_) {
        const Dtype* bias = this->blobs_[1]->cpu_data();
        this->forward_cpu_bias(top_data + n * this->top_dim_, bias);
      }
    }
  }
}

template <typename Dtype>
void ConvolutionLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const Dtype* weight = this->blobs_[0]->cpu_data();
  Dtype* weight_diff = this->blobs_[0]->mutable_cpu_diff();
  for (int i = 0; i < top.size(); ++i) {
    const Dtype* top_diff = top[i]->cpu_diff();
    const Dtype* bottom_data = bottom[i]->cpu_data();
    Dtype* bottom_diff = bottom[i]->mutable_cpu_diff();
    // Bias gradient, if necessary.
    if (this->bias_term_ && this->param_propagate_down_[1]) {
      Dtype* bias_diff = this->blobs_[1]->mutable_cpu_diff();
      for (int n = 0; n < this->num_; ++n) {
        this->backward_cpu_bias(bias_diff, top_diff + n * this->top_dim_);
      }
    }
    if (this->param_propagate_down_[0] || propagate_down[i]) {
      for (int n = 0; n < this->num_; ++n) {
        // gradient w.r.t. weight. Note that we will accumulate diffs.
        if (this->param_propagate_down_[0]) {
          this->weight_cpu_gemm(bottom_data + n * this->bottom_dim_,
              top_diff + n * this->top_dim_, weight_diff);
        }
        Dtype *weight_diff = this->blobs_[0]->mutable_cpu_diff();
        for (int j = 0; j < count; ++j)
        {
          weight_diff[j] *= this->masks_[j];
        }
        vector<Dtype> tmp_diff(CONV_QUNUM);
        vector<int> freq(CONV_QUNUM);
        for (int j = 0; j < count; ++j)
        {
          if (this->masks_[j])
          {
            tmp_diff[this->indices_[j]] += weight_diff[j];
            freq[this->indices_[j]]++;
          }
        }
        for (int j = 0; j < count; ++j)
        {
          if (this->masks_[j])
            weight_diff[j] = tmp_diff[this->indices_[j]] / freq[this->indices_[j]];
        }
        // gradient w.r.t. bottom data, if necessary.
        if (propagate_down[i]) {
          this->backward_cpu_gemm(top_diff + n * this->top_dim_, weight,
              bottom_diff + n * this->bottom_dim_);
        }
      }
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(ConvolutionLayer);
#endif

INSTANTIATE_CLASS(ConvolutionLayer);

}  // namespace caffe
