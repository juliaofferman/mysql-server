// Copyright 2026 Google LLC

#include "scann_interface.h"

#include <memory>
#include <string>
#include <vector>
#include <typeinfo>
#include <cassert>

#include "scann/proto/scann.pb.h"
#include "scann/base/single_machine_base.h"
#include "scann/base/single_machine_factory_scann.h"
#include "scann/partitioning/partitioner_factory.h"
#include "scann/partitioning/partitioner_factory_base.h"
#include "scann/partitioning/kmeans_tree_partitioner.h"
#include "scann/partitioning/kmeans_tree_like_partitioner.h"
#include "scann/utils/fast_top_neighbors.h"
#include "scann/utils/types.h"
#include "scann/utils/threads.h"
#include "scann/data_format/dataset.h"
#include "scann/data_format/datapoint.h"
#include "scann/data_format/docid_collection.h"
#include "scann/distance_measures/one_to_one/dot_product.h"
#include "scann/utils/scalar_quantization_helpers.h"
#include "scann/utils/top_n_amortized_constant.h"

#include "google/protobuf/text_format.h"

namespace kmeans_wrapper {

class KMeansConfigWrapperImpl : public KMeansConfigWrapper {
 public:
  research_scann::ScannConfig config;
  KMeansConfigWrapperImpl() = default;
};

class DatapointWrapperImpl : public DatapointWrapper {
 public:
  bool is_float;
  std::vector<float> float_dp;
  std::vector<int8_t> int8_dp;

  explicit DatapointWrapperImpl(const std::vector<float> &values)
      : is_float(true), float_dp(values) {}
  DatapointWrapperImpl(const int8_t *val, size_t size) : is_float(false) {
    int8_dp.assign(val, val + size);
  }

  const float *float_values() const override {
    if (is_float) return float_dp.empty() ? nullptr : float_dp.data();
    return nullptr;
  }
  float *mutable_float_values() override {
    if (is_float) return float_dp.empty() ? nullptr : float_dp.data();
    return nullptr;
  }
  size_t dimensionality() const override {
    return is_float ? float_dp.size() : int8_dp.size();
  }

  research_scann::DatapointPtr<float> GetFloatPtr() {
    assert(is_float);
    return research_scann::DatapointPtr<float>(
        nullptr, float_dp.data(), float_dp.size(), float_dp.size());
  }

  research_scann::DatapointPtr<int8_t> GetInt8Ptr() {
    return research_scann::DatapointPtr<int8_t>(nullptr, int8_dp.data(),
                                                int8_dp.size(), int8_dp.size());
  }
};

class DatasetWrapperImpl : public DatasetWrapper {
 public:
  std::shared_ptr<research_scann::DenseDataset<float>> dataset;

  DatasetWrapperImpl() {
    dataset = std::make_shared<research_scann::DenseDataset<float>>();
    dataset->set_docids_no_checks(
        std::make_shared<research_scann::VariableLengthDocidCollection>());
  }

  void set_dimensionality(int dim) override {
    dataset->set_dimensionality(dim);
  }

  size_t size() const override { return dataset->size(); }

  bool Append(DatapointWrapper *dp) override {
    auto *float_dp = static_cast<DatapointWrapperImpl *>(dp);
    if (float_dp && float_dp->is_float) {
      auto status = dataset->Append(float_dp->GetFloatPtr(), "");
      return status.ok();
    }
    return false;
  }

  bool NormalizeUnitL2() override {
    auto status = dataset->NormalizeUnitL2();
    return status.ok();
  }
};

class PartitionerWrapperImpl : public PartitionerWrapper {
 public:
  std::unique_ptr<research_scann::Partitioner<float>> partitioner;

  explicit PartitionerWrapperImpl(
      std::unique_ptr<research_scann::Partitioner<float>> p)
      : partitioner(std::move(p)) {}

  std::unique_ptr<PartitionerWrapper> Clone() const override {
    if (!partitioner) return std::make_unique<PartitionerWrapperImpl>(nullptr);
    auto cloned = partitioner->Clone();
    cloned->set_tokenization_mode(partitioner->tokenization_mode());
    return std::make_unique<PartitionerWrapperImpl>(std::move(cloned));
  }

  void set_tokenization_mode(TokenizationMode mode) override {
    if (!partitioner) return;
    partitioner->set_tokenization_mode(
        mode == TokenizationMode::DATABASE
            ? research_scann::UntypedPartitioner::DATABASE
            : research_scann::UntypedPartitioner::QUERY);
  }

  void tokenize(DatapointWrapper *dp, std::vector<int32_t> *parts,
                int32_t num_parts) override {
    // If called directly on base with num_parts, just fallback to standard
    tokenize(dp, parts);
  }

  void tokenize(DatapointWrapper *dp, std::vector<int32_t> *parts) override {
    auto *float_dp = static_cast<DatapointWrapperImpl *>(dp);
    if (!float_dp || !float_dp->is_float) return;
    research_scann::DatapointPtr<float> ptr = float_dp->GetFloatPtr();
    auto *raw_part = partitioner.get();
    if (!raw_part) return;

    auto status = partitioner->TokensForDatapointWithSpilling(ptr, parts);
    assert(status.ok());
  }

  void SerializeToString(std::string *out) const override {
    research_scann::SerializedPartitioner proto;
    partitioner->CopyToProto(&proto);
    proto.SerializeToString(out);
  }

  int32_t query_spilling_max_centers() const override {
    auto *raw_part = partitioner.get();
    if (!raw_part) return 0;
    auto *kmeans_part =
        dynamic_cast<const research_scann::KMeansTreeLikePartitioner<float> *>(
            raw_part);
    if (kmeans_part) {
      return kmeans_part->query_spilling_max_centers();
    }
    return 0;
  }
};

class KMeansTreeLikePartitionerWrapperImpl : public KMeansTreeLikePartitionerWrapper {
 public:
  std::unique_ptr<research_scann::Partitioner<float>> partitioner;

  explicit KMeansTreeLikePartitionerWrapperImpl(
      std::unique_ptr<research_scann::Partitioner<float>> p)
      : partitioner(std::move(p)) {}

  std::unique_ptr<PartitionerWrapper> Clone() const override {
    if (!partitioner)
      return std::make_unique<KMeansTreeLikePartitionerWrapperImpl>(nullptr);
    auto cloned = partitioner->Clone();
    cloned->set_tokenization_mode(partitioner->tokenization_mode());
    return std::make_unique<KMeansTreeLikePartitionerWrapperImpl>(
        std::move(cloned));
  }

  void set_tokenization_mode(TokenizationMode mode) override {
    if (!partitioner) return;
    partitioner->set_tokenization_mode(
        mode == TokenizationMode::DATABASE
            ? research_scann::UntypedPartitioner::DATABASE
            : research_scann::UntypedPartitioner::QUERY);
  }

  void tokenize(DatapointWrapper *dp, std::vector<int32_t> *parts,
                int32_t num_parts) override {
    // If called directly on base with num_parts, just fallback to standard
    tokenize(dp, parts);
  }

  void tokenize(DatapointWrapper *dp, std::vector<int32_t> *parts) override {
    auto *float_dp = static_cast<DatapointWrapperImpl *>(dp);
    if (!float_dp || !float_dp->is_float) return;
    research_scann::DatapointPtr<float> ptr = float_dp->GetFloatPtr();
    auto *raw_part = partitioner.get();
    if (!raw_part) return;

    auto status = partitioner->TokensForDatapointWithSpilling(ptr, parts);
    assert(status.ok());
  }

  void SerializeToString(std::string *out) const override {
    research_scann::SerializedPartitioner proto;
    partitioner->CopyToProto(&proto);
    proto.SerializeToString(out);
  }

  int32_t query_spilling_max_centers() const override {
    auto *raw_part = partitioner.get();
    if (!raw_part) return 0;
    auto *kmeans_part =
        dynamic_cast<const research_scann::KMeansTreeLikePartitioner<float> *>(
            raw_part);
    if (kmeans_part) {
      return kmeans_part->query_spilling_max_centers();
    }
    return 0;
  }

  void TokensForDatapointWithSpilling(
      DatapointWrapper *dp, int32_t num_results,
      std::vector<std::pair<uint32_t, float>> *results) override {
    auto *float_dp = static_cast<DatapointWrapperImpl *>(dp);
    if (!float_dp || !float_dp->is_float) return;
    research_scann::DatapointPtr<float> ptr = float_dp->GetFloatPtr();
    auto *raw_part = partitioner.get();
    if (!raw_part) return;

    auto *kmeans_part = dynamic_cast<
        const research_scann::KMeansTreeLikePartitioner<float> *>(raw_part);
    if (kmeans_part) {
      std::vector<std::pair<uint32_t, float>> pairs;
      auto status =
          kmeans_part->TokensForDatapointWithSpilling(ptr, num_results, &pairs);
      assert(status.ok());
      results->assign(pairs.begin(), pairs.end());
    }
  }
};

// Global functions
std::unique_ptr<KMeansConfigWrapper> ParseKMeansConfigBinary(
    const uint8_t *proto_bytes, size_t length) {
  auto config = std::make_unique<KMeansConfigWrapperImpl>();
  bool success = config->config.ParseFromArray(proto_bytes, length);
  if (!success) return nullptr;
  return config;
}

std::unique_ptr<PartitionerWrapper> CreatePartitioner(
    const uint8_t *partitioner_bytes, size_t length,
    KMeansConfigWrapper *config) {
  research_scann::SerializedPartitioner proto;
  proto.ParseFromArray(partitioner_bytes, length);
  auto *config_impl = dynamic_cast<KMeansConfigWrapperImpl *>(config);
  auto result = research_scann::PartitionerFromSerialized<float>(
      proto, config_impl->config.partitioning());
  if (result.ok()) {
    auto p = std::move(result.value());
    if (dynamic_cast<const research_scann::KMeansTreeLikePartitioner<float> *>(p.get())) {
      return std::make_unique<KMeansTreeLikePartitionerWrapperImpl>(std::move(p));
    }
    return std::make_unique<PartitionerWrapperImpl>(std::move(p));
  }
  return nullptr;
}

class ThreadPoolWrapperImpl : public ThreadPoolWrapper {
 public:
  std::shared_ptr<research_scann::ThreadPool> pool;
  ThreadPoolWrapperImpl(int num_threads) {
    pool.reset(research_scann::StartThreadPool("scann_train", num_threads).release());
  }
};

std::unique_ptr<ThreadPoolWrapper> CreateThreadPool(int num_threads) {
  return std::make_unique<ThreadPoolWrapperImpl>(num_threads);
}

std::unique_ptr<PartitionerWrapper> TrainPartitioner(DatasetWrapper *dataset,
                                                     KMeansConfigWrapper *config,
                                                     ThreadPoolWrapper* pool) {
  auto *dataset_impl = dynamic_cast<DatasetWrapperImpl *>(dataset);
  auto *config_impl = dynamic_cast<KMeansConfigWrapperImpl *>(config);
  research_scann::SingleMachineFactoryOptions opt;
  if (pool) {
    auto* pool_impl = dynamic_cast<ThreadPoolWrapperImpl*>(pool);
    opt.parallelization_pool = pool_impl->pool;
  }
  auto result = research_scann::PartitionerFactory<float>(
      dataset_impl->dataset.get(), config_impl->config.partitioning(),
      opt.parallelization_pool);
  if (result.ok()) {
    auto p = std::move(result.value());
    if (dynamic_cast<const research_scann::KMeansTreeLikePartitioner<float> *>(p.get())) {
      return std::make_unique<KMeansTreeLikePartitionerWrapperImpl>(std::move(p));
    }
    return std::make_unique<PartitionerWrapperImpl>(std::move(p));
  }
  return nullptr;
}

bool MaybeAddTopLevelPartitioner(PartitionerWrapper *part,
                                 const uint8_t *config_bytes, size_t length) {
  research_scann::Partitioner<float> *raw_partitioner = nullptr;
  auto *part_impl = dynamic_cast<PartitionerWrapperImpl *>(part);
  auto *kmeans_impl = dynamic_cast<KMeansTreeLikePartitionerWrapperImpl *>(part);

  if (part_impl) raw_partitioner = part_impl->partitioner.get();
  else if (kmeans_impl) raw_partitioner = kmeans_impl->partitioner.get();

  if (!raw_partitioner) return false;

  research_scann::PartitioningConfig part_config;
  if (!part_config.ParseFromArray(config_bytes, length)) return false;

  auto cloned = raw_partitioner->Clone();
  auto status =
      research_scann::MaybeAddTopLevelPartitioner<float>(cloned, part_config);
  if (status.ok()) {
    if (part_impl) part_impl->partitioner = std::move(cloned);
    else if (kmeans_impl) kmeans_impl->partitioner = std::move(cloned);
    return true;
  }
  return false;
}

std::unique_ptr<DatasetWrapper> CreateDataset() {
  return std::make_unique<DatasetWrapperImpl>();
}

std::unique_ptr<DatapointWrapper> CreateDatapointFromPtr(const float *values,
                                                         size_t size) {
  std::vector<float> vec(values, values + size);
  return std::make_unique<DatapointWrapperImpl>(vec);
}

std::unique_ptr<DatapointWrapper> CreateQuantizedDatapoint(const int8_t *values,
                                                           size_t size) {
  return std::make_unique<DatapointWrapperImpl>(values, size);
}

float DenseDotProduct(DatapointWrapper *q, DatapointWrapper *dp) {
  auto *q_impl = static_cast<DatapointWrapperImpl *>(q);
  auto *dp_impl = static_cast<DatapointWrapperImpl *>(dp);
  if (q_impl && dp_impl) {
    if (q_impl->is_float && !dp_impl->is_float) {
      return research_scann::DenseDotProduct(q_impl->GetFloatPtr(),
                                             dp_impl->GetInt8Ptr());
    } else if (q_impl->is_float && dp_impl->is_float) {
      return research_scann::DenseDotProduct(q_impl->GetFloatPtr(),
                                             dp_impl->GetFloatPtr());
    }
  }
  return 0.0f;
}

float DenseDotProductRaw(DatapointWrapper *q, const int8_t *dp_values, size_t size) {
  auto *q_impl = static_cast<DatapointWrapperImpl *>(q);
  if (q_impl && q_impl->is_float) {
    research_scann::DatapointPtr<int8_t> dp_ptr(nullptr, dp_values, size, size);
    return research_scann::DenseDotProduct(q_impl->GetFloatPtr(), dp_ptr);
  }
  return 0.0f;
}

float SquaredL2Norm(DatapointWrapper *dp) {
  auto *dp_impl = static_cast<DatapointWrapperImpl *>(dp);
  if (dp_impl && dp_impl->is_float) {
    return research_scann::SquaredL2Norm(dp_impl->GetFloatPtr());
  }
  return 0.0f;
}

std::vector<float> ComputeMaxQuantizationMultipliers(DatasetWrapper *dataset) {
  auto *dataset_impl = dynamic_cast<DatasetWrapperImpl *>(dataset);
  return research_scann::ComputeMaxQuantizationMultipliers(
      *(dataset_impl->dataset));
}

void NormalizeUnitL2(DatapointWrapper *dp) {
  auto *casted = static_cast<DatapointWrapperImpl *>(dp);
  if (casted && casted->is_float && !casted->float_dp.empty()) {
    research_scann::Datapoint<float> scann_dp;
    scann_dp.mutable_values()->assign(casted->float_dp.begin(),
                                      casted->float_dp.end());
    research_scann::NormalizeUnitL2(&scann_dp);
    casted->float_dp.assign(scann_dp.values().begin(), scann_dp.values().end());
  }
}

void ScalarQuantizeFloatDatapoint(const float *src, size_t dim,
                                  const float *multipliers, int8_t *ret) {
  research_scann::DatapointPtr<float> p(nullptr, src, dim, dim);
  auto mul = absl::MakeConstSpan(multipliers, dim);
  auto ret_span = absl::MakeSpan(ret, dim);
  research_scann::ScalarQuantizeFloatDatapoint(p, mul, ret_span);
}

}  // namespace kmeans_wrapper
