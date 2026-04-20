/* Copyright 2024 The MediaPipe Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "mediapipe/framework/deps/scoped_model_resources.h"

#include <memory>
#include <utility>

#include "absl/log/absl_log.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"

// Only include tasks headers if available
#ifdef MEDIAPIPE_TASKS_AVAILABLE
#include "mediapipe/tasks/cc/core/model_resources_cache.h"
#include "mediapipe/tasks/cc/core/model_resources.h"
#include "mediapipe/tasks/cc/core/model_asset_bundle_resources.h"
#include "mediapipe/tasks/cc/core/base_options.h"
#endif

namespace mediapipe {

ScopedModelResourcesCache::ScopedModelResourcesCache(const std::string& model_id)
    : model_id_(model_id) {
#ifdef MEDIAPIPE_TASKS_AVAILABLE
  // Create the underlying cache with a builtin op resolver
  cache_ = absl::make_unique<tasks::core::ModelResourcesCache>(
      absl::make_unique<tasks::core::MediaPipeBuiltinOpResolver>());
#endif
  ABSL_LOG(INFO) << "Created ScopedModelResourcesCache for model: " << model_id_;
}

ScopedModelResourcesCache::~ScopedModelResourcesCache() {
  Cleanup();
}

#ifdef MEDIAPIPE_TASKS_AVAILABLE
absl::Status ScopedModelResourcesCache::AddModelResources(
    std::unique_ptr<tasks::core::ModelResources> model_resources) {
  if (!cache_) {
    return absl::InternalError("Cache not initialized");
  }
  
  const std::string& tag = model_resources->GetTag();
  auto status = cache_->AddModelResources(std::move(model_resources));
  
  if (status.ok()) {
    registered_resource_tags_.push_back(tag);
  }
  
  return status;
}

absl::Status ScopedModelResourcesCache::AddModelAssetBundleResources(
    std::unique_ptr<tasks::core::ModelAssetBundleResources> model_asset_bundle_resources) {
  if (!cache_) {
    return absl::InternalError("Cache not initialized");
  }
  
  const std::string& tag = model_asset_bundle_resources->GetTag();
  auto status = cache_->AddModelAssetBundleResources(std::move(model_asset_bundle_resources));
  
  if (status.ok()) {
    registered_bundle_tags_.push_back(tag);
  }
  
}

absl::StatusOr<const tasks::core::ModelResources*> 
ScopedModelResourcesCache::GetModelResources(const std::string& tag) const {
  if (!cache_) {
    return absl::InternalError("Cache not initialized");
  }
  return cache_->GetModelResources(tag);
}

absl::StatusOr<const tasks::core::ModelAssetBundleResources*> 
ScopedModelResourcesCache::GetModelAssetBundleResources(const std::string& tag) const {
  if (!cache_) {
    return absl::InternalError("Cache not initialized");
  }
  return cache_->GetModelAssetBundleResources(tag);
}
#endif

// These methods are available in both cases
bool ScopedModelResourcesCache::Exists(const std::string& tag) const {
#ifdef MEDIAPIPE_TASKS_AVAILABLE
  return cache_ && cache_->Exists(tag);
#else
  return false;
#endif
}

bool ScopedModelResourcesCache::ModelAssetBundleExists(const std::string& tag) const {
#ifdef MEDIAPIPE_TASKS_AVAILABLE
  return cache_ && cache_->ModelAssetBundleExists(tag);
#else
  return false;
#endif
}

void ScopedModelResourcesCache::RegisterWithLifecycleManager(
    ModelLifecycleManager* manager) {
  if (manager) {
    // Only register cleanup function, don't pass the cache since we own it
    manager->RegisterCleanupFunction([this]() { this->Cleanup(); });
  }
}

void ScopedModelResourcesCache::Cleanup() {
#ifdef MEDIAPIPE_TASKS_AVAILABLE
  if (!cache_) {
    return;
  }
  
  ABSL_LOG(INFO) << "Cleaning up ScopedModelResourcesCache for model: " << model_id_;
  
  // Clear all registration tokens first
  registration_tokens_.clear();
  
  // The underlying cache will be automatically cleaned up when the unique_ptr is reset
  registered_resource_tags_.clear();
  registered_bundle_tags_.clear();
  
  cache_.reset();
#else
  ABSL_LOG(INFO) << "Cleaning up ScopedModelResourcesCache for model: " << model_id_;
  
  // Clear all registration tokens and tags
  registration_tokens_.clear();
  registered_resource_tags_.clear();
  registered_bundle_tags_.clear();
  
  cache_ = nullptr;
#endif
}

std::unique_ptr<ScopedModelResourcesCache> CreateScopedModelResourcesCache(
    const std::string& model_id) {
  return absl::make_unique<ScopedModelResourcesCache>(model_id);
}

}  // namespace mediapipe
