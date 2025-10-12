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

#ifndef MEDIAPIPE_FRAMEWORK_DEPS_SCOPED_MODEL_RESOURCES_H_
#define MEDIAPIPE_FRAMEWORK_DEPS_SCOPED_MODEL_RESOURCES_H_

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "mediapipe/framework/deps/registration_token.h"
#include "mediapipe/framework/deps/model_lifecycle_manager.h"

namespace mediapipe {

#ifdef MEDIAPIPE_TASKS_AVAILABLE
// Forward declarations when tasks are available
namespace tasks {
namespace core {
class ModelResources;
class ModelAssetBundleResources; 
class ModelResourcesCache;
}
}
#endif

// A scoped version of ModelResourcesCache that automatically cleans up
// resources when destroyed, preventing memory leaks across model instances.
class ScopedModelResourcesCache {
 public:
  explicit ScopedModelResourcesCache(const std::string& model_id);
  ~ScopedModelResourcesCache();
  
#ifdef MEDIAPIPE_TASKS_AVAILABLE
  // Add model resources with automatic cleanup tracking
  absl::Status AddModelResources(
      std::unique_ptr<tasks::core::ModelResources> model_resources);
  
  // Add model asset bundle resources with automatic cleanup tracking  
  absl::Status AddModelAssetBundleResources(
      std::unique_ptr<tasks::core::ModelAssetBundleResources> model_asset_bundle_resources);
  
  // Get model resources (returns const pointer)
  absl::StatusOr<const tasks::core::ModelResources*> GetModelResources(
      const std::string& tag) const;
      
  // Get model asset bundle resources (returns const pointer)
  absl::StatusOr<const tasks::core::ModelAssetBundleResources*> GetModelAssetBundleResources(
      const std::string& tag) const;
  
  // Get the underlying cache (for compatibility with existing code)
  tasks::core::ModelResourcesCache* GetCache() const { return cache_.get(); }
#else
  // Stub implementations when tasks are not available
  absl::Status AddModelResources(void* model_resources) {
    return absl::UnimplementedError("MediaPipe Tasks not available");
  }
  
  absl::Status AddModelAssetBundleResources(void* model_asset_bundle_resources) {
    return absl::UnimplementedError("MediaPipe Tasks not available");
  }
  
  absl::StatusOr<void*> GetModelResources(const std::string& tag) const {
    return absl::UnimplementedError("MediaPipe Tasks not available");
  }
      
  absl::StatusOr<void*> GetModelAssetBundleResources(const std::string& tag) const {
    return absl::UnimplementedError("MediaPipe Tasks not available");
  }
  
  void* GetCache() const { return nullptr; }
#endif
  
  // Check if model resources exist
  bool Exists(const std::string& tag) const;
  
  // Check if model asset bundle resources exist  
  bool ModelAssetBundleExists(const std::string& tag) const;
  
  // Get the model ID
  const std::string& GetModelId() const { return model_id_; }
  
  // Register this cache with a lifecycle manager for automatic cleanup
  void RegisterWithLifecycleManager(ModelLifecycleManager* manager);

 private:
  std::string model_id_;
#ifdef MEDIAPIPE_TASKS_AVAILABLE
  std::unique_ptr<tasks::core::ModelResourcesCache> cache_;
#else
  void* cache_ = nullptr; // Placeholder when tasks not available
#endif
  std::vector<std::string> registered_resource_tags_;
  std::vector<std::string> registered_bundle_tags_;
  
  // Track registration tokens for cleanup
  std::vector<RegistrationToken> registration_tokens_;
  
  void Cleanup();
};

// Factory function for creating scoped model resources cache
std::unique_ptr<ScopedModelResourcesCache> CreateScopedModelResourcesCache(
    const std::string& model_id);

}  // namespace mediapipe

#endif  // MEDIAPIPE_FRAMEWORK_DEPS_SCOPED_MODEL_RESOURCES_H_
