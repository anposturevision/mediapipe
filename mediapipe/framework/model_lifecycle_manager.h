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

#ifndef MEDIAPIPE_FRAMEWORK_MODEL_LIFECYCLE_MANAGER_H_
#define MEDIAPIPE_FRAMEWORK_MODEL_LIFECYCLE_MANAGER_H_

#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/synchronization/mutex.h"
#include "mediapipe/framework/deps/registration_token.h"

namespace mediapipe {

// Manages the lifecycle of model-related global resources to prevent memory leaks
// when creating and destroying multiple models. This class tracks all global
// registrations and caches associated with a model instance and ensures they
// are properly cleaned up when the model is destroyed.
class ModelLifecycleManager {
 public:
  using CleanupFunction = std::function<void()>;
  using ModelId = std::string;

  // Creates a new model lifecycle manager for the given model ID
  explicit ModelLifecycleManager(const ModelId& model_id);
  
  // Destructor automatically cleans up all registered resources
  ~ModelLifecycleManager();

  // Registers a cleanup function to be called when the model is destroyed
  void RegisterCleanupFunction(CleanupFunction cleanup);
  
  // Registers a registration token for automatic cleanup
  void RegisterToken(RegistrationToken token);
  
  // Registers a model-scoped resource cache
  template<typename CacheType>
  void RegisterCache(std::shared_ptr<CacheType> cache);
  
  // Manually trigger cleanup (called automatically in destructor)
  void Cleanup();
  
  // Get the model ID
  const ModelId& GetModelId() const { return model_id_; }
  
  // Get singleton instance for global cleanup coordination
  static ModelLifecycleManager& GetGlobalManager();
  
  // Register a model instance with the global manager
  static void RegisterModelInstance(const ModelId& model_id, 
                                    std::shared_ptr<ModelLifecycleManager> manager);
  
  // Unregister a model instance from the global manager
  static void UnregisterModelInstance(const ModelId& model_id);
  
  // Cleanup all model instances
  static void CleanupAllModels();

 private:
  ModelId model_id_;
  std::vector<CleanupFunction> cleanup_functions_;
  std::vector<RegistrationToken> registration_tokens_;
  std::vector<std::shared_ptr<void>> managed_caches_;
  
  mutable absl::Mutex mutex_;
  bool cleaned_up_ ABSL_GUARDED_BY(mutex_) = false;
  
  // Global state for managing all model instances
  static absl::Mutex global_mutex_;
  static std::unordered_map<ModelId, std::weak_ptr<ModelLifecycleManager>> 
      global_model_managers_ ABSL_GUARDED_BY(global_mutex_);
};

// RAII wrapper for automatic model lifecycle management
class ScopedModelManager {
 public:
  explicit ScopedModelManager(const std::string& model_id)
      : manager_(std::make_shared<ModelLifecycleManager>(model_id)) {
    ModelLifecycleManager::RegisterModelInstance(model_id, manager_);
  }
  
  ~ScopedModelManager() {
    ModelLifecycleManager::UnregisterModelInstance(manager_->GetModelId());
  }
  
  ModelLifecycleManager* operator->() { return manager_.get(); }
  ModelLifecycleManager& operator*() { return *manager_; }
  
 private:
  std::shared_ptr<ModelLifecycleManager> manager_;
};

template<typename CacheType>
void ModelLifecycleManager::RegisterCache(std::shared_ptr<CacheType> cache) {
  absl::MutexLock lock(&mutex_);
  if (!cleaned_up_) {
    managed_caches_.push_back(std::static_pointer_cast<void>(cache));
  }
}

}  // namespace mediapipe

#endif  // MEDIAPIPE_FRAMEWORK_MODEL_LIFECYCLE_MANAGER_H_
