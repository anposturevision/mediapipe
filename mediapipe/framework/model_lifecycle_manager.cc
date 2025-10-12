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

#include "mediapipe/framework/model_lifecycle_manager.h"

#include <memory>
#include <utility>

#include "absl/log/absl_log.h"
#include "absl/synchronization/mutex.h"

namespace mediapipe {

// Static member definitions
absl::Mutex ModelLifecycleManager::global_mutex_;
std::unordered_map<ModelLifecycleManager::ModelId, 
                   std::weak_ptr<ModelLifecycleManager>> 
    ModelLifecycleManager::global_model_managers_;

ModelLifecycleManager::ModelLifecycleManager(const ModelId& model_id)
    : model_id_(model_id) {
  ABSL_LOG(INFO) << "Created ModelLifecycleManager for model: " << model_id_;
}

ModelLifecycleManager::~ModelLifecycleManager() {
  Cleanup();
}

void ModelLifecycleManager::RegisterCleanupFunction(CleanupFunction cleanup) {
  absl::MutexLock lock(&mutex_);
  if (!cleaned_up_) {
    cleanup_functions_.push_back(std::move(cleanup));
  }
}

void ModelLifecycleManager::RegisterToken(RegistrationToken token) {
  absl::MutexLock lock(&mutex_);
  if (!cleaned_up_) {
    registration_tokens_.push_back(std::move(token));
  }
}

void ModelLifecycleManager::Cleanup() {
  absl::MutexLock lock(&mutex_);
  if (cleaned_up_) {
    return;
  }
  
  ABSL_LOG(INFO) << "Cleaning up model resources for: " << model_id_;
  
  // Clear registration tokens first (triggers unregistration)
  registration_tokens_.clear();
  
  // Execute cleanup functions
  for (auto& cleanup : cleanup_functions_) {
    try {
      cleanup();
    } catch (const std::exception& e) {
      ABSL_LOG(ERROR) << "Error during cleanup: " << e.what();
    }
  }
  cleanup_functions_.clear();
  
  // Clear managed caches
  managed_caches_.clear();
  
  cleaned_up_ = true;
}

ModelLifecycleManager& ModelLifecycleManager::GetGlobalManager() {
  static ModelLifecycleManager* global_manager = 
      new ModelLifecycleManager("__global__");
  return *global_manager;
}

void ModelLifecycleManager::RegisterModelInstance(
    const ModelId& model_id, 
    std::shared_ptr<ModelLifecycleManager> manager) {
  absl::MutexLock lock(&global_mutex_);
  global_model_managers_[model_id] = manager;
}

void ModelLifecycleManager::UnregisterModelInstance(const ModelId& model_id) {
  absl::MutexLock lock(&global_mutex_);
  global_model_managers_.erase(model_id);
}

void ModelLifecycleManager::CleanupAllModels() {
  std::vector<std::shared_ptr<ModelLifecycleManager>> managers_to_cleanup;
  
  {
    absl::MutexLock lock(&global_mutex_);
    for (auto it = global_model_managers_.begin(); 
         it != global_model_managers_.end();) {
      if (auto manager = it->second.lock()) {
        managers_to_cleanup.push_back(manager);
        ++it;
      } else {
        it = global_model_managers_.erase(it);
      }
    }
  }
  
  // Cleanup outside the lock to avoid deadlocks
  for (auto& manager : managers_to_cleanup) {
    manager->Cleanup();
  }
}

}  // namespace mediapipe
