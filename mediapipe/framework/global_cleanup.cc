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

#include "mediapipe/framework/global_cleanup.h"

#include <algorithm>
#include <cstdlib>
#include <memory>

#include "absl/log/absl_log.h"
#include "absl/synchronization/mutex.h"

namespace mediapipe {

// Static member definitions
absl::Mutex GlobalCleanupManager::mutex_;

GlobalCleanupManager& GlobalCleanupManager::GetInstance() {
  static GlobalCleanupManager* instance = new GlobalCleanupManager();
  return *instance;
}

GlobalCleanupManager::~GlobalCleanupManager() {
  CleanupAllModels();
}

void GlobalCleanupManager::RegisterLifecycleManager(
    std::shared_ptr<ModelLifecycleManager> manager) {
  if (!manager) {
    return;
  }
  
  absl::MutexLock lock(&mutex_);
  
  // Clean up expired weak_ptrs
  lifecycle_managers_.erase(
      std::remove_if(lifecycle_managers_.begin(), lifecycle_managers_.end(),
                     [](const std::weak_ptr<ModelLifecycleManager>& ptr) {
                       return ptr.expired();
                     }),
      lifecycle_managers_.end());
  
  lifecycle_managers_.push_back(manager);
}

void GlobalCleanupManager::CleanupAllModels() {
  std::vector<std::shared_ptr<ModelLifecycleManager>> managers_to_cleanup;
  
  {
    absl::MutexLock lock(&mutex_);
    for (auto it = lifecycle_managers_.begin(); it != lifecycle_managers_.end();) {
      if (auto manager = it->lock()) {
        managers_to_cleanup.push_back(manager);
        ++it;
      } else {
        it = lifecycle_managers_.erase(it);
      }
    }
  }
  
  ABSL_LOG(INFO) << "Cleaning up " << managers_to_cleanup.size() << " active models";
  
  // Cleanup outside the lock to avoid deadlocks
  for (auto& manager : managers_to_cleanup) {
    try {
      manager->Cleanup();
    } catch (const std::exception& e) {
      ABSL_LOG(ERROR) << "Error during model cleanup: " << e.what();
    }
  }
  
  // Clear the list after cleanup
  {
    absl::MutexLock lock(&mutex_);
    lifecycle_managers_.clear();
  }
}

void GlobalCleanupManager::CleanupModel(const std::string& model_id) {
  std::shared_ptr<ModelLifecycleManager> target_manager;
  
  {
    absl::MutexLock lock(&mutex_);
    for (auto it = lifecycle_managers_.begin(); it != lifecycle_managers_.end();) {
      if (auto manager = it->lock()) {
        if (manager->GetModelId() == model_id) {
          target_manager = manager;
          it = lifecycle_managers_.erase(it);
          break;
        } else {
          ++it;
        }
      } else {
        it = lifecycle_managers_.erase(it);
      }
    }
  }
  
  if (target_manager) {
    ABSL_LOG(INFO) << "Cleaning up model: " << model_id;
    target_manager->Cleanup();
  }
}

size_t GlobalCleanupManager::GetActiveModelCount() const {
  absl::MutexLock lock(&mutex_);
  size_t count = 0;
  for (const auto& weak_manager : lifecycle_managers_) {
    if (!weak_manager.expired()) {
      count++;
    }
  }
  return count;
}

void GlobalCleanupManager::ForceGlobalCleanup() {
  ABSL_LOG(WARNING) << "Forcing global MediaPipe cleanup - this may cause issues if MediaPipe is still in use";
  
  // Clean up all models first
  CleanupAllModels();
  
  // Clean up global model managers
  ModelLifecycleManager::CleanupAllModels();
  
  ABSL_LOG(INFO) << "Global MediaPipe cleanup completed";
}

void GlobalCleanupManager::RegisterAtExitCleanup() {
  static bool registered = false;
  if (!registered) {
    std::atexit(AtExitCleanup);
    registered = true;
  }
}

void GlobalCleanupManager::AtExitCleanup() {
  GetInstance().CleanupAllModels();
}

// Utility functions
void CleanupAllMediaPipeModels() {
  GlobalCleanupManager::GetInstance().CleanupAllModels();
}

void ForceCleanupMediaPipeGlobals() {
  GlobalCleanupManager::GetInstance().ForceGlobalCleanup();
}

}  // namespace mediapipe
