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

#ifndef MEDIAPIPE_FRAMEWORK_GLOBAL_CLEANUP_H_
#define MEDIAPIPE_FRAMEWORK_GLOBAL_CLEANUP_H_

#include <memory>
#include <vector>

#include "mediapipe/framework/model_lifecycle_manager.h"

namespace mediapipe {

// Global cleanup utility to ensure all MediaPipe resources are properly
// cleaned up to prevent memory leaks, especially important when creating
// and destroying multiple models.
class GlobalCleanupManager {
 public:
  // Get the singleton instance
  static GlobalCleanupManager& GetInstance();
  
  // Register a lifecycle manager for global cleanup
  void RegisterLifecycleManager(std::shared_ptr<ModelLifecycleManager> manager);
  
  // Cleanup all registered lifecycle managers
  void CleanupAllModels();
  
  // Cleanup specific model by ID
  void CleanupModel(const std::string& model_id);
  
  // Get the number of active models
  size_t GetActiveModelCount() const;
  
  // Force cleanup of all global singletons (use with caution)
  void ForceGlobalCleanup();
  
  // Register an at-exit cleanup function (called automatically)
  static void RegisterAtExitCleanup();

 private:
  GlobalCleanupManager() = default;
  ~GlobalCleanupManager();
  
  std::vector<std::weak_ptr<ModelLifecycleManager>> lifecycle_managers_;
  mutable absl::Mutex mutex_;
  
  // At-exit cleanup function
  static void AtExitCleanup();
};

// RAII helper to ensure cleanup on scope exit
class ScopedGlobalCleanup {
 public:
  ScopedGlobalCleanup() {
    GlobalCleanupManager::RegisterAtExitCleanup();
  }
  
  ~ScopedGlobalCleanup() {
    GlobalCleanupManager::GetInstance().CleanupAllModels();
  }
};

// Utility functions
void CleanupAllMediaPipeModels();
void ForceCleanupMediaPipeGlobals();

}  // namespace mediapipe

#endif  // MEDIAPIPE_FRAMEWORK_GLOBAL_CLEANUP_H_
