// Copyright 2025 The MediaPipe Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef MEDIAPIPE_FRAMEWORK_TOOL_OPTIONS_REGISTRY_CLEANUP_H_
#define MEDIAPIPE_FRAMEWORK_TOOL_OPTIONS_REGISTRY_CLEANUP_H_

#include <memory>
#include <vector>

#include "mediapipe/framework/deps/registration_token.h"

namespace mediapipe {
namespace tool {

// A utility class to manage cleanup of global registration tokens.
// This helps prevent memory leaks by providing a way to unregister 
// options that were registered during static initialization.
class OptionsRegistryCleanupManager {
 public:
  // Get the singleton instance
  static OptionsRegistryCleanupManager& GetInstance();
  
  // Register a token for cleanup
  void RegisterToken(RegistrationToken token);
  
  // Clean up all registered tokens (usually called at program exit)
  void CleanupAll();
  
  // Destructor automatically cleans up remaining tokens
  ~OptionsRegistryCleanupManager();
  
 private:
  OptionsRegistryCleanupManager() = default;
  std::vector<RegistrationToken> tokens_;
};

// RAII helper class for automatic cleanup of registration tokens
class ScopedRegistrationToken {
 public:
  explicit ScopedRegistrationToken(RegistrationToken token);
  ~ScopedRegistrationToken();
  
  ScopedRegistrationToken(const ScopedRegistrationToken&) = delete;
  ScopedRegistrationToken& operator=(const ScopedRegistrationToken&) = delete;
  
  ScopedRegistrationToken(ScopedRegistrationToken&& other) noexcept;
  ScopedRegistrationToken& operator=(ScopedRegistrationToken&& other) noexcept;
  
 private:
  RegistrationToken token_;
};

}  // namespace tool
}  // namespace mediapipe

#endif  // MEDIAPIPE_FRAMEWORK_TOOL_OPTIONS_REGISTRY_CLEANUP_H_
