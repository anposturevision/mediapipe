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

#ifndef MEDIAPIPE_FRAMEWORK_DEPS_SCOPED_OPTIONS_REGISTRY_H_
#define MEDIAPIPE_FRAMEWORK_DEPS_SCOPED_OPTIONS_REGISTRY_H_

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/synchronization/mutex.h"
#include "mediapipe/framework/deps/registration_token.h"
#include "mediapipe/framework/deps/model_lifecycle_manager.h"
#include "mediapipe/framework/tool/field_data.pb.h"

namespace mediapipe {

// Forward declarations from tool namespace
namespace tool {
class Descriptor;
class FieldDescriptor;
}

// A scoped version of OptionsRegistry that tracks registrations and can clean them up
// to prevent memory leaks when models are created and destroyed multiple times.
class ScopedOptionsRegistry {
 public:
  explicit ScopedOptionsRegistry(const std::string& model_id);
  ~ScopedOptionsRegistry();
  
  // Register protobuf descriptors with automatic cleanup tracking
  RegistrationToken Register(const mediapipe::FieldData& file_descriptor_set);
  
  // Find descriptor (delegates to global registry for compatibility)
  static const tool::Descriptor* GetProtobufDescriptor(const std::string& type_name);
  
  // Find all extensions (delegates to global registry for compatibility) 
  static void FindAllExtensions(absl::string_view extendee,
                                std::vector<const tool::FieldDescriptor*>* result);
  
  // Register this registry with a lifecycle manager for automatic cleanup
  void RegisterWithLifecycleManager(ModelLifecycleManager* manager);
  
  // Get the model ID
  const std::string& GetModelId() const { return model_id_; }
  
  // Manual cleanup (called automatically in destructor)
  void Cleanup();

 private:
  std::string model_id_;
  std::vector<RegistrationToken> registration_tokens_;
  std::vector<std::string> registered_type_names_;
  
  mutable absl::Mutex mutex_;
  bool cleaned_up_ ABSL_GUARDED_BY(mutex_) = false;
};

// Enhanced global options registry that supports reference counting
class EnhancedOptionsRegistry {
 public:
  struct RegistryEntry {
    std::shared_ptr<tool::Descriptor> descriptor;
    int ref_count = 0;
  };
  
  // Register with reference counting 
  static RegistrationToken RegisterWithRefCounting(const mediapipe::FieldData& file_descriptor_set,
                                                    const std::string& model_id);
  
  // Get current reference count for a type
  static int GetRefCount(const std::string& type_name);
  
  // Get total number of registered types
  static size_t GetRegisteredTypeCount();
  
 private:
  static absl::Mutex& GetMutex();
  static absl::flat_hash_map<std::string, RegistryEntry>& GetRefCountedDescriptors();
  
  static void DecrementRefCount(const std::string& type_name);
};

// Factory function for creating scoped options registry
std::unique_ptr<ScopedOptionsRegistry> CreateScopedOptionsRegistry(
    const std::string& model_id);

}  // namespace mediapipe

#endif  // MEDIAPIPE_FRAMEWORK_DEPS_SCOPED_OPTIONS_REGISTRY_H_
