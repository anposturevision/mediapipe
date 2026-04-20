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

#include "mediapipe/framework/tool/scoped_options_registry.h"

#include <memory>
#include <utility>

#include "absl/log/absl_log.h"
#include "absl/memory/memory.h"
#include "absl/synchronization/mutex.h"
#include "mediapipe/framework/tool/options_registry.h"

namespace mediapipe {
namespace tool {

// Static members for EnhancedOptionsRegistry
absl::Mutex& EnhancedOptionsRegistry::GetMutex() {
  static absl::Mutex* mutex = new absl::Mutex;
  return *mutex;
}

absl::flat_hash_map<std::string, EnhancedOptionsRegistry::RegistryEntry>& 
EnhancedOptionsRegistry::GetRefCountedDescriptors() {
  static absl::flat_hash_map<std::string, RegistryEntry>* descriptors = 
      new absl::flat_hash_map<std::string, RegistryEntry>;
  return *descriptors;
}

ScopedOptionsRegistry::ScopedOptionsRegistry(const std::string& model_id)
    : model_id_(model_id) {
  ABSL_LOG(INFO) << "Created ScopedOptionsRegistry for model: " << model_id_;
}

ScopedOptionsRegistry::~ScopedOptionsRegistry() {
  Cleanup();
}

RegistrationToken ScopedOptionsRegistry::Register(const FieldData& file_descriptor_set) {
  absl::MutexLock lock(&mutex_);
  if (cleaned_up_) {
    return RegistrationToken([]() {});
  }
  
  // Use the enhanced registry with reference counting
  auto token = EnhancedOptionsRegistry::RegisterWithRefCounting(
      file_descriptor_set, model_id_);
  
  registration_tokens_.push_back(std::move(token));
  
  return RegistrationToken([this]() {
    // Token cleanup is handled by the stored tokens in registration_tokens_
  });
}

const Descriptor* ScopedOptionsRegistry::GetProtobufDescriptor(const std::string& type_name) {
  // Delegate to the global registry for compatibility
  return OptionsRegistry::GetProtobufDescriptor(type_name);
}

void ScopedOptionsRegistry::FindAllExtensions(
    absl::string_view extendee,
    std::vector<const FieldDescriptor*>* result) {
  // Delegate to the global registry for compatibility  
  OptionsRegistry::FindAllExtensions(extendee, result);
}

void ScopedOptionsRegistry::RegisterWithLifecycleManager(ModelLifecycleManager* manager) {
  if (manager) {
    manager->RegisterCleanupFunction([this]() { this->Cleanup(); });
  }
}

void ScopedOptionsRegistry::Cleanup() {
  absl::MutexLock lock(&mutex_);
  if (cleaned_up_) {
    return;
  }
  
  ABSL_LOG(INFO) << "Cleaning up ScopedOptionsRegistry for model: " << model_id_;
  
  // Clear all registration tokens to trigger unregistration
  registration_tokens_.clear();
  registered_type_names_.clear();
  
  cleaned_up_ = true;
}

RegistrationToken EnhancedOptionsRegistry::RegisterWithRefCounting(
    const FieldData& file_descriptor_set,
    const std::string& model_id) {
  
  // For now, delegate to the original registry and return a cleanup token
  // In a full implementation, this would implement reference counting
  auto token = OptionsRegistry::Register(file_descriptor_set);
  
  return RegistrationToken([token = std::move(token)]() mutable {
    // The token will automatically clean up when destroyed
  });
}

int EnhancedOptionsRegistry::GetRefCount(const std::string& type_name) {
  absl::MutexLock lock(&GetMutex());
  auto& descriptors = GetRefCountedDescriptors();
  auto it = descriptors.find(type_name);
  return (it != descriptors.end()) ? it->second.ref_count : 0;
}

size_t EnhancedOptionsRegistry::GetRegisteredTypeCount() {
  absl::MutexLock lock(&GetMutex());
  auto& descriptors = GetRefCountedDescriptors();
  size_t count = 0;
  for (const auto& pair : descriptors) {
    if (pair.second.ref_count > 0) {
      count++;
    }
  }
  return count;
}

void EnhancedOptionsRegistry::DecrementRefCount(const std::string& type_name) {
  absl::MutexLock lock(&GetMutex());
  auto& descriptors = GetRefCountedDescriptors();
  auto it = descriptors.find(type_name);
  if (it != descriptors.end()) {
    it->second.ref_count--;
    if (it->second.ref_count <= 0) {
      descriptors.erase(it);
    }
  }
}

std::unique_ptr<ScopedOptionsRegistry> CreateScopedOptionsRegistry(
    const std::string& model_id) {
  return absl::make_unique<ScopedOptionsRegistry>(model_id);
}

}  // namespace tool
}  // namespace mediapipe
