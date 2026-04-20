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

#ifndef MEDIAPIPE_FRAMEWORK_REFERENCE_COUNTED_REGISTRY_H_
#define MEDIAPIPE_FRAMEWORK_REFERENCE_COUNTED_REGISTRY_H_

#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>

#include "absl/strings/string_view.h"
#include "mediapipe/framework/deps/no_destructor.h"
#include "mediapipe/framework/deps/registration_token.h"

namespace mediapipe {

// A registry that tracks reference counts for registered entries.
// When the reference count reaches zero, the entry is automatically removed.
// This helps prevent memory leaks from accumulated registry entries.
template <typename ReturnType, typename... Args>
class ReferenceCountedRegistry {
 public:
  using Function = std::function<ReturnType(Args...)>;
  
  struct RegistryEntry {
    Function function;
    int ref_count = 0;
    
    RegistryEntry() = default;
    RegistryEntry(Function func) : function(std::move(func)), ref_count(1) {}
  };
  
  // Register a function with automatic reference counting
  RegistrationToken Register(absl::string_view name, Function func) {
    std::string name_str(name);
    
    {
      absl::MutexLock lock(&mutex_);
      auto& entry = entries_[name_str];
      if (entry.ref_count == 0) {
        entry.function = std::move(func);
      }
      entry.ref_count++;
    }
    
    // Return a token that decrements the reference count when destroyed
    return RegistrationToken([this, name_str]() {
      this->DecrementRefCount(name_str);
    });
  }
  
  // Get a registered function
  ReturnType Invoke(absl::string_view name, Args... args) {
    std::string name_str(name);
    absl::MutexLock lock(&mutex_);
    
    auto it = entries_.find(name_str);
    if (it != entries_.end() && it->second.ref_count > 0) {
      return it->second.function(std::forward<Args>(args)...);
    }
    
    return ReturnType{};  // Default constructed return value
  }
  
  // Check if a function is registered and has references
  bool IsRegistered(absl::string_view name) const {
    std::string name_str(name);
    absl::MutexLock lock(&mutex_);
    
    auto it = entries_.find(name_str);
    return it != entries_.end() && it->second.ref_count > 0;
  }
  
  // Get the current reference count for a registered function
  int GetRefCount(absl::string_view name) const {
    std::string name_str(name);
    absl::MutexLock lock(&mutex_);
    
    auto it = entries_.find(name_str);
    return (it != entries_.end()) ? it->second.ref_count : 0;
  }
  
  // Get number of registered entries with non-zero reference counts
  size_t Size() const {
    absl::MutexLock lock(&mutex_);
    size_t count = 0;
    for (const auto& pair : entries_) {
      if (pair.second.ref_count > 0) {
        count++;
      }
    }
    return count;
  }
  
  // Clear all entries (for testing/cleanup)
  void Clear() {
    absl::MutexLock lock(&mutex_);
    entries_.clear();
  }
  
 private:
  void DecrementRefCount(const std::string& name) {
    absl::MutexLock lock(&mutex_);
    
    auto it = entries_.find(name);
    if (it != entries_.end()) {
      it->second.ref_count--;
      if (it->second.ref_count <= 0) {
        entries_.erase(it);
      }
    }
  }
  
  mutable absl::Mutex mutex_;
  std::unordered_map<std::string, RegistryEntry> entries_ ABSL_GUARDED_BY(mutex_);
};

// Template specialization for creating global reference-counted registries
template <typename ReturnType, typename... Args>
class GlobalReferenceCountedRegistry {
 public:
  using Registry = ReferenceCountedRegistry<ReturnType, Args...>;
  
  static RegistrationToken Register(absl::string_view name, 
                                    typename Registry::Function func) {
    return GetRegistry()->Register(name, std::move(func));
  }
  
  static ReturnType Invoke(absl::string_view name, Args... args) {
    return GetRegistry()->Invoke(name, std::forward<Args>(args)...);
  }
  
  static bool IsRegistered(absl::string_view name) {
    return GetRegistry()->IsRegistered(name);
  }
  
  static Registry* GetRegistry() {
    static NoDestructor<Registry> registry;
    return registry.get();
  }
};

}  // namespace mediapipe

#endif  // MEDIAPIPE_FRAMEWORK_REFERENCE_COUNTED_REGISTRY_H_
