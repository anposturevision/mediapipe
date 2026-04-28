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

#include "mediapipe/framework/tool/options_registry_cleanup.h"

#include <utility>

#include "mediapipe/framework/deps/no_destructor.h"

namespace mediapipe {
namespace tool {

OptionsRegistryCleanupManager& OptionsRegistryCleanupManager::GetInstance() {
  // Use NoDestructor to avoid issues during static destruction
  static NoDestructor<OptionsRegistryCleanupManager> instance;
  return *instance;
}

void OptionsRegistryCleanupManager::RegisterToken(RegistrationToken token) {
  tokens_.push_back(std::move(token));
}

void OptionsRegistryCleanupManager::CleanupAll() {
  for (auto& token : tokens_) {
    token.Unregister();
  }
  tokens_.clear();
}

OptionsRegistryCleanupManager::~OptionsRegistryCleanupManager() {
  CleanupAll();
}

ScopedRegistrationToken::ScopedRegistrationToken(RegistrationToken token)
    : token_(std::move(token)) {}

ScopedRegistrationToken::~ScopedRegistrationToken() {
  token_.Unregister();
}

ScopedRegistrationToken::ScopedRegistrationToken(ScopedRegistrationToken&& other) noexcept
    : token_(std::move(other.token_)) {}

ScopedRegistrationToken& ScopedRegistrationToken::operator=(ScopedRegistrationToken&& other) noexcept {
  if (this != &other) {
    token_.Unregister();
    token_ = std::move(other.token_);
  }
  return *this;
}

}  // namespace tool
}  // namespace mediapipe
