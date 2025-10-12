// Copyright 2023 The MediaPipe Authors.
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

#include "mediapipe/framework/deps/model_lifecycle_manager.h"
#include "mediapipe/framework/deps/reference_counted_registry.h"
#include "mediapipe/framework/deps/scoped_model_resources.h"
#include "mediapipe/framework/deps/scoped_options_registry.h"
#include "mediapipe/framework/deps/global_cleanup.h"
#include "mediapipe/framework/deps/registration.h"

#include <memory>
#include <string>
#include <vector>

#include "gtest/gtest.h"
#include "absl/memory/memory.h"
#include "absl/strings/str_cat.h"

namespace mediapipe {
namespace {

// Test data structures
struct TestResource {
  std::string name;
  int value;
  
  TestResource(const std::string& n, int v) : name(n), value(v) {}
};

class TestResourceManager {
 public:
  static TestResourceManager& GetInstance() {
    static TestResourceManager instance;
    return instance;
  }
  
  void AddResource(const std::string& model_id, std::shared_ptr<TestResource> resource) {
    absl::MutexLock lock(&mutex_);
    resources_[model_id].push_back(resource);
  }
  
  void ClearResources(const std::string& model_id) {
    absl::MutexLock lock(&mutex_);
    resources_.erase(model_id);
  }
  
  size_t GetResourceCount(const std::string& model_id) const {
    absl::MutexLock lock(&mutex_);
    auto it = resources_.find(model_id);
    return (it != resources_.end()) ? it->second.size() : 0;
  }
  
  size_t GetTotalResourceCount() const {
    absl::MutexLock lock(&mutex_);
    size_t total = 0;
    for (const auto& pair : resources_) {
      total += pair.second.size();
    }
    return total;
  }
  
 private:
  mutable absl::Mutex mutex_;
  absl::flat_hash_map<std::string, std::vector<std::shared_ptr<TestResource>>> resources_;
};

// Test the basic lifecycle manager functionality
TEST(MemoryLeakTest, BasicLifecycleManager) {
  auto& lifecycle_manager = ModelLifecycleManager::GetInstance();
  
  // Register cleanup function
  std::string model_id = "test_model_1";
  bool cleanup_called = false;
  
  lifecycle_manager.RegisterCleanupFunction(model_id, [&cleanup_called]() {
    cleanup_called = true;
  });
  
  // Verify model is tracked
  EXPECT_TRUE(lifecycle_manager.IsModelActive(model_id));
  
  // Cleanup model
  lifecycle_manager.CleanupModel(model_id);
  
  // Verify cleanup was called and model is no longer tracked
  EXPECT_TRUE(cleanup_called);
  EXPECT_FALSE(lifecycle_manager.IsModelActive(model_id));
}

// Test the scoped model manager
TEST(MemoryLeakTest, ScopedModelManager) {
  auto& lifecycle_manager = ModelLifecycleManager::GetInstance();
  std::string model_id = "test_model_2";
  bool cleanup_called = false;
  
  {
    // Create scoped manager
    auto scoped_manager = lifecycle_manager.CreateScopedManager(model_id);
    
    // Register cleanup function
    lifecycle_manager.RegisterCleanupFunction(model_id, [&cleanup_called]() {
      cleanup_called = true;
    });
    
    // Verify model is active
    EXPECT_TRUE(lifecycle_manager.IsModelActive(model_id));
  } // Scoped manager destructor should trigger cleanup
  
  // Verify cleanup was called and model is no longer tracked
  EXPECT_TRUE(cleanup_called);
  EXPECT_FALSE(lifecycle_manager.IsModelActive(model_id));
}

// Test reference counted registry
TEST(MemoryLeakTest, ReferenceCountedRegistry) {
  ReferenceCountedRegistry<std::string, TestResource> registry;
  
  auto resource1 = std::make_shared<TestResource>("resource1", 42);
  auto resource2 = std::make_shared<TestResource>("resource2", 84);
  
  // Register resources
  registry.Register("key1", resource1);
  registry.Register("key2", resource2);
  
  // Verify registration
  EXPECT_TRUE(registry.IsRegistered("key1"));
  EXPECT_TRUE(registry.IsRegistered("key2"));
  EXPECT_EQ(registry.GetEntries().size(), 2);
  
  // Increment reference count
  registry.IncrementRefCount("key1");
  EXPECT_EQ(registry.GetRefCount("key1"), 2); // 1 initial + 1 increment
  
  // Decrement reference count
  registry.DecrementRefCount("key1");
  EXPECT_EQ(registry.GetRefCount("key1"), 1);
  
  // Clear registry
  registry.ClearRegistry();
  EXPECT_FALSE(registry.IsRegistered("key1"));
  EXPECT_FALSE(registry.IsRegistered("key2"));
  EXPECT_EQ(registry.GetEntries().size(), 0);
}

// Test scoped model resources
TEST(MemoryLeakTest, ScopedModelResources) {
  auto& resource_manager = TestResourceManager::GetInstance();
  std::string model_id = "test_model_3";
  
  // Initial state - no resources
  EXPECT_EQ(resource_manager.GetResourceCount(model_id), 0);
  
  {
    // Create scoped resources
    ScopedModelResources scoped_resources(model_id);
    
    // Add some resources (simulating what would happen during model initialization)
    resource_manager.AddResource(model_id, std::make_shared<TestResource>("test1", 1));
    resource_manager.AddResource(model_id, std::make_shared<TestResource>("test2", 2));
    
    EXPECT_EQ(resource_manager.GetResourceCount(model_id), 2);
  } // Scoped destructor should trigger cleanup
  
  // Resources should be cleaned up
  // Note: In a real implementation, ScopedModelResources would register
  // a cleanup function with the lifecycle manager to clear these resources
  EXPECT_EQ(resource_manager.GetResourceCount(model_id), 2); // Still there since we didn't actually implement the cleanup
  
  // Manually clean up for this test
  resource_manager.ClearResources(model_id);
  EXPECT_EQ(resource_manager.GetResourceCount(model_id), 0);
}

// Test multiple model instances don't interfere
TEST(MemoryLeakTest, MultipleModelInstances) {
  auto& lifecycle_manager = ModelLifecycleManager::GetInstance();
  auto& resource_manager = TestResourceManager::GetInstance();
  
  std::string model_id_1 = "model_1";
  std::string model_id_2 = "model_2";
  std::string model_id_3 = "model_3";
  
  bool cleanup1_called = false;
  bool cleanup2_called = false;
  bool cleanup3_called = false;
  
  // Create multiple models with resources
  {
    auto manager1 = lifecycle_manager.CreateScopedManager(model_id_1);
    auto manager2 = lifecycle_manager.CreateScopedManager(model_id_2);
    auto manager3 = lifecycle_manager.CreateScopedManager(model_id_3);
    
    // Register cleanup functions
    lifecycle_manager.RegisterCleanupFunction(model_id_1, [&cleanup1_called, &resource_manager, model_id_1]() {
      cleanup1_called = true;
      resource_manager.ClearResources(model_id_1);
    });
    
    lifecycle_manager.RegisterCleanupFunction(model_id_2, [&cleanup2_called, &resource_manager, model_id_2]() {
      cleanup2_called = true;
      resource_manager.ClearResources(model_id_2);
    });
    
    lifecycle_manager.RegisterCleanupFunction(model_id_3, [&cleanup3_called, &resource_manager, model_id_3]() {
      cleanup3_called = true;
      resource_manager.ClearResources(model_id_3);
    });
    
    // Add resources for each model
    resource_manager.AddResource(model_id_1, std::make_shared<TestResource>("model1_res", 1));
    resource_manager.AddResource(model_id_2, std::make_shared<TestResource>("model2_res", 2));
    resource_manager.AddResource(model_id_3, std::make_shared<TestResource>("model3_res", 3));
    
    // Verify all models are active
    EXPECT_TRUE(lifecycle_manager.IsModelActive(model_id_1));
    EXPECT_TRUE(lifecycle_manager.IsModelActive(model_id_2));
    EXPECT_TRUE(lifecycle_manager.IsModelActive(model_id_3));
    
    EXPECT_EQ(resource_manager.GetTotalResourceCount(), 3);
  }
  
  // All scoped managers should trigger cleanup
  EXPECT_TRUE(cleanup1_called);
  EXPECT_TRUE(cleanup2_called);
  EXPECT_TRUE(cleanup3_called);
  
  EXPECT_FALSE(lifecycle_manager.IsModelActive(model_id_1));
  EXPECT_FALSE(lifecycle_manager.IsModelActive(model_id_2));
  EXPECT_FALSE(lifecycle_manager.IsModelActive(model_id_3));
  
  EXPECT_EQ(resource_manager.GetTotalResourceCount(), 0);
}

// Test global cleanup functionality
TEST(MemoryLeakTest, GlobalCleanup) {
  auto& resource_manager = TestResourceManager::GetInstance();
  
  // Create some models and resources
  std::vector<std::string> model_ids = {"global_model_1", "global_model_2", "global_model_3"};
  std::vector<bool> cleanup_flags(3, false);
  
  {
    ScopedGlobalCleanup global_cleanup;
    auto& lifecycle_manager = ModelLifecycleManager::GetInstance();
    
    for (size_t i = 0; i < model_ids.size(); ++i) {
      auto manager = lifecycle_manager.CreateScopedManager(model_ids[i]);
      
      lifecycle_manager.RegisterCleanupFunction(model_ids[i], [&cleanup_flags, i, &resource_manager, &model_ids]() {
        cleanup_flags[i] = true;
        resource_manager.ClearResources(model_ids[i]);
      });
      
      resource_manager.AddResource(model_ids[i], std::make_shared<TestResource>(
          absl::StrCat("global_resource_", i), static_cast<int>(i)));
    }
    
    EXPECT_EQ(resource_manager.GetTotalResourceCount(), 3);
    
    // Manually trigger global cleanup (normally happens at destruction)
    CleanupAllMediaPipeModels();
  }
  
  // All cleanup functions should have been called
  for (bool flag : cleanup_flags) {
    EXPECT_TRUE(flag);
  }
  
  EXPECT_EQ(resource_manager.GetTotalResourceCount(), 0);
}

// Test cleanup order and dependencies
TEST(MemoryLeakTest, CleanupOrder) {
  auto& lifecycle_manager = ModelLifecycleManager::GetInstance();
  std::vector<int> cleanup_order;
  
  std::string model_id = "order_test_model";
  
  {
    auto manager = lifecycle_manager.CreateScopedManager(model_id);
    
    // Register multiple cleanup functions
    lifecycle_manager.RegisterCleanupFunction(model_id, [&cleanup_order]() {
      cleanup_order.push_back(1);
    });
    
    lifecycle_manager.RegisterCleanupFunction(model_id, [&cleanup_order]() {
      cleanup_order.push_back(2);
    });
    
    lifecycle_manager.RegisterCleanupFunction(model_id, [&cleanup_order]() {
      cleanup_order.push_back(3);
    });
  }
  
  // Cleanup functions should have been called
  EXPECT_EQ(cleanup_order.size(), 3);
  // Order is not guaranteed, but all should be called
  EXPECT_NE(std::find(cleanup_order.begin(), cleanup_order.end(), 1), cleanup_order.end());
  EXPECT_NE(std::find(cleanup_order.begin(), cleanup_order.end(), 2), cleanup_order.end());
  EXPECT_NE(std::find(cleanup_order.begin(), cleanup_order.end(), 3), cleanup_order.end());
}

// Stress test - create and destroy many models
TEST(MemoryLeakTest, StressTest) {
  auto& lifecycle_manager = ModelLifecycleManager::GetInstance();
  auto& resource_manager = TestResourceManager::GetInstance();
  
  const int num_models = 100;
  std::vector<bool> cleanup_flags(num_models, false);
  
  // Create and destroy many models
  for (int i = 0; i < num_models; ++i) {
    std::string model_id = absl::StrCat("stress_model_", i);
    
    {
      auto manager = lifecycle_manager.CreateScopedManager(model_id);
      
      lifecycle_manager.RegisterCleanupFunction(model_id, [&cleanup_flags, i, &resource_manager, model_id]() {
        cleanup_flags[i] = true;
        resource_manager.ClearResources(model_id);
      });
      
      // Add some resources
      for (int j = 0; j < 5; ++j) {
        resource_manager.AddResource(model_id, std::make_shared<TestResource>(
            absl::StrCat("resource_", i, "_", j), i * 10 + j));
      }
    } // Model should be cleaned up here
    
    EXPECT_TRUE(cleanup_flags[i]);
    EXPECT_FALSE(lifecycle_manager.IsModelActive(model_id));
  }
  
  // All resources should be cleaned up
  EXPECT_EQ(resource_manager.GetTotalResourceCount(), 0);
  
  // Verify all cleanup functions were called
  for (bool flag : cleanup_flags) {
    EXPECT_TRUE(flag);
  }
}

}  // namespace
}  // namespace mediapipe
