#include "offload_read_miss_evictor.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <vector>

namespace mooncake {
namespace {

// Polls until pred() holds or 5 s pass.
template <typename Pred>
bool WaitUntil(Pred pred) {
    const auto until =
        std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (!pred()) {
        if (std::chrono::steady_clock::now() > until) return false;
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    return true;
}

// Records every batch handed to the evict call; can hold the call until
// Open(), the way a slow master RPC does.
class Recorder {
   public:
    explicit Recorder(bool start_open = true) : open_(start_open) {}

    OffloadReadMissEvictor::EvictFn Fn(ErrorCode result = ErrorCode::OK) {
        return [this, result](const std::vector<std::string>& keys)
                   -> tl::expected<void, ErrorCode> {
            {
                std::unique_lock<std::mutex> lock(mu_);
                batches_.push_back(keys);
                cv_.notify_all();
                cv_.wait(lock, [this] { return open_; });
            }
            if (result != ErrorCode::OK) return tl::make_unexpected(result);
            return {};
        };
    }
    void Open() {
        {
            std::lock_guard<std::mutex> lock(mu_);
            open_ = true;
        }
        cv_.notify_all();
    }
    std::vector<std::vector<std::string>> batches() {
        std::lock_guard<std::mutex> lock(mu_);
        return batches_;
    }
    std::multiset<std::string> keys() {
        std::lock_guard<std::mutex> lock(mu_);
        std::multiset<std::string> all;
        for (const auto& b : batches_) all.insert(b.begin(), b.end());
        return all;
    }

   private:
    std::mutex mu_;
    std::condition_variable cv_;
    bool open_;
    std::vector<std::vector<std::string>> batches_;
};

std::vector<std::string> Keys(const std::string& prefix, int n) {
    std::vector<std::string> keys;
    for (int i = 0; i < n; ++i) keys.push_back(prefix + std::to_string(i));
    return keys;
}

TEST(OffloadReadMissEvictorTest, SendsQueuedKeysInBoundedBatches) {
    Recorder recorder;
    OffloadReadMissEvictor evictor(recorder.Fn(), /*max_pending=*/100,
                                   /*max_batch=*/4);
    const auto keys = Keys("k", 10);
    auto added = evictor.Add(keys);
    EXPECT_EQ(added.queued, 10u);
    EXPECT_EQ(added.dropped, 0u);
    ASSERT_TRUE(WaitUntil([&] { return evictor.stats().sent_keys == 10; }));
    EXPECT_EQ(recorder.keys(),
              std::multiset<std::string>(keys.begin(), keys.end()));
    for (const auto& batch : recorder.batches()) {
        EXPECT_LE(batch.size(), 4u);
    }
    EXPECT_EQ(evictor.stats().pending, 0u);
}

// A key queued twice before it is sent is sent once; a key that fails again
// after it was sent is queued again.
TEST(OffloadReadMissEvictorTest, DeduplicatesKeysWhileQueued) {
    Recorder recorder(/*start_open=*/false);
    OffloadReadMissEvictor evictor(recorder.Fn(), 100, 100);
    EXPECT_EQ(evictor.Add({"a"}).queued, 1u);
    // "a" is now in the held evict call; the next keys wait in the queue.
    ASSERT_TRUE(WaitUntil([&] { return recorder.batches().size() == 1; }));
    EXPECT_EQ(evictor.Add({"b", "b", "c"}).queued, 2u);
    EXPECT_EQ(evictor.Add({"b"}).queued, 0u);
    EXPECT_EQ(evictor.Add({"a"}).queued, 1u);
    recorder.Open();
    ASSERT_TRUE(WaitUntil([&] { return evictor.stats().sent_keys == 4; }));
    EXPECT_EQ(recorder.keys(),
              (std::multiset<std::string>{"a", "a", "b", "c"}));
    EXPECT_EQ(evictor.stats().queued_keys, 4u);
}

// Add() never waits for the master: with the evict call held, keys beyond
// max_pending are dropped and counted.
TEST(OffloadReadMissEvictorTest, DropsKeysBeyondMaxPending) {
    Recorder recorder(/*start_open=*/false);
    OffloadReadMissEvictor evictor(recorder.Fn(), /*max_pending=*/2, 100);
    evictor.Add({"held"});
    ASSERT_TRUE(WaitUntil([&] { return recorder.batches().size() == 1; }));
    auto added = evictor.Add({"a", "b", "c", "a"});
    EXPECT_EQ(added.queued, 2u);
    EXPECT_EQ(added.dropped, 1u);
    EXPECT_EQ(evictor.stats().dropped_keys, 1u);
    recorder.Open();
    ASSERT_TRUE(WaitUntil([&] { return evictor.stats().sent_keys == 3; }));
    EXPECT_EQ(recorder.keys(), (std::multiset<std::string>{"held", "a", "b"}));
}

// A failed evict call is counted and not retried: the key stays as it was
// (disk-only), the behavior without the evictor.
TEST(OffloadReadMissEvictorTest, FailedBatchIsCountedAndNotRetried) {
    Recorder recorder;
    OffloadReadMissEvictor evictor(
        recorder.Fn(ErrorCode::UNAVAILABLE_IN_CURRENT_STATUS), 100, 100);
    evictor.Add({"a", "b"});
    ASSERT_TRUE(WaitUntil([&] { return evictor.stats().failed_batches == 1; }));
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    EXPECT_EQ(recorder.batches().size(), 1u);
    EXPECT_EQ(evictor.stats().sent_keys, 2u);
    EXPECT_EQ(evictor.stats().pending, 0u);
}

// The destructor waits for a call in progress and does not send what is
// still queued.
TEST(OffloadReadMissEvictorTest, DestructorWaitsForTheCallInProgress) {
    Recorder recorder(/*start_open=*/false);
    std::atomic<bool> destroyed{false};
    auto evictor =
        std::make_unique<OffloadReadMissEvictor>(recorder.Fn(), 100, 1);
    evictor->Add({"a"});
    ASSERT_TRUE(WaitUntil([&] { return recorder.batches().size() == 1; }));
    evictor->Add({"b"});
    std::thread destroyer([&] {
        evictor.reset();
        destroyed = true;
    });
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    EXPECT_FALSE(destroyed.load());
    recorder.Open();
    destroyer.join();
    EXPECT_TRUE(destroyed.load());
    // "b" may or may not have been taken before the stop was seen; nothing
    // after the stop is sent.
    EXPECT_LE(recorder.batches().size(), 2u);
}

}  // namespace
}  // namespace mooncake
