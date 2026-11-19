// Reads that arrive while a bucket eviction is waiting for the master.
//
// An eviction removes the keys from the local index (PrepareEviction), tells
// the master (the eviction handler, BatchEvictDiskReplica in production),
// then commits and deletes the files (CommitPreparedEviction,
// FinalizeEviction). Until the master has processed the notification it keeps
// handing out the disk replica, so a BatchLoad can land inside the handler.
// These tests call BatchLoad from inside the handler to pin what such a read
// sees, with and without read_pending_eviction.

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <functional>
#include <future>
#include <memory>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "storage_backend.h"

namespace fs = std::filesystem;

namespace mooncake::test {
namespace {

constexpr size_t kValueSize = 4 * 1024;

std::string ValueFor(const std::string& key) {
    std::string value(kValueSize, '\0');
    for (size_t i = 0; i < value.size(); ++i) {
        value[i] =
            static_cast<char>('a' + (std::hash<std::string>{}(key) + i) % 26);
    }
    return value;
}

struct ReadResult {
    tl::expected<void, ErrorCode> status;
    bool bytes_match = false;
};

class BucketEvictionReadTest : public ::testing::Test {
   protected:
    void SetUp() override {
        FLAGS_logtostderr = true;
        const auto* info =
            ::testing::UnitTest::GetInstance()->current_test_info();
        data_path_ = (fs::current_path() / "bucket_eviction_read_test_data" /
                      info->name())
                         .string();
        fs::remove_all(data_path_);
        fs::create_directories(data_path_);
    }

    void TearDown() override { fs::remove_all(data_path_); }

    // Three buckets of two 4 KiB keys fill about 24 KiB of a 40 KiB quota.
    // EvictAboveDiskWatermark(kHigh, kLow) then evicts exactly the oldest
    // bucket (usage drops to about 16 KiB, under 18 KiB).
    static constexpr double kHigh = 0.50;
    static constexpr double kLow = 0.45;

    std::unique_ptr<BucketStorageBackend> MakeBackend(bool read_pending) {
        FileStorageConfig config;
        config.storage_filepath = data_path_;
        BucketBackendConfig bucket_config;
        bucket_config.bucket_keys_limit = 10;
        bucket_config.bucket_size_limit = 64 * 1024;
        bucket_config.max_total_size = 40 * 1024;
        bucket_config.eviction_policy = BucketEvictionPolicy::FIFO;
        bucket_config.read_pending_eviction = read_pending;
        auto backend =
            std::make_unique<BucketStorageBackend>(config, bucket_config);
        EXPECT_TRUE(backend->Init().has_value());
        return backend;
    }

    // Writes one bucket holding `keys` and returns its id.
    int64_t Offload(
        BucketStorageBackend& backend, const std::vector<std::string>& keys,
        StorageBackendInterface::EvictionHandler eviction_handler = nullptr) {
        std::vector<std::string> values;
        values.reserve(keys.size());
        std::unordered_map<std::string, std::vector<Slice>> batch;
        for (const auto& key : keys) {
            values.push_back(ValueFor(key));
            batch.emplace(key, std::vector<Slice>{Slice{values.back().data(),
                                                        values.back().size()}});
        }
        auto result = backend.BatchOffload(
            batch,
            [](const std::vector<std::string>&,
               std::vector<StorageObjectMetadata>&) { return ErrorCode::OK; },
            std::move(eviction_handler));
        EXPECT_TRUE(result.has_value())
            << (result ? ErrorCode::OK : result.error());
        return result ? result.value() : -1;
    }

    // Reads `keys` in one BatchLoad and checks the bytes.
    static ReadResult Read(BucketStorageBackend& backend,
                           const std::vector<std::string>& keys) {
        std::vector<std::unique_ptr<char[]>> buffers;
        std::unordered_map<std::string, Slice> slices;
        for (const auto& key : keys) {
            buffers.push_back(std::make_unique<char[]>(kValueSize));
            slices.emplace(key, Slice{buffers.back().get(), kValueSize});
        }
        ReadResult result{backend.BatchLoad(slices), false};
        if (result.status) {
            result.bytes_match = true;
            for (const auto& key : keys) {
                const auto& slice = slices.at(key);
                if (std::string(static_cast<char*>(slice.ptr), slice.size) !=
                    ValueFor(key)) {
                    result.bytes_match = false;
                }
            }
        }
        return result;
    }

    fs::path DataFile(int64_t bucket_id) const {
        return fs::path(data_path_) / (std::to_string(bucket_id) + ".bucket");
    }

    // Fills three buckets; the first one is the eviction candidate.
    std::vector<int64_t> FillThreeBuckets(BucketStorageBackend& backend) {
        return {Offload(backend, {"old_a", "old_b"}),
                Offload(backend, {"mid_a", "mid_b"}),
                Offload(backend, {"new_a", "new_b"})};
    }

    static bool IsExist(BucketStorageBackend& backend, const std::string& key) {
        auto exists = backend.IsExist(key);
        EXPECT_TRUE(exists.has_value());
        return exists && exists.value();
    }

    std::string data_path_;
};

const std::vector<std::string> kOldKeys = {"old_a", "old_b"};

// Default: a read inside the handler fails with INVALID_KEY although the
// bucket file is still on disk. This is the race behind the engine errors.
TEST_F(BucketEvictionReadTest, DefaultReadInsideTheWindowFailsWithInvalidKey) {
    auto backend = MakeBackend(/*read_pending=*/false);
    const auto ids = FillThreeBuckets(*backend);

    bool handler_ran = false;
    auto evicted = backend->EvictAboveDiskWatermark(
        kHigh, kLow,
        [&](const std::vector<std::string>& keys)
            -> tl::expected<void, ErrorCode> {
            handler_ran = true;
            EXPECT_EQ(keys.size(), 2u);
            EXPECT_TRUE(fs::exists(DataFile(ids[0])));
            EXPECT_EQ(backend->PendingEvictionReadableKeysForTest(), 0u);
            auto read = Read(*backend, kOldKeys);
            EXPECT_FALSE(read.status.has_value());
            if (!read.status) {
                EXPECT_EQ(read.status.error(), ErrorCode::INVALID_KEY);
            }
            // One evicted key fails the whole batch, live keys included.
            auto mixed = Read(*backend, {"old_a", "new_a"});
            EXPECT_FALSE(mixed.status.has_value());
            return {};
        });
    ASSERT_TRUE(evicted.has_value());
    EXPECT_TRUE(handler_ran);
    EXPECT_FALSE(fs::exists(DataFile(ids[0])));
    EXPECT_TRUE(Read(*backend, {"mid_a", "new_b"}).bytes_match);
}

// Opt-in: the same read returns the right bytes. After the commit the keys
// are gone exactly as before and the file is deleted.
TEST_F(BucketEvictionReadTest, OptInReadInsideTheWindowReturnsTheBytes) {
    auto backend = MakeBackend(/*read_pending=*/true);
    const auto ids = FillThreeBuckets(*backend);

    bool handler_ran = false;
    auto evicted = backend->EvictAboveDiskWatermark(
        kHigh, kLow,
        [&](const std::vector<std::string>& keys)
            -> tl::expected<void, ErrorCode> {
            handler_ran = true;
            EXPECT_EQ(keys.size(), 2u);
            EXPECT_TRUE(fs::exists(DataFile(ids[0])));
            EXPECT_EQ(backend->PendingEvictionReadableKeysForTest(), 2u);
            // The index no longer has the keys; only BatchLoad looks further.
            EXPECT_FALSE(IsExist(*backend, "old_a"));
            auto read = Read(*backend, kOldKeys);
            EXPECT_TRUE(read.status.has_value());
            EXPECT_TRUE(read.bytes_match);
            auto mixed = Read(*backend, {"old_b", "mid_a", "new_b"});
            EXPECT_TRUE(mixed.status.has_value());
            EXPECT_TRUE(mixed.bytes_match);
            // A key that was never stored still fails.
            auto unknown = Read(*backend, {"old_a", "never_stored"});
            EXPECT_FALSE(unknown.status.has_value());
            if (!unknown.status) {
                EXPECT_EQ(unknown.status.error(), ErrorCode::INVALID_KEY);
            }
            return {};
        });
    ASSERT_TRUE(evicted.has_value());
    EXPECT_TRUE(handler_ran);
    EXPECT_EQ(evicted.value().size(), 2u);

    EXPECT_EQ(backend->PendingEvictionReadableKeysForTest(), 0u);
    EXPECT_FALSE(fs::exists(DataFile(ids[0])));
    auto after = Read(*backend, kOldKeys);
    ASSERT_FALSE(after.status.has_value());
    EXPECT_EQ(after.status.error(), ErrorCode::INVALID_KEY);
    EXPECT_TRUE(Read(*backend, {"mid_a", "mid_b", "new_a"}).bytes_match);
}

// The eviction that BatchOffload runs to make room for a new bucket has the
// same window and the same fix.
TEST_F(BucketEvictionReadTest, OffloadDrivenEvictionHasTheSameWindow) {
    for (bool read_pending : {false, true}) {
        SCOPED_TRACE(read_pending ? "opt-in" : "default");
        fs::remove_all(data_path_);
        fs::create_directories(data_path_);
        auto backend = MakeBackend(read_pending);
        const auto ids = FillThreeBuckets(*backend);
        // A fourth and fifth bucket push total over the 40 KiB quota.
        Offload(*backend, {"more_a", "more_b"});

        bool handler_ran = false;
        Offload(*backend, {"last_a", "last_b"},
                [&](const std::vector<std::string>& keys)
                    -> tl::expected<void, ErrorCode> {
                    handler_ran = true;
                    EXPECT_EQ(keys.size(), 2u);
                    auto read = Read(*backend, kOldKeys);
                    EXPECT_EQ(read.status.has_value(), read_pending);
                    EXPECT_EQ(read.bytes_match, read_pending);
                    if (!read.status) {
                        EXPECT_EQ(read.status.error(), ErrorCode::INVALID_KEY);
                    }
                    return {};
                });
        EXPECT_TRUE(handler_ran);
        EXPECT_FALSE(fs::exists(DataFile(ids[0])));
        EXPECT_FALSE(Read(*backend, kOldKeys).status.has_value());
        EXPECT_TRUE(Read(*backend, {"last_a", "mid_b"}).bytes_match);
        EXPECT_EQ(backend->PendingEvictionReadableKeysForTest(), 0u);
    }
}

// When the master notification fails the eviction is restored: the keys go
// back to the index, the side list is emptied, and a later eviction works.
TEST_F(BucketEvictionReadTest, RestoreReturnsTheKeysToTheIndex) {
    auto backend = MakeBackend(/*read_pending=*/true);
    const auto ids = FillThreeBuckets(*backend);

    auto failed = backend->EvictAboveDiskWatermark(
        kHigh, kLow,
        [&](const std::vector<std::string>&) -> tl::expected<void, ErrorCode> {
            EXPECT_TRUE(Read(*backend, kOldKeys).bytes_match);
            return tl::make_unexpected(ErrorCode::RPC_FAIL);
        });
    ASSERT_FALSE(failed.has_value());
    EXPECT_EQ(failed.error(), ErrorCode::RPC_FAIL);

    EXPECT_EQ(backend->PendingEvictionReadableKeysForTest(), 0u);
    EXPECT_TRUE(IsExist(*backend, "old_a"));
    EXPECT_TRUE(IsExist(*backend, "old_b"));
    EXPECT_TRUE(fs::exists(DataFile(ids[0])));
    EXPECT_TRUE(Read(*backend, kOldKeys).bytes_match);

    // The next attempt evicts the same bucket and the keys are gone after it.
    std::vector<std::string> notified;
    auto evicted = backend->EvictAboveDiskWatermark(
        kHigh, kLow,
        [&](const std::vector<std::string>& keys)
            -> tl::expected<void, ErrorCode> {
            notified = keys;
            EXPECT_EQ(backend->PendingEvictionReadableKeysForTest(), 2u);
            EXPECT_TRUE(Read(*backend, kOldKeys).bytes_match);
            return {};
        });
    ASSERT_TRUE(evicted.has_value());
    EXPECT_EQ(notified.size(), 2u);
    EXPECT_FALSE(IsExist(*backend, "old_a"));
    EXPECT_FALSE(fs::exists(DataFile(ids[0])));
    EXPECT_EQ(backend->PendingEvictionReadableKeysForTest(), 0u);
    EXPECT_FALSE(Read(*backend, kOldKeys).status.has_value());
}

// Blocks the first BatchLoad that reaches the hook until Release().
class FirstReadBlocker {
   public:
    void Install(BucketStorageBackend& backend) {
        backend.SetBatchLoadAfterPlanHookForTest([this] {
            if (!armed_.exchange(false)) return;
            at_hook_.set_value();
            release_future_.wait();
        });
    }
    // Bounded so a reader that never reaches the hook fails the test
    // instead of hanging it.
    bool WaitUntilBlocked() {
        return at_hook_future_.wait_for(std::chrono::seconds(10)) ==
               std::future_status::ready;
    }
    void Release() { release_.set_value(); }

   private:
    std::atomic<bool> armed_{true};
    std::promise<void> at_hook_;
    std::future<void> at_hook_future_ = at_hook_.get_future();
    std::promise<void> release_;
    std::shared_future<void> release_future_ = release_.get_future().share();
};

// A read that took its guard inside the window holds off the file deletion
// until it is done, even though the commit has already happened.
TEST_F(BucketEvictionReadTest, FinalizeWaitsForAReadStartedInTheWindow) {
    auto backend = MakeBackend(/*read_pending=*/true);
    const auto ids = FillThreeBuckets(*backend);
    FirstReadBlocker blocker;
    blocker.Install(*backend);

    std::future<ReadResult> reader;
    std::atomic<bool> handler_returned{false};
    auto eviction = std::async(std::launch::async, [&] {
        return backend->EvictAboveDiskWatermark(
            kHigh, kLow,
            [&](const std::vector<std::string>&)
                -> tl::expected<void, ErrorCode> {
                reader = std::async(std::launch::async,
                                    [&] { return Read(*backend, kOldKeys); });
                EXPECT_TRUE(blocker.WaitUntilBlocked());
                handler_returned = true;
                return {};
            });
    });

    // Bounded: if the eviction never reaches the handler (nothing above the
    // watermark), fail instead of spinning forever.
    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::seconds(10);
    while (!handler_returned &&
           eviction.wait_for(std::chrono::milliseconds(1)) !=
               std::future_status::ready &&
           std::chrono::steady_clock::now() < deadline) {
    }
    ASSERT_TRUE(handler_returned.load());
    // Commit runs right after the handler; FinalizeEviction then spins on
    // the reader's guard.
    EXPECT_EQ(eviction.wait_for(std::chrono::milliseconds(300)),
              std::future_status::timeout);
    EXPECT_TRUE(fs::exists(DataFile(ids[0])));
    EXPECT_EQ(backend->PendingEvictionReadableKeysForTest(), 0u);
    // A read that starts after the commit fails as it does by default.
    // (EXPECT, not ASSERT, before Release so a failure cannot hang the test.)
    auto late = Read(*backend, kOldKeys);
    EXPECT_FALSE(late.status.has_value());
    if (!late.status) {
        EXPECT_EQ(late.status.error(), ErrorCode::INVALID_KEY);
    }

    blocker.Release();
    auto held = reader.get();
    EXPECT_TRUE(held.status.has_value());
    EXPECT_TRUE(held.bytes_match);
    auto evicted = eviction.get();
    ASSERT_TRUE(evicted.has_value());
    EXPECT_EQ(evicted.value().size(), 2u);
    EXPECT_FALSE(fs::exists(DataFile(ids[0])));
}

// A read that took its guard inside a window that then got restored keeps
// that guard on the restored bucket; the next eviction of that bucket waits
// for it.
TEST_F(BucketEvictionReadTest, RestoredBucketKeepsAHeldReadGuarded) {
    auto backend = MakeBackend(/*read_pending=*/true);
    const auto ids = FillThreeBuckets(*backend);
    FirstReadBlocker blocker;
    blocker.Install(*backend);

    std::future<ReadResult> reader;
    auto failed = backend->EvictAboveDiskWatermark(
        kHigh, kLow,
        [&](const std::vector<std::string>&) -> tl::expected<void, ErrorCode> {
            reader = std::async(std::launch::async,
                                [&] { return Read(*backend, kOldKeys); });
            EXPECT_TRUE(blocker.WaitUntilBlocked());
            return tl::make_unexpected(ErrorCode::RPC_FAIL);
        });
    EXPECT_FALSE(failed.has_value());
    EXPECT_TRUE(IsExist(*backend, "old_a"));

    auto eviction = std::async(std::launch::async, [&] {
        return backend->EvictAboveDiskWatermark(
            kHigh, kLow,
            [](const std::vector<std::string>&)
                -> tl::expected<void, ErrorCode> { return {}; });
    });
    EXPECT_EQ(eviction.wait_for(std::chrono::milliseconds(300)),
              std::future_status::timeout);
    EXPECT_TRUE(fs::exists(DataFile(ids[0])));

    blocker.Release();
    auto held = reader.get();
    EXPECT_TRUE(held.status.has_value());
    EXPECT_TRUE(held.bytes_match);
    auto evicted = eviction.get();
    ASSERT_TRUE(evicted.has_value());
    EXPECT_EQ(evicted.value().size(), 2u);
    EXPECT_FALSE(fs::exists(DataFile(ids[0])));
    EXPECT_FALSE(IsExist(*backend, "old_a"));
}

// Several readers hammer the evicted keys while the handler stands in for a
// slow master. Every read that both started and finished inside the window
// succeeds with the right bytes when opted in, and fails when not.
TEST_F(BucketEvictionReadTest, ConcurrentReadersInsideTheWindow) {
    constexpr int kReaders = 4;
    constexpr int kReadsPerReader = 25;
    for (bool read_pending : {false, true}) {
        SCOPED_TRACE(read_pending ? "opt-in" : "default");
        fs::remove_all(data_path_);
        fs::create_directories(data_path_);
        auto backend = MakeBackend(read_pending);
        const auto ids = FillThreeBuckets(*backend);

        std::atomic<bool> in_window{false};
        std::atomic<bool> stop{false};
        std::atomic<int> window_ok{0};
        std::atomic<int> window_invalid_key{0};
        std::atomic<int> window_other{0};
        std::atomic<int> window_bad_bytes{0};
        std::vector<std::atomic<int>> per_reader(kReaders);
        std::vector<std::thread> readers;
        for (int r = 0; r < kReaders; ++r) {
            readers.emplace_back([&, r] {
                // Alternate between evicted-only and mixed batches.
                const std::vector<std::vector<std::string>> batches = {
                    kOldKeys, {"old_b", "mid_a"}, {"new_b", "old_a"}};
                // Read only inside the window. Readers looping on the
                // shared lock outside it would starve the exclusive lock
                // that PrepareEviction and CommitPreparedEviction take.
                while (!in_window && !stop) std::this_thread::yield();
                for (size_t i = 0; in_window; ++i) {
                    auto read = Read(*backend, batches[i % batches.size()]);
                    // Count only reads that also ended inside the window:
                    // they planned before the handler returned, i.e. before
                    // the commit.
                    if (!in_window) break;
                    per_reader[r]++;
                    if (read.status) {
                        window_ok++;
                        if (!read.bytes_match) window_bad_bytes++;
                    } else if (read.status.error() == ErrorCode::INVALID_KEY) {
                        window_invalid_key++;
                    } else {
                        window_other++;
                    }
                }
            });
        }

        auto evicted = backend->EvictAboveDiskWatermark(
            kHigh, kLow,
            [&](const std::vector<std::string>&)
                -> tl::expected<void, ErrorCode> {
                in_window = true;
                const auto deadline =
                    std::chrono::steady_clock::now() + std::chrono::seconds(5);
                for (int r = 0; r < kReaders; ++r) {
                    while (per_reader[r] < kReadsPerReader &&
                           std::chrono::steady_clock::now() < deadline) {
                        std::this_thread::yield();
                    }
                }
                in_window = false;
                return {};
            });
        stop = true;
        for (auto& t : readers) t.join();

        ASSERT_TRUE(evicted.has_value());
        EXPECT_EQ(window_other.load(), 0);
        EXPECT_EQ(window_bad_bytes.load(), 0);
        EXPECT_GE(window_ok + window_invalid_key, kReaders * kReadsPerReader);
        if (read_pending) {
            EXPECT_EQ(window_invalid_key.load(), 0);
        } else {
            EXPECT_EQ(window_ok.load(), 0);
        }
        EXPECT_FALSE(fs::exists(DataFile(ids[0])));
        EXPECT_FALSE(Read(*backend, kOldKeys).status.has_value());
        EXPECT_EQ(backend->PendingEvictionReadableKeysForTest(), 0u);
    }
}

TEST_F(BucketEvictionReadTest, DefaultConfigKeepsTheOptInOff) {
    EXPECT_FALSE(BucketBackendConfig{}.read_pending_eviction);
}

}  // namespace
}  // namespace mooncake::test
