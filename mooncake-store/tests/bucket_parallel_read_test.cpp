// Parallel reads within one BucketStorageBackend::BatchLoad
// (offload_read_parallelism > 1, POSIX read path).
//
// The default reads a batch's keys one after another, so a batch takes about
// "keys x device latency". These tests pin that the parallel path returns the
// same bytes, keeps at most `parallelism` reads on the disk per batch, never
// returns while one of its reads still runs (the caller frees the buffers
// right after a failure), and makes progress when the helper pool is busy.

#include <fcntl.h>
#include <gtest/gtest.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <functional>
#include <iterator>
#include <mutex>
#include <memory>
#include <random>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "storage_backend.h"

namespace fs = std::filesystem;

namespace mooncake::test {
namespace {

std::string ValueFor(const std::string& key, size_t size) {
    std::string value(size, '\0');
    const size_t seed = std::hash<std::string>{}(key);
    for (size_t i = 0; i < value.size(); ++i) {
        value[i] = static_cast<char>('a' + (seed + i * 7) % 26);
    }
    return value;
}

// Counts reads in flight (inside the before-read hook) and their maximum.
struct ReadTracker {
    std::atomic<int> in_flight{0};
    std::atomic<int> max_in_flight{0};
    std::atomic<int> reads{0};

    void Enter() {
        const int now = in_flight.fetch_add(1) + 1;
        int seen = max_in_flight.load();
        while (now > seen && !max_in_flight.compare_exchange_weak(seen, now)) {
        }
        reads.fetch_add(1);
    }
    void Exit() { in_flight.fetch_sub(1); }
};

class BucketParallelReadTest : public ::testing::Test {
   protected:
    void SetUp() override {
        FLAGS_logtostderr = true;
        const auto* info =
            ::testing::UnitTest::GetInstance()->current_test_info();
        data_path_ = (fs::current_path() / "bucket_parallel_read_test_data" /
                      info->name())
                         .string();
        fs::remove_all(data_path_);
        fs::create_directories(data_path_);
    }

    void TearDown() override { fs::remove_all(data_path_); }

    std::unique_ptr<BucketStorageBackend> MakeBackend(
        uint32_t parallelism, uint32_t io_threads = 64) {
        FileStorageConfig config;
        config.storage_filepath = data_path_;
        config.offload_read_parallelism = parallelism;
        config.offload_read_io_threads = io_threads;
        BucketBackendConfig bucket_config;
        auto backend =
            std::make_unique<BucketStorageBackend>(config, bucket_config);
        EXPECT_TRUE(backend->Init().has_value());
        return backend;
    }

    // Writes one bucket holding `keys` (value size from `size_of`).
    static void Offload(BucketStorageBackend& backend,
                        const std::vector<std::string>& keys,
                        const std::function<size_t(size_t)>& size_of) {
        std::vector<std::string> values;
        values.reserve(keys.size());
        std::unordered_map<std::string, std::vector<Slice>> batch;
        for (size_t i = 0; i < keys.size(); ++i) {
            values.push_back(ValueFor(keys[i], size_of(i)));
            batch.emplace(keys[i],
                          std::vector<Slice>{Slice{values.back().data(),
                                                   values.back().size()}});
        }
        auto result = backend.BatchOffload(
            batch,
            [](const std::vector<std::string>&,
               std::vector<StorageObjectMetadata>&) { return ErrorCode::OK; });
        ASSERT_TRUE(result.has_value())
            << (result ? ErrorCode::OK : result.error());
    }

    // Three buckets of `per_bucket` keys with sizes between 1 and 20 KiB.
    static std::vector<std::pair<std::string, size_t>> FillBuckets(
        BucketStorageBackend& backend, size_t per_bucket = 20) {
        std::vector<std::pair<std::string, size_t>> all;
        for (int b = 0; b < 3; ++b) {
            std::vector<std::string> keys;
            for (size_t i = 0; i < per_bucket; ++i) {
                keys.push_back("b" + std::to_string(b) + "_k" +
                               std::to_string(i));
            }
            auto size_of = [b](size_t i) {
                return 1024 + ((i * 3 + b) % 20) * 1000;
            };
            Offload(backend, keys, size_of);
            for (size_t i = 0; i < keys.size(); ++i) {
                all.emplace_back(keys[i], size_of(i));
            }
        }
        return all;
    }

    struct ReadResult {
        tl::expected<void, ErrorCode> status;
        bool bytes_match = false;
    };

    static ReadResult Read(
        BucketStorageBackend& backend,
        const std::vector<std::pair<std::string, size_t>>& keys) {
        std::vector<std::unique_ptr<char[]>> buffers;
        std::unordered_map<std::string, Slice> slices;
        for (const auto& [key, size] : keys) {
            buffers.push_back(std::make_unique<char[]>(size));
            std::memset(buffers.back().get(), 0, size);
            slices.emplace(key, Slice{buffers.back().get(), size});
        }
        ReadResult result{backend.BatchLoad(slices), false};
        if (result.status) {
            result.bytes_match = true;
            for (const auto& [key, size] : keys) {
                const auto& slice = slices.at(key);
                if (std::string(static_cast<char*>(slice.ptr), slice.size) !=
                    ValueFor(key, size)) {
                    result.bytes_match = false;
                }
            }
        }
        return result;
    }

    // Data file of the bucket that holds `key`, or empty.
    fs::path BucketFileOf(BucketStorageBackend& backend,
                          const std::string& key) const {
        std::unordered_map<std::string, StorageObjectMetadata> metadata;
        auto res = backend.BatchQuery({key}, metadata);
        if (!res || !metadata.contains(key)) return {};
        return fs::path(data_path_) /
               (std::to_string(metadata.at(key).bucket_id) + ".bucket");
    }

    std::string data_path_;
};

TEST_F(BucketParallelReadTest, ParallelReadReturnsTheSameBytes) {
    for (uint32_t parallelism : {1u, 2u, 8u, 64u}) {
        SCOPED_TRACE("parallelism=" + std::to_string(parallelism));
        fs::remove_all(data_path_);
        fs::create_directories(data_path_);
        auto backend = MakeBackend(parallelism);
        auto keys = FillBuckets(*backend);
        auto all = Read(*backend, keys);
        ASSERT_TRUE(all.status.has_value()) << all.status.error();
        EXPECT_TRUE(all.bytes_match);
        // A single key takes the one-after-another loop.
        auto one = Read(*backend, {keys[5]});
        ASSERT_TRUE(one.status.has_value());
        EXPECT_TRUE(one.bytes_match);
    }
}

TEST_F(BucketParallelReadTest, DefaultReadsOneKeyAtATimeOnTheCaller) {
    auto backend = MakeBackend(1);
    auto keys = FillBuckets(*backend, 6);
    ReadTracker tracker;
    const auto caller = std::this_thread::get_id();
    std::atomic<bool> other_thread{false};
    backend->SetBeforeKeyReadHookForTest([&](const std::string&) {
        tracker.Enter();
        if (std::this_thread::get_id() != caller) other_thread = true;
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
        tracker.Exit();
    });
    auto result = Read(*backend, keys);
    ASSERT_TRUE(result.status.has_value());
    EXPECT_TRUE(result.bytes_match);
    EXPECT_EQ(tracker.reads.load(), static_cast<int>(keys.size()));
    EXPECT_EQ(tracker.max_in_flight.load(), 1);
    EXPECT_FALSE(other_thread.load());
}

TEST_F(BucketParallelReadTest, ParallelReadOverlapsSlowReadsUpToTheLimit) {
    constexpr auto kReadDelay = std::chrono::milliseconds(40);
    constexpr uint32_t kParallelism = 8;
    auto backend = MakeBackend(kParallelism);
    auto keys = FillBuckets(*backend, 8);  // 24 keys in 3 buckets
    ReadTracker tracker;
    backend->SetBeforeKeyReadHookForTest([&](const std::string&) {
        tracker.Enter();
        std::this_thread::sleep_for(kReadDelay);
        tracker.Exit();
    });
    const auto start = std::chrono::steady_clock::now();
    auto result = Read(*backend, keys);
    const auto elapsed = std::chrono::steady_clock::now() - start;
    ASSERT_TRUE(result.status.has_value());
    EXPECT_TRUE(result.bytes_match);
    EXPECT_EQ(tracker.reads.load(), static_cast<int>(keys.size()));
    EXPECT_LE(tracker.max_in_flight.load(), static_cast<int>(kParallelism));
    EXPECT_GE(tracker.max_in_flight.load(), 2);
    // One after another would take 24 x 40 ms = 960 ms; 8 at a time about
    // 120 ms. The bound leaves room for a slow test machine.
    EXPECT_LT(elapsed, keys.size() * kReadDelay / 2);
}

TEST_F(BucketParallelReadTest, FailureWaitsForStartedReadsAndStopsNewOnes) {
    auto backend = MakeBackend(4);
    auto keys = FillBuckets(*backend, 20);
    // Cut the middle bucket's data file (b1_*): its keys read short.
    const auto cut = BucketFileOf(*backend, "b1_k0");
    ASSERT_FALSE(cut.empty());
    fs::resize_file(cut, 100);

    ReadTracker tracker;
    backend->SetBeforeKeyReadHookForTest([&](const std::string&) {
        tracker.Enter();
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        tracker.Exit();
    });
    auto result = Read(*backend, keys);
    ASSERT_FALSE(result.status.has_value());
    EXPECT_EQ(result.status.error(), ErrorCode::FILE_READ_FAIL);
    // Every read of this batch that started has left its hook (reads start
    // right after it); that the buffers are not written after BatchLoad
    // returns is what an ASan build of this test checks.
    EXPECT_EQ(tracker.in_flight.load(), 0);
    // Reads stop after the failure instead of reading all 60 keys.
    EXPECT_LT(tracker.reads.load(), static_cast<int>(keys.size()));

    // A later batch on the healthy buckets still reads.
    std::vector<std::pair<std::string, size_t>> healthy;
    for (const auto& entry : keys) {
        if (entry.first.rfind("b1_", 0) != 0) healthy.push_back(entry);
    }
    backend->SetBeforeKeyReadHookForTest(nullptr);
    auto ok = Read(*backend, healthy);
    ASSERT_TRUE(ok.status.has_value()) << ok.status.error();
    EXPECT_TRUE(ok.bytes_match);
}

TEST_F(BucketParallelReadTest, MissingFileFailsLikeTheSerialPath) {
    for (uint32_t parallelism : {1u, 4u}) {
        SCOPED_TRACE("parallelism=" + std::to_string(parallelism));
        fs::remove_all(data_path_);
        fs::create_directories(data_path_);
        auto backend = MakeBackend(parallelism);
        auto keys = FillBuckets(*backend, 4);
        const auto lost = BucketFileOf(*backend, "b2_k0");
        ASSERT_FALSE(lost.empty());
        fs::remove(lost);
        auto result = Read(*backend, keys);
        ASSERT_FALSE(result.status.has_value());
        EXPECT_EQ(result.status.error(), ErrorCode::FILE_OPEN_FAIL);
    }
}

TEST_F(BucketParallelReadTest, BusyPoolStillCompletesEveryBatch) {
    // Two helper threads for 8 callers that each want 7 helpers: most helper
    // tasks start late or after their batch is done, and the callers read
    // their own keys meanwhile.
    auto backend = MakeBackend(8, 2);
    auto keys = FillBuckets(*backend, 20);
    backend->SetBeforeKeyReadHookForTest([](const std::string&) {
        std::this_thread::sleep_for(std::chrono::microseconds(200));
    });
    std::atomic<int> failures{0};
    std::vector<std::thread> callers;
    for (int t = 0; t < 8; ++t) {
        callers.emplace_back([&, t] {
            std::mt19937 rng(t);
            for (int round = 0; round < 15; ++round) {
                auto subset = keys;
                std::shuffle(subset.begin(), subset.end(), rng);
                subset.resize(5 + rng() % 40);
                auto result = Read(*backend, subset);
                if (!result.status || !result.bytes_match) failures++;
            }
        });
    }
    for (auto& caller : callers) caller.join();
    EXPECT_EQ(failures.load(), 0);
    // Late helper tasks find no work and must not touch freed batches.
    backend.reset();
}

// Microbenchmark, not run by default. On the target disk:
//   MOONCAKE_READ_BENCH_DIR=/path/on/disk ./bucket_parallel_read_test \
//     --gtest_also_run_disabled_tests --gtest_filter='*ReadBench*'
// Writes MOONCAKE_READ_BENCH_KEYS (default 1200) keys of 455680 B (one rig
// KV object), drops them from the page cache before every round, and times
// batches of 60 keys (60 keys of one bucket, or keys spread over buckets) for
// each parallelism, with 1 and 4 callers that read disjoint keys at once.
// Buffers are allocated before and data checked after the timed part; gbps is
// bytes read over the round's wall time from start to the last batch done.
// MOONCAKE_READ_BENCH_DELAY_US adds that much sleep before every key read, as
// a slow device that keeps serving reads in parallel (like dm-delay) would.
TEST(BucketParallelReadBench, DISABLED_ReadBench) {
    const char* dir_env = std::getenv("MOONCAKE_READ_BENCH_DIR");
    ASSERT_NE(dir_env, nullptr) << "set MOONCAKE_READ_BENCH_DIR";
    const char* keys_env = std::getenv("MOONCAKE_READ_BENCH_KEYS");
    const size_t total_keys =
        keys_env ? std::strtoull(keys_env, nullptr, 10) : 1200;
    const char* rounds_env = std::getenv("MOONCAKE_READ_BENCH_ROUNDS");
    const int rounds = rounds_env ? std::atoi(rounds_env) : 30;
    const char* delay_env = std::getenv("MOONCAKE_READ_BENCH_DELAY_US");
    const auto delay = std::chrono::microseconds(
        delay_env ? std::strtoull(delay_env, nullptr, 10) : 0);
    constexpr size_t kValueSize = 455680;
    constexpr size_t kBatch = 60;
    constexpr size_t kKeysPerBucket = 120;
    constexpr int kMaxCallers = 4;
    ASSERT_GE(total_keys, kBatch * kMaxCallers * 2);
    const fs::path dir = fs::path(dir_env) / "bucket_parallel_read_bench";
    fs::remove_all(dir);
    fs::create_directories(dir);

    std::vector<std::string> keys;
    {
        FileStorageConfig config;
        config.storage_filepath = dir.string();
        BucketStorageBackend writer(config, BucketBackendConfig{});
        ASSERT_TRUE(writer.Init().has_value());
        for (size_t first = 0; first < total_keys; first += kKeysPerBucket) {
            std::vector<std::string> bucket_keys;
            std::vector<std::string> values;
            std::unordered_map<std::string, std::vector<Slice>> batch;
            values.reserve(kKeysPerBucket);
            for (size_t i = first;
                 i < std::min(total_keys, first + kKeysPerBucket); ++i) {
                bucket_keys.push_back("k" + std::to_string(i));
                values.push_back(ValueFor(bucket_keys.back(), kValueSize));
                batch.emplace(bucket_keys.back(),
                              std::vector<Slice>{Slice{values.back().data(),
                                                       values.back().size()}});
            }
            auto res = writer.BatchOffload(
                batch, [](const std::vector<std::string>&,
                          std::vector<StorageObjectMetadata>&) {
                    return ErrorCode::OK;
                });
            ASSERT_TRUE(res.has_value());
            keys.insert(keys.end(), bucket_keys.begin(), bucket_keys.end());
        }
    }
    auto drop_page_cache = [&dir] {
        for (const auto& entry : fs::directory_iterator(dir)) {
            if (entry.path().extension() != ".bucket") continue;
            int fd = ::open(entry.path().c_str(), O_RDONLY | O_CLOEXEC);
            if (fd < 0) continue;
            ::fdatasync(fd);
            ::posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);
            ::close(fd);
        }
    };

    for (const bool spread : {false, true}) {
        for (const int callers : {1, 4}) {
            for (const uint32_t parallelism : {1u, 4u, 16u, 32u}) {
                FileStorageConfig config;
                config.storage_filepath = dir.string();
                config.offload_read_parallelism = parallelism;
                config.offload_read_io_threads = 64;
                BucketStorageBackend backend(config, BucketBackendConfig{});
                ASSERT_TRUE(backend.Init().has_value());
                if (delay.count() > 0) {
                    backend.SetBeforeKeyReadHookForTest(
                        [delay](const std::string&) {
                            std::this_thread::sleep_for(delay);
                        });
                }
                std::vector<double> latencies_ms;
                std::atomic<int> errors{0};
                double total_seconds = 0;
                for (int round = 0; round < rounds; ++round) {
                    // Pick disjoint keys per caller and allocate before timing.
                    std::mt19937 rng(round * 131 + callers * 7 + parallelism);
                    const size_t ranges = keys.size() / kBatch;
                    const size_t base = rng() % ranges;
                    std::vector<std::unordered_map<std::string, Slice>> slices(
                        callers);
                    std::vector<std::vector<std::unique_ptr<char[]>>> buffers(
                        callers);
                    for (int c = 0; c < callers; ++c) {
                        std::vector<std::string> batch_keys;
                        if (spread) {
                            std::vector<std::string> own;
                            for (size_t i = c; i < keys.size(); i += callers) {
                                own.push_back(keys[i]);
                            }
                            std::sample(own.begin(), own.end(),
                                        std::back_inserter(batch_keys), kBatch,
                                        rng);
                        } else {
                            const size_t first = ((base + c) % ranges) * kBatch;
                            batch_keys.assign(keys.begin() + first,
                                              keys.begin() + first + kBatch);
                        }
                        for (const auto& key : batch_keys) {
                            buffers[c].push_back(
                                std::make_unique<char[]>(kValueSize));
                            slices[c].emplace(
                                key,
                                Slice{buffers[c].back().get(), kValueSize});
                        }
                    }
                    drop_page_cache();
                    std::atomic<bool> go{false};
                    std::vector<double> round_ms(callers, 0);
                    std::vector<std::thread> threads;
                    for (int c = 0; c < callers; ++c) {
                        threads.emplace_back([&, c] {
                            while (!go.load(std::memory_order_acquire)) {
                            }
                            const auto start = std::chrono::steady_clock::now();
                            if (!backend.BatchLoad(slices[c])) errors++;
                            round_ms[c] =
                                std::chrono::duration<double, std::milli>(
                                    std::chrono::steady_clock::now() - start)
                                    .count();
                        });
                    }
                    const auto round_start = std::chrono::steady_clock::now();
                    go.store(true, std::memory_order_release);
                    for (auto& thread : threads) thread.join();
                    total_seconds +=
                        std::chrono::duration<double>(
                            std::chrono::steady_clock::now() - round_start)
                            .count();
                    latencies_ms.insert(latencies_ms.end(), round_ms.begin(),
                                        round_ms.end());
                    // Check every key to catch wrong offsets.
                    for (int c = 0; c < callers; ++c) {
                        for (const auto& [key, slice] : slices[c]) {
                            if (std::memcmp(slice.ptr,
                                            ValueFor(key, kValueSize).data(),
                                            kValueSize) != 0) {
                                errors++;
                            }
                        }
                    }
                }
                std::sort(latencies_ms.begin(), latencies_ms.end());
                const double p50 = latencies_ms[latencies_ms.size() / 2];
                const double p99 = latencies_ms[std::min(
                    latencies_ms.size() - 1, latencies_ms.size() * 99 / 100)];
                const double bytes =
                    static_cast<double>(rounds) * callers * kBatch * kValueSize;
                std::printf(
                    "BENCH pattern=%s callers=%d parallelism=%u rounds=%d "
                    "delay_us=%lld p50_ms=%.2f p99_ms=%.2f gbps=%.3f "
                    "errors=%d\n",
                    spread ? "spread" : "contig", callers, parallelism, rounds,
                    static_cast<long long>(delay.count()), p50, p99,
                    bytes / total_seconds / 1e9, errors.load());
                EXPECT_EQ(errors.load(), 0);
            }
        }
    }
    fs::remove_all(dir);
}

}  // namespace
}  // namespace mooncake::test
