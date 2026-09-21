// RIG ONLY (INF-522): offload write throughput of BucketStorageBackend,
// to compare a tree without upstream #3601 (no fdatasync on the PosixFile
// write path) against one with it. Knobs come from the environment so the
// same binary sweeps shapes:
//   OFFLOAD_DIR (required), OFFLOAD_TOTAL_MB (default 2048),
//   OFFLOAD_OBJ_KB (default 1024), OFFLOAD_KEYS_PER_BATCH (default 16),
//   OFFLOAD_THREADS (default 1).
// Prints one machine-readable line: OFFLOAD_TPUT ... mb_per_s=<x>.
#include <gtest/gtest.h>
#include <glog/logging.h>

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <memory>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "storage_backend.h"

namespace mooncake {
namespace {

int64_t EnvInt(const char* name, int64_t fallback) {
    const char* v = std::getenv(name);
    return v ? std::atoll(v) : fallback;
}

TEST(OffloadTput, BucketBackendWrite) {
    const char* dir = std::getenv("OFFLOAD_DIR");
    ASSERT_NE(dir, nullptr) << "set OFFLOAD_DIR";
    const int64_t total_mb = EnvInt("OFFLOAD_TOTAL_MB", 2048);
    const int64_t obj_kb = EnvInt("OFFLOAD_OBJ_KB", 1024);
    const int64_t keys_per_batch = EnvInt("OFFLOAD_KEYS_PER_BATCH", 16);
    const int64_t threads = EnvInt("OFFLOAD_THREADS", 1);
    std::filesystem::remove_all(dir);
    std::filesystem::create_directories(dir);

    FileStorageConfig config;
    config.storage_filepath = dir;
    BucketBackendConfig bucket_config;
    BucketStorageBackend backend(config, bucket_config);
    ASSERT_TRUE(backend.Init());

    const size_t obj_bytes = static_cast<size_t>(obj_kb) * 1024;
    const int64_t total_objs = total_mb * 1024 / obj_kb;
    const int64_t batches = total_objs / keys_per_batch;
    std::vector<char> payload(obj_bytes);
    for (size_t i = 0; i < obj_bytes; ++i) payload[i] = static_cast<char>(i * 131);

    std::atomic<int64_t> next{0}, failed{0}, written{0};
    auto worker = [&](int tid) {
        for (;;) {
            int64_t b = next.fetch_add(1);
            if (b >= batches) return;
            std::unordered_map<std::string, std::vector<Slice>> batch;
            for (int64_t k = 0; k < keys_per_batch; ++k) {
                batch.emplace("t" + std::to_string(tid) + "_b" +
                                  std::to_string(b) + "_k" + std::to_string(k),
                              std::vector<Slice>{Slice{payload.data(), obj_bytes}});
            }
            auto r = backend.BatchOffload(
                batch, [](const std::vector<std::string>&,
                          std::vector<StorageObjectMetadata>&) {
                    return ErrorCode::OK;
                });
            if (!r) failed.fetch_add(1);
            else written.fetch_add(keys_per_batch);
        }
    };
    const auto t0 = std::chrono::steady_clock::now();
    std::vector<std::thread> ts;
    for (int i = 0; i < threads; ++i) ts.emplace_back(worker, i);
    for (auto& t : ts) t.join();
    const double sec = std::chrono::duration<double>(
                           std::chrono::steady_clock::now() - t0)
                           .count();
    const double mb = static_cast<double>(written.load()) * obj_kb / 1024.0;
    printf("OFFLOAD_TPUT total_mb=%.0f obj_kb=%lld keys_per_batch=%lld "
           "threads=%lld batches=%lld failed_batches=%lld sec=%.3f "
           "mb_per_s=%.1f\n",
           mb, (long long)obj_kb, (long long)keys_per_batch,
           (long long)threads, (long long)batches, (long long)failed.load(),
           sec, mb / sec);
    EXPECT_EQ(failed.load(), 0);
    std::filesystem::remove_all(dir);
}

}  // namespace
}  // namespace mooncake
