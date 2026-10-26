#include "storage_backend.h"

#include <cmath>
#include <filesystem>
#include <locale>
#include <sstream>
#include <string>
#include <sys/stat.h>
#include <unistd.h>

#include "bool_parser.h"
#include "environ.h"
#include "environment_variables.h"

namespace mooncake {

namespace {

double ParseDiskEvictionRatioOr(const std::string& raw_value,
                                double default_value) {
    if (raw_value.empty()) {
        return default_value;
    }

    std::istringstream stream(raw_value);
    stream.imbue(std::locale::classic());

    double value = 0.0;
    stream >> value;
    if (stream.fail()) {
        return default_value;
    }
    if (!stream.eof() || !std::isfinite(value) || value <= 0.0 || value > 1.0) {
        return default_value;
    }
    return value;
}

bool ParseStrictBoolOr(const std::string& raw_value, bool default_value) {
    return TryParseBool(raw_value, {.token_set = BoolTokenSet::kTrueFalse,
                                    .trim_ascii_whitespace = false})
        .value_or(default_value);
}

}  // namespace

FileStorageConfig FileStorageConfig::FromEnvironment() {
    FileStorageConfig config;
    using Variables = FileStorageEnvironmentVariables;

    const auto storage_backend_descriptor =
        Environ::ReadOr(Variables::MOONCAKE_OFFLOAD_STORAGE_BACKEND_DESCRIPTOR,
                        std::string{"bucket_storage_backend"});

    if (storage_backend_descriptor == "bucket_storage_backend") {
        config.storage_backend_type = StorageBackendType::kBucket;
    } else if (storage_backend_descriptor == "file_per_key_storage_backend") {
        config.storage_backend_type = StorageBackendType::kFilePerKey;
    } else if (storage_backend_descriptor ==
               "offset_allocator_storage_backend") {
        config.storage_backend_type = StorageBackendType::kOffsetAllocator;
    } else if (storage_backend_descriptor == "distributed_storage_backend") {
        config.storage_backend_type = StorageBackendType::kDistributed;
    } else if (storage_backend_descriptor == "nvme_kv_storage_backend") {
        config.storage_backend_type = StorageBackendType::kNvmeKv;
    } else {
        LOG(ERROR) << "Unknown storage backend.";
    }

    config.storage_filepath = Environ::ReadOr(
        Variables::MOONCAKE_OFFLOAD_FILE_STORAGE_PATH, config.storage_filepath);

    config.local_buffer_size =
        Environ::ReadOr(Variables::MOONCAKE_OFFLOAD_LOCAL_BUFFER_SIZE_BYTES,
                        config.local_buffer_size);

    config.pinned_restore_arena_size =
        Environ::ReadOr(Variables::MC_STORE_PINNED_RESTORE_ARENA_SIZE_BYTES,
                        config.pinned_restore_arena_size);

    const auto legacy_scanmeta_iterator_keys_limit =
        Environ::ReadOr(Variables::MOONCAKE_SCANMETA_ITERATOR_KEYS_LIMIT,
                        config.scanmeta_iterator_keys_limit);
    config.scanmeta_iterator_keys_limit = Environ::ReadOr(
        Variables::MOONCAKE_OFFLOAD_SCANMETA_ITERATOR_KEYS_LIMIT,
        legacy_scanmeta_iterator_keys_limit);

    config.total_keys_limit = Environ::ReadOr(
        Variables::MOONCAKE_OFFLOAD_TOTAL_KEYS_LIMIT, config.total_keys_limit);

    config.total_size_limit =
        Environ::ReadOr(Variables::MOONCAKE_OFFLOAD_TOTAL_SIZE_LIMIT_BYTES,
                        config.total_size_limit);

    config.heartbeat_interval_seconds =
        Environ::ReadOr(Variables::MOONCAKE_OFFLOAD_HEARTBEAT_INTERVAL_SECONDS,
                        config.heartbeat_interval_seconds);
    config.client_buffer_gc_interval_seconds = Environ::ReadOr(
        Variables::MOONCAKE_OFFLOAD_CLIENT_BUFFER_GC_INTERVAL_SECONDS,
        config.client_buffer_gc_interval_seconds);

    config.client_buffer_gc_ttl_ms =
        Environ::ReadOr(Variables::MOONCAKE_OFFLOAD_CLIENT_BUFFER_GC_TTL_MS,
                        config.client_buffer_gc_ttl_ms);

    const auto enable_disk_watermark_eviction = Environ::ReadOr(
        Variables::MOONCAKE_OFFLOAD_ENABLE_DISK_WATERMARK_EVICTION,
        std::string{config.enable_disk_watermark_eviction ? "true" : "false"});
    config.enable_disk_watermark_eviction = ParseStrictBoolOr(
        enable_disk_watermark_eviction, config.enable_disk_watermark_eviction);

    const auto high_watermark_ratio = Environ::ReadOr(
        Variables::MOONCAKE_OFFLOAD_DISK_EVICTION_HIGH_WATERMARK_RATIO,
        std::string{});
    const auto high_watermark_ratio_or_alias =
        high_watermark_ratio.empty()
            ? Environ::ReadOr(
                  Variables::MOONCAKE_DISK_EVICTION_HIGH_WATERMARK_RATIO,
                  std::string{})
            : high_watermark_ratio;
    config.disk_eviction_high_watermark_ratio =
        ParseDiskEvictionRatioOr(high_watermark_ratio_or_alias,
                                 config.disk_eviction_high_watermark_ratio);

    const auto low_watermark_ratio = Environ::ReadOr(
        Variables::MOONCAKE_OFFLOAD_DISK_EVICTION_LOW_WATERMARK_RATIO,
        std::string{});
    const auto low_watermark_ratio_or_alias =
        low_watermark_ratio.empty()
            ? Environ::ReadOr(
                  Variables::MOONCAKE_DISK_EVICTION_LOW_WATERMARK_RATIO,
                  std::string{})
            : low_watermark_ratio;
    config.disk_eviction_low_watermark_ratio = ParseDiskEvictionRatioOr(
        low_watermark_ratio_or_alias, config.disk_eviction_low_watermark_ratio);

    const auto legacy_use_uring =
        Environ::ReadOr(Variables::MOONCAKE_USE_URING, std::string{"false"});
    const auto use_uring = Environ::ReadOr(
        Variables::MOONCAKE_OFFLOAD_USE_URING, legacy_use_uring);
    config.use_uring = ParseStrictBoolOr(use_uring, false);

    config.offload_read_deadline_ms =
        Environ::ReadOr(Variables::MOONCAKE_OFFLOAD_READ_DEADLINE_MS,
                        config.offload_read_deadline_ms);
    config.offload_read_shed_ms = Environ::ReadOr(
        Variables::MOONCAKE_OFFLOAD_READ_SHED_MS, config.offload_read_shed_ms);
    config.offload_read_threads = Environ::ReadOr(
        Variables::MOONCAKE_OFFLOAD_READ_THREADS, config.offload_read_threads);
    config.offload_read_miss_evict = ParseStrictBoolOr(
        Environ::ReadOr(Variables::MOONCAKE_OFFLOAD_READ_MISS_EVICT,
                        std::string{}),
        config.offload_read_miss_evict);
    config.offload_read_parallelism =
        Environ::ReadOr(Variables::MOONCAKE_OFFLOAD_READ_PARALLELISM,
                        config.offload_read_parallelism);
    config.offload_read_io_threads =
        Environ::ReadOr(Variables::MOONCAKE_OFFLOAD_READ_IO_THREADS,
                        config.offload_read_io_threads);

    return config;
}

bool FileStorageConfig::ValidatePath(std::string path) const {
    if (path.empty()) {
        LOG(ERROR) << "FileStorageConfig: storage_filepath is invalid";
        return false;
    }
    namespace fs = std::filesystem;
    // 1. Must be an absolute path
    if (!fs::path(path).is_absolute()) {
        LOG(ERROR)
            << "FileStorageConfig: storage_filepath must be an absolute path: "
            << path;
        return false;
    }

    // 2. Check if the path contains ".." components that could lead to path
    // traversal (static check)
    fs::path p(path);
    for (const auto& component : p) {
        if (component == "..") {
            LOG(ERROR) << "FileStorageConfig: path traversal is not allowed: "
                       << path;
            return false;
        }
    }

    struct stat stat_buf;

    // 3. Use stat() to check if the path exists
    if (::stat(path.c_str(), &stat_buf) != 0) {
        LOG(ERROR) << "FileStorageConfig: storage_filepath does not exist: "
                   << path;
        return false;
    }
    // Path exists — check if it is a directory
    if (!S_ISDIR(stat_buf.st_mode)) {
        LOG(ERROR) << "FileStorageConfig: storage_filepath is not a directory: "
                   << path;
        return false;
    }

    // (Optional) Check write permission
    if (::access(path.c_str(), W_OK) != 0) {
        LOG(ERROR) << "FileStorageConfig: no write permission on directory: "
                   << path;
        return false;
    }

    // 4. Additional security: prevent symlink bypass (optional)
    // Use lstat to avoid automatic dereferencing of symbolic links
    struct stat lstat_buf;
    if (::lstat(path.c_str(), &lstat_buf) == 0) {
        if (S_ISLNK(lstat_buf.st_mode)) {
            LOG(ERROR) << "FileStorageConfig: symbolic link is not allowed: "
                       << path;
            return false;
        }
    }

    return true;
}

bool FileStorageConfig::Validate() const {
    if (!ValidatePath(storage_filepath)) {
        return false;
    }
    if (total_keys_limit <= 0) {
        LOG(ERROR) << "FileStorageConfig: total_keys_limit must > 0";
        return false;
    }
    if (total_size_limit == 0) {
        LOG(ERROR) << "FileStorageConfig: total_size_limit should not be zero";
        return false;
    }
    if (pinned_restore_arena_size < 0) {
        LOG(ERROR) << "FileStorageConfig: pinned_restore_arena_size must be "
                      "non-negative";
        return false;
    }
    if (heartbeat_interval_seconds <= 0) {
        LOG(ERROR) << "FileStorageConfig: heartbeat_interval_seconds must > 0";
        return false;
    }
    if (disk_eviction_low_watermark_ratio <= 0.0 ||
        disk_eviction_low_watermark_ratio > 1.0) {
        LOG(ERROR) << "FileStorageConfig: "
                   << "disk_eviction_low_watermark_ratio must be in (0, 1]";
        return false;
    }
    if (disk_eviction_high_watermark_ratio <= 0.0 ||
        disk_eviction_high_watermark_ratio > 1.0) {
        LOG(ERROR) << "FileStorageConfig: "
                   << "disk_eviction_high_watermark_ratio must be in (0, 1]";
        return false;
    }
    if (disk_eviction_low_watermark_ratio >=
        disk_eviction_high_watermark_ratio) {
        LOG(ERROR) << "FileStorageConfig: "
                   << "disk_eviction_low_watermark_ratio must be lower than "
                   << "disk_eviction_high_watermark_ratio";
        return false;
    }
    constexpr uint32_t kMaxReadParallelism = 256;
    constexpr uint32_t kMaxReadIoThreads = 1024;
    if (offload_read_parallelism == 0 ||
        offload_read_parallelism > kMaxReadParallelism) {
        LOG(ERROR) << "FileStorageConfig: offload_read_parallelism must be in "
                      "[1, "
                   << kMaxReadParallelism << "]";
        return false;
    }
    if (offload_read_parallelism > 1 &&
        (offload_read_io_threads == 0 ||
         offload_read_io_threads > kMaxReadIoThreads)) {
        LOG(ERROR) << "FileStorageConfig: offload_read_io_threads must be in "
                      "[1, "
                   << kMaxReadIoThreads
                   << "] when offload_read_parallelism is above 1";
        return false;
    }
    if (offload_read_deadline_ms > 0) {
        constexpr uint64_t kMaxReadGuardMs = 3'600'000;  // 1 hour
        constexpr uint32_t kMaxReadGuardThreads = 1024;
        if (offload_read_deadline_ms > kMaxReadGuardMs ||
            offload_read_shed_ms > kMaxReadGuardMs) {
            LOG(ERROR) << "FileStorageConfig: offload_read_deadline_ms and "
                          "offload_read_shed_ms must be at most "
                       << kMaxReadGuardMs;
            return false;
        }
        if (offload_read_threads == 0 ||
            offload_read_threads > kMaxReadGuardThreads) {
            LOG(ERROR) << "FileStorageConfig: offload_read_threads must be in "
                          "[1, "
                       << kMaxReadGuardThreads
                       << "] when offload_read_deadline_ms is set";
            return false;
        }
        if (offload_read_deadline_ms >= client_buffer_gc_ttl_ms) {
            // The reader already fails a read that took this long
            // (OBJECT_HAS_LEASE), so the guard would never answer first.
            LOG(WARNING) << "FileStorageConfig: offload_read_deadline_ms ("
                         << offload_read_deadline_ms
                         << ") is not below client_buffer_gc_ttl_ms ("
                         << client_buffer_gc_ttl_ms << ")";
        }
    }
    if (offload_read_miss_evict && offload_read_deadline_ms == 0) {
        // Only the read guard reports read misses.
        LOG(WARNING) << "FileStorageConfig: offload_read_miss_evict has no "
                        "effect without offload_read_deadline_ms";
    }
    return true;
}

}  // namespace mooncake
