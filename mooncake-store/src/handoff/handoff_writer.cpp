#include "handoff/handoff_writer.h"

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <sstream>
#include <thread>
#include <vector>

#include <glog/logging.h>

#include "crc32c.h"
#include "ha/snapshot/batch_oplog/codec.h"
#include "ha/snapshot/batch_oplog/metadata.h"
#include "master_service.h"

namespace mooncake {

namespace {

namespace fs = std::filesystem;
using SteadyClock = std::chrono::steady_clock;

uint64_t NsSince(SteadyClock::time_point start) {
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(SteadyClock::now() -
                                                             start)
            .count());
}

uint64_t ProcStatusKiB(const char* field) {
    std::ifstream f("/proc/self/status");
    std::string line;
    const std::string prefix = std::string(field) + ":";
    while (std::getline(f, line)) {
        if (line.rfind(prefix, 0) == 0) {
            return std::strtoull(line.c_str() + prefix.size(), nullptr, 10);
        }
    }
    return 0;
}

std::string Errno(const std::string& what, const std::string& path) {
    return what + " " + path + ": " + std::strerror(errno);
}

bool WriteAll(int fd, const void* data, size_t size) {
    const auto* p = static_cast<const uint8_t*>(data);
    while (size > 0) {
        const ssize_t n = ::write(fd, p, size);
        if (n < 0 && errno == EINTR) continue;
        if (n <= 0) return false;
        p += n;
        size -= static_cast<size_t>(n);
    }
    return true;
}

// Starts writeback of a freshly written file without waiting.
void StartWriteback(int fd) {
#ifdef __linux__
    (void)::sync_file_range(fd, 0, 0, SYNC_FILE_RANGE_WRITE);
#else
    (void)fd;
#endif
}

// Waits for the file's data, makes it durable and drops it from page cache,
// so the writer does not leave gigabytes of dirty or cached pages behind.
bool FinishDurable(int fd) {
#ifdef __linux__
    (void)::sync_file_range(fd, 0, 0,
                            SYNC_FILE_RANGE_WAIT_BEFORE |
                                SYNC_FILE_RANGE_WRITE |
                                SYNC_FILE_RANGE_WAIT_AFTER);
#endif
    if (::fdatasync(fd) != 0) return false;
#ifdef __linux__
    (void)::posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);
#endif
    return true;
}

bool FsyncDir(const fs::path& dir) {
    const int fd = ::open(dir.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC);
    if (fd < 0) return false;
    const bool ok = ::fsync(fd) == 0;
    ::close(fd);
    return ok;
}

// Writes a small file completely (optionally durable) and returns its info.
tl::expected<handoff::BlobInfo, std::string> WriteBlob(
    const fs::path& root, const std::string& key, const void* data,
    size_t size, bool durable) {
    const fs::path path = root / key;
    std::error_code ec;
    fs::create_directories(path.parent_path(), ec);
    const int fd =
        ::open(path.c_str(), O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0644);
    if (fd < 0) return tl::make_unexpected(Errno("open", path.string()));
    if (!WriteAll(fd, data, size) || (durable && ::fdatasync(fd) != 0)) {
        const std::string err = Errno("write", path.string());
        ::close(fd);
        return tl::make_unexpected(err);
    }
    ::close(fd);
    return handoff::BlobInfo{key, size, Crc32cValue(data, size)};
}

}  // namespace

std::string HandoffWriteResult::ToJson() const {
    std::ostringstream o;
    o << "{\"objects\":" << stats.objects << ",\"replicas\":" << stats.replicas
      << ",\"skipped_processing\":" << stats.skipped_processing
      << ",\"skipped_no_exportable_replica\":"
      << stats.skipped_no_exportable_replica
      << ",\"dropped_non_memory_replicas\":"
      << stats.dropped_non_memory_replicas
      << ",\"dropped_incomplete_replicas\":"
      << stats.dropped_incomplete_replicas
      << ",\"dropped_unknown_segment_replicas\":"
      << stats.dropped_unknown_segment_replicas
      << ",\"quarantine_ranges\":" << stats.quarantine_ranges
      << ",\"quarantine_bytes\":" << stats.quarantine_bytes
      << ",\"total_bytes\":" << stats.total_bytes
      << ",\"workers\":" << stats.workers << ",\"chunks\":" << chunks
      << ",\"segments_ns\":" << segments_ns
      << ",\"chunks_wall_ns\":" << chunks_wall_ns
      << ",\"export_cpu_ns\":" << export_cpu_ns
      << ",\"encode_cpu_ns\":" << encode_cpu_ns
      << ",\"write_cpu_ns\":" << write_cpu_ns << ",\"tail_ns\":" << tail_ns
      << ",\"total_ns\":" << total_ns
      << ",\"peak_rss_kib\":" << stats.peak_rss_kib << "}";
    return o.str();
}

tl::expected<HandoffWriteResult, std::string> WriteHandoffSnapshot(
    MasterService& service, const HandoffWriteOptions& options) {
    const auto total_start = SteadyClock::now();
    HandoffWriteResult result;
    const fs::path root(options.dir);
    std::error_code ec;
    fs::create_directories(root, ec);
    if (ec) {
        return tl::make_unexpected("create " + options.dir + ": " +
                                   ec.message());
    }
    if (fs::exists(root / handoff::kHandoffManifestFile)) {
        return tl::make_unexpected("handoff manifest already exists in " +
                                   options.dir);
    }

    handoff::Manifest manifest;
    manifest.created_at_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch())
            .count();
    manifest.snapshot_id =
        ha::BuildBatchOpLogSnapshotId(0, std::max<int64_t>(1, manifest.created_at_ms));
    manifest.codec = handoff::CurrentCodecFingerprint();
    manifest.memory_allocator = "offset";
    manifest.multi_tenant = service.HandoffMultiTenant();

    auto t = SteadyClock::now();
    manifest.segments = service.ExportHandoffSegments();
    const handoff::SegmentLocator locator(manifest.segments);
    result.segments_ns = NsSince(t);

    // ---- Object chunks: one per metadata shard, written as they are made.
    const size_t num_chunks = MasterService::HandoffShardCount();
    const size_t workers =
        std::max<size_t>(1, std::min(options.workers, num_chunks));
    struct WorkerState {
        handoff::WriterStats stats;
        std::vector<handoff::QuarantineRange> quarantine;
        uint64_t export_ns{0}, encode_ns{0}, write_ns{0};
        int previous_fd{-1};
        std::string error;
    };
    std::vector<WorkerState> states(workers);
    std::vector<handoff::ChunkInfo> chunks(num_chunks);
    std::atomic<size_t> next{0};
    std::atomic<bool> failed{false};
    auto body = [&](size_t w) {
        auto& st = states[w];
        auto finish_previous = [&]() {
            if (st.previous_fd < 0) return true;
            const bool ok = !options.durable || FinishDurable(st.previous_fd);
            ::close(st.previous_fd);
            st.previous_fd = -1;
            return ok;
        };
        for (;;) {
            const size_t shard = next.fetch_add(1);
            if (shard >= num_chunks || failed.load()) break;
            auto ts = SteadyClock::now();
            std::vector<StandbyObjectEntry> objects;
            service.ExportHandoffShard(shard, locator, objects, st.quarantine,
                                       st.stats);
            st.export_ns += NsSince(ts);

            ts = SteadyClock::now();
            const uint64_t count = objects.size();
            const auto encoded =
                EncodeBatchOpLogSnapshotObjectChunk(shard, std::move(objects));
            const uint32_t crc = Crc32cValue(encoded.data(), encoded.size());
            st.encode_ns += NsSince(ts);

            ts = SteadyClock::now();
            const std::string key = ha::BuildBatchOpLogSnapshotObjectChunkKey(
                handoff::kHandoffSnapshotRoot, manifest.snapshot_id, shard);
            const fs::path path = root / key;
            std::error_code dir_ec;
            fs::create_directories(path.parent_path(), dir_ec);
            const int fd = ::open(path.c_str(),
                                  O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC,
                                  0644);
            if (fd < 0 || !WriteAll(fd, encoded.data(), encoded.size())) {
                st.error = Errno("write", path.string());
                if (fd >= 0) ::close(fd);
                failed = true;
                break;
            }
            if (options.durable) StartWriteback(fd);
            // Finish the previous chunk while this one is in flight.
            if (!finish_previous()) {
                st.error = "fdatasync failed";
                ::close(fd);
                failed = true;
                break;
            }
            st.previous_fd = fd;
            st.write_ns += NsSince(ts);
            chunks[shard] = {shard, key, count, encoded.size(), crc};
            st.stats.total_bytes += encoded.size();
        }
        if (!finish_previous() && st.error.empty()) {
            st.error = "fdatasync failed";
            failed = true;
        }
    };
    t = SteadyClock::now();
    {
        std::vector<std::thread> threads;
        for (size_t w = 1; w < workers; ++w) threads.emplace_back(body, w);
        body(0);
        for (auto& th : threads) th.join();
    }
    result.chunks_wall_ns = NsSince(t);
    std::vector<handoff::QuarantineRange> quarantine;
    for (auto& st : states) {
        if (!st.error.empty()) return tl::make_unexpected(st.error);
        auto& s = manifest.writer;
        s.objects += st.stats.objects;
        s.replicas += st.stats.replicas;
        s.skipped_processing += st.stats.skipped_processing;
        s.skipped_no_exportable_replica +=
            st.stats.skipped_no_exportable_replica;
        s.dropped_non_memory_replicas += st.stats.dropped_non_memory_replicas;
        s.dropped_incomplete_replicas += st.stats.dropped_incomplete_replicas;
        s.dropped_unknown_segment_replicas +=
            st.stats.dropped_unknown_segment_replicas;
        s.quarantine_ranges += st.stats.quarantine_ranges;
        s.quarantine_bytes += st.stats.quarantine_bytes;
        s.total_bytes += st.stats.total_bytes;
        result.export_cpu_ns += st.export_ns;
        result.encode_cpu_ns += st.encode_ns;
        result.write_cpu_ns += st.write_ns;
        quarantine.insert(quarantine.end(), st.quarantine.begin(),
                          st.quarantine.end());
    }
    manifest.chunks = std::move(chunks);
    manifest.object_count = manifest.writer.objects;
    manifest.replica_count = manifest.writer.replicas;
    // Read after the export: every id the old master handed out is below it,
    // including ids of objects that were left out.
    manifest.next_replica_id = MasterService::HandoffNextReplicaId();

    // ---- Small files: weight metadata, quarantine, upstream compatibility
    // artifacts, then the handoff manifest last.
    t = SteadyClock::now();
    const auto weight = service.ExportHandoffWeightMetadata();
    const auto weight_bytes = EncodeBatchOpLogSnapshotWeightMetadata(weight);
    auto weight_blob = WriteBlob(
        root,
        ha::BuildBatchOpLogSnapshotWeightMetadataKey(
            handoff::kHandoffSnapshotRoot, manifest.snapshot_id),
        weight_bytes.data(), weight_bytes.size(), options.durable);
    if (!weight_blob) return tl::make_unexpected(weight_blob.error());
    manifest.weight_metadata = *weight_blob;

    const auto quarantine_bytes = handoff::EncodeQuarantine(quarantine);
    auto quarantine_blob =
        WriteBlob(root, handoff::kHandoffQuarantineFile,
                  quarantine_bytes.data(), quarantine_bytes.size(),
                  options.durable);
    if (!quarantine_blob) return tl::make_unexpected(quarantine_blob.error());
    manifest.quarantine = *quarantine_blob;
    manifest.quarantine_ranges = quarantine.size();

    // Upstream layout (segments.bin, manifest.json, descriptor.json), so that
    // upstream tooling can read the directory. The restore does not use them.
    std::vector<StandbySegmentInfo> standby_segments;
    for (const auto& s : manifest.segments) {
        StandbySegmentInfo info;
        info.segment_name = s.name;
        info.transport_endpoint = s.te_endpoint;
        info.capacity = s.size;
        info.is_memory_segment = true;
        standby_segments.push_back(std::move(info));
    }
    const auto segments_bytes =
        EncodeBatchOpLogSnapshotSegments(standby_segments);
    auto segments_blob = WriteBlob(
        root,
        ha::BuildBatchOpLogSnapshotSegmentsKey(handoff::kHandoffSnapshotRoot,
                                               manifest.snapshot_id),
        segments_bytes.data(), segments_bytes.size(), options.durable);
    if (!segments_blob) return tl::make_unexpected(segments_blob.error());
    ha::BatchOpLogSnapshotManifest upstream_manifest;
    upstream_manifest.schema_version = ha::kBatchOpLogWeightSnapshotSchemaVersion;
    upstream_manifest.snapshot_format = ha::kBatchOpLogWeightSnapshotFormat;
    upstream_manifest.snapshot_id = manifest.snapshot_id;
    upstream_manifest.segments = {segments_blob->key, segments_blob->bytes,
                                  segments_blob->crc32c};
    for (const auto& c : manifest.chunks) {
        upstream_manifest.object_chunks.push_back(
            {c.index, c.key, c.objects, c.bytes, c.crc32c});
    }
    upstream_manifest.weight_metadata =
        ha::BatchOpLogSnapshotObjectDescriptor{
            weight_blob->key, weight_blob->bytes, weight_blob->crc32c};
    const std::string upstream_manifest_json =
        ha::EncodeBatchOpLogSnapshotManifest(upstream_manifest);
    const std::string upstream_manifest_key =
        ha::BuildBatchOpLogSnapshotManifestKey(handoff::kHandoffSnapshotRoot,
                                               manifest.snapshot_id);
    auto upstream_manifest_blob =
        WriteBlob(root, upstream_manifest_key, upstream_manifest_json.data(),
                  upstream_manifest_json.size(), options.durable);
    if (!upstream_manifest_blob) {
        return tl::make_unexpected(upstream_manifest_blob.error());
    }
    ha::BatchOpLogSnapshotDescriptor descriptor;
    descriptor.schema_version = ha::kBatchOpLogWeightSnapshotSchemaVersion;
    descriptor.snapshot_format = ha::kBatchOpLogWeightSnapshotFormat;
    descriptor.snapshot_id = manifest.snapshot_id;
    descriptor.manifest_key = upstream_manifest_key;
    descriptor.manifest_size = upstream_manifest_json.size();
    descriptor.manifest_crc32c = upstream_manifest_blob->crc32c;
    descriptor.created_at_ms = manifest.created_at_ms;
    const std::string descriptor_json =
        ha::EncodeBatchOpLogSnapshotDescriptor(descriptor);
    auto descriptor_blob = WriteBlob(
        root,
        ha::BuildBatchOpLogSnapshotDescriptorKey(handoff::kHandoffSnapshotRoot,
                                                 manifest.snapshot_id),
        descriptor_json.data(), descriptor_json.size(), options.durable);
    if (!descriptor_blob) return tl::make_unexpected(descriptor_blob.error());

    if (options.durable) {
        // Directory entries of every file written so far.
        for (auto it = fs::recursive_directory_iterator(root, ec);
             !ec && it != fs::recursive_directory_iterator(); it.increment(ec)) {
            if (it->is_directory() && !FsyncDir(it->path())) {
                return tl::make_unexpected("fsync dir " +
                                           it->path().string());
            }
        }
    }

    manifest.writer.workers = workers;
    manifest.writer.peak_rss_kib = ProcStatusKiB("VmHWM");
    manifest.writer.duration_ms = NsSince(total_start) / 1000000;
    const std::string manifest_text = handoff::EncodeManifest(manifest);
    const std::string tmp_name = std::string(handoff::kHandoffManifestFile) + ".tmp";
    auto manifest_blob = WriteBlob(root, tmp_name, manifest_text.data(),
                                   manifest_text.size(), options.durable);
    if (!manifest_blob) return tl::make_unexpected(manifest_blob.error());
    fs::rename(root / tmp_name, root / handoff::kHandoffManifestFile, ec);
    if (ec) {
        return tl::make_unexpected("rename manifest: " + ec.message());
    }
    if (options.durable && !FsyncDir(root)) {
        return tl::make_unexpected("fsync dir " + root.string());
    }
    result.tail_ns = NsSince(t);
    result.stats = manifest.writer;
    result.stats.workers = workers;
    result.chunks = num_chunks;
    result.total_ns = NsSince(total_start);
    LOG(INFO) << "handoff_write=ok, dir=" << options.dir
              << ", stats=" << result.ToJson();
    return result;
}

}  // namespace mooncake
