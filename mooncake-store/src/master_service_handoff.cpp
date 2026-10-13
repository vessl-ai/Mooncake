// Shutdown handoff snapshot: export and trusted restore (INF-600, fork-only).
//
// Restore outline (all under client_mutex_ + snapshot_mutex_ held
// exclusively, before the master serves):
//   0. Read and check the handoff manifest. Mount every restored memory
//      segment with its original identity and owner, an empty real
//      OffsetBufferAllocator, allocation disabled, and its endpoint in the
//      read gate (invalid_replica_endpoints_).
//   1. Stream object chunks in parallel and insert objects into the metadata
//      shards with placeholder buffers; remember (address, size, Replica*) per
//      segment.
//   2. Per segment, in parallel: sort, import the live and quarantined ranges
//      into the real allocator, and swap each placeholder buffer for the
//      imported one, bound to the segment registration.
//   3. Finalize: replica id floor, tenant quota, metrics, quarantine
//      deadline, fresh liveness timestamps.
// A remount by the owner with the same segment identity then only opens the
// segment (see ConfirmPendingRestoredSegmentsLocked); a mismatch drops it
// (see FindHandoffMismatchesLocked / DropPendingRestoredSegmentLocked).
//
// L4 (schema 2): LOCAL_DISK replicas travel in the same chunks. For every
// store that owns a restored memory segment, its LocalSSD registration is
// restored too, so its next offload heartbeat succeeds and it does not
// re-register its disk keys. A restored LOCAL_DISK replica is unreadable
// until any RPC from its owner's client id arrives (the id is random per
// process, so a restarted store can never open it); see HandoffDiskGated.

#include <algorithm>
#include <atomic>
#include <chrono>
#include <fstream>
#include <limits>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <glog/logging.h>

#include "ha/snapshot/batch_oplog/codec.h"
#include "handoff/handoff_format.h"
#include "master_service.h"

namespace mooncake {

namespace {

using SteadyClock = std::chrono::steady_clock;

uint64_t NsSince(SteadyClock::time_point start) {
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            SteadyClock::now() - start)
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

// Runs fn(index, worker) for index in [0, count) on `workers` threads with
// dynamic scheduling.
template <typename Fn>
void ParallelFor(size_t count, size_t workers, Fn&& fn) {
    workers = std::max<size_t>(1, std::min(workers, count));
    std::atomic<size_t> next{0};
    auto body = [&](size_t worker) {
        for (;;) {
            const size_t index = next.fetch_add(1, std::memory_order_relaxed);
            if (index >= count) {
                return;
            }
            fn(index, worker);
        }
    };
    std::vector<std::thread> threads;
    threads.reserve(workers - 1);
    for (size_t w = 1; w < workers; ++w) {
        threads.emplace_back(body, w);
    }
    body(0);
    for (auto& t : threads) {
        t.join();
    }
}

struct RestoredRef {
    uint64_t address{0};
    uint64_t size{0};
    Replica* replica{nullptr};  // nullptr: quarantined range
};

}  // namespace

std::string MasterService::HandoffRestoreStats::ToJson() const {
    std::ostringstream o;
    o << "{\"objects\":" << objects << ",\"replicas\":" << replicas
      << ",\"segments\":" << segments << ",\"chunks\":" << chunks
      << ",\"quarantine_ranges\":" << quarantine_ranges
      << ",\"schema\":" << schema
      << ",\"local_disk_clients\":" << local_disk_clients
      << ",\"local_disk_replicas\":" << local_disk_replicas
      << ",\"disk_only_objects\":" << disk_only_objects
      << ",\"offload_pins\":" << offload_pins << ",\"workers\":" << workers
      << ",\"manifest_ns\":" << manifest_ns << ",\"mount_ns\":" << mount_ns
      << ",\"insert_wall_ns\":" << insert_wall_ns
      << ",\"load_cpu_ns\":" << load_cpu_ns
      << ",\"prepare_cpu_ns\":" << prepare_cpu_ns
      << ",\"insert_cpu_ns\":" << insert_cpu_ns
      << ",\"rebuild_wall_ns\":" << rebuild_wall_ns
      << ",\"sort_cpu_ns\":" << sort_cpu_ns
      << ",\"import_cpu_ns\":" << import_cpu_ns
      << ",\"finalize_ns\":" << finalize_ns << ",\"total_ns\":" << total_ns
      << ",\"rss_kib_start\":" << rss_kib_start
      << ",\"rss_kib_after_insert\":" << rss_kib_after_insert
      << ",\"rss_kib_end\":" << rss_kib_end
      << ",\"hwm_kib_end\":" << hwm_kib_end << "}";
    return o.str();
}

// ------------------------------------------------------------------ export

std::vector<handoff::SegmentIdentity> MasterService::ExportHandoffSegments() {
    std::vector<handoff::SegmentIdentity> out;
    ScopedSegmentAccess segment_access = segment_manager_.getSegmentAccess();
    std::vector<std::pair<Segment, UUID>> all;
    segment_access.GetAllSegments(all);
    for (const auto& [segment, owner] : all) {
        // DRAINING and DRAINED segments are exported too: their owner lists
        // them in its next remount, so leaving them out would remount them
        // with an empty allocator, allocatable at once, while a late write
        // may still land in their in-flight ranges. GRACEFULLY_UNMOUNTING
        // segments are left out on purpose: their owner has already removed
        // them from its own list, so it never remounts them. UNMOUNTING ones
        // are on their way out.
        SegmentStatus status = SegmentStatus::UNDEFINED;
        if (segment_access.GetSegmentStatusById(segment.id, status) !=
                ErrorCode::OK ||
            (status != SegmentStatus::OK && status != SegmentStatus::DRAINING &&
             status != SegmentStatus::DRAINED)) {
            continue;
        }
        if (owner == UUID{0, 0} || segment.protocol == "cxl" ||
            segment.base == 0 || segment.size == 0 ||
            !segment_access.GetAllocator(segment.id)) {
            continue;
        }
        auto identity = handoff::SegmentIdentity::FromSegment(segment, owner);
        identity.status = status;
        out.push_back(std::move(identity));
    }
    return out;
}

std::vector<LocalSsdHandoffClient> MasterService::ExportHandoffLocalDiskClients(
    const std::vector<handoff::SegmentIdentity>& segments) const {
    HandoffClientSet owners;
    for (const auto& segment : segments) {
        owners.insert(segment.owner_client_id);
    }
    std::vector<LocalSsdHandoffClient> out;
    for (auto& client : local_ssd_manager_.ExportHandoffClients()) {
        if (owners.contains(client.client_id)) {
            out.push_back(client);
        }
    }
    std::sort(
        out.begin(), out.end(),
        [](const LocalSsdHandoffClient& a, const LocalSsdHandoffClient& b) {
            return a.client_id < b.client_id;
        });
    return out;
}

void MasterService::ExportHandoffShard(
    size_t shard_idx, const handoff::SegmentLocator& locator,
    const HandoffClientSet& disk_clients,
    std::vector<StandbyObjectEntry>& objects,
    std::vector<handoff::QuarantineRange>& quarantine,
    std::vector<handoff::OffloadPin>& pins, handoff::WriterStats& stats) const {
    MetadataShardAccessorRO shard(this, shard_idx);
    size_t count = 0;
    for (const auto& [tenant_id, tenant_state] : shard->tenants) {
        (void)tenant_id;
        count += tenant_state.metadata.size();
    }
    objects.reserve(objects.size() + count);
    for (const auto& [tenant_id, tenant_state] : shard->tenants) {
        for (const auto& [key, metadata] : tenant_state.metadata) {
            // An in-flight put (PutStart without PutEnd) is never exported;
            // its ranges are quarantined below.
            const bool processing = tenant_state.processing_keys.contains(key);
            StandbyObjectEntry entry;
            size_t memory_replicas = 0;
            bool has_disk = false;
            for (const auto& replica : metadata.GetAllReplicas()) {
                if (replica.is_local_disk_replica()) {
                    // L4: kept only for an exported owner (all-or-nothing
                    // per store), complete, at most one per object, and
                    // only if it describes this object's size. A size
                    // mismatch is a stale re-registration: never carry it.
                    const auto owner = replica.get_local_disk_client_id();
                    if (processing || has_disk || !replica.is_completed() ||
                        !owner || !disk_clients.contains(*owner)) {
                        ++stats.dropped_local_disk_replicas;
                        continue;
                    }
                    auto descriptor = replica.get_descriptor();
                    const auto& disk = descriptor.get_local_disk_descriptor();
                    if (disk.object_size != metadata.size ||
                        disk.transport_endpoint.empty()) {
                        ++stats.dropped_local_disk_size_mismatch;
                        continue;
                    }
                    entry.metadata.replicas.push_back(std::move(descriptor));
                    has_disk = true;
                    continue;
                }
                if (!replica.is_memory_replica()) {
                    // DISK / NoF / DFS replicas are out of scope.
                    ++stats.dropped_non_memory_replicas;
                    continue;
                }
                if (replica.has_invalid_mem_handle()) {
                    // Its segment is gone; the range may already belong to a
                    // newer mount, so it is neither exported nor quarantined.
                    ++stats.dropped_unknown_segment_replicas;
                    continue;
                }
                auto descriptor = replica.get_descriptor();
                const auto& buffer =
                    descriptor.get_memory_descriptor().buffer_descriptor;
                const auto segment_index =
                    locator.Find(buffer.transport_endpoint_,
                                 buffer.buffer_address_, buffer.size_);
                if (!segment_index) {
                    ++stats.dropped_unknown_segment_replicas;
                    continue;
                }
                if (processing || !replica.is_completed() ||
                    buffer.size_ != metadata.size) {
                    // The remaining time of an in-flight write is unknown:
                    // hold the range for the full release timeout.
                    quarantine.push_back({*segment_index,
                                          buffer.buffer_address_, buffer.size_,
                                          HandoffFullHoldMs()});
                    ++stats.dropped_incomplete_replicas;
                    ++stats.quarantine_ranges;
                    stats.quarantine_bytes += buffer.size_;
                    continue;
                }
                entry.metadata.replicas.push_back(std::move(descriptor));
                ++memory_replicas;
            }
            if (entry.metadata.replicas.empty()) {
                if (processing) {
                    ++stats.skipped_processing;
                } else {
                    ++stats.skipped_no_exportable_replica;
                }
                continue;
            }
            entry.tenant_id = tenant_id.value();
            entry.key = key;
            entry.metadata.client_id = metadata.client_id;
            entry.metadata.size = metadata.size;
            entry.metadata.group_id = metadata.group_id;
            entry.metadata.data_type = metadata.data_type;
            entry.metadata.hard_pinned = metadata.IsHardPinned();
            ++stats.objects;
            stats.replicas += memory_replicas;
            if (has_disk) {
                ++stats.local_disk_replicas;
                if (memory_replicas == 0) ++stats.disk_only_objects;
            }
            // A store may still be copying this object to its disk from
            // its own memory: keep the memory replicas pinned in the next
            // master until the quarantine deadline.
            if (memory_replicas > 0 &&
                tenant_state.offloading_tasks.contains(key)) {
                pins.push_back({entry.tenant_id, key});
                ++stats.offload_pins;
            }
            objects.push_back(std::move(entry));
        }
    }
}

void MasterService::ExportHandoffPendingQuarantine(
    const std::vector<handoff::SegmentIdentity>& segments,
    std::vector<handoff::QuarantineRange>& quarantine,
    handoff::WriterStats& stats) const {
    std::unordered_map<UUID, uint32_t, boost::hash<UUID>> index_by_id;
    for (uint32_t i = 0; i < segments.size(); ++i) {
        index_by_id.emplace(segments[i].segment_id, i);
    }
    const auto now = SteadyClock::now();
    std::lock_guard<std::mutex> lock(handoff_quarantine_.mutex);
    for (const auto& held : handoff_quarantine_.buffers) {
        const auto it = index_by_id.find(held.segment_id);
        // A dropped segment's ranges may belong to a newer mount already.
        if (it == index_by_id.end() || !held.buffer ||
            !held.buffer->isAllocatorValid() || held.deadline <= now) {
            continue;
        }
        const auto remaining =
            std::chrono::duration_cast<std::chrono::milliseconds>(
                held.deadline - now)
                .count();
        quarantine.push_back(
            {it->second, reinterpret_cast<uint64_t>(held.buffer->data()),
             held.buffer->size(), static_cast<uint64_t>(remaining) + 1});
        ++stats.quarantine_ranges;
        stats.quarantine_bytes += held.buffer->size();
    }
}

void MasterService::ExportHandoffDiscardedQuarantine(
    const handoff::SegmentLocator& locator,
    std::vector<handoff::QuarantineRange>& quarantine,
    handoff::WriterStats& stats) {
    const auto now = std::chrono::system_clock::now();
    std::lock_guard<std::mutex> lock(discarded_replicas_mutex_);
    for (const auto& item : discarded_replicas_) {
        if (item.isExpired(now)) {
            continue;  // this master would release it at its next sweep
        }
        const auto remaining =
            std::chrono::duration_cast<std::chrono::milliseconds>(item.ttl() -
                                                                  now)
                .count();
        for (const auto& replica : item.replicas()) {
            if (!replica.is_memory_replica()) {
                continue;
            }
            if (replica.has_invalid_mem_handle()) {
                // Its segment was unmounted; the range may already belong to
                // a newer mount (same rule as for live replicas).
                ++stats.dropped_discarded_ranges;
                continue;
            }
            const auto descriptor = replica.get_descriptor();
            const auto& buffer =
                descriptor.get_memory_descriptor().buffer_descriptor;
            const auto segment_index =
                locator.Find(buffer.transport_endpoint_, buffer.buffer_address_,
                             buffer.size_);
            if (!segment_index) {
                // On a segment that is not exported (e.g. gracefully
                // unmounting): its owner never remounts it, so the range is
                // never allocated by the next master.
                ++stats.dropped_discarded_ranges;
                continue;
            }
            quarantine.push_back({*segment_index, buffer.buffer_address_,
                                  buffer.size_,
                                  static_cast<uint64_t>(remaining) + 1});
            ++stats.discarded_ranges;
            ++stats.quarantine_ranges;
            stats.quarantine_bytes += buffer.size_;
        }
    }
}

uint64_t MasterService::HandoffLocalDiskObjectCount() const {
    uint64_t count = 0;
    for (size_t s = 0; s < kNumShards; ++s) {
        MetadataShardAccessorRO shard(this, s);
        if (shard->disk_object_count > 0) {
            count += static_cast<uint64_t>(shard->disk_object_count);
        }
    }
    return count;
}

uint64_t MasterService::HandoffObjectCount() const {
    uint64_t count = 0;
    for (size_t s = 0; s < kNumShards; ++s) {
        MetadataShardAccessorRO shard(this, s);
        for (const auto& [tenant_id, tenant_state] : shard->tenants) {
            (void)tenant_id;
            count += tenant_state.metadata.size();
        }
    }
    return count;
}

// ----------------------------------------------------------------- restore

tl::expected<MasterService::HandoffRestoreStats, std::string>
MasterService::RestoreFromHandoff(const std::string& dir,
                                  const HandoffRestoreOptions& options) {
    if (!HandoffMasterIsEmpty()) {
        // Never reset a master that already holds state.
        return tl::make_unexpected("master is not empty");
    }
    auto result = RestoreFromHandoffLocked(dir, options);
    if (!result) {
        LOG(ERROR) << "handoff_restore=failed, dir=" << dir
                   << ", error=" << result.error()
                   << ", action=reset_to_empty_master";
        ResetHandoffState();
        ResetStateAfterFailedRestoreAttempt();
        return result;
    }
    LOG(INFO) << "handoff_restore=ok, dir=" << dir
              << ", stats=" << result->ToJson();
    return result;
}

void MasterService::ResetHandoffState() {
    std::vector<HandoffQuarantinedBuffer> released;
    {
        std::unique_lock<std::shared_mutex> client_lock(client_mutex_);
        std::unique_lock<std::shared_mutex> snapshot_lock(snapshot_mutex_);
        pending_restored_segments_.clear();
        invalid_replica_endpoints_.clear();
        handoff_disk_gate_.active.store(false, std::memory_order_release);
        handoff_disk_gate_.opened.clear();
        if (handoff_restored_file_capacity_ > 0) {
            MasterMetricManager::instance().dec_total_file_capacity(
                handoff_restored_file_capacity_);
        }
        handoff_restored_file_capacity_ = 0;
        std::lock_guard<std::mutex> lock(handoff_quarantine_.mutex);
        released.swap(handoff_quarantine_.buffers);
        handoff_offload_pins_.clear();
    }
}

bool MasterService::HandoffMasterIsEmpty() {
    std::shared_lock<std::shared_mutex> client_lock(client_mutex_);
    std::shared_lock<std::shared_mutex> snapshot_lock(snapshot_mutex_);
    {
        ScopedSegmentAccess segment_access =
            segment_manager_.getSegmentAccess();
        std::vector<std::pair<Segment, UUID>> mounted;
        segment_access.GetAllSegments(mounted);
        if (!mounted.empty()) return false;
    }
    if (!client_liveness_records_.empty() ||
        !pending_restored_segments_.empty() ||
        !local_ssd_manager_.GetClientIds().empty()) {
        return false;
    }
    for (size_t s = 0; s < kNumShards; ++s) {
        MetadataShardAccessorRO shard(this, s);
        if (!shard->tenants.empty()) return false;
    }
    return true;
}

size_t MasterService::HandoffPendingSegmentCount() const {
    std::shared_lock<std::shared_mutex> lock(client_mutex_);
    return pending_restored_segments_.size();
}

size_t MasterService::HandoffQuarantineBufferCount() const {
    std::lock_guard<std::mutex> lock(handoff_quarantine_.mutex);
    return handoff_quarantine_.buffers.size();
}

tl::expected<MasterService::HandoffRestoreStats, std::string>
MasterService::RestoreFromHandoffLocked(const std::string& dir,
                                        const HandoffRestoreOptions& options) {
    const auto total_start = SteadyClock::now();
    HandoffRestoreStats stats;
    stats.rss_kib_start = ProcStatusKiB("VmRSS");
    const size_t workers = std::max<size_t>(1, options.workers);
    stats.workers = workers;

    // ---- Phase 0a: manifest and preconditions (no locks needed yet).
    auto phase_start = SteadyClock::now();
    if (enable_ha_ || enable_dfs_) {
        return tl::make_unexpected(
            "handoff restore requires a standalone "
            "master without DFS");
    }
    if (memory_allocator_type_ != BufferAllocatorType::OFFSET) {
        return tl::make_unexpected(
            "handoff restore supports only the offset "
            "allocator");
    }
    auto manifest_bytes = handoff::ReadFileBytes(
        dir + "/" + handoff::kHandoffManifestFile, handoff::kMaxManifestBytes);
    if (!manifest_bytes) {
        return tl::make_unexpected(manifest_bytes.error());
    }
    auto manifest = handoff::DecodeManifest(
        std::string_view(reinterpret_cast<const char*>(manifest_bytes->data()),
                         manifest_bytes->size()));
    if (!manifest) {
        return tl::make_unexpected(manifest.error());
    }
    if (!handoff::CodecMatches(manifest->codec, manifest->schema_version)) {
        return tl::make_unexpected("codec fingerprint mismatch");
    }
    if (manifest->memory_allocator != "offset") {
        return tl::make_unexpected("snapshot allocator mismatch");
    }
    if (manifest->multi_tenant != enable_multi_tenants_) {
        return tl::make_unexpected("multi-tenant mode mismatch");
    }
    if (auto invalid = handoff::ValidateSegments(manifest->segments)) {
        return tl::make_unexpected(*invalid);
    }
    // This fork's master has no weight management, so its writer never
    // writes weight metadata. A snapshot that carries some was written by
    // another build; restoring it would silently drop that state.
    if (manifest->weight_metadata) {
        return tl::make_unexpected(
            "weight metadata is not supported by this build");
    }
    std::vector<handoff::QuarantineRange> quarantine_ranges;
    {
        auto bytes = handoff::ReadVerifiedBlob(dir, manifest->quarantine.key,
                                               manifest->quarantine.bytes,
                                               manifest->quarantine.crc32c);
        if (!bytes) return tl::make_unexpected(bytes.error());
        auto decoded = handoff::DecodeQuarantine(*bytes);
        if (!decoded) return tl::make_unexpected(decoded.error());
        quarantine_ranges = std::move(*decoded);
        if (quarantine_ranges.size() != manifest->quarantine_ranges) {
            return tl::make_unexpected("quarantine range count mismatch");
        }
    }
    const auto& segments = manifest->segments;
    const size_t nseg = segments.size();
    const handoff::SegmentLocator locator(segments);
    for (const auto& range : quarantine_ranges) {
        if (range.segment_index >= nseg ||
            locator.Find(segments[range.segment_index].te_endpoint,
                         range.address, range.size) != range.segment_index) {
            return tl::make_unexpected("quarantine range outside segment");
        }
    }
    // L4 state (schema 2). A schema-1 snapshot carries none: its stores
    // re-register their disk keys as after a cold start.
    std::vector<LocalSsdHandoffClient> disk_clients;
    std::vector<handoff::OffloadPin> offload_pins;
    if (manifest->schema_version >= 2) {
        auto disk_bytes = handoff::ReadVerifiedBlob(
            dir, manifest->local_disk->key, manifest->local_disk->bytes,
            manifest->local_disk->crc32c);
        if (!disk_bytes) return tl::make_unexpected(disk_bytes.error());
        auto decoded_clients = handoff::DecodeLocalDiskClients(*disk_bytes);
        if (!decoded_clients) {
            return tl::make_unexpected(decoded_clients.error());
        }
        disk_clients = std::move(*decoded_clients);
        auto pin_bytes = handoff::ReadVerifiedBlob(
            dir, manifest->offload_pins->key, manifest->offload_pins->bytes,
            manifest->offload_pins->crc32c);
        if (!pin_bytes) return tl::make_unexpected(pin_bytes.error());
        auto decoded_pins = handoff::DecodeOffloadPins(*pin_bytes);
        if (!decoded_pins) return tl::make_unexpected(decoded_pins.error());
        offload_pins = std::move(*decoded_pins);
        if (disk_clients.size() != manifest->local_disk_clients ||
            offload_pins.size() != manifest->offload_pin_count) {
            return tl::make_unexpected("L4 blob count mismatch");
        }
    }
    if (manifest->local_disk_replica_count > 0 && disk_clients.empty()) {
        return tl::make_unexpected("LOCAL_DISK replicas without owners");
    }
    if (!disk_clients.empty() && !enable_offload_) {
        // Never restore without the L4 state the snapshot carries.
        return tl::make_unexpected(
            "snapshot carries L4 state but offload is disabled");
    }
    {
        std::unordered_set<UUID, boost::hash<UUID>> owners, seen;
        for (const auto& segment : segments) {
            owners.insert(segment.owner_client_id);
        }
        for (const auto& client : disk_clients) {
            if (!owners.contains(client.client_id) ||
                !seen.insert(client.client_id).second ||
                client.total_capacity_bytes < 0) {
                return tl::make_unexpected("invalid L4 client entry");
            }
        }
    }
    stats.schema = manifest->schema_version;
    stats.local_disk_clients = disk_clients.size();
    stats.segments = nseg;
    stats.chunks = manifest->chunks.size();
    stats.quarantine_ranges = quarantine_ranges.size();
    stats.manifest_ns = NsSince(phase_start);

    std::unique_lock<std::shared_mutex> client_lock(client_mutex_);
    std::unique_lock<std::shared_mutex> snapshot_lock(snapshot_mutex_);

    // ---- Phase 0b: mount restored segments, closed for reads and writes.
    phase_start = SteadyClock::now();
    std::vector<std::shared_ptr<OffsetBufferAllocator>> allocators(nseg);
    std::vector<std::shared_ptr<BufferAllocatorBase>> placeholders(nseg);
    std::vector<std::function<void(AllocatedBuffer&)>> binders(nseg);
    std::unordered_map<UUID, std::shared_ptr<ClientLivenessRecord>,
                       boost::hash<UUID>>
        records;
    {
        ScopedSegmentAccess segment_access =
            segment_manager_.getSegmentAccess();
        for (size_t i = 0; i < nseg; ++i) {
            const auto& identity = segments[i];
            auto& record = records[identity.owner_client_id];
            if (!record) {
                record = std::make_shared<ClientLivenessRecord>(
                    ClientLivenessRecord::Clock::now());
            }
            allocators[i] = std::make_shared<OffsetBufferAllocator>(
                identity.name, identity.base, identity.size,
                identity.te_endpoint, ReplicaType::MEMORY);
            const Segment segment = identity.ToSegment();
            const auto mounted = segment_access.MountRestoredSegment(
                segment, identity.owner_client_id, record, allocators[i],
                identity.status);
            if (mounted != ErrorCode::OK) {
                return tl::make_unexpected("mount of restored segment " +
                                           identity.name +
                                           " failed: " + toString(mounted));
            }
            binders[i] = segment_access.MakeBufferBinder(segment.id);
            if (!binders[i]) {
                return tl::make_unexpected("no registration for segment " +
                                           identity.name);
            }
            placeholders[i] = std::make_shared<DummyBufferAllocator>(
                identity.name, identity.te_endpoint);
            pending_restored_segments_[segment.id] = {identity.owner_client_id,
                                                      segment};
            invalid_replica_endpoints_.insert(identity.te_endpoint);
            invalid_replica_endpoints_.insert(identity.name);
        }
    }
    for (auto& [client_id, record] : records) {
        client_liveness_records_.emplace(client_id, record);
        MasterMetricManager::instance().client_liveness_record_created();
    }
    // Restored LOCAL_DISK replicas are bound to their owner's restored
    // liveness record (the owner also owns a restored memory segment), so
    // the existing TTL offboarding removes them if the owner never returns.
    std::unordered_map<UUID, std::shared_ptr<ClientLivenessRecord>,
                       boost::hash<UUID>>
        disk_records;
    for (const auto& client : disk_clients) {
        disk_records.emplace(client.client_id, records.at(client.client_id));
    }
    stats.mount_ns = NsSince(phase_start);

    // ---- Phase 1: stream chunks and insert objects with placeholders.
    struct WorkerState {
        std::vector<std::vector<RestoredRef>> refs;
        ReplicaID max_id{0};
        uint64_t objects{0};
        uint64_t replicas{0};  // memory
        uint64_t disk_replicas{0};
        uint64_t memory_objects{0};
        std::unordered_map<UUID, int64_t, boost::hash<UUID>> disk_bytes;
        uint64_t load_ns{0};
        uint64_t prepare_ns{0};
        uint64_t insert_ns{0};
        std::string error;
    };
    std::vector<WorkerState> states(workers);
    for (auto& st : states) {
        st.refs.resize(nseg);
    }
    std::atomic<bool> failed{false};
    const auto now = std::chrono::system_clock::now();
    // Read leases: the old master granted none after the export started,
    // so a lease it handed out ends by export + kv_lease_ttl (assuming both
    // masters run with the same --default_kv_lease_ttl). Restored objects
    // keep that floor, so a reader still inside its lease is not undercut
    // by an eviction, remove or upsert right after a short roll.
    const auto lease_floor =
        std::chrono::system_clock::time_point(
            std::chrono::milliseconds(manifest->created_at_ms)) +
        std::chrono::milliseconds(default_kv_lease_ttl_);
    struct Prepared {
        size_t shard;
        TenantId tenant_id;
        std::string user_key;
        const StandbyObjectEntry* entry;
        std::vector<Replica> replicas;
        bool has_memory{false};
        bool has_disk{false};
        // Per memory replica, in order: segment index, address, size.
        std::vector<std::pair<uint32_t, std::pair<uint64_t, uint64_t>>>
            placement;
    };

    phase_start = SteadyClock::now();
    ParallelFor(
        manifest->chunks.size(), workers, [&](size_t chunk_index, size_t w) {
            if (failed.load(std::memory_order_relaxed)) return;
            auto& st = states[w];
            const auto fail = [&](std::string message) {
                if (st.error.empty()) st.error = std::move(message);
                failed = true;
            };
            const auto& chunk = manifest->chunks[chunk_index];

            auto t = SteadyClock::now();
            auto bytes = handoff::ReadVerifiedBlob(dir, chunk.key, chunk.bytes,
                                                   chunk.crc32c);
            if (!bytes) {
                fail(bytes.error());
                return;
            }
            auto decoded = DecodeBatchOpLogSnapshotObjectChunk(
                *bytes, chunk.index, chunk.objects);
            if (!decoded) {
                fail("chunk " + std::to_string(chunk.index) + ": " +
                     decoded.error());
                return;
            }
            std::vector<uint8_t>().swap(*bytes);
            const auto& objects = decoded->objects;
            st.load_ns += NsSince(t);

            t = SteadyClock::now();
            std::vector<Prepared> prepared;
            prepared.reserve(objects.size());
            for (const auto& entry : objects) {
                auto [scoped_tenant_id, user_key] =
                    TenantId::ParseScopedKey(entry.key);
                TenantId tenant_id(entry.tenant_id);
                if (tenant_id.IsDefault() && !scoped_tenant_id.IsDefault()) {
                    tenant_id = std::move(scoped_tenant_id);
                }
                if (!tenant_id.IsValid() || user_key.empty()) {
                    fail("invalid tenant or key");
                    return;
                }
                const auto& meta = entry.metadata;
                if (meta.replicas.empty()) {
                    fail("object without replicas");
                    return;
                }
                const size_t shard_idx = getShardIndex(tenant_id, user_key);
                Prepared p{shard_idx,
                           std::move(tenant_id),
                           std::move(user_key),
                           &entry,
                           {},
                           false,
                           false,
                           {}};
                p.replicas.reserve(meta.replicas.size());
                for (size_t r = 0; r < meta.replicas.size(); ++r) {
                    const auto& desc = meta.replicas[r];
                    if (desc.id == 0 ||
                        desc.id == std::numeric_limits<ReplicaID>::max()) {
                        fail("invalid replica id");
                        return;
                    }
                    for (size_t q = 0; q < r; ++q) {
                        if (meta.replicas[q].id == desc.id) {
                            fail("duplicate replica id");
                            return;
                        }
                    }
                    if (desc.status != ReplicaStatus::COMPLETE) {
                        fail("unexpected replica status");
                        return;
                    }
                    if (desc.is_local_disk_replica()) {
                        const auto& disk = desc.get_local_disk_descriptor();
                        const auto owner = disk_records.find(disk.client_id);
                        if (p.has_disk || owner == disk_records.end() ||
                            disk.object_size != meta.size ||
                            disk.transport_endpoint.empty()) {
                            fail("invalid local disk descriptor");
                            return;
                        }
                        st.max_id = std::max(st.max_id, desc.id);
                        p.replicas.push_back(
                            Replica(desc.id, disk.client_id, disk.object_size,
                                    disk.transport_endpoint,
                                    ReplicaStatus::COMPLETE, owner->second));
                        p.has_disk = true;
                        st.disk_bytes[disk.client_id] +=
                            static_cast<int64_t>(disk.object_size);
                        continue;
                    }
                    if (!desc.is_memory_replica()) {
                        fail("unexpected replica kind");
                        return;
                    }
                    const auto& buffer =
                        desc.get_memory_descriptor().buffer_descriptor;
                    if (buffer.size_ != meta.size ||
                        buffer.protocol_ == "cxl") {
                        fail("invalid memory descriptor");
                        return;
                    }
                    const auto si =
                        locator.Find(buffer.transport_endpoint_,
                                     buffer.buffer_address_, buffer.size_);
                    if (!si) {
                        fail(
                            "memory descriptor outside every restored segment");
                        return;
                    }
                    st.max_id = std::max(st.max_id, desc.id);
                    p.replicas.push_back(
                        Replica(desc.id,
                                std::make_unique<AllocatedBuffer>(
                                    placeholders[*si], buffer),
                                desc.status));
                    p.placement.push_back(
                        {*si, {buffer.buffer_address_, buffer.size_}});
                    p.has_memory = true;
                }
                prepared.push_back(std::move(p));
            }
            std::sort(prepared.begin(), prepared.end(),
                      [](const Prepared& a, const Prepared& b) {
                          return a.shard < b.shard;
                      });
            st.prepare_ns += NsSince(t);

            t = SteadyClock::now();
            size_t i = 0;
            while (i < prepared.size()) {
                const size_t shard_idx = prepared[i].shard;
                MetadataShardAccessorRW shard(this, shard_idx);
                for (; i < prepared.size() && prepared[i].shard == shard_idx;
                     ++i) {
                    auto& object = prepared[i];
                    const auto& meta = object.entry->metadata;
                    auto& tenant_state =
                        GetOrCreateTenantState(shard.get(), object.tenant_id);
                    auto [it, inserted] = tenant_state.metadata.emplace(
                        std::piecewise_construct,
                        std::forward_as_tuple(object.user_key),
                        std::forward_as_tuple(
                            meta.client_id, now, meta.size,
                            std::move(object.replicas), std::nullopt,
                            meta.hard_pinned.value_or(false), meta.data_type,
                            meta.group_id, object.tenant_id, object.user_key));
                    if (!inserted) {
                        fail("duplicate object " + object.user_key);
                        return;
                    }
                    auto& metadata = it->second;
                    if (!meta.group_id.empty()) {
                        metadata.lease_ = RegisterGroupMember(
                            object.tenant_id, object.user_key, meta.group_id);
                    }
                    metadata.lease_->ExtendTo(lease_floor);
                    // Every restored replica is complete.
                    metadata.memory_cache_total_accounted = object.has_memory;
                    metadata.disk_cache_total_accounted = object.has_disk;
                    if (object.has_disk) {
                        shard.OnDiskReplicaAdded(metadata);
                        ++st.disk_replicas;
                    }
                    if (object.has_memory) ++st.memory_objects;
                    size_t r = 0;
                    metadata.VisitReplicas(
                        &Replica::fn_is_memory_replica, [&](Replica& replica) {
                            const auto& [si, range] = object.placement[r];
                            st.refs[si].push_back(
                                {range.first, range.second, &replica});
                            ++r;
                        });
                    st.replicas += r;
                    ++st.objects;
                }
            }
            st.insert_ns += NsSince(t);
        });
    stats.insert_wall_ns = NsSince(phase_start);
    for (const auto& st : states) {
        if (!st.error.empty()) {
            return tl::make_unexpected(st.error);
        }
    }
    ReplicaID max_live_id = 0;
    uint64_t memory_objects = 0;
    std::unordered_map<UUID, int64_t, boost::hash<UUID>> disk_bytes;
    for (const auto& st : states) {
        max_live_id = std::max(max_live_id, st.max_id);
        stats.objects += st.objects;
        stats.replicas += st.replicas;
        stats.local_disk_replicas += st.disk_replicas;
        memory_objects += st.memory_objects;
        for (const auto& [client_id, bytes] : st.disk_bytes) {
            disk_bytes[client_id] += bytes;
        }
        stats.load_cpu_ns += st.load_ns;
        stats.prepare_cpu_ns += st.prepare_ns;
        stats.insert_cpu_ns += st.insert_ns;
    }
    stats.disk_only_objects = stats.objects - memory_objects;
    if (stats.objects != manifest->object_count ||
        stats.replicas != manifest->replica_count ||
        stats.local_disk_replicas != manifest->local_disk_replica_count) {
        return tl::make_unexpected("object or replica count mismatch");
    }
    if (max_live_id != 0 && max_live_id >= manifest->next_replica_id) {
        return tl::make_unexpected("replica id above next_replica_id");
    }
    stats.rss_kib_after_insert = ProcStatusKiB("VmRSS");

    // ---- Phase 2: per-segment real allocator rebuild.
    phase_start = SteadyClock::now();
    std::vector<std::vector<handoff::QuarantineRange>> quarantine_by_segment(
        nseg);
    for (const auto& range : quarantine_ranges) {
        quarantine_by_segment[range.segment_index].push_back(range);
    }
    std::vector<std::string> segment_errors(nseg);
    // Per segment: (hold_ms, buffer) of every quarantined range.
    std::vector<
        std::vector<std::pair<uint64_t, std::unique_ptr<AllocatedBuffer>>>>
        quarantined_buffers(nseg);
    std::vector<uint64_t> sort_ns(nseg, 0), import_ns(nseg, 0);
    std::vector<uint64_t> swapped(nseg, 0);
    ParallelFor(nseg, std::min(workers, nseg), [&](size_t si, size_t) {
        auto t = SteadyClock::now();
        size_t total = quarantine_by_segment[si].size();
        for (const auto& st : states) total += st.refs[si].size();
        std::vector<RestoredRef> refs;
        refs.reserve(total);
        for (auto& st : states) {
            refs.insert(refs.end(), st.refs[si].begin(), st.refs[si].end());
            std::vector<RestoredRef>().swap(st.refs[si]);
        }
        auto& segment_quarantine = quarantine_by_segment[si];
        std::sort(segment_quarantine.begin(), segment_quarantine.end(),
                  [](const handoff::QuarantineRange& a,
                     const handoff::QuarantineRange& b) {
                      return a.address < b.address;
                  });
        for (const auto& range : segment_quarantine) {
            refs.push_back({range.address, range.size, nullptr});
        }
        std::sort(refs.begin(), refs.end(),
                  [](const RestoredRef& a, const RestoredRef& b) {
                      return a.address < b.address;
                  });
        const uint64_t base = segments[si].base;
        std::vector<LiveAllocation> allocations;
        allocations.reserve(refs.size());
        for (size_t k = 0; k < refs.size(); ++k) {
            if (k > 0 &&
                refs[k].address < refs[k - 1].address + refs[k - 1].size) {
                segment_errors[si] =
                    "overlapping ranges in segment " + segments[si].name;
                return;
            }
            allocations.push_back({refs[k].address - base, refs[k].size});
        }
        sort_ns[si] = NsSince(t);

        t = SteadyClock::now();
        bool swap_failed = false;
        const auto& bind = binders[si];
        auto& quarantined = quarantined_buffers[si];
        const bool imported = ImportOffsetBufferAllocations(
            allocators[si], allocations,
            [&](size_t index, std::unique_ptr<AllocatedBuffer> buffer) {
                Replica* replica = refs[index].replica;
                if (replica == nullptr) {
                    // Ranges are disjoint (checked above), so the address
                    // identifies the quarantine entry.
                    const auto q = std::lower_bound(
                        segment_quarantine.begin(), segment_quarantine.end(),
                        refs[index].address,
                        [](const handoff::QuarantineRange& range,
                           uint64_t address) {
                            return range.address < address;
                        });
                    const uint64_t hold_ms =
                        q != segment_quarantine.end() &&
                                q->address == refs[index].address
                            ? q->hold_ms
                            : HandoffFullHoldMs();
                    quarantined.emplace_back(hold_ms, std::move(buffer));
                    return;
                }
                bind(*buffer);
                if (replica->replace_memory_buffer(std::move(buffer))) {
                    ++swapped[si];
                } else {
                    swap_failed = true;
                }
            });
        import_ns[si] = NsSince(t);
        if (!imported || swap_failed) {
            segment_errors[si] =
                "allocator import failed for segment " + segments[si].name;
        }
    });
    stats.rebuild_wall_ns = NsSince(phase_start);
    for (size_t si = 0; si < nseg; ++si) {
        stats.sort_cpu_ns += sort_ns[si];
        stats.import_cpu_ns += import_ns[si];
        if (!segment_errors[si].empty()) {
            return tl::make_unexpected(segment_errors[si]);
        }
    }

    // ---- Phase 3: finalize.
    phase_start = SteadyClock::now();
    // Every placeholder buffer must have been replaced by an imported one.
    uint64_t total_swapped = 0;
    for (const uint64_t n : swapped) total_swapped += n;
    if (total_swapped != stats.replicas) {
        return tl::make_unexpected("not every restored replica was rebound");
    }
    placeholders.clear();
    {
        const ReplicaID desired_next_id =
            static_cast<ReplicaID>(manifest->next_replica_id);
        ReplicaID current_next_id = Replica::next_id_.load();
        while (current_next_id < desired_next_id &&
               !Replica::next_id_.compare_exchange_weak(current_next_id,
                                                        desired_next_id)) {
        }
    }
    if (enable_multi_tenants_) {
        RebuildTenantQuotaUsageFromMetadata();
    }
    // L4 registrations: the owners' next offload heartbeat finds them, so
    // they do not re-register their disk keys. Their restored LOCAL_DISK
    // replicas stay unreadable until an RPC from the owner arrives.
    for (const auto& client : disk_clients) {
        const auto it = disk_bytes.find(client.client_id);
        const int64_t used = it == disk_bytes.end() ? 0 : it->second;
        if (local_ssd_manager_.RestoreHandoffClient(client, used) !=
            ErrorCode::OK) {
            return tl::make_unexpected("L4 client already registered");
        }
        if (client.total_capacity_bytes > 0) {
            MasterMetricManager::instance().inc_total_file_capacity(
                client.total_capacity_bytes);
            handoff_restored_file_capacity_ += client.total_capacity_bytes;
        }
        handoff_disk_gate_.opened.emplace(
            client.client_id, std::make_unique<std::atomic<bool>>(false));
    }
    handoff_disk_gate_.active.store(!handoff_disk_gate_.opened.empty(),
                                    std::memory_order_release);
    // Pins of in-flight offload sources (released at the quarantine
    // deadline together with the quarantined ranges).
    std::vector<HandoffOffloadPin> pins;
    pins.reserve(offload_pins.size());
    for (const auto& pin : offload_pins) {
        const auto object_id =
            MakeObjectIdentity(pin.key, TenantId(pin.tenant_id));
        MetadataAccessorRW accessor(this, object_id);
        if (!accessor.Exists()) continue;
        HandoffOffloadPin held{object_id.tenant_id, object_id.user_key, {}};
        accessor.Get().VisitReplicas(
            &Replica::fn_is_memory_replica, [&held](Replica& replica) {
                replica.inc_refcnt();
                held.replica_ids.push_back(replica.id());
            });
        if (!held.replica_ids.empty()) pins.push_back(std::move(held));
    }
    stats.offload_pins = pins.size();
    MasterMetricManager::instance().inc_mem_cache_nums(
        static_cast<int64_t>(memory_objects));
    MasterMetricManager::instance().inc_file_cache_nums(
        static_cast<int64_t>(stats.local_disk_replicas));
    {
        // Each range is held for the rest of the time the old master would
        // have held it, counted from now (the gap only adds to it).
        const auto quarantine_now = SteadyClock::now();
        std::lock_guard<std::mutex> lock(handoff_quarantine_.mutex);
        handoff_quarantine_.deadline =
            quarantine_now + put_start_release_timeout_sec_;
        for (size_t si = 0; si < nseg; ++si) {
            for (auto& [hold_ms, buffer] : quarantined_buffers[si]) {
                handoff_quarantine_.buffers.push_back(
                    {segments[si].segment_id,
                     quarantine_now + std::chrono::milliseconds(hold_ms),
                     std::move(buffer)});
            }
        }
        handoff_offload_pins_ = std::move(pins);
    }
    // Liveness clocks start now, when serving starts, not when the restore
    // started.
    const auto liveness_now = ClientLivenessRecord::Clock::now();
    for (auto& [client_id, record] : records) {
        (void)client_id;
        (void)record->Observe(liveness_now);
    }
    stats.finalize_ns = NsSince(phase_start);
    stats.total_ns = NsSince(total_start);
    stats.rss_kib_end = ProcStatusKiB("VmRSS");
    stats.hwm_kib_end = ProcStatusKiB("VmHWM");
    return stats;
}

// ----------------------------------------------------- remount integration

std::vector<UUID> MasterService::FindHandoffMismatchesLocked(
    const std::vector<Segment>& segments, const UUID& client_id,
    bool is_remount) const {
    std::vector<UUID> mismatched;
    for (const auto& [pending_id, pending] : pending_restored_segments_) {
        const Segment& restored = pending.segment;
        bool stale = false;
        bool listed_by_owner = false;
        for (const auto& segment : segments) {
            if (pending.owner != client_id) {
                // Another store process presents the transport endpoint the
                // restored segment was reachable at: the restored owner is
                // gone (store restarted during the roll).
                if (segment.te_endpoint == restored.te_endpoint) {
                    stale = true;
                }
                continue;
            }
            if (segment.id == pending_id) {
                listed_by_owner = true;
                if (!(segment == restored)) {
                    stale = true;  // same id, different base/size/endpoint
                }
            } else if (segment.te_endpoint == restored.te_endpoint &&
                       segment.base < restored.base + restored.size &&
                       restored.base < segment.base + segment.size) {
                // The owner re-registered this memory under a new segment.
                stale = true;
            }
        }
        if (is_remount && pending.owner == client_id && !listed_by_owner) {
            stale = true;  // the owner no longer has this segment
        }
        if (stale) {
            mismatched.push_back(pending_id);
        }
    }
    return mismatched;
}

void MasterService::DropPendingRestoredSegmentLocked(const UUID& segment_id,
                                                     const char* reason,
                                                     const UUID& requester) {
    const auto it = pending_restored_segments_.find(segment_id);
    if (it == pending_restored_segments_.end()) {
        return;
    }
    const PendingRestoredSegment pending = it->second;
    {
        ScopedSegmentAccess segment_access =
            segment_manager_.getSegmentAccess();
        size_t capacity = 0;
        if (segment_access.PrepareUnmountSegment(segment_id, capacity) ==
            ErrorCode::OK) {
            // Invalidates every restored buffer of the segment at once; the
            // objects become unreadable now and are erased by the cleanup
            // worker.
            (void)segment_access.CommitUnmountSegment(segment_id, pending.owner,
                                                      capacity);
        }
    }
    ForgetPendingRestoredSegmentLocked(it);
    if (requester != pending.owner && HandoffDiskGated(pending.owner)) {
        // Another store process took over the owner's endpoint: the owner
        // restarted during the roll. Drop its restored L4 registration so
        // its successor's re-registration can replace the old (still
        // unreadable) disk replicas; the TTL offboarding erases the rest.
        const auto capacity =
            local_ssd_manager_.UnregisterClient(pending.owner);
        if (capacity && *capacity > 0) {
            MasterMetricManager::instance().dec_total_file_capacity(*capacity);
        }
        LOG(WARNING) << "handoff_local_disk=dropped, owner=" << pending.owner
                     << ", reason=" << reason;
    }
    replica_cleanup_worker_.Schedule();
    handoff_segments_dropped_.fetch_add(1, std::memory_order_relaxed);
    LOG(WARNING) << "handoff_segment=dropped, segment_name="
                 << pending.segment.name
                 << ", endpoint=" << pending.segment.te_endpoint
                 << ", owner=" << pending.owner << ", reason=" << reason;
}

void MasterService::ForgetPendingRestoredSegmentLocked(
    std::unordered_map<UUID, PendingRestoredSegment,
                       boost::hash<UUID>>::iterator it) {
    const UUID segment_id = it->first;
    const PendingRestoredSegment pending = it->second;
    pending_restored_segments_.erase(it);
    bool alias_still_pending = false;
    for (const auto& [other_id, other] : pending_restored_segments_) {
        (void)other_id;
        if (other.segment.te_endpoint == pending.segment.te_endpoint ||
            other.segment.name == pending.segment.name ||
            other.segment.te_endpoint == pending.segment.name ||
            other.segment.name == pending.segment.te_endpoint) {
            alias_still_pending = true;
            break;
        }
    }
    if (!alias_still_pending) {
        invalid_replica_endpoints_.erase(pending.segment.te_endpoint);
        invalid_replica_endpoints_.erase(pending.segment.name);
    }
    std::vector<HandoffQuarantinedBuffer> released;
    {
        std::lock_guard<std::mutex> lock(handoff_quarantine_.mutex);
        auto& buffers = handoff_quarantine_.buffers;
        for (auto b = buffers.begin(); b != buffers.end();) {
            if (b->segment_id == segment_id) {
                released.push_back(std::move(*b));
                b = buffers.erase(b);
            } else {
                ++b;
            }
        }
    }
}

size_t MasterService::ReconcileHandoffPendingSegments() {
    {
        std::shared_lock<std::shared_mutex> client_lock(client_mutex_);
        if (pending_restored_segments_.empty()) return 0;
    }
    std::unique_lock<std::shared_mutex> client_lock(client_mutex_);
    std::unique_lock<std::shared_mutex> snapshot_lock(snapshot_mutex_);
    std::vector<UUID> gone;
    {
        ScopedSegmentAccess segment_access =
            segment_manager_.getSegmentAccess();
        for (const auto& [segment_id, pending] : pending_restored_segments_) {
            (void)pending;
            // Unmounted (e.g. by the TTL offboarding), or unmounted and
            // mounted again as a fresh, empty segment.
            if (!segment_access.IsRestoredSegmentPending(segment_id)) {
                gone.push_back(segment_id);
            }
        }
    }
    for (const auto& segment_id : gone) {
        const auto it = pending_restored_segments_.find(segment_id);
        if (it == pending_restored_segments_.end()) continue;
        LOG(WARNING) << "handoff_segment=forgotten, segment_name="
                     << it->second.segment.name
                     << ", endpoint=" << it->second.segment.te_endpoint
                     << ", owner=" << it->second.owner
                     << ", reason=unmounted_or_remounted_fresh";
        ForgetPendingRestoredSegmentLocked(it);
        handoff_segments_dropped_.fetch_add(1, std::memory_order_relaxed);
    }
    return gone.size();
}

void MasterService::ConfirmPendingRestoredSegmentsLocked(
    const std::vector<Segment>& segments, const UUID& client_id) {
    ScopedSegmentAccess segment_access = segment_manager_.getSegmentAccess();
    for (const auto& segment : segments) {
        const auto it = pending_restored_segments_.find(segment.id);
        if (it == pending_restored_segments_.end() ||
            it->second.owner != client_id || !(it->second.segment == segment)) {
            continue;
        }
        // An OK segment becomes allocatable; a DRAINING / DRAINED one keeps
        // its status (its drain job did not survive the roll; an operator
        // starts a new one or sets it back to OK).
        (void)segment_access.ConfirmRestoredSegment(segment.id);
        pending_restored_segments_.erase(it);
        handoff_segments_confirmed_.fetch_add(1, std::memory_order_relaxed);
        LOG(INFO) << "handoff_segment=confirmed, segment_name=" << segment.name
                  << ", owner=" << client_id;
    }
}

void MasterService::ReleaseExpiredHandoffQuarantine(
    std::chrono::steady_clock::time_point now) {
    std::vector<HandoffQuarantinedBuffer> released;
    std::vector<HandoffOffloadPin> pins;
    {
        std::lock_guard<std::mutex> lock(handoff_quarantine_.mutex);
        std::vector<HandoffQuarantinedBuffer> kept;
        for (auto& held : handoff_quarantine_.buffers) {
            if (held.deadline <= now) {
                released.push_back(std::move(held));
            } else {
                kept.push_back(std::move(held));
            }
        }
        handoff_quarantine_.buffers.swap(kept);
        if (now >= handoff_quarantine_.deadline) {
            pins.swap(handoff_offload_pins_);
        }
        if (released.empty() && pins.empty()) {
            return;
        }
    }
    size_t unpinned = 0;
    if (!pins.empty()) {
        std::shared_lock<std::shared_mutex> shared_lock(snapshot_mutex_);
        for (const auto& pin : pins) {
            MetadataAccessorRW accessor(this, {pin.tenant_id, pin.key});
            if (!accessor.Exists()) continue;
            // By id: the object may have been replaced meanwhile, and a
            // new replica must not lose a reference it never got.
            accessor.Get().VisitReplicas(
                [&pin](const Replica& replica) {
                    return std::find(pin.replica_ids.begin(),
                                     pin.replica_ids.end(),
                                     replica.id()) != pin.replica_ids.end();
                },
                [&unpinned](Replica& replica) {
                    replica.dec_refcnt();
                    ++unpinned;
                });
        }
    }
    LOG(INFO) << "handoff_quarantine=released, ranges=" << released.size()
              << ", offload_pins=" << pins.size()
              << ", unpinned_replicas=" << unpinned;
}

size_t MasterService::HandoffOffloadPinCount() const {
    std::lock_guard<std::mutex> lock(handoff_quarantine_.mutex);
    return handoff_offload_pins_.size();
}

// ------------------------------------------------------------ L4 gate

bool MasterService::HandoffDiskGated(const UUID& client_id) const {
    if (!handoff_disk_gate_.active.load(std::memory_order_acquire)) {
        return false;
    }
    const auto it = handoff_disk_gate_.opened.find(client_id);
    return it != handoff_disk_gate_.opened.end() &&
           !it->second->load(std::memory_order_acquire);
}

void MasterService::OpenHandoffDiskGate(const UUID& client_id) {
    if (!handoff_disk_gate_.active.load(std::memory_order_acquire)) {
        return;
    }
    const auto it = handoff_disk_gate_.opened.find(client_id);
    if (it == handoff_disk_gate_.opened.end() ||
        it->second->exchange(true, std::memory_order_acq_rel)) {
        return;
    }
    LOG(INFO) << "handoff_local_disk=opened, owner=" << client_id;
}

size_t MasterService::HandoffDiskClientsPending() const {
    if (!handoff_disk_gate_.active.load(std::memory_order_acquire)) return 0;
    size_t pending = 0;
    for (const auto& [client_id, opened] : handoff_disk_gate_.opened) {
        (void)client_id;
        if (!opened->load(std::memory_order_acquire)) ++pending;
    }
    return pending;
}

bool MasterService::EraseUnregisteredOwnerDiskReplica(
    MetadataShardAccessorRW& shard, ObjectMetadata& metadata,
    const UUID& client_id) {
    const auto stale = [this, &client_id](const Replica& replica) {
        if (!replica.is_local_disk_replica()) return false;
        const auto owner = replica.get_local_disk_client_id();
        return owner && *owner != client_id &&
               !HasMountedLocalDiskSegment(*owner);
    };
    if (!metadata.HasReplica(stale)) return false;
    const bool had_completed_disk = metadata.HasReplica([](const Replica& r) {
        return r.is_local_disk_replica() && r.is_completed();
    });
    if (EraseReplicasWithCacheTotalAccounting(metadata, stale) == 0) {
        return false;
    }
    shard.OnDiskReplicaRemoved(had_completed_disk, metadata);
    return true;
}

}  // namespace mooncake
