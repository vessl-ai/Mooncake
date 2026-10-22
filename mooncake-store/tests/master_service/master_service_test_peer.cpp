#include "master_service/master_service_test_peer.h"

#include <algorithm>
#include <cassert>

#include "ha/snapshot/catalog/snapshot_catalog_store.h"
#ifdef USE_NOF
#include "spdk/spdk_wrapper.h"
#endif

namespace mooncake::test {

ErrorCode MasterServiceTestPeer::SetBatchOpLogBackendForTesting(
    std::shared_ptr<HaKvBackend> backend) {
    // Explicit test injection keeps the zero-view fixture API. A configured
    // view still exercises the production fenced path.
    return service_.InitializeBatchOpLogWriter(std::move(backend),
                                               service_.view_version_ > 0);
}

void MasterServiceTestPeer::SetBatchOpLogWriterFactoryForTesting(
    MasterService::BatchOpLogWriterFactory factory) {
    assert(factory);
    assert(!service_.ordered_oplog_writer_);
    service_.batch_oplog_writer_factory_ = std::move(factory);
}

void MasterServiceTestPeer::RunBatchEvictForTesting(
    double evict_ratio_target, double evict_ratio_lowerbound) {
    service_.BatchEvict(evict_ratio_target, evict_ratio_lowerbound);
}

void MasterServiceTestPeer::PrepareOffloadAdmissionSpareForTesting() {
    if (service_.offload_admission_filter_) {
        service_.offload_admission_filter_->PrepareSpare(SIZE_MAX);
    }
}

bool MasterServiceTestPeer::AdmitEagerOffloadForTesting(
    const std::string& key) {
    const auto object_id =
        service_.MakeObjectIdentity(key, TenantId::Default());
    MasterService::MetadataAccessorRW accessor(&service_, object_id);
    if (!accessor.Exists()) {
        return false;
    }
    return service_.AdmitEagerOffload(object_id, accessor.Get());
}

ReuseSampler* MasterServiceTestPeer::GetReuseSamplerForTesting() const {
    return service_.reuse_sampler_.get();
}

void MasterServiceTestPeer::RunOffloadStallMetricsUpdateForTesting() {
    service_.UpdateOffloadStallMetrics();
}

void MasterServiceTestPeer::SetDiskUnmountSweepShardHookForTesting(
    std::function<void(size_t)> hook) {
    service_.disk_unmount_sweep_shard_hook_ = std::move(hook);
}

bool MasterServiceTestPeer::WaitForDiskUnmountSweepsForTesting(
    std::chrono::milliseconds timeout) {
    std::unique_lock<std::shared_mutex> lock(
        service_.disk_unmount_sweep_mutex_);
    return service_.disk_unmount_sweep_cv_.wait_for(lock, timeout, [this] {
        return service_.disk_unmount_sweep_jobs_.empty();
    });
}

bool MasterServiceTestPeer::DiskUnmountSweepPendingForTesting(
    const UUID& owner) const {
    return service_.DiskUnmountSweepPending(owner);
}

void MasterServiceTestPeer::InsertDiskOnlyObjectForTesting(
    const std::string& key, uint64_t size, const UUID& owner,
    const std::string& endpoint) {
    std::shared_ptr<ClientLivenessRecord> record;
    {
        std::shared_lock<std::shared_mutex> client_lock(service_.client_mutex_);
        const auto it = service_.client_liveness_records_.find(owner);
        if (it != service_.client_liveness_records_.end()) {
            record = it->second;
        }
    }
    const TenantId tenant = TenantId::Default();
    std::vector<Replica> replicas;
    replicas.emplace_back(owner, size, endpoint, ReplicaStatus::COMPLETE,
                          record);
    std::shared_lock<std::shared_mutex> snapshot_lock(service_.snapshot_mutex_);
    MasterService::MetadataShardAccessorRW shard(
        &service_, service_.getShardIndex(tenant, key));
    auto& tenant_state = service_.GetOrCreateTenantState(shard.get(), tenant);
    auto [it, inserted] = tenant_state.metadata.emplace(
        std::piecewise_construct, std::forward_as_tuple(key),
        std::forward_as_tuple(owner, std::chrono::system_clock::now(), size,
                              std::move(replicas), std::nullopt, false,
                              ObjectDataType::UNKNOWN, "", tenant, key));
    assert(inserted);
    (void)inserted;
    shard.OnDiskReplicaAdded(it->second);
    service_.SyncCacheTotalAccounting(it->second);
    service_.local_ssd_manager_.AdjustUsedBytes(owner,
                                                static_cast<int64_t>(size));
}

MasterServiceTestPeer::OwnerSweepStats
MasterServiceTestPeer::SweepLocalDiskOwnerWithStatsForTesting(
    const UUID& owner) {
    const std::function<bool(const Replica&)> is_stale =
        [&owner](const Replica& replica) {
            return replica.is_local_disk_replica() && replica.is_completed() &&
                   replica.get_local_disk_client_id() == owner;
        };
    OwnerSweepStats total;
    for (size_t i = 0; i < MasterService::kNumShards; ++i) {
        MasterService::StaleHandleShardStats stats;
        (void)service_.ClearStaleHandlesInShard(i, is_stale, nullptr, &stats);
        total.keys_scanned += stats.keys_scanned;
        total.removed_replicas += stats.removed_replicas;
        total.scan_hold_ns += stats.scan_hold_ns;
        total.max_scan_hold_ns =
            std::max(total.max_scan_hold_ns, stats.scan_hold_ns);
        total.batch_hold_ns += stats.batch_hold_ns;
        total.max_batch_hold_ns =
            std::max(total.max_batch_hold_ns, stats.max_batch_hold_ns);
    }
    return total;
}

void MasterServiceTestPeer::RunDiscardExpiredForTesting(
    std::chrono::system_clock::time_point now) {
    std::shared_lock<std::shared_mutex> shared_lock(service_.snapshot_mutex_);
    for (size_t i = 0; i < MasterService::kNumShards; i++) {
        MasterService::MetadataShardAccessorRW shard(&service_, i);
        service_.DiscardExpiredProcessingReplicas(shard, now);
    }
}

void MasterServiceTestPeer::RunNoFBatchEvictForTesting(
    double evict_ratio_target, double evict_ratio_lowerbound) {
    service_.NoFBatchEvict(evict_ratio_target, evict_ratio_lowerbound);
}

void MasterServiceTestPeer::RunDfsEvictionForTesting() {
    service_.RunDfsEviction();
}

void MasterServiceTestPeer::RunTenantEvictForTesting() {
    service_.EvictTenantsOverWatermark();
}

void MasterServiceTestPeer::SetKvTenantEpochTrackingForTesting(bool enabled) {
    service_.kv_track_tenant_epochs_ = enabled;
}

void MasterServiceTestPeer::SetRemoveAllShardHookForTesting(
    std::function<void(size_t)> hook) {
    service_.kv_remove_all_shard_hook_ = std::move(hook);
}

uint64_t MasterServiceTestPeer::GetKvClearedPublishedForTesting() const {
    return service_.kv_cleared_published_.load(std::memory_order_relaxed);
}

uint64_t MasterServiceTestPeer::GetKvClearedSuppressedForTesting() const {
    return service_.kv_cleared_suppressed_by_epoch_.load(
        std::memory_order_relaxed);
}

void MasterServiceTestPeer::SetNoFProbeFnForTesting(
    MasterService::NoFProbeFn fn) {
#ifdef USE_NOF
    std::lock_guard<std::mutex> lock(service_.nof_probe_fn_mutex_);
    if (fn) {
        service_.nof_probe_fn_ = std::move(fn);
        return;
    }
    service_.nof_probe_fn_ = [](const std::string& te_endpoint,
                                uint32_t timeout_ms,
                                std::string* error_reason) {
        return SpdkWrapper::GetInstance().ProbeNofSegment(
            te_endpoint, timeout_ms, error_reason);
    };
#else
    (void)fn;
#endif
}

size_t MasterServiceTestPeer::GetMountedNoFSegmentCountForTesting() {
    std::vector<MountedNoFSegmentSnapshot> mounted_segments;
    service_.nof_segment_manager_.GetMountedSegmentsSnapshot(mounted_segments);
    return mounted_segments.size();
}

bool MasterServiceTestPeer::IsNoFSegmentMountedForTesting(
    const UUID& segment_id) {
    std::vector<MountedNoFSegmentSnapshot> mounted_segments;
    service_.nof_segment_manager_.GetMountedSegmentsSnapshot(mounted_segments);
    return std::any_of(
        mounted_segments.begin(), mounted_segments.end(),
        [&segment_id](const MountedNoFSegmentSnapshot& snapshot) {
            return snapshot.segment_id == segment_id &&
                   snapshot.status == SegmentStatus::OK;
        });
}

std::optional<uint32_t>
MasterServiceTestPeer::GetNoFHeartbeatFailureCountForTesting(
    const UUID& segment_id) {
    std::lock_guard<std::mutex> lock(service_.nof_heartbeat_mutex_);
    auto it = service_.nof_heartbeat_states_.find(segment_id);
    if (it == service_.nof_heartbeat_states_.end()) {
        return std::nullopt;
    }
    return it->second.consecutive_failures;
}

size_t MasterServiceTestPeer::RunPromotionCandidateRetryForTesting() {
    return service_.RunPromotionCandidateRetry(MasterService::kNumShards);
}

size_t MasterServiceTestPeer::CountCandidatesForTesting(
    const TenantId& tenant_id) {
    size_t count = 0;
    std::shared_lock<std::shared_mutex> lock(service_.snapshot_mutex_);
    for (size_t i = 0; i < MasterService::kNumShards; i++) {
        MetadataShardAccessorRO shard(&service_, i);
        auto it = shard->tenants.find(tenant_id);
        if (it != shard->tenants.end()) {
            count += it->second.promotion_candidates.size();
        }
    }
    return count;
}

void MasterServiceTestPeer::ResetCandidateBackoffsForTesting() {
    const auto epoch = std::chrono::steady_clock::time_point{};
    for (size_t i = 0; i < MasterService::kNumShards; i++) {
        MetadataShardAccessorRW shard(&service_, i);
        for (auto& [tenant_id, tenant_state] : shard->tenants) {
            (void)tenant_id;
            for (auto& [key, candidate] : tenant_state.promotion_candidates) {
                (void)key;
                candidate.retry_after = epoch;
            }
        }
    }
}

size_t MasterServiceTestPeer::SoftPinHeapSize() const {
    const auto& index = service_.soft_pin_deadline_index_;
    std::lock_guard lock(index.mutex_);
    return index.heap_.size();
}

size_t MasterServiceTestPeer::SoftPinRegistrationCount() const {
    const auto& index = service_.soft_pin_deadline_index_;
    std::lock_guard lock(index.mutex_);
    return index.registrations_.size();
}

std::unique_ptr<ha::SnapshotCatalogStore>
MasterServiceTestPeer::CreateSnapshotCatalogStore(
    const MasterServiceConfig& config) {
    return service_.CreateSnapshotCatalogStore(config);
}

}  // namespace mooncake::test
