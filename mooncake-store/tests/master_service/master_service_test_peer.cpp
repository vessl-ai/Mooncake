#include "master_service/master_service_test_peer.h"

#include <algorithm>
#include <cassert>
#include <limits>
#include <mutex>

#include "storage/distributed/kvcs/kvcs_capacity_provider.h"

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

std::string MasterServiceTestPeer::SerializeMetadataForOpLogForTesting(
    MasterService& service, const ObjectMetadata& metadata) {
    return service.SerializeMetadataForOpLog(metadata);
}

std::string MasterServiceTestPeer::
    SerializeMetadataForOpLogFromReplicaDescriptorsForTesting(
        MasterService& service, const ObjectMetadata& metadata,
        const std::vector<Replica::Descriptor>& replicas) {
    return service.SerializeMetadataForOpLogFromReplicaDescriptors(metadata,
                                                                   replicas);
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

void MasterServiceTestPeer::SetDiskUnmountSweepStripeHookForTesting(
    std::function<void(size_t)> hook) {
    service_.disk_unmount_sweep_stripe_hook_ = std::move(hook);
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
    std::shared_lock<std::shared_mutex> state_lock(
        service_.master_state_mutex_);
    MasterService::MetadataAccessorRW accessor(
        &service_, MasterService::ObjectIdentity{tenant, key},
        MasterService::MetadataAccessorRW::CreateKey{.cleanup_invalid = false});
    assert(!accessor.IsPublished());
    accessor.Publish(owner, std::chrono::system_clock::now(), size,
                     std::move(replicas), std::nullopt, false,
                     ObjectDataType::UNKNOWN, "", tenant, key);
    MasterService::SyncCacheTotalAccounting(accessor.Get());
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
    const std::function<void(const Replica&)> on_removed =
        [&total](const Replica&) { ++total.removed_replicas; };
    std::optional<ErrorCode> persist_error;
    for (size_t stripe = 0; stripe < kStripeCount; ++stripe) {
        const auto start = std::chrono::steady_clock::now();
        service_.VisitNamespaces(
            [&](const TenantId& tenant_id, metadata::Namespace& ns) {
                service_.ClearStaleHandlesIn(tenant_id, ns, stripe, is_stale,
                                             &on_removed, persist_error);
            });
        const auto ns = static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - start)
                .count());
        total.stripe_ns += ns;
        total.max_stripe_ns = std::max(total.max_stripe_ns, ns);
    }
    return total;
}

void MasterServiceTestPeer::RunDiscardExpiredForTesting(
    std::chrono::system_clock::time_point now) {
    std::shared_lock<std::shared_mutex> state_lock(
        service_.master_state_mutex_);
    service_.VisitNamespaces(
        [&](const TenantId& /*tenant_id*/, metadata::Namespace& ns) {
            service_.DiscardExpiredProcessingReplicas(ns, now);
        });
}

void MasterServiceTestPeer::RunNoFBatchEvictForTesting(
    double evict_ratio_target, double evict_ratio_lowerbound) {
    service_.NoFBatchEvict(evict_ratio_target, evict_ratio_lowerbound);
}

void MasterServiceTestPeer::RunDfsEvictionForTesting() {
    service_.RunDfsEviction();
}

void MasterServiceTestPeer::ConfigureKvcsCapacityForTesting(
    std::unique_ptr<KvcsCapacityProvider> provider, bool eviction_enabled,
    double high_watermark, double low_watermark) {
    service_.enable_dfs_ = true;
    service_.use_kvcs_ = true;
    service_.dfs_kvcs_capacity_provider_ = std::move(provider);
    service_.dfs_kvcs_eviction_enabled_ = eviction_enabled;
    service_.dfs_kvcs_eviction_high_watermark_ = high_watermark;
    service_.dfs_kvcs_eviction_low_watermark_ = low_watermark;
    service_.InitKvcsDfsEviction();
}

void MasterServiceTestPeer::RefreshKvcsDfsUsageForTesting() {
    service_.RefreshKvcsDfsUsage();
}

void MasterServiceTestPeer::CleanupHttpMetadataForTesting(
    const std::string& segment_name) {
    service_.cleanupHttpMetadata(segment_name);
}

size_t MasterServiceTestPeer::HttpMetadataCleanupQueueSizeForTesting() const {
    std::lock_guard<std::mutex> lk(service_.http_metadata_cleanup_mutex_);
    return service_.http_metadata_cleanup_queue_.size();
}

void MasterServiceTestPeer::RunTenantEvictForTesting() {
    // The policy throttles its pass; a test asks for one now.
    service_.tenant_quota_policy_->next_eviction_check_ = {};
    service_.tenant_quota_policy_->OnMaintenance();
}

void MasterServiceTestPeer::SetKvTenantEpochTrackingForTesting(bool enabled) {
    service_.kv_track_tenant_epochs_ = enabled;
}

void MasterServiceTestPeer::SetRemoveAllTenantHookForTesting(
    std::function<void(const TenantId&)> hook) {
    service_.kv_remove_all_tenant_hook_ = std::move(hook);
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
    service_.nof_probe_fn_ = &MasterService::DefaultProbeNofSegment;
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
    return service_.RunPromotionCandidateRetry();
}

size_t MasterServiceTestPeer::CountCandidatesForTesting(
    const TenantId& tenant_id) {
    // The count is the tenant's share of the retry index; the candidate state
    // itself lives on each object.
    const auto& tracker = service_.promotion_candidates_;
    std::lock_guard<std::mutex> lock(tracker.mutex_);
    return std::count_if(
        tracker.due_.begin(), tracker.due_.end(),
        [&](const auto& entry) { return entry.first.tenant_id == tenant_id; });
}

void MasterServiceTestPeer::ResetCandidateBackoffsForTesting() {
    const auto epoch = std::chrono::steady_clock::time_point{};
    auto& tracker = service_.promotion_candidates_;
    // The index only names keys, so each key is resolved again under its
    // shard lock and refiled as due now.
    for (const auto& [tenant_id, key] :
         tracker.Due(std::chrono::steady_clock::time_point::max(),
                     std::numeric_limits<size_t>::max())) {
        MasterService::MetadataAccessorRW accessor(
            &service_, MasterService::ObjectIdentity{.tenant_id = tenant_id,
                                                     .user_key = key});
        if (accessor.Exists() && accessor.GetState().promotion_candidate) {
            accessor.GetState().promotion_candidate->retry_after = epoch;
            tracker.File(tenant_id, key, epoch);
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

}  // namespace mooncake::test
