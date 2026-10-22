#include "master_service.h"
#include "master_service/master_service_test_peer.h"

#include <glog/logging.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#include "master_metric_manager.h"
#include "types.h"

namespace mooncake::test {

class OffloadOnEvictTest : public ::testing::Test {
   protected:
    void SetUp() override {
        google::InitGoogleLogging("OffloadOnEvictTest");
        FLAGS_logtostderr = true;
    }

    void TearDown() override { google::ShutdownGoogleLogging(); }

    static constexpr size_t kDefaultSegmentBase = 0x300000000;

    Segment MakeSegment(std::string name, size_t base, size_t size) const {
        Segment segment;
        segment.id = generate_uuid();
        segment.name = std::move(name);
        segment.base = base;
        segment.size = size;
        segment.te_endpoint = segment.name;
        return segment;
    }

    struct MountedSegmentContext {
        UUID segment_id;
        UUID client_id;
    };

    MountedSegmentContext PrepareSegment(MasterService& service,
                                         std::string name, size_t base,
                                         size_t size) const {
        Segment segment = MakeSegment(std::move(name), base, size);
        UUID client_id = generate_uuid();
        auto mount_result = service.MountSegment(segment, client_id);
        EXPECT_TRUE(mount_result.has_value());
        return {.segment_id = segment.id, .client_id = client_id};
    }

    // Put an object and complete it.
    void PutObject(MasterService& service, const UUID& client_id,
                   const std::string& key, size_t size = 1024) {
        ReplicateConfig config;
        config.replica_num = 1;
        auto put_start =
            service.PutStart(client_id, key, TenantId::Default(), size, config);
        ASSERT_TRUE(put_start.has_value()) << "PutStart failed for key=" << key;
        auto put_end = service.PutEnd(client_id, key, TenantId::Default(),
                                      ReplicaType::MEMORY);
        ASSERT_TRUE(put_end.has_value()) << "PutEnd failed for key=" << key;
    }

    // Drain the offload queue via OffloadObjectHeartbeat.
    std::unordered_map<std::string, int64_t> DrainOffloadQueue(
        MasterService& service, const UUID& client_id) {
        auto res = service.OffloadObjectHeartbeat(client_id, true);
        if (!res) {
            return {};
        }
        std::unordered_map<std::string, int64_t> queued;
        for (const auto& task : res.value()) {
            queued[task.key] = task.size;
        }
        return queued;
    }
};

// =============================================================================
// UpsertStart interaction with an outstanding offload task.
//
// The task passes through two observable states in offloading_tasks[key]:
//   QUEUED    - mirror entry still present in offloading_objects; the store
//               worker has not observed the task. UpsertStart cancels the
//               task in place and allocates a fresh replica.
//   IN-FLIGHT - mirror already drained by OffloadObjectHeartbeat; the worker
//               is reading the source buffer for SSD write. UpsertStart
//               returns OBJECT_HAS_REPLICATION_TASK; the caller retries
//               after NotifyOffloadSuccess clears the marker.
// =============================================================================

TEST_F(OffloadOnEvictTest, UpsertPreemptsQueuedOffloadWithOffloadOnEvict) {
    // With offload_on_evict the task is queued by eviction rather than by
    // PutEnd.
    const uint64_t kv_lease_ttl = 500;
    MasterServiceConfig config;
    config.enable_offload = true;
    config.offload_on_evict = true;
    config.default_kv_lease_ttl = kv_lease_ttl;
    auto service = std::make_unique<MasterService>(config);

    constexpr size_t seg_size = 1024 * 1024 * 16;
    constexpr size_t object_size = 1024 * 1024;
    auto ctx =
        PrepareSegment(*service, "test_segment", kDefaultSegmentBase, seg_size);
    ASSERT_TRUE(service->MountLocalDiskSegment(ctx.client_id, true));

    std::vector<std::string> keys;
    for (int i = 0; i < 8; ++i) {
        keys.push_back("evict_upsert_" + std::to_string(i));
        PutObject(*service, ctx.client_id, keys.back(), object_size);
    }

    // PutEnd does not queue offloads in this mode; eviction does.
    auto queued = DrainOffloadQueue(*service, ctx.client_id);
    ASSERT_TRUE(queued.empty())
        << "offload_on_evict: PutEnd must not queue offloads";

    // Let leases expire so the keys are evictable, then run one eviction
    // cycle to queue their offloads without draining the mirrors.
    std::this_thread::sleep_for(std::chrono::milliseconds(kv_lease_ttl * 2));
    MasterServiceTestPeer(*service).RunBatchEvictForTesting(1.0, 1.0);

    std::string offloading_key;
    for (const auto& k : keys) {
        auto upsert =
            service->UpsertStart(ctx.client_id, k, TenantId::Default(),
                                 object_size, ReplicateConfig{});
        if (upsert.has_value()) {
            offloading_key = k;
            EXPECT_EQ(upsert->size(), 1u);
            EXPECT_EQ(upsert->at(0).status, ReplicaStatus::PROCESSING);
            ASSERT_TRUE(service
                            ->PutEnd(ctx.client_id, k, TenantId::Default(),
                                     ReplicaType::MEMORY)
                            .has_value());
            break;
        }
    }
    ASSERT_FALSE(offloading_key.empty())
        << "expected UpsertStart to preempt an offload queued by eviction";

    // Preempt cleared the marker and the mirror, so no stale entry is handed
    // to the store worker for the preempted key.
    auto stale = DrainOffloadQueue(*service, ctx.client_id);
    EXPECT_EQ(stale.count(offloading_key), 0u)
        << "preempted key must not remain queued";

    service->RemoveAll();
}

// =============================================================================
// Eviction-time offload, driven one BatchEvict cycle at a time.
//
// These tests pin the current behavior of the default eviction path in
// offload_on_evict mode (INF-651 C-1). Every object is 1 MiB, leases are zero
// so an object is an eviction candidate right after PutEnd, and the high
// watermark is 1.0 so the background eviction thread stays idle unless a
// PutStart fails. A store is a client id with one memory segment and, when
// `with_disk` is set, a LOCAL_DISK registration.
// =============================================================================

class OffloadOnEvictCycleTest : public ::testing::Test {
   protected:
    static constexpr size_t kSegmentSize = 64 * 1024 * 1024;
    static constexpr size_t kObjectSize = 1024 * 1024;

    struct Store {
        std::string segment_name;
        UUID client_id;
    };

    void SetUp() override {
        google::InitGoogleLogging("OffloadOnEvictCycleTest");
        FLAGS_logtostderr = true;
    }

    void TearDown() override {
        if (service_) {
            service_->RemoveAll();
            service_.reset();
        }
        google::ShutdownGoogleLogging();
    }

    static MasterServiceConfig Config() {
        MasterServiceConfig config;
        config.enable_offload = true;
        config.offload_on_evict = true;
        config.default_kv_lease_ttl = 0;
        config.eviction_ratio = 0.05;
        config.eviction_high_watermark_ratio = 1.0;
        return config;
    }

    void Start(const MasterServiceConfig& config) {
        service_ = std::make_unique<MasterService>(config);
    }

    Store AddStore(const std::string& name, bool with_disk,
                   size_t segment_size = kSegmentSize) {
        Segment segment;
        segment.id = generate_uuid();
        segment.name = name;
        segment.base = next_base_;
        segment.size = segment_size;
        segment.te_endpoint = name;
        next_base_ += 0x100000000ULL;
        Store store{name, generate_uuid()};
        EXPECT_TRUE(service_->MountSegment(segment, store.client_id));
        if (with_disk) {
            EXPECT_TRUE(service_->MountLocalDiskSegment(store.client_id, true));
            EXPECT_TRUE(
                service_->ReportSsdCapacity(store.client_id, 1LL << 34));
        }
        return store;
    }

    std::vector<std::string> Put(const Store& store, const std::string& prefix,
                                 int count, size_t object_size = kObjectSize) {
        std::vector<std::string> keys;
        for (int i = 0; i < count; ++i) {
            const std::string key = prefix + std::to_string(i);
            ReplicateConfig config;
            config.replica_num = 1;
            config.preferred_segment = store.segment_name;
            auto started = service_->PutStart(
                store.client_id, key, TenantId::Default(), object_size, config);
            EXPECT_TRUE(started.has_value()) << key;
            EXPECT_TRUE(service_->PutEnd(store.client_id, key,
                                         TenantId::Default(),
                                         ReplicaType::MEMORY));
            keys.push_back(key);
        }
        return keys;
    }

    // Overwrite the lease deadline of a key (the eviction order).
    void SetLease(const std::string& key,
                  std::chrono::system_clock::time_point deadline) {
        MasterServiceTestPeer::MetadataAccessorRW accessor(
            service_.get(),
            MasterServiceTestPeer::ObjectIdentity{TenantId::Default(), key});
        ASSERT_TRUE(accessor.Exists()) << key;
        SpinLocker locker(&accessor.Get().lock);
        accessor.Get().lease_->SetDeadline(deadline);
    }

    // One BatchEvict cycle with the given target (and lower bound).
    void Evict(double ratio = 1.0) {
        MasterServiceTestPeer(*service_).RunBatchEvictForTesting(ratio, ratio);
    }

    std::vector<OffloadTaskItem> Drain(const Store& store) {
        auto taken = service_->OffloadObjectHeartbeat(store.client_id, true);
        EXPECT_TRUE(taken.has_value());
        return taken ? *taken : std::vector<OffloadTaskItem>{};
    }

    void Report(const Store& store, const std::vector<OffloadTaskItem>& items,
                int64_t data_size) {
        std::vector<StorageObjectMetadata> metas;
        for (size_t i = 0; i < items.size(); ++i) {
            StorageObjectMetadata meta;
            meta.data_size = data_size;
            meta.transport_endpoint = store.segment_name;
            metas.push_back(meta);
        }
        EXPECT_TRUE(
            service_->NotifyOffloadSuccess(store.client_id, items, metas));
    }

    void Complete(const Store& store,
                  const std::vector<OffloadTaskItem>& items) {
        Report(store, items, static_cast<int64_t>(kObjectSize));
    }

    // The store's NACK: it did not persist the items.
    void Nack(const Store& store, const std::vector<OffloadTaskItem>& items) {
        Report(store, items, -1);
    }

    struct Copies {
        bool exists{false};
        size_t memory{0};
        size_t local_disk{0};
    };

    Copies CopiesOf(const std::string& key) {
        Copies copies;
        auto list = service_->GetReplicaList(key, TenantId::Default());
        if (!list) {
            EXPECT_EQ(list.error(), ErrorCode::OBJECT_NOT_FOUND) << key;
            return copies;
        }
        copies.exists = true;
        for (const auto& replica : list->replicas) {
            if (replica.is_memory_replica()) ++copies.memory;
            if (replica.is_local_disk_replica()) ++copies.local_disk;
        }
        return copies;
    }

    static std::vector<OffloadTaskItem> Select(
        const std::vector<OffloadTaskItem>& items,
        const std::vector<std::string>& keys) {
        std::vector<OffloadTaskItem> out;
        for (const auto& item : items) {
            if (std::find(keys.begin(), keys.end(), item.key) != keys.end()) {
                out.push_back(item);
            }
        }
        return out;
    }

    // A per-client offload counter, read from the Prometheus text the same
    // way master_metrics_test does. Client ids are fresh per test, so the
    // value belongs to this test only. Absent means zero.
    static int64_t ClientCounter(const std::string& metric, const UUID& id) {
        const std::string out =
            MasterMetricManager::instance().serialize_metrics();
        const std::string needle =
            metric + "{client_id=\"" + UuidToString(id) + "\"} ";
        const auto pos = out.find(needle);
        if (pos == std::string::npos) return 0;
        return std::stoll(out.substr(pos + needle.size()));
    }

    std::unique_ptr<MasterService> service_;
    uintptr_t next_base_{0x300000000};
};

TEST_F(OffloadOnEvictCycleTest, OneCycleQueuesEveryCandidateWhateverTheTarget) {
    // A queued object frees nothing and is not counted as evicted, so a cycle
    // with a 5% target keeps walking the candidates and queues all of them.
    Start(Config());
    const Store a = AddStore("store_a", /*with_disk=*/true);
    const auto keys = Put(a, "all_", 20);

    Evict(/*ratio=*/0.05);

    EXPECT_EQ(Drain(a).size(), keys.size());
    for (const auto& key : keys) {
        EXPECT_EQ(CopiesOf(key).memory, 1u) << key;
        EXPECT_EQ(CopiesOf(key).local_disk, 0u) << key;
    }
}

TEST_F(OffloadOnEvictCycleTest, MemoryIsFreedOnlyByTheCycleAfterCompletion) {
    Start(Config());
    const Store a = AddStore("store_a", /*with_disk=*/true);
    const auto keys = Put(a, "freed_", 6);
    auto& metrics = MasterMetricManager::instance();
    const int64_t evicted_before = metrics.get_evicted_key_count();
    const int64_t success_before = metrics.get_eviction_success();

    // Queueing alone counts as a successful cycle that evicted nothing.
    Evict();
    EXPECT_EQ(metrics.get_eviction_success(), success_before + 1);
    EXPECT_EQ(metrics.get_evicted_key_count(), evicted_before);

    const auto taken = Drain(a);
    ASSERT_EQ(taken.size(), keys.size());
    Complete(a, Select(taken, {keys[0], keys[1]}));

    // Completed: memory and disk (inclusive) until a later cycle picks it.
    EXPECT_EQ(CopiesOf(keys[0]).memory, 1u);
    EXPECT_EQ(CopiesOf(keys[0]).local_disk, 1u);
    EXPECT_EQ(CopiesOf(keys[2]).local_disk, 0u);

    Evict();
    for (int i = 0; i < 2; ++i) {
        EXPECT_EQ(CopiesOf(keys[i]).memory, 0u) << keys[i];
        EXPECT_EQ(CopiesOf(keys[i]).local_disk, 1u) << keys[i];
    }
    // Still being copied: pinned, so not even a candidate.
    for (int i = 2; i < 6; ++i) {
        EXPECT_EQ(CopiesOf(keys[i]).memory, 1u) << keys[i];
        EXPECT_EQ(CopiesOf(keys[i]).local_disk, 0u) << keys[i];
    }
    EXPECT_EQ(metrics.get_evicted_key_count(), evicted_before + 2);
    EXPECT_TRUE(Drain(a).empty()) << "pinned objects must not be re-queued";
}

TEST_F(OffloadOnEvictCycleTest, PutStartFailsUntilAnOffloadCompletes) {
    // DRAM full: the writer sees NO_AVAILABLE_HANDLE until a copy completes
    // and the next cycle frees the source. Queueing frees nothing.
    Start(Config());
    const Store a = AddStore("store_a", /*with_disk=*/true,
                             /*segment_size=*/8 * kObjectSize);
    ReplicateConfig rc;
    rc.replica_num = 1;
    auto put = [&](const std::string& key) {
        return service_->PutStart(a.client_id, key, TenantId::Default(),
                                  kObjectSize, rc);
    };
    std::vector<std::string> keys;
    for (int i = 0; i < 16; ++i) {
        const std::string key = "full_" + std::to_string(i);
        if (!put(key)) break;
        ASSERT_TRUE(service_->PutEnd(a.client_id, key, TenantId::Default(),
                                     ReplicaType::MEMORY));
        keys.push_back(key);
    }
    ASSERT_FALSE(keys.empty());
    auto& metrics = MasterMetricManager::instance();
    const int64_t alloc_failures_before =
        metrics.get_put_start_alloc_failures();

    auto first = put("full_new");
    ASSERT_FALSE(first.has_value());
    EXPECT_EQ(first.error(), ErrorCode::NO_AVAILABLE_HANDLE);
    Evict();
    auto second = put("full_new");
    ASSERT_FALSE(second.has_value());
    EXPECT_EQ(second.error(), ErrorCode::NO_AVAILABLE_HANDLE);
    EXPECT_GE(metrics.get_put_start_alloc_failures(),
              alloc_failures_before + 2);

    const auto taken = Drain(a);
    ASSERT_EQ(taken.size(), keys.size());
    Complete(a, Select(taken, {keys[0]}));
    Evict();
    EXPECT_EQ(CopiesOf(keys[0]).memory, 0u);
    EXPECT_TRUE(put("full_new").has_value());
}

TEST_F(OffloadOnEvictCycleTest, NackReleasesThePinAndTheNextCycleRequeues) {
    // A store that keeps NACKing (EIO, or an orphan file of the key) makes a
    // loop: every cycle queues the object again with a new task, and its
    // memory is never freed.
    Start(Config());
    const Store a = AddStore("store_a", /*with_disk=*/true);
    const auto keys = Put(a, "nack_", 3);
    const int64_t failed_before =
        ClientCounter("master_offload_failed_total", a.client_id);
    const int64_t enqueued_before =
        ClientCounter("master_offload_enqueued_total", a.client_id);

    std::map<std::string, L4TaskId> previous;
    for (int round = 0; round < 3; ++round) {
        Evict();
        const auto taken = Drain(a);
        ASSERT_EQ(taken.size(), keys.size()) << "round " << round;
        for (const auto& item : taken) {
            ASSERT_TRUE(item.task_id.has_value());
            auto it = previous.find(item.key);
            if (it != previous.end()) {
                EXPECT_NE(*item.task_id, it->second) << item.key;
            }
            previous[item.key] = *item.task_id;
        }
        Nack(a, taken);
        for (const auto& key : keys) {
            EXPECT_EQ(CopiesOf(key).memory, 1u) << key;
            EXPECT_EQ(CopiesOf(key).local_disk, 0u) << key;
        }
    }
    EXPECT_EQ(ClientCounter("master_offload_failed_total", a.client_id),
              failed_before + 9);
    EXPECT_EQ(ClientCounter("master_offload_enqueued_total", a.client_id),
              enqueued_before + 9);
}

TEST_F(OffloadOnEvictCycleTest, QueueOverflowSkipsAndCountsEveryAttempt) {
    // Without force the per-cycle cap is not used: a cycle queues up to the
    // store's mailbox limit (4 here, not the cap of 2), and every further
    // push is rejected. The rejected counter counts attempts, so it grows by
    // the same number again on the next cycle.
    MasterServiceConfig config = Config();
    config.offloading_queue_limit = 4;
    config.offload_cap_ratio = 0.5;
    Start(config);
    const Store a = AddStore("store_a", /*with_disk=*/true);
    const auto keys = Put(a, "overflow_", 10);
    const auto rejected = [&] {
        return ClientCounter("master_offload_enqueue_rejected_total",
                             a.client_id);
    };
    const int64_t rejected_before = rejected();

    Evict();
    EXPECT_EQ(rejected(), rejected_before + 6);
    Evict();
    EXPECT_EQ(rejected(), rejected_before + 12);
    for (const auto& key : keys) {
        EXPECT_TRUE(CopiesOf(key).exists) << key;
    }

    EXPECT_EQ(Drain(a).size(), 4u);
    // The mailbox is empty again: the next cycle queues four more of the six
    // that were left in memory, and rejects the last two.
    Evict();
    EXPECT_EQ(Drain(a).size(), 4u);
    EXPECT_EQ(rejected(), rejected_before + 14);
}

TEST_F(OffloadOnEvictCycleTest, ForceCapIsPerCycleAcrossStoresAndDropsTheRest) {
    // With force the cap (limit 4 x ratio 0.5 = 2) counts queued objects of
    // the whole cycle, not per store. Once it is reached the remaining
    // candidates are evicted without a disk copy.
    MasterServiceConfig config = Config();
    config.offload_force_evict = true;
    config.offloading_queue_limit = 4;
    config.offload_cap_ratio = 0.5;
    Start(config);
    const Store a = AddStore("store_a", /*with_disk=*/true);
    const Store b = AddStore("store_b", /*with_disk=*/true);
    auto keys = Put(a, "cap_a_", 5);
    const auto keys_b = Put(b, "cap_b_", 5);
    keys.insert(keys.end(), keys_b.begin(), keys_b.end());

    Evict();

    const size_t queued = Drain(a).size() + Drain(b).size();
    EXPECT_EQ(queued, 2u);
    size_t remaining = 0;
    for (const auto& key : keys) {
        if (CopiesOf(key).exists) ++remaining;
    }
    EXPECT_EQ(remaining, 2u) << "only the queued objects survive the cycle";
}

TEST_F(OffloadOnEvictCycleTest, MixedStoresFreezeOnlyTheStoreWithoutLocalDisk) {
    Start(Config());
    const Store a = AddStore("store_a", /*with_disk=*/true);
    const Store b = AddStore("store_b", /*with_disk=*/false);
    const auto keys_a = Put(a, "mixed_a_", 4);
    const auto keys_b = Put(b, "mixed_b_", 4);

    Evict();
    const auto taken = Drain(a);
    EXPECT_EQ(taken.size(), keys_a.size());
    Complete(a, taken);
    Evict();
    Evict();

    for (const auto& key : keys_a) {
        EXPECT_EQ(CopiesOf(key).memory, 0u) << key;
        EXPECT_EQ(CopiesOf(key).local_disk, 1u) << key;
    }
    for (const auto& key : keys_b) {
        EXPECT_EQ(CopiesOf(key).memory, 1u) << key;
    }
    // The push for store b fails with UNABLE_OFFLOADING, which the rejected
    // counter does not count: nothing reports the frozen store.
    EXPECT_EQ(
        ClientCounter("master_offload_enqueue_rejected_total", b.client_id), 0);
}

TEST_F(OffloadOnEvictCycleTest, MixedStoresEvictWithoutLocalDiskWhenOptedIn) {
    MasterServiceConfig config = Config();
    config.offload_evict_without_local_disk = true;
    Start(config);
    const Store a = AddStore("store_a", /*with_disk=*/true);
    const Store b = AddStore("store_b", /*with_disk=*/false);
    const auto keys_a = Put(a, "optin_a_", 4);
    const auto keys_b = Put(b, "optin_b_", 4);

    Evict();

    for (const auto& key : keys_b) {
        EXPECT_FALSE(CopiesOf(key).exists) << key;
    }
    // The store with a disk tier is unchanged: queued, not freed.
    const auto taken = Drain(a);
    EXPECT_EQ(taken.size(), keys_a.size());
    for (const auto& key : keys_a) {
        EXPECT_EQ(CopiesOf(key).memory, 1u) << key;
    }
    Complete(a, taken);
    Evict();
    for (const auto& key : keys_a) {
        EXPECT_EQ(CopiesOf(key).memory, 0u) << key;
        EXPECT_EQ(CopiesOf(key).local_disk, 1u) << key;
    }
}

TEST_F(OffloadOnEvictCycleTest, OptInStillSkipsWhenTheQueueIsFull) {
    // A full mailbox is not "no disk tier": the opt-in leaves those objects
    // in memory, as the default does.
    MasterServiceConfig config = Config();
    config.offload_evict_without_local_disk = true;
    config.offloading_queue_limit = 2;
    Start(config);
    const Store a = AddStore("store_a", /*with_disk=*/true);
    const auto keys = Put(a, "full_q_", 5);

    Evict();

    for (const auto& key : keys) {
        EXPECT_TRUE(CopiesOf(key).exists) << key;
    }
    EXPECT_EQ(Drain(a).size(), 2u);
}

TEST_F(OffloadOnEvictCycleTest, FenceReleasesQueuedTasksAndFreezesMemory) {
    // A disk fence first sends OffloadObjectHeartbeat(false): the queued
    // tasks end and their sources are unpinned, and the store takes no new
    // offloads, so eviction skips its objects.
    Start(Config());
    const Store a = AddStore("store_a", /*with_disk=*/true);
    const auto keys = Put(a, "fence_", 4);

    Evict();
    auto fenced = service_->OffloadObjectHeartbeat(a.client_id, false);
    ASSERT_TRUE(fenced.has_value());
    EXPECT_TRUE(fenced->empty());

    Evict();
    for (const auto& key : keys) {
        EXPECT_EQ(CopiesOf(key).memory, 1u) << key;
    }
    // Re-enabled: nothing was queued while fenced, and the objects are
    // unpinned candidates again.
    EXPECT_TRUE(Drain(a).empty());
    Evict();
    EXPECT_EQ(Drain(a).size(), keys.size());
}

TEST_F(OffloadOnEvictCycleTest, FenceEvictsQueuedObjectsWhenOptedIn) {
    MasterServiceConfig config = Config();
    config.offload_evict_without_local_disk = true;
    Start(config);
    const Store a = AddStore("store_a", /*with_disk=*/true);
    const auto keys = Put(a, "fence_optin_", 4);

    Evict();
    ASSERT_TRUE(service_->OffloadObjectHeartbeat(a.client_id, false));
    Evict();

    for (const auto& key : keys) {
        EXPECT_FALSE(CopiesOf(key).exists) << key;
    }
}

TEST_F(OffloadOnEvictCycleTest, FenceKeepsTasksTheStoreAlreadyTook) {
    // OffloadObjectHeartbeat(false) ends only tasks still in the mailbox.
    // A task the store already took keeps its source pinned until the store
    // reports it (a fenced store NACKs its carried tasks) or the task
    // expires; the opt-in cannot evict a pinned source either.
    MasterServiceConfig config = Config();
    config.offload_evict_without_local_disk = true;
    Start(config);
    const Store a = AddStore("store_a", /*with_disk=*/true);
    const auto keys = Put(a, "fence_taken_", 3);

    Evict();
    const auto taken = Drain(a);
    ASSERT_EQ(taken.size(), keys.size());
    ASSERT_TRUE(service_->OffloadObjectHeartbeat(a.client_id, false));
    Evict();
    for (const auto& key : keys) {
        EXPECT_EQ(CopiesOf(key).memory, 1u) << key;
    }

    Nack(a, taken);
    Evict();
    for (const auto& key : keys) {
        EXPECT_FALSE(CopiesOf(key).exists) << key;
    }
}

TEST_F(OffloadOnEvictCycleTest, UnmountSweepsDiskOnlyObjectsAndFreezesTheRest) {
    // After a fence the store unmounts its disk tier. The unmount erases the
    // objects whose only copy was on that disk, and leaves the memory-only
    // objects stuck. Tasks that were queued or taken before the unmount are
    // not released by it; their sources stay pinned.
    Start(Config());
    const Store a = AddStore("store_a", /*with_disk=*/true);
    const auto keys = Put(a, "unmount_", 5);

    Evict();
    const auto taken = Drain(a);
    ASSERT_EQ(taken.size(), keys.size());
    Complete(a, Select(taken, {keys[0], keys[1]}));
    Nack(a, Select(taken, {keys[3], keys[4]}));
    Evict();  // frees keys 0 and 1 (disk only); re-queues 3 and 4
    ASSERT_EQ(CopiesOf(keys[0]).memory, 0u);
    ASSERT_EQ(CopiesOf(keys[0]).local_disk, 1u);
    // keys[2] is taken (copying); keys[3], keys[4] are queued again.

    ASSERT_TRUE(service_->UnmountLocalDiskSegment(a.client_id));

    EXPECT_FALSE(CopiesOf(keys[0]).exists);
    EXPECT_FALSE(CopiesOf(keys[1]).exists);
    Evict();
    for (int i = 2; i < 5; ++i) {
        EXPECT_EQ(CopiesOf(keys[i]).memory, 1u) << keys[i];
    }
}

TEST_F(OffloadOnEvictCycleTest, UnmountedStoreIsEvictedWhenOptedIn) {
    // With the opt-in, a memory-only object of the unmounted store is
    // evicted. One whose task was outstanding at the unmount stays pinned.
    MasterServiceConfig config = Config();
    config.offload_evict_without_local_disk = true;
    Start(config);
    const Store a = AddStore("store_a", /*with_disk=*/true);
    const auto keys = Put(a, "unmount_optin_", 3);

    Evict();
    const auto taken = Drain(a);
    ASSERT_EQ(taken.size(), keys.size());
    Nack(a, Select(taken, {keys[0]}));  // keys[0] is back in memory only
    ASSERT_TRUE(service_->UnmountLocalDiskSegment(a.client_id));
    Evict();

    EXPECT_FALSE(CopiesOf(keys[0]).exists);
    EXPECT_EQ(CopiesOf(keys[1]).memory, 1u);
    EXPECT_EQ(CopiesOf(keys[2]).memory, 1u);
}

TEST_F(OffloadOnEvictCycleTest, DiskCopyEvictedByTheStoreIsWrittenAgain) {
    // An object whose disk copy the store's bucket eviction removed while it
    // was still in memory goes back to the queue on its next eviction.
    Start(Config());
    const Store a = AddStore("store_a", /*with_disk=*/true);
    const auto keys = Put(a, "rewrite_", 1);

    Evict();
    const auto taken = Drain(a);
    ASSERT_EQ(taken.size(), 1u);
    Complete(a, taken);
    ASSERT_EQ(CopiesOf(keys[0]).local_disk, 1u);
    ASSERT_TRUE(service_->EvictDiskReplica(
        a.client_id, keys[0], TenantId::Default(), ReplicaType::LOCAL_DISK));
    ASSERT_EQ(CopiesOf(keys[0]).local_disk, 0u);
    ASSERT_EQ(CopiesOf(keys[0]).memory, 1u);

    Evict();
    const auto again = Drain(a);
    ASSERT_EQ(again.size(), 1u);
    ASSERT_TRUE(again[0].task_id.has_value());
    EXPECT_NE(*again[0].task_id, *taken[0].task_id);
}

// --- Offload stall metrics (enable_offload_stall_metrics) ---------------
//
// The metric manager is process-wide, so these tests compare gauges after
// the action they cause and counters as deltas.

TEST_F(OffloadOnEvictCycleTest, StallGaugeCountsObjectsKeptForNoLocalDisk) {
    MasterServiceConfig config = Config();
    config.enable_offload_stall_metrics = true;
    Start(config);
    const Store a = AddStore("store_a", /*with_disk=*/true);
    const Store b = AddStore("store_b", /*with_disk=*/false);
    Put(a, "stall_a_", 4);
    Put(b, "stall_b_", 6);
    auto& metrics = MasterMetricManager::instance();

    Evict();
    // Only store_b's objects are kept for lack of a disk tier; store_a's are
    // queued. The gauge is per cycle, not cumulative.
    EXPECT_EQ(metrics.get_offload_no_local_disk_skipped_objects(), 6);
    Evict();
    EXPECT_EQ(metrics.get_offload_no_local_disk_skipped_objects(), 6);
}

TEST_F(OffloadOnEvictCycleTest, StallCounterCountsOptInEvictions) {
    MasterServiceConfig config = Config();
    config.enable_offload_stall_metrics = true;
    config.offload_evict_without_local_disk = true;
    Start(config);
    const Store b = AddStore("store_b", /*with_disk=*/false);
    Put(b, "optin_metric_", 5);
    auto& metrics = MasterMetricManager::instance();
    const int64_t evicted_before = metrics.get_offload_no_local_disk_evicted();

    Evict();
    EXPECT_EQ(metrics.get_offload_no_local_disk_evicted() - evicted_before, 5);
    EXPECT_EQ(metrics.get_offload_no_local_disk_skipped_objects(), 0);
}

TEST_F(OffloadOnEvictCycleTest, StallGaugesShowTasksLeftByAnUnmount) {
    MasterServiceConfig config = Config();
    config.enable_offload_stall_metrics = true;
    Start(config);
    const Store a = AddStore("store_a", /*with_disk=*/true);
    const auto keys = Put(a, "unmount_metric_", 5);
    auto& metrics = MasterMetricManager::instance();
    MasterServiceTestPeer peer(*service_);

    Evict();
    const auto taken = Drain(a);
    ASSERT_EQ(taken.size(), keys.size());
    peer.RunOffloadStallMetricsUpdateForTesting();
    EXPECT_EQ(metrics.get_offload_tasks_pending(), 5);
    EXPECT_EQ(metrics.get_offload_tasks_without_local_disk(), 0);
    EXPECT_GE(metrics.get_offload_task_oldest_age_seconds(), 0);

    // The unmount leaves the tasks (and the pinned sources) in place, and
    // no store can finish them any more.
    ASSERT_TRUE(service_->UnmountLocalDiskSegment(a.client_id));
    peer.RunOffloadStallMetricsUpdateForTesting();
    EXPECT_EQ(metrics.get_offload_tasks_pending(), 5);
    EXPECT_EQ(metrics.get_offload_tasks_without_local_disk(), 5);

    // Only the expiry sweep releases them.
    const int64_t expired_before = metrics.get_offload_tasks_expired();
    peer.RunDiscardExpiredForTesting(std::chrono::system_clock::now() +
                                     std::chrono::hours(24));
    EXPECT_EQ(metrics.get_offload_tasks_expired() - expired_before, 5);
    peer.RunOffloadStallMetricsUpdateForTesting();
    EXPECT_EQ(metrics.get_offload_tasks_pending(), 0);
    EXPECT_EQ(metrics.get_offload_tasks_without_local_disk(), 0);
    EXPECT_EQ(metrics.get_offload_task_oldest_age_seconds(), 0);
}

TEST_F(OffloadOnEvictCycleTest, StallMetricsStayUntouchedByDefault) {
    Start(Config());
    const Store a = AddStore("store_a", /*with_disk=*/true);
    const Store b = AddStore("store_b", /*with_disk=*/false);
    Put(a, "quiet_a_", 3);
    Put(b, "quiet_b_", 3);
    auto& metrics = MasterMetricManager::instance();
    // Park the gauges on values the default path would never write.
    metrics.set_offload_no_local_disk_skipped_objects(-7);
    metrics.set_offload_tasks_pending(-7);
    const int64_t expired_before = metrics.get_offload_tasks_expired();
    const int64_t evicted_before = metrics.get_offload_no_local_disk_evicted();

    Evict();
    Drain(a);
    ASSERT_TRUE(service_->UnmountLocalDiskSegment(a.client_id));
    MasterServiceTestPeer(*service_).RunDiscardExpiredForTesting(
        std::chrono::system_clock::now() + std::chrono::hours(24));

    EXPECT_EQ(metrics.get_offload_no_local_disk_skipped_objects(), -7);
    EXPECT_EQ(metrics.get_offload_tasks_pending(), -7);
    EXPECT_EQ(metrics.get_offload_tasks_expired(), expired_before);
    EXPECT_EQ(metrics.get_offload_no_local_disk_evicted(), evicted_before);
}

// =============================================================================
// Which keys a cycle queues, and in what order (INF-651 hole analysis).
//
// 2000 objects of 4 KiB, all expired, key i with the i-th oldest lease. A 5%
// target asks for 100 objects; the frontier is those plus the reserve slack
// (1024), i.e. the 1124 oldest keys. Queueing frees nothing, so the first
// pass never reaches the target and the refill pass walks the 876 newer keys
// in metadata (hash) order. A 50% target (above the compact-frontier ratio)
// takes every candidate at once; its frontier is the 1000 oldest.
// =============================================================================

class OffloadFrontierTest : public OffloadOnEvictCycleTest {
   protected:
    static constexpr int kKeys = 2000;
    static constexpr size_t kSmallObject = 4096;
    static constexpr int kCompactFrontier = 1124;

    std::vector<std::string> PutAgedKeys(const Store& store) {
        auto keys = Put(store, "lease_", kKeys, kSmallObject);
        for (int i = 0; i < kKeys; ++i) {
            SetLease(keys[i], base_ + std::chrono::milliseconds(i));
        }
        return keys;
    }

    // Index (lease rank) of each drained key, in task id order (the order
    // the master queued them).
    static std::vector<int> RanksInQueueOrder(
        std::vector<OffloadTaskItem> items) {
        std::sort(items.begin(), items.end(), [](const auto& a, const auto& b) {
            return a.task_id->seq < b.task_id->seq;
        });
        std::vector<int> ranks;
        for (const auto& item : items) {
            ranks.push_back(std::stoi(item.key.substr(6)));  // "lease_<i>"
        }
        return ranks;
    }

    static std::vector<int> Range(int begin, int end) {
        std::vector<int> out;
        for (int i = begin; i < end; ++i) out.push_back(i);
        return out;
    }

    MasterServiceConfig FrontierConfig(bool frontier_only) {
        MasterServiceConfig config = Config();
        config.offload_evict_frontier_only = frontier_only;
        return config;
    }

    const std::chrono::system_clock::time_point base_ =
        std::chrono::system_clock::now() - std::chrono::hours(1);
};

TEST_F(OffloadFrontierTest, ByDefaultTheRefillQueuesKeysNewerThanTheFrontier) {
    Start(FrontierConfig(false));
    const Store a = AddStore("store_a", /*with_disk=*/true);
    PutAgedKeys(a);

    Evict(/*ratio=*/0.05);

    const auto ranks = RanksInQueueOrder(Drain(a));
    ASSERT_EQ(ranks.size(), static_cast<size_t>(kKeys));
    // The 876 refill keys are queued after the frontier, in hash order.
    const std::vector<int> refill(ranks.begin() + kCompactFrontier,
                                  ranks.end());
    EXPECT_EQ(*std::min_element(refill.begin(), refill.end()),
              kCompactFrontier);
    EXPECT_FALSE(std::is_sorted(refill.begin(), refill.end()));
}

TEST_F(OffloadFrontierTest, FrontierOnlyQueuesTheOldestLeasesInLeaseOrder) {
    Start(FrontierConfig(true));
    const Store a = AddStore("store_a", /*with_disk=*/true);
    const auto keys = PutAgedKeys(a);

    Evict(/*ratio=*/0.05);

    EXPECT_EQ(RanksInQueueOrder(Drain(a)), Range(0, kCompactFrontier));
    // Past the frontier nothing is queued, and nothing is lost.
    for (int i = kCompactFrontier; i < kKeys; ++i) {
        ASSERT_EQ(CopiesOf(keys[i]).memory, 1u) << keys[i];
    }
    // The next cycle takes the next oldest: the queued keys are pinned.
    Evict(/*ratio=*/0.05);
    EXPECT_EQ(RanksInQueueOrder(Drain(a)), Range(kCompactFrontier, kKeys));
}

TEST_F(OffloadFrontierTest, FrontierOnlyFullCandidateScanQueuesTheEvictNum) {
    // A 50% target skips the compact frontier: every candidate is collected
    // and the frontier is the evict_num (1000) oldest.
    for (const bool frontier_only : {false, true}) {
        SCOPED_TRACE(frontier_only);
        Start(FrontierConfig(frontier_only));
        const Store a = AddStore("store_a", /*with_disk=*/true);
        PutAgedKeys(a);

        Evict(/*ratio=*/0.5);

        const auto ranks = RanksInQueueOrder(Drain(a));
        if (frontier_only) {
            EXPECT_EQ(ranks, Range(0, kKeys / 2));
        } else {
            EXPECT_EQ(ranks.size(), static_cast<size_t>(kKeys));
        }
        service_->RemoveAll();
    }
}

TEST_F(OffloadFrontierTest, FrontierOnlyStillFreesDiskBackedKeysPastIt) {
    // Keys 0..49 get a disk copy and then the newest leases, so the next
    // cycle meets them in the refill pass. Both modes free them there (they
    // are safe to drop from memory); only the default also queues the S2
    // keys of the refill.
    for (const bool frontier_only : {false, true}) {
        SCOPED_TRACE(frontier_only);
        Start(FrontierConfig(frontier_only));
        const Store a = AddStore("store_a", /*with_disk=*/true);
        const auto keys = PutAgedKeys(a);
        constexpr int kOnDisk = 50;

        Evict(/*ratio=*/0.05);
        const auto taken = Drain(a);
        std::vector<OffloadTaskItem> done, nacked;
        for (const auto& item : taken) {
            (std::stoi(item.key.substr(6)) < kOnDisk ? done : nacked)
                .push_back(item);
        }
        ASSERT_EQ(done.size(), static_cast<size_t>(kOnDisk));
        Report(a, done, static_cast<int64_t>(kSmallObject));  // completed
        Nack(a, nacked);
        for (int i = 0; i < kOnDisk; ++i) {
            SetLease(keys[i], base_ + std::chrono::minutes(30) +
                                  std::chrono::milliseconds(i));
        }

        Evict(/*ratio=*/0.05);

        for (int i = 0; i < kOnDisk; ++i) {
            EXPECT_EQ(CopiesOf(keys[i]).memory, 0u) << keys[i];
            EXPECT_EQ(CopiesOf(keys[i]).local_disk, 1u) << keys[i];
        }
        const auto ranks = RanksInQueueOrder(Drain(a));
        if (frontier_only) {
            EXPECT_EQ(ranks, Range(kOnDisk, kOnDisk + kCompactFrontier));
        } else {
            EXPECT_EQ(ranks.size(), static_cast<size_t>(kKeys - kOnDisk));
        }
        service_->RemoveAll();
    }
}

// =============================================================================
// Offload admission (INF-651 C4-1).
//
// (a) offload_admission_second_put: eager mode, PutEnd queues the offload
//     only for a key put before within the admission window.
// (b) offload_admission_read_since_put: offload_on_evict, eviction queues an
//     offload only for an object read from memory since its PutEnd; others
//     are evicted without a disk copy.
//
// Inspection here goes through the metadata accessor (Peek, Unread), not
// GetReplicaList, because a GetReplicaList is itself a read for (b).
// =============================================================================

TEST(OffloadAdmissionFilterTest, RemembersAtLeastOneWindowAndAtMostTwo) {
    constexpr size_t kWindow = 1000;
    OffloadAdmissionFilter filter(kWindow);
    EXPECT_EQ(filter.MemoryBytes(), 3u * 4u * 2u * kWindow);

    // The first window: nothing was put before (up to false positives).
    size_t false_seen = 0;
    for (size_t i = 0; i < kWindow; ++i) {
        false_seen += filter.ObservePut("a" + std::to_string(i));
    }
    EXPECT_EQ(filter.rotations(), 1u);
    EXPECT_LE(false_seen, kWindow * 2 / 100);
    // One window later every key is still remembered: no false negatives.
    for (size_t i = 0; i < kWindow; ++i) {
        ASSERT_TRUE(filter.Seen("a" + std::to_string(i))) << i;
    }
    // A second window of other keys retires the first generation.
    EXPECT_TRUE(filter.PrepareSpare(SIZE_MAX));
    for (size_t i = 0; i < kWindow; ++i) {
        filter.ObservePut("b" + std::to_string(i));
    }
    EXPECT_EQ(filter.rotations(), 2u);
    EXPECT_EQ(filter.deferred_rotations(), 0u);
    size_t still_seen = 0;
    for (size_t i = 0; i < kWindow; ++i) {
        ASSERT_TRUE(filter.Seen("b" + std::to_string(i))) << i;
        still_seen += filter.Seen("a" + std::to_string(i));
    }
    // Only false positives of the half-full previous generation remain.
    EXPECT_LE(still_seen, kWindow * 5 / 100);
}

TEST(OffloadAdmissionFilterTest, RotationWaitsForAZeroedSpare) {
    constexpr size_t kWindow = 100;
    OffloadAdmissionFilter filter(kWindow);
    // The spare made at construction serves the first rotation.
    for (size_t i = 0; i < kWindow; ++i) {
        filter.ObservePut("a" + std::to_string(i));
    }
    ASSERT_EQ(filter.rotations(), 1u);
    // No spare until the retired generation is zeroed: the window is
    // reached again but the rotation waits, and puts stay in the current
    // generation.
    for (size_t i = 0; i < kWindow + 50; ++i) {
        filter.ObservePut("b" + std::to_string(i));
    }
    EXPECT_EQ(filter.rotations(), 1u);
    EXPECT_EQ(filter.deferred_rotations(), 1u);
    EXPECT_TRUE(filter.Seen("a0"));  // previous generation still held
    // Zeroing in slices: 4 rows of 2 * kWindow counters, 150 per call.
    int calls = 0;
    while (!filter.PrepareSpare(/*max_bytes=*/150)) {
        ASSERT_LT(++calls, 100);
    }
    EXPECT_EQ(calls, 5);  // 800 counters: 5 slices short, the 6th finishes
    // The next put rotates: "b" keys move to previous, "a" keys drop out.
    filter.ObservePut("c0");
    EXPECT_EQ(filter.rotations(), 2u);
    EXPECT_TRUE(filter.Seen("b0"));
    size_t a_seen = 0;
    for (size_t i = 0; i < kWindow; ++i) {
        a_seen += filter.Seen("a" + std::to_string(i));
    }
    // Only false positives of "b", whose generation filled past half
    // (150 puts in 200 counters per row) while the rotation waited.
    EXPECT_LE(a_seen, kWindow * 20 / 100);
    // The zeroed generation reads as empty.
    EXPECT_FALSE(filter.Seen("never-put"));
}

namespace {

// A key in the shape SGLang writes: <64 hex digits>_<rank>_<k|v>.
std::string PageKey(uint64_t page, int rank, char half) {
    uint64_t x = page;
    std::string hex;
    for (int lane = 0; lane < 4; ++lane) {
        x += 0x9e3779b97f4a7c15ULL;
        uint64_t z = x;
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
        z ^= z >> 31;
        char buf[17];
        snprintf(buf, sizeof(buf), "%016llx",
                 static_cast<unsigned long long>(z));
        hex += buf;
    }
    return hex + "_" + std::to_string(rank) + "_" + half;
}

// Share of first puts the filter calls "put before", over the second and
// third windows of distinct keys (a full previous generation from then on).
double SteadyFalseSeenRate(OffloadAdmissionFilter::Kind kind, size_t window) {
    OffloadAdmissionFilter filter(window, kind);
    uint64_t page = 0;
    size_t false_seen = 0;
    size_t asked = 0;
    for (int w = 0; w < 3; ++w) {
        EXPECT_TRUE(filter.PrepareSpare(SIZE_MAX));
        for (size_t i = 0; i < window; i += 2, ++page) {
            for (char half : {'k', 'v'}) {
                const bool seen = filter.ObservePut(PageKey(page, 0, half));
                if (w > 0) {
                    false_seen += seen;
                    ++asked;
                }
            }
        }
    }
    EXPECT_EQ(filter.deferred_rotations(), 0u);
    return static_cast<double>(false_seen) / static_cast<double>(asked);
}

}  // namespace

TEST(OffloadAdmissionFilterTest, BloomFalseSeenIsFarBelowTheSketch) {
    // The rig run (window 2^20) admitted 0.415 of puts where the model with
    // no false positives admits 0.400: about 2% of first puts were admitted
    // by mistake. Same window, same memory, keys of the same shape.
    constexpr size_t kWindow = 1 << 20;
    const double sketch = SteadyFalseSeenRate(
        OffloadAdmissionFilter::Kind::kCountMinSketch, kWindow);
    const double bloom =
        SteadyFalseSeenRate(OffloadAdmissionFilter::Kind::kBloom, kWindow);
    LOG(INFO) << "false seen rate: CountMinSketch " << sketch << ", Bloom "
              << bloom;
    EXPECT_GT(sketch, 0.005);
    EXPECT_LT(sketch, 0.05);
    EXPECT_LT(bloom, 1e-4);
}

TEST(OffloadAdmissionFilterTest, BloomRemembersAtLeastOneWindowAndAtMostTwo) {
    constexpr size_t kWindow = 1000;
    OffloadAdmissionFilter filter(kWindow,
                                  OffloadAdmissionFilter::Kind::kBloom);
    EXPECT_EQ(filter.kind(), OffloadAdmissionFilter::Kind::kBloom);
    // Same memory as the CountMinSketch filter.
    EXPECT_EQ(filter.MemoryBytes(), 3u * 4u * 2u * kWindow);

    size_t false_seen = 0;
    for (size_t i = 0; i < kWindow; ++i) {
        false_seen += filter.ObservePut("a" + std::to_string(i));
    }
    EXPECT_EQ(filter.rotations(), 1u);
    EXPECT_LE(false_seen, 1u);
    for (size_t i = 0; i < kWindow; ++i) {
        ASSERT_TRUE(filter.Seen("a" + std::to_string(i))) << i;
    }
    EXPECT_TRUE(filter.PrepareSpare(SIZE_MAX));
    false_seen = 0;
    for (size_t i = 0; i < kWindow; ++i) {
        false_seen += filter.ObservePut("b" + std::to_string(i));
    }
    EXPECT_LE(false_seen, 1u);
    EXPECT_EQ(filter.rotations(), 2u);
    size_t still_seen = 0;
    for (size_t i = 0; i < kWindow; ++i) {
        ASSERT_TRUE(filter.Seen("b" + std::to_string(i))) << i;
        still_seen += filter.Seen("a" + std::to_string(i));
    }
    // The retired generation is gone; a full Bloom generation of the same
    // bytes leaves almost no false positives.
    EXPECT_LE(still_seen, 1u);
    // A put seen again in the current generation is "put before".
    EXPECT_TRUE(filter.ObservePut("b0"));
    EXPECT_FALSE(filter.ObservePut("never-put"));
    EXPECT_TRUE(filter.ObservePut("never-put"));
}

TEST(OffloadAdmissionFilterTest, BloomRotationWaitsForAZeroedSpare) {
    constexpr size_t kWindow = 100;
    OffloadAdmissionFilter filter(kWindow,
                                  OffloadAdmissionFilter::Kind::kBloom);
    for (size_t i = 0; i < kWindow; ++i) {
        filter.ObservePut("a" + std::to_string(i));
    }
    ASSERT_EQ(filter.rotations(), 1u);
    for (size_t i = 0; i < kWindow + 50; ++i) {
        filter.ObservePut("b" + std::to_string(i));
    }
    EXPECT_EQ(filter.rotations(), 1u);
    EXPECT_EQ(filter.deferred_rotations(), 1u);
    EXPECT_TRUE(filter.Seen("a0"));
    // 800 bytes (8 per window put) in slices of 150: 5 short, the 6th
    // finishes, as for the CountMinSketch generation of the same size.
    int calls = 0;
    while (!filter.PrepareSpare(/*max_bytes=*/150)) {
        ASSERT_LT(++calls, 100);
    }
    EXPECT_EQ(calls, 5);
    filter.ObservePut("c0");
    EXPECT_EQ(filter.rotations(), 2u);
    EXPECT_TRUE(filter.Seen("b0"));
    size_t a_seen = 0;
    for (size_t i = 0; i < kWindow; ++i) {
        a_seen += filter.Seen("a" + std::to_string(i));
    }
    EXPECT_LE(a_seen, 1u);
    EXPECT_FALSE(filter.Seen("never-put"));
}

TEST(OffloadAdmissionFilterTest, BloomReusedGenerationStartsEmpty) {
    // A generation that was written, retired, zeroed in slices and reused as
    // the current one must not remember its old keys.
    constexpr size_t kWindow = 100;
    OffloadAdmissionFilter filter(kWindow,
                                  OffloadAdmissionFilter::Kind::kBloom);
    auto put_all = [&](const std::string& prefix) {
        for (size_t i = 0; i < kWindow; ++i) {
            filter.ObservePut(prefix + std::to_string(i));
        }
    };
    auto prepare = [&] {
        int calls = 0;
        while (!filter.PrepareSpare(/*max_bytes=*/150)) {
            ASSERT_LT(++calls, 100);
        }
    };
    put_all("a");  // rotation 1: "a" generation becomes previous
    prepare();
    put_all("b");  // rotation 2: "a" generation retired
    prepare();     // zeroes it into the spare
    put_all("c");  // rotation 3: the zeroed "a" generation is current
    EXPECT_EQ(filter.rotations(), 3u);
    EXPECT_EQ(filter.deferred_rotations(), 0u);
    size_t a_seen = 0;
    for (size_t i = 0; i < kWindow; ++i) {
        a_seen += filter.Seen("a" + std::to_string(i));
    }
    EXPECT_LE(a_seen, 1u);
    // Rotation 3 retired the "b" generation and nothing zeroed it yet, so
    // rotation 4 waits. One slice, then an unbounded call (as the test peer
    // makes after the eviction thread's slices) completes the spare.
    put_all("d");
    EXPECT_EQ(filter.deferred_rotations(), 1u);
    EXPECT_FALSE(filter.PrepareSpare(/*max_bytes=*/150));
    EXPECT_TRUE(filter.PrepareSpare(SIZE_MAX));
    filter.ObservePut("e0");
    EXPECT_EQ(filter.rotations(), 4u);
}

class OffloadAdmissionTest : public OffloadOnEvictCycleTest {
   protected:
    static constexpr size_t kSmallObject = 4096;

    static MasterServiceConfig EagerConfig(bool second_put,
                                           size_t window_puts = 1024,
                                           bool bloom_filter = false) {
        MasterServiceConfig config = Config();
        config.offload_on_evict = false;
        config.offload_admission_second_put = second_put;
        config.offload_admission_window_puts = window_puts;
        config.offload_admission_bloom_filter = bloom_filter;
        return config;
    }

    static MasterServiceConfig ExclusiveConfig(bool read_since_put) {
        MasterServiceConfig config = Config();
        config.offload_admission_read_since_put = read_since_put;
        return config;
    }

    // Replicas of a key, without counting as a read.
    Copies Peek(const std::string& key) {
        Copies copies;
        MasterServiceTestPeer::MetadataAccessorRW accessor(
            service_.get(),
            MasterServiceTestPeer::ObjectIdentity{TenantId::Default(), key});
        if (!accessor.Exists()) return copies;
        copies.exists = true;
        for (const auto& replica : accessor.Get().GetAllReplicas()) {
            if (replica.is_memory_replica()) ++copies.memory;
            if (replica.is_local_disk_replica()) ++copies.local_disk;
        }
        return copies;
    }

    // Whether the current write of `key` has asked the admission filter.
    bool Judged(const std::string& key) {
        MasterServiceTestPeer::MetadataAccessorRW accessor(
            service_.get(),
            MasterServiceTestPeer::ObjectIdentity{TenantId::Default(), key});
        EXPECT_TRUE(accessor.Exists()) << key;
        return accessor.Exists() && accessor.Get().eager_offload_judged;
    }

    bool Unread(const std::string& key) {
        MasterServiceTestPeer::MetadataAccessorRW accessor(
            service_.get(),
            MasterServiceTestPeer::ObjectIdentity{TenantId::Default(), key});
        EXPECT_TRUE(accessor.Exists()) << key;
        return accessor.Exists() &&
               accessor.Get().unread_since_put.load(std::memory_order_relaxed);
    }

    void RemoveAll(const std::vector<std::string>& keys) {
        for (const auto& key : keys) {
            EXPECT_TRUE(service_->Remove(key, TenantId::Default(),
                                         /*force=*/true))
                << key;
        }
    }

    static std::vector<std::string> Keys(
        const std::vector<OffloadTaskItem>& items) {
        std::vector<std::string> keys;
        for (const auto& item : items) keys.push_back(item.key);
        std::sort(keys.begin(), keys.end());
        return keys;
    }

    static std::vector<std::string> Sorted(std::vector<std::string> keys) {
        std::sort(keys.begin(), keys.end());
        return keys;
    }

    struct AdmissionCounters {
        int64_t admitted;
        int64_t skipped;
        int64_t unread_evicted;
    };

    static AdmissionCounters Counters() {
        auto& metrics = MasterMetricManager::instance();
        return {metrics.get_offload_admission_put_admitted(),
                metrics.get_offload_admission_put_skipped(),
                metrics.get_offload_admission_unread_evicted()};
    }
};

// --- Defaults
// -----------------------------------------------------------------

TEST_F(OffloadAdmissionTest, ByDefaultEagerQueuesEveryPutEnd) {
    Start(EagerConfig(/*second_put=*/false));
    const Store a = AddStore("store_a", /*with_disk=*/true);
    const auto before = Counters();

    const auto keys = Put(a, "eager_", 20, kSmallObject);
    EXPECT_EQ(Keys(Drain(a)), Sorted(keys));
    RemoveAll(keys);
    // A second put is queued the same way.
    Put(a, "eager_", 20, kSmallObject);
    EXPECT_EQ(Keys(Drain(a)), Sorted(keys));

    const auto after = Counters();
    EXPECT_EQ(after.admitted, before.admitted);
    EXPECT_EQ(after.skipped, before.skipped);
    for (const auto& key : keys) EXPECT_FALSE(Unread(key)) << key;
}

TEST_F(OffloadAdmissionTest, ByDefaultExclusiveQueuesUnreadObjects) {
    Start(ExclusiveConfig(/*read_since_put=*/false));
    const Store a = AddStore("store_a", /*with_disk=*/true);
    const auto before = Counters();
    const auto keys = Put(a, "unread_", 20, kSmallObject);
    for (const auto& key : keys) EXPECT_FALSE(Unread(key)) << key;

    Evict(/*ratio=*/1.0);

    EXPECT_EQ(Keys(Drain(a)), Sorted(keys));
    for (const auto& key : keys) EXPECT_EQ(Peek(key).memory, 1u) << key;
    EXPECT_EQ(Counters().unread_evicted, before.unread_evicted);
}

TEST_F(OffloadAdmissionTest, EachAdmissionIsIgnoredInTheOtherMode) {
    // (a) needs eager mode, (b) needs offload_on_evict.
    {
        MasterServiceConfig config = ExclusiveConfig(false);
        config.offload_admission_second_put = true;
        Start(config);
        const Store a = AddStore("store_a", /*with_disk=*/true);
        const auto keys = Put(a, "x_", 10, kSmallObject);
        EXPECT_TRUE(Drain(a).empty());  // PutEnd never queues here
        Evict(/*ratio=*/1.0);
        EXPECT_EQ(Keys(Drain(a)), Sorted(keys));
        service_->RemoveAll();
    }
    {
        MasterServiceConfig config = EagerConfig(false);
        config.offload_admission_read_since_put = true;
        Start(config);
        const Store a = AddStore("store_a", /*with_disk=*/true);
        const auto keys = Put(a, "y_", 10, kSmallObject);
        EXPECT_EQ(Keys(Drain(a)), Sorted(keys));
        for (const auto& key : keys) EXPECT_FALSE(Unread(key)) << key;
    }
}

// --- (a) second put
// -------------------------------------------------------------

TEST_F(OffloadAdmissionTest, SecondPutSkipsTheFirstPutAndQueuesTheSecond) {
    Start(EagerConfig(/*second_put=*/true));
    const Store a = AddStore("store_a", /*with_disk=*/true);
    const auto before = Counters();

    const auto keys = Put(a, "page_", 20, kSmallObject);
    EXPECT_TRUE(Drain(a).empty());
    for (const auto& key : keys) {
        EXPECT_EQ(Peek(key).memory, 1u) << key;  // the put itself is kept
    }
    auto mid = Counters();
    EXPECT_EQ(mid.skipped - before.skipped, 20);
    EXPECT_EQ(mid.admitted, before.admitted);

    // The key leaves memory and comes back: that put is written to disk.
    RemoveAll(keys);
    Put(a, "page_", 20, kSmallObject);
    const auto taken = Drain(a);
    EXPECT_EQ(Keys(taken), Sorted(keys));
    Report(a, taken, static_cast<int64_t>(kSmallObject));
    for (const auto& key : keys) {
        EXPECT_EQ(Peek(key).local_disk, 1u) << key;
        EXPECT_EQ(Peek(key).memory, 1u) << key;
    }
    const auto after = Counters();
    EXPECT_EQ(after.admitted - mid.admitted, 20);
    EXPECT_EQ(after.skipped, mid.skipped);
}

TEST_F(OffloadAdmissionTest, SecondPutFalseAdmissionsStayWithinTheSketchBound) {
    // 2 windows of distinct first puts: the first fills one generation to
    // half its width, the second also meets the full previous generation.
    constexpr size_t kWindow = 4096;
    Start(EagerConfig(/*second_put=*/true, kWindow));
    const Store a = AddStore("store_a", /*with_disk=*/true);

    Put(a, "first_", static_cast<int>(kWindow), kSmallObject);
    const size_t first_window = Drain(a).size();
    Put(a, "second_", static_cast<int>(kWindow), kSmallObject);
    const size_t second_window = Drain(a).size();

    EXPECT_LE(first_window, kWindow * 2 / 100);
    EXPECT_LE(second_window, kWindow * 5 / 100);
}

TEST_F(OffloadAdmissionTest,
       SecondPutBloomFilterSkipsFirstPutsAndQueuesTheSecond) {
    constexpr size_t kWindow = 4096;
    Start(EagerConfig(/*second_put=*/true, kWindow, /*bloom_filter=*/true));
    const Store a = AddStore("store_a", /*with_disk=*/true);
    const auto before = Counters();

    // The second put of a key is written to disk, as with the sketch.
    const auto keys = Put(a, "page_", 20, kSmallObject);
    EXPECT_TRUE(Drain(a).empty());
    RemoveAll(keys);
    Put(a, "page_", 20, kSmallObject);
    const auto taken = Drain(a);
    EXPECT_EQ(Keys(taken), Sorted(keys));
    Report(a, taken, static_cast<int64_t>(kSmallObject));
    const auto mid = Counters();
    EXPECT_EQ(mid.admitted - before.admitted, 20);
    EXPECT_EQ(mid.skipped - before.skipped, 20);

    // Two windows of distinct first puts (compare
    // SecondPutFalseAdmissionsStayWithinTheSketchBound: up to 2% and 5%).
    MasterServiceTestPeer(*service_).PrepareOffloadAdmissionSpareForTesting();
    Put(a, "first_", static_cast<int>(kWindow), kSmallObject);
    const size_t first_window = Drain(a).size();
    MasterServiceTestPeer(*service_).PrepareOffloadAdmissionSpareForTesting();
    Put(a, "second_", static_cast<int>(kWindow), kSmallObject);
    const size_t second_window = Drain(a).size();
    EXPECT_LE(first_window, 1u);
    EXPECT_LE(second_window, 1u);
}

TEST_F(OffloadAdmissionTest, BloomFilterAloneChangesNothing) {
    // Without offload_admission_second_put the filter kind is ignored:
    // every eager PutEnd is queued and no admission counter moves.
    Start(EagerConfig(/*second_put=*/false, 1024, /*bloom_filter=*/true));
    const Store a = AddStore("store_a", /*with_disk=*/true);
    const auto before = Counters();
    const auto keys = Put(a, "page_", 20, kSmallObject);
    EXPECT_EQ(Keys(Drain(a)), Sorted(keys));
    const auto after = Counters();
    EXPECT_EQ(after.admitted, before.admitted);
    EXPECT_EQ(after.skipped, before.skipped);
}

TEST_F(OffloadAdmissionTest, SecondPutRemembersTheWindowAndForgetsAfterTwo) {
    constexpr int kWindow = 256;
    constexpr int kTargets = 50;
    Start(EagerConfig(/*second_put=*/true, kWindow));
    const Store a = AddStore("store_a", /*with_disk=*/true);

    // 50 targets and 206 fillers: the 256th put retires the generation.
    const auto targets = Put(a, "target_", kTargets, kSmallObject);
    const auto fillers = Put(a, "fill_a_", kWindow - kTargets, kSmallObject);
    Drain(a);
    RemoveAll(targets);
    RemoveAll(fillers);

    // One window later the targets are remembered: all are admitted.
    Put(a, "target_", kTargets, kSmallObject);
    EXPECT_EQ(Keys(Drain(a)), Sorted(targets));
    RemoveAll(targets);

    // Two more retirements (206 + 256 puts) drop the generation holding the
    // targets' second puts; only false positives can admit them now. Each
    // rotation needs a zeroed spare (the eviction thread prepares one too).
    MasterServiceTestPeer(*service_).PrepareOffloadAdmissionSpareForTesting();
    Put(a, "fill_b_", kWindow - kTargets, kSmallObject);
    MasterServiceTestPeer(*service_).PrepareOffloadAdmissionSpareForTesting();
    Put(a, "fill_c_", kWindow, kSmallObject);
    Drain(a);
    Put(a, "target_", kTargets, kSmallObject);
    EXPECT_LE(Drain(a).size(), static_cast<size_t>(kTargets * 10 / 100));
}

TEST_F(OffloadAdmissionTest, SecondPutKeepsPutOrderForQueueOrderBuckets) {
    // MOONCAKE_OFFLOAD_BUCKET_KEEP_QUEUE_ORDER (store side) fills buckets in
    // task id order. Admission only removes pushes, so the admitted items
    // still carry ids in put order, and skipped puts take no id.
    Start(EagerConfig(/*second_put=*/true));
    const Store a = AddStore("store_a", /*with_disk=*/true);
    std::vector<std::string> order;
    for (int i = 0; i < 10; ++i) {
        for (const char* half : {"_k", "_v"}) {
            order.push_back("p" + std::to_string(i) + half);
        }
    }
    auto put_in_order = [&] {
        for (const auto& key : order) {
            ReplicateConfig config;
            config.replica_num = 1;
            config.preferred_segment = a.segment_name;
            ASSERT_TRUE(service_->PutStart(
                a.client_id, key, TenantId::Default(), kSmallObject, config));
            ASSERT_TRUE(service_->PutEnd(a.client_id, key, TenantId::Default(),
                                         ReplicaType::MEMORY));
        }
    };
    put_in_order();
    EXPECT_TRUE(Drain(a).empty());
    RemoveAll(order);
    put_in_order();

    auto taken = Drain(a);
    ASSERT_EQ(taken.size(), order.size());
    std::sort(taken.begin(), taken.end(), [](const auto& x, const auto& y) {
        return x.task_id->seq < y.task_id->seq;
    });
    for (size_t i = 0; i < order.size(); ++i) {
        EXPECT_EQ(taken[i].key, order[i]) << i;
    }
    EXPECT_EQ(taken.back().task_id->seq - taken.front().task_id->seq,
              order.size() - 1);
}

TEST_F(OffloadAdmissionTest, SecondPutLaterEndOfTheSameWriteFollowsTheFirst) {
    // A write can end with more than one PutEnd (one per replica type). The
    // first End decides; a later End must not count the first one as an
    // earlier put of the key.
    Start(EagerConfig(/*second_put=*/true));
    const Store a = AddStore("store_a", /*with_disk=*/true);
    const auto keys = Put(a, "two_ends_", 4, kSmallObject);  // first puts
    MasterServiceTestPeer peer(*service_);
    for (const auto& key : keys) {
        EXPECT_FALSE(peer.AdmitEagerOffloadForTesting(key)) << key;
    }
    // The next write of each key is admitted, and so is its later End.
    RemoveAll(keys);
    Put(a, "two_ends_", 4, kSmallObject);
    EXPECT_EQ(Drain(a).size(), keys.size());
    for (const auto& key : keys) {
        EXPECT_TRUE(peer.AdmitEagerOffloadForTesting(key)) << key;
    }
}

// Sets an environment variable for its lifetime.
class ScopedEnv {
   public:
    ScopedEnv(const char* name, const std::string& value) : name_(name) {
        if (const char* previous = std::getenv(name)) previous_ = previous;
        ::setenv(name, value.c_str(), 1);
    }
    ~ScopedEnv() {
        if (previous_) {
            ::setenv(name_.c_str(), previous_->c_str(), 1);
        } else {
            ::unsetenv(name_.c_str());
        }
    }

   private:
    std::string name_;
    std::optional<std::string> previous_;
};

// A directory removed with its contents when the test leaves, even on a
// failed ASSERT.
class ScopedTempDir {
   public:
    explicit ScopedTempDir(const std::string& name)
        : path_((std::filesystem::temp_directory_path() /
                 (name + "_" + std::to_string(::getpid())))
                    .string()) {
        std::filesystem::create_directories(path_);
    }
    ~ScopedTempDir() {
        std::error_code ec;
        std::filesystem::remove_all(path_, ec);
    }
    const std::string& path() const { return path_; }

   private:
    std::string path_;
};

// The regression test of the admission fix: on the code before it, the DFS
// End skipped the filter and the MEMORY End took "not admitted", so neither
// write below was counted and the second was not queued.
TEST_F(OffloadAdmissionTest, SecondPutJudgesAWriteWhoseDfsEndComesFirst) {
    // A write with a DFS replica can end with its DFS End before its MEMORY
    // End. The DFS End cannot queue an offload but still records the put,
    // so the MEMORY End follows this write's answer. UpsertStart case B
    // (same size, the metadata is reused) is a new write and asks again.
    const ScopedTempDir dfs_dir("offload_admission_dfs");
    const std::string& dfs_root = dfs_dir.path();
    {
        ScopedEnv enable_dfs("MOONCAKE_ENABLE_DFS", "1");
        ScopedEnv fs_adapter("MOONCAKE_DFS_FS_ADAPTER", "posix");
        ScopedEnv root_dir("MOONCAKE_DFS_ROOT_DIR", dfs_root);
        ScopedEnv shard_count("MOONCAKE_DFS_SHARD_COUNT", "1");
        ScopedEnv shard_capacity("MOONCAKE_DFS_SHARD_CAPACITY", "1048576");
        ScopedEnv alignment("MOONCAKE_DFS_ALIGNMENT", "4096");
        ScopedEnv eviction("MOONCAKE_DFS_EVICTION_ENABLED", "0");
        ScopedEnv deferred_free("MOONCAKE_DFS_DEFERRED_FREE_SECONDS", "0");
        ScopedEnv single_tenant("MOONCAKE_DFS_SINGLE_TENANT", "true");

        for (const bool bloom : {false, true}) {
            SCOPED_TRACE(bloom ? "Bloom" : "CountMinSketch");
            Start(EagerConfig(/*second_put=*/true, 1024, bloom));
            const Store a = AddStore("store_a", /*with_disk=*/true);
            const auto before = Counters();
            ReplicateConfig config;
            config.replica_num = 1;
            config.dfs_replica_num = 1;
            config.preferred_segment = a.segment_name;
            const std::string key = "dfs_end_first";
            const auto tenant = TenantId::Default();

            // First write: a first put, judged by the DFS End, not queued.
            auto started = service_->PutStart(a.client_id, key, tenant,
                                              kSmallObject, config);
            ASSERT_TRUE(started.has_value());
            ASSERT_EQ(started->size(), 2u);
            ASSERT_TRUE(
                service_->PutEnd(a.client_id, key, tenant, ReplicaType::DFS));
            EXPECT_TRUE(Judged(key));
            ASSERT_TRUE(service_->PutEnd(a.client_id, key, tenant,
                                         ReplicaType::MEMORY));
            EXPECT_TRUE(Drain(a).empty());
            const auto mid = Counters();
            EXPECT_EQ(mid.skipped - before.skipped, 1);
            EXPECT_EQ(mid.admitted, before.admitted);

            // Second write in place, DFS End first again: the second put of
            // the key is queued by its MEMORY End.
            ASSERT_TRUE(service_->UpsertStart(a.client_id, key, tenant,
                                              kSmallObject, config));
            EXPECT_FALSE(Judged(key));
            ASSERT_TRUE(service_->UpsertEnd(a.client_id, key, tenant,
                                            ReplicaType::DFS));
            ASSERT_TRUE(service_->UpsertEnd(a.client_id, key, tenant,
                                            ReplicaType::MEMORY));
            EXPECT_EQ(Keys(Drain(a)), std::vector<std::string>{key});
            const auto after = Counters();
            EXPECT_EQ(after.admitted - mid.admitted, 1);
            EXPECT_EQ(after.skipped, mid.skipped);

            service_->RemoveAll();
            service_.reset();
        }
    }
}

TEST_F(OffloadAdmissionTest, SecondPutUpsertInPlaceIsANewWrite) {
    // UpsertStart case B reuses the metadata of the previous write. Its End
    // must ask the filter again rather than follow the previous answer. With
    // MEMORY replicas only this also held before the admission fix (the End
    // judged because no replica was complete); it pins that case B clears
    // the per-write mark.
    for (const bool bloom : {false, true}) {
        SCOPED_TRACE(bloom ? "Bloom" : "CountMinSketch");
        Start(EagerConfig(/*second_put=*/true, 1024, bloom));
        const Store a = AddStore("store_a", /*with_disk=*/true);
        const auto keys = Put(a, "upsert_", 8, kSmallObject);  // first puts
        EXPECT_TRUE(Drain(a).empty());
        ReplicateConfig config;
        config.replica_num = 1;
        config.preferred_segment = a.segment_name;
        for (const auto& key : keys) {
            EXPECT_TRUE(Judged(key)) << key;
            ASSERT_TRUE(service_->UpsertStart(
                a.client_id, key, TenantId::Default(), kSmallObject, config))
                << key;
            EXPECT_FALSE(Judged(key)) << key;
            ASSERT_TRUE(service_->UpsertEnd(
                a.client_id, key, TenantId::Default(), ReplicaType::MEMORY))
                << key;
        }
        EXPECT_EQ(Keys(Drain(a)), Sorted(keys));
        service_->RemoveAll();
        service_.reset();
    }
}

// --- (b) read since put
// -------------------------------------------------------

TEST_F(OffloadAdmissionTest, ReadSincePutEvictsUnreadAndQueuesRead) {
    Start(ExclusiveConfig(/*read_since_put=*/true));
    const Store a = AddStore("store_a", /*with_disk=*/true);
    const auto before = Counters();
    const auto keys = Put(a, "obj_", 20, kSmallObject);
    for (const auto& key : keys) EXPECT_TRUE(Unread(key)) << key;

    // Keys 0..4 are read by GetReplicaList, 5..9 by BatchGetReplicaList;
    // 10..19 only answer exists queries, which are not reads.
    std::vector<std::string> read(keys.begin(), keys.begin() + 10);
    std::vector<std::string> unread(keys.begin() + 10, keys.end());
    for (int i = 0; i < 5; ++i) {
        ASSERT_TRUE(service_->GetReplicaList(keys[i], TenantId::Default()));
    }
    for (const auto& r : service_->BatchGetReplicaList(
             std::vector<std::string>(keys.begin() + 5, keys.begin() + 10),
             TenantId::Default())) {
        ASSERT_TRUE(r.has_value());
    }
    for (const auto& r : service_->BatchExistKey(unread, TenantId::Default())) {
        ASSERT_TRUE(r.has_value() && *r);
    }
    for (const auto& key : read) EXPECT_FALSE(Unread(key)) << key;
    for (const auto& key : unread) EXPECT_TRUE(Unread(key)) << key;

    Evict(/*ratio=*/1.0);

    const auto taken = Drain(a);
    EXPECT_EQ(Keys(taken), Sorted(read));
    for (const auto& key : unread) EXPECT_FALSE(Peek(key).exists) << key;
    EXPECT_EQ(Counters().unread_evicted - before.unread_evicted, 10);

    // The read keys follow the usual exclusive path.
    Report(a, taken, static_cast<int64_t>(kSmallObject));
    Evict(/*ratio=*/1.0);
    for (const auto& key : read) {
        EXPECT_EQ(Peek(key).memory, 0u) << key;
        EXPECT_EQ(Peek(key).local_disk, 1u) << key;
    }
}

TEST_F(OffloadAdmissionTest, ReadSincePutMarkIsResetByTheNextPut) {
    Start(ExclusiveConfig(/*read_since_put=*/true));
    const Store a = AddStore("store_a", /*with_disk=*/true);
    const auto keys = Put(a, "again_", 4, kSmallObject);
    for (const auto& key : keys) {
        ASSERT_TRUE(service_->GetReplicaList(key, TenantId::Default()));
        EXPECT_FALSE(Unread(key));
    }
    RemoveAll(keys);
    Put(a, "again_", 4, kSmallObject);
    for (const auto& key : keys) EXPECT_TRUE(Unread(key)) << key;
}

TEST_F(OffloadAdmissionTest, ReadSincePutWithFrontierOnlyKeepsLeaseOrder) {
    // 2000 keys, key i holds the i-th oldest lease; the odd ones are read.
    // With the 5% target (evict_num 100) the cycle walks the frontier from
    // the oldest: even keys are evicted, odd keys are queued, and it stops
    // once 100 objects are evicted. Nothing past that is touched.
    constexpr int kKeys = 2000;
    MasterServiceConfig config = ExclusiveConfig(/*read_since_put=*/true);
    config.offload_evict_frontier_only = true;
    Start(config);
    const Store a = AddStore("store_a", /*with_disk=*/true);
    const auto keys = Put(a, "lease_", kKeys, kSmallObject);
    for (int i = 1; i < kKeys; i += 2) {
        ASSERT_TRUE(service_->GetReplicaList(keys[i], TenantId::Default()));
    }
    const auto base = std::chrono::system_clock::now() - std::chrono::hours(1);
    for (int i = 0; i < kKeys; ++i) {
        SetLease(keys[i], base + std::chrono::milliseconds(i));
    }

    Evict(/*ratio=*/0.05);

    auto taken = Drain(a);
    std::sort(taken.begin(), taken.end(), [](const auto& x, const auto& y) {
        return x.task_id->seq < y.task_id->seq;
    });
    std::vector<int> queued;
    for (const auto& item : taken)
        queued.push_back(std::stoi(item.key.substr(6)));
    std::vector<int> expected;
    for (int i = 1; i < 199; i += 2) expected.push_back(i);
    EXPECT_EQ(queued, expected);
    for (int i = 0; i < 200; i += 2) {
        EXPECT_FALSE(Peek(keys[i]).exists) << keys[i];
    }
    for (int i = 199; i < kKeys; ++i) {
        ASSERT_EQ(Peek(keys[i]).memory, 1u) << keys[i];
    }
}

TEST_F(OffloadAdmissionTest, ReadSincePutWithDefaultOrderDropsOnlyUnread) {
    // Same keys without frontier_only: the frontier is walked in nth_element
    // order, so which keys are visited is not fixed, but every evicted key is
    // unread and every queued key was read.
    constexpr int kKeys = 2000;
    Start(ExclusiveConfig(/*read_since_put=*/true));
    const Store a = AddStore("store_a", /*with_disk=*/true);
    const auto keys = Put(a, "lease_", kKeys, kSmallObject);
    for (int i = 1; i < kKeys; i += 2) {
        ASSERT_TRUE(service_->GetReplicaList(keys[i], TenantId::Default()));
    }

    Evict(/*ratio=*/0.05);

    for (const auto& item : Drain(a)) {
        EXPECT_EQ(std::stoi(item.key.substr(6)) % 2, 1) << item.key;
    }
    int evicted = 0;
    for (int i = 0; i < kKeys; ++i) {
        if (!Peek(keys[i]).exists) {
            ++evicted;
            EXPECT_EQ(i % 2, 0) << keys[i];
        }
    }
    EXPECT_EQ(evicted, 100);
}

TEST_F(OffloadAdmissionTest, ReadSincePutDropsNothingPastTheFrontier) {
    // The 1124 oldest keys (the compact frontier) were read, the 876 newer
    // ones were not. The frontier only queues, so the cycle goes on past it.
    // There an unread key is never dropped: without frontier_only it is
    // queued as before, with it it stays in memory.
    constexpr int kKeys = 2000;
    constexpr int kFrontier = 1124;
    for (const bool frontier_only : {false, true}) {
        SCOPED_TRACE(frontier_only);
        MasterServiceConfig config = ExclusiveConfig(/*read_since_put=*/true);
        config.offload_evict_frontier_only = frontier_only;
        Start(config);
        const Store a = AddStore("store_a", /*with_disk=*/true);
        const auto keys = Put(a, "lease_", kKeys, kSmallObject);
        for (int i = 0; i < kFrontier; ++i) {
            ASSERT_TRUE(service_->GetReplicaList(keys[i], TenantId::Default()));
        }
        const auto base =
            std::chrono::system_clock::now() - std::chrono::hours(1);
        for (int i = 0; i < kKeys; ++i) {
            SetLease(keys[i], base + std::chrono::milliseconds(i));
        }
        const auto before = Counters();

        Evict(/*ratio=*/0.05);

        EXPECT_EQ(Drain(a).size(),
                  static_cast<size_t>(frontier_only ? kFrontier : kKeys));
        for (int i = kFrontier; i < kKeys; ++i) {
            ASSERT_EQ(Peek(keys[i]).memory, 1u) << keys[i];
        }
        EXPECT_EQ(Counters().unread_evicted, before.unread_evicted);
        service_->RemoveAll();
    }
}

TEST_F(OffloadAdmissionTest, ReadSincePutOnStoresWithoutLocalDisk) {
    // Store b has no LOCAL_DISK tier. Unread objects are evicted on both
    // stores. A read object of b stays in memory by default and is evicted
    // when offload_evict_without_local_disk is set.
    for (const bool without_local_disk : {false, true}) {
        SCOPED_TRACE(without_local_disk);
        MasterServiceConfig config = ExclusiveConfig(/*read_since_put=*/true);
        config.offload_evict_without_local_disk = without_local_disk;
        Start(config);
        const Store a = AddStore("store_a", /*with_disk=*/true);
        const Store b = AddStore("store_b", /*with_disk=*/false);
        const auto a_keys = Put(a, "a_", 10, kSmallObject);
        const auto b_keys = Put(b, "b_", 10, kSmallObject);
        for (int i = 0; i < 5; ++i) {
            ASSERT_TRUE(
                service_->GetReplicaList(a_keys[i], TenantId::Default()));
            ASSERT_TRUE(
                service_->GetReplicaList(b_keys[i], TenantId::Default()));
        }
        const auto before = Counters();

        Evict(/*ratio=*/1.0);

        EXPECT_EQ(Keys(Drain(a)), Sorted(std::vector<std::string>(
                                      a_keys.begin(), a_keys.begin() + 5)));
        for (int i = 5; i < 10; ++i) {
            EXPECT_FALSE(Peek(a_keys[i]).exists) << a_keys[i];
            EXPECT_FALSE(Peek(b_keys[i]).exists) << b_keys[i];
        }
        for (int i = 0; i < 5; ++i) {
            EXPECT_EQ(Peek(b_keys[i]).exists, !without_local_disk) << b_keys[i];
        }
        EXPECT_EQ(Counters().unread_evicted - before.unread_evicted, 10);
        service_->RemoveAll();
    }
}

TEST_F(OffloadAdmissionTest, ReadSincePutUnreadDoesNotUseTheForceCap) {
    // With force_evict and a cap of 2, unread objects are evicted by
    // admission before the cap is counted, so the read ones still get
    // the two queue slots.
    MasterServiceConfig config = ExclusiveConfig(/*read_since_put=*/true);
    config.offload_force_evict = true;
    config.offloading_queue_limit = 4;
    config.offload_cap_ratio = 0.5;
    Start(config);
    const Store a = AddStore("store_a", /*with_disk=*/true);
    const auto keys = Put(a, "cap_", 10, kSmallObject);
    for (int i = 0; i < 2; ++i) {
        ASSERT_TRUE(service_->GetReplicaList(keys[i], TenantId::Default()));
    }

    Evict(/*ratio=*/1.0);

    EXPECT_EQ(Keys(Drain(a)), Sorted({keys[0], keys[1]}));
}

// unread_since_put, eager_offload_judged and eager_offload_admitted sit in
// padding. On the production toolchain (x86-64 libstdc++, no debug
// containers) the struct stays at its size before the admission fields.
#if defined(__x86_64__) && defined(__GLIBCXX__) && !defined(_GLIBCXX_DEBUG)
static_assert(sizeof(ObjectMetadata) == 392,
              "ObjectMetadata grew; the master metadata cost model "
              "(handoff_admission.h) assumes its size");
#endif

// A store whose keys cannot be queued must not hold the frontier. Store a
// holds the 1500 oldest leases, store b the 500 newest, so the compact
// frontier (1124) is all a's.
class OffloadFrontierSlotTest : public OffloadFrontierTest {
   protected:
    void PutRanks(const Store& store, int begin, int end) {
        for (int i = begin; i < end; ++i) {
            const std::string key = "lease_" + std::to_string(i);
            ReplicateConfig config;
            config.replica_num = 1;
            config.preferred_segment = store.segment_name;
            ASSERT_TRUE(service_->PutStart(store.client_id, key,
                                           TenantId::Default(), kSmallObject,
                                           config));
            ASSERT_TRUE(service_->PutEnd(store.client_id, key,
                                         TenantId::Default(),
                                         ReplicaType::MEMORY));
            SetLease(key, base_ + std::chrono::milliseconds(i));
        }
    }
};

TEST_F(OffloadFrontierSlotTest, FrontierOnlyPassesSlotsOfAFullMailbox) {
    // Mailboxes hold 10. a's 10 oldest are queued and its other 1114
    // frontier keys fail; their slots go to the oldest keys past the
    // frontier: a's (fail again) and then b's, until b's mailbox is full.
    MasterServiceConfig config = FrontierConfig(true);
    config.offloading_queue_limit = 10;
    Start(config);
    const Store a = AddStore("store_a", /*with_disk=*/true);
    const Store b = AddStore("store_b", /*with_disk=*/true);
    PutRanks(a, 0, 1500);
    PutRanks(b, 1500, 2000);

    Evict(/*ratio=*/0.05);

    EXPECT_EQ(RanksInQueueOrder(Drain(a)), Range(0, 10));
    EXPECT_EQ(RanksInQueueOrder(Drain(b)), Range(1500, 1510));
}

TEST_F(OffloadFrontierSlotTest, FrontierOnlyPassesSlotsOfAStoreWithoutDisk) {
    // a has no local disk (offload_evict_without_local_disk off): every
    // frontier push fails and the keys stay in memory, unpinned, so they
    // are the frontier again next cycle. b's keys take their slots, oldest
    // first. Without that, b would never be queued and nothing evicted.
    Start(FrontierConfig(true));
    const Store a = AddStore("store_a", /*with_disk=*/false);
    const Store b = AddStore("store_b", /*with_disk=*/true);
    PutRanks(a, 0, 1500);
    PutRanks(b, 1500, 2000);

    Evict(/*ratio=*/0.05);

    EXPECT_EQ(RanksInQueueOrder(Drain(b)), Range(1500, 2000));
    EXPECT_EQ(CopiesOf("lease_0").memory, 1u);
}

TEST_F(OffloadFrontierSlotTest, FrontierOnlyQueuesSoftPinnedKeysInPassB) {
    // Soft-pinned keys never enter the first pass; with
    // allow_evict_soft_pinned_objects the second pass takes the oldest. It
    // must queue them with frontier_only too, or they never leave memory.
    for (const bool frontier_only : {false, true}) {
        SCOPED_TRACE(frontier_only);
        MasterServiceConfig config = FrontierConfig(frontier_only);
        config.allow_evict_soft_pinned_objects = true;
        Start(config);
        const Store a = AddStore("store_a", /*with_disk=*/true);
        for (int i = 0; i < 20; ++i) {
            const std::string key = "lease_" + std::to_string(i);
            ReplicateConfig replicate;
            replicate.replica_num = 1;
            replicate.preferred_segment = a.segment_name;
            replicate.soft_pin_action = SoftPinAction::ENABLE;
            ASSERT_TRUE(service_->PutStart(a.client_id, key,
                                           TenantId::Default(), kSmallObject,
                                           replicate));
            ASSERT_TRUE(service_->PutEnd(a.client_id, key, TenantId::Default(),
                                         ReplicaType::MEMORY));
            SetLease(key, base_ + std::chrono::milliseconds(i));
        }

        Evict(/*ratio=*/0.05);

        EXPECT_EQ(RanksInQueueOrder(Drain(a)), Range(0, 1));
        service_->RemoveAll();
    }
}

TEST_F(OffloadOnEvictCycleTest, DefaultsKeepTheOptInOff) {
    MasterServiceConfig config;
    EXPECT_FALSE(config.offload_admission_second_put);
    EXPECT_FALSE(config.offload_admission_read_since_put);
    EXPECT_EQ(config.offload_admission_window_puts, 33554432u);
    EXPECT_FALSE(config.offload_admission_bloom_filter);
    EXPECT_FALSE(config.offload_evict_without_local_disk);
    EXPECT_FALSE(config.offload_evict_frontier_only);
    EXPECT_FALSE(config.offload_force_evict);
    EXPECT_FALSE(config.offload_on_evict);
    EXPECT_FALSE(config.enable_offload_stall_metrics);
}

}  // namespace mooncake::test

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
