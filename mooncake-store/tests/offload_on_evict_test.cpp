#include "master_service.h"
#include "master_service/master_service_test_peer.h"

#include <glog/logging.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <map>
#include <memory>
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
                                 int count) {
        std::vector<std::string> keys;
        for (int i = 0; i < count; ++i) {
            const std::string key = prefix + std::to_string(i);
            ReplicateConfig config;
            config.replica_num = 1;
            config.preferred_segment = store.segment_name;
            auto started = service_->PutStart(
                store.client_id, key, TenantId::Default(), kObjectSize, config);
            EXPECT_TRUE(started.has_value()) << key;
            EXPECT_TRUE(service_->PutEnd(store.client_id, key,
                                         TenantId::Default(),
                                         ReplicaType::MEMORY));
            keys.push_back(key);
        }
        return keys;
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

TEST_F(OffloadOnEvictCycleTest, DefaultsKeepTheOptInOff) {
    MasterServiceConfig config;
    EXPECT_FALSE(config.offload_evict_without_local_disk);
    EXPECT_FALSE(config.offload_force_evict);
    EXPECT_FALSE(config.offload_on_evict);
}

}  // namespace mooncake::test

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
