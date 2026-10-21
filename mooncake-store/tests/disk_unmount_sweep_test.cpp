// Tests for the background LOCAL_DISK unmount sweep
// (--async_disk_unmount_sweep, fork-only, default off).
//
// The sweep thread calls a test hook before each shard step, so a test can
// hold it at a chosen point and run puts, reads, mounts or a handoff export
// against a sweep that is known to be in progress.

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include "master_metric_manager.h"
#include "master_service.h"
#include "master_service/master_service_test_peer.h"
#include "types.h"

namespace mooncake::test {
namespace {

constexpr int64_t kObjectSize = 1024;
constexpr auto kSweepTimeout = std::chrono::seconds(120);

MasterServiceConfig SweepConfig(bool async) {
    MasterServiceConfig config;
    config.enable_offload = true;
    config.default_kv_lease_ttl = 0;
    config.eviction_high_watermark_ratio = 1.0;
    config.async_disk_unmount_sweep = async;
    // No store heartbeats here: keep the clients alive through long setups.
    config.client_active_ttl_sec = 3600;
    config.client_suspicion_ttl_sec = 3600;
    return config;
}

// A store as the master sees it: one memory segment and a LOCAL_DISK tier.
struct Store {
    UUID client_id;
    std::string segment;
};

Store MountStore(MasterService& service, int index) {
    Store store{generate_uuid(), "sweep-store-" + std::to_string(index)};
    Segment segment;
    segment.id = generate_uuid();
    segment.name = store.segment;
    segment.base = 0x100000000000ULL + index * 0x10000000000ULL;
    segment.size = 1ULL << 30;
    segment.te_endpoint = segment.name;
    EXPECT_TRUE(service.MountSegment(segment, store.client_id).has_value());
    EXPECT_TRUE(
        service.MountLocalDiskSegment(store.client_id, true).has_value());
    EXPECT_TRUE(
        service.ReportSsdCapacity(store.client_id, 1LL << 40).has_value());
    return store;
}

void Put(MasterService& service, const Store& store, const std::string& key) {
    ReplicateConfig config;
    config.replica_num = 1;
    config.preferred_segment = store.segment;
    auto started = service.PutStart(store.client_id, key, TenantId::Default(),
                                    kObjectSize, config);
    ASSERT_TRUE(started.has_value()) << key << " " << toString(started.error());
    ASSERT_TRUE(service
                    .PutEnd(store.client_id, key, TenantId::Default(),
                            ReplicaType::MEMORY)
                    .has_value());
}

// Puts the keys on `store` and completes their offload: each ends with a
// memory and a LOCAL_DISK replica of `store`.
void PutWithDiskCopy(MasterService& service, const Store& store,
                     const std::vector<std::string>& keys) {
    for (const auto& key : keys) Put(service, store, key);
    auto tasks = service.OffloadObjectHeartbeat(store.client_id, true);
    ASSERT_TRUE(tasks.has_value());
    ASSERT_EQ(tasks->size(), keys.size());
    StorageObjectMetadata meta;
    meta.data_size = kObjectSize;
    meta.transport_endpoint = store.segment + "-disk";
    std::vector<StorageObjectMetadata> metas(tasks->size(), meta);
    ASSERT_TRUE(service.NotifyOffloadSuccess(store.client_id, *tasks, metas)
                    .has_value());
}

// As PutWithDiskCopy, then the memory copy is dropped: disk-only objects.
void PutDiskOnly(MasterService& service, const Store& store,
                 const std::vector<std::string>& keys) {
    PutWithDiskCopy(service, store, keys);
    auto cleared =
        service.BatchReplicaClear(keys, store.client_id, store.segment);
    ASSERT_TRUE(cleared.has_value());
    ASSERT_EQ(cleared->size(), keys.size());
}

std::vector<std::string> Keys(const std::string& prefix, int count) {
    std::vector<std::string> keys;
    for (int i = 0; i < count; ++i) keys.push_back(prefix + std::to_string(i));
    return keys;
}

std::vector<Replica::Descriptor> Replicas(MasterService& service,
                                          const std::string& key) {
    auto r = service.GetReplicaList(key, TenantId::Default());
    return r ? r->replicas : std::vector<Replica::Descriptor>{};
}

bool HasDiskOf(const std::vector<Replica::Descriptor>& replicas,
               const UUID& owner) {
    for (const auto& d : replicas) {
        if (d.is_local_disk_replica() &&
            d.get_local_disk_descriptor().client_id == owner) {
            return true;
        }
    }
    return false;
}

bool Exists(MasterService& service, const std::string& key) {
    auto r = service.ExistKey(key, TenantId::Default());
    return r.has_value() && *r;
}

// Holds the sweep thread before its (block_after + 1)-th shard step until
// Release(). Steps before that run freely, as do all steps after Release().
class SweepGate {
   public:
    explicit SweepGate(size_t block_after) : block_after_(block_after) {}

    void Hook(size_t /*shard*/) {
        std::unique_lock<std::mutex> lock(mutex_);
        ++steps_;
        cv_.notify_all();
        if (steps_ > block_after_) {
            cv_.wait(lock, [this] { return released_; });
        }
    }

    // Waits until the sweep thread is parked in the hook.
    void WaitBlocked() {
        std::unique_lock<std::mutex> lock(mutex_);
        ASSERT_TRUE(cv_.wait_for(lock, kSweepTimeout,
                                 [this] { return steps_ > block_after_; }));
    }

    void Release() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            released_ = true;
        }
        cv_.notify_all();
    }

    size_t steps() {
        std::lock_guard<std::mutex> lock(mutex_);
        return steps_;
    }

   private:
    std::mutex mutex_;
    std::condition_variable cv_;
    const size_t block_after_;
    size_t steps_ = 0;
    bool released_ = false;
};

// Declared after the service (and the gate before it), so a failing test
// frees a parked sweep step before the service's destructor joins the
// sweep thread.
struct ReleaseOnExit {
    SweepGate& gate;
    ~ReleaseOnExit() { gate.Release(); }
};

void InstallGate(MasterService& service, SweepGate& gate) {
    MasterServiceTestPeer(service).SetDiskUnmountSweepShardHookForTesting(
        [&gate](size_t shard) { gate.Hook(shard); });
}

bool WaitForSweeps(MasterService& service) {
    return MasterServiceTestPeer(service).WaitForDiskUnmountSweepsForTesting(
        std::chrono::duration_cast<std::chrono::milliseconds>(kSweepTimeout));
}

auto& Metrics() { return MasterMetricManager::instance(); }

bool SweepSeriesExported() {
    return Metrics().serialize_metrics().find("master_disk_unmount_sweep") !=
           std::string::npos;
}

// ---- Flag off: the synchronous sweep, unchanged.

// Declared first so that, in a normal run, no earlier test of this binary
// has exported the sweep series yet.
TEST(DiskUnmountSweepTest, FlagOffSweepsBeforeTheUnmountReturns) {
    const bool series_before = SweepSeriesExported();
    MasterService service(SweepConfig(false));
    MasterServiceTestPeer peer(service);
    EXPECT_FALSE(peer.AsyncDiskUnmountSweepEnabledForTesting());
    Store x = MountStore(service, 0);
    const auto disk_only = Keys("off-disk-only-", 5);
    const auto both = Keys("off-both-", 5);
    PutDiskOnly(service, x, disk_only);
    PutWithDiskCopy(service, x, both);
    const size_t keys_before = service.GetKeyCount();
    const int64_t removed_before =
        Metrics().get_disk_unmount_sweep_removed_replicas();

    ASSERT_TRUE(service.UnmountLocalDiskSegment(x.client_id).has_value());

    // Already swept when the call returns.
    EXPECT_EQ(service.GetKeyCount(), keys_before - disk_only.size());
    for (const auto& key : disk_only) {
        auto r = service.GetReplicaList(key, TenantId::Default());
        ASSERT_FALSE(r.has_value());
        EXPECT_EQ(r.error(), ErrorCode::OBJECT_NOT_FOUND);
    }
    for (const auto& key : both) {
        auto replicas = Replicas(service, key);
        ASSERT_EQ(replicas.size(), 1u);
        EXPECT_TRUE(replicas[0].is_memory_replica());
    }
    // The same store mounts again at once.
    EXPECT_TRUE(service.MountLocalDiskSegment(x.client_id, true).has_value());
    EXPECT_FALSE(peer.DiskUnmountSweepPendingForTesting(x.client_id));
    EXPECT_EQ(Metrics().get_disk_unmount_sweep_removed_replicas(),
              removed_before);
    if (!series_before) {
        EXPECT_FALSE(SweepSeriesExported());
    }
}

TEST(DiskUnmountSweepTest, FlagNeedsOffloadAndNoHaSnapshotOrCxl) {
    MasterServiceConfig no_offload = SweepConfig(true);
    no_offload.enable_offload = false;
    MasterService without_offload(no_offload);
    EXPECT_FALSE(MasterServiceTestPeer(without_offload)
                     .AsyncDiskUnmountSweepEnabledForTesting());

    MasterService enabled(SweepConfig(true));
    EXPECT_TRUE(MasterServiceTestPeer(enabled)
                    .AsyncDiskUnmountSweepEnabledForTesting());
}

// ---- Flag on.

TEST(DiskUnmountSweepTest, UnmountReturnsBeforeTheSweepAndHidesTheReplicas) {
    SweepGate gate(0);
    MasterService service(SweepConfig(true));
    ReleaseOnExit release_on_exit{gate};
    MasterServiceTestPeer peer(service);
    InstallGate(service, gate);
    Store x = MountStore(service, 0);
    Store y = MountStore(service, 1);
    const auto disk_only = Keys("x-disk-only-", 5);
    const auto both = Keys("x-both-", 5);
    const auto other = Keys("y-disk-only-", 5);
    PutDiskOnly(service, x, disk_only);
    PutWithDiskCopy(service, x, both);
    PutDiskOnly(service, y, other);
    const size_t keys_before = service.GetKeyCount();
    const int64_t removed_before =
        Metrics().get_disk_unmount_sweep_removed_replicas();

    ASSERT_TRUE(service.UnmountLocalDiskSegment(x.client_id).has_value());
    gate.WaitBlocked();  // nothing swept yet

    EXPECT_TRUE(peer.DiskUnmountSweepPendingForTesting(x.client_id));
    EXPECT_FALSE(peer.DiskUnmountSweepPendingForTesting(y.client_id));
    EXPECT_EQ(service.GetKeyCount(), keys_before);
    EXPECT_EQ(Metrics().get_disk_unmount_sweep_pending_owners(), 1);
    EXPECT_EQ(Metrics().get_disk_unmount_sweep_remaining_shards(),
              static_cast<int64_t>(MasterServiceTestPeer::kNumShards));
    // Unmounting again is a no-op.
    EXPECT_TRUE(service.UnmountLocalDiskSegment(x.client_id).has_value());
    EXPECT_EQ(Metrics().get_disk_unmount_sweep_pending_owners(), 1);

    // Disk-only objects of the unmounted store: not readable, not existing.
    for (const auto& key : disk_only) {
        auto r = service.GetReplicaList(key, TenantId::Default());
        ASSERT_FALSE(r.has_value()) << key;
        EXPECT_EQ(r.error(), ErrorCode::REPLICA_IS_NOT_READY);
        EXPECT_FALSE(Exists(service, key));
    }
    for (const auto& r :
         service.BatchGetReplicaList(disk_only, TenantId::Default())) {
        EXPECT_FALSE(r.has_value());
    }
    for (const auto& r :
         service.BatchExistKey(disk_only, TenantId::Default())) {
        EXPECT_FALSE(r.has_value() && *r);
    }
    // Objects with a memory copy: only the memory copy is handed out.
    for (const auto& key : both) {
        auto replicas = Replicas(service, key);
        ASSERT_EQ(replicas.size(), 1u) << key;
        EXPECT_TRUE(replicas[0].is_memory_replica());
        EXPECT_TRUE(Exists(service, key));
    }
    for (const auto& r :
         service.BatchGetReplicaList(both, TenantId::Default())) {
        ASSERT_TRUE(r.has_value());
        EXPECT_FALSE(HasDiskOf(r->replicas, x.client_id));
    }
    // The other store is untouched.
    for (const auto& key : other) {
        EXPECT_TRUE(HasDiskOf(Replicas(service, key), y.client_id)) << key;
    }

    gate.Release();
    ASSERT_TRUE(WaitForSweeps(service));
    EXPECT_FALSE(peer.DiskUnmountSweepPendingForTesting(x.client_id));
    EXPECT_EQ(service.GetKeyCount(), keys_before - disk_only.size());
    for (const auto& key : disk_only) {
        auto r = service.GetReplicaList(key, TenantId::Default());
        ASSERT_FALSE(r.has_value());
        EXPECT_EQ(r.error(), ErrorCode::OBJECT_NOT_FOUND);
    }
    for (const auto& key : both) {
        auto replicas = Replicas(service, key);
        ASSERT_EQ(replicas.size(), 1u);
        EXPECT_TRUE(replicas[0].is_memory_replica());
    }
    for (const auto& key : other) {
        EXPECT_TRUE(HasDiskOf(Replicas(service, key), y.client_id));
    }
    EXPECT_EQ(
        Metrics().get_disk_unmount_sweep_removed_replicas() - removed_before,
        static_cast<int64_t>(disk_only.size() + both.size()));
    EXPECT_EQ(Metrics().get_disk_unmount_sweep_pending_owners(), 0);
    EXPECT_EQ(Metrics().get_disk_unmount_sweep_remaining_shards(), 0);
    EXPECT_EQ(Metrics().get_disk_unmount_sweep_remaining_replicas(), 0);
    const std::string exposition = Metrics().serialize_metrics();
    EXPECT_NE(exposition.find("master_disk_unmount_sweep_duration_ms_count"),
              std::string::npos);
    EXPECT_NE(
        exposition.find("master_disk_unmount_sweep_removed_replicas_total"),
        std::string::npos);
}

TEST(DiskUnmountSweepTest, PutAndRemoveDuringTheSweepActOnAFreshObject) {
    SweepGate gate(0);
    MasterService service(SweepConfig(true));
    ReleaseOnExit release_on_exit{gate};
    InstallGate(service, gate);
    Store x = MountStore(service, 0);
    Store y = MountStore(service, 1);
    PutDiskOnly(service, x, {"reput", "removed", "untouched"});

    ASSERT_TRUE(service.UnmountLocalDiskSegment(x.client_id).has_value());
    gate.WaitBlocked();

    // exists() is false, so a client puts the page again. The hidden disk
    // replica is treated as stale: the put creates a fresh object instead of
    // answering OBJECT_ALREADY_EXISTS for an object nobody can read.
    EXPECT_FALSE(Exists(service, "reput"));
    Put(service, y, "reput");
    auto replicas = Replicas(service, "reput");
    ASSERT_EQ(replicas.size(), 1u);
    EXPECT_TRUE(replicas[0].is_memory_replica());

    // Remove of a hidden object leaves nothing behind.
    (void)service.Remove("removed", TenantId::Default(), true);
    auto removed = service.GetReplicaList("removed", TenantId::Default());
    ASSERT_FALSE(removed.has_value());
    EXPECT_EQ(removed.error(), ErrorCode::OBJECT_NOT_FOUND);

    gate.Release();
    ASSERT_TRUE(WaitForSweeps(service));
    // The fresh object survives the sweep with its memory copy.
    replicas = Replicas(service, "reput");
    ASSERT_EQ(replicas.size(), 1u);
    EXPECT_TRUE(replicas[0].is_memory_replica());
    auto untouched = service.GetReplicaList("untouched", TenantId::Default());
    ASSERT_FALSE(untouched.has_value());
    EXPECT_EQ(untouched.error(), ErrorCode::OBJECT_NOT_FOUND);
}

TEST(DiskUnmountSweepTest, TheSameStoreMountsAgainOnlyAfterItsSweep) {
    SweepGate gate(100);
    MasterService service(SweepConfig(true));
    ReleaseOnExit release_on_exit{gate};
    MasterServiceTestPeer peer(service);
    InstallGate(service, gate);
    Store x = MountStore(service, 0);
    const auto disk_only = Keys("remount-disk-only-", 200);
    PutDiskOnly(service, x, disk_only);
    // An offload taken before the unmount, completed after it.
    Put(service, x, "late");
    auto taken = service.OffloadObjectHeartbeat(x.client_id, true);
    ASSERT_TRUE(taken.has_value());
    ASSERT_EQ(taken->size(), 1u);
    const int64_t refused_before = Metrics().get_disk_unmount_remount_refused();

    ASSERT_TRUE(service.UnmountLocalDiskSegment(x.client_id).has_value());
    gate.WaitBlocked();  // 100 shards swept, the rest not

    auto mount = service.MountLocalDiskSegment(x.client_id, true);
    ASSERT_FALSE(mount.has_value());
    EXPECT_EQ(mount.error(), ErrorCode::UNAVAILABLE_IN_CURRENT_STATUS);
    EXPECT_EQ(Metrics().get_disk_unmount_remount_refused() - refused_before, 1);
    // The store's heartbeat keeps answering SEGMENT_NOT_FOUND (the store
    // retries the mount on its next heartbeat).
    auto heartbeat = service.OffloadObjectHeartbeat(x.client_id, true);
    ASSERT_FALSE(heartbeat.has_value());
    EXPECT_EQ(heartbeat.error(), ErrorCode::SEGMENT_NOT_FOUND);
    // A completion from the unregistered store attaches nothing.
    StorageObjectMetadata meta;
    meta.data_size = kObjectSize;
    meta.transport_endpoint = x.segment + "-disk";
    (void)service.NotifyOffloadSuccess(x.client_id, *taken, {meta});
    EXPECT_FALSE(HasDiskOf(Replicas(service, "late"), x.client_id));
    for (const auto& key : disk_only) {
        EXPECT_FALSE(Exists(service, key)) << key;
    }

    gate.Release();
    ASSERT_TRUE(WaitForSweeps(service));
    // Now the mount succeeds, and the new registration starts with no replica
    // of the old one: every old disk-only object is gone.
    ASSERT_TRUE(service.MountLocalDiskSegment(x.client_id, true).has_value());
    for (const auto& key : disk_only) {
        auto r = service.GetReplicaList(key, TenantId::Default());
        ASSERT_FALSE(r.has_value());
        EXPECT_EQ(r.error(), ErrorCode::OBJECT_NOT_FOUND);
    }
    // New offloads register and are readable as usual.
    PutWithDiskCopy(service, x, {"after-remount"});
    EXPECT_TRUE(HasDiskOf(Replicas(service, "after-remount"), x.client_id));
}

TEST(DiskUnmountSweepTest, AnotherStoreMountsAndOffloadsDuringTheSweep) {
    SweepGate gate(10);
    MasterService service(SweepConfig(true));
    ReleaseOnExit release_on_exit{gate};
    InstallGate(service, gate);
    Store x = MountStore(service, 0);
    PutDiskOnly(service, x, Keys("leaving-", 50));

    ASSERT_TRUE(service.UnmountLocalDiskSegment(x.client_id).has_value());
    gate.WaitBlocked();

    // A new store process (its own client id) at the same time.
    Store late = MountStore(service, 1);
    const auto late_keys = Keys("late-", 50);
    PutDiskOnly(service, late, late_keys);
    for (const auto& key : late_keys) {
        EXPECT_TRUE(HasDiskOf(Replicas(service, key), late.client_id));
    }

    gate.Release();
    ASSERT_TRUE(WaitForSweeps(service));
    for (const auto& key : late_keys) {
        EXPECT_TRUE(HasDiskOf(Replicas(service, key), late.client_id)) << key;
    }
}

TEST(DiskUnmountSweepTest, AStoreUnmountingMidPassGetsAFullPass) {
    SweepGate gate(300);
    MasterService service(SweepConfig(true));
    ReleaseOnExit release_on_exit{gate};
    MasterServiceTestPeer peer(service);
    InstallGate(service, gate);
    Store x = MountStore(service, 0);
    Store z = MountStore(service, 1);
    const auto x_keys = Keys("first-", 2000);
    const auto z_keys = Keys("second-", 2000);
    PutDiskOnly(service, x, x_keys);
    PutDiskOnly(service, z, z_keys);
    const size_t keys_before = service.GetKeyCount();

    ASSERT_TRUE(service.UnmountLocalDiskSegment(x.client_id).has_value());
    gate.WaitBlocked();  // 300 shards swept for x
    ASSERT_TRUE(service.UnmountLocalDiskSegment(z.client_id).has_value());
    EXPECT_EQ(Metrics().get_disk_unmount_sweep_pending_owners(), 2);
    EXPECT_EQ(
        Metrics().get_disk_unmount_sweep_remaining_shards(),
        static_cast<int64_t>(2 * MasterServiceTestPeer::kNumShards - 300));
    for (const auto& key : z_keys) {
        EXPECT_FALSE(Exists(service, key));
    }

    gate.Release();
    ASSERT_TRUE(WaitForSweeps(service));
    // The parked step (shard 300) had taken its owner list before z
    // unmounted, so z was swept over 301..1023 and then 0..300.
    EXPECT_EQ(gate.steps(), 301u + MasterServiceTestPeer::kNumShards);
    EXPECT_EQ(service.GetKeyCount(),
              keys_before - x_keys.size() - z_keys.size());
    for (const auto& key : z_keys) {
        auto r = service.GetReplicaList(key, TenantId::Default());
        ASSERT_FALSE(r.has_value());
        EXPECT_EQ(r.error(), ErrorCode::OBJECT_NOT_FOUND);
    }
}

TEST(DiskUnmountSweepTest, QuiesceStopsAPendingSweepAndKeepsItHidden) {
    SweepGate gate(5);
    auto service = std::make_unique<MasterService>(SweepConfig(true));
    ReleaseOnExit release_on_exit{gate};
    MasterServiceTestPeer peer(*service);
    InstallGate(*service, gate);
    Store x = MountStore(*service, 0);
    const auto keys = Keys("quiesce-", 100);
    PutDiskOnly(*service, x, keys);

    ASSERT_TRUE(service->UnmountLocalDiskSegment(x.client_id).has_value());
    gate.WaitBlocked();
    std::thread quiesce([&] { service->QuiesceBackgroundWork(); });
    // Free the parked step only once the quiesce has set the stop flag.
    while (!peer.DiskUnmountSweepStoppingForTesting()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    gate.Release();
    quiesce.join();
    EXPECT_TRUE(service->BackgroundWorkQuiesced());
    EXPECT_TRUE(peer.DiskUnmountSweepPendingForTesting(x.client_id));
    for (const auto& key : keys) {
        EXPECT_FALSE(Exists(*service, key));
    }
}

// ---- Scale: a million disk-only objects, with readers and writers running.

TEST(DiskUnmountSweepTest,
     AMillionDiskOnlyObjectsWithConcurrentReadersAndPuts) {
    constexpr int kLeaving = 1000000;
    constexpr int kStaying = 200000;
    MasterService service(SweepConfig(true));
    MasterServiceTestPeer peer(service);
    Store x = MountStore(service, 0);
    Store y = MountStore(service, 1);
    for (int i = 0; i < kLeaving; ++i) {
        peer.InsertDiskOnlyObjectForTesting("big-x-" + std::to_string(i),
                                            kObjectSize, x.client_id, "x-disk");
    }
    for (int i = 0; i < kStaying; ++i) {
        peer.InsertDiskOnlyObjectForTesting("big-y-" + std::to_string(i),
                                            kObjectSize, y.client_id, "y-disk");
    }
    ASSERT_EQ(service.GetKeyCount(), static_cast<size_t>(kLeaving + kStaying));
    const int64_t removed_before =
        Metrics().get_disk_unmount_sweep_removed_replicas();

    std::atomic<bool> unmounted{false};
    std::atomic<bool> done{false};
    std::atomic<uint64_t> checked_reads{0};
    std::atomic<uint64_t> violations{0};
    std::vector<std::thread> readers;
    for (int t = 0; t < 4; ++t) {
        readers.emplace_back([&, t] {
            std::mt19937 rng(t);
            std::uniform_int_distribution<int> pick(0, kLeaving - 1);
            while (!done.load()) {
                const int index = pick(rng);
                // The writer below puts these again, so they may exist.
                const bool reput = index % 499 == 0 && index / 499 < 2000;
                const std::string key = "big-x-" + std::to_string(index);
                // Only reads issued after the unmount returned must not see
                // the unmounted store's disk.
                const bool after = unmounted.load();
                auto r = service.GetReplicaList(key, TenantId::Default());
                const bool exists = Exists(service, key);
                if (!after) continue;
                ++checked_reads;
                if ((r && HasDiskOf(r->replicas, x.client_id)) ||
                    (exists && !reput)) {
                    ++violations;
                }
            }
        });
    }
    // Re-puts of leaving keys during the sweep, by the staying store.
    std::atomic<int> reputs_ok{0};
    std::thread writer([&] {
        while (!unmounted.load()) std::this_thread::yield();
        for (int i = 0; i < 2000; ++i) {
            ReplicateConfig config;
            config.replica_num = 1;
            config.preferred_segment = y.segment;
            const std::string key = "big-x-" + std::to_string(i * 499);
            auto started = service.PutStart(
                y.client_id, key, TenantId::Default(), kObjectSize, config);
            if (started && service
                               .PutEnd(y.client_id, key, TenantId::Default(),
                                       ReplicaType::MEMORY)
                               .has_value()) {
                ++reputs_ok;
            }
        }
    });

    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    const auto unmount_start = std::chrono::steady_clock::now();
    ASSERT_TRUE(service.UnmountLocalDiskSegment(x.client_id).has_value());
    unmounted.store(true);
    const auto unmount_us =
        std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now() - unmount_start)
            .count();
    ASSERT_TRUE(WaitForSweeps(service));
    const auto sweep_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                              std::chrono::steady_clock::now() - unmount_start)
                              .count();
    writer.join();
    // A few more reads after the sweep, then stop.
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    done.store(true);
    for (auto& reader : readers) reader.join();

    std::cout << "unmount_us=" << unmount_us << " sweep_ms=" << sweep_ms
              << " checked_reads=" << checked_reads.load()
              << " reputs_ok=" << reputs_ok.load() << std::endl;
    EXPECT_GT(checked_reads.load(), 0u);
    EXPECT_EQ(violations.load(), 0u);
    // Every re-put succeeded: a hidden object never blocks a fresh put.
    EXPECT_EQ(reputs_ok.load(), 2000);
    EXPECT_EQ(service.GetKeyCount(), static_cast<size_t>(kStaying + 2000));
    // A re-put that reaches its key before the sweep erases the hidden
    // replica itself; the sweep counts only the rest.
    const int64_t swept =
        Metrics().get_disk_unmount_sweep_removed_replicas() - removed_before;
    EXPECT_GE(swept, kLeaving - 2000);
    EXPECT_LE(swept, kLeaving);
    for (int i = 0; i < kStaying; i += 997) {
        EXPECT_TRUE(HasDiskOf(Replicas(service, "big-y-" + std::to_string(i)),
                              y.client_id));
    }
    for (int i = 0; i < 2000; ++i) {
        auto replicas = Replicas(service, "big-x-" + std::to_string(i * 499));
        ASSERT_EQ(replicas.size(), 1u);
        EXPECT_TRUE(replicas[0].is_memory_replica());
    }
}

}  // namespace
}  // namespace mooncake::test
