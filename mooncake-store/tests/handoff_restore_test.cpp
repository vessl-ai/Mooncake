// Tests for the shutdown handoff snapshot (INF-600, fork-only): writer,
// trusted restore with real allocators, remount identity check, quarantine.

#include <gtest/gtest.h>

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <map>
#include <string>
#include <thread>
#include <vector>

#include "crc32c.h"
#include "handoff/handoff_format.h"
#include "handoff/handoff_writer.h"
#include "master_service.h"

namespace mooncake::test {
namespace {

namespace fs = std::filesystem;

constexpr uint64_t kSegmentSize = 256ULL * 1024 * 1024;
constexpr uint64_t kObjectSize = 64 * 1024;

MasterServiceConfig MakeConfig(int64_t live_ttl_sec = 3600,
                               int64_t suspicion_ttl_sec = 3600,
                               BufferAllocatorType allocator =
                                   BufferAllocatorType::OFFSET) {
    return MasterServiceConfig::builder()
        .set_memory_allocator(allocator)
        .set_eviction_ratio(0.0)
        .set_eviction_high_watermark_ratio(1.0)
        .set_client_live_ttl_sec(live_ttl_sec)
        .set_client_suspicion_ttl_sec(suspicion_ttl_sec)
        .build();
}

Segment MakeSegment(int index) {
    Segment segment;
    segment.id = generate_uuid();
    segment.name = "10.0.0." + std::to_string(10 + index) + ":15651";
    segment.base = 0x100000000000ULL + index * 0x10000000000ULL;
    segment.size = kSegmentSize;
    segment.te_endpoint = segment.name;
    segment.protocol = "rdma";
    segment.host_id = "host-" + std::to_string(index);
    return segment;
}

std::string Key(int segment, int i) {
    return "key-" + std::to_string(segment) + "-" + std::to_string(i);
}

class TempDir {
   public:
    TempDir() {
        std::string pattern =
            (fs::temp_directory_path() / "handoff-test-XXXXXX").string();
        path_ = mkdtemp(pattern.data());
    }
    ~TempDir() {
        std::error_code ec;
        fs::remove_all(path_, ec);
    }
    const std::string& path() const { return path_; }

   private:
    std::string path_;
};

// A store process as seen by the master: its client id and segments.
struct Store {
    UUID client_id;
    std::vector<Segment> segments;
};

class HandoffRestoreTest : public ::testing::Test {
   protected:
    // Mounts `num_stores` stores with one segment each and puts
    // `objects_per_store` objects on each.
    void Populate(MasterService& service, int num_stores,
                  int objects_per_store) {
        for (int s = 0; s < num_stores; ++s) {
            Store store{generate_uuid(), {MakeSegment(s)}};
            ASSERT_TRUE(
                service.MountSegment(store.segments[0], store.client_id)
                    .has_value());
            stores_.push_back(store);
            for (int i = 0; i < objects_per_store; ++i) {
                Put(service, store, Key(s, i));
            }
        }
    }

    Replica::Descriptor Put(MasterService& service, const Store& store,
                            const std::string& key, bool end = true) {
        ReplicateConfig config;
        config.replica_num = 1;
        config.preferred_segment = store.segments[0].name;
        auto started = service.PutStart(store.client_id, key,
                                        TenantId::Default(), kObjectSize,
                                        config);
        EXPECT_TRUE(started.has_value()) << toString(started.error());
        if (!started) return {};
        if (end) {
            EXPECT_TRUE(service
                            .PutEnd(store.client_id, key, TenantId::Default(),
                                    ReplicaType::MEMORY)
                            .has_value());
        }
        return started->front();
    }

    HandoffWriteResult Write(MasterService& service, const std::string& dir) {
        HandoffWriteOptions options;
        options.dir = dir;
        options.workers = 3;
        options.durable = false;
        auto written = WriteHandoffSnapshot(service, options);
        EXPECT_TRUE(written.has_value()) << written.error();
        return written ? *written : HandoffWriteResult{};
    }

    static bool Readable(MasterService& service, const std::string& key) {
        return service.GetReplicaList(key, TenantId::Default()).has_value();
    }

    static uint64_t Used(MasterService& service, const Segment& segment) {
        auto q = service.QuerySegments(segment.name);
        return q ? q->first : UINT64_MAX;
    }

    static bool Remount(MasterService& service, const Store& store) {
        return service.ReMountSegment(store.segments, store.client_id)
            .has_value();
    }

    static void ExpectEmpty(MasterService& service) {
        EXPECT_EQ(service.GetKeyCount(), 0u);
        EXPECT_EQ(service.HandoffPendingSegmentCount(), 0u);
        // A fresh store can still mount and put.
        Segment segment = MakeSegment(42);
        const UUID client = generate_uuid();
        ASSERT_TRUE(service.MountSegment(segment, client).has_value());
        ReplicateConfig config;
        config.replica_num = 1;
        ASSERT_TRUE(service
                        .PutStart(client, "after-cold-start",
                                  TenantId::Default(), kObjectSize, config)
                        .has_value());
    }

    std::vector<Store> stores_;
};

TEST_F(HandoffRestoreTest, RestoredObjectsAreGatedUntilOwnerRemounts) {
    TempDir dir;
    std::map<std::string, Replica::Descriptor> before;
    std::vector<uint64_t> used_before;
    {
        MasterService old_master(MakeConfig());
        Populate(old_master, 3, 200);
        for (int s = 0; s < 3; ++s) {
            used_before.push_back(Used(old_master, stores_[s].segments[0]));
            for (int i = 0; i < 200; ++i) {
                auto r = old_master.GetReplicaList(Key(s, i),
                                                   TenantId::Default());
                ASSERT_TRUE(r.has_value());
                before[Key(s, i)] = r->replicas.front();
            }
        }
        auto written = Write(old_master, dir.path());
        EXPECT_EQ(written.stats.objects, 600u);
    }

    MasterService master(MakeConfig());
    auto restored = master.RestoreFromHandoff(dir.path(), {4});
    ASSERT_TRUE(restored.has_value()) << restored.error();
    EXPECT_EQ(restored->objects, 600u);
    EXPECT_EQ(master.GetKeyCount(), 600u);
    EXPECT_EQ(master.HandoffPendingSegmentCount(), 3u);
    for (int s = 0; s < 3; ++s) {
        EXPECT_EQ(Used(master, stores_[s].segments[0]), used_before[s]);
    }

    // Gated: unreadable, and no segment accepts allocations.
    auto gated = master.GetReplicaList(Key(0, 0), TenantId::Default());
    ASSERT_FALSE(gated.has_value());
    EXPECT_EQ(gated.error(), ErrorCode::REPLICA_IS_NOT_READY);
    ReplicateConfig config;
    config.replica_num = 1;
    EXPECT_FALSE(master
                     .PutStart(stores_[0].client_id, "new", TenantId::Default(),
                               kObjectSize, config)
                     .has_value());

    // Ping of a restored owner is accepted and asks for a remount.
    auto ping = master.Ping(stores_[0].client_id);
    ASSERT_TRUE(ping.has_value());
    EXPECT_EQ(ping->client_status, ClientStatus::NEED_REMOUNT);

    ASSERT_TRUE(Remount(master, stores_[0]));
    EXPECT_EQ(master.HandoffSegmentsConfirmed(), 1u);
    EXPECT_LT(master.LastRemountLockHoldNs(), 50ull * 1000 * 1000);
    for (int i = 0; i < 200; ++i) {
        auto r = master.GetReplicaList(Key(0, i), TenantId::Default());
        ASSERT_TRUE(r.has_value()) << toString(r.error());
        const auto& got =
            r->replicas.front().get_memory_descriptor().buffer_descriptor;
        const auto& want =
            before[Key(0, i)].get_memory_descriptor().buffer_descriptor;
        EXPECT_EQ(got.buffer_address_, want.buffer_address_);
        EXPECT_EQ(got.size_, want.size_);
        EXPECT_EQ(got.transport_endpoint_, want.transport_endpoint_);
        EXPECT_EQ(got.protocol_, want.protocol_);
    }
    // Other stores are still gated.
    EXPECT_FALSE(Readable(master, Key(1, 0)));

    // New allocations on the confirmed segment avoid every restored range.
    auto fresh = Put(master, stores_[0], "new");
    const auto& fb = fresh.get_memory_descriptor().buffer_descriptor;
    for (int i = 0; i < 200; ++i) {
        const auto& b =
            before[Key(0, i)].get_memory_descriptor().buffer_descriptor;
        EXPECT_TRUE(fb.buffer_address_ + fb.size_ <= b.buffer_address_ ||
                    b.buffer_address_ + b.size_ <= fb.buffer_address_);
    }
    // Replica ids continue above the old master's.
    for (const auto& [key, desc] : before) {
        EXPECT_GT(fresh.id, desc.id) << key;
    }

    ASSERT_TRUE(Remount(master, stores_[1]));
    ASSERT_TRUE(Remount(master, stores_[2]));
    EXPECT_EQ(master.HandoffPendingSegmentCount(), 0u);
    EXPECT_TRUE(Readable(master, Key(2, 199)));
}

TEST_F(HandoffRestoreTest, InFlightPutIsExcludedAndItsRangeQuarantined) {
    TempDir dir;
    Replica::Descriptor in_flight;
    uint64_t used_with_in_flight = 0;
    {
        MasterService old_master(MakeConfig());
        Populate(old_master, 1, 50);
        in_flight = Put(old_master, stores_[0], "in-flight", /*end=*/false);
        used_with_in_flight = Used(old_master, stores_[0].segments[0]);
        auto written = Write(old_master, dir.path());
        EXPECT_EQ(written.stats.objects, 50u);
        EXPECT_EQ(written.stats.skipped_processing, 1u);
        EXPECT_EQ(written.stats.quarantine_ranges, 1u);
    }
    MasterService master(MakeConfig());
    ASSERT_TRUE(master.RestoreFromHandoff(dir.path(), {2}).has_value());
    EXPECT_FALSE(master.GetReplicaList("in-flight", TenantId::Default())
                     .has_value());
    EXPECT_EQ(master.GetKeyCount(), 50u);
    EXPECT_EQ(master.HandoffQuarantineBufferCount(), 1u);
    // The quarantined range is still allocated.
    EXPECT_EQ(Used(master, stores_[0].segments[0]), used_with_in_flight);
    ASSERT_TRUE(Remount(master, stores_[0]));

    // Fill the segment: no allocation may land on the quarantined range.
    const auto& q = in_flight.get_memory_descriptor().buffer_descriptor;
    for (int i = 0; i < 1000; ++i) {
        ReplicateConfig config;
        config.replica_num = 1;
        auto r = master.PutStart(stores_[0].client_id,
                                 "fill-" + std::to_string(i),
                                 TenantId::Default(), kObjectSize, config);
        if (!r) break;
        const auto& b = r->front().get_memory_descriptor().buffer_descriptor;
        ASSERT_TRUE(b.buffer_address_ + b.size_ <= q.buffer_address_ ||
                    q.buffer_address_ + q.size_ <= b.buffer_address_);
    }
    // Not yet expired: nothing released.
    master.ReleaseExpiredHandoffQuarantine(std::chrono::steady_clock::now());
    EXPECT_EQ(master.HandoffQuarantineBufferCount(), 1u);
    master.ReleaseExpiredHandoffQuarantine(std::chrono::steady_clock::now() +
                                           std::chrono::hours(1));
    EXPECT_EQ(master.HandoffQuarantineBufferCount(), 0u);
}

TEST_F(HandoffRestoreTest, LeaseExpiredObjectsAreKept) {
    TempDir dir;
    {
        MasterService old_master(MakeConfig());
        Populate(old_master, 1, 20);
        // Leases granted by reads have long expired by the time of a roll.
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        Write(old_master, dir.path());
    }
    MasterService master(MakeConfig());
    ASSERT_TRUE(master.RestoreFromHandoff(dir.path(), {1}).has_value());
    EXPECT_EQ(master.GetKeyCount(), 20u);
}

TEST_F(HandoffRestoreTest, RestartedStoreDropsItsRestoredSegment) {
    TempDir dir;
    {
        MasterService old_master(MakeConfig());
        Populate(old_master, 2, 100);
        Write(old_master, dir.path());
    }
    MasterService master(MakeConfig());
    ASSERT_TRUE(master.RestoreFromHandoff(dir.path(), {2}).has_value());

    // Store 0 restarted during the roll: new process, new client id, new
    // segment id, same endpoint, same base.
    Store restarted{generate_uuid(), {stores_[0].segments[0]}};
    restarted.segments[0].id = generate_uuid();
    ASSERT_TRUE(master.MountSegment(restarted.segments[0], restarted.client_id)
                    .has_value());
    EXPECT_EQ(master.HandoffSegmentsDropped(), 1u);
    EXPECT_FALSE(Readable(master, Key(0, 0)));

    // The new process's own objects are readable at once (the endpoint gate
    // belonged to the dropped segment).
    Put(master, restarted, "fresh");
    EXPECT_TRUE(Readable(master, "fresh"));

    // Store 1 is unaffected.
    ASSERT_TRUE(Remount(master, stores_[1]));
    EXPECT_TRUE(Readable(master, Key(1, 99)));

    // The dropped objects are erased by the cleanup worker.
    for (int i = 0; i < 100 && master.GetKeyCount() != 101; ++i) {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    EXPECT_EQ(master.GetKeyCount(), 101u);
}

TEST_F(HandoffRestoreTest, OwnerRemountWithChangedIdentityDrops) {
    TempDir dir;
    {
        MasterService old_master(MakeConfig());
        Populate(old_master, 1, 30);
        // Second segment of the same store.
        stores_[0].segments.push_back(MakeSegment(1));
        stores_[0].segments[1].name = stores_[0].segments[0].name;
        stores_[0].segments[1].te_endpoint = stores_[0].segments[0].te_endpoint;
        ASSERT_TRUE(old_master
                        .MountSegment(stores_[0].segments[1],
                                      stores_[0].client_id)
                        .has_value());
        Write(old_master, dir.path());
    }
    {
        // Same id, different size: drop, then the new identity mounts fresh.
        MasterService master(MakeConfig());
        ASSERT_TRUE(master.RestoreFromHandoff(dir.path(), {1}).has_value());
        Store changed = stores_[0];
        changed.segments[0].size = kSegmentSize / 2;
        ASSERT_TRUE(Remount(master, changed));
        EXPECT_EQ(master.HandoffSegmentsDropped(), 1u);
        EXPECT_EQ(master.HandoffSegmentsConfirmed(), 1u);
        EXPECT_FALSE(Readable(master, Key(0, 0)));
    }
    {
        // The owner no longer lists its first segment: drop it, keep the
        // listed one.
        MasterService master(MakeConfig());
        ASSERT_TRUE(master.RestoreFromHandoff(dir.path(), {1}).has_value());
        Store subset{stores_[0].client_id, {stores_[0].segments[1]}};
        ASSERT_TRUE(Remount(master, subset));
        EXPECT_EQ(master.HandoffSegmentsDropped(), 1u);
        EXPECT_EQ(master.HandoffPendingSegmentCount(), 0u);
        EXPECT_FALSE(Readable(master, Key(0, 0)));
    }
}

TEST_F(HandoffRestoreTest, OwnerThatNeverReturnsExpires) {
    TempDir dir;
    {
        MasterService old_master(MakeConfig());
        Populate(old_master, 2, 40);
        Write(old_master, dir.path());
    }
    MasterService master(MakeConfig(/*live=*/1, /*suspicion=*/1));
    ASSERT_TRUE(master.RestoreFromHandoff(dir.path(), {2}).has_value());
    ASSERT_TRUE(Remount(master, stores_[1]));
    // Store 1 keeps pinging; store 0 never comes back.
    for (int i = 0; i < 80 && master.GetKeyCount() != 40; ++i) {
        (void)master.Ping(stores_[1].client_id);
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    EXPECT_EQ(master.GetKeyCount(), 40u);
    EXPECT_TRUE(Readable(master, Key(1, 39)));
    EXPECT_FALSE(Readable(master, Key(0, 0)));
}

TEST_F(HandoffRestoreTest, RemoveWhileGatedFreesTheRange) {
    TempDir dir;
    {
        MasterService old_master(MakeConfig());
        Populate(old_master, 1, 10);
        Write(old_master, dir.path());
    }
    MasterService master(MakeConfig());
    ASSERT_TRUE(master.RestoreFromHandoff(dir.path(), {1}).has_value());
    const uint64_t used = Used(master, stores_[0].segments[0]);
    ASSERT_TRUE(master.Remove(Key(0, 0), TenantId::Default(), true)
                    .has_value());
    EXPECT_LT(Used(master, stores_[0].segments[0]), used);
    ASSERT_TRUE(Remount(master, stores_[0]));
    EXPECT_EQ(master.GetKeyCount(), 9u);
}

TEST_F(HandoffRestoreTest, NonMemoryReplicasAreDropped) {
    TempDir dir;
    {
        MasterService old_master(MakeConfig());
        Populate(old_master, 1, 2);
        // Key(0,0) gains a LOCAL_DISK replica; "disk-only" has only one.
        StorageObjectMetadata disk;
        disk.data_size = kObjectSize;
        disk.transport_endpoint = "disk-endpoint";
        std::vector<OffloadTaskItem> tasks{
            OffloadTaskItem{.tenant_id = "default",
                            .key = Key(0, 0),
                            .size = static_cast<int64_t>(kObjectSize)},
            OffloadTaskItem{.tenant_id = "default",
                            .key = "disk-only",
                            .size = static_cast<int64_t>(kObjectSize)}};
        ASSERT_TRUE(old_master
                        .NotifyOffloadSuccess(stores_[0].client_id, tasks,
                                              {disk, disk})
                        .has_value());
        auto written = Write(old_master, dir.path());
        EXPECT_EQ(written.stats.objects, 2u);
        EXPECT_EQ(written.stats.dropped_non_memory_replicas, 2u);
        EXPECT_EQ(written.stats.skipped_no_exportable_replica, 1u);
    }
    MasterService master(MakeConfig());
    ASSERT_TRUE(master.RestoreFromHandoff(dir.path(), {1}).has_value());
    ASSERT_TRUE(Remount(master, stores_[0]));
    auto r = master.GetReplicaList(Key(0, 0), TenantId::Default());
    ASSERT_TRUE(r.has_value());
    ASSERT_EQ(r->replicas.size(), 1u);
    EXPECT_TRUE(r->replicas.front().is_memory_replica());
    EXPECT_FALSE(master.GetReplicaList("disk-only", TenantId::Default())
                     .has_value());
}

TEST_F(HandoffRestoreTest, UnmountedSegmentIsNotExported) {
    TempDir dir;
    {
        MasterService old_master(MakeConfig());
        Populate(old_master, 2, 10);
        ASSERT_TRUE(old_master
                        .UnmountSegment(stores_[0].segments[0].id,
                                        stores_[0].client_id)
                        .has_value());
        auto written = Write(old_master, dir.path());
        EXPECT_EQ(written.stats.quarantine_ranges, 0u);
    }
    MasterService master(MakeConfig());
    auto restored = master.RestoreFromHandoff(dir.path(), {1});
    ASSERT_TRUE(restored.has_value()) << restored.error();
    EXPECT_EQ(restored->segments, 1u);
    EXPECT_EQ(master.GetKeyCount(), 10u);
}

// ---- Corrupt or incompatible snapshots: cold start, never a crash.

class HandoffColdStartTest : public HandoffRestoreTest {
   protected:
    void SetUp() override {
        MasterService old_master(MakeConfig());
        Populate(old_master, 2, 50);
        Write(old_master, dir_.path());
        auto bytes = handoff::ReadFileBytes(Path(handoff::kHandoffManifestFile));
        ASSERT_TRUE(bytes.has_value());
        auto manifest = handoff::DecodeManifest(
            std::string_view(reinterpret_cast<const char*>(bytes->data()),
                             bytes->size()));
        ASSERT_TRUE(manifest.has_value()) << manifest.error();
        manifest_ = *manifest;
    }
    std::string Path(const std::string& key) const {
        return dir_.path() + "/" + key;
    }
    void WriteManifest(const handoff::Manifest& m) {
        std::ofstream(Path(handoff::kHandoffManifestFile), std::ios::trunc)
            << handoff::EncodeManifest(m);
    }
    void ExpectColdStart(BufferAllocatorType allocator =
                             BufferAllocatorType::OFFSET,
                         const std::string& reason = "") {
        MasterService master(MakeConfig(3600, 3600, allocator));
        auto restored = master.RestoreFromHandoff(dir_.path(), {2});
        ASSERT_FALSE(restored.has_value());
        if (!reason.empty()) {
            EXPECT_NE(restored.error().find(reason), std::string::npos)
                << restored.error();
        }
        if (allocator == BufferAllocatorType::OFFSET) {
            ExpectEmpty(master);
        }
    }
    TempDir dir_;
    handoff::Manifest manifest_;
};

TEST_F(HandoffColdStartTest, MissingManifest) {
    fs::remove(Path(handoff::kHandoffManifestFile));
    ExpectColdStart();
}

TEST_F(HandoffColdStartTest, CorruptManifestBody) {
    std::fstream f(Path(handoff::kHandoffManifestFile),
                   std::ios::in | std::ios::out);
    f.seekp(100);
    f.put('#');
    f.close();
    ExpectColdStart();
}

TEST_F(HandoffColdStartTest, UnknownSchemaVersion) {
    std::string text = handoff::EncodeManifest(manifest_);
    text.replace(text.find(" 1 "), 3, " 9 ");
    std::ofstream(Path(handoff::kHandoffManifestFile), std::ios::trunc)
        << text;
    ExpectColdStart();
}

TEST_F(HandoffColdStartTest, CodecFingerprintMismatch) {
    manifest_.codec.object_entry ^= 1;
    WriteManifest(manifest_);
    ExpectColdStart();
}

TEST_F(HandoffColdStartTest, ChunkCrcMismatch) {
    for (const auto& chunk : manifest_.chunks) {
        if (chunk.objects == 0) continue;
        std::fstream f(Path(chunk.key), std::ios::in | std::ios::out |
                                            std::ios::binary);
        f.seekp(static_cast<std::streamoff>(chunk.bytes / 2));
        f.put('\x5a');
        break;
    }
    ExpectColdStart();
}

TEST_F(HandoffColdStartTest, TruncatedChunk) {
    for (const auto& chunk : manifest_.chunks) {
        if (chunk.objects == 0) continue;
        fs::resize_file(Path(chunk.key), chunk.bytes / 2);
        break;
    }
    ExpectColdStart();
}

TEST_F(HandoffColdStartTest, DescriptorOutsideItsSegment) {
    manifest_.segments[0].size = 4096;
    WriteManifest(manifest_);
    ExpectColdStart();
}

TEST_F(HandoffColdStartTest, SegmentIdentityMissingOwner) {
    manifest_.segments[1].owner_client_id = {0, 0};
    WriteManifest(manifest_);
    ExpectColdStart();
}

TEST_F(HandoffColdStartTest, QuarantineOverlappingALiveRange) {
    // Find one live descriptor and claim its range is quarantined.
    MasterService probe(MakeConfig());
    auto restored = probe.RestoreFromHandoff(dir_.path(), {1});
    ASSERT_TRUE(restored.has_value()) << restored.error();
    ASSERT_TRUE(Remount(probe, stores_[0]));
    auto r = probe.GetReplicaList(Key(0, 7), TenantId::Default());
    ASSERT_TRUE(r.has_value());
    const auto& b = r->replicas.front().get_memory_descriptor().buffer_descriptor;
    uint32_t segment_index = 0;
    while (manifest_.segments[segment_index].te_endpoint !=
           b.transport_endpoint_) {
        ++segment_index;
    }
    std::vector<handoff::QuarantineRange> ranges{
        {segment_index, b.buffer_address_, b.size_}};
    const auto bytes = handoff::EncodeQuarantine(ranges);
    std::ofstream(Path(handoff::kHandoffQuarantineFile),
                  std::ios::trunc | std::ios::binary)
        .write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
    manifest_.quarantine.bytes = bytes.size();
    manifest_.quarantine.crc32c = Crc32cValue(bytes.data(), bytes.size());
    manifest_.quarantine_ranges = 1;
    WriteManifest(manifest_);
    ExpectColdStart(BufferAllocatorType::OFFSET, "overlapping ranges");
}

TEST_F(HandoffColdStartTest, AllocatorTypeMismatch) {
    ExpectColdStart(BufferAllocatorType::CACHELIB);
}

TEST_F(HandoffColdStartTest, MasterNotEmpty) {
    MasterService master(MakeConfig());
    Segment segment = MakeSegment(9);
    ASSERT_TRUE(master.MountSegment(segment, generate_uuid()).has_value());
    EXPECT_FALSE(master.RestoreFromHandoff(dir_.path(), {1}).has_value());
}

}  // namespace
}  // namespace mooncake::test
