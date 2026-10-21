#include "reuse_sampler.h"

#include "labeled_histogram.h"

#include <glog/logging.h>
#include <gtest/gtest.h>

#include <chrono>
#include <limits>
#include <map>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "master_metric_manager.h"
#include "master_service.h"
#include "master_service/master_service_test_peer.h"
#include "types.h"

namespace mooncake::test {

using Clock = ReuseSampler::Clock;
using std::chrono::seconds;

namespace {

// Fingerprints of distinct keys that land in the given sampler shard.
std::vector<uint64_t> FingerprintsInShard(size_t shard, size_t count) {
    std::vector<uint64_t> out;
    for (int i = 0; out.size() < count; ++i) {
        const uint64_t fp = ReuseSampler::Fingerprint(
            TenantId::Default(), "shard_key_" + std::to_string(i));
        if ((fp & (ReuseSampler::kNumShards - 1)) == shard) {
            out.push_back(fp);
        }
    }
    return out;
}

}  // namespace

// ---------------------------------------------------------------------------
// Sample selection
// ---------------------------------------------------------------------------

TEST(ReuseSamplerTest, SelectsTheRequestedFractionOfKeys) {
    ReuseSampler sampler(0.01, 1024);
    constexpr int kKeys = 200000;
    int sampled = 0;
    // Split by the master's metadata shard parity: the sampling bits must
    // not follow the bits that pick the metadata shard.
    int sampled_even_shard = 0;
    int even_shard = 0;
    for (int i = 0; i < kKeys; ++i) {
        const std::string key = "page_" + std::to_string(i);
        const uint64_t fp = ReuseSampler::Fingerprint(TenantId::Default(), key);
        const bool even = (std::hash<std::string>{}(key) % 1024) % 2 == 0;
        even_shard += even ? 1 : 0;
        if (sampler.IsSampled(fp)) {
            ++sampled;
            sampled_even_shard += even ? 1 : 0;
        }
    }
    // Binomial(200000, 0.01): mean 2000, sd ~44.5. Allow ~4.5 sd.
    EXPECT_NEAR(sampled, 2000, 200);
    const double even_rate = static_cast<double>(sampled_even_shard) /
                             static_cast<double>(even_shard);
    EXPECT_NEAR(even_rate, 0.01, 0.0015);
}

TEST(ReuseSamplerTest, SelectionIsStableAndScopedByTenant) {
    ReuseSampler sampler(0.5, 1024);
    const uint64_t a = ReuseSampler::Fingerprint(TenantId::Default(), "k");
    EXPECT_EQ(a, ReuseSampler::Fingerprint(TenantId::Default(), "k"));
    EXPECT_EQ(sampler.IsSampled(a), sampler.IsSampled(a));
    // The same user key in another tenant is another object.
    EXPECT_NE(a, ReuseSampler::Fingerprint(TenantId("tenant_b"), "k"));
}

TEST(ReuseSamplerTest, FullRateSamplesEveryKey) {
    ReuseSampler sampler(1.0, 1024);
    for (int i = 0; i < 10000; ++i) {
        EXPECT_TRUE(sampler.IsSampled(ReuseSampler::Fingerprint(
            TenantId::Default(), "any_" + std::to_string(i))));
    }
}

TEST(ReuseSamplerTest, RejectsInvalidSettings) {
    EXPECT_THROW(ReuseSampler(0.0, 1024), std::invalid_argument);
    EXPECT_THROW(ReuseSampler(1.5, 1024), std::invalid_argument);
    EXPECT_THROW(ReuseSampler(0.01, 0), std::invalid_argument);
    EXPECT_THROW(ReuseSampler(0.01, ReuseSampler::kMaxEntriesLimit + 1),
                 std::invalid_argument);
}

// ---------------------------------------------------------------------------
// Time calculation
// ---------------------------------------------------------------------------

TEST(ReuseSamplerTest, ReuseAgeIsTheTimeSinceTheMemoryEviction) {
    ReuseSampler sampler(1.0, 1024);
    const uint64_t fp = ReuseSampler::Fingerprint(TenantId::Default(), "k");
    const auto t0 = Clock::now();

    sampler.RecordMemoryEviction(fp, /*disk_copy_kept=*/true, t0);
    auto reuse = sampler.RecordReference(fp, t0 + seconds(42));
    ASSERT_TRUE(reuse.has_value());
    ASSERT_TRUE(reuse->since_memory_eviction.has_value());
    EXPECT_EQ(*reuse->since_memory_eviction, seconds(42));
    EXPECT_FALSE(reuse->since_disk_eviction.has_value());
    EXPECT_TRUE(reuse->disk_copy_present);

    // Only the first reference after the eviction reports an age.
    EXPECT_FALSE(sampler.RecordReference(fp, t0 + seconds(50)).has_value());
    EXPECT_EQ(sampler.GetStats().entries, 0u);
    EXPECT_EQ(sampler.GetStats().reuses, 1u);
}

TEST(ReuseSamplerTest, MemoryEvictionWithoutCopyIsAMiss) {
    ReuseSampler sampler(1.0, 1024);
    const uint64_t fp = ReuseSampler::Fingerprint(TenantId::Default(), "k");
    const auto t0 = Clock::now();
    sampler.RecordMemoryEviction(fp, /*disk_copy_kept=*/false, t0);
    auto reuse = sampler.RecordReference(fp, t0 + seconds(7));
    ASSERT_TRUE(reuse.has_value());
    EXPECT_FALSE(reuse->disk_copy_present);
    EXPECT_EQ(*reuse->since_memory_eviction, seconds(7));
}

TEST(ReuseSamplerTest, DiskEvictionKeepsTheMemoryEvictionTime) {
    ReuseSampler sampler(1.0, 1024);
    const uint64_t fp = ReuseSampler::Fingerprint(TenantId::Default(), "k");
    const auto t0 = Clock::now();
    sampler.RecordMemoryEviction(fp, /*disk_copy_kept=*/true, t0);
    sampler.RecordDiskEviction(fp, t0 + seconds(100));
    EXPECT_EQ(sampler.GetStats().entries, 1u);

    auto reuse = sampler.RecordReference(fp, t0 + seconds(130));
    ASSERT_TRUE(reuse.has_value());
    EXPECT_EQ(*reuse->since_memory_eviction, seconds(130));
    EXPECT_EQ(*reuse->since_disk_eviction, seconds(30));
    EXPECT_FALSE(reuse->disk_copy_present);
}

TEST(ReuseSamplerTest, DiskEvictionAloneHasNoMemoryAge) {
    ReuseSampler sampler(1.0, 1024);
    const uint64_t fp = ReuseSampler::Fingerprint(TenantId::Default(), "k");
    const auto t0 = Clock::now();
    sampler.RecordDiskEviction(fp, t0);
    auto reuse = sampler.RecordReference(fp, t0 + seconds(3));
    ASSERT_TRUE(reuse.has_value());
    EXPECT_FALSE(reuse->since_memory_eviction.has_value());
    EXPECT_EQ(*reuse->since_disk_eviction, seconds(3));
}

TEST(ReuseSamplerTest, LaterEvictionReplacesTheGhost) {
    ReuseSampler sampler(1.0, 1024);
    const uint64_t fp = ReuseSampler::Fingerprint(TenantId::Default(), "k");
    const auto t0 = Clock::now();
    sampler.RecordMemoryEviction(fp, false, t0);
    sampler.RecordMemoryEviction(fp, true, t0 + seconds(10));
    EXPECT_EQ(sampler.GetStats().entries, 1u);
    auto reuse = sampler.RecordReference(fp, t0 + seconds(15));
    ASSERT_TRUE(reuse.has_value());
    EXPECT_EQ(*reuse->since_memory_eviction, seconds(5));
    EXPECT_TRUE(reuse->disk_copy_present);
}

TEST(ReuseSamplerTest, ReferenceWithoutGhostReportsNothing) {
    ReuseSampler sampler(1.0, 1024);
    const uint64_t fp = ReuseSampler::Fingerprint(TenantId::Default(), "k");
    EXPECT_FALSE(sampler.RecordReference(fp, Clock::now()).has_value());
    EXPECT_EQ(sampler.GetStats().reuses, 0u);
}

// ---------------------------------------------------------------------------
// Table overflow
// ---------------------------------------------------------------------------

TEST(ReuseSamplerTest, FullTableOverwritesTheOldestGhost) {
    // 4 ghosts per shard.
    ReuseSampler sampler(1.0, 4 * ReuseSampler::kNumShards);
    ASSERT_EQ(sampler.capacity(), 4 * ReuseSampler::kNumShards);
    const auto fps = FingerprintsInShard(3, 5);
    const auto t0 = Clock::now();
    const int64_t overflow_before =
        MasterMetricManager::instance().get_reuse_sampler_ghost_overflow();

    for (size_t i = 0; i < fps.size(); ++i) {
        sampler.RecordMemoryEviction(fps[i], false, t0 + seconds(10 * i));
    }
    const auto stats = sampler.GetStats();
    EXPECT_EQ(stats.entries, 4u);
    EXPECT_EQ(stats.overflowed, 1u);
    EXPECT_EQ(
        MasterMetricManager::instance().get_reuse_sampler_ghost_overflow() -
            overflow_before,
        1);
    // The oldest ghost (written at t0) was overwritten at t0 + 40 s.
    EXPECT_EQ(
        MasterMetricManager::instance().get_reuse_sampler_window_seconds(), 40);

    const auto now = t0 + seconds(100);
    EXPECT_FALSE(sampler.RecordReference(fps[0], now).has_value());
    for (size_t i = 1; i < fps.size(); ++i) {
        auto reuse = sampler.RecordReference(fps[i], now);
        ASSERT_TRUE(reuse.has_value()) << i;
        EXPECT_EQ(*reuse->since_memory_eviction, seconds(100 - 10 * i));
    }
}

TEST(ReuseSamplerTest, ConsumedSlotIsReusedWithoutOverflow) {
    ReuseSampler sampler(1.0, 2 * ReuseSampler::kNumShards);
    const auto fps = FingerprintsInShard(5, 3);
    const auto t0 = Clock::now();
    sampler.RecordMemoryEviction(fps[0], false, t0);
    sampler.RecordMemoryEviction(fps[1], false, t0);
    ASSERT_TRUE(sampler.RecordReference(fps[0], t0).has_value());
    // The ring wraps onto the consumed slot: nothing live is dropped.
    sampler.RecordMemoryEviction(fps[2], false, t0);
    EXPECT_EQ(sampler.GetStats().overflowed, 0u);
    EXPECT_TRUE(sampler.RecordReference(fps[1], t0).has_value());
    EXPECT_TRUE(sampler.RecordReference(fps[2], t0).has_value());
}

// Random operations against a model of the ring: shard s writes its n-th
// ghost into slot n mod C, so a ghost written as the shard's n-th survives
// until the shard writes its (n + C)-th, unless referenced or replaced.
// A small table keeps the open-addressing index crowded, which exercises
// probing and the backward-shift deletion.
TEST(ReuseSamplerTest, MatchesARingModelUnderRandomOperations) {
    constexpr size_t kPerShard = 8;
    ReuseSampler sampler(1.0, kPerShard * ReuseSampler::kNumShards);
    struct Ghost {
        int64_t memory_s{-1};
        int64_t disk_s{-1};
        bool disk_copy{false};
        uint64_t seq{0};
    };
    std::unordered_map<uint64_t, Ghost> model;
    std::vector<uint64_t> shard_seq(ReuseSampler::kNumShards, 0);
    // (shard, seq) -> fingerprint of the live ghost written then.
    std::map<std::pair<size_t, uint64_t>, uint64_t> by_seq;
    uint64_t model_overflow = 0;

    std::vector<uint64_t> keys;
    for (int i = 0; i < 400; ++i) {
        keys.push_back(ReuseSampler::Fingerprint(TenantId::Default(),
                                                 "model_" + std::to_string(i)));
    }
    auto shard_of = [](uint64_t fp) {
        return static_cast<size_t>(fp & (ReuseSampler::kNumShards - 1));
    };
    auto drop = [&](uint64_t fp) {
        auto it = model.find(fp);
        if (it == model.end()) return;
        by_seq.erase({shard_of(fp), it->second.seq});
        model.erase(it);
    };
    auto write = [&](uint64_t fp, Ghost ghost) {
        const size_t shard = shard_of(fp);
        const uint64_t seq = shard_seq[shard]++;
        if (seq >= kPerShard) {
            auto old = by_seq.find({shard, seq - kPerShard});
            if (old != by_seq.end()) {
                model.erase(old->second);
                by_seq.erase(old);
                ++model_overflow;
            }
        }
        ghost.seq = seq;
        model[fp] = ghost;
        by_seq[{shard, seq}] = fp;
    };

    std::mt19937 rng(12345);
    const auto t0 = Clock::now();
    for (int64_t step = 1; step <= 20000; ++step) {
        const uint64_t fp = keys[rng() % keys.size()];
        const auto now = t0 + seconds(step);
        switch (rng() % 3) {
            case 0: {
                const bool disk = rng() % 2 == 0;
                sampler.RecordMemoryEviction(fp, disk, now);
                drop(fp);
                write(fp, Ghost{.memory_s = step, .disk_copy = disk});
                break;
            }
            case 1: {
                sampler.RecordDiskEviction(fp, now);
                Ghost ghost{.disk_s = step};
                auto it = model.find(fp);
                if (it != model.end()) ghost.memory_s = it->second.memory_s;
                drop(fp);
                write(fp, ghost);
                break;
            }
            default: {
                auto reuse = sampler.RecordReference(fp, now);
                auto it = model.find(fp);
                ASSERT_EQ(reuse.has_value(), it != model.end())
                    << "step " << step;
                if (reuse) {
                    const Ghost& g = it->second;
                    ASSERT_EQ(reuse->since_memory_eviction.has_value(),
                              g.memory_s >= 0);
                    if (g.memory_s >= 0) {
                        ASSERT_EQ(*reuse->since_memory_eviction,
                                  seconds(step - g.memory_s));
                    }
                    ASSERT_EQ(reuse->since_disk_eviction.has_value(),
                              g.disk_s >= 0);
                    if (g.disk_s >= 0) {
                        ASSERT_EQ(*reuse->since_disk_eviction,
                                  seconds(step - g.disk_s));
                    }
                    ASSERT_EQ(reuse->disk_copy_present, g.disk_copy);
                    drop(fp);
                }
                break;
            }
        }
    }
    const auto stats = sampler.GetStats();
    EXPECT_EQ(stats.entries, model.size());
    EXPECT_EQ(stats.overflowed, model_overflow);
    EXPECT_GT(model_overflow, 0u);
    EXPECT_LE(stats.entries, sampler.capacity());
}

// ---------------------------------------------------------------------------
// Histogram exposition
// ---------------------------------------------------------------------------

TEST(LabeledHistogramTest, SerializesEachSeriesOnceWithItsLabels) {
    LabeledHistogram<2> histogram(
        "demo_seconds", "Demo.", {1, 10}, {"state", "op"},
        {{"disk", "get"}, {"none", "get"}, {"none", "exist"}});
    std::string empty;
    histogram.serialize(empty);
    EXPECT_TRUE(empty.empty()) << "no observation, no output";

    histogram.observe({"disk", "get"}, 0.5);
    histogram.observe({"disk", "get"}, 10);  // le is inclusive
    histogram.observe({"none", "get"}, 30);
    std::string text;
    histogram.serialize(text);
    EXPECT_EQ(text,
              "# HELP demo_seconds Demo.\n"
              "# TYPE demo_seconds histogram\n"
              "demo_seconds_bucket{state=\"disk\",op=\"get\",le=\"1\"} 1\n"
              "demo_seconds_bucket{state=\"disk\",op=\"get\",le=\"10\"} 2\n"
              "demo_seconds_bucket{state=\"disk\",op=\"get\",le=\"+Inf\"} 2\n"
              "demo_seconds_sum{state=\"disk\",op=\"get\"} 10.5\n"
              "demo_seconds_count{state=\"disk\",op=\"get\"} 2\n"
              "demo_seconds_bucket{state=\"none\",op=\"get\",le=\"1\"} 0\n"
              "demo_seconds_bucket{state=\"none\",op=\"get\",le=\"10\"} 0\n"
              "demo_seconds_bucket{state=\"none\",op=\"get\",le=\"+Inf\"} 1\n"
              "demo_seconds_sum{state=\"none\",op=\"get\"} 30\n"
              "demo_seconds_count{state=\"none\",op=\"get\"} 1\n");
    EXPECT_EQ(histogram.count_at_or_below({"disk", "get"}, 1), 1);
    EXPECT_EQ(histogram.count_at_or_below(
                  {"disk", "get"}, std::numeric_limits<double>::infinity()),
              2);
    EXPECT_DOUBLE_EQ(histogram.sum({"none", "get"}), 30);
    EXPECT_THROW(histogram.observe({"disk", "put_start"}, 1),
                 std::invalid_argument);
}

// ---------------------------------------------------------------------------
// MasterService integration
// ---------------------------------------------------------------------------

class ReuseSamplerServiceTest : public ::testing::Test {
   protected:
    static constexpr size_t kSegmentSize = 64 * 1024 * 1024;
    static constexpr size_t kObjectSize = 1024 * 1024;

    struct Store {
        std::string segment_name;
        UUID client_id;
    };

    void SetUp() override {
        google::InitGoogleLogging("ReuseSamplerServiceTest");
        FLAGS_logtostderr = true;
    }

    void TearDown() override {
        if (service_) {
            service_->RemoveAll();
            service_.reset();
        }
        google::ShutdownGoogleLogging();
    }

    // Lease 0 and watermark 1.0: every completed object is a candidate and
    // nothing is evicted unless a test runs a cycle.
    static MasterServiceConfig BaseConfig() {
        MasterServiceConfig config;
        config.default_kv_lease_ttl = 0;
        config.eviction_ratio = 0.05;
        config.eviction_high_watermark_ratio = 1.0;
        return config;
    }

    static MasterServiceConfig SamplerConfig() {
        MasterServiceConfig config = BaseConfig();
        config.enable_reuse_sampler = true;
        config.reuse_sampler_rate = 1.0;
        config.reuse_sampler_max_entries = 1024;
        return config;
    }

    void Start(const MasterServiceConfig& config) {
        service_ = std::make_unique<MasterService>(config);
    }

    Store AddStore(const std::string& name, bool with_disk) {
        Segment segment;
        segment.id = generate_uuid();
        segment.name = name;
        segment.base = 0x300000000;
        segment.size = kSegmentSize;
        segment.te_endpoint = name;
        Store store{name, generate_uuid()};
        EXPECT_TRUE(service_->MountSegment(segment, store.client_id));
        if (with_disk) {
            EXPECT_TRUE(service_->MountLocalDiskSegment(store.client_id, true));
            EXPECT_TRUE(
                service_->ReportSsdCapacity(store.client_id, 1LL << 34));
        }
        return store;
    }

    void Put(const Store& store, const std::string& key) {
        ReplicateConfig config;
        config.replica_num = 1;
        config.preferred_segment = store.segment_name;
        ASSERT_TRUE(service_->PutStart(
            store.client_id, key, TenantId::Default(), kObjectSize, config));
        ASSERT_TRUE(service_->PutEnd(store.client_id, key, TenantId::Default(),
                                     ReplicaType::MEMORY));
    }

    void Evict() {
        MasterServiceTestPeer(*service_).RunBatchEvictForTesting(1.0, 1.0);
    }

    // offload_on_evict: queue, copy and complete every candidate, then evict
    // again so the memory copies go and the disk copies stay.
    void EvictToDisk(const Store& store) {
        Evict();
        auto taken = service_->OffloadObjectHeartbeat(store.client_id, true);
        ASSERT_TRUE(taken.has_value());
        std::vector<StorageObjectMetadata> metas(taken->size());
        for (auto& meta : metas) {
            meta.data_size = static_cast<int64_t>(kObjectSize);
            meta.transport_endpoint = store.segment_name;
        }
        ASSERT_TRUE(
            service_->NotifyOffloadSuccess(store.client_id, *taken, metas));
        Evict();
    }

    ReuseSampler* Sampler() {
        return MasterServiceTestPeer(*service_).GetReuseSamplerForTesting();
    }

    static int64_t MemoryReuses(
        const std::string& state, const std::string& op,
        double le = std::numeric_limits<double>::infinity()) {
        return MasterMetricManager::instance()
            .get_reuse_sampler_memory_reuse_age_count(state, op, le);
    }
    static int64_t DiskReuses(const std::string& op) {
        return MasterMetricManager::instance()
            .get_reuse_sampler_disk_reuse_age_count(op);
    }
    static int64_t Evicted(const std::string& tier) {
        return MasterMetricManager::instance()
            .get_reuse_sampler_evicted_age_count(tier);
    }

    // Every series the sampler can write, so a test can tell nothing moved.
    static std::vector<int64_t> AllSamplerCounts() {
        std::vector<int64_t> counts;
        for (const char* state : {"disk", "none"}) {
            for (const char* op : {"exist", "get", "put_start"}) {
                counts.push_back(MemoryReuses(state, op));
            }
        }
        for (const char* op : {"exist", "get", "put_start"}) {
            counts.push_back(DiskReuses(op));
        }
        counts.push_back(Evicted("memory"));
        counts.push_back(Evicted("disk"));
        counts.push_back(
            MasterMetricManager::instance().get_reuse_sampler_ghost_overflow());
        return counts;
    }

    std::unique_ptr<MasterService> service_;
};

TEST_F(ReuseSamplerServiceTest, DefaultsKeepTheSamplerOff) {
    MasterServiceConfig defaults;
    EXPECT_FALSE(defaults.enable_reuse_sampler);
    EXPECT_FALSE(defaults.enable_offload_stall_metrics);
    EXPECT_DOUBLE_EQ(defaults.reuse_sampler_rate, 0.01);
    EXPECT_EQ(defaults.reuse_sampler_max_entries, 1000000u);
}

TEST_F(ReuseSamplerServiceTest, FlagOffRecordsNothing) {
    Start(BaseConfig());
    EXPECT_EQ(Sampler(), nullptr);
    const Store store = AddStore("store_off", /*with_disk=*/false);
    const auto before = AllSamplerCounts();

    Put(store, "off_key");
    Evict();
    ASSERT_FALSE(service_->ExistKey("off_key", TenantId::Default()).value());
    EXPECT_FALSE(service_->GetReplicaList("off_key", TenantId::Default()));
    Put(store, "off_key");

    EXPECT_EQ(AllSamplerCounts(), before);
}

TEST_F(ReuseSamplerServiceTest, EvictedKeyAskedForAgainIsAMemoryMiss) {
    Start(SamplerConfig());
    ASSERT_NE(Sampler(), nullptr);
    const Store store = AddStore("store_a", /*with_disk=*/false);
    const int64_t evicted_before = Evicted("memory");
    const int64_t miss_before = MemoryReuses("none", "exist");
    const int64_t miss_fast_before = MemoryReuses("none", "exist", 1);
    const int64_t put_before = MemoryReuses("none", "put_start");

    Put(store, "miss_key");
    Evict();
    ASSERT_FALSE(service_->GetReplicaList("miss_key", TenantId::Default()));
    EXPECT_EQ(Evicted("memory") - evicted_before, 1);
    EXPECT_EQ(Sampler()->GetStats().memory_evictions, 1u);

    // GetReplicaList above was the first reference: it consumed the ghost.
    // Evict again and ask with ExistKey this time.
    Put(store, "miss_key");
    EXPECT_EQ(MemoryReuses("none", "put_start") - put_before, 0);
    Evict();
    ASSERT_FALSE(service_->ExistKey("miss_key", TenantId::Default()).value());
    EXPECT_EQ(MemoryReuses("none", "exist") - miss_before, 1);
    // Asked again within a second: the age lands in the le=1 bucket.
    EXPECT_EQ(MemoryReuses("none", "exist", 1) - miss_fast_before, 1);

    // Only the first reference counts; the re-put that follows does not.
    ASSERT_FALSE(service_->ExistKey("miss_key", TenantId::Default()).value());
    Put(store, "miss_key");
    EXPECT_EQ(MemoryReuses("none", "exist") - miss_before, 1);
    EXPECT_EQ(MemoryReuses("none", "put_start") - put_before, 0);
    EXPECT_EQ(Sampler()->GetStats().entries, 0u);
}

TEST_F(ReuseSamplerServiceTest, ReputWithoutLookupCountsAsPutStart) {
    Start(SamplerConfig());
    const Store store = AddStore("store_a", /*with_disk=*/false);
    const int64_t put_before = MemoryReuses("none", "put_start");
    Put(store, "reput_key");
    Evict();
    Put(store, "reput_key");
    EXPECT_EQ(MemoryReuses("none", "put_start") - put_before, 1);
}

TEST_F(ReuseSamplerServiceTest, KeyKeptOnDiskIsAMemoryReuseWithACopy) {
    MasterServiceConfig config = SamplerConfig();
    config.enable_offload = true;
    config.offload_on_evict = true;
    Start(config);
    const Store store = AddStore("store_d", /*with_disk=*/true);
    const int64_t disk_get_before = MemoryReuses("disk", "get");

    Put(store, "disk_key");
    Evict();  // queues the offload; the pinned memory copy stays
    EXPECT_EQ(Sampler()->GetStats().memory_evictions, 0u);
    auto taken = service_->OffloadObjectHeartbeat(store.client_id, true);
    ASSERT_TRUE(taken.has_value());
    ASSERT_EQ(taken->size(), 1u);
    std::vector<StorageObjectMetadata> metas(1);
    metas[0].data_size = static_cast<int64_t>(kObjectSize);
    metas[0].transport_endpoint = store.segment_name;
    ASSERT_TRUE(service_->NotifyOffloadSuccess(store.client_id, *taken, metas));
    Evict();  // the memory copy goes, the disk copy stays
    EXPECT_EQ(Sampler()->GetStats().memory_evictions, 1u);

    auto list = service_->GetReplicaList("disk_key", TenantId::Default());
    ASSERT_TRUE(list.has_value());
    ASSERT_EQ(list->replicas.size(), 1u);
    EXPECT_TRUE(list->replicas[0].is_local_disk_replica());
    EXPECT_EQ(MemoryReuses("disk", "get") - disk_get_before, 1);
}

TEST_F(ReuseSamplerServiceTest, DiskEvictionThenReputReportsBothAges) {
    MasterServiceConfig config = SamplerConfig();
    config.enable_offload = true;
    config.offload_on_evict = true;
    Start(config);
    const Store store = AddStore("store_d", /*with_disk=*/true);
    const int64_t evicted_disk_before = Evicted("disk");
    const int64_t none_put_before = MemoryReuses("none", "put_start");
    const int64_t disk_put_before = DiskReuses("put_start");

    Put(store, "gone_key");
    EvictToDisk(store);
    ASSERT_TRUE(service_->EvictDiskReplica(store.client_id, "gone_key",
                                           TenantId::Default(),
                                           ReplicaType::LOCAL_DISK));
    EXPECT_EQ(Evicted("disk") - evicted_disk_before, 1);
    EXPECT_EQ(Sampler()->GetStats().disk_evictions, 1u);
    EXPECT_EQ(Sampler()->GetStats().entries, 1u);

    Put(store, "gone_key");
    EXPECT_EQ(MemoryReuses("none", "put_start") - none_put_before, 1);
    EXPECT_EQ(DiskReuses("put_start") - disk_put_before, 1);
}

TEST_F(ReuseSamplerServiceTest, DiskEvictionOfAKeyStillInMemoryLeavesNoGhost) {
    MasterServiceConfig config = SamplerConfig();
    config.enable_offload = true;
    config.offload_on_evict = true;
    Start(config);
    const Store store = AddStore("store_d", /*with_disk=*/true);
    Put(store, "inclusive_key");
    Evict();
    auto taken = service_->OffloadObjectHeartbeat(store.client_id, true);
    ASSERT_TRUE(taken.has_value());
    std::vector<StorageObjectMetadata> metas(taken->size());
    for (auto& meta : metas) {
        meta.data_size = static_cast<int64_t>(kObjectSize);
        meta.transport_endpoint = store.segment_name;
    }
    ASSERT_TRUE(service_->NotifyOffloadSuccess(store.client_id, *taken, metas));
    // Memory and disk copies now; drop the disk copy only.
    ASSERT_TRUE(service_->EvictDiskReplica(store.client_id, "inclusive_key",
                                           TenantId::Default(),
                                           ReplicaType::LOCAL_DISK));
    const auto stats = Sampler()->GetStats();
    EXPECT_EQ(stats.disk_evictions, 0u);
    EXPECT_EQ(stats.memory_evictions, 0u);
    EXPECT_EQ(stats.entries, 0u);
}

TEST_F(ReuseSamplerServiceTest, ProbeIsNotAReference) {
    Start(SamplerConfig());
    const Store store = AddStore("store_a", /*with_disk=*/false);
    Put(store, "probe_key");
    Evict();
    ASSERT_FALSE(service_->ProbeKey("probe_key", TenantId::Default()).value());
    EXPECT_EQ(Sampler()->GetStats().entries, 1u);
    auto batch = service_->BatchExistKey({"probe_key"}, TenantId::Default());
    ASSERT_EQ(batch.size(), 1u);
    EXPECT_EQ(Sampler()->GetStats().entries, 0u);
    EXPECT_EQ(Sampler()->GetStats().reuses, 1u);
}

TEST_F(ReuseSamplerServiceTest, UnsampledKeysNeverReachTheTable) {
    MasterServiceConfig config = SamplerConfig();
    config.reuse_sampler_rate = 0.05;
    Start(config);
    const Store store = AddStore("store_a", /*with_disk=*/false);
    ReuseSampler probe(0.05, 16);
    int sampled = 0;
    for (int i = 0; i < 40; ++i) {
        const std::string key = "few_" + std::to_string(i);
        Put(store, key);
        sampled +=
            probe.IsSampled(ReuseSampler::Fingerprint(TenantId::Default(), key))
                ? 1
                : 0;
    }
    Evict();
    EXPECT_EQ(Sampler()->GetStats().memory_evictions,
              static_cast<uint64_t>(sampled));
}

TEST_F(ReuseSamplerServiceTest, MetricsAreExportedWhenOn) {
    Start(SamplerConfig());
    const Store store = AddStore("store_a", /*with_disk=*/false);
    Put(store, "exported_key");
    Evict();
    ASSERT_FALSE(
        service_->ExistKey("exported_key", TenantId::Default()).value());
    const std::string text =
        MasterMetricManager::instance().serialize_metrics();
    EXPECT_NE(text.find("master_reuse_sampler_memory_reuse_age_seconds_bucket{"
                        "state=\"none\",op=\"exist\""),
              std::string::npos);
    EXPECT_NE(text.find("master_reuse_sampler_evicted_age_seconds_count{"
                        "tier=\"memory\"}"),
              std::string::npos);
    EXPECT_NE(text.find("master_reuse_sampler_sampling_rate 1"),
              std::string::npos);
    EXPECT_NE(text.find("master_reuse_sampler_capacity_entries 1024"),
              std::string::npos);
}

}  // namespace mooncake::test

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
