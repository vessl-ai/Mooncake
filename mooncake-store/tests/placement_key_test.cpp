// Tests for placement keys (--placement_key_hrw, fork-only, default off;
// placement_key.h): a put names "placement-key:<key>" in preferred_segments
// and the master ranks the serving memory segments by rendezvous hash.

#include <gtest/gtest.h>

#include <map>
#include <set>
#include <string>
#include <vector>

#include "master_service.h"
#include "placement_key.h"
#include "types.h"

namespace mooncake::test {
namespace {

constexpr uint64_t kSegmentSize = 256ULL * 1024 * 1024;

MasterServiceConfig Config(bool placement_key_hrw) {
    MasterServiceConfig config;
    config.default_kv_lease_ttl = 0;
    config.eviction_ratio = 0.0;
    config.eviction_high_watermark_ratio = 1.0;
    config.client_active_ttl_sec = 3600;
    config.client_suspicion_ttl_sec = 3600;
    config.placement_key_hrw = placement_key_hrw;
    return config;
}

struct Store {
    UUID client_id;
    Segment segment;
};

std::vector<Store> MountStores(MasterService& service, int count) {
    std::vector<Store> stores;
    for (int i = 0; i < count; ++i) {
        Store store{generate_uuid(), {}};
        store.segment.id = generate_uuid();
        store.segment.name = "10.1.0." + std::to_string(10 + i) + ":15651";
        store.segment.base = 0x100000000000ULL + i * 0x10000000000ULL;
        store.segment.size = kSegmentSize;
        store.segment.te_endpoint = store.segment.name;
        EXPECT_TRUE(
            service.MountSegment(store.segment, store.client_id).has_value());
        stores.push_back(store);
    }
    return stores;
}

std::vector<std::string> Names(const std::vector<Store>& stores) {
    std::vector<std::string> names;
    for (const auto& store : stores) names.push_back(store.segment.name);
    return names;
}

// Puts `key` and returns the segment its replica went to ("" on failure).
std::string PutTo(MasterService& service, const std::string& key,
                  const std::vector<std::string>& preferred,
                  uint64_t size = 64 * 1024) {
    ReplicateConfig config;
    config.replica_num = 1;
    config.preferred_segments = preferred;
    const UUID writer = generate_uuid();
    auto started =
        service.PutStart(writer, key, TenantId::Default(), size, config);
    if (!started.has_value() || started->size() != 1 ||
        !started->front().is_memory_replica()) {
        return "";
    }
    EXPECT_TRUE(
        service.PutEnd(writer, key, TenantId::Default(), ReplicaType::MEMORY)
            .has_value());
    return started->front()
        .get_memory_descriptor()
        .buffer_descriptor.transport_endpoint_;
}

std::string Placement(const std::string& key) {
    return std::string(kPlacementKeyPrefix) + key;
}

TEST(PlacementKeyTest, ParsesOnlyThePrefixedNonEmptyForm) {
    EXPECT_EQ(ParsePlacementKey("placement-key:abc"), "abc");
    EXPECT_FALSE(ParsePlacementKey("placement-key:").has_value());
    EXPECT_FALSE(ParsePlacementKey("10.0.0.1:15651").has_value());
    EXPECT_FALSE(ParsePlacementKey("").has_value());
    EXPECT_FALSE(ParsePlacementKey("xplacement-key:abc").has_value());
}

// The rendezvous properties the design relies on: every key ranks every
// segment, keys spread over the segments, and taking one segment out moves
// only the keys that ranked it first, each to its own second choice.
TEST(PlacementKeyTest, RankingSpreadsKeysAndMovesOnlyTheLeavingSegmentsShare) {
    std::vector<std::string> segments;
    for (int i = 0; i < 9; ++i) {
        segments.push_back("10.2.0." + std::to_string(i) + ":15651");
    }
    std::map<std::string, int> firsts;
    const int keys = 9000;
    std::vector<std::string> without_three = segments;
    without_three.erase(without_three.begin() + 3);
    int moved = 0;
    for (int k = 0; k < keys; ++k) {
        const std::string key = "chain-" + std::to_string(k);
        auto ranked = RankSegmentsForPlacementKey(key, segments);
        ASSERT_EQ(ranked.size(), segments.size());
        EXPECT_EQ(std::set<std::string>(ranked.begin(), ranked.end()).size(),
                  segments.size());
        EXPECT_EQ(ranked, RankSegmentsForPlacementKey(key, segments));
        ++firsts[ranked[0]];

        auto after = RankSegmentsForPlacementKey(key, without_three);
        if (ranked[0] == segments[3]) {
            EXPECT_EQ(after[0], ranked[1]) << key;
            ++moved;
        } else {
            EXPECT_EQ(after[0], ranked[0]) << key;
        }
    }
    ASSERT_EQ(firsts.size(), segments.size());
    for (const auto& [segment, count] : firsts) {
        EXPECT_GT(count, keys / 9 * 0.8) << segment;
        EXPECT_LT(count, keys / 9 * 1.2) << segment;
    }
    // About 1/9 of the keys ranked the removed segment first.
    EXPECT_GT(moved, keys / 9 * 0.8);
}

// Flag off (the default): the entry names no segment, so it is skipped and
// puts are placed at random as before.
TEST(PlacementKeyTest, FlagOffLeavesThePlacementRandom) {
    MasterService service(Config(false));
    auto stores = MountStores(service, 6);
    std::set<std::string> used;
    for (int i = 0; i < 64; ++i) {
        auto segment =
            PutTo(service, "off-" + std::to_string(i), {Placement("chain-a")});
        ASSERT_FALSE(segment.empty());
        used.insert(segment);
    }
    EXPECT_GT(used.size(), 1u);
}

TEST(PlacementKeyTest, OneKeyLandsOnItsFirstRankedSegment) {
    MasterService service(Config(true));
    auto stores = MountStores(service, 6);
    const auto names = Names(stores);
    std::set<std::string> firsts;
    for (const std::string chain : {"chain-a", "chain-b", "chain-c"}) {
        const auto expected = RankSegmentsForPlacementKey(chain, names)[0];
        firsts.insert(expected);
        for (int i = 0; i < 32; ++i) {
            EXPECT_EQ(PutTo(service, chain + "-" + std::to_string(i),
                            {Placement(chain)}),
                      expected)
                << chain << " " << i;
        }
    }
    // Real segment names keep their place in front of a placement key.
    const auto real =
        names[0] == RankSegmentsForPlacementKey("chain-a", names)[0] ? names[1]
                                                                     : names[0];
    EXPECT_EQ(PutTo(service, "mixed", {real, Placement("chain-a")}), real);
}

// A full first segment spills to the second-ranked one, the same for every
// put of the key; an unmounted segment is skipped the same way.
TEST(PlacementKeyTest, SpillAndLeaveFollowTheRanking) {
    MasterService service(Config(true));
    auto stores = MountStores(service, 4);
    const auto ranked = RankSegmentsForPlacementKey("chain-z", Names(stores));

    const uint64_t big = 16ULL * 1024 * 1024;
    int on_first = 0;
    std::string spilled;
    for (int i = 0; i < 64 && spilled.empty(); ++i) {
        auto segment = PutTo(service, "fill-" + std::to_string(i),
                             {Placement("chain-z")}, big);
        ASSERT_FALSE(segment.empty()) << i;
        if (segment == ranked[0]) {
            ++on_first;
        } else {
            spilled = segment;
        }
    }
    EXPECT_GT(on_first, 4);
    EXPECT_EQ(spilled, ranked[1]);
    EXPECT_EQ(PutTo(service, "spill-again", {Placement("chain-z")}, big),
              ranked[1]);

    // The second-ranked segment leaves: with the first still full, the
    // key's puts go on to its third-ranked segment.
    for (const auto& store : stores) {
        if (store.segment.name == ranked[1]) {
            ASSERT_TRUE(
                service.UnmountSegment(store.segment.id, store.client_id)
                    .has_value());
        }
    }
    EXPECT_EQ(PutTo(service, "after-leave", {Placement("chain-z")}, big),
              ranked[2]);
}

}  // namespace
}  // namespace mooncake::test
