// Tests for Client::UnmountMemorySegmentsForDrain (fork-only; called by
// nothing unless a caller asks, e.g. the store's preStop through
// /api/unmount_memory_segments).
//
// A leaving store takes its memory segments off the master while it can
// still serve them: from that moment the master names none of its replicas
// (exists is false, get finds nothing), a read that already holds a
// descriptor still completes, and the segments never come back.

#include <gflags/gflags.h>
#include <glog/logging.h>
#include <gtest/gtest.h>

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "allocator.h"
#include "client_service.h"
#include "common/client_buffer_allocation.h"
#include "common/network.h"
#include "default_config.h"
#include "test_server_helpers.h"
#include "types.h"

namespace mooncake {
namespace testing {
namespace {

constexpr size_t kSegmentSize = 32 * 1024 * 1024;
constexpr size_t kLocalBufferSize = 16 * 1024 * 1024;
constexpr size_t kValueSize = 4096;
constexpr int kKeys = 16;

std::shared_ptr<Client> CreateClient(const std::string& host_name,
                                     const std::string& master_address) {
    auto client = Client::Create(host_name, "P2PHANDSHAKE", "tcp", std::nullopt,
                                 master_address);
    EXPECT_TRUE(client.has_value()) << host_name;
    return client ? *client : nullptr;
}

std::string Value(int i) {
    std::string value(kValueSize, '\0');
    for (size_t b = 0; b < kValueSize; ++b) {
        value[b] = static_cast<char>((i * 131 + b) & 0xff);
    }
    return value;
}

bool WaitVisible(const InProcMaster& master, const std::string& name,
                 bool visible, std::chrono::seconds timeout) {
    const auto deadline = std::chrono::steady_clock::now() + timeout;
    while (std::chrono::steady_clock::now() < deadline) {
        auto v = CheckSegmentVisible(master, name);
        if (v.has_value() && *v == visible) return true;
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }
    return false;
}

// A leaving store, a staying store and a writer/reader without a segment,
// against one in-process master.
class SegmentDrainTest : public ::testing::Test {
   protected:
    void SetUp() override {
        ASSERT_TRUE(master_.Start(InProcMasterConfigBuilder().build()));
        const auto ports = getFreeTcpPorts(3);
        ASSERT_EQ(ports.size(), 3u);
        for (int port : ports) ASSERT_GE(port, 10000);
        leaving_name_ = "127.0.0.1:" + std::to_string(ports[0]);
        staying_name_ = "127.0.0.1:" + std::to_string(ports[1]);
        reader_name_ = "127.0.0.1:" + std::to_string(ports[2]);
        leaving_ = CreateClient(leaving_name_, master_.master_address());
        staying_ = CreateClient(staying_name_, master_.master_address());
        reader_ = CreateClient(reader_name_, master_.master_address());
        ASSERT_TRUE(leaving_ && staying_ && reader_);

        leaving_segment_ = allocate_buffer_allocator_memory(kSegmentSize);
        staying_segment_ = allocate_buffer_allocator_memory(kSegmentSize);
        ASSERT_TRUE(leaving_segment_ && staying_segment_);
        ASSERT_TRUE(
            leaving_->MountSegment(leaving_segment_, kSegmentSize, "tcp")
                .has_value());
        ASSERT_TRUE(
            staying_->MountSegment(staying_segment_, kSegmentSize, "tcp")
                .has_value());

        local_ = std::make_unique<SimpleAllocator>(kLocalBufferSize);
        ASSERT_TRUE(reader_
                        ->RegisterLocalMemory(local_->getBase(),
                                              kLocalBufferSize, "cpu:0", false,
                                              false)
                        .has_value());
    }

    void TearDown() override {
        // Clients first (their destructors unmount and unregister), then the
        // memory they registered.
        reader_.reset();
        leaving_.reset();
        staying_.reset();
        local_.reset();
        free(leaving_segment_);
        free(staying_segment_);
        master_.Stop();
    }

    std::string Key(int i) { return "drain-key-" + std::to_string(i); }

    void PutOn(const std::string& segment_name, const std::string& key,
               const std::string& value) {
        void* buffer = local_->allocate(value.size());
        ASSERT_NE(buffer, nullptr);
        memcpy(buffer, value.data(), value.size());
        std::vector<Slice> slices{Slice{buffer, value.size()}};
        ReplicateConfig config;
        config.replica_num = 1;
        config.preferred_segment = segment_name;
        auto put = reader_->Put(key, slices, config);
        local_->deallocate(buffer, value.size());
        ASSERT_TRUE(put.has_value()) << key << " " << toString(put.error());
    }

    // Reads `key` with `query` (a descriptor taken earlier) and compares.
    void ExpectRead(const std::string& key, const QueryResult& query,
                    const std::string& value) {
        void* buffer = local_->allocate(value.size());
        ASSERT_NE(buffer, nullptr);
        std::vector<Slice> slices{Slice{buffer, value.size()}};
        auto got = reader_->Get(key, query, slices);
        ASSERT_TRUE(got.has_value()) << key << " " << toString(got.error());
        EXPECT_EQ(memcmp(buffer, value.data(), value.size()), 0) << key;
        local_->deallocate(buffer, value.size());
    }

    std::string EndpointOf(const QueryResult& query) {
        if (query.replicas.size() != 1 ||
            !query.replicas[0].is_memory_replica()) {
            return "";
        }
        return query.replicas[0]
            .get_memory_descriptor()
            .buffer_descriptor.transport_endpoint_;
    }

    // Client names (and so segment names) on free ports, so parallel ctest
    // runs do not collide. Ephemeral ports are all 5 digits, so no name is a
    // substring of another (CheckSegmentVisible matches by substring).
    std::string leaving_name_;
    std::string staying_name_;
    std::string reader_name_;

    InProcMaster master_;
    std::shared_ptr<Client> leaving_;
    std::shared_ptr<Client> staying_;
    std::shared_ptr<Client> reader_;
    void* leaving_segment_ = nullptr;
    void* staying_segment_ = nullptr;
    std::unique_ptr<SimpleAllocator> local_;
};

// Default path, pinned: without a drain the leaving store's replicas stay
// readable until its client tears down, which unmounts them.
TEST_F(SegmentDrainTest, WithoutADrainTheSegmentLeavesAtTeardown) {
    for (int i = 0; i < kKeys; ++i) PutOn(leaving_name_, Key(i), Value(i));
    std::this_thread::sleep_for(std::chrono::seconds(2));
    for (int i = 0; i < kKeys; ++i) {
        auto exists = reader_->IsExist(Key(i));
        ASSERT_TRUE(exists.has_value());
        EXPECT_TRUE(*exists) << Key(i);
    }

    leaving_.reset();
    for (int i = 0; i < kKeys; ++i) {
        auto exists = reader_->IsExist(Key(i));
        ASSERT_TRUE(exists.has_value());
        EXPECT_FALSE(*exists) << Key(i);
    }
    EXPECT_TRUE(
        WaitVisible(master_, leaving_name_, false, std::chrono::seconds(5)));
}

TEST_F(SegmentDrainTest, DrainHidesTheReplicasAtOnceAndHeldReadsFinish) {
    std::vector<QueryResult> held;
    for (int i = 0; i < kKeys; ++i) {
        PutOn(leaving_name_, Key(i), Value(i));
        auto query = reader_->Query(Key(i));
        ASSERT_TRUE(query.has_value()) << Key(i);
        ASSERT_EQ(EndpointOf(*query), leaving_->GetTransportEndpoint());
        held.push_back(*query);
    }
    PutOn(staying_name_, "staying-key", Value(99));

    // The drained-segments gauge has no series until a drain is called.
    const std::string gauge = "mooncake_client_memory_segments_drained";
    auto metrics_before = leaving_->SerializeMetrics();
    if (metrics_before.has_value()) {
        EXPECT_EQ(metrics_before->find(gauge), std::string::npos);
    }

    auto drained = leaving_->UnmountMemorySegmentsForDrain();
    ASSERT_TRUE(drained.has_value()) << toString(drained.error());
    EXPECT_EQ(*drained, 1u);

    auto metrics_after = leaving_->SerializeMetrics();
    if (metrics_after.has_value()) {
        // The sample line: the name, optional labels, then the value 1.
        bool sampled = false;
        std::istringstream lines(*metrics_after);
        for (std::string line; std::getline(lines, line);) {
            if (line.rfind(gauge, 0) == 0 && line.size() > 2 &&
                line.compare(line.size() - 2, 2, " 1") == 0) {
                sampled = true;
            }
        }
        EXPECT_TRUE(sampled) << *metrics_after;
    }
    auto staying_metrics = staying_->SerializeMetrics();
    if (staying_metrics.has_value()) {
        EXPECT_EQ(staying_metrics->find(gauge), std::string::npos);
    }

    // Right away: no key of the leaving store exists or is handed out.
    std::vector<std::string> keys;
    for (int i = 0; i < kKeys; ++i) {
        keys.push_back(Key(i));
        auto exists = reader_->IsExist(Key(i));
        ASSERT_TRUE(exists.has_value());
        EXPECT_FALSE(*exists) << Key(i);
        // No replica is handed out. Until the master's background metadata
        // sweep reaches the key the code is REPLICA_IS_NOT_READY, after it
        // OBJECT_NOT_FOUND; both are a miss to the reader.
        auto query = reader_->Query(Key(i));
        ASSERT_FALSE(query.has_value()) << Key(i);
        EXPECT_TRUE(query.error() == ErrorCode::OBJECT_NOT_FOUND ||
                    query.error() == ErrorCode::REPLICA_IS_NOT_READY)
            << Key(i) << " " << toString(query.error());
    }
    for (const auto& exists : reader_->BatchIsExist(keys)) {
        ASSERT_TRUE(exists.has_value());
        EXPECT_FALSE(*exists);
    }
    auto staying_exists = reader_->IsExist("staying-key");
    ASSERT_TRUE(staying_exists.has_value());
    EXPECT_TRUE(*staying_exists);

    // A read that took its descriptor before the drain still gets the bytes:
    // the leaving store keeps the memory registered while it lives.
    for (int i = 0; i < kKeys; ++i) ExpectRead(Key(i), held[i], Value(i));

    // The writer re-puts the missing key; it lands on the staying store.
    PutOn(leaving_name_, Key(0), Value(1000));
    auto requery = reader_->Query(Key(0));
    ASSERT_TRUE(requery.has_value());
    EXPECT_EQ(EndpointOf(*requery), staying_->GetTransportEndpoint());
    ExpectRead(Key(0), *requery, Value(1000));

    // The master no longer lists the segment; a second drain has nothing
    // left; the leaving store takes no new segment.
    EXPECT_TRUE(
        WaitVisible(master_, leaving_name_, false, std::chrono::seconds(5)));
    EXPECT_TRUE(
        WaitVisible(master_, staying_name_, true, std::chrono::seconds(5)));
    auto again = leaving_->UnmountMemorySegmentsForDrain();
    ASSERT_TRUE(again.has_value());
    EXPECT_EQ(*again, 0u);
    void* extra = allocate_buffer_allocator_memory(kSegmentSize);
    ASSERT_NE(extra, nullptr);
    auto mount = leaving_->MountSegment(extra, kSegmentSize, "tcp");
    ASSERT_FALSE(mount.has_value());
    EXPECT_EQ(mount.error(), ErrorCode::UNAVAILABLE_IN_CURRENT_STATUS);
    free(extra);
}

// A master that forgets the leaving store (restart) gets NEED_REMOUNT from
// its heartbeat; the remount carries no drained segment.
TEST_F(SegmentDrainTest, AHeartbeatRemountDoesNotBringADrainedSegmentBack) {
    ASSERT_TRUE(leaving_->UnmountMemorySegmentsForDrain().has_value());
    ASSERT_TRUE(
        WaitVisible(master_, leaving_name_, false, std::chrono::seconds(5)));

    const int rpc_port = master_.rpc_port();
    const int metrics_port = master_.http_metrics_port();
    master_.Stop();
    std::this_thread::sleep_for(std::chrono::seconds(3));
    ASSERT_TRUE(master_.Start(InProcMasterConfigBuilder()
                                  .set_rpc_port(rpc_port)
                                  .set_http_metrics_port(metrics_port)
                                  .build()));

    // The staying store remounts; then give the leaving store's heartbeat
    // the same time and more.
    ASSERT_TRUE(
        WaitVisible(master_, staying_name_, true, std::chrono::seconds(20)));
    std::this_thread::sleep_for(std::chrono::seconds(3));
    auto leaving_visible = CheckSegmentVisible(master_, leaving_name_);
    ASSERT_TRUE(leaving_visible.has_value());
    EXPECT_FALSE(*leaving_visible);

    // New writes go to the staying store only.
    PutOn(leaving_name_, "after-restart", Value(7));
    auto query = reader_->Query("after-restart");
    ASSERT_TRUE(query.has_value());
    EXPECT_EQ(EndpointOf(*query), staying_->GetTransportEndpoint());
}

// A master error leaves the segment mounted (and the store refusing new
// mounts); calling again once the master answers finishes the drain.
TEST_F(SegmentDrainTest, ADrainThatFailsIsRetriedByCallingAgain) {
    PutOn(leaving_name_, Key(0), Value(0));
    const int rpc_port = master_.rpc_port();
    const int metrics_port = master_.http_metrics_port();
    master_.Stop();

    auto failed = leaving_->UnmountMemorySegmentsForDrain();
    ASSERT_FALSE(failed.has_value());

    ASSERT_TRUE(master_.Start(InProcMasterConfigBuilder()
                                  .set_rpc_port(rpc_port)
                                  .set_http_metrics_port(metrics_port)
                                  .build()));
    // The segment is still the store's own, so its heartbeat remounts it.
    ASSERT_TRUE(
        WaitVisible(master_, leaving_name_, true, std::chrono::seconds(20)));

    auto retried = leaving_->UnmountMemorySegmentsForDrain();
    ASSERT_TRUE(retried.has_value()) << toString(retried.error());
    EXPECT_EQ(*retried, 1u);
    EXPECT_TRUE(
        WaitVisible(master_, leaving_name_, false, std::chrono::seconds(5)));
}

}  // namespace
}  // namespace testing
}  // namespace mooncake

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    gflags::ParseCommandLineFlags(&argc, &argv, false);
    mooncake::init_ylt_log_level();
    return RUN_ALL_TESTS();
}
