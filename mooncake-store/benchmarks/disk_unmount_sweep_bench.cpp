// Microbenchmark for the LOCAL_DISK unmount sweep (fork-only): how long
// UnmountLocalDiskSegment takes to return, how long the sweep takes in all,
// how long it holds the metadata shard locks, and what readers of other
// stores' objects see meanwhile, with the synchronous sweep (default) and
// with --async_disk_unmount_sweep.
//
// Each run builds a fresh master with `num_keys` disk-only objects spread
// evenly over `owners` stores, unmounts the first store's disk tier, and
// prints one CSV line.
//   mode=sync   UnmountLocalDiskSegment with the flag off (the call returns
//               after the sweep)
//   mode=async  the same with the flag on (the call returns at once; the
//               sweep runs on the background thread)
//   mode=stats  the owner sweep driven shard by shard with lock-hold timers
//               (the same per-shard routine both modes run)

#include "master_service.h"
#include "master_service/master_service_test_peer.h"

#include <algorithm>
#include <atomic>
#include <fstream>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "gflags/gflags.h"
#include "glog/logging.h"
#include "types.h"

DEFINE_string(num_keys, "1000000,5000000,20000000",
              "Comma-separated total object counts");
DEFINE_string(modes, "sync,async,stats", "Comma-separated modes to run");
DEFINE_uint32(owners, 9, "Stores the objects are spread over");
DEFINE_uint32(readers, 4,
              "Reader threads calling GetReplicaList on other stores' "
              "objects while the sweep runs (0 = none)");
DEFINE_uint32(insert_threads, 16, "Threads used to build the population");
DEFINE_uint64(object_size, 450 * 1024, "Object size in bytes");

namespace mooncake::benchmarks {
namespace {

using Clock = std::chrono::steady_clock;
using mooncake::test::MasterServiceTestPeer;

std::vector<std::string> Split(const std::string& s) {
    std::vector<std::string> out;
    std::stringstream in(s);
    std::string item;
    while (std::getline(in, item, ',')) {
        if (!item.empty()) out.push_back(item);
    }
    return out;
}

// Resident set of this process in GiB (VmRSS), 0 if unreadable.
double RssGiB() {
    std::ifstream status("/proc/self/status");
    std::string line;
    while (std::getline(status, line)) {
        if (line.rfind("VmRSS:", 0) == 0) {
            return std::stod(line.substr(6)) / (1024.0 * 1024.0);
        }
    }
    return 0;
}

double Ms(Clock::duration d) {
    return std::chrono::duration<double, std::milli>(d).count();
}

std::string Key(uint32_t owner, uint64_t i) {
    return "bench-" + std::to_string(owner) + "-" + std::to_string(i);
}

struct Population {
    std::vector<UUID> owners;
    uint64_t per_owner = 0;
};

Population Build(MasterService& service, uint64_t num_keys) {
    MasterServiceTestPeer peer(service);
    Population pop;
    pop.per_owner = num_keys / FLAGS_owners;
    for (uint32_t o = 0; o < FLAGS_owners; ++o) {
        const UUID id = generate_uuid();
        CHECK(service.MountLocalDiskSegment(id, true).has_value());
        CHECK(service.ReportSsdCapacity(id, 1LL << 50).has_value());
        pop.owners.push_back(id);
    }
    std::vector<std::thread> threads;
    const uint64_t total = pop.per_owner * FLAGS_owners;
    for (uint32_t t = 0; t < FLAGS_insert_threads; ++t) {
        threads.emplace_back([&, t] {
            for (uint64_t n = t; n < total; n += FLAGS_insert_threads) {
                const uint32_t owner = n % FLAGS_owners;
                const uint64_t i = n / FLAGS_owners;
                peer.InsertDiskOnlyObjectForTesting(
                    Key(owner, i), FLAGS_object_size, pop.owners[owner],
                    "disk-" + std::to_string(owner));
            }
        });
    }
    for (auto& thread : threads) thread.join();
    CHECK_EQ(service.GetKeyCount(), total);
    return pop;
}

// Readers of the other stores' objects; latencies in microseconds.
class Readers {
   public:
    Readers(MasterService& service, const Population& pop)
        : service_(service), pop_(pop) {}

    void Start() {
        for (uint32_t t = 0; t < FLAGS_readers; ++t) {
            threads_.emplace_back([this, t] {
                std::mt19937_64 rng(t + 1);
                std::uniform_int_distribution<uint32_t> owner(1,
                                                              FLAGS_owners - 1);
                std::uniform_int_distribution<uint64_t> index(
                    0, pop_.per_owner - 1);
                std::vector<uint32_t> local;
                while (!stop_.load(std::memory_order_relaxed)) {
                    const std::string key = Key(owner(rng), index(rng));
                    const auto start = Clock::now();
                    auto r = service_.GetReplicaList(key, TenantId::Default());
                    const auto us =
                        std::chrono::duration_cast<std::chrono::microseconds>(
                            Clock::now() - start)
                            .count();
                    CHECK(r.has_value());
                    if (measuring_.load(std::memory_order_relaxed)) {
                        local.push_back(static_cast<uint32_t>(us));
                    }
                }
                std::lock_guard<std::mutex> lock(mutex_);
                samples_.insert(samples_.end(), local.begin(), local.end());
            });
        }
    }
    void Measure(bool on) { measuring_.store(on); }
    void Stop() {
        stop_.store(true);
        for (auto& thread : threads_) thread.join();
        threads_.clear();
        std::sort(samples_.begin(), samples_.end());
    }
    uint64_t Count() const { return samples_.size(); }
    uint32_t Quantile(double q) const {
        if (samples_.empty()) return 0;
        return samples_[std::min(samples_.size() - 1,
                                 static_cast<size_t>(q * samples_.size()))];
    }
    uint32_t Max() const { return samples_.empty() ? 0 : samples_.back(); }

   private:
    MasterService& service_;
    const Population& pop_;
    std::atomic<bool> stop_{false};
    std::atomic<bool> measuring_{false};
    std::vector<std::thread> threads_;
    std::mutex mutex_;
    std::vector<uint32_t> samples_;
};

MasterServiceConfig Config(bool async) {
    MasterServiceConfig config;
    config.enable_offload = true;
    config.eviction_high_watermark_ratio = 1.0;
    config.async_disk_unmount_sweep = async;
    // No store heartbeats here: keep the clients alive through long setups.
    config.client_active_ttl_sec = 3600;
    config.client_suspicion_ttl_sec = 3600;
    return config;
}

void RunOne(const std::string& mode, uint64_t num_keys) {
    const bool async = mode == "async";
    MasterService service(Config(async));
    MasterServiceTestPeer peer(service);
    CHECK_EQ(peer.AsyncDiskUnmountSweepEnabledForTesting(), async);
    const auto build_start = Clock::now();
    const Population pop = Build(service, num_keys);
    const double build_ms = Ms(Clock::now() - build_start);
    const double rss_gib = RssGiB();
    const UUID leaving = pop.owners[0];
    const size_t keys_before = service.GetKeyCount();

    if (mode == "stats") {
        CHECK(MasterServiceTestPeer::LocalSsdManager(service)
                  .UnregisterClient(leaving)
                  .has_value());
        const auto start = Clock::now();
        const auto stats = peer.SweepLocalDiskOwnerWithStatsForTesting(leaving);
        const double sweep_ms = Ms(Clock::now() - start);
        CHECK_EQ(stats.removed_replicas, pop.per_owner);
        std::cout << "mode=stats,num_keys=" << keys_before
                  << ",removed=" << stats.removed_replicas
                  << ",sweep_ms=" << sweep_ms
                  << ",scan_hold_total_ms=" << stats.scan_hold_ns / 1e6
                  << ",scan_hold_max_shard_ms=" << stats.max_scan_hold_ns / 1e6
                  << ",erase_hold_total_ms=" << stats.batch_hold_ns / 1e6
                  << ",erase_hold_max_batch_us="
                  << stats.max_batch_hold_ns / 1e3
                  << ",ns_per_key=" << (sweep_ms * 1e6) / keys_before
                  << ",build_ms=" << build_ms << ",rss_gib=" << rss_gib
                  << std::endl;
        return;
    }

    Readers readers(service, pop);
    readers.Start();
    // Readers' baseline, then the window from the unmount to the sweep end.
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    readers.Measure(true);
    const auto start = Clock::now();
    CHECK(service.UnmountLocalDiskSegment(leaving).has_value());
    const double call_ms = Ms(Clock::now() - start);
    if (async) {
        CHECK(peer.WaitForDiskUnmountSweepsForTesting(std::chrono::hours(1)));
    }
    const double sweep_ms = Ms(Clock::now() - start);
    readers.Measure(false);
    readers.Stop();
    CHECK_EQ(service.GetKeyCount(), keys_before - pop.per_owner);
    std::cout << "mode=" << mode << ",num_keys=" << keys_before
              << ",removed=" << pop.per_owner << ",unmount_call_ms=" << call_ms
              << ",sweep_ms=" << sweep_ms
              << ",ns_per_key=" << (sweep_ms * 1e6) / keys_before
              << ",reads=" << readers.Count()
              << ",read_p50_us=" << readers.Quantile(0.50)
              << ",read_p99_us=" << readers.Quantile(0.99)
              << ",read_p999_us=" << readers.Quantile(0.999)
              << ",read_max_us=" << readers.Max() << ",build_ms=" << build_ms
              << ",rss_gib=" << rss_gib << std::endl;
}

}  // namespace
}  // namespace mooncake::benchmarks

int main(int argc, char** argv) {
    gflags::ParseCommandLineFlags(&argc, &argv, true);
    google::InitGoogleLogging(argv[0]);
    FLAGS_minloglevel = 1;  // keep the unmount INFO lines out of the CSV
    for (const auto& n : mooncake::benchmarks::Split(FLAGS_num_keys)) {
        for (const auto& mode : mooncake::benchmarks::Split(FLAGS_modes)) {
            mooncake::benchmarks::RunOne(mode, std::stoull(n));
        }
    }
    return 0;
}
