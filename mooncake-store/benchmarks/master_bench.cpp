// Copyright 2025 Alibaba Cloud and its affiliates
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <netdb.h>
#include <sys/socket.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <latch>
#include <mutex>
#include <random>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_set>
#include <vector>

#include "gflags/gflags.h"
#include "glog/logging.h"

#include "master_client.h"

// Size units for better readability
static constexpr size_t KiB = 1024;
static constexpr size_t MiB = 1024 * KiB;
static constexpr size_t GiB = 1024 * MiB;
static constexpr uintptr_t kSegmentBase = 0x100000000ULL;

DEFINE_string(master_server, "127.0.0.1:50051", "Master server address");
DEFINE_uint64(num_segments, 128, "Number of segments to mount");
DEFINE_uint64(segment_size, 64 * GiB, "Size of each segment");
DEFINE_uint64(num_clients, 4, "Number of clients to perform operations");
DEFINE_uint64(num_threads, 1,
              "Number of threads in each client to perform operations");
DEFINE_string(operation, "BatchPut",
              "Operation to perform: Put, Get, BatchPut, BatchGet or "
              "BatchExistKey. BatchExistKey is what a hicache fleet actually "
              "drives at a master -- it is the lookup a prefix-cache hit test "
              "makes, and it dominates a serving cell's master traffic");
DEFINE_uint64(num_keys, 10 * 1000,
              "Number of keys to prefill for Get operations on each thread");
DEFINE_uint64(batch_size, 128, "Batch size for batch operations");
DEFINE_uint64(value_size, 4096, "Size of object values");
DEFINE_uint64(duration, 60, "Test duration in seconds");
DEFINE_double(
    prefill_ratio, 0.0,
    "Ratio of segment capacity to prefill before test (0.0-1.0). "
    "E.g., 0.95 means fill segments to 95% before starting benchmark");
DEFINE_bool(independent_ping, false,
            "Give every client its own ping thread instead of pinging all of "
            "them in turn from one. A store pod pings on its own timer, so "
            "one shared pinger cannot distinguish a master that stopped "
            "answering from a pinger that is blocked on the first client in "
            "its list. Off by default, which is the previous behaviour");
DEFINE_uint64(wait_register_sec, 0,
              "After the opening mounts, wait up to this many seconds for "
              "every client to reach the master's OK state before starting "
              "load. A client only becomes OK by remounting, and a remount of "
              "a segment that already holds replicas fails, so a run that "
              "needs the master to have clients it can expire has to register "
              "them while the master is still empty. 0 keeps the previous "
              "behaviour: start load immediately and leave the clients "
              "unregistered");
DEFINE_int64(mount_extra_at_sec, -1,
             "Second of the load phase at which one further client joins and "
             "mounts one more segment. Negative disables it. This is the "
             "control-plane event a loaded master has to absorb: the joining "
             "client is new, so the mount arrives while the pre-existing "
             "clients are pinging and the bench clients are driving load");
DEFINE_uint64(extra_segment_size, 0,
              "Size of the segment --mount_extra_at_sec mounts. 0 means the "
              "same size as every other segment (--segment_size)");
DEFINE_string(metrics_server, "",
              "host:port of the master's --metrics_port endpoint. When set, "
              "it is scraped once per second into --metrics_csv");
DEFINE_string(metrics_path, "/metrics", "HTTP path of the metrics endpoint");
DEFINE_string(metrics_csv, "",
              "Where the once-per-second scrape is written, as "
              "elapsed_ms,epoch_ms,metric,value. Empty disables the sampler "
              "even when --metrics_server is set");
DEFINE_string(metrics_names, "",
              "Comma-separated metric names to keep from each scrape. Empty "
              "keeps every metric the endpoint exports, which is the safe "
              "default: a name guessed wrong is a column of silence");
DEFINE_string(ping_csv, "",
              "Where each client's own Ping outcome is written, as "
              "elapsed_ms,epoch_ms,segment_name,latency_us,outcome. This is "
              "the client-side view of the same window the master-side "
              "sampler records");

static inline void unset_cpu_affinity() {
    // Ensure that the worker threads are not bound to any CPU cores.
    cpu_set_t cpuset;
    memset(&cpuset, -1, sizeof(cpuset));
    pthread_setaffinity_np(pthread_self(), sizeof(cpuset), &cpuset);
}

// One origin for every timestamp this benchmark writes, so the master-side
// scrape, the client-side ping log and the master's own glog lines can be laid
// on the same axis afterwards. epoch_ms is what joins them to the master log;
// elapsed_ms is what makes a run readable on its own.
static const std::chrono::steady_clock::time_point gBenchStart =
    std::chrono::steady_clock::now();

static inline int64_t ElapsedMs() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
               std::chrono::steady_clock::now() - gBenchStart)
        .count();
}

static inline int64_t EpochMs() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
               std::chrono::system_clock::now().time_since_epoch())
        .count();
}

// A blocking HTTP/1.0 GET over a plain socket. The master's metrics endpoint
// is the only thing this benchmark fetches, and it answers a bare GET with a
// text body, so nothing here needs an HTTP library the benchmark does not
// already link.
static std::string HttpGet(const std::string& host, const std::string& port,
                           const std::string& path) {
    struct addrinfo hints {};
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;

    struct addrinfo* result = nullptr;
    if (getaddrinfo(host.c_str(), port.c_str(), &hints, &result) != 0) {
        return {};
    }

    int fd = -1;
    for (auto* ai = result; ai != nullptr; ai = ai->ai_next) {
        fd = socket(ai->ai_family, ai->ai_socktype, ai->ai_protocol);
        if (fd < 0) {
            continue;
        }
        if (connect(fd, ai->ai_addr, ai->ai_addrlen) == 0) {
            break;
        }
        close(fd);
        fd = -1;
    }
    freeaddrinfo(result);
    if (fd < 0) {
        return {};
    }

    const std::string request = "GET " + path +
                                " HTTP/1.0\r\nHost: " + host +
                                "\r\nConnection: close\r\n\r\n";
    size_t sent = 0;
    while (sent < request.size()) {
        auto n = send(fd, request.data() + sent, request.size() - sent, 0);
        if (n <= 0) {
            close(fd);
            return {};
        }
        sent += static_cast<size_t>(n);
    }

    std::string response;
    char buffer[16 * KiB];
    while (true) {
        auto n = recv(fd, buffer, sizeof(buffer), 0);
        if (n <= 0) {
            break;
        }
        response.append(buffer, static_cast<size_t>(n));
    }
    close(fd);

    auto body = response.find("\r\n\r\n");
    if (body == std::string::npos) {
        return {};
    }
    return response.substr(body + 4);
}

// Scrapes the master's own metrics once per second into a long-format CSV.
//
// The names are not hardcoded: --metrics_names defaults to empty, which keeps
// every metric the endpoint exports. A previous session lost a window to a
// guessed metric name that the master does not export, which reads exactly
// like a metric that never moved, so the default here is to keep everything
// and select afterwards.
class MetricsSampler {
   public:
    MetricsSampler(const std::string& server, const std::string& path,
                   const std::string& csv_path, const std::string& names)
        : path_(path) {
        auto colon = server.rfind(':');
        if (colon == std::string::npos) {
            throw std::invalid_argument(
                "--metrics_server must be host:port, got " + server);
        }
        host_ = server.substr(0, colon);
        port_ = server.substr(colon + 1);

        size_t begin = 0;
        while (begin < names.size()) {
            auto end = names.find(',', begin);
            if (end == std::string::npos) {
                end = names.size();
            }
            if (end > begin) {
                wanted_.insert(names.substr(begin, end - begin));
            }
            begin = end + 1;
        }

        out_.open(csv_path, std::ios::out | std::ios::trunc);
        if (!out_) {
            throw std::runtime_error("Cannot open --metrics_csv " + csv_path);
        }
        out_ << "elapsed_ms,epoch_ms,metric,value\n";

        thread_ = std::jthread([this](std::stop_token stop_token) {
            unset_cpu_affinity();
            static const auto OneSecond = std::chrono::seconds(1);
            while (!stop_token.stop_requested()) {
                auto start_time = std::chrono::steady_clock::now();
                Sample();
                auto time_elapsed =
                    std::chrono::steady_clock::now() - start_time;
                if (OneSecond > time_elapsed) {
                    std::this_thread::sleep_for(OneSecond - time_elapsed);
                }
            }
        });
    }

   private:
    void Sample() {
        // The scrape's own timestamps: a scrape that takes seconds because
        // the master is stalled is itself a measurement, so record when it
        // was asked for and how long the answer took.
        const int64_t elapsed_ms = ElapsedMs();
        const int64_t epoch_ms = EpochMs();
        const auto begin = std::chrono::steady_clock::now();
        const std::string body = HttpGet(host_, port_, path_);
        const int64_t scrape_us =
            std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::steady_clock::now() - begin)
                .count();

        Write(elapsed_ms, epoch_ms, "bench_scrape_latency_us",
              std::to_string(scrape_us));
        Write(elapsed_ms, epoch_ms, "bench_scrape_bytes",
              std::to_string(body.size()));
        if (body.empty()) {
            out_.flush();
            return;
        }

        size_t line_begin = 0;
        while (line_begin < body.size()) {
            auto line_end = body.find('\n', line_begin);
            if (line_end == std::string::npos) {
                line_end = body.size();
            }
            std::string_view line(body.data() + line_begin,
                                  line_end - line_begin);
            line_begin = line_end + 1;

            if (line.empty() || line.front() == '#') {
                continue;
            }
            auto space = line.rfind(' ');
            if (space == std::string_view::npos) {
                continue;
            }
            std::string_view series = line.substr(0, space);
            std::string_view value = line.substr(space + 1);

            // Keep the labels in the series name -- per-segment gauges are
            // the ones that say which segment went away.
            auto brace = series.find('{');
            std::string name(
                series.substr(0, brace == std::string_view::npos
                                     ? series.size()
                                     : brace));
            if (!wanted_.empty() && !wanted_.contains(name)) {
                continue;
            }
            Write(elapsed_ms, epoch_ms, std::string(series),
                  std::string(value));
        }
        out_.flush();
    }

    void Write(int64_t elapsed_ms, int64_t epoch_ms, const std::string& metric,
               const std::string& value) {
        out_ << elapsed_ms << ',' << epoch_ms << ",\"" << metric << "\","
             << value << '\n';
    }

    std::string host_;
    std::string port_;
    std::string path_;
    std::unordered_set<std::string> wanted_;
    std::ofstream out_;
    std::jthread thread_;
};

// The client-side half of the same window: what each client's Ping actually
// returned, and how long it took. The master-side sampler cannot show a Ping
// that never reached the master, and that is the observation this whole
// benchmark exists to make.
class PingLog {
   public:
    explicit PingLog(const std::string& csv_path) {
        out_.open(csv_path, std::ios::out | std::ios::trunc);
        if (!out_) {
            throw std::runtime_error("Cannot open --ping_csv " + csv_path);
        }
        out_ << "elapsed_ms,epoch_ms,segment_name,latency_us,outcome\n";
    }

    void Append(int64_t elapsed_ms, int64_t epoch_ms,
                const std::string& segment_name, int64_t latency_us,
                const std::string& outcome) {
        std::lock_guard<std::mutex> guard(mutex_);
        out_ << elapsed_ms << ',' << epoch_ms << ",\"" << segment_name
             << "\"," << latency_us << ',' << outcome << '\n';
        out_.flush();
    }

   private:
    std::mutex mutex_;
    std::ofstream out_;
};

static std::unique_ptr<PingLog> gPingLog;

class SegmentClient {
   public:
    SegmentClient(const std::string& name, const std::string& master_server,
                  uintptr_t segment_base, uint64_t segment_size)
        : master_client_(mooncake::generate_uuid()) {
        auto ec = master_client_.Connect(master_server);
        if (ec != mooncake::ErrorCode::OK) {
            throw std::invalid_argument("Cannot connect to master server at " +
                                        master_server + ", ec=" + toString(ec));
        }

        segment_.id = mooncake::generate_uuid();
        segment_.name = name;
        segment_.base = segment_base;
        segment_.size = segment_size;
        segment_.te_endpoint = name;
        const auto mount_begin = std::chrono::steady_clock::now();
        auto mount_ec = master_client_.MountSegment(segment_);
        mount_latency_us_ =
            std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::steady_clock::now() - mount_begin)
                .count();
        if (!mount_ec.has_value()) {
            throw std::runtime_error("Failed to mount segment " + name +
                                     ", ec=" + toString(mount_ec.error()));
        }
    }

    const std::string& name() const { return segment_.name; }

    int64_t mount_latency_us() const { return mount_latency_us_; }

    // True once a Ping has come back OK, which is the master saying this
    // client is in ok_client_ and therefore something it can expire.
    bool registered() const { return registered_.load(); }

    ~SegmentClient() {
        if (remount_future_.valid()) {
            remount_future_.wait();
        }

        auto unmount_result = master_client_.UnmountSegment(segment_.id);
        if (!unmount_result.has_value()) {
            LOG(ERROR) << "Failed to unmount segment " << segment_.name;
        }
    }

    void Ping() {
        if (remount_future_.valid() &&
            remount_future_.wait_for(std::chrono::seconds(0)) ==
                std::future_status::ready) {
            remount_future_.get();
            remount_future_ = std::future<void>();
        }

        const int64_t elapsed_ms = ElapsedMs();
        const int64_t epoch_ms = EpochMs();
        const auto ping_begin = std::chrono::steady_clock::now();
        auto ping_result = master_client_.Ping();
        const int64_t latency_us =
            std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::steady_clock::now() - ping_begin)
                .count();

        // A failed Ping is recorded and returned from, not thrown on. Losing
        // the client that is meant to be observing the master is the one
        // outcome that would destroy the observation: after an expiry the
        // master answers again, and what the run needs to show is the
        // recovery.
        if (!ping_result.has_value()) {
            RecordPing(elapsed_ms, epoch_ms, latency_us, "error");
            LOG(ERROR) << "segment_name=" << segment_.name
                       << ", action=ping_failed, ec="
                       << toString(ping_result.error());
            return;
        }

        const bool need_remount = ping_result.value().client_status ==
                                  mooncake::ClientStatus::NEED_REMOUNT;
        registered_.store(!need_remount);
        RecordPing(elapsed_ms, epoch_ms, latency_us,
                   need_remount ? "need_remount" : "ok");

        if (need_remount && !remount_future_.valid()) {
            remount_future_ = std::async(std::launch::async, [&]() {
                const auto remount_begin = std::chrono::steady_clock::now();
                auto remount_ec = master_client_.ReMountSegment({segment_});
                const int64_t remount_us =
                    std::chrono::duration_cast<std::chrono::microseconds>(
                        std::chrono::steady_clock::now() - remount_begin)
                        .count();
                LOG(INFO) << "segment_name=" << segment_.name
                          << ", action=remount_segment, latency_us="
                          << remount_us << ", ok=" << remount_ec.has_value();
                if (gPingLog) {
                    gPingLog->Append(ElapsedMs(), EpochMs(), segment_.name,
                                     remount_us,
                                     remount_ec.has_value()
                                         ? "remount_ok"
                                         : "remount_failed");
                }
            });
        }
    }

    // Ping this client on its own timer, the way a store pod does. Stopped by
    // the jthread's destructor, which runs before the members Ping() touches.
    void StartOwnPingThread() {
        own_ping_thread_ = std::jthread([this](std::stop_token stop_token) {
            unset_cpu_affinity();
            static const auto OneSecond = std::chrono::seconds(1);
            while (!stop_token.stop_requested()) {
                auto start_time = std::chrono::steady_clock::now();
                Ping();
                auto time_elapsed =
                    std::chrono::steady_clock::now() - start_time;
                if (OneSecond > time_elapsed) {
                    std::this_thread::sleep_for(OneSecond - time_elapsed);
                }
            }
        });
    }

   private:
    void RecordPing(int64_t elapsed_ms, int64_t epoch_ms, int64_t latency_us,
                    const char* outcome) {
        if (gPingLog) {
            gPingLog->Append(elapsed_ms, epoch_ms, segment_.name, latency_us,
                             outcome);
        }
    }

    mooncake::MasterClient master_client_;
    mooncake::Segment segment_;
    std::future<void> remount_future_;
    int64_t mount_latency_us_ = 0;
    std::atomic<bool> registered_{false};
    // Declared last so its destructor stops the thread before anything the
    // thread's Ping() reads is destroyed.
    std::jthread own_ping_thread_;
};

static std::atomic<uint64_t> gCompletedOperations = 0;

enum class BenchOperation {
    PUT,
    GET,
    BATCH_PUT,
    BATCH_GET,
    BATCH_EXIST_KEY,
};

static inline BenchOperation ParseOperation(const std::string& operation_str) {
    if (operation_str == "Put") {
        return BenchOperation::PUT;
    } else if (operation_str == "Get") {
        return BenchOperation::GET;
    } else if (operation_str == "BatchPut") {
        return BenchOperation::BATCH_PUT;
    } else if (operation_str == "BatchGet") {
        return BenchOperation::BATCH_GET;
    } else if (operation_str == "BatchExistKey") {
        return BenchOperation::BATCH_EXIST_KEY;
    } else {
        throw std::invalid_argument("Invalid operation");
    }
}

class BenchClient {
   public:
    BenchClient(const std::string& master_server)
        : master_client_(mooncake::generate_uuid()), running_(false) {
        auto ec = master_client_.Connect(master_server);
        if (ec != mooncake::ErrorCode::OK) {
            throw std::invalid_argument("Cannot connect to master server at " +
                                        master_server + ", ec=" + toString(ec));
        }
    }

    void StartBench(std::shared_ptr<std::latch> barrier,
                    BenchOperation operation, uint64_t num_threads,
                    uint64_t batch_size, uint64_t value_size,
                    uint64_t num_keys) {
        if (running_.load()) {
            return;
        }

        running_.store(true);
        for (size_t i = 0; i < num_threads; i++) {
            threads_.push_back(std::thread(
                std::bind(&BenchClient::BenchFn, this, barrier, operation,
                          batch_size, value_size, num_keys)));
        }
    }

    void StopBench() {
        if (!running_.load()) {
            return;
        }

        running_.store(false);
        for (auto& thread : threads_) {
            if (thread.joinable()) {
                thread.join();
            }
        }

        threads_.clear();
    }

   private:
    bool Put(const std::string& key, const std::vector<uint64_t>& slice_lengths,
             const mooncake::ReplicateConfig& config) {
        auto put_start_result =
            master_client_.PutStart(key, slice_lengths, config);
        if (!put_start_result.has_value()) {
            return false;
        }

        auto put_end_result = master_client_.PutEnd(
            {key, std::nullopt}, mooncake::ReplicaType::MEMORY);
        if (!put_end_result.has_value()) {
            return false;
        }

        return true;
    }

    bool Get(const std::string& key) {
        auto get_result = master_client_.GetReplicaList(key);
        return get_result.has_value() ||
               get_result.error() == mooncake::ErrorCode::OBJECT_NOT_FOUND;
    }

    uint64_t BatchPut(const std::vector<std::string>& keys,
                      const std::vector<std::vector<uint64_t>>& slice_lengths,
                      const mooncake::ReplicateConfig& config) {
        std::vector<std::string> started_keys;
        uint64_t success_cnt = 0;

        auto put_start_result =
            master_client_.BatchPutStart(keys, slice_lengths, config);
        for (size_t i = 0; i < keys.size(); i++) {
            if (put_start_result[i].has_value()) {
                started_keys.push_back(keys[i]);
            }
        }

        if (started_keys.empty()) {
            return 0;
        }

        std::vector<mooncake::ObjectMeta> object_metas;
        object_metas.reserve(started_keys.size());
        for (const auto& key : started_keys) {
            object_metas.emplace_back(mooncake::ObjectMeta{key, std::nullopt});
        }
        auto put_end_result = master_client_.BatchPutEnd(object_metas);
        for (auto& result : put_end_result) {
            if (result.has_value()) {
                success_cnt++;
            }
        }

        return success_cnt;
    }

    uint64_t BatchExistKey(const std::vector<std::string>& keys) {
        uint64_t success_cnt = 0;
        auto exist_results = master_client_.BatchExistKey(keys);
        for (auto& exist_result : exist_results) {
            if (exist_result.has_value()) {
                success_cnt++;
            }
        }
        return success_cnt;
    }

    uint64_t BatchGet(const std::vector<std::string>& keys) {
        uint64_t success_cnt = 0;
        auto get_results = master_client_.BatchGetReplicaList(keys);
        for (auto& get_result : get_results) {
            if (get_result.has_value() ||
                get_result.error() == mooncake::ErrorCode::OBJECT_NOT_FOUND) {
                success_cnt++;
            }
        }
        return success_cnt;
    }

    std::vector<std::string> GeneratePutKeys(uint64_t& key_id,
                                             uint64_t num_keys) {
        const auto self_id = std::this_thread::get_id();
        std::vector<std::string> keys;
        keys.reserve(num_keys);

        for (size_t i = 0; i < num_keys; i++, key_id++) {
            std::stringstream ss;
            ss << self_id << "_" << key_id;
            keys.push_back(ss.str());
        }

        return keys;
    }

    std::vector<std::string> GenerateGetKeys(uint64_t key_id,
                                             uint64_t num_keys) {
        static thread_local std::mt19937 generator(std::random_device{}());

        const auto self_id = std::this_thread::get_id();
        std::vector<std::string> keys;
        keys.reserve(num_keys);

        if (key_id == 0) {
            // No keys have been created yet; return empty vector.
            return keys;
        }

        std::uniform_int_distribution<uint64_t> distribution(0, key_id - 1);
        for (size_t i = 0; i < num_keys; i++) {
            std::stringstream ss;
            ss << self_id << "_" << distribution(generator);
            keys.push_back(ss.str());
        }

        return keys;
    }

    void PrefillKeys(uint64_t& key_id, uint64_t batch_size, uint64_t value_size,
                     uint64_t num_keys) {
        const mooncake::ReplicateConfig config;

        uint64_t num_to_prefill = std::min(batch_size, num_keys);
        while (num_to_prefill > 0) {
            auto keys = GeneratePutKeys(key_id, num_to_prefill);
            std::vector<std::vector<uint64_t>> slice_lengths(num_to_prefill,
                                                             {value_size});
            BatchPut(keys, slice_lengths, config);
            num_keys -= num_to_prefill;
            num_to_prefill = std::min(batch_size, num_keys);
        }
    }

    void BenchFn(std::shared_ptr<std::latch> barrier, BenchOperation operation,
                 uint64_t batch_size, uint64_t value_size, uint64_t num_keys) {
        unset_cpu_affinity();

        uint64_t key_id = 0;
        const mooncake::ReplicateConfig config;

        std::vector<std::string> keys;
        std::vector<std::vector<uint64_t>> slice_lengths;
        slice_lengths.reserve(batch_size);
        for (size_t i = 0; i < batch_size; i++) {
            slice_lengths.push_back({value_size});
        }

        if (operation == BenchOperation::GET ||
            operation == BenchOperation::BATCH_GET ||
            operation == BenchOperation::BATCH_EXIST_KEY) {
            PrefillKeys(key_id, batch_size, value_size, num_keys);
        }

        barrier->arrive_and_wait();

        while (running_.load()) {
            switch (operation) {
                case BenchOperation::PUT:
                    keys = GeneratePutKeys(key_id, 1);
                    if (Put(keys[0], slice_lengths[0], config)) {
                        gCompletedOperations.fetch_add(1);
                    }
                    break;
                case BenchOperation::GET:
                    keys = GenerateGetKeys(key_id, 1);
                    if (Get(keys[0])) {
                        gCompletedOperations.fetch_add(1);
                    }
                    break;
                case BenchOperation::BATCH_PUT:
                    keys = GeneratePutKeys(key_id, batch_size);
                    gCompletedOperations.fetch_add(
                        BatchPut(keys, slice_lengths, config));
                    break;
                case BenchOperation::BATCH_GET:
                    keys = GenerateGetKeys(key_id, batch_size);
                    gCompletedOperations.fetch_add(BatchGet(keys));
                    break;
                case BenchOperation::BATCH_EXIST_KEY:
                    keys = GenerateGetKeys(key_id, batch_size);
                    gCompletedOperations.fetch_add(BatchExistKey(keys));
                    break;
                default:
                    break;
            }
        }
    }

    mooncake::MasterClient master_client_;

    std::atomic<bool> running_;
    std::vector<std::thread> threads_;
};

int main(int argc, char** argv) {
    std::vector<std::unique_ptr<SegmentClient>> segment_clients;
    std::mutex segment_clients_mutex;
    std::jthread ping_thread;
    std::vector<std::unique_ptr<BenchClient>> bench_clients;

    google::InitGoogleLogging("MasterBench");
    FLAGS_logtostderr = true;

    gflags::ParseCommandLineFlags(&argc, &argv, false);

    if (!FLAGS_ping_csv.empty()) {
        gPingLog = std::make_unique<PingLog>(FLAGS_ping_csv);
    }

    // Started before the first mount: the opening mounts are themselves
    // control-plane work on an empty master, and a baseline for them is what
    // makes the mid-run mount's cost readable.
    std::unique_ptr<MetricsSampler> metrics_sampler;
    if (!FLAGS_metrics_server.empty() && !FLAGS_metrics_csv.empty()) {
        metrics_sampler = std::make_unique<MetricsSampler>(
            FLAGS_metrics_server, FLAGS_metrics_path, FLAGS_metrics_csv,
            FLAGS_metrics_names);
        LOG(INFO) << "action=metrics_sampler_started server="
                  << FLAGS_metrics_server << " csv=" << FLAGS_metrics_csv;
    }

    ping_thread = std::jthread([&](std::stop_token stop_token) {
        static const auto OneSecond = std::chrono::seconds(1);

        unset_cpu_affinity();

        while (!stop_token.stop_requested()) {
            std::chrono::nanoseconds time_elapsed;
            auto start_time = std::chrono::steady_clock::now();
            if (!FLAGS_independent_ping) {
                std::lock_guard<std::mutex> guard(segment_clients_mutex);
                for (auto& segment_client : segment_clients) {
                    segment_client->Ping();
                }
            }
            time_elapsed = std::chrono::steady_clock::now() - start_time;

            if (OneSecond > time_elapsed) {
                std::this_thread::sleep_for(OneSecond - time_elapsed);
            }
        }
    });

    LOG(INFO) << "Mounting " << FLAGS_num_segments << " segments...";
    for (size_t i = 0; i < FLAGS_num_segments; i++) {
        auto segment_client = std::make_unique<SegmentClient>(
            "segment_client_" + std::to_string(i), FLAGS_master_server,
            kSegmentBase + i * FLAGS_segment_size, FLAGS_segment_size);
        if (FLAGS_independent_ping) {
            segment_client->StartOwnPingThread();
        }
        LOG(INFO) << "segment_name=" << segment_client->name()
                  << ", action=mount_segment, epoch_ms=" << EpochMs()
                  << ", latency_us=" << segment_client->mount_latency_us();
        {
            std::lock_guard<std::mutex> guard(segment_clients_mutex);
            segment_clients.push_back(std::move(segment_client));
        }
    }
    LOG(INFO) << "Segments mounted";

    if (FLAGS_wait_register_sec > 0) {
        const auto deadline = std::chrono::steady_clock::now() +
                              std::chrono::seconds(FLAGS_wait_register_sec);
        size_t registered = 0;
        while (std::chrono::steady_clock::now() < deadline) {
            registered = 0;
            {
                std::lock_guard<std::mutex> guard(segment_clients_mutex);
                for (auto& segment_client : segment_clients) {
                    if (segment_client->registered()) {
                        registered++;
                    }
                }
            }
            if (registered == segment_clients.size()) {
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
        }
        LOG(INFO) << "action=wait_register_done, registered=" << registered
                  << ", of=" << segment_clients.size()
                  << ", elapsed_ms=" << ElapsedMs();
        if (registered != segment_clients.size()) {
            LOG(ERROR) << "action=wait_register_incomplete: the master has "
                          "fewer clients than this run assumes, so an expiry "
                          "it triggers will not be the one being measured";
        }
    }

    // Prefill segments if requested
    if (FLAGS_prefill_ratio > 0.0) {
        LOG(INFO) << "Prefilling segments to " << (FLAGS_prefill_ratio * 100)
                  << "% capacity...";

        // Calculate total capacity and target fill size
        uint64_t total_capacity = FLAGS_num_segments * FLAGS_segment_size;
        uint64_t target_bytes =
            static_cast<uint64_t>(total_capacity * FLAGS_prefill_ratio);
        uint64_t bytes_per_object = FLAGS_value_size;
        uint64_t target_objects = target_bytes / bytes_per_object;

        LOG(INFO) << "Target: " << target_objects << " objects ("
                  << (target_bytes / GiB) << " GiB)";

        // Create a temporary client for prefilling
        mooncake::MasterClient prefill_client(mooncake::generate_uuid());
        auto ec = prefill_client.Connect(FLAGS_master_server);
        if (ec != mooncake::ErrorCode::OK) {
            LOG(ERROR) << "Failed to connect prefill client";
            return 1;
        }

        const mooncake::ReplicateConfig config;
        uint64_t filled_objects = 0;
        uint64_t key_id = 0;

        while (filled_objects < target_objects) {
            uint64_t batch =
                std::min(FLAGS_batch_size, target_objects - filled_objects);
            std::vector<std::string> keys;
            std::vector<std::vector<uint64_t>> slice_lengths;

            keys.reserve(batch);
            slice_lengths.reserve(batch);

            for (uint64_t i = 0; i < batch; i++) {
                keys.push_back("prefill_" + std::to_string(key_id++));
                slice_lengths.push_back({bytes_per_object});
            }

            auto put_start_result =
                prefill_client.BatchPutStart(keys, slice_lengths, config);
            std::vector<std::string> started_keys;
            for (size_t i = 0; i < keys.size(); i++) {
                if (put_start_result[i].has_value()) {
                    started_keys.push_back(keys[i]);
                }
            }

            if (!started_keys.empty()) {
                std::vector<mooncake::ObjectMeta> object_metas;
                object_metas.reserve(started_keys.size());
                for (const auto& key : started_keys) {
                    object_metas.emplace_back(
                        mooncake::ObjectMeta{key, std::nullopt});
                }
                auto put_end_result = prefill_client.BatchPutEnd(object_metas);
                for (auto& result : put_end_result) {
                    if (result.has_value()) {
                        filled_objects++;
                    }
                }
            }

            if (filled_objects % 10000 == 0) {
                double progress =
                    (double)filled_objects / target_objects * 100.0;
                LOG(INFO) << "Prefill progress: " << filled_objects << "/"
                          << target_objects << " (" << std::fixed
                          << std::setprecision(1) << progress << "%)";
            }
        }

        LOG(INFO) << "Prefill completed: " << filled_objects << " objects";
    }

    LOG(INFO) << "Starting " << FLAGS_num_clients << " bench clients with "
              << FLAGS_num_threads << " threads for each...";
    for (size_t i = 0; i < FLAGS_num_clients; i++) {
        bench_clients.emplace_back(
            std::make_unique<BenchClient>(FLAGS_master_server));
    }

    const auto operation = ParseOperation(FLAGS_operation);
    auto num_threads = FLAGS_num_clients * FLAGS_num_threads;
    auto barrier = std::make_shared<std::latch>(num_threads +
                                                1);  // +1 for the main thread

    for (auto& bench_client : bench_clients) {
        bench_client->StartBench(barrier, operation, FLAGS_num_threads,
                                 FLAGS_batch_size, FLAGS_value_size,
                                 FLAGS_num_keys);
    }
    barrier->arrive_and_wait();
    LOG(INFO) << "Clients started";

    const uint64_t extra_segment_size = FLAGS_extra_segment_size == 0
                                            ? FLAGS_segment_size
                                            : FLAGS_extra_segment_size;

    uint64_t last_completed = 0;
    for (size_t i = 0; i < FLAGS_duration; i++) {
        std::this_thread::sleep_for(std::chrono::seconds(1));
        auto curr_completed = gCompletedOperations.load();
        LOG(INFO) << "Completed operations: "
                  << curr_completed - last_completed
                  << ", elapsed_ms=" << ElapsedMs();
        last_completed = curr_completed;

        if (FLAGS_mount_extra_at_sec < 0 ||
            static_cast<int64_t>(i) != FLAGS_mount_extra_at_sec) {
            continue;
        }

        // The trigger. Constructed outside segment_clients_mutex on purpose:
        // the mount is the event under study, and holding the mutex the ping
        // thread walks would stop this benchmark's own pings for the mount's
        // duration -- an observer that produces the symptom it is measuring.
        LOG(INFO) << "action=extra_mount_begin, epoch_ms=" << EpochMs()
                  << ", elapsed_ms=" << ElapsedMs()
                  << ", segment_size=" << extra_segment_size;
        std::unique_ptr<SegmentClient> extra_client;
        try {
            extra_client = std::make_unique<SegmentClient>(
                "segment_client_extra", FLAGS_master_server,
                kSegmentBase + FLAGS_num_segments * FLAGS_segment_size,
                extra_segment_size);
        } catch (const std::exception& e) {
            LOG(ERROR) << "action=extra_mount_failed, error=" << e.what();
            continue;
        }
        if (FLAGS_independent_ping) {
            extra_client->StartOwnPingThread();
        }
        LOG(INFO) << "action=extra_mount_end, epoch_ms=" << EpochMs()
                  << ", elapsed_ms=" << ElapsedMs() << ", latency_us="
                  << extra_client->mount_latency_us();
        {
            std::lock_guard<std::mutex> guard(segment_clients_mutex);
            segment_clients.push_back(std::move(extra_client));
        }
    }

    auto num_completed_operations = gCompletedOperations.load();

    LOG(INFO) << "Stopping bench clients...";
    for (auto& bench_client : bench_clients) {
        bench_client->StopBench();
    }
    LOG(INFO) << "Clients stopped";

    LOG(INFO) << "Stopping ping thread...";
    if (ping_thread.joinable()) {
        ping_thread.request_stop();
        ping_thread.join();
    }
    LOG(INFO) << "Ping thread stopped";

    if (metrics_sampler) {
        LOG(INFO) << "Stopping metrics sampler...";
        metrics_sampler.reset();
    }

    LOG(INFO) << "Disconnecting from master...";
    bench_clients.clear();
    segment_clients.clear();
    LOG(INFO) << "Disconnected from master";
    gPingLog.reset();

    std::cout << "Operations per second: " << std::fixed << std::setprecision(2)
              << num_completed_operations / (double)FLAGS_duration << "\n";

    google::ShutdownGoogleLogging();

    return 0;
}
