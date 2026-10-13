#pragma once

// Shutdown and startup sequences of the handoff (INF-600, fork-only), the
// deadline watchdog and a few metrics. The steps that touch the RPC server
// and MasterService are injected, so tests can check ordering and outcomes
// without a real server.

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <mutex>
#include <string>
#include <thread>

#include <ylt/util/tl/expected.hpp>

#include "handoff/handoff_admission.h"
#include "handoff/handoff_lifecycle.h"

namespace mooncake::handoff {

// Runs `on_fire` on its own thread once `deadline` has passed, unless
// Cancel() was called first. In the master, on_fire records the outcome and
// calls _exit(), so a slow or stuck snapshot never outlives the grace period.
class Watchdog {
   public:
    using Action = std::function<void()>;
    Watchdog(std::chrono::milliseconds deadline, Action on_fire);
    ~Watchdog();
    Watchdog(const Watchdog&) = delete;
    Watchdog& operator=(const Watchdog&) = delete;

    void Cancel();
    bool fired() const { return fired_.load(); }

   private:
    std::mutex mutex_;
    std::condition_variable cv_;
    bool cancelled_{false};
    std::atomic<bool> fired_{false};
    std::thread thread_;
};

// ---------------------------------------------------------------- shutdown

struct ShutdownWriteResult {
    uint64_t objects{0};
    uint64_t bytes{0};
    uint64_t disk_replicas{0};
    std::string json;  // writer stats
};

struct ShutdownSteps {
    // Stops the RPC and admin servers and returns only after every request
    // handler has returned.
    std::function<void()> stop_serving;
    // Stops and joins every background thread that mutates metadata.
    std::function<void()> quiesce;
    // malloc_trim(0); may be empty. Only called when the first admission
    // check gives fewer workers than wanted (or none).
    std::function<void()> trim;
    std::function<tl::expected<MemorySnapshot, std::string>()> read_memory;
    std::function<uint64_t()> object_count;
    // Objects with a LOCAL_DISK replica; may be empty (0).
    std::function<uint64_t()> local_disk_count;
    std::function<uint64_t(const std::string&)> disk_free;
    // Writes a complete, fsynced snapshot into `dir` (created by the step).
    std::function<tl::expected<ShutdownWriteResult, std::string>(
        const std::string& dir, size_t workers)>
        write;
    // Called right before the READY marker is published; tests use it to
    // inject races (e.g. a POISON written by another master).
    std::function<void()> before_publish;
};

struct ShutdownOptions {
    std::string root;
    bool owner{false};  // holds the HandoffDirLock
    size_t max_workers{8};
    size_t cpus{1};
    uint64_t restore_bytes_per_object{1280};
    uint64_t restore_bytes_per_disk_replica{256};
};

struct ShutdownReport {
    // ok | skipped_not_owner | skipped_poisoned | skipped_memory_headroom |
    // skipped_disk_space | skipped_restore_memory | error | deadline
    std::string result;
    std::string detail;
    std::string snapshot;
    uint64_t objects{0};
    uint64_t local_disk_objects{0};
    uint64_t bytes{0};
    size_t workers{0};
    uint64_t stop_ns{0};
    uint64_t quiesce_ns{0};
    uint64_t trim_ns{0};
    uint64_t write_ns{0};
    uint64_t publish_ns{0};
    uint64_t total_ns{0};
    int64_t finished_at_ms{0};
    std::string admission;  // ShutdownAdmission JSON
    std::string memory;     // MemorySnapshot JSON
    std::string writer;     // writer JSON
    std::string ToJson() const;
};

// Order: stop_serving -> quiesce -> ownership / poison checks -> reset the
// directory -> admission (trim and re-check if short) -> write into
// snap-<id>.tmp -> poison re-check -> publish (rename + READY). Writes
// last_shutdown at the end.
// Never throws; the caller exits the process afterwards.
ShutdownReport RunHandoffShutdown(const ShutdownSteps& steps,
                                  const ShutdownOptions& options,
                                  std::atomic<bool>* published = nullptr);

// ----------------------------------------------------------------- startup

struct RestoreSummary {
    uint64_t objects{0};
    uint64_t segments{0};
    uint64_t local_disk_replicas{0};
    std::string json;  // restore stats
};

struct StartupSteps {
    std::function<tl::expected<MemorySnapshot, std::string>()> read_memory;
    // Restores the snapshot in `dir` into the (empty) master. On error the
    // master must be empty again (cold start).
    std::function<tl::expected<RestoreSummary, std::string>(
        const std::string& dir, size_t workers)>
        restore;
};

struct StartupOptions {
    std::string root;
    bool owner{false};
    bool restore_enabled{true};
    int64_t max_age_ms{600000};
    uint64_t restore_bytes_per_object{1280};
    uint64_t restore_bytes_per_disk_replica{256};
    size_t workers{8};
    int64_t now_ms{0};  // 0: system clock
};

struct StartupReport {
    // ok | none | disabled | not_owner | aborted_previous | poisoned |
    // invalid | stale | memory_headroom | error
    std::string result;
    std::string detail;
    std::string snapshot;
    uint64_t objects{0};
    uint64_t segments{0};
    uint64_t local_disk_replicas{0};
    uint64_t age_ms{0};
    uint64_t claim_ns{0};
    uint64_t restore_ns{0};
    uint64_t total_ns{0};
    std::string restore;  // restore stats JSON
    std::string ToJson() const;
};

// Claims READY (before anything is allocated), checks age and memory, runs
// the restore once and always releases the claim afterwards.
StartupReport RunHandoffStartup(const StartupSteps& steps,
                                const StartupOptions& options);

// ----------------------------------------------------------------- metrics

// Small process-wide holder exported on the admin /metrics endpoint.
class HandoffMetrics {
   public:
    static HandoffMetrics& Instance();
    void SetStartup(const StartupReport& report);
    void SetLastShutdownJson(const std::string& json);
    // Live counters of restored segments (pending / confirmed / dropped).
    void SetSegmentCounters(
        std::function<void(uint64_t*, uint64_t*, uint64_t*)> provider);
    // Restored L4 owners not yet seen, and pinned offload sources.
    void SetLocalDiskCounters(
        std::function<void(uint64_t*, uint64_t*)> provider);
    std::string Serialize() const;

   private:
    mutable std::mutex mutex_;
    bool has_startup_{false};
    StartupReport startup_;
    std::string last_shutdown_json_;
    std::function<void(uint64_t*, uint64_t*, uint64_t*)> segment_counters_;
    std::function<void(uint64_t*, uint64_t*)> local_disk_counters_;
};

}  // namespace mooncake::handoff
