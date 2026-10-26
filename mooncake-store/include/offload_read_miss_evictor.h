#pragma once

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_set>
#include <utility>
#include <vector>

#include <glog/logging.h>
#include <ylt/util/tl/expected.hpp>

#include "types.h"

namespace mooncake {

/**
 * @brief Drops, in the background, this store's LOCAL_DISK replicas of keys
 * whose peer offload read the read guard failed (OffloadReadGuard deadline
 * miss or shed).
 *
 * Why: a reader that gets such a miss recomputes the keys and puts them back,
 * but the put path first asks the master whether the key exists, and a key
 * that has only a disk replica exists. So the recomputed value is never
 * written to memory, the key stays on the slow disk, and every later read of
 * it misses again until the disk is fast. Dropping the master's pointer to
 * the disk replica makes the key absent, so the next put of it lands in
 * memory.
 *
 * Only master metadata changes, through the RPC the store already uses when
 * it evicts disk data (BatchEvictDiskReplica, LOCAL_DISK, this store's
 * replicas only). The bytes stay on the disk, like the file of a removed
 * object: the master registers a LOCAL_DISK replica only on completion of an
 * offload task, never from a file the store already has, so no new reader is
 * pointed at them (one that already holds the replica still reads them). With
 * the bucket backend they stay in its index until bucket eviction, and an
 * offload of the key to this store is skipped while they are there.
 *
 * The drop is asynchronous, so a reader that checks existence right after
 * its failed read may still see the key and skip its put; the next reader of
 * the key then sees it absent.
 *
 * Add() never blocks on the master: keys are queued (deduplicated, at most
 * `max_pending`; more are dropped and counted) and a thread hands them to
 * `evict` in batches of at most `max_batch`. A failed batch is logged and not
 * retried (the key then stays disk-only, the behavior without this class),
 * and the thread waits `kFailureBackoff` before the next batch.
 */
class OffloadReadMissEvictor {
   public:
    using EvictFn = std::function<tl::expected<void, ErrorCode>(
        const std::vector<std::string>&)>;

    OffloadReadMissEvictor(EvictFn evict, size_t max_pending, size_t max_batch)
        : evict_(std::move(evict)),
          max_pending_(max_pending),
          max_batch_(max_batch == 0 ? 1 : max_batch),
          thread_([this] { Loop(); }) {}

    OffloadReadMissEvictor(const OffloadReadMissEvictor&) = delete;
    OffloadReadMissEvictor& operator=(const OffloadReadMissEvictor&) = delete;

    // Keys still queued are dropped. A batch already handed to `evict` is
    // waited for.
    ~OffloadReadMissEvictor() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stopping_ = true;
        }
        cv_.notify_all();
        if (thread_.joinable()) thread_.join();
    }

    struct AddResult {
        size_t queued = 0;   // newly queued (a key already queued is not)
        size_t dropped = 0;  // refused because the queue was full
    };

    // Queues the keys of one failed read.
    AddResult Add(const std::vector<std::string>& keys) {
        size_t queued = 0;
        size_t dropped = 0;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (stopping_) return {};
            for (const auto& key : keys) {
                if (pending_.size() >= max_pending_ && !pending_.count(key)) {
                    ++dropped;
                    continue;
                }
                if (pending_.insert(key).second) ++queued;
            }
            queued_keys_ += queued;
            dropped_keys_ += dropped;
        }
        if (dropped > 0) {
            LOG_EVERY_N(WARNING, 1000)
                << "action=offload_read_miss_evict_dropped, keys=" << dropped
                << ", max_pending=" << max_pending_ << " (logged every 1000)";
        }
        if (queued > 0) cv_.notify_one();
        return AddResult{.queued = queued, .dropped = dropped};
    }

    struct Stats {
        uint64_t queued_keys = 0;   // keys accepted by Add()
        uint64_t dropped_keys = 0;  // keys refused because the queue was full
        uint64_t sent_keys = 0;     // keys handed to `evict`
        uint64_t failed_batches = 0;
        size_t pending = 0;
    };

    Stats stats() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return Stats{.queued_keys = queued_keys_,
                     .dropped_keys = dropped_keys_,
                     .sent_keys = sent_keys_,
                     .failed_batches = failed_batches_,
                     .pending = pending_.size()};
    }

   private:
    void Loop() {
        std::unique_lock<std::mutex> lock(mutex_);
        while (true) {
            cv_.wait(lock, [this] { return stopping_ || !pending_.empty(); });
            if (stopping_) return;
            std::vector<std::string> batch;
            batch.reserve(std::min(pending_.size(), max_batch_));
            for (auto it = pending_.begin();
                 it != pending_.end() && batch.size() < max_batch_;) {
                batch.push_back(std::move(pending_.extract(it++).value()));
            }
            lock.unlock();
            auto result = evict_(batch);
            if (!result) {
                LOG_EVERY_N(WARNING, 100)
                    << "action=offload_read_miss_evict_failed, keys="
                    << batch.size() << ", error=" << result.error()
                    << " (logged every 100)";
            }
            lock.lock();
            sent_keys_ += batch.size();
            if (!result) {
                ++failed_batches_;
                // A master outage fails every batch; do not spin through the
                // queue meanwhile.
                cv_.wait_for(lock, kFailureBackoff,
                             [this] { return stopping_; });
            }
        }
    }

    static constexpr std::chrono::milliseconds kFailureBackoff{1000};

    const EvictFn evict_;
    const size_t max_pending_;
    const size_t max_batch_;
    mutable std::mutex mutex_;
    std::condition_variable cv_;
    std::unordered_set<std::string> pending_;
    bool stopping_ = false;
    uint64_t queued_keys_ = 0;
    uint64_t dropped_keys_ = 0;
    uint64_t sent_keys_ = 0;
    uint64_t failed_batches_ = 0;
    // Last: started after every member above is ready.
    std::thread thread_;
};

}  // namespace mooncake
