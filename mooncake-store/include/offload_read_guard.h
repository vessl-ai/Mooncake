#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <exception>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <utility>

#include <glog/logging.h>
#include <ylt/util/tl/expected.hpp>

#include "thread_pool.h"
#include "types.h"

namespace mooncake {

/**
 * @brief Bounds how long an offload (SSD tier) read waits for the disk.
 *
 * Without it a reader waits for the slowest read of its batch: the store reads
 * the batch's keys one after another, so a disk that answers each read in
 * 20 ms makes a 60-key batch take over a second, and the reader's RPC has no
 * deadline below the 30 s transport timeout. The reader cannot use the memory
 * replicas of the same batch until then either.
 *
 * Run() hands the read to its own threads and waits at most `deadline`, time
 * spent waiting for a free thread included. A read that is not done by then is
 * answered with UNAVAILABLE_IN_CURRENT_STATUS (the reader treats those keys as
 * misses and recomputes them). If it had not started yet it never starts;
 * otherwise it finishes in the background and `release_late` frees what it
 * allocated. A late read is never answered, so no reader can see its buffer.
 *
 * After a deadline miss the guard sheds reads (answers at once, without disk
 * I/O) while a late read is still running and for `shed` after the last miss,
 * so a slow disk does not collect more stuck reads and every reader of the
 * store gets a fast miss instead of one deadline each. After that window one
 * read probes the disk while the others are still shed: if it finishes in
 * time, reads resume; if it misses, the window starts again.
 */
class OffloadReadGuard {
   public:
    using Clock = std::chrono::steady_clock;

    enum class Outcome {
        kDone,          // the read finished (or failed) within the deadline
        kDeadlineMiss,  // the read was not done at the deadline
        kShed,          // refused without disk I/O
    };

    OffloadReadGuard(std::chrono::milliseconds deadline,
                     std::chrono::milliseconds shed, size_t threads)
        : deadline_(deadline), shed_(shed), pool_(threads) {}

    OffloadReadGuard(const OffloadReadGuard&) = delete;
    OffloadReadGuard& operator=(const OffloadReadGuard&) = delete;

    // Queued reads are dropped; a read already on the disk is waited for
    // (it still uses the counters and its release callback, so the owner must
    // outlive this), which blocks as long as a hung device does.
    ~OffloadReadGuard() {
        stopping_.store(true, std::memory_order_release);
        const auto late = late_running_.load(std::memory_order_relaxed);
        if (late > 0) {
            LOG(WARNING) << "action=offload_read_guard_stop, waiting for "
                         << late << " late reads";
        }
        pool_.stop();
    }

    template <typename T>
    tl::expected<T, ErrorCode> Run(
        std::function<tl::expected<T, ErrorCode>()> read,
        std::function<void(const T&)> release_late,
        Outcome* outcome = nullptr) {
        bool probe = false;
        if (!Admit(Clock::now(), &probe)) {
            shed_count_.fetch_add(1, std::memory_order_relaxed);
            if (outcome) *outcome = Outcome::kShed;
            return tl::make_unexpected(
                ErrorCode::UNAVAILABLE_IN_CURRENT_STATUS);
        }

        struct Call {
            std::mutex mu;
            std::condition_variable cv;
            bool done = false;
            bool abandoned = false;
            std::optional<tl::expected<T, ErrorCode>> result;
        };
        auto call = std::make_shared<Call>();
        const auto start = Clock::now();
        try {
            pool_.enqueue([this, call, start, read = std::move(read),
                           release_late = std::move(release_late)]() {
                Work<T>(*call, start, read, release_late);
            });
        } catch (const std::exception& e) {
            LOG(ERROR) << "action=offload_read_enqueue_failed, what="
                       << e.what();
            if (probe) EndProbe(/*in_time=*/false);
            if (outcome) *outcome = Outcome::kDone;
            return tl::make_unexpected(ErrorCode::INTERNAL_ERROR);
        }

        std::unique_lock<std::mutex> lock(call->mu);
        if (call->cv.wait_for(lock, deadline_, [&] { return call->done; })) {
            if (probe) EndProbe(/*in_time=*/true);
            if (outcome) *outcome = Outcome::kDone;
            return std::move(*call->result);
        }
        // Counted before the worker can see the flag (both under call->mu),
        // so its decrement always follows this increment.
        call->abandoned = true;
        late_running_.fetch_add(1, std::memory_order_acq_rel);
        lock.unlock();

        {
            std::lock_guard<std::mutex> state_lock(state_mu_);
            shed_until_ = Clock::now() + shed_;
            suspect_ = true;
        }
        if (probe) EndProbe(/*in_time=*/false);
        deadline_miss_count_.fetch_add(1, std::memory_order_relaxed);
        LOG(WARNING) << "action=offload_read_deadline_miss, deadline_ms="
                     << deadline_.count() << ", shed_ms=" << shed_.count()
                     << ", late_running="
                     << late_running_.load(std::memory_order_relaxed);
        if (outcome) *outcome = Outcome::kDeadlineMiss;
        return tl::make_unexpected(ErrorCode::UNAVAILABLE_IN_CURRENT_STATUS);
    }

    // Whether a read arriving at `now` would be refused (test hook; Run
    // decides under the same lock).
    bool ShouldShed(Clock::time_point now) const {
        if (late_running_.load(std::memory_order_acquire) > 0) return true;
        std::lock_guard<std::mutex> lock(state_mu_);
        return now < shed_until_ || probing_;
    }

    // Guard-local counts (the store exports its own counters).
    uint64_t deadline_miss_count() const {
        return deadline_miss_count_.load(std::memory_order_relaxed);
    }
    uint64_t shed_count() const {
        return shed_count_.load(std::memory_order_relaxed);
    }
    int64_t late_running() const {
        return late_running_.load(std::memory_order_relaxed);
    }

   private:
    // Decides whether a read may go to the disk; sets *probe for the single
    // read allowed after a shed window.
    bool Admit(Clock::time_point now, bool* probe) {
        if (late_running_.load(std::memory_order_acquire) > 0) return false;
        std::lock_guard<std::mutex> lock(state_mu_);
        if (!suspect_) return true;
        if (now < shed_until_ || probing_) return false;
        probing_ = true;
        *probe = true;
        return true;
    }

    void EndProbe(bool in_time) {
        std::lock_guard<std::mutex> lock(state_mu_);
        probing_ = false;
        if (in_time) suspect_ = false;
    }

    template <typename T, typename Call>
    void Work(Call& call, Clock::time_point start,
              const std::function<tl::expected<T, ErrorCode>()>& read,
              const std::function<void(const T&)>& release_late) {
        bool skip = stopping_.load(std::memory_order_acquire);
        {
            std::lock_guard<std::mutex> lock(call.mu);
            // Gave up on while still queued: do not touch the disk at all.
            if (call.abandoned) skip = true;
        }
        tl::expected<T, ErrorCode> result =
            tl::make_unexpected(ErrorCode::INTERNAL_ERROR);
        if (!skip) {
            try {
                result = read();
            } catch (const std::exception& e) {
                LOG(ERROR) << "action=offload_read_exception, what="
                           << e.what();
            } catch (...) {
                LOG(ERROR) << "action=offload_read_exception";
            }
        }
        bool abandoned = false;
        {
            std::lock_guard<std::mutex> lock(call.mu);
            call.done = true;
            abandoned = call.abandoned;
            if (!abandoned) call.result = std::move(result);
        }
        if (!abandoned) {
            call.cv.notify_one();
            return;
        }
        // Nobody waits for this read any more: free its buffer now instead of
        // at the buffer GC.
        if (result) {
            try {
                release_late(*result);
            } catch (...) {
                LOG(ERROR) << "action=offload_read_release_exception";
            }
        }
        const auto elapsed_ms =
            std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now() -
                                                                  start)
                .count();
        const auto left =
            late_running_.fetch_sub(1, std::memory_order_acq_rel) - 1;
        LOG(INFO) << "action=offload_read_late_done, elapsed_ms=" << elapsed_ms
                  << ", read=" << (skip ? "skipped" : "done")
                  << ", ok=" << static_cast<bool>(result)
                  << ", late_running=" << left;
    }

    const std::chrono::milliseconds deadline_;
    const std::chrono::milliseconds shed_;

    mutable std::mutex state_mu_;
    Clock::time_point shed_until_{};  // GUARDED_BY(state_mu_)
    bool suspect_ = false;            // a miss since the last read in time
    bool probing_ = false;            // the post-window probe is running
    std::atomic<bool> stopping_{false};
    std::atomic<int64_t> late_running_{0};
    std::atomic<uint64_t> deadline_miss_count_{0};
    std::atomic<uint64_t> shed_count_{0};

    // Last member: destroyed first, joining reads that still touch the
    // fields above.
    ThreadPool pool_;
};

}  // namespace mooncake
