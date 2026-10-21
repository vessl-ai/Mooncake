#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>

#include "count_min_sketch.h"

namespace mooncake {

// "Has this key been put before?" for the eager offload admission
// (offload_admission_second_put). Two CountMinSketch generations: every put
// is recorded in the current one, and once it has recorded `window_puts`
// puts it becomes the previous one and a zeroed spare becomes the current
// one. A key is remembered for at least `window_puts` and at most
// 2 * `window_puts` later puts (of any key), as long as a spare is ready.
//
// Each generation is 4 rows of 2 * `window_puts` one-byte counters, so a
// generation is at most half full and its own false "seen" rate stays below
// (1 - e^-0.5)^4, about 2.4%. The sketch's own halving (at width * depth
// increments) never runs, because a generation is retired after
// `window_puts` increments.
//
// Three generations are allocated once, at construction: current, previous
// and a spare (24 bytes per window put). ObservePut never allocates, zeroes
// or frees memory: a rotation only moves pointers, and the generation that
// falls out of the window is zeroed later, a slice at a time, by
// PrepareSpare on a background thread (the master's eviction thread). If
// the spare is not ready when the window is reached, the rotation waits:
// puts keep landing in the current generation, which fills past half its
// width (false "seen" rises) until the next ObservePut after the spare is
// ready.
//
// A false "seen" only writes a first put to disk (eager behavior); a
// forgotten key only skips one disk write. Neither changes which bytes the
// master registers.
class OffloadAdmissionFilter {
   public:
    static constexpr size_t kDepth = 4;
    static constexpr size_t kGenerations = 3;

    explicit OffloadAdmissionFilter(size_t window_puts)
        : window_puts_(window_puts > 0 ? window_puts : 1),
          current_(NewGeneration()),
          previous_(NewGeneration()),
          spare_(NewGeneration()) {}

    // Record a put of |key|. Returns true when |key| was put before within
    // the remembered window.
    bool ObservePut(const std::string& key) {
        std::lock_guard<std::mutex> lock(mu_);
        // increment() returns the count after this put, so >= 2 means the
        // current generation saw the key before.
        const bool seen =
            current_->increment(key) >= 2 || previous_->count(key) > 0;
        ++puts_in_current_;
        if (puts_in_current_ >= window_puts_) {
            if (spare_) {
                retired_ = std::move(previous_);
                previous_ = std::move(current_);
                current_ = std::move(spare_);
                puts_in_current_ = 0;
                ++rotations_;
            } else if (puts_in_current_ == window_puts_) {
                ++deferred_rotations_;
            }
        }
        return seen;
    }

    // Whether |key| was put before within the window, without recording.
    bool Seen(const std::string& key) const {
        std::lock_guard<std::mutex> lock(mu_);
        return current_->count(key) > 0 || previous_->count(key) > 0;
    }

    // Zeroes up to `max_bytes` of the generation retired by the last
    // rotation; once it is all zero it becomes the spare. Returns true when
    // a spare is ready. Callers are serialized by their own lock; the filter
    // lock is held only to move pointers, never while zeroing.
    bool PrepareSpare(size_t max_bytes) {
        std::lock_guard<std::mutex> prepare_lock(prepare_mu_);
        if (!preparing_) {
            std::lock_guard<std::mutex> lock(mu_);
            if (spare_) {
                return true;
            }
            if (!retired_) {
                return false;
            }
            preparing_ = std::move(retired_);
            prepared_counters_ = 0;
        }
        prepared_counters_ =
            preparing_->clear_range(prepared_counters_, max_bytes);
        if (prepared_counters_ < preparing_->size()) {
            return false;
        }
        std::lock_guard<std::mutex> lock(mu_);
        spare_ = std::move(preparing_);
        return true;
    }

    size_t window_puts() const { return window_puts_; }

    // Bytes held by the generations (allocated once, at construction).
    size_t MemoryBytes() const { return kGenerations * kDepth * Width(); }

    uint64_t rotations() const {
        std::lock_guard<std::mutex> lock(mu_);
        return rotations_;
    }

    // Times the window was reached with no spare ready yet.
    uint64_t deferred_rotations() const {
        std::lock_guard<std::mutex> lock(mu_);
        return deferred_rotations_;
    }

   private:
    size_t Width() const { return 2 * window_puts_; }

    std::unique_ptr<CountMinSketch> NewGeneration() const {
        return std::make_unique<CountMinSketch>(Width(), kDepth);
    }

    const size_t window_puts_;
    mutable std::mutex mu_;
    std::unique_ptr<CountMinSketch> current_;
    std::unique_ptr<CountMinSketch> previous_;
    // Zeroed and ready to become current_ (null while being prepared).
    std::unique_ptr<CountMinSketch> spare_;
    // Fell out of the window, not yet zeroed.
    std::unique_ptr<CountMinSketch> retired_;
    size_t puts_in_current_{0};
    uint64_t rotations_{0};
    uint64_t deferred_rotations_{0};
    // Owned by PrepareSpare (under prepare_mu_).
    std::mutex prepare_mu_;
    std::unique_ptr<CountMinSketch> preparing_;
    size_t prepared_counters_{0};
};

}  // namespace mooncake
