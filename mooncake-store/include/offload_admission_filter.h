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
// puts it becomes the previous one and a fresh current one starts. A key is
// remembered for at least `window_puts` and at most 2 * `window_puts` later
// puts (of any key).
//
// Each generation is 4 rows of 2 * `window_puts` one-byte counters, so a
// generation is at most half full and its own false "seen" rate stays below
// (1 - e^-0.5)^4, about 2.4%. Memory is 16 bytes per window put. The
// sketch's own halving (at width * depth increments) never runs, because a
// generation is retired after `window_puts` increments.
//
// A false "seen" only writes a first put to disk (eager behavior); a
// forgotten key only skips one disk write. Neither changes which bytes the
// master registers.
class OffloadAdmissionFilter {
   public:
    static constexpr size_t kDepth = 4;

    explicit OffloadAdmissionFilter(size_t window_puts)
        : window_puts_(window_puts > 0 ? window_puts : 1),
          current_(NewGeneration()),
          previous_(NewGeneration()) {}

    // Record a put of |key|. Returns true when |key| was put before within
    // the remembered window.
    bool ObservePut(const std::string& key) {
        bool seen = false;
        bool retire = false;
        {
            std::lock_guard<std::mutex> lock(mu_);
            // increment() returns the count after this put, so >= 2 means
            // the current generation saw the key before.
            seen = current_->increment(key) >= 2 || previous_->count(key) > 0;
            retire = ++puts_in_current_ == window_puts_;
        }
        if (retire) {
            Rotate();
        }
        return seen;
    }

    // Whether |key| was put before within the window, without recording.
    bool Seen(const std::string& key) const {
        std::lock_guard<std::mutex> lock(mu_);
        return current_->count(key) > 0 || previous_->count(key) > 0;
    }

    size_t window_puts() const { return window_puts_; }

    // Bytes held by the two generations.
    size_t MemoryBytes() const { return 2 * kDepth * Width(); }

    uint64_t rotations() const {
        std::lock_guard<std::mutex> lock(mu_);
        return rotations_;
    }

   private:
    size_t Width() const { return 2 * window_puts_; }

    std::unique_ptr<CountMinSketch> NewGeneration() const {
        return std::make_unique<CountMinSketch>(Width(), kDepth);
    }

    // Exactly one ObservePut sees the count reach the window, so one thread
    // rotates. The fresh generation is zeroed and the old one freed outside
    // the lock; puts recorded meanwhile land in the retiring generation and
    // stay remembered as the previous one.
    void Rotate() {
        auto fresh = NewGeneration();
        {
            std::lock_guard<std::mutex> lock(mu_);
            previous_.swap(current_);
            current_.swap(fresh);
            puts_in_current_ = 0;
            ++rotations_;
        }
        // `fresh` now holds the generation that fell out of the window.
    }

    const size_t window_puts_;
    mutable std::mutex mu_;
    std::unique_ptr<CountMinSketch> current_;
    std::unique_ptr<CountMinSketch> previous_;
    size_t puts_in_current_{0};
    uint64_t rotations_{0};
};

}  // namespace mooncake
