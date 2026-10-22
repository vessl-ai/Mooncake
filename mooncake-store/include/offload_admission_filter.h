#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "count_min_sketch.h"

namespace mooncake {

// One generation of the put history kept by OffloadAdmissionFilter. The
// filter serializes all access to a generation (its lock for the current and
// previous generations, PrepareSpare for the one being zeroed).
class OffloadAdmissionGeneration {
   public:
    virtual ~OffloadAdmissionGeneration() = default;
    // Records a put of |key|; returns true when this generation had
    // recorded |key| before.
    virtual bool ObservePut(const std::string& key) = 0;
    virtual bool Contains(const std::string& key) const = 0;
    virtual size_t Bytes() const = 0;
    // Zeroes up to |max_bytes| bytes starting at byte |begin| and returns
    // the byte where the next call continues (Bytes() once all are zero).
    virtual size_t ClearRange(size_t begin, size_t max_bytes) = 0;
};

// CountMinSketch generation (the default): 4 rows of 2 * window_puts
// one-byte counters. A key counts as recorded when every row's counter is
// at least 1, so a key whose four counters were all raised by other keys
// reads as recorded. A generation is retired at half fill, where it answers
// "recorded" for a new key about (1 - e^-0.5)^4 = 2.4% of the time; with the
// filling current generation added, about 3% of first puts read as put
// before (measured 2.9% with SGLang-shaped keys). The sketch's own halving
// (at width * depth increments) never runs, because a generation is retired
// after `window_puts` increments.
class CountMinSketchAdmissionGeneration final
    : public OffloadAdmissionGeneration {
   public:
    static constexpr size_t kDepth = 4;

    explicit CountMinSketchAdmissionGeneration(size_t window_puts)
        : sketch_(2 * window_puts, kDepth) {}

    bool ObservePut(const std::string& key) override {
        // increment() returns the count after this put, so >= 2 means the
        // generation saw the key before.
        return sketch_.increment(key) >= 2;
    }
    bool Contains(const std::string& key) const override {
        return sketch_.count(key) > 0;
    }
    size_t Bytes() const override { return sketch_.size(); }
    size_t ClearRange(size_t begin, size_t max_bytes) override {
        return sketch_.clear_range(begin, max_bytes);
    }

   private:
    CountMinSketch sketch_;
};

// Bloom filter generation (offload_admission_bloom_filter): the same bytes
// as the CountMinSketch generation (8 * window_puts) used as single bits,
// 64 bits per window put, with kProbes bit positions per key. A full
// generation answers "recorded" for a new key about (1 - e^(-4/64))^4 =
// 1.4e-5 of the time. It only answers "put before or not", which is all the
// second-put admission asks.
class BloomAdmissionGeneration final : public OffloadAdmissionGeneration {
   public:
    static constexpr size_t kProbes = 4;
    static constexpr size_t kBitsPerWindowPut = 64;

    explicit BloomAdmissionGeneration(size_t window_puts)
        : bits_(window_puts * kBitsPerWindowPut),
          bytes_(window_puts * kBitsPerWindowPut / 8, 0) {}

    bool ObservePut(const std::string& key) override {
        size_t pos[kProbes];
        Positions(key, pos);
        bool recorded = true;
        for (size_t p : pos) {
            const uint8_t mask = static_cast<uint8_t>(1u << (p & 7));
            uint8_t& byte = bytes_[p >> 3];
            if (!(byte & mask)) {
                recorded = false;
                byte |= mask;
            }
        }
        return recorded;
    }
    bool Contains(const std::string& key) const override {
        size_t pos[kProbes];
        Positions(key, pos);
        for (size_t p : pos) {
            if (!(bytes_[p >> 3] & (1u << (p & 7)))) {
                return false;
            }
        }
        return true;
    }
    size_t Bytes() const override { return bytes_.size(); }
    size_t ClearRange(size_t begin, size_t max_bytes) override {
        begin = std::min(begin, bytes_.size());
        const size_t end = begin + std::min(max_bytes, bytes_.size() - begin);
        if (begin < end) {
            std::fill(bytes_.begin() + begin, bytes_.begin() + end, uint8_t{0});
        }
        return end;
    }

   private:
    static uint64_t Mix(uint64_t x) {
        // splitmix64 finalizer
        x += 0x9e3779b97f4a7c15ULL;
        x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
        x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
        return x ^ (x >> 31);
    }

    // kProbes positions in [0, bits_) from one string hash: each probe
    // mixes the hash with its own offset, then maps the 64-bit value onto
    // the range with a multiply-high (no modulo bias worth noting).
    void Positions(const std::string& key, size_t* pos) const {
        const uint64_t h = std::hash<std::string>{}(key);
        for (size_t i = 0; i < kProbes; ++i) {
            const uint64_t x = Mix(h + i * 0xd6e8feb86659fd93ULL);
            pos[i] = static_cast<size_t>(
                (static_cast<unsigned __int128>(x) * bits_) >> 64);
        }
    }

    const size_t bits_;
    std::vector<uint8_t> bytes_;
};

// "Has this key been put before?" for the eager offload admission
// (offload_admission_second_put). Two generations (CountMinSketch by default,
// Bloom filter bits with offload_admission_bloom_filter): every put is
// recorded in the current one, and once it has recorded `window_puts` puts
// it becomes the previous one and a zeroed spare becomes the current one. A
// key is remembered for at least `window_puts` and at most 2 * `window_puts`
// later puts (of any key), as long as a spare is ready.
//
// Three generations are allocated once, at construction: current, previous
// and a spare (24 bytes per window put for either kind). ObservePut never
// allocates, zeroes or frees memory: a rotation only moves pointers, and the
// generation that falls out of the window is zeroed later, a slice at a
// time, by PrepareSpare on a background thread (the master's eviction
// thread). If the spare is not ready when the window is reached, the
// rotation waits: puts keep landing in the current generation, which fills
// past its design load (false "seen" rises) until the next ObservePut after
// the spare is ready.
//
// A false "seen" writes a first put to disk (eager behavior); a forgotten
// key only skips one disk write. Neither changes which bytes the master
// registers. A false "seen" still costs hit rate: it writes one object of a
// chunk whose other objects stay memory-only, and when that older disk copy
// is evicted it leaves a gap inside the chunk's later copies. That is why
// the Bloom kind exists.
//
// The filter lives only in the master's memory: a master restart or handoff
// starts it empty, so for up to two windows afterwards every put counts as
// a first put and is not written to disk.
class OffloadAdmissionFilter {
   public:
    static constexpr size_t kGenerations = 3;

    enum class Kind { kCountMinSketch, kBloom };

    explicit OffloadAdmissionFilter(size_t window_puts,
                                    Kind kind = Kind::kCountMinSketch)
        : window_puts_(window_puts > 0 ? window_puts : 1),
          kind_(kind),
          current_(NewGeneration()),
          previous_(NewGeneration()),
          spare_(NewGeneration()) {}

    // Record a put of |key|. Returns true when |key| was put before within
    // the remembered window.
    bool ObservePut(const std::string& key) {
        std::lock_guard<std::mutex> lock(mu_);
        const bool seen = current_->ObservePut(key) || previous_->Contains(key);
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
        return current_->Contains(key) || previous_->Contains(key);
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
            prepared_bytes_ = 0;
        }
        prepared_bytes_ = preparing_->ClearRange(prepared_bytes_, max_bytes);
        if (prepared_bytes_ < preparing_->Bytes()) {
            return false;
        }
        std::lock_guard<std::mutex> lock(mu_);
        spare_ = std::move(preparing_);
        return true;
    }

    size_t window_puts() const { return window_puts_; }
    Kind kind() const { return kind_; }

    // Bytes held by the generations (allocated once, at construction):
    // 8 * window_puts per generation for both kinds.
    static_assert(BloomAdmissionGeneration::kBitsPerWindowPut / 8 ==
                      2 * CountMinSketchAdmissionGeneration::kDepth,
                  "both kinds use 8 bytes per window put and generation");
    size_t MemoryBytes() const { return kGenerations * 8 * window_puts_; }

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
    std::unique_ptr<OffloadAdmissionGeneration> NewGeneration() const {
        if (kind_ == Kind::kBloom) {
            return std::make_unique<BloomAdmissionGeneration>(window_puts_);
        }
        return std::make_unique<CountMinSketchAdmissionGeneration>(
            window_puts_);
    }

    const size_t window_puts_;
    const Kind kind_;
    mutable std::mutex mu_;
    std::unique_ptr<OffloadAdmissionGeneration> current_;
    std::unique_ptr<OffloadAdmissionGeneration> previous_;
    // Zeroed and ready to become current_ (null while being prepared).
    std::unique_ptr<OffloadAdmissionGeneration> spare_;
    // Fell out of the window, not yet zeroed.
    std::unique_ptr<OffloadAdmissionGeneration> retired_;
    size_t puts_in_current_{0};
    uint64_t rotations_{0};
    uint64_t deferred_rotations_{0};
    // Owned by PrepareSpare (under prepare_mu_).
    std::mutex prepare_mu_;
    std::unique_ptr<OffloadAdmissionGeneration> preparing_;
    size_t prepared_bytes_{0};
};

}  // namespace mooncake
