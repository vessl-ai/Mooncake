#pragma once

// Ghost-key reuse-age sampler.
//
// The master knows when a key leaves the cache (its last memory copy is
// evicted, or its last disk copy is evicted) and when a key is asked for
// again (ExistKey, GetReplicaList, PutStart). The time between the two is
// the key's reuse age: a cache that kept the key that much longer would
// have served the request. Its distribution is the right half of the
// miss-ratio curve, which the master cannot otherwise observe.
//
// Only a fixed hash-selected fraction of keys is tracked (spatial sampling,
// as in SHARDS, Waldspurger et al. 2015): a key is sampled iff its hash falls
// below `sampling_rate`, so every event of a sampled key is seen and the
// distribution of the sample is an unbiased estimate of the whole. Each
// eviction of a sampled key leaves a "ghost" (hash and time) in a bounded
// table. The first later reference of that key reports the ghost's age and
// drops it. The table is a ring per shard: when full, the oldest ghost is
// overwritten, so the table always covers the most recent `max_entries`
// sampled evictions; reuse ages beyond that window are counted as overflow.
//
// Cost: callers check IsSampled() (one hash) before calling in, so an
// unsampled key never takes a lock. Each shard has its own small mutex,
// which is a leaf lock: nothing else is acquired while holding it, so it is
// safe to call in while holding a metadata shard lock.

#include <atomic>
#include <chrono>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "mutex.h"
#include "tenant_id.h"

namespace mooncake {

class ReuseSampler {
   public:
    using Clock = std::chrono::steady_clock;

    // Number of independently locked sub-tables.
    static constexpr size_t kNumShards = 16;
    // Upper bound for max_entries; keeps ring slots addressable by uint32.
    static constexpr size_t kMaxEntriesLimit = 100'000'000;

    // What the first reference after an eviction found in the ghost table.
    struct Reuse {
        // Time since the key's last memory copy was evicted. Empty when the
        // ghost was left by a disk eviction alone (the memory eviction was
        // not seen, e.g. the key reached the disk tier another way).
        std::optional<Clock::duration> since_memory_eviction;
        // Time since the key's last disk copy was evicted. Empty when the
        // disk copy has not been evicted (or the key never had one).
        std::optional<Clock::duration> since_disk_eviction;
        // True when the memory eviction left a disk copy and that copy has
        // not been evicted since: the reference could be served from disk.
        bool disk_copy_present{false};
    };

    struct Stats {
        uint64_t memory_evictions{0};  // ghosts left by memory evictions
        uint64_t disk_evictions{0};    // ghosts left by disk evictions
        uint64_t reuses{0};            // references that found a ghost
        uint64_t overflowed{0};        // ghosts overwritten unreferenced
        uint64_t entries{0};           // live ghosts now
    };

    // sampling_rate must be in (0, 1]; max_entries in [1, kMaxEntriesLimit].
    // Invalid values throw std::invalid_argument.
    ReuseSampler(double sampling_rate, size_t max_entries);

    ReuseSampler(const ReuseSampler&) = delete;
    ReuseSampler& operator=(const ReuseSampler&) = delete;

    // 64-bit key fingerprint. The same (tenant, key) always maps to the same
    // value; the bits used for sampling and for shard choice are independent
    // of the master's metadata shard index.
    static uint64_t Fingerprint(const TenantId& tenant_id,
                                const std::string& user_key);

    bool IsSampled(uint64_t fingerprint) const {
        return (fingerprint >> kSampleShift) < sample_threshold_;
    }

    // The key's last memory copy was evicted. `disk_copy_kept` says whether
    // a disk copy remains. Replaces an existing ghost of the same key.
    // The fingerprint must be sampled.
    void RecordMemoryEviction(uint64_t fingerprint, bool disk_copy_kept,
                              Clock::time_point now);

    // The key's last copy, a disk copy, was evicted: the key is gone. Keeps
    // the memory eviction time of an existing ghost and moves the ghost to
    // the newest end of the window. The fingerprint must be sampled.
    void RecordDiskEviction(uint64_t fingerprint, Clock::time_point now);

    // The key was referenced. Returns the ghost it left, and drops it, or
    // nullopt when the key has no ghost. The fingerprint must be sampled.
    std::optional<Reuse> RecordReference(uint64_t fingerprint,
                                         Clock::time_point now);

    Stats GetStats() const;
    double sampling_rate() const { return sampling_rate_; }
    size_t capacity() const { return shard_capacity_ * kNumShards; }

   private:
    // Sampling compares the top 24 bits of the fingerprint with the rate.
    static constexpr int kSampleShift = 40;
    static constexpr uint64_t kSampleSpace = uint64_t{1} << (64 - kSampleShift);

    struct Entry {
        uint64_t fingerprint{0};
        int64_t memory_evicted_ns{0};
        int64_t disk_evicted_ns{0};
        bool live{false};
        bool has_memory_eviction{false};
        bool has_disk_eviction{false};
        bool disk_copy{false};
    };

    // A ring of ghosts in insertion order plus an open-addressing index
    // (linear probing, load factor <= 0.5) from fingerprint to ring slot.
    struct Shard {
        Mutex mutex;
        std::vector<Entry> ring GUARDED_BY(mutex);
        size_t next_slot GUARDED_BY(mutex) = 0;
        size_t filled GUARDED_BY(mutex) = 0;
        // 0 = empty, otherwise ring slot + 1.
        std::vector<uint32_t> index GUARDED_BY(mutex);
        size_t index_mask GUARDED_BY(mutex) = 0;
    };

    Shard& ShardFor(uint64_t fingerprint) const {
        return *shards_[fingerprint & (kNumShards - 1)];
    }
    static size_t HomeSlot(uint64_t fingerprint, size_t mask) {
        return static_cast<size_t>(fingerprint >> 4) & mask;
    }
    static int64_t ToNs(Clock::time_point t) {
        return std::chrono::duration_cast<std::chrono::nanoseconds>(
                   t.time_since_epoch())
            .count();
    }

    // Index position holding `fingerprint`, or nullopt.
    static std::optional<size_t> FindLocked(Shard& shard, uint64_t fingerprint)
        REQUIRES(shard.mutex);
    // Removes the index entry at `pos` (Knuth's algorithm R, no tombstones)
    // and marks its ring slot dead.
    void EraseLocked(Shard& shard, size_t pos) REQUIRES(shard.mutex);
    // Writes `entry` at the ring head, overwriting the oldest slot when full.
    void AppendLocked(Shard& shard, const Entry& entry, int64_t now_ns)
        REQUIRES(shard.mutex);

    const double sampling_rate_;
    const uint64_t sample_threshold_;
    const size_t shard_capacity_;
    std::vector<std::unique_ptr<Shard>> shards_;

    std::atomic<uint64_t> memory_evictions_{0};
    std::atomic<uint64_t> disk_evictions_{0};
    std::atomic<uint64_t> reuses_{0};
    std::atomic<uint64_t> overflowed_{0};
    std::atomic<int64_t> entries_{0};
};

}  // namespace mooncake
