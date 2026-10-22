#include "reuse_sampler.h"

#include <algorithm>
#include <cmath>
#include <functional>
#include <stdexcept>

#include <boost/functional/hash.hpp>

#include "master_metric_manager.h"

namespace mooncake {

namespace {

// splitmix64 finalizer: a bijective mix, so the sampling bits it yields are
// independent of the low bits the metadata shard index uses.
uint64_t Mix64(uint64_t x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

size_t NextPowerOfTwo(size_t value) {
    size_t result = 1;
    while (result < value) {
        result <<= 1;
    }
    return result;
}

double ValidatedRate(double sampling_rate) {
    if (!(sampling_rate > 0.0 && sampling_rate <= 1.0)) {
        throw std::invalid_argument(
            "reuse sampler sampling rate must be in (0, 1]");
    }
    return sampling_rate;
}

size_t ValidatedMaxEntries(size_t max_entries) {
    if (max_entries == 0 || max_entries > ReuseSampler::kMaxEntriesLimit) {
        throw std::invalid_argument(
            "reuse sampler max entries must be in [1, 100000000]");
    }
    return max_entries;
}

}  // namespace

ReuseSampler::ReuseSampler(double sampling_rate, size_t max_entries)
    : sampling_rate_(ValidatedRate(sampling_rate)),
      sample_threshold_(std::max<uint64_t>(
          1, static_cast<uint64_t>(std::llround(
                 sampling_rate_ * static_cast<double>(kSampleSpace))))),
      shard_capacity_((ValidatedMaxEntries(max_entries) + kNumShards - 1) /
                      kNumShards) {
    const size_t index_size =
        NextPowerOfTwo(std::max<size_t>(2, 2 * shard_capacity_));
    shards_.reserve(kNumShards);
    for (size_t i = 0; i < kNumShards; ++i) {
        auto shard = std::make_unique<Shard>();
        MutexLocker lock(&shard->mutex);
        shard->ring.resize(shard_capacity_);
        shard->index.assign(index_size, 0);
        shard->index_mask = index_size - 1;
        shards_.push_back(std::move(shard));
    }
    MasterMetricManager::instance().set_reuse_sampler_config(sampling_rate_,
                                                             capacity());
}

uint64_t ReuseSampler::Fingerprint(const TenantId& tenant_id,
                                   const std::string& user_key) {
    size_t seed = 0;
    if (tenant_id.IsDefault()) {
        seed = std::hash<std::string>{}(user_key);
    } else {
        seed = std::hash<std::string>{}(tenant_id.value());
        boost::hash_combine(seed, user_key);
    }
    return Mix64(static_cast<uint64_t>(seed));
}

std::optional<size_t> ReuseSampler::FindLocked(Shard& shard,
                                               uint64_t fingerprint) {
    size_t pos = HomeSlot(fingerprint, shard.index_mask);
    while (shard.index[pos] != 0) {
        if (shard.ring[shard.index[pos] - 1].fingerprint == fingerprint) {
            return pos;
        }
        pos = (pos + 1) & shard.index_mask;
    }
    return std::nullopt;
}

void ReuseSampler::EraseLocked(Shard& shard, size_t pos) {
    shard.ring[shard.index[pos] - 1].live = false;
    entries_.fetch_sub(1, std::memory_order_relaxed);
    // Shift later entries of the probe run back so lookups never stop at
    // the hole: an entry at j may move to the hole at i unless its home
    // slot lies cyclically in (i, j].
    size_t hole = pos;
    size_t j = pos;
    while (true) {
        j = (j + 1) & shard.index_mask;
        if (shard.index[j] == 0) {
            break;
        }
        const size_t home = HomeSlot(shard.ring[shard.index[j] - 1].fingerprint,
                                     shard.index_mask);
        const bool home_in_range =
            hole <= j ? (hole < home && home <= j) : (hole < home || home <= j);
        if (!home_in_range) {
            shard.index[hole] = shard.index[j];
            hole = j;
        }
    }
    shard.index[hole] = 0;
}

void ReuseSampler::AppendLocked(Shard& shard, const Entry& entry,
                                int64_t now_ns) {
    const size_t slot = shard.next_slot;
    Entry& old = shard.ring[slot];
    if (shard.filled == shard_capacity_) {
        // The ring has wrapped: the slot holds the oldest ghost. Its age is
        // the window this shard covers.
        if (old.live) {
            auto pos = FindLocked(shard, old.fingerprint);
            if (pos) {
                EraseLocked(shard, *pos);
            }
            overflowed_.fetch_add(1, std::memory_order_relaxed);
            MasterMetricManager::instance().inc_reuse_sampler_ghost_overflow();
        }
        const int64_t recorded_ns =
            std::max(old.has_memory_eviction ? old.memory_evicted_ns : 0,
                     old.has_disk_eviction ? old.disk_evicted_ns : 0);
        MasterMetricManager::instance().set_reuse_sampler_window_seconds(
            (now_ns - recorded_ns) / 1'000'000'000);
    } else {
        ++shard.filled;
    }
    old = entry;
    old.live = true;
    size_t pos = HomeSlot(entry.fingerprint, shard.index_mask);
    while (shard.index[pos] != 0) {
        pos = (pos + 1) & shard.index_mask;
    }
    shard.index[pos] = static_cast<uint32_t>(slot + 1);
    shard.next_slot = (slot + 1) % shard_capacity_;
    entries_.fetch_add(1, std::memory_order_relaxed);
}

void ReuseSampler::RecordMemoryEviction(uint64_t fingerprint,
                                        bool disk_copy_kept,
                                        Clock::time_point now) {
    const int64_t now_ns = ToNs(now);
    Entry entry;
    entry.fingerprint = fingerprint;
    entry.has_memory_eviction = true;
    entry.memory_evicted_ns = now_ns;
    entry.disk_copy = disk_copy_kept;
    {
        Shard& shard = ShardFor(fingerprint);
        MutexLocker lock(&shard.mutex);
        if (auto pos = FindLocked(shard, fingerprint)) {
            EraseLocked(shard, *pos);
        }
        AppendLocked(shard, entry, now_ns);
    }
    memory_evictions_.fetch_add(1, std::memory_order_relaxed);
    MasterMetricManager::instance().set_reuse_sampler_ghost_entries(
        entries_.load(std::memory_order_relaxed));
}

void ReuseSampler::RecordDiskEviction(uint64_t fingerprint,
                                      Clock::time_point now) {
    const int64_t now_ns = ToNs(now);
    Entry entry;
    entry.fingerprint = fingerprint;
    entry.has_disk_eviction = true;
    entry.disk_evicted_ns = now_ns;
    {
        Shard& shard = ShardFor(fingerprint);
        MutexLocker lock(&shard.mutex);
        if (auto pos = FindLocked(shard, fingerprint)) {
            const Entry& previous = shard.ring[shard.index[*pos] - 1];
            entry.has_memory_eviction = previous.has_memory_eviction;
            entry.memory_evicted_ns = previous.memory_evicted_ns;
            EraseLocked(shard, *pos);
        }
        AppendLocked(shard, entry, now_ns);
    }
    disk_evictions_.fetch_add(1, std::memory_order_relaxed);
    MasterMetricManager::instance().set_reuse_sampler_ghost_entries(
        entries_.load(std::memory_order_relaxed));
}

std::optional<ReuseSampler::Reuse> ReuseSampler::RecordReference(
    uint64_t fingerprint, Clock::time_point now) {
    Entry found;
    {
        Shard& shard = ShardFor(fingerprint);
        MutexLocker lock(&shard.mutex);
        auto pos = FindLocked(shard, fingerprint);
        if (!pos) {
            return std::nullopt;
        }
        found = shard.ring[shard.index[*pos] - 1];
        EraseLocked(shard, *pos);
    }
    reuses_.fetch_add(1, std::memory_order_relaxed);
    MasterMetricManager::instance().set_reuse_sampler_ghost_entries(
        entries_.load(std::memory_order_relaxed));

    const int64_t now_ns = ToNs(now);
    Reuse reuse;
    if (found.has_memory_eviction) {
        reuse.since_memory_eviction =
            std::chrono::nanoseconds(now_ns - found.memory_evicted_ns);
    }
    if (found.has_disk_eviction) {
        reuse.since_disk_eviction =
            std::chrono::nanoseconds(now_ns - found.disk_evicted_ns);
    }
    reuse.disk_copy_present = found.disk_copy;
    return reuse;
}

ReuseSampler::Stats ReuseSampler::GetStats() const {
    Stats stats;
    stats.memory_evictions = memory_evictions_.load(std::memory_order_relaxed);
    stats.disk_evictions = disk_evictions_.load(std::memory_order_relaxed);
    stats.reuses = reuses_.load(std::memory_order_relaxed);
    stats.overflowed = overflowed_.load(std::memory_order_relaxed);
    stats.entries = static_cast<uint64_t>(
        std::max<int64_t>(0, entries_.load(std::memory_order_relaxed)));
    return stats;
}

}  // namespace mooncake
