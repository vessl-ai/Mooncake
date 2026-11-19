#pragma once

// What one eviction run did, object by object (Effect) and in total (Report).

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

#include "replica.h"

namespace mooncake {
namespace eviction {

// What reclaiming an object's memory came to.
enum class Outcome : uint8_t {
    // The object lost its last valid replica and goes.
    kDropped,
    // The object lost memory and lives on through its other replicas.
    kDemoted,
    // One memory replica was queued for offload to local disk and is freed by
    // a later run once the copy lands; the rest were released now.
    kOffloadQueued,
    // Nothing was released: the object waits for a later run.
    kKept,
};
inline constexpr size_t kOutcomeCount = 4;

// Why an object went without the offload that was due for it.
enum class DropReason : uint8_t {
    kNone,
    // The run had queued as many offloads as it may.
    kOffloadCap,
    // The offload could not be queued.
    kOffloadRejected,
    // No local-disk tier takes the offload (evict_without_local_disk).
    kNoLocalDisk,
    // Not read from memory since its put (drop_unread).
    kUnread,
};

// Why an object that was due an offload kept its memory.
enum class KeepReason : uint8_t {
    kNone,
    // No local-disk tier takes the offload.
    kNoLocalDisk,
    // Past the run's frontier (frontier_only), with no disk copy.
    kBeyondFrontier,
};

// What reclaiming one object did. Move-only: it carries the replicas taken off
// the object, which must outlive the object's lock.
struct Effect {
    Outcome outcome = Outcome::kKept;
    DropReason reason = DropReason::kNone;
    KeepReason keep_reason = KeepReason::kNone;
    // Memory taken off the object: freed now, or once the OpLog entry that
    // records it is durable.
    uint64_t released_bytes = 0;
    // Memory pinned for an offload, freed by a later run.
    uint64_t offload_bytes = 0;
    // The object holds nothing valid any more and must be torn down.
    bool needs_teardown = false;
    // The replicas taken off the object, freed when this is destroyed.
    std::vector<Replica> detached;

    Effect() = default;
    Effect(Effect&&) = default;
    Effect& operator=(Effect&&) = default;
    Effect(const Effect&) = delete;
    Effect& operator=(const Effect&) = delete;
};

// What a run did in total.
struct Report {
    // Keys the eviction indexes had filed when the run began.
    uint64_t filed = 0;
    // For an object goal, the Taken() count at which the namespace being
    // walked has met its share of the target; unset for a byte goal.
    std::optional<uint64_t> frontier_end;
    // Objects that released memory, and the memory they released.
    uint64_t evicted_objects = 0;
    uint64_t released_bytes = 0;
    // Memory pinned for offloads, freed by a later run.
    uint64_t offload_bytes = 0;
    std::array<uint64_t, kOutcomeCount> outcomes{};
    // Objects dropped without the offload due for them, by DropReason.
    uint64_t offload_cap_drops = 0;
    uint64_t offload_rejected_drops = 0;
    uint64_t offload_no_local_disk_drops = 0;
    uint64_t offload_unread_drops = 0;
    // Objects that kept their memory, by KeepReason.
    uint64_t offload_no_local_disk_kept = 0;
    uint64_t offload_beyond_frontier_kept = 0;
    // An OpLog write failed and cut the run short.
    bool stopped = false;

    [[nodiscard]] uint64_t Count(Outcome outcome) const {
        return outcomes[static_cast<size_t>(outcome)];
    }

    // Objects dropped, demoted or queued for offload.
    [[nodiscard]] uint64_t Taken() const {
        return Count(Outcome::kDropped) + Count(Outcome::kDemoted) +
               Count(Outcome::kOffloadQueued);
    }

    // Something was released or queued for offload.
    [[nodiscard]] bool MadeProgress() const {
        return evicted_objects > 0 || Count(Outcome::kOffloadQueued) > 0;
    }

    void Add(const Effect& effect) {
        ++outcomes[static_cast<size_t>(effect.outcome)];
        released_bytes += effect.released_bytes;
        offload_bytes += effect.offload_bytes;
        if (effect.released_bytes > 0) {
            ++evicted_objects;
        }
        if (effect.reason == DropReason::kOffloadCap) {
            ++offload_cap_drops;
        } else if (effect.reason == DropReason::kOffloadRejected) {
            ++offload_rejected_drops;
        } else if (effect.reason == DropReason::kNoLocalDisk) {
            ++offload_no_local_disk_drops;
        } else if (effect.reason == DropReason::kUnread) {
            ++offload_unread_drops;
        }
        if (effect.keep_reason == KeepReason::kNoLocalDisk) {
            ++offload_no_local_disk_kept;
        } else if (effect.keep_reason == KeepReason::kBeyondFrontier) {
            ++offload_beyond_frontier_kept;
        }
    }
};

}  // namespace eviction
}  // namespace mooncake
