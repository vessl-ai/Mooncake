#include "eviction/offloading_reclaimer.h"

#include <utility>

#include "object_runtime_state.h"

namespace mooncake {
namespace eviction {

namespace {

bool InFrontier(const Report& so_far) {
    return !so_far.frontier_end || so_far.Taken() < *so_far.frontier_end;
}

}  // namespace

tl::expected<Effect, ErrorCode> OffloadingReclaimer::Apply(
    metadata::Namespace& ns, const route::WriteGuard& hold, const Plan& plan,
    const Report& so_far) {
    ObjectMetadata& metadata = hold.metadata();
    // A local-disk replica whose removal awaits durability no longer keeps
    // the data, so it does not spare the object its offload.
    if (metadata.HasReplica([this](const Replica& replica) {
            return replica.is_local_disk_replica() && replica.is_completed() &&
                   queue_.KeepsData(replica);
        })) {
        return inner_.Apply(ns, hold, plan, so_far);
    }
    // The replica an offload in flight pinned keeps the data until it lands.
    if (hold.state().offloading_task.has_value()) {
        return inner_.Apply(ns, hold, plan, so_far);
    }
    const bool in_frontier = InFrontier(so_far);
    if (options_.frontier_only && !in_frontier) {
        Effect kept;
        kept.keep_reason = KeepReason::kBeyondFrontier;
        return kept;
    }
    if (options_.drop_unread && in_frontier &&
        metadata.unread_since_put.load(std::memory_order_relaxed)) {
        return WithoutOffload(ns, hold, plan, so_far, DropReason::kUnread);
    }
    if (options_.force_evict && static_cast<long>(so_far.Count(
                                    Outcome::kOffloadQueued)) >= options_.cap) {
        return WithoutOffload(ns, hold, plan, so_far, DropReason::kOffloadCap);
    }

    bool push_failed = false;
    bool only_no_local_disk = true;
    for (const ReplicaID id : plan.release) {
        Replica* source = metadata.GetReplicaByID(id);
        if (source == nullptr) {
            continue;
        }
        std::vector<UUID> mirror_clients;
        const auto task_id =
            queue_.Push(ns.id(), hold.key(), *source, &mirror_clients);
        if (!task_id) {
            push_failed = true;
            only_no_local_disk = only_no_local_disk &&
                                 task_id.error() == ErrorCode::UNABLE_OFFLOADING;
            continue;
        }
        source->inc_refcnt();
        // Only this replica was pushed, so every mirror copies from it.
        std::vector<UUID> source_clients = mirror_clients;
        hold.state().offloading_task =
            OffloadingTask{.source_id = id,
                           .start_time = Clock::now(),
                           .mirror_clients = std::move(mirror_clients),
                           .source_clients = std::move(source_clients),
                           .task_id = *task_id};

        Plan rest;
        for (const ReplicaID other : plan.release) {
            if (other != id) {
                rest.release.push_back(other);
            }
        }
        auto effect = rest.release.empty()
                          ? tl::expected<Effect, ErrorCode>(Effect{})
                          : inner_.Apply(ns, hold, rest, so_far);
        if (effect) {
            effect->outcome = Outcome::kOffloadQueued;
            effect->offload_bytes = metadata.TotalLength();
        }
        return effect;
    }

    const bool no_local_disk = push_failed && only_no_local_disk;
    if (options_.evict_without_local_disk && no_local_disk) {
        return WithoutOffload(ns, hold, plan, so_far, DropReason::kNoLocalDisk);
    }
    if (options_.force_evict) {
        return WithoutOffload(ns, hold, plan, so_far,
                              DropReason::kOffloadRejected);
    }
    Effect kept;
    if (no_local_disk) {
        kept.keep_reason = KeepReason::kNoLocalDisk;
    }
    return kept;
}

tl::expected<Effect, ErrorCode> OffloadingReclaimer::WithoutOffload(
    metadata::Namespace& ns, const route::WriteGuard& hold, const Plan& plan,
    const Report& so_far, DropReason reason) {
    auto effect = inner_.Apply(ns, hold, plan, so_far);
    if (effect) {
        effect->reason = reason;
    }
    return effect;
}

}  // namespace eviction
}  // namespace mooncake
