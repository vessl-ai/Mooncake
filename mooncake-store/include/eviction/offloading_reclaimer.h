#pragma once

// OffloadingReclaimer: a Reclaimer that keeps an object's data on local disk
// before its memory goes. Layered over the reclaimer that does the releasing:
//
// - An object that already has a completed local-disk replica loses nothing,
//   so its memory goes through the inner reclaimer as it is. One being
//   removed does not count: once its removal is durable it is gone; nor does
//   one the queue says no longer keeps the data (OffloadQueue::KeepsData).
// - An object with an offload already in flight keeps its data through the
//   replica that offload pinned, which no plan names while pinned, so its
//   memory goes through the inner reclaimer too. Another offload would pin a
//   replica nothing unpins: an object tracks one offload at a time.
// - A namespace's frontier is the first objects a run meets there, as many
//   as the namespace's share of the object goal, counted as they are freed
//   or queued (Report::frontier_end); a run with a byte goal is all
//   frontier. With frontier_only an object past the frontier keeps its
//   memory. With drop_unread an object in the frontier that nobody read from
//   memory since its put goes without an offload.
// - Otherwise one memory replica the plan names is pinned and queued for
//   offload, and the rest are released now: the data survives through the
//   pinned one, which a later run frees once the copy lands.
// - When no offload can be queued, the object waits for a later run, unless
//   force_evict lets it go without one, or evict_without_local_disk does
//   because every push failed for want of a local-disk tier. With
//   force_evict, a run that has queued `cap` offloads lets the rest go
//   without one too.

#include <string>
#include <vector>

#include <ylt/util/tl/expected.hpp>

#include "eviction/reclaimer.h"
#include "replica.h"
#include "tenant_id.h"
#include "types.h"

namespace mooncake {
namespace eviction {

// Where offloads are queued.
class OffloadQueue {
   public:
    // Queues `source`, a memory replica of the object `key` names, for a copy
    // to its host's local disk. When it was queued, returns the id every
    // queued item carries and appends the destinations to `mirror_clients`;
    // otherwise why not. UNABLE_OFFLOADING means no local-disk tier takes it.
    virtual tl::expected<L4TaskId, ErrorCode> Push(
        const TenantId& tenant_id, const std::string& key, Replica& source,
        std::vector<UUID>* mirror_clients) = 0;
    // Whether `replica`, a completed local-disk replica, still keeps the
    // object's data.
    [[nodiscard]] virtual bool KeepsData(const Replica& replica) const = 0;

   protected:
    ~OffloadQueue() = default;
};

class OffloadingReclaimer final : public Reclaimer {
   public:
    struct Options {
        // How many offloads one run may queue; enforced with force_evict.
        long cap = 0;
        // Let an object go without its offload rather than keep it.
        bool force_evict = false;
        // Let an object go without its offload when no local-disk tier
        // takes it.
        bool evict_without_local_disk = false;
        // Queue offloads only for the run's frontier.
        bool frontier_only = false;
        // Let an unread object in the frontier go without its offload.
        bool drop_unread = false;
    };

    // `inner` and `queue` outlive the reclaimer.
    OffloadingReclaimer(Reclaimer& inner, OffloadQueue& queue, Options options)
        : inner_(inner), queue_(queue), options_(options) {}

    tl::expected<Effect, ErrorCode> Apply(metadata::Namespace& ns,
                                          const route::WriteGuard& hold,
                                          const Plan& plan,
                                          const Report& so_far) override;
    void TearDown(metadata::Namespace& ns,
                  const route::WriteGuard& hold) override {
        inner_.TearDown(ns, hold);
    }

   private:
    // The plan carried out by the inner reclaimer, without an offload.
    tl::expected<Effect, ErrorCode> WithoutOffload(
        metadata::Namespace& ns, const route::WriteGuard& hold,
        const Plan& plan, const Report& so_far, DropReason reason);

    Reclaimer& inner_;
    OffloadQueue& queue_;
    const Options options_;
};

}  // namespace eviction
}  // namespace mooncake
