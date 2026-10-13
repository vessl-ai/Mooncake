#pragma once

// Fork-only shutdown handoff snapshot format (INF-600).
//
// A handoff snapshot directory holds the upstream batch-OpLog snapshot
// artifacts (object chunks, segments, weight metadata, manifest and
// descriptor; see ha/snapshot/batch_oplog/) plus two fork files:
//
//   handoff_manifest        versioned header: segment identities, chunk
//                           index with CRCs, next replica id, config and
//                           codec fingerprints. Written last.
//   handoff_quarantine.bin  memory ranges that a writer or reader of the old
//                           master may still use (in-flight puts, other
//                           incomplete replicas, discarded replicas still
//                           held back, ranges still quarantined by a previous
//                           restore), each with its remaining hold time.
//
// Segment identity (segment id, base, owner client id, ...) lives here rather
// than in upstream StandbySegmentInfo, so upstream struct changes cannot
// change how our files decode. SegmentIdentity maps 1:1 onto the fields an
// upstream "segment incarnation" change would add to StandbySegmentInfo.

#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <ylt/struct_pack.hpp>
#include <ylt/util/tl/expected.hpp>

#include "local_ssd/manager.h"
#include "segment/status.h"
#include "types.h"

namespace mooncake::handoff {

inline constexpr char kHandoffFormat[] = "mooncake-handoff";
// Schema 1: memory replicas only (stage 1/2). Schema 2 adds the LocalSSD
// (L4) state: LOCAL_DISK replicas in the chunks (same chunk codec) and the
// per-client registrations. Schema 3 adds the remaining hold time of every
// quarantine range, the ranges of the old master's discarded replicas, and
// DRAINING / DRAINED segments with their status. Schema 4 replaces the
// offload pins of schema 2/3 with the offload takes (OffloadTake). Older
// snapshots lack ranges that may still be written or takes whose report may
// still arrive, so the reader accepts schema 4 only (an older snapshot means
// one cold start).
inline constexpr uint32_t kHandoffSchemaVersion = 4;
inline constexpr uint32_t kHandoffMinSchemaVersion = 4;
inline constexpr char kHandoffManifestFile[] = "handoff_manifest";
inline constexpr char kHandoffQuarantineFile[] = "handoff_quarantine.bin";
inline constexpr char kHandoffLocalDiskFile[] = "handoff_local_disk.bin";
inline constexpr char kHandoffOffloadTakesFile[] = "handoff_offload_takes.bin";
inline constexpr char kHandoffSnapshotRoot[] = "handoff";

struct SegmentIdentity {
    UUID segment_id{0, 0};
    std::string name;
    std::string te_endpoint;
    uint64_t base{0};
    uint64_t size{0};
    std::string protocol;
    std::string host_id;
    UUID owner_client_id{0, 0};
    // OK, DRAINING or DRAINED (schema 3). A restored DRAINING / DRAINED
    // segment keeps that status and never becomes allocatable on its own.
    SegmentStatus status{SegmentStatus::OK};

    Segment ToSegment() const;
    static SegmentIdentity FromSegment(const Segment& segment,
                                       const UUID& owner);
};

// A file in the snapshot directory, verified by size and CRC32C.
struct BlobInfo {
    std::string key;
    uint64_t bytes{0};
    uint32_t crc32c{0};
};

struct ChunkInfo {
    uint64_t index{0};
    std::string key;
    uint64_t objects{0};
    uint64_t bytes{0};
    uint32_t crc32c{0};
};

struct QuarantineRange {
    uint32_t segment_index{0};
    uint64_t address{0};
    uint64_t size{0};
    // How long the old master would still have held the range back, counted
    // from the export. The restore holds it this long after the restore ends
    // (the gap between export and restore only adds to it).
    uint64_t hold_ms{0};
    YLT_REFL(QuarantineRange, segment_index, address, size, hold_ms);
};

// A store whose report (completion or NACK) of an offload of `key` may still
// arrive although no master tracks that offload any more: a taker of a task
// the old master still had (tasks are not carried, so the roll drops them),
// or a stale take the old master still held. The restore records it as a
// stale take for `hold_ms` after the restore ends, so the store cannot take
// a newer offload of the key before such a report can no longer arrive.
struct OffloadTake {
    std::string tenant_id;
    std::string key;
    UUID client_id{0, 0};
    uint64_t hold_ms{0};
    YLT_REFL(OffloadTake, tenant_id, key, client_id, hold_ms);
};

// Fingerprints of the struct_pack layouts the chunk and segment files were
// written with. A reader built from a different layout rejects the snapshot
// before decoding anything.
struct CodecFingerprint {
    uint32_t object_entry{0};
    uint32_t segments{0};
    uint32_t quarantine{0};
    // Schema 2 only; 0 in a schema-1 manifest.
    uint32_t local_disk{0};
    uint32_t offload_takes{0};
    bool operator==(const CodecFingerprint&) const = default;
};
CodecFingerprint CurrentCodecFingerprint();
// True if `fp` (from a manifest of `schema`) matches this build.
bool CodecMatches(const CodecFingerprint& fp, uint32_t schema);

struct WriterStats {
    uint64_t objects{0};
    uint64_t replicas{0};
    uint64_t skipped_processing{0};
    uint64_t skipped_no_exportable_replica{0};
    uint64_t dropped_non_memory_replicas{0};
    uint64_t dropped_incomplete_replicas{0};
    uint64_t dropped_unknown_segment_replicas{0};
    uint64_t quarantine_ranges{0};
    uint64_t quarantine_bytes{0};
    // Schema 3: ranges of the old master's discarded replicas that are still
    // held back, and those that could not be carried (segment not exported).
    uint64_t discarded_ranges{0};
    uint64_t dropped_discarded_ranges{0};
    // L4 (schema 2).
    uint64_t local_disk_replicas{0};
    uint64_t disk_only_objects{0};
    uint64_t dropped_local_disk_replicas{0};       // owner not exported
    uint64_t dropped_local_disk_size_mismatch{0};  // object_size != size
    uint64_t offload_takes{0};
    uint64_t total_bytes{0};
    uint64_t workers{0};
    uint64_t duration_ms{0};
    uint64_t peak_rss_kib{0};
};

struct Manifest {
    std::string format{kHandoffFormat};
    uint32_t schema_version{kHandoffSchemaVersion};
    std::string snapshot_id;
    int64_t created_at_ms{0};
    CodecFingerprint codec;
    std::string memory_allocator;  // "offset" | "cachelib"
    bool multi_tenant{false};
    uint64_t next_replica_id{0};
    uint64_t object_count{0};
    uint64_t replica_count{0};
    std::vector<SegmentIdentity> segments;
    std::vector<ChunkInfo> chunks;
    std::optional<BlobInfo> weight_metadata;
    BlobInfo quarantine;
    uint64_t quarantine_ranges{0};
    // L4 state (schema 2). `replica_count` above counts memory replicas.
    bool enable_offload{false};
    uint64_t local_disk_replica_count{0};
    uint64_t disk_only_object_count{0};
    std::optional<BlobInfo> local_disk;  // vector<LocalSsdHandoffClient>
    uint64_t local_disk_clients{0};
    std::optional<BlobInfo> offload_takes;  // vector<OffloadTake>
    uint64_t offload_take_count{0};
    WriterStats writer;
};

// Maps a memory descriptor (endpoint, address, size) to the index of the
// segment that contains it. Segment name and transport endpoint are both
// accepted as aliases; several segments may share an alias and are told apart
// by address range.
class SegmentLocator {
   public:
    explicit SegmentLocator(const std::vector<SegmentIdentity>& segments);
    std::optional<uint32_t> Find(const std::string& endpoint, uint64_t address,
                                 uint64_t size) const;

   private:
    const std::vector<SegmentIdentity>& segments_;
    std::unordered_map<std::string, std::vector<uint32_t>> by_alias_;
};

// Checks a segment list from a manifest: non-empty names, non-zero ranges,
// unique ids, no CXL, a status of OK / DRAINING / DRAINED, and no overlapping
// ranges among segments sharing an alias. Returns an error string on the
// first violation.
std::optional<std::string> ValidateSegments(
    const std::vector<SegmentIdentity>& segments);

// File layout: "MOONCAKE-HANDOFF <schema> <crc32c> <body bytes>\n<json body>".
std::string EncodeManifest(const Manifest& manifest);
tl::expected<Manifest, std::string> DecodeManifest(std::string_view file);

std::vector<uint8_t> EncodeQuarantine(
    const std::vector<QuarantineRange>& ranges);
tl::expected<std::vector<QuarantineRange>, std::string> DecodeQuarantine(
    const std::vector<uint8_t>& bytes);

std::vector<uint8_t> EncodeLocalDiskClients(
    const std::vector<LocalSsdHandoffClient>& clients);
tl::expected<std::vector<LocalSsdHandoffClient>, std::string>
DecodeLocalDiskClients(const std::vector<uint8_t>& bytes);
std::vector<uint8_t> EncodeOffloadTakes(const std::vector<OffloadTake>& takes);
tl::expected<std::vector<OffloadTake>, std::string> DecodeOffloadTakes(
    const std::vector<uint8_t>& bytes);

// Reads a whole file. Error string on I/O failure, or if the file is larger
// than `max_bytes` (checked before anything is allocated).
tl::expected<std::vector<uint8_t>, std::string> ReadFileBytes(
    const std::string& path,
    uint64_t max_bytes = std::numeric_limits<uint64_t>::max());
// Upper bound for the handoff manifest file.
inline constexpr uint64_t kMaxManifestBytes = 64ULL << 20;
// Reads a blob and checks its size and CRC32C.
tl::expected<std::vector<uint8_t>, std::string> ReadVerifiedBlob(
    const std::string& dir, const std::string& key, uint64_t bytes,
    uint32_t crc32c);

}  // namespace mooncake::handoff
