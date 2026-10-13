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
//   handoff_quarantine.bin  memory ranges of replicas that were left out
//                           (in-flight puts and other incomplete replicas).
//
// Segment identity (segment id, base, owner client id, ...) lives here rather
// than in upstream StandbySegmentInfo, so upstream struct changes cannot
// change how our files decode. SegmentIdentity maps 1:1 onto the fields an
// upstream "segment incarnation" change would add to StandbySegmentInfo.

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <ylt/struct_pack.hpp>
#include <ylt/util/tl/expected.hpp>

#include "types.h"

namespace mooncake::handoff {

inline constexpr char kHandoffFormat[] = "mooncake-handoff";
inline constexpr uint32_t kHandoffSchemaVersion = 1;
inline constexpr char kHandoffManifestFile[] = "handoff_manifest";
inline constexpr char kHandoffQuarantineFile[] = "handoff_quarantine.bin";
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
    YLT_REFL(QuarantineRange, segment_index, address, size);
};

// Fingerprints of the struct_pack layouts the chunk and segment files were
// written with. A reader built from a different layout rejects the snapshot
// before decoding anything.
struct CodecFingerprint {
    uint32_t object_entry{0};
    uint32_t segments{0};
    uint32_t quarantine{0};
    bool operator==(const CodecFingerprint&) const = default;
};
CodecFingerprint CurrentCodecFingerprint();

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
// unique ids, no CXL, and no overlapping ranges among segments sharing an
// alias. Returns an error string on the first violation.
std::optional<std::string> ValidateSegments(
    const std::vector<SegmentIdentity>& segments);

// File layout: "MOONCAKE-HANDOFF <schema> <crc32c> <body bytes>\n<json body>".
std::string EncodeManifest(const Manifest& manifest);
tl::expected<Manifest, std::string> DecodeManifest(std::string_view file);

std::vector<uint8_t> EncodeQuarantine(
    const std::vector<QuarantineRange>& ranges);
tl::expected<std::vector<QuarantineRange>, std::string> DecodeQuarantine(
    const std::vector<uint8_t>& bytes);

// Reads a whole file. Error string on I/O failure.
tl::expected<std::vector<uint8_t>, std::string> ReadFileBytes(
    const std::string& path);
// Reads a blob and checks its size and CRC32C.
tl::expected<std::vector<uint8_t>, std::string> ReadVerifiedBlob(
    const std::string& dir, const std::string& key, uint64_t bytes,
    uint32_t crc32c);

}  // namespace mooncake::handoff
