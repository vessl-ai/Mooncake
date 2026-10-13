#include "handoff/handoff_admission.h"

#include <sched.h>
#include <sys/statvfs.h>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>
#include <sstream>

namespace mooncake::handoff {

namespace {

constexpr uint64_t kMiB = 1024ULL * 1024;
constexpr uint64_t kMinSafetyBytes = 512 * kMiB;
constexpr double kSafetyFraction = 0.03;
constexpr uint64_t kWorkerFixedBytes = 16 * kMiB;
constexpr uint64_t kWorkerBytesPerShardObject = 1350;
constexpr uint64_t kWriterBaseBytes = 64 * kMiB;
constexpr uint64_t kShards = 1024;
constexpr uint64_t kDiskBytesPerObject = 200;
constexpr uint64_t kWorkerBytesPerShardDiskReplica = 200;
constexpr uint64_t kDiskBytesPerDiskReplica = 100;
constexpr uint64_t kDiskFixedBytes = 16 * kMiB;
constexpr double kDiskFactor = 1.3;

bool ReadFirstLine(const std::string& path, std::string* out) {
    if (path.empty()) return false;
    std::ifstream in(path);
    if (!in) return false;
    return static_cast<bool>(std::getline(in, *out));
}

// Reads a cgroup value file: a number or "max". Returns false if missing or
// malformed; *unlimited is set for "max".
bool ReadCgroupValue(const std::string& path, uint64_t* value,
                     bool* unlimited) {
    std::string line;
    if (!ReadFirstLine(path, &line)) return false;
    if (line == "max") {
        *unlimited = true;
        *value = std::numeric_limits<uint64_t>::max();
        return true;
    }
    try {
        size_t pos = 0;
        *value = std::stoull(line, &pos);
        *unlimited = false;
        return pos > 0;
    } catch (const std::exception&) {
        return false;
    }
}

// "key value" lines (memory.stat, and /proc/meminfo with a "kB" suffix).
uint64_t ReadKeyedValue(const std::string& path, const std::string& key,
                        uint64_t multiplier, bool* found) {
    *found = false;
    if (path.empty()) return 0;
    std::ifstream in(path);
    std::string line;
    while (std::getline(in, line)) {
        std::istringstream ls(line);
        std::string k;
        uint64_t v = 0;
        if (!(ls >> k >> v)) continue;
        if (!k.empty() && k.back() == ':') k.pop_back();
        if (k == key) {
            *found = true;
            return v * multiplier;
        }
    }
    return 0;
}

uint64_t SaturatingSub(uint64_t a, uint64_t b) { return a > b ? a - b : 0; }

}  // namespace

CgroupPaths CgroupPaths::Under(const std::string& cgroup_root,
                               const std::string& meminfo) {
    CgroupPaths p;
    p.memory_max = cgroup_root + "/memory.max";
    p.memory_high = cgroup_root + "/memory.high";
    p.memory_current = cgroup_root + "/memory.current";
    p.memory_stat = cgroup_root + "/memory.stat";
    p.cpu_max = cgroup_root + "/cpu.max";
    p.v1_limit = cgroup_root + "/memory/memory.limit_in_bytes";
    p.v1_usage = cgroup_root + "/memory/memory.usage_in_bytes";
    p.meminfo = meminfo;
    return p;
}

std::string MemorySnapshot::ToJson() const {
    std::ostringstream o;
    o << "{\"source\":\"" << source << "\",\"limit\":" << limit
      << ",\"limited\":" << (limited ? "true" : "false")
      << ",\"current\":" << current << ",\"anon\":" << anon
      << ",\"file\":" << file << ",\"file_dirty\":" << file_dirty
      << ",\"file_writeback\":" << file_writeback
      << ",\"inactive_file\":" << inactive_file << "}";
    return o.str();
}

tl::expected<MemorySnapshot, std::string> ReadMemorySnapshot(
    const CgroupPaths& paths) {
    MemorySnapshot snap;
    bool found = false;
    const uint64_t mem_total =
        ReadKeyedValue(paths.meminfo, "MemTotal", 1024, &found);
    const uint64_t mem_available =
        ReadKeyedValue(paths.meminfo, "MemAvailable", 1024, &found);

    uint64_t max_value = 0;
    bool max_unlimited = false;
    if (ReadCgroupValue(paths.memory_max, &max_value, &max_unlimited)) {
        snap.source = "cgroup_v2";
        uint64_t high_value = 0;
        bool high_unlimited = true;
        if (!ReadCgroupValue(paths.memory_high, &high_value, &high_unlimited)) {
            high_unlimited = true;
        }
        uint64_t limit = std::numeric_limits<uint64_t>::max();
        if (!max_unlimited) limit = max_value;
        if (!high_unlimited) limit = std::min(limit, high_value);
        bool current_unlimited = false;
        if (!ReadCgroupValue(paths.memory_current, &snap.current,
                             &current_unlimited) ||
            current_unlimited) {
            return tl::make_unexpected("unreadable " + paths.memory_current);
        }
        snap.limited = limit != std::numeric_limits<uint64_t>::max();
        snap.limit = snap.limited ? limit : mem_total;
        snap.anon = ReadKeyedValue(paths.memory_stat, "anon", 1, &found);
        snap.file = ReadKeyedValue(paths.memory_stat, "file", 1, &found);
        snap.file_dirty =
            ReadKeyedValue(paths.memory_stat, "file_dirty", 1, &found);
        snap.file_writeback =
            ReadKeyedValue(paths.memory_stat, "file_writeback", 1, &found);
        snap.inactive_file =
            ReadKeyedValue(paths.memory_stat, "inactive_file", 1, &found);
    } else if (ReadCgroupValue(paths.v1_limit, &max_value, &max_unlimited)) {
        snap.source = "cgroup_v1";
        bool usage_unlimited = false;
        if (!ReadCgroupValue(paths.v1_usage, &snap.current, &usage_unlimited)) {
            return tl::make_unexpected("unreadable " + paths.v1_usage);
        }
        // v1 reports a huge page-aligned number when unlimited.
        snap.limited = !max_unlimited && mem_total > 0 && max_value < mem_total;
        snap.limit = snap.limited ? max_value : mem_total;
    } else if (mem_total > 0) {
        snap.source = "meminfo";
        snap.limit = mem_total;
        snap.current = SaturatingSub(mem_total, mem_available);
    } else {
        return tl::make_unexpected("no cgroup memory files and no meminfo");
    }
    if (snap.limit == 0) {
        return tl::make_unexpected("memory limit is unknown");
    }
    return snap;
}

size_t AvailableCpus(const CgroupPaths& paths) {
    size_t cpus = 1;
#ifdef __linux__
    cpu_set_t set;
    CPU_ZERO(&set);
    if (sched_getaffinity(0, sizeof(set), &set) == 0) {
        cpus = std::max<size_t>(1, CPU_COUNT(&set));
    }
#endif
    std::string line;
    if (ReadFirstLine(paths.cpu_max, &line)) {
        std::istringstream ls(line);
        std::string quota;
        uint64_t period = 0;
        if (ls >> quota >> period && quota != "max" && period > 0) {
            try {
                const double q = std::stod(quota) / static_cast<double>(period);
                cpus = std::min<size_t>(
                    cpus,
                    std::max<size_t>(1, static_cast<size_t>(std::ceil(q))));
            } catch (const std::exception&) {
            }
        }
    }
    return cpus;
}

uint64_t FreeDiskBytes(const std::string& path) {
    struct statvfs st{};
    if (::statvfs(path.c_str(), &st) != 0) return 0;
    return static_cast<uint64_t>(st.f_bavail) * st.f_frsize;
}

uint64_t AdmissionSafetyBytes(uint64_t limit) {
    return std::max<uint64_t>(
        kMinSafetyBytes,
        static_cast<uint64_t>(static_cast<double>(limit) * kSafetyFraction));
}

std::string ShutdownAdmission::ToJson() const {
    std::ostringstream o;
    o << "{\"admit\":" << (admit ? "true" : "false")
      << ",\"workers\":" << workers << ",\"reason\":\"" << reason
      << "\",\"headroom\":" << headroom << ",\"safety\":" << safety
      << ",\"per_worker\":" << per_worker << ",\"base\":" << base
      << ",\"disk_needed\":" << disk_needed
      << ",\"restore_needed\":" << restore_needed << "}";
    return o.str();
}

ShutdownAdmission DecideShutdownAdmission(const ShutdownAdmissionInput& in) {
    ShutdownAdmission out;
    out.headroom = in.memory.Headroom();
    out.safety = AdmissionSafetyBytes(in.memory.limit);
    out.base = kWriterBaseBytes;
    const uint64_t objects_per_shard = (in.objects + kShards - 1) / kShards;
    const uint64_t disk_per_shard =
        (in.local_disk_objects + kShards - 1) / kShards;
    out.per_worker = kWorkerFixedBytes +
                     objects_per_shard * kWorkerBytesPerShardObject +
                     disk_per_shard * kWorkerBytesPerShardDiskReplica;
    out.disk_needed = static_cast<uint64_t>(
        kDiskFactor *
        static_cast<double>(in.objects * kDiskBytesPerObject +
                            in.local_disk_objects * kDiskBytesPerDiskReplica +
                            kDiskFixedBytes));
    out.restore_needed = RestoreNeedBytes(
        in.memory.limit, in.objects, in.restore_bytes_per_object,
        in.local_disk_objects, in.restore_bytes_per_disk_replica);

    const uint64_t budget = SaturatingSub(out.headroom, out.safety + out.base);
    const uint64_t fit = budget / out.per_worker;
    out.workers = static_cast<size_t>(std::min<uint64_t>(
        fit, std::min(in.max_workers, std::max<size_t>(1, in.cpus))));
    if (out.workers == 0) {
        out.reason = "memory_headroom";
        return out;
    }
    if (in.disk_free_bytes < out.disk_needed) {
        out.workers = 0;
        out.reason = "disk_space";
        return out;
    }
    if (out.restore_needed > in.memory.limit) {
        // The next master could not restore it (with its L4 state); do not
        // spend the shutdown on a snapshot that would be discarded.
        out.workers = 0;
        out.reason = "restore_memory";
        return out;
    }
    out.admit = true;
    return out;
}

std::string RestoreAdmission::ToJson() const {
    std::ostringstream o;
    o << "{\"admit\":" << (admit ? "true" : "false") << ",\"needed\":" << needed
      << ",\"headroom\":" << headroom << "}";
    return o.str();
}

uint64_t RestoreNeedBytes(uint64_t limit, uint64_t objects,
                          uint64_t bytes_per_object, uint64_t disk_replicas,
                          uint64_t bytes_per_disk_replica) {
    return objects * bytes_per_object + disk_replicas * bytes_per_disk_replica +
           AdmissionSafetyBytes(limit);
}

RestoreAdmission DecideRestoreAdmission(const MemorySnapshot& memory,
                                        uint64_t objects,
                                        uint64_t bytes_per_object,
                                        uint64_t disk_replicas,
                                        uint64_t bytes_per_disk_replica) {
    RestoreAdmission out;
    out.headroom = memory.Headroom();
    out.needed = RestoreNeedBytes(memory.limit, objects, bytes_per_object,
                                  disk_replicas, bytes_per_disk_replica);
    out.admit = out.needed <= out.headroom;
    return out;
}

}  // namespace mooncake::handoff
