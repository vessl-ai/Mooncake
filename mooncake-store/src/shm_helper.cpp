#include "shm_helper.h"

#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <fcntl.h>
#include <unistd.h>
#include <glog/logging.h>

#include <algorithm>
#include <thread>
#include <vector>

#include "common/client_buffer_allocation.h"
#include "config.h"
#include "config/hugepage_config.h"
#ifdef USE_NOF
#include "spdk/spdk_wrapper.h"
#endif
#if defined(USE_ASCEND_DIRECT)
#include "ascend_allocator.h"
#endif

#ifndef MOONCAKE_SHM_NAME
#define MOONCAKE_SHM_NAME "mooncake_shm"
#endif

namespace mooncake {

std::mutex ShmHelper::shm_mutex_;

static int memfd_create_wrapper(const char* name, unsigned int flags) {
#ifdef __NR_memfd_create
    return syscall(__NR_memfd_create, name, flags);
#else
    return -1;
#endif
}

// Build a clear error message for a failed shared-memory allocation step.
// When the allocation uses hugepages (forced by MC_STORE_REGISTER_SPDK=1), the
// usual failure is an insufficient hugepage pool (ENOMEM at ftruncate/mmap);
// spell out exactly how many hugepages of what size are needed so the operator
// can configure them. Non-hugepage messages keep the pre-existing text exactly.
static std::string shm_alloc_error(const char* step, size_t size,
                                   bool use_hugepage, size_t hp_size, int err) {
    if (!use_hugepage) {
        return std::string("Failed to ") + step + ": " + strerror(err);
    }
    std::string msg = std::string("Failed to ") + step + " using ";
    if (hp_size >= SZ_1GB) {
        msg += std::to_string(hp_size / SZ_1GB) + "GB hugepages";
    } else {
        msg += std::to_string(hp_size / (1024 * 1024)) + "MB hugepages";
    }
    msg += " (size " + std::to_string(size) + " bytes, need " +
           std::to_string(size / hp_size) + " hugepages): " + strerror(err);
    msg +=
        ". Configure enough hugepages via /proc/sys/vm/nr_hugepages (or "
        "hugepages-2048kB/hugepages-1048576kB nr_hugepages), or unset "
        "MC_STORE_REGISTER_SPDK to skip SPDK registration";
    return msg;
}

ShmHelper* ShmHelper::getInstance() {
    static ShmHelper instance;
    return &instance;
}

ShmHelper::ShmHelper() {
#ifdef USE_NOF
    // Force SpdkWrapper to complete construction before ShmHelper does, so that
    // at process exit ShmHelper is destroyed FIRST: its cleanup() (which calls
    // SpdkWrapper::UnregisterMemory) runs while the SPDK env is still alive,
    // before ~SpdkWrapper -> Cleanup() -> spdk_env_fini(). Without this, the
    // first host-pool allocation constructs SpdkWrapper AFTER ShmHelper, so at
    // exit SpdkWrapper is destroyed first and cleanup() would call
    // spdk_mem_unregister after spdk_env_fini() (UB / NULL-deref on SPDK
    // >= 26.09). Constructing SpdkWrapper here is side-effect free: its ctor is
    // `= default` and the env is only initialized lazily via InitializeEnv().
    SpdkWrapper::GetInstance();
#endif
    use_hugepage_ = HugepageConfig::IsEnabledFromEnvironment();
    // Read once at construction (ShmHelper is a singleton). Opt-in only: with
    // MC_STORE_REGISTER_SPDK=1, ShmHelper mappings are registered with SPDK so
    // NoF zero-copy transfers can DMA to/from them; otherwise the previous
    // allocation behavior is preserved exactly.
    register_spdk_ = is_register_spdk_enabled();
    if (register_spdk_) {
#ifdef USE_NOF
        // spdk_mem_register() requires hugepage-backed memory: SPDK's vtophys
        // notify checks each 2MB segment's PHYSICAL address for 2MB alignment,
        // which only hugepage pages satisfy (4KB-backed memory always fails
        // with -EINVAL, on v23.01.1 and every later version in iova=pa mode).
        // Virtual-address alignment alone (mmap_shm_2mb_aligned) is not
        // sufficient, so force hugepages here; if not enough are configured the
        // allocation below fails with a clear error instead of silently losing
        // NoF zero-copy. This is an override on top of
        // HugepageConfig::IsEnabledFromEnvironment() above.
        use_hugepage_ = true;
#endif
        LOG(INFO) << "MC_STORE_REGISTER_SPDK=1: shared memory will be "
                     "registered with SPDK for NoF zero-copy transfers";
    }
}

bool ShmHelper::is_register_spdk_enabled() {
    const char* rs = std::getenv("MC_STORE_REGISTER_SPDK");
    return rs != nullptr && std::strcmp(rs, "1") == 0;
}

ShmHelper::~ShmHelper() { cleanup(); }

bool ShmHelper::cleanup() {
    std::lock_guard<std::mutex> lock(shm_mutex_);
    bool ret = true;
    for (auto& shm : shms_) {
        if (shm->fd != -1) {
            close(shm->fd);
            shm->fd = -1;
        }
        if (shm->base_addr) {
#ifdef USE_NOF
            if (shm->spdk_registered) {
                if (SpdkWrapper::GetInstance().UnregisterMemory(
                        shm->base_addr, shm->size) != 0) {
                    // SPDK still holds a translation for this range: do NOT
                    // munmap or clear spdk_registered, or a later mmap could
                    // reuse the VA and NoF would silently DMA to the wrong
                    // memory. Retain the mapping (released by the OS at process
                    // exit; cleanup() only runs from ~ShmHelper).
                    LOG(ERROR) << "Failed to unregister shared memory from "
                                  "SPDK during cleanup; retaining mapping "
                                  "(never munmap): "
                               << shm->base_addr;
                    ret = false;
                    continue;
                }
                shm->spdk_registered = false;
            }
#endif
#if defined(USE_ASCEND_DIRECT)
            if (globalConfig().ascend_agent_mode &&
                globalConfig().ascend_use_fabric_mem) {
                free_memory("ascend", shm->base_addr);
                continue;
            }
#endif
            if (munmap(shm->base_addr, shm->size) == -1) {
                LOG(ERROR) << "Failed to unmap shared memory: "
                           << strerror(errno);
                ret = false;
            }
            shm->base_addr = nullptr;
        }
    }
    shms_.clear();
    return ret;
}

#ifndef MADV_POPULATE_WRITE
#define MADV_POPULATE_WRITE 23
#endif

namespace {

// Fault a freshly mapped segment in, using several threads.
//
// mmap(MAP_POPULATE) reaches the same end state, but it does the whole
// mapping on the calling thread. On a 2 TiB two-socket host that runs at
// about 3.1 GB/s, so the 65.6 GB of HiCache host pools an SGLang prefill
// server allocates costs 21.2 s of its startup (measured on H200, kernel
// 6.8, 4 KiB pages). MADV_POPULATE_WRITE is the range form of the same
// operation, so populating disjoint chunks concurrently produces an
// identical mapping and reaches about 7.6 GB/s.
//
// MC_STORE_POPULATE_THREADS=1 restores the single-threaded behaviour.
size_t populate_thread_count() {
    if (const char* env = std::getenv("MC_STORE_POPULATE_THREADS")) {
        long n = std::strtol(env, nullptr, 10);
        if (n > 0) return static_cast<size_t>(std::min(n, 64L));
    }
    unsigned hw = std::thread::hardware_concurrency();
    return std::min<size_t>(hw ? hw : 1, 8);
}

// Populate is best effort, exactly as MAP_POPULATE is: the kernel may decline
// and the mapping is still valid, with the declined pages faulted in on first
// touch. So a failure is logged and the segment is returned, never thrown.
void populate_parallel(void* base, size_t size) {
    const size_t threads = populate_thread_count();
    const long page = sysconf(_SC_PAGESIZE);
    if (threads <= 1 || page <= 0 || size < (1UL << 30)) {
        if (madvise(base, size, MADV_POPULATE_WRITE) != 0) {
            LOG(WARNING) << "MADV_POPULATE_WRITE failed for " << size
                         << " bytes: " << strerror(errno)
                         << "; pages will fault in on first touch";
        }
        return;
    }

    size_t chunk = (size / threads / static_cast<size_t>(page)) *
                   static_cast<size_t>(page);
    if (chunk == 0) chunk = size;
    const size_t n = (size + chunk - 1) / chunk;

    std::vector<std::thread> workers;
    std::vector<int> failures(n, 0);
    workers.reserve(n);
    size_t spawned = 0;
    for (size_t i = 0; i < n; ++i) {
        try {
            workers.emplace_back([&, i] {
                char* start = static_cast<char*>(base) + i * chunk;
                size_t len = (i == n - 1) ? size - i * chunk : chunk;
                if (madvise(start, len, MADV_POPULATE_WRITE) != 0) {
                    failures[i] = errno;
                }
            });
            ++spawned;
        } catch (const std::exception& e) {
            // std::thread's constructor throws on resource exhaustion (e.g. a
            // container's pids limit). A joinable std::thread's destructor
            // calls std::terminate, so an unjoined worker here would crash
            // the process instead of staying best effort; stop spawning and
            // let the retry loop below finish the rest on this thread.
            LOG(WARNING) << "failed to start populate worker " << i << "/" << n
                         << ": " << e.what()
                         << "; finishing remaining chunks on this thread";
            break;
        }
    }
    for (auto& w : workers) w.join();

    // Any chunk that never got a worker, or whose worker's madvise was
    // declined, is retried once on this thread, so the common transient case
    // still ends fully populated.
    for (size_t i = 0; i < n; ++i) {
        if (i < spawned && failures[i] == 0) continue;
        char* start = static_cast<char*>(base) + i * chunk;
        size_t len = (i == n - 1) ? size - i * chunk : chunk;
        if (madvise(start, len, MADV_POPULATE_WRITE) != 0) {
            LOG(WARNING) << "MADV_POPULATE_WRITE declined chunk " << i << "/"
                         << n << " (" << len << " bytes): " << strerror(errno)
                         << "; those pages will fault in on first touch";
        }
    }
}

}  // namespace

void* ShmHelper::allocate(size_t size) {
    std::lock_guard<std::mutex> lock(shm_mutex_);
    // Dummy-real: FabricMem host uses VMM; non-Fabric host uses memfd+mmap like
    // non-agent / GPU shm path.
#ifdef USE_ASCEND_DIRECT
    if (globalConfig().ascend_agent_mode) {
        if (globalConfig().ascend_use_fabric_mem) {
            void* base_addr = nullptr;
            size_t alloc_size = size;
            base_addr = ascend_allocate_vmm_memory_direct(alloc_size);
            if (base_addr == nullptr) {
                throw std::runtime_error(
                    "Failed to allocate VMM shared memory");
            }
            auto shm = std::make_shared<ShmSegment>();
            shm->fd = -1;
            shm->base_addr = base_addr;
            shm->size = alloc_size;
            shm->requested_size = alloc_size;
            shm->name = MOONCAKE_SHM_NAME;
            shm->registered = false;
            shms_.push_back(shm);
            return base_addr;
        }
        // ascend_agent_mode && !ascend_use_fabric_mem: fall through to memfd
    }
#endif

    // Remember the caller-requested size before any alignment padding (hugepage
    // or 2MB for SPDK registration). shm->size is the padded size; consumers
    // that must match the original request (e.g. DummyClient::register_buffer)
    // use requested_size instead of re-deriving the alignment policy.
    const size_t requested = size;

    unsigned int flags = MFD_CLOEXEC;
    size_t hp_size = 0;  // hugepage size when use_hugepage_ (0 otherwise)
    if (use_hugepage_) {
        bool use_memfd = true;
        hp_size = get_hugepage_size_from_env(&flags, use_memfd);
        if (!(flags & MFD_HUGETLB)) {
            // get_hugepage_size_from_env() returns 0 and sets no MFD_HUGETLB
            // flag when MC_STORE_USE_HUGEPAGE is unset -- exactly the case when
            // MC_STORE_REGISTER_SPDK=1 forces hugepages (see constructor).
            // Default to 2MB hugepages.
            flags |= MFD_HUGETLB | MFD_HUGE_2MB;
            if (hp_size == 0) {
                hp_size = SZ_2MB;
            }
            LOG(INFO) << "Using 2MB hugepages (set MC_STORE_USE_HUGEPAGE and "
                         "MC_STORE_HUGEPAGE_SIZE=1GB to use 1GB)";
        }
        size = align_up(size, hp_size);
        LOG(INFO) << "Using huge pages for shared memory, size: " << size;
    }
    // When MC_STORE_REGISTER_SPDK=1 forces hugepages, the branch above already
    // aligned size to the hugepage size, and hugetlb mmap returns a
    // hugepage-aligned base, so no separate 2MB size padding or aligned base
    // mapping (mmap_shm_2mb_aligned, common/mmap_aligned.h) is needed on the
    // sender side.
    // (The receiver maps the shared fd and aligns its own base in
    // RealClient::map_shm_internal_with_device.)

    int fd = memfd_create_wrapper(MOONCAKE_SHM_NAME, flags);
    if (fd == -1) {
        throw std::runtime_error(
            shm_alloc_error("create anonymous shared memory", size,
                            use_hugepage_, hp_size, errno));
    }

    if (ftruncate(fd, size) == -1) {
        int err = errno;
        close(fd);
        throw std::runtime_error(shm_alloc_error("set shared memory size", size,
                                                 use_hugepage_, hp_size, err));
    }

    // MAP_POPULATE is deliberately absent: populate_parallel below reaches the
    // same end state with several threads. See its comment for the numbers.
    void* base_addr =
        mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (base_addr == MAP_FAILED) {
        int err = errno;
        close(fd);
        throw std::runtime_error(shm_alloc_error("map shared memory", size,
                                                 use_hugepage_, hp_size, err));
    }
    populate_parallel(base_addr, size);

    auto shm = std::make_shared<ShmSegment>();
    shm->fd = fd;
    shm->base_addr = base_addr;
    shm->size = size;
    shm->requested_size = requested;
    shm->name = MOONCAKE_SHM_NAME;
    shm->registered = false;
#ifdef USE_NOF
    // Register the mapping with SPDK so NoF (NVMe-oF) RDMA transfers can DMA
    // to/from this buffer directly (spdk_rdma_get_translation). Registration
    // failure is non-fatal: the buffer stays usable for all non-NoF paths.
    // Enabled only with MC_STORE_REGISTER_SPDK=1 (read once at construction).
    if (register_spdk_) {
        if (!SpdkWrapper::IsRegistrableRange(base_addr, size)) {
            // The hugepage forcing in the constructor makes this unreachable
            // today (hugetlb mmap returns a hugepage-aligned base and size is
            // aligned to the hugepage size), but a range SPDK would reject
            // up-front must not be registered at all: the rejection happens
            // before any state is marked, so munmap below stays safe.
            LOG(WARNING) << "Shared memory is not 2MB-aligned; not registering "
                            "with SPDK: addr="
                         << base_addr << ", size=" << size
                         << "; NoF zero-copy transfers to this buffer will be "
                            "unavailable";
        } else {
            const int register_rc =
                SpdkWrapper::GetInstance().RegisterMemory(base_addr, size);
            if (register_rc == 0) {
                shm->spdk_registered = true;
            } else if (register_rc == -EBUSY) {
                // The range is already registered, so SPDK marked nothing new
                // for us. Do NOT roll back: the unregister below would clear
                // the existing registration (memory.c:358-366 detects it,
                // 445-466 clears it). Retain the mapping and let teardown
                // retry.
                LOG(ERROR) << "Shared memory is already registered with SPDK: "
                           << base_addr << ", size: " << size
                           << "; retaining mapping";
                shm->spdk_registered = true;
            } else {
                LOG(WARNING) << "Failed to register shared memory with SPDK: "
                             << "addr=" << base_addr << ", size=" << size
                             << "; NoF zero-copy transfers to this buffer will "
                                "be unavailable";
                // spdk_mem_register() marks the range in g_mem_reg_map before
                // running its notify callbacks and does NOT roll back on
                // failure (SPDK v23.01.1, memory.c:370-384), and in iova=va it
                // can already have installed the IOMMU mapping when a later
                // step fails (memory.c:1092-1107). This unregister is the only
                // chance to undo that, and only a 0 return proves it worked:
                // -EINVAL is also returned for a half-marked range
                // (memory.c:426) and, in iova=va, for an incomplete
                // translation before the IOMMU is unmapped
                // (memory.c:1216-1224). -EINVAL covers both a failure before
                // anything was mapped and one after the DMA mapping was
                // installed, and the codes cannot be told apart, so retaining
                // a mapping that turns out to have been clean is the accepted
                // cost of never munmapping a live translation.
                const int rollback_rc =
                    SpdkWrapper::GetInstance().UnregisterMemory(base_addr,
                                                                size);
                if (rollback_rc != 0) {
                    LOG(ERROR)
                        << "Failed to roll back incomplete SPDK registration: "
                        << base_addr << ", size: " << size
                        << ", rc: " << rollback_rc
                        << "; treating the range as registered so free()/"
                           "cleanup() retry the unregister and quarantine the "
                           "mapping instead of munmapping";
                    shm->spdk_registered = true;
                }
            }
        }
    }
#endif
    shms_.push_back(shm);

    return base_addr;
}

int ShmHelper::free(void* addr) {
    std::lock_guard<std::mutex> lock(shm_mutex_);
    for (auto it = shms_.begin(); it != shms_.end(); ++it) {
        if ((*it)->base_addr == addr) {
            if ((*it)->fd != -1) {
                close((*it)->fd);
                (*it)->fd = -1;
            }
            if ((*it)->base_addr) {
#ifdef USE_NOF
                if ((*it)->spdk_registered) {
                    if (SpdkWrapper::GetInstance().UnregisterMemory(
                            (*it)->base_addr, (*it)->size) != 0) {
                        // SPDK still holds a translation for this range: do NOT
                        // munmap, clear spdk_registered, or erase the segment.
                        // Retain the mapping so the VA cannot be reused while
                        // SPDK's translation is live (a later free() retries).
                        LOG(ERROR) << "Failed to unregister shared memory from "
                                      "SPDK during free; retaining mapping "
                                      "(never munmap): "
                                   << (*it)->base_addr;
                        return -1;
                    }
                    (*it)->spdk_registered = false;
                }
#endif
#if defined(USE_ASCEND_DIRECT)
                if (globalConfig().ascend_agent_mode &&
                    globalConfig().ascend_use_fabric_mem) {
                    free_memory("ascend", (*it)->base_addr);
                } else
#endif
                    if (munmap((*it)->base_addr, (*it)->size) == -1) {
                    LOG(ERROR) << "Failed to unmap shared memory during free: "
                               << strerror(errno);
                    return -1;
                }
            }
            LOG(INFO) << "Freed shared memory at " << addr
                      << ", size: " << (*it)->size;
            shms_.erase(it);
            return 0;
        }
    }
    LOG(ERROR) << "Attempted to free unknown shared memory address: " << addr;
    return -1;
}

std::shared_ptr<ShmHelper::ShmSegment> ShmHelper::get_shm(void* addr) {
    std::lock_guard<std::mutex> lock(shm_mutex_);
    const uintptr_t address = reinterpret_cast<uintptr_t>(addr);
    for (auto& shm : shms_) {
        const uintptr_t base = reinterpret_cast<uintptr_t>(shm->base_addr);
        if (address >= base && address - base < shm->size) {
            return shm;
        }
    }
    return nullptr;
}

}  // namespace mooncake
