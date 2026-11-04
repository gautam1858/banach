// Operating-system layer of banach-engine: page size, read-only file handles, residency
// checks, prefetch, eviction and I/O counters.
//
//   Linux    readahead + MADV_POPULATE_READ, MADV_PAGEOUT / MADV_DONTNEED + fadvise, mincore
//   macOS    MADV_WILLNEED + touching pages, MADV_DONTNEED (a soft budget), mincore
//   Windows  PrefetchVirtualMemory + touching pages, VirtualUnlock to trim the working set
//            (a soft budget), no residency ground truth
#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <psapi.h>
#include <fcntl.h>
#include <io.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/resource.h>
#include <unistd.h>
#ifndef MADV_PAGEOUT
#define MADV_PAGEOUT 21
#endif
#ifndef MADV_POPULATE_READ
#define MADV_POPULATE_READ 22
#endif
#endif

namespace plat {

inline size_t page_size() {
#if defined(_WIN32)
    SYSTEM_INFO si;
    GetSystemInfo(&si);
    return (size_t) si.dwPageSize;
#else
    return (size_t) sysconf(_SC_PAGESIZE);
#endif
}

// Read-only descriptor of a model file (no symlinks on POSIX). -1 on failure.
inline int open_ro(const std::string & path) {
#if defined(_WIN32)
    return _open(path.c_str(), _O_RDONLY | _O_BINARY);
#else
    return open(path.c_str(), O_RDONLY | O_NOFOLLOW);
#endif
}

inline void close_fd(int fd) {
#if defined(_WIN32)
    _close(fd);
#else
    close(fd);
#endif
}

// Fraction of [a, a+len) resident in memory, or -1 when the platform cannot tell.
inline double residency(void * a, size_t len, size_t page) {
#if defined(_WIN32)
    (void) a; (void) len; (void) page;
    return -1.0;
#else
    std::vector<unsigned char> vec(len / page);
#if defined(__linux__)
    if (mincore(a, len, vec.data()) != 0) return -1.0;
#else
    if (mincore(a, len, (char *) vec.data()) != 0) return -1.0;
#endif
    size_t present = 0;
    for (unsigned char c : vec) present += c & 1;
    return vec.empty() ? 1.0 : (double) present / (double) vec.size();
#endif
}

// Bring a file range into memory and map it: [a, a+len) is its page-aligned mapping.
inline void fetch(int fd, uint64_t off, size_t bytes, uint8_t * a, size_t len, size_t page) {
#if defined(__linux__)
    (void) page;
    if (readahead(fd, (off64_t) off, bytes) != 0) {
        posix_fadvise(fd, (off_t) off, (off_t) bytes, POSIX_FADV_WILLNEED);
    }
    madvise(a, len, MADV_POPULATE_READ);
#else
    (void) fd; (void) off; (void) bytes;
#if defined(_WIN32)
    WIN32_MEMORY_RANGE_ENTRY entry{a, len};
    PrefetchVirtualMemory(GetCurrentProcess(), 1, &entry, 0);
#else
    madvise(a, len, MADV_WILLNEED);
#endif
    volatile uint8_t sink = 0;
    for (size_t o = 0; o < len; o += page) sink ^= a[o];  // fault the pages in
    (void) sink;
#endif
}

// Let the OS reclaim a file range: [a, a+len) is the page-aligned part fully inside it.
inline void drop(int fd, uint64_t off, size_t bytes, uint8_t * a, size_t len, bool pageout) {
#if defined(__linux__)
    if (len) madvise(a, len, pageout ? MADV_PAGEOUT : MADV_DONTNEED);
    posix_fadvise(fd, (off_t) off, (off_t) bytes, POSIX_FADV_DONTNEED);
#elif defined(_WIN32)
    (void) fd; (void) off; (void) bytes; (void) pageout;
    // on unlocked pages this removes them from the working set; the file cache keeps or
    // drops them as memory pressure dictates
    if (len) VirtualUnlock(a, len);
#else
    (void) fd; (void) off; (void) bytes; (void) pageout;
    if (len) madvise(a, len, MADV_DONTNEED);
#endif
}

struct IoCounters {
    uint64_t read_bytes = 0;
    long major_faults = 0;
    long minor_faults = 0;
};

inline IoCounters io_counters() {
    IoCounters c;
#if defined(_WIN32)
    IO_COUNTERS io{};
    if (GetProcessIoCounters(GetCurrentProcess(), &io)) c.read_bytes = io.ReadTransferCount;
    PROCESS_MEMORY_COUNTERS pmc{};
    if (GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof pmc)) c.major_faults = (long) pmc.PageFaultCount;
#else
    std::ifstream f("/proc/self/io");  // Linux only; zero elsewhere
    std::string k;
    uint64_t v;
    while (f >> k >> v) if (k == "read_bytes:") c.read_bytes = v;
    rusage ru{};
    getrusage(RUSAGE_SELF, &ru);
    c.major_faults = ru.ru_majflt;
    c.minor_faults = ru.ru_minflt;
#endif
    return c;
}

}  // namespace plat
