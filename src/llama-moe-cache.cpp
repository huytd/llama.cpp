#include "llama-moe-cache.h"

#include "llama-graph.h"
#include "llama-impl.h"
#include "llama-memory.h"
#include "llama-mmap.h"
#include "llama-model.h"

#include "ggml-cpp.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#ifdef __linux__
#include <fcntl.h>
#include <unistd.h>
#endif

namespace {

// ClassLRU: one LRU list per frequency class, over the experts of a group of layers
// slot_map[il][e] is the slot of expert e of layer il, n_slots or later if not cached, the host slot map tensor
// CRF (combined recency and frequency) = uses of an expert, decayed with a half-life of N tokens, retained across evictions
// class = floor(log2(CRF)), at most n_class - 1; each slot is in the LRU list of the class, as of its last use
// victim: an empty slot, or the least recently used slot of each list not in the current plan() with the max age/2^class
struct moe_cache_lru {
    static constexpr int   n_class  = 8; // floor(log2(1/(1 - 2^(-1/crf_half_life)))) = 7, class of CRF_max for half-life of 128
    static constexpr float crf_half_life = 128.0f; // tokens

    int32_t n_expert = 0;
    int32_t n_slots  = 0;
    int32_t empty    = 0; // slot of the experts that are not cached

    // [n_layer] data of the slot maps, the experts at slot n_slots or later are not cached
    // the experts computed on the CPU read zero slots after the cached ones on the device
    std::vector<int32_t *> slot_map;
    std::vector<int32_t>   key_of;   // [n_slots] key of the expert in the slot, -1 if empty

    // one LRU list per class, the head is the least recently used end
    std::vector<int32_t> prev;    // [n_slots] -1 at the ends
    std::vector<int32_t> next;    // [n_slots]
    int32_t head[n_class];
    int32_t tail[n_class];
    std::vector<int8_t>  list_of; // [n_slots] class list of the slot
    int32_t n_filled = 0;         // slots that hold an expert

    std::vector<float>   crf;      // [n_layer*n_expert]
    std::vector<int64_t> last_use; // [n_layer*n_expert] layer clock at the last use, -1 if not used
    std::vector<int64_t> clock;    // [n_layer] tokens seen by the layer (as added by plan() for each ubatch)

    std::vector<uint32_t> seen; // [n_expert] seen[e] == seen_gen if expert e is selected by the current plan()
    uint32_t seen_gen = 0;
    std::vector<int32_t> uniq;  // the distinct selected experts
    std::vector<int32_t> misses;
    std::vector<int32_t> evicted; // keys evicted by the last plan()
    std::vector<int32_t> evicted_slot; // their slots
    std::vector<uint64_t> token_zero; // [n_tokens] zero slots used by each token

    void init(int32_t n_layer, int32_t n_expert, int32_t n_slots) {
        GGML_ASSERT(n_class == (int) std::log2(1.0f/(1.0f - std::exp2(-1.0f/crf_half_life))) + 1);
        this->n_expert = n_expert;
        this->n_slots  = n_slots;
        this->empty    = n_slots;
        slot_map.assign(n_layer, nullptr);
        key_of.assign(n_slots, -1);
        prev.resize(n_slots);
        next.resize(n_slots);
        for (int32_t s = 0; s < n_slots; ++s) {
            prev[s] = s - 1;
            next[s] = s + 1 < n_slots ? s + 1 : -1;
        }
        for (int c = 0; c < n_class; ++c) {
            head[c] = -1;
            tail[c] = -1;
        }
        head[0] = 0;
        tail[0] = n_slots - 1;
        list_of.assign(n_slots, 0);
        crf.assign((size_t) n_layer*n_expert, 0.0f);
        last_use.assign((size_t) n_layer*n_expert, -1);
        clock.assign(n_layer, 0);
        seen.assign(n_expert, 0);
    }

    // move slot s to the most recently used end of list c
    void move(int32_t s, int c) {
        const int from = list_of[s];
        if (from == c && s == tail[from]) {
            return;
        }
        if (prev[s] >= 0) {
            next[prev[s]] = next[s];
        } else {
            head[from] = next[s];
        }
        if (next[s] >= 0) {
            prev[next[s]] = prev[s];
        } else {
            tail[from] = prev[s];
        }

        prev[s] = tail[c];
        next[s] = -1;
        if (tail[c] >= 0) {
            next[tail[c]] = s;
        } else {
            head[c] = s;
        }
        tail[c] = s;
        list_of[s] = c;
    }

    // key = index of an expert of a layer in crf and last_use
    int32_t make_key(int32_t il, int32_t e) const { return il*n_expert + e; }
    int32_t key_layer (int32_t key) const { return key / n_expert; }
    int32_t key_expert(int32_t key) const { return key % n_expert; }

    // for a key after use(): its CRF is >= 1, so log2 is >= 0 and the cast is the floor
    int class_of(int32_t key) const {
        return std::min(n_class - 1, (int) std::log2(crf[key]));
    }

    // the CRF of a key, decayed by the time since its last use
    float decayed_crf(int32_t il, int32_t key) const {
        return crf[key]*std::exp2(-(clock[il] - last_use[key])/crf_half_life);
    }

    // an unused key gets 1 because crf[key] is 0 prior to first use
    void use(int32_t il, int32_t key) {
        crf[key]      = decayed_crf(il, key) + 1.0f;
        last_use[key] = clock[il];
    }

    // the experts selected by this plan() must not be evicted, nor the slots this plan() filled already
    bool is_selected(int32_t il, int32_t key) const {
        if (key < 0) {
            return false; // unused slot
        }
        if (key_layer(key) != il) {
            return false; // seen is per expert index, an expert of another layer can have the same index
        }
        return seen[key_expert(key)] == seen_gen;
    }

    // the least recently used slot of list c that this plan() does not select, -1 if none
    int32_t candidate(int c, int32_t il) const {
        for (int32_t s = head[c]; s >= 0; s = next[s]) {
            if (!is_selected(il, key_of[s])) {
                return s;
            }
        }
        return -1;
    }

    int32_t victim(int32_t il) const {
        int32_t res = -1;
        double res_score = -1.0;
        for (int c = 0; c < n_class; ++c) {
            const int32_t s = candidate(c, il);
            if (s < 0) {
                continue;
            }
            const int32_t key = key_of[s];
            if (key < 0) {
                return s; // unused slot
            }
            const int64_t age   = clock[key_layer(key)] - last_use[key];
            const double  score = std::ldexp((double) age, -c); // age/2^class
            if (score > res_score) {
                res       = s;
                res_score = score;
            }
        }
        if (res < 0) {
            GGML_ABORT("no MoE cache victim");
        }
        return res;
    }

    struct fill {
        int32_t expert;
        int32_t slot;
    };

    // give a slot to each expert selected by ids (n_tokens tokens) in layer il
    // n_cpu(n) of the n misses are left to the CPU and read a zero slot on the device, they are not cached
    // a zero slot is used at most once per token, a miss that does not get one is cached instead
    // returns false if the ids select more distinct experts than there are slots
    template <typename F>
    bool plan(int32_t il, const int32_t * ids, size_t n_ids, int64_t n_tokens, int32_t n_used, int32_t n_zero, F && n_cpu,
            std::vector<fill> & fills, std::vector<fill> & cpu, size_t & n_hit) {
        fills.clear();
        evicted.clear();
        evicted_slot.clear();
        cpu.clear();
        n_hit = 0;

        if (++seen_gen == 0) {
            std::fill(seen.begin(), seen.end(), 0);
            seen_gen = 1;
        }
        uniq.clear();
        for (size_t i = 0; i < n_ids; ++i) {
            GGML_ASSERT(ids[i] >= 0 && ids[i] < n_expert);
            if (seen[ids[i]] != seen_gen) {
                seen[ids[i]] = seen_gen;
                uniq.push_back(ids[i]);
            }
        }
        if (uniq.size() > (size_t) n_slots) {
            return false;
        }

        int32_t * slots = slot_map[il];

        // update the CRF of all selected experts and promote hits before choosing victims for misses
        misses.clear();
        for (int32_t e : uniq) {
            const int32_t key = make_key(il, e);
            use(il, key);
            if (slots[e] < n_slots) {
                move(slots[e], class_of(key));
                n_hit++;
            } else {
                misses.push_back(e);
            }
        }

        // the misses with the lowest CRF are left to the CPU, a miss that keeps being used gains CRF and enters the cache
        // all misses are used at the same clock, so their CRF compare without decay
        const size_t n_c = std::min(n_cpu(misses.size()), misses.size());
        if (n_c > 0) {
            std::sort(misses.begin(), misses.end(), [&](int32_t a, int32_t b) {
                const float crf_a = crf[make_key(il, a)];
                const float crf_b = crf[make_key(il, b)];
                return crf_a != crf_b ? crf_a < crf_b : a < b;
            });
            GGML_ASSERT((size_t) n_tokens*n_used == n_ids);
            token_zero.assign(n_tokens, 0);
            size_t n_keep = n_c;
            for (size_t i = 0; i < n_keep; ) {
                const int32_t e = misses[i];
                uint64_t used = 0;
                for (int64_t it = 0; it < n_tokens; ++it) {
                    for (int32_t k = 0; k < n_used; ++k) {
                        if (ids[it*n_used + k] == e) {
                            used |= token_zero[it];
                        }
                    }
                }
                int32_t z = 0;
                while (z < 64 && (used >> z & 1)) {
                    z++;
                }
                if (z >= n_zero) {
                    std::swap(misses[i], misses[--n_keep]);
                    continue;
                }
                for (int64_t it = 0; it < n_tokens; ++it) {
                    for (int32_t k = 0; k < n_used; ++k) {
                        if (ids[it*n_used + k] == e) {
                            token_zero[it] |= 1ull << z;
                        }
                    }
                }
                slots[e] = n_slots + z;
                cpu.push_back({ e, n_slots + z });
                i++;
            }
            misses.erase(misses.begin(), misses.begin() + n_keep);
        }

        // fill the misses in expert order: victim() takes unused slots, and slots filled by one plan(), in list order,
        // so consecutive experts often get consecutive slots and one upload
        std::sort(misses.begin(), misses.end());
        for (int32_t e : misses) {
            const int32_t s = victim(il);
            if (key_of[s] >= 0) {
                slot_map[key_layer(key_of[s])][key_expert(key_of[s])] = empty;
                evicted.push_back(key_of[s]);
                evicted_slot.push_back(s);
            } else {
                n_filled++;
            }
            const int32_t key = make_key(il, e);
            key_of[s] = key;
            slots[e]  = s;
            move(s, class_of(key));
            fills.push_back({ e, s });
        }
        clock[il] += n_tokens;
        return true;
    }
};

// gate, up, down or gate_up, down
static std::vector<ggml_tensor *> llama_moe_cache_layer_experts(const llama_layer & layer) {
    std::vector<ggml_tensor *> res;
    for (ggml_tensor * t : { layer.ffn_gate_up_exps, layer.ffn_gate_exps, layer.ffn_up_exps, layer.ffn_down_exps }) {
        if (t != nullptr) {
            res.push_back(t);
        }
    }
    return res;
}

static bool llama_moe_cache_same_layout(const std::vector<ggml_tensor *> & a, const std::vector<ggml_tensor *> & b) {
    if (a.size() != b.size()) {
        return false;
    }
    for (size_t i = 0; i < a.size(); ++i) {
        if (a[i]->type != b[i]->type || !ggml_are_same_shape(a[i], b[i]) || a[i]->nb[2] != b[i]->nb[2]) {
            return false;
        }
    }
    return true;
}

static bool llama_moe_cache_is_host_weight(const ggml_tensor * t) {
    return t->buffer != nullptr &&
        ggml_backend_buffer_get_usage(t->buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS &&
        ggml_backend_buffer_is_host(t->buffer);
}

}

// host pool of experts evicted from the device slots, exclusive with them: a pool hit moves the expert back to the device
// LRU over pool slots, the head is the least recently used end
struct moe_cache_pool {
    int32_t n = 0;
    std::vector<int32_t> slot_of; // [n_layer*n_expert] pool slot of a key, -1 if not in the pool
    std::vector<int32_t> key_of;  // [n] -1 if free
    std::vector<int32_t> prev;
    std::vector<int32_t> next;
    int32_t head = -1;
    int32_t tail = -1;
    std::vector<int32_t> free_slots;

    void init(int32_t n_keys, int32_t n_slots) {
        n = n_slots;
        slot_of.assign(n_keys, -1);
        key_of.assign(n, -1);
        prev.assign(n, -1);
        next.assign(n, -1);
        free_slots.clear();
        for (int32_t s = n - 1; s >= 0; --s) {
            free_slots.push_back(s);
        }
    }

    void unlink(int32_t s) {
        if (prev[s] >= 0) { next[prev[s]] = next[s]; } else { head = next[s]; }
        if (next[s] >= 0) { prev[next[s]] = prev[s]; } else { tail = prev[s]; }
        prev[s] = next[s] = -1;
    }

    void push_tail(int32_t s) {
        prev[s] = tail;
        next[s] = -1;
        if (tail >= 0) { next[tail] = s; } else { head = s; }
        tail = s;
    }

    // take the slot of key out of the pool, the caller frees it with release() once the upload from it is queued
    int32_t take(int32_t key) {
        const int32_t s = slot_of[key];
        if (s >= 0) {
            unlink(s);
            slot_of[key] = -1;
            key_of[s]    = -1;
        }
        return s;
    }

    void release(int32_t s) { free_slots.push_back(s); }

    // a slot for key, a free one or the least recently used one
    int32_t insert(int32_t key) {
        int32_t s;
        if (!free_slots.empty()) {
            s = free_slots.back();
            free_slots.pop_back();
        } else {
            s = head;
            if (s < 0) {
                return -1;
            }
            unlink(s);
            slot_of[key_of[s]] = -1;
        }
        key_of[s]    = key;
        slot_of[key] = s;
        push_tail(s);
        return s;
    }
};

struct llama_moe_cache::impl {
    // a GPU with its own budget and banks, it caches the layers assigned to it
    struct device {
        ggml_backend_t backend;
        ggml_backend_buffer_type_t buft;
        size_t host_bytes = 0; // host experts of the layers it caches
        double split = 0.0;    // share of the budget

        // banks and their views
        ggml_context_ptr ctx;
        ggml_backend_buffer_ptr buf;
        size_t buf_size = 0;

        // pool overlap: second stream for the D2H of evicted experts, so it runs at the same time as the uploads
        ggml_backend_ptr       backend_d2h;
        ggml_backend_event_ptr ev_stage; // on backend: evicted experts are in the stage slots
        ggml_backend_event_ptr ev_d2h;   // on backend_d2h: the last D2H is done
        bool d2h_pending = false;
        ggml_context_ptr ctx_stage;
        ggml_backend_buffer_ptr buf_stage;
    };

    // layers of the same device with the same expert tensor layout share the banks and the LRU of a group
    struct group {
        int32_t id;                       // device
        std::vector<ggml_tensor *> ref;   // expert tensors of the first layer
        std::vector<int32_t> layers;
        std::vector<ggml_tensor *> banks; // device storage of all slots, one per expert tensor
        size_t host_bytes = 0;
        int32_t n_slots = 0;
        moe_cache_lru lru;

        moe_cache_pool pool;
        std::vector<ggml_tensor *> pool_banks; // host storage of the pool slots, one per expert tensor
        std::vector<ggml_tensor *> stage_banks; // device copies of evicted experts, read by the D2H, one per expert tensor
        int32_t n_stage = 0;
    };

    struct layer {
        int32_t ig = -1;                    // -1 if the layer is not cached
        ggml_tensor * slot_map = nullptr;   // I32 [1, n_expert] in host memory
        ggml_tensor * slot_map_cpu = nullptr; // looked up in place of slot_map by the graphs with a CPU part, if cpu_share > 0, its data is not used
        ggml_tensor * cpu_map  = nullptr;   // I32 [1, n_expert] in host memory, if cpu_share > 0
        std::vector<int32_t> cpu_experts;   // experts set in cpu_map
        uint32_t cpu_acc = 0;               // fraction of a miss carried over to the next plan, in 1/65536
        std::vector<ggml_tensor *> experts; // host expert tensors, in the order of the banks
    };

    struct binding {
        int32_t il;
        int32_t ip;           // index of the bank
        ggml_tensor * cached; // view of the bank used in place of the host experts
    };

    struct stats {
        size_t hits   = 0;
        size_t misses = 0;
        size_t cpu    = 0; // misses computed on the CPU
        size_t bytes  = 0;
        size_t pool_hits = 0;
        size_t pool_put  = 0;
    };

    // LLAMA_MOE_CACHE_DDIO=n: n threads read the misses not in the pool with O_DIRECT, in chunks, into pinned stage slots, then upload from there
    // up to ddio_slots misses per layer, the rest go through the mmap
    struct ddio_job {
        int fd;
        off_t off;  // aligned
        size_t len; // aligned
        size_t need; // fail if fewer bytes are read
        uint8_t * dst;
        std::atomic<int32_t> * left;
        std::atomic<int32_t> * fail;
    };
    struct ddio_src {
        int fd;
        off_t off; // file offset of the tensor data
    };
    static constexpr size_t  ddio_chunk = 256*1024;
    static constexpr int32_t ddio_slots = 64;
    int32_t ddio        = 0;
    size_t  ddio_stride = 0; // bytes per expert tensor in a stage slot
    size_t  ddio_nt     = 0; // expert tensors per stage slot
    ggml_backend_buffer_ptr buf_ddio;
    uint8_t * ddio_base = nullptr;
    std::unique_ptr<std::atomic<int32_t>[]> ddio_left; // [ddio_slots] chunks in flight
    std::unique_ptr<std::atomic<int32_t>[]> ddio_fail; // [ddio_slots]
    std::vector<int32_t> ddio_slot_of; // [fills] stage slot an upload reads, -1 if none
    std::unordered_map<const ggml_tensor *, ddio_src> ddio_srcs;
    std::vector<int> ddio_fds;
    std::vector<std::thread> ddio_threads;
    std::mutex ddio_mutex;
    std::condition_variable ddio_cv;
    std::deque<ddio_job> ddio_queue;
    std::vector<ddio_job> ddio_jobs;
    bool   ddio_stop = false;
    size_t ddio_max_len = 0;
    size_t stats_ddio_read = 0;
    size_t stats_ddio_fail = 0;

    void ddio_run() {
        std::unique_lock<std::mutex> lock(ddio_mutex);
        while (true) {
            ddio_cv.wait(lock, [&] { return ddio_stop || !ddio_queue.empty(); });
            if (ddio_stop) {
                return;
            }
            const ddio_job j = ddio_queue.front();
            ddio_queue.pop_front();
            lock.unlock();
            size_t done = 0;
#ifdef __linux__
            while (done < j.len) {
                const ssize_t r = pread(j.fd, j.dst + done, j.len - done, j.off + done);
                if (r <= 0) {
                    break;
                }
                done += r;
            }
#endif
            if (done < j.need) {
                j.fail->store(1, std::memory_order_relaxed);
            }
            j.left->fetch_sub(1, std::memory_order_release);
            lock.lock();
        }
    }

    // find the file and offset of the expert tensors, from /proc/self/maps
    void ddio_init() {
#ifdef __linux__
        struct range {
            uintptr_t beg;
            uintptr_t end;
            off_t off;
            std::string path;
        };
        std::vector<range> ranges;
        FILE * f = fopen("/proc/self/maps", "r");
        if (f == nullptr) {
            return;
        }
        char line[4096];
        while (fgets(line, sizeof(line), f)) {
            unsigned long beg;
            unsigned long end;
            unsigned long off;
            char path[4096] = {};
            if (sscanf(line, "%lx-%lx %*s %lx %*s %*s %4095[^\n]", &beg, &end, &off, path) == 4 && path[0] == '/') {
                ranges.push_back({ beg, end, (off_t) off, path });
            }
        }
        fclose(f);
        std::unordered_map<std::string, int> fds;
        for (const layer & l : layers) {
            if (l.ig < 0) {
                continue;
            }
            for (const ggml_tensor * t : l.experts) {
                const uintptr_t p = (uintptr_t) t->data;
                const auto it = std::find_if(ranges.begin(), ranges.end(), [&](const range & r) { return p >= r.beg && p + ggml_nbytes(t) <= r.end; });
                if (it == ranges.end()) {
                    LLAMA_LOG_WARN("%s: %s is not in a file mapping, ddio is disabled\n", __func__, t->name);
                    ddio_srcs.clear();
                    return;
                }
                auto it_fd = fds.find(it->path);
                if (it_fd == fds.end()) {
                    const int fd = open(it->path.c_str(), O_RDONLY | O_DIRECT);
                    if (fd < 0) {
                        LLAMA_LOG_WARN("%s: failed to open %s with O_DIRECT, ddio is disabled\n", __func__, it->path.c_str());
                        ddio_srcs.clear();
                        return;
                    }
                    ddio_fds.push_back(fd);
                    it_fd = fds.emplace(it->path, fd).first;
                }
                ddio_srcs[t] = { it_fd->second, it->off + (off_t) (p - it->beg) };
                ddio_max_len = std::max(ddio_max_len, (size_t) t->nb[2]);
            }
        }
#endif
    }

    static constexpr int64_t max_batch = 32;

    int32_t n_expert_used;

    stats stats_small; // up to 8 tokens per ubatch
    stats stats_large;
    stats stats_copy;  // experts copied from the cache for large batches

    std::vector<device> devices;
    std::vector<group> groups;
    std::vector<layer> layers;
    std::unordered_map<const ggml_tensor *, binding> bindings; // host experts -> cached experts
    std::unordered_map<const ggml_tensor *, std::pair<int32_t, bool>> layer_of; // slot map -> layer, and if the graph that looks it up has a CPU part

    std::vector<int32_t> ids;
    std::vector<moe_cache_lru::fill> fills;
    std::vector<moe_cache_lru::fill> cpu;

    float    cpu_share = 0.0f;
    uint32_t cpu_share_q = 0; // cpu_share in 1/65536, integer so that a simulator can reproduce the rounding
    int32_t  n_zero    = 0; // zero slots per group

    // LLAMA_MOE_CACHE_TRACE: the selected experts of each slot lookup, for a simulator of the cache
    FILE * trace = nullptr;

    // slot maps in host memory
    ggml_context_ptr ctx_host;
    ggml_backend_buffer_ptr buf_host;
    size_t buf_host_size = 0;

    // LLAMA_MOE_CACHE_POOL_MIB: size of the host pool of evicted experts, in pinned memory if the device has a host buffer type
    size_t pool_size = 0;
    ggml_context_ptr ctx_pool;
    ggml_backend_buffer_ptr buf_pool;
    std::vector<int32_t> pool_src; // [fills] pool slot an upload reads, -1 if from the host experts
    std::vector<std::pair<int32_t, int32_t>> staged; // (stage slot, pool slot) of the evicted experts copied by backend_d2h

    // views used by copy_experts
    ggml_context_ptr ctx_views;

    impl(const llama_model & model, bool no_alloc, const std::vector<ggml_backend_t> & backends, const std::vector<ggml_backend_buffer_type_t> & bufts, size_t size, float cpu_share) :
            n_expert_used(model.hparams.n_expert_used_max()), layers(model.layers.size()), cpu_share(std::clamp(cpu_share, 0.0f, 1.0f)) {
        for (size_t i = 0; i < backends.size(); ++i) {
            const auto dev_type = ggml_backend_dev_type(ggml_backend_get_device(backends[i]));
            if (dev_type == GGML_BACKEND_DEVICE_TYPE_GPU || dev_type == GGML_BACKEND_DEVICE_TYPE_IGPU) {
                auto & d = devices.emplace_back();
                d.backend = backends[i];
                d.buft    = bufts[i];
            }
        }
        if (devices.empty()) {
            throw std::runtime_error("MoE cache requires a GPU backend");
        }
        if (model.split_mode() == LLAMA_SPLIT_MODE_TENSOR) {
            throw std::runtime_error("MoE cache does not support tensor parallelism");
        }
        if (model.hparams.n_expert == 0 || n_expert_used == 0) {
            throw std::runtime_error("MoE cache requires a MoE model");
        }

        // only cache layers that keep all of their experts in host memory, on the device the layer is assigned to
        for (size_t il = 0; il < model.layers.size(); ++il) {
            auto experts = llama_moe_cache_layer_experts(model.layers[il]);
            if (experts.empty() || !std::all_of(experts.begin(), experts.end(), llama_moe_cache_is_host_weight)) {
                continue;
            }
            const auto it_dev = std::find_if(devices.begin(), devices.end(), [&](const device & d) { return ggml_backend_get_device(d.backend) == model.dev_layer(il); });
            if (it_dev == devices.end()) {
                continue;
            }
            const int32_t id = (int32_t) (it_dev - devices.begin());
            auto it = std::find_if(groups.begin(), groups.end(), [&](const group & g) { return g.id == id && llama_moe_cache_same_layout(g.ref, experts); });
            if (it == groups.end()) {
                groups.emplace_back();
                it = groups.end() - 1;
                it->id  = id;
                it->ref = experts;
            }
            it->layers.push_back(il);
            for (const ggml_tensor * t : experts) {
                it->host_bytes         += ggml_nbytes(t);
                devices[id].host_bytes += ggml_nbytes(t);
            }
        }
        if (groups.empty()) {
            LLAMA_LOG_WARN("%s: no layer has all of its experts in host memory, MoE cache is disabled\n", __func__);
            return;
        }

        // with misses on the CPU, the n_zero slots after the cached ones stay zero: the device computes these misses as zero
        // one extra slot at the end, CUDA MMQ can read past the last expert
        n_zero = cpu_share > 0.0f ? std::min<int32_t>(n_expert_used, 64) : 0;
        auto alloc_size = [&](const group & g, int32_t n_slots) {
            const size_t alignment = ggml_backend_buft_get_alignment(devices[g.id].buft);
            size_t res = 0;
            for (const ggml_tensor * t : g.ref) {
                res += GGML_PAD(t->nb[2]*(n_slots + n_zero + 1), alignment);
            }
            return res;
        };

        // the budget is split among the devices with host experts like the layers, by the tensor split or by default by free memory
        const float * tensor_split = model.tensor_split();
        const bool split_by_free = tensor_split == nullptr ||
            std::all_of(tensor_split, tensor_split + model.n_devices(), [](float x) { return x == 0.0f; });
        double split_sum = 0.0;
        for (device & d : devices) {
            if (d.host_bytes == 0) {
                continue;
            }
            ggml_backend_dev_t dev = ggml_backend_get_device(d.backend);
            if (split_by_free) {
                size_t free;
                size_t total;
                ggml_backend_dev_memory(dev, &free, &total);
                d.split = (double) free;
            } else {
                const auto it = std::find_if(model.devices.begin(), model.devices.end(), [&](const llama_device & ld) { return ld.dev == dev; });
                GGML_ASSERT(it != model.devices.end());
                d.split = (double) tensor_split[it - model.devices.begin()];
            }
            split_sum += d.split;
        }
        if (split_sum == 0.0) {
            // the devices do not report their free memory
            for (device & d : devices) {
                d.split    = d.host_bytes > 0 ? 1.0 : 0.0;
                split_sum += d.split;
            }
        }

        // within a device the budget is split by the size of the experts, so each group caches the same fraction of its experts
        cpu_share_q = (uint32_t) std::lround(cpu_share*65536.0f);
        if (const char * s = getenv("LLAMA_MOE_CACHE_DDIO"); s != nullptr) {
            ddio = std::max(0, atoi(s));
        }
        if (const char * path = getenv("LLAMA_MOE_CACHE_TRACE"); path != nullptr && *path != '\0') {
            trace = fopen(path, "wb");
            if (trace == nullptr) {
                throw std::runtime_error(std::string("failed to open the MoE cache trace ") + path);
            }
            write_trace_header(split_sum);
        }
        std::vector<size_t> n_tensors(devices.size(), 0);
        size_t n_tensors_host = 0;
        for (group & g : groups) {
            const device & d = devices[g.id];
            const int32_t n_expert  = g.ref[0]->ne[2];
            const size_t  budget    = (size_t) ((double) size*d.split/split_sum*g.host_bytes/d.host_bytes);
            const int32_t max_slots = g.layers.size()*n_expert;
            while (g.n_slots < max_slots && alloc_size(g, g.n_slots + 1) <= budget) {
                g.n_slots++;
            }
            if (g.n_slots < n_expert_used) {
                LLAMA_LOG_WARN("%s: MoE cache budget is too small for %zu layers, they are not cached\n", __func__, g.layers.size());
                g.n_slots = 0;
                continue;
            }
            g.lru.init(model.layers.size(), n_expert, g.n_slots);
            n_tensors[g.id] += g.ref.size()*(1 + g.layers.size());
            n_tensors_host  += g.layers.size()*(n_zero > 0 ? 3 : 1); // slot maps and CPU map
        }
        if (n_tensors_host == 0) {
            throw std::runtime_error("MoE cache is too small to hold the experts of one token");
        }

        auto init_ctx = [](size_t n_tensors) {
            ggml_init_params params = {
                /*.mem_size   =*/ n_tensors*ggml_tensor_overhead(),
                /*.mem_buffer =*/ nullptr,
                /*.no_alloc   =*/ true,
            };
            ggml_context_ptr res(ggml_init(params));
            if (!res) {
                throw std::runtime_error("failed to create the MoE cache context");
            }
            return res;
        };
        for (size_t id = 0; id < devices.size(); ++id) {
            if (n_tensors[id] > 0) {
                devices[id].ctx = init_ctx(n_tensors[id]);
            }
        }
        ctx_host  = init_ctx(n_tensors_host);
        ctx_views = init_ctx(2);

        ggml_backend_buffer_type_t buft_host = ggml_backend_cpu_buffer_type();
        const size_t alignment_host = ggml_backend_buft_get_alignment(buft_host);

        for (size_t ig = 0; ig < groups.size(); ++ig) {
            group & g = groups[ig];
            if (g.n_slots == 0) {
                continue;
            }
            ggml_context * ctx = devices[g.id].ctx.get();
            for (const ggml_tensor * t : g.ref) {
                ggml_tensor * bank = ggml_new_tensor_3d(ctx, t->type, t->ne[0], t->ne[1], g.n_slots + n_zero + 1);
                GGML_ASSERT(bank->nb[2] == t->nb[2]);
                ggml_format_name(bank, "moe_cache.%zu.%s", ig, t->name);
                g.banks.push_back(bank);
            }
            for (int32_t il : g.layers) {
                layer & l = layers[il];
                l.ig      = (int32_t) ig;
                l.experts = llama_moe_cache_layer_experts(model.layers[il]);
                for (size_t ip = 0; ip < l.experts.size(); ++ip) {
                    ggml_tensor * bank   = g.banks[ip];
                    ggml_tensor * cached = ggml_view_3d(ctx, bank, bank->ne[0], bank->ne[1], g.n_slots + n_zero, bank->nb[1], bank->nb[2], 0);
                    ggml_format_name(cached, "moe_cache.%s", l.experts[ip]->name);
                    bindings[l.experts[ip]] = { il, (int32_t) ip, cached };
                }
                l.slot_map = ggml_new_tensor_2d(ctx_host.get(), GGML_TYPE_I32, 1, g.ref[0]->ne[2]);
                ggml_format_name(l.slot_map, "moe_cache.slot_map-%d", il);
                layer_of[l.slot_map] = { il, false };
                buf_host_size += GGML_PAD(ggml_nbytes(l.slot_map), alignment_host);
                if (n_zero > 0) {
                    l.slot_map_cpu = ggml_new_tensor_2d(ctx_host.get(), GGML_TYPE_I32, 1, g.ref[0]->ne[2]);
                    ggml_format_name(l.slot_map_cpu, "moe_cache.slot_map_cpu-%d", il);
                    layer_of[l.slot_map_cpu] = { il, true };
                    buf_host_size += GGML_PAD(ggml_nbytes(l.slot_map_cpu), alignment_host);
                    l.cpu_map = ggml_new_tensor_2d(ctx_host.get(), GGML_TYPE_I32, 1, g.ref[0]->ne[2]);
                    ggml_format_name(l.cpu_map, "moe_cache.cpu_map-%d", il);
                    buf_host_size += GGML_PAD(ggml_nbytes(l.cpu_map), alignment_host);
                }
            }
            devices[g.id].buf_size += alloc_size(g, g.n_slots);
        }

        // with no_alloc, dummy buffers are attached so the memory use can still be measured, see llama_context::memory_breakdown
        for (device & d : devices) {
            if (!d.ctx) {
                continue;
            }
            d.buf = llama_memory_bufs::alloc_buf(d.ctx.get(), d.buft, no_alloc, "MoE cache");
            if (!no_alloc) {
                ggml_backend_buffer_clear(d.buf.get(), 0);
                d.buf_size = ggml_backend_buffer_get_size(d.buf.get());
            }
        }
        buf_host = llama_memory_bufs::alloc_buf(ctx_host.get(), buft_host, no_alloc, "MoE cache slot maps");
        if (!no_alloc) {
            ggml_backend_buffer_clear(buf_host.get(), 0xff); // the CPU maps are -1
            buf_host_size = ggml_backend_buffer_get_size(buf_host.get());

            for (group & g : groups) {
                for (int32_t il : g.layers) {
                    if (layers[il].slot_map != nullptr) {
                        int32_t * slots = (int32_t *) layers[il].slot_map->data;
                        std::fill(slots, slots + layers[il].slot_map->ne[1], g.lru.empty);
                        g.lru.slot_map[il] = slots;
                    }
                }
            }
        }

        if (const char * e = getenv("LLAMA_MOE_CACHE_POOL_MIB"); e != nullptr && !no_alloc) {
            pool_size = (size_t) std::max(0, atoi(e))*1024*1024;
        }
        if (pool_size > 0) {
            size_t host_total = 0;
            for (const group & g : groups) {
                host_total += g.n_slots > 0 ? g.host_bytes : 0;
            }
            ctx_pool = init_ctx(groups.size()*4);
            for (size_t ig = 0; ig < groups.size(); ++ig) {
                group & g = groups[ig];
                if (g.n_slots == 0) {
                    continue;
                }
                size_t expert_bytes = 0;
                for (const ggml_tensor * t : g.ref) {
                    expert_bytes += t->nb[2];
                }
                const int32_t n_expert = g.ref[0]->ne[2];
                const int32_t max_pool = g.layers.size()*n_expert - g.n_slots;
                const int32_t n_pool = std::min<int64_t>(max_pool, (int64_t) ((double) pool_size*g.host_bytes/host_total/expert_bytes));
                if (n_pool <= 0) {
                    continue;
                }
                g.pool.init(model.layers.size()*n_expert, n_pool);
                for (const ggml_tensor * t : g.ref) {
                    ggml_tensor * pb = ggml_new_tensor_3d(ctx_pool.get(), t->type, t->ne[0], t->ne[1], n_pool);
                    ggml_format_name(pb, "moe_cache_pool.%zu.%s", ig, t->name);
                    g.pool_banks.push_back(pb);
                }
            }
            ggml_backend_dev_t dev = ggml_backend_get_device(devices[0].backend);
            ggml_backend_buffer_type_t buft_pool = ggml_backend_dev_host_buffer_type(dev);
            if (buft_pool == nullptr) {
                buft_pool = ggml_backend_cpu_buffer_type();
            }
            ggml_backend_buffer_t b = ggml_backend_alloc_ctx_tensors_from_buft(ctx_pool.get(), buft_pool);
            if (b == nullptr) {
                throw std::runtime_error("failed to allocate the MoE cache pool");
            }
            buf_pool.reset(b);
            LLAMA_LOG_INFO("%s: %10s MoE cache pool size = %8.2f MiB\n", __func__, ggml_backend_buft_name(buft_pool), ggml_backend_buffer_get_size(b)/1024.0/1024.0);
            for (const group & g : groups) {
                if (g.pool.n > 0) {
                    LLAMA_LOG_INFO("%s: %2zu layers: %5d pool slots\n", __func__, g.layers.size(), g.pool.n);
                }
            }

            // evicted experts go D2D to stage slots, then D2H on a second stream, so the D2H runs at the same time as the uploads
            const int32_t n_stage = 16;
            for (group & g : groups) {
                if (g.pool.n == 0) {
                    continue;
                }
                device & d = devices[g.id];
                ggml_backend_dev_t dev_g = ggml_backend_get_device(d.backend);
                if (!d.backend_d2h) {
                    ggml_backend_dev_props props;
                    ggml_backend_dev_get_props(dev_g, &props);
                    if (!props.caps.events) {
                        continue;
                    }
                    d.backend_d2h.reset(ggml_backend_dev_init(dev_g, nullptr));
                    d.ev_stage.reset(ggml_backend_event_new(dev_g));
                    d.ev_d2h.reset(ggml_backend_event_new(dev_g));
                    d.ctx_stage = init_ctx(groups.size()*4);
                    if (!d.backend_d2h || !d.ev_stage || !d.ev_d2h) {
                        throw std::runtime_error("failed to init the MoE cache pool stream");
                    }
                }
                g.n_stage = n_stage;
                for (const ggml_tensor * t : g.ref) {
                    ggml_tensor * sb = ggml_new_tensor_3d(d.ctx_stage.get(), t->type, t->ne[0], t->ne[1], n_stage);
                    ggml_format_name(sb, "moe_cache_stage.%d.%s", g.id, t->name);
                    g.stage_banks.push_back(sb);
                }
            }
            for (device & d : devices) {
                if (!d.backend_d2h) {
                    continue;
                }
                ggml_backend_buffer_t bs = ggml_backend_alloc_ctx_tensors_from_buft(d.ctx_stage.get(), d.buft);
                if (bs == nullptr) {
                    throw std::runtime_error("failed to allocate the MoE cache stage slots");
                }
                d.buf_stage.reset(bs);
                LLAMA_LOG_INFO("%s: %10s MoE cache stage size = %8.2f MiB (%d slots per group)\n", __func__, ggml_backend_buft_name(d.buft), ggml_backend_buffer_get_size(bs)/1024.0/1024.0, n_stage);
            }
        }

        if (ddio > 0 && !no_alloc) {
            ddio_init();
        }
        if (ddio > 0 && !ddio_srcs.empty()) {
            for (const group & g : groups) {
                ddio_nt = std::max(ddio_nt, g.ref.size());
            }
            ddio_stride = GGML_PAD(ddio_max_len, 4096) + 4096;
            ggml_backend_buffer_type_t buft_ddio = ggml_backend_dev_host_buffer_type(ggml_backend_get_device(devices[0].backend));
            if (buft_ddio == nullptr) {
                buft_ddio = ggml_backend_cpu_buffer_type();
            }
            buf_ddio.reset(ggml_backend_buft_alloc_buffer(buft_ddio, ddio_slots*ddio_nt*ddio_stride + 4096));
            if (!buf_ddio) {
                throw std::runtime_error("failed to allocate the MoE cache ddio stage slots");
            }
            ddio_base = (uint8_t *) GGML_PAD((uintptr_t) ggml_backend_buffer_get_base(buf_ddio.get()), 4096);
            ddio_left.reset(new std::atomic<int32_t>[ddio_slots]);
            ddio_fail.reset(new std::atomic<int32_t>[ddio_slots]);
            for (int32_t i = 0; i < ddio_slots; ++i) {
                ddio_left[i].store(0);
                ddio_fail[i].store(0);
            }
            for (int32_t i = 0; i < ddio; ++i) {
                ddio_threads.emplace_back([this] { ddio_run(); });
            }
            LLAMA_LOG_INFO("%s: ddio: %d reader threads, %d slots, %.2f MiB pinned\n", __func__,
                ddio, ddio_slots, ggml_backend_buffer_get_size(buf_ddio.get())/1024.0/1024.0);
        } else {
            ddio = 0;
        }

        // as weights, the ops that read the banks run on the device and the slot maps are copied with the copy callback
        for (device & d : devices) {
            if (d.buf) {
                ggml_backend_buffer_set_usage(d.buf.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
            }
        }
        ggml_backend_buffer_set_usage(buf_host.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

        for (size_t id = 0; id < devices.size(); ++id) {
            const device & d = devices[id];
            if (d.host_bytes == 0) {
                continue;
            }
            LLAMA_LOG_INFO("%s: %10s MoE cache size = %8.2f MiB for %.2f MiB of host experts, %.0f%% of the misses on the CPU\n", __func__,
                ggml_backend_buft_name(d.buft), d.buf_size/1024.0/1024.0, d.host_bytes/1024.0/1024.0, 100.0*cpu_share);
            for (const group & g : groups) {
                if (g.id == (int32_t) id) {
                    LLAMA_LOG_INFO("%s: %2zu layers, %s: %5d slots (%.1f%%, %.1f tokens worth)\n", __func__,
                        g.layers.size(), ggml_type_name(g.ref.back()->type), g.n_slots, 100.0*g.n_slots/(g.layers.size()*g.ref[0]->ne[2]),
                        (double) g.n_slots/(g.layers.size()*n_expert_used));
                }
            }
        }
    }

    // the device that caches layer il
    ggml_backend_t backend(int32_t il) const {
        return devices[groups[layers[il].ig].id].backend;
    }

    // trace format, little endian: "MCT1", then u32 n_layer, n_expert_used, n_group
    // per group: u32 device, n_expert, n_layers, the layers (u32), f64 split, split_sum, u64 group host bytes, device host bytes, alignment,
    //            u32 n_tensors, the bytes of one expert of each tensor (u64)
    // then one record per slot lookup: i32 layer, experts per token, n_ids, the ids (i32)
    void write_trace_header(double split_sum) {
        auto u32 = [&](uint32_t v) { fwrite(&v, sizeof(v), 1, trace); };
        auto u64 = [&](uint64_t v) { fwrite(&v, sizeof(v), 1, trace); };
        auto f64 = [&](double   v) { fwrite(&v, sizeof(v), 1, trace); };
        fwrite("MCT1", 1, 4, trace);
        u32(layers.size());
        u32(n_expert_used);
        u32(groups.size());
        for (const group & g : groups) {
            const device & d = devices[g.id];
            u32(g.id);
            u32(g.ref[0]->ne[2]);
            u32(g.layers.size());
            for (int32_t il : g.layers) {
                u32(il);
            }
            f64(d.split);
            f64(split_sum);
            u64(g.host_bytes);
            u64(d.host_bytes);
            u64(ggml_backend_buft_get_alignment(d.buft));
            u32(g.ref.size());
            for (const ggml_tensor * t : g.ref) {
                u64(t->nb[2]);
            }
        }
    }

    void write_trace(int32_t il, int32_t n_used) {
        const int32_t rec[3] = { il, n_used, (int32_t) ids.size() };
        fwrite(rec, sizeof(int32_t), 3, trace);
        fwrite(ids.data(), sizeof(int32_t), ids.size(), trace);
    }

    ~impl() {
        for (device & d : devices) {
            if (d.backend_d2h) {
                ggml_backend_synchronize(d.backend_d2h.get());
            }
        }
        {
            std::lock_guard<std::mutex> lock(ddio_mutex);
            ddio_stop = true;
        }
        ddio_cv.notify_all();
        for (auto & t : ddio_threads) {
            t.join();
        }
#ifdef __linux__
        for (int fd : ddio_fds) {
            close(fd);
        }
#endif
        log_stats();
        if (trace != nullptr) {
            fclose(trace);
        }
    }

    // the slot map of layer il, if its experts can be read from the cache for n_tokens tokens, nullptr otherwise
    // cpu_part: the graph computes the misses in the CPU map of the layer (see get_cpu_map)
    ggml_tensor * get_slot_map(int32_t il, int64_t n_tokens, int64_t n_expert_used, bool cpu_part) const {
        if (il < 0 || il >= (int32_t) layers.size() || layers[il].ig < 0) {
            return nullptr;
        }
        const layer & l = layers[il];

        // large batches use most experts of a layer, so they gain little from the cache and would evict the experts used in generation
        if (n_tokens == 0 || n_tokens > max_batch || std::min(n_tokens*n_expert_used, l.slot_map->ne[1]) > groups[l.ig].n_slots) {
            return nullptr;
        }
        GGML_ASSERT(!cpu_part || l.slot_map_cpu != nullptr);
        return cpu_part ? l.slot_map_cpu : l.slot_map;
    }

    // the experts of layer il computed on the CPU: I32 [1, n_expert] in host memory, e for these experts, -1 for the others
    // nullptr if no misses of layer il are computed on the CPU for ubatches of n_tokens tokens
    ggml_tensor * get_cpu_map(int32_t il, int64_t n_tokens) const {
        if (il < 0 || il >= (int32_t) layers.size() || layers[il].cpu_map == nullptr) {
            return nullptr;
        }
        // the scheduler would run the CPU part on a device that offloads such batches from host weights
        ggml_tensor probe = {};
        probe.op    = GGML_OP_MUL_MAT_ID;
        probe.type  = GGML_TYPE_F32;
        probe.ne[0] = 1;
        probe.ne[1] = n_expert_used;
        probe.ne[2] = n_tokens;
        probe.ne[3] = 1;
        for (const device & d : devices) {
            if (ggml_backend_offload_op(d.backend, &probe)) {
                return nullptr;
            }
        }
        return layers[il].cpu_map;
    }

    ggml_tensor * get_experts(const ggml_tensor * w) const {
        const auto it = bindings.find(w);
        return it != bindings.end() ? it->second.cached : nullptr;
    }

    int64_t copy_experts(ggml_backend_t backend, const ggml_tensor * w, ggml_tensor * dst, int64_t e, int64_t last) {
        const auto it = bindings.find(w);
        if (it == bindings.end()) {
            return 0;
        }
        const binding & b = it->second;
        const group & g = groups[layers[b.il].ig];
        if (backend != devices[g.id].backend) {
            return 0;
        }

        // large batches only read the cache, so the experts used in generation stay in it
        const int32_t * slots = g.lru.slot_map[b.il];
        if (slots == nullptr || slots[e] >= g.n_slots) {
            return 0;
        }
        int64_t n = 1;
        while (e + n <= last && slots[e + n] < g.n_slots && slots[e + n] == slots[e] + n) {
            n++;
        }

        ggml_tensor * bank = g.banks[b.ip];
        ggml_reset(ctx_views.get());
        ggml_tensor * src_view = ggml_view_3d(ctx_views.get(), bank, bank->ne[0], bank->ne[1], n, bank->nb[1], bank->nb[2], slots[e]*bank->nb[2]);
        ggml_tensor * dst_view = ggml_view_3d(ctx_views.get(), dst,  dst->ne[0],  dst->ne[1],  n, dst->nb[1],  dst->nb[2],  e*dst->nb[2]);
        ggml_backend_view_init(src_view);
        ggml_backend_view_init(dst_view);
        ggml_backend_tensor_copy_async(backend, backend, src_view, dst_view);

        stats_copy.hits  += n;
        stats_copy.bytes += ggml_nbytes(src_view);

        return n;
    }

    bool copy(ggml_backend_t backend, const ggml_tensor * src, ggml_tensor * dst, ggml_cgraph * graph) {
        const auto it = layer_of.find(src);
        if (it == layer_of.end()) {
            return false;
        }
        const int32_t il      = it->second.first;
        const bool    has_cpu = it->second.second;
        layer & l = layers[il];
        group & g = groups[l.ig];

        GGML_ASSERT(backend == devices[g.id].backend);

        // the get_rows that looks up the slots of the selected experts
        const int n_nodes = ggml_graph_n_nodes(graph);
        const ggml_tensor * lookup = nullptr;
        for (int i = 0; i < n_nodes && lookup == nullptr; ++i) {
            const ggml_tensor * node = ggml_graph_node(graph, i);
            if (node->op == GGML_OP_GET_ROWS && node->src[0] == dst) {
                lookup = node;
            }
        }
        GGML_ASSERT(lookup != nullptr);

        // the selected experts must be computed in an earlier split
        // the scheduler starts a new split at the lookup because it reads a host weight, unless the lookup is the first node of its split
        const ggml_tensor * sel = lookup->src[1];
        for (int i = 0; i < n_nodes; ++i) {
            const ggml_tensor * node = ggml_graph_node(graph, i);
            if (node == sel || node == sel->view_src) {
                GGML_ABORT("the experts of layer %d are selected in the same split as their MoE cache lookup", il);
            }
        }
        GGML_ASSERT(ggml_is_contiguous(sel));
        GGML_ASSERT(sel->op == GGML_OP_RESHAPE); // of the [n_expert_used, n_tokens] ids, see build_lookup

        const int32_t n_used = (int32_t) sel->src[0]->ne[0];

        ids.resize(ggml_nelements(sel));
        ggml_backend_tensor_get_async(backend, sel, ids.data(), 0, ggml_nbytes(sel));
        ggml_backend_synchronize(backend);

        if (trace != nullptr) {
            write_trace(il, n_used);
        }

        // part of the misses is computed on the CPU once the cache is full, the carry keeps the share right on average
        const uint32_t share_q = has_cpu && g.lru.n_filled == g.lru.n_slots ? cpu_share_q : 0;
        auto n_cpu = [&](size_t n_miss) {
            const uint64_t acc = (uint64_t) n_miss*share_q + l.cpu_acc;
            l.cpu_acc = acc & 0xffff;
            return (size_t) (acc >> 16);
        };
        size_t n_hit = 0;
        if (!g.lru.plan(il, ids.data(), ids.size(), sel->src[0]->ne[1], n_used, n_zero, n_cpu, fills, cpu, n_hit)) {
            GGML_ABORT("the MoE cache is too small for the experts selected in layer %d", il);
        }
        if (has_cpu) {
            int32_t * cpu_map = (int32_t *) l.cpu_map->data;
            for (int32_t e : l.cpu_experts) {
                cpu_map[e] = -1;
            }
            l.cpu_experts.clear();
            for (const auto & f : cpu) {
                cpu_map[f.expert] = f.expert;
                l.cpu_experts.push_back(f.expert);
            }
        }

        // pool: take the misses found in the pool out of it, then copy the evicted experts to the pool before their slots are overwritten
        // an upload from a pool slot is queued on backend before a later D2H can reuse that slot
        pool_src.assign(fills.size(), -1);
        size_t n_pool_hit = 0;
        size_t n_pool_put = 0;
        if (g.pool.n > 0) {
            for (size_t i = 0; i < fills.size(); ++i) {
                pool_src[i] = g.pool.take(g.lru.make_key(il, fills[i].expert));
                n_pool_hit += pool_src[i] >= 0;
            }
            device & d = devices[g.id];
            if (d.d2h_pending && (!fills.empty() || !g.lru.evicted.empty())) {
                // the stage slots and the pool slots written by the last D2H must be ready
                ggml_backend_event_wait(backend, d.ev_d2h.get());
                d.d2h_pending = false;
            }
            // evicted experts go D2D to the stage slots (fast), the D2H from there runs on backend_d2h while backend uploads
            staged.clear();
            for (size_t i = 0; i < g.lru.evicted.size(); ++i) {
                const int32_t ps = g.pool.insert(g.lru.evicted[i]);
                if (ps < 0) {
                    continue;
                }
                const bool stage = (int32_t) staged.size() < g.n_stage;
                for (size_t ip = 0; ip < g.banks.size(); ++ip) {
                    ggml_tensor * bank = g.banks[ip];
                    const size_t es = bank->nb[2];
                    if (stage) {
                        ggml_tensor * sb = g.stage_banks[ip];
                        ggml_reset(ctx_views.get());
                        ggml_tensor * src_view = ggml_view_3d(ctx_views.get(), bank, bank->ne[0], bank->ne[1], 1, bank->nb[1], es, g.lru.evicted_slot[i]*es);
                        ggml_tensor * dst_view = ggml_view_3d(ctx_views.get(), sb,   sb->ne[0],   sb->ne[1],   1, sb->nb[1],   es, staged.size()*es);
                        ggml_backend_view_init(src_view);
                        ggml_backend_view_init(dst_view);
                        ggml_backend_tensor_copy_async(backend, backend, src_view, dst_view);
                    } else {
                        ggml_backend_tensor_get_async(backend, bank, (uint8_t *) g.pool_banks[ip]->data + ps*es, g.lru.evicted_slot[i]*es, es);
                    }
                }
                if (stage) {
                    staged.push_back({ (int32_t) staged.size(), ps });
                }
                n_pool_put++;
            }
            if (!staged.empty()) {
                ggml_backend_t b2 = d.backend_d2h.get();
                ggml_backend_event_record(d.ev_stage.get(), backend);
                ggml_backend_event_wait(b2, d.ev_stage.get());
                for (const auto & [ss, ps] : staged) {
                    for (size_t ip = 0; ip < g.stage_banks.size(); ++ip) {
                        const size_t es = g.stage_banks[ip]->nb[2];
                        ggml_backend_tensor_get_async(b2, g.stage_banks[ip], (uint8_t *) g.pool_banks[ip]->data + ps*es, ss*es, es);
                    }
                }
                ggml_backend_event_record(d.ev_d2h.get(), b2);
                d.d2h_pending = true;
            }
        }

        // ddio: read the misses not in the pool with O_DIRECT, many chunks in flight, into pinned stage slots
        ddio_slot_of.assign(fills.size(), -1);
        int32_t n_ddio = 0;
        if (ddio > 0 && !fills.empty()) {
            ddio_jobs.clear();
            for (size_t i = 0; i < fills.size() && n_ddio < ddio_slots; ++i) {
                if (pool_src[i] >= 0) {
                    continue;
                }
                const int32_t k = n_ddio++;
                ddio_slot_of[i] = k;
                ddio_fail[k].store(0, std::memory_order_relaxed);
                int32_t n_chunks = 0;
                for (size_t ip = 0; ip < l.experts.size(); ++ip) {
                    const ggml_tensor * w = l.experts[ip];
                    const ddio_src & src = ddio_srcs.at(w);
                    const size_t es  = w->nb[2];
                    const off_t  off = src.off + (off_t) (fills[i].expert*es);
                    const off_t  beg = off & ~(off_t) 4095;
                    const off_t  end = GGML_PAD(off + (off_t) es, 4096);
                    uint8_t * dst = ddio_base + (k*ddio_nt + ip)*ddio_stride;
                    for (off_t c = beg; c < end; c += ddio_chunk) {
                        const size_t len  = std::min<off_t>(ddio_chunk, end - c);
                        const size_t need = std::max<off_t>(0, std::min<off_t>(c + len, off + es) - c);
                        ddio_jobs.push_back({ src.fd, c, len, need, dst + (c - beg), &ddio_left[k], &ddio_fail[k] });
                        n_chunks++;
                    }
                }
                ddio_left[k].store(n_chunks, std::memory_order_relaxed);
            }
            if (!ddio_jobs.empty()) {
                {
                    std::lock_guard<std::mutex> lock(ddio_mutex);
                    ddio_queue.insert(ddio_queue.end(), ddio_jobs.begin(), ddio_jobs.end());
                }
                ddio_cv.notify_all();
            }
            stats_ddio_read += n_ddio;
        }

        // mmap'd misses may be on disk: request all of them first, so the reads run in parallel and not one page fault at a time
        if (!fills.empty() || !cpu.empty()) {
            llama_memory_ranges mr;
            for (const ggml_tensor * w : l.experts) {
                for (size_t i = 0; i < fills.size(); ++i) {
                    if (pool_src[i] < 0 && ddio_slot_of[i] < 0) {
                        mr.push_back({ (const uint8_t *) w->data + fills[i].expert*w->nb[2], w->nb[2] });
                    }
                }
                for (const auto & f : cpu) {
                    mr.push_back({ (const uint8_t *) w->data + f.expert*w->nb[2], w->nb[2] });
                }
            }
            llama_prefetch(std::move(mr));
        }

        // upload the missing experts, consecutive experts going to consecutive slots are uploaded together
        size_t bytes = 0;
        for (size_t ip = 0; ip < l.experts.size(); ++ip) {
            const ggml_tensor * w    = l.experts[ip];
            ggml_tensor       * bank = g.banks[ip];
            const size_t expert_size = w->nb[2];
            for (size_t i = 0; i < fills.size();) {
                if (ddio_slot_of[i] >= 0) {
                    i++;
                    continue;
                }
                if (pool_src[i] >= 0) {
                    ggml_backend_tensor_set_async(backend, bank, (const uint8_t *) g.pool_banks[ip]->data + pool_src[i]*expert_size, fills[i].slot*expert_size, expert_size);
                    bytes += expert_size;
                    i++;
                    continue;
                }
                size_t n = 1;
                while (i + n < fills.size() && pool_src[i + n] < 0 && ddio_slot_of[i + n] < 0 && fills[i + n].expert == fills[i].expert + (int32_t) n && fills[i + n].slot == fills[i].slot + (int32_t) n) {
                    n++;
                }
                ggml_backend_tensor_set_async(backend, bank, (const uint8_t *) w->data + fills[i].expert*expert_size, fills[i].slot*expert_size, n*expert_size);
                bytes += n*expert_size;
                i += n;
            }
        }
        // ddio misses, in the order their reads were queued, from the mmap if a read failed
        for (size_t i = 0; i < fills.size() && n_ddio > 0; ++i) {
            const int32_t k = ddio_slot_of[i];
            if (k < 0) {
                continue;
            }
            while (ddio_left[k].load(std::memory_order_acquire) > 0) {
                std::this_thread::yield();
            }
            const bool failed = ddio_fail[k].load(std::memory_order_relaxed) != 0;
            stats_ddio_fail += failed;
            for (size_t ip = 0; ip < l.experts.size(); ++ip) {
                const ggml_tensor * w = l.experts[ip];
                const size_t es  = w->nb[2];
                const off_t  off = ddio_srcs.at(w).off + (off_t) (fills[i].expert*es);
                const uint8_t * src = failed ? (const uint8_t *) w->data + fills[i].expert*es : ddio_base + (k*ddio_nt + ip)*ddio_stride + (off & 4095);
                ggml_backend_tensor_set_async(backend, g.banks[ip], src, fills[i].slot*es, es);
                bytes += es;
            }
        }
        if (n_ddio > 0) {
            // the next copy reuses the ddio stage slots
            ggml_backend_synchronize(backend);
        }
        for (int32_t ps : pool_src) {
            if (ps >= 0) {
                g.pool.release(ps);
            }
        }

        stats & st = ids.size() <= (size_t) 8*n_expert_used ? stats_small : stats_large;
        st.hits   += n_hit;
        st.misses += fills.size() + cpu.size();
        st.cpu    += cpu.size();
        st.bytes  += bytes;
        st.pool_hits += n_pool_hit;
        st.pool_put  += n_pool_put;

        // the next copy synchronizes the backend before it changes the slot map again
        ggml_backend_tensor_set_async(backend, dst, l.slot_map->data, 0, ggml_nbytes(l.slot_map));

        return true;
    }

    void log_stats() const {
        auto log = [](const char * name, const stats & st) {
            const size_t n = st.hits + st.misses;
            if (n == 0) {
                return;
            }
            LLAMA_LOG_INFO("llama_moe_cache: %s: hits = %zu, misses = %zu (%zu on the CPU), hit rate = %.2f%%, uploaded = %.2f MiB\n",
                name, st.hits, st.misses, st.cpu, 100.0*st.hits/n, st.bytes/1024.0/1024.0);
        };
        log("ubatch <= 8", stats_small);
        log("ubatch  > 8", stats_large);
        if (pool_size > 0) {
            LLAMA_LOG_INFO("llama_moe_cache: pool: %zu misses read from the pool, %zu evicted experts copied to it\n",
                stats_small.pool_hits + stats_large.pool_hits, stats_small.pool_put + stats_large.pool_put);
        }
        if (ddio > 0) {
            LLAMA_LOG_INFO("llama_moe_cache: ddio: %zu misses read, %zu failed reads\n", stats_ddio_read, stats_ddio_fail);
        }
        if (stats_copy.hits > 0) {
            LLAMA_LOG_INFO("llama_moe_cache: large batches: %zu experts copied from the cache, %.2f MiB\n", stats_copy.hits, stats_copy.bytes/1024.0/1024.0);
        }
    }
};

llama_moe_cache::llama_moe_cache(const llama_model & model, bool no_alloc, const std::vector<ggml_backend_t> & backends, const std::vector<ggml_backend_buffer_type_t> & bufts, size_t size, float cpu_share) :
    pimpl(new impl(model, no_alloc, backends, bufts, size, cpu_share)) {
}

llama_moe_cache::~llama_moe_cache() = default;

ggml_tensor * llama_moe_cache::get_experts(const ggml_tensor * w) const {
    return pimpl->get_experts(w);
}

llama_moe_cache_lookup llama_moe_cache::build_lookup(const llm_graph_context & g, const llama_moe_cache_layer & layer, ggml_tensor * selected_experts) const {
    const int32_t il = layer.il;
    ggml_context * ctx0 = g.ctx0;

    // part of the misses may be computed from the host experts, but not with weights before the FFN, LoRAs or rotated experts
    bool cpu_part = !layer.weight_before_ffn && g.loras->empty();
    for (ggml_tensor * w : { layer.up_exps, layer.gate_exps, layer.down_exps, layer.gate_up_exps }) {
        cpu_part = cpu_part && !(w != nullptr && g.hdmd != nullptr && g.hdmd->rot.count(w) > 0);
    }

    ggml_tensor * cpu_map  = cpu_part ? pimpl->get_cpu_map(il, selected_experts->ne[1]) : nullptr;
    ggml_tensor * slot_map = pimpl->get_slot_map(il, selected_experts->ne[1], selected_experts->ne[0], cpu_map != nullptr);
    if (slot_map == nullptr) {
        return {};
    }
    for (ggml_tensor * w : { layer.up_exps, layer.gate_exps, layer.down_exps, layer.gate_up_exps }) {
        if (w == nullptr) {
            continue;
        }
        ggml_tensor * experts = pimpl->get_experts(w);
        if (experts == nullptr) {
            return {};
        }
        // the device of the cache must compute the matmul, else the scheduler copies the cache to another backend
        ggml_tensor * b = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, experts->ne[0], w == layer.down_exps ? selected_experts->ne[0] : 1, selected_experts->ne[1]);
        if (!ggml_backend_supports_op(pimpl->backend(il), ggml_mul_mat_id(ctx0, experts, b, selected_experts))) {
            return {};
        }
    }

    ggml_tensor * ids = selected_experts;
    if (!ggml_is_contiguous(ids)) {
        ids = ggml_cont(ctx0, ids);
    }
    ids = ggml_reshape_1d(ctx0, ids, ggml_nelements(ids));

    // the slot map is a host weight, so the scheduler starts a new split here and copies it with the copy callback
    // the callback reads the selected experts, uploads the missing ones and updates the slot map
    ggml_tensor * slots = ggml_get_rows(ctx0, slot_map, ids); // [1, n_expert_used*n_tokens]
    if (!ggml_backend_supports_op(pimpl->backend(il), slots)) {
        return {};
    }
    ggml_backend_sched_set_tensor_backend(g.sched, slots, pimpl->backend(il));
    g.cb(slots, "ffn_moe_slots", il);

    // add the lookup now: the transform of a Hadamard-folded expert reads host weights and would start its split first
    ggml_build_forward_expand(g.gf, slots);

    // the misses computed from the host experts, read after the lookup has picked them
    ggml_tensor * cpu_ids = nullptr;
    if (cpu_map != nullptr) {
        cpu_ids = ggml_get_rows(ctx0, cpu_map, ids); // [1, n_expert_used*n_tokens]
        cpu_ids = ggml_reshape_2d(ctx0, cpu_ids, selected_experts->ne[0], selected_experts->ne[1]);
    }

    return { il, ggml_reshape_2d(ctx0, slots, selected_experts->ne[0], selected_experts->ne[1]), cpu_ids };
}

ggml_tensor * llama_moe_cache::build_experts(const llm_graph_context & g, const llama_moe_cache_lookup & lookup,
        ggml_tensor * cur, ggml_tensor * selected_experts, llama_moe_cache_build_experts_t build, void * user_data) const {
    GGML_ASSERT(lookup.slots != nullptr);

    ggml_tensor * experts = build(cur, selected_experts, lookup.slots, true, user_data);
    if (lookup.cpu_ids != nullptr) {
        // the cached experts and the misses computed from the host experts, each reads the others as zero
        // the cached experts come first in the graph, so that they are computed while the misses are
        ggml_build_forward_expand(g.gf, experts);
        experts = ggml_add(g.ctx0, experts, build(cur, selected_experts, lookup.cpu_ids, false, user_data));
        g.cb(experts, "ffn_moe_down_merged", lookup.il);
    }

    return experts;
}

bool llama_moe_cache::copy(ggml_backend_t backend, const ggml_tensor * src, ggml_tensor * dst, ggml_cgraph * graph) {
    return pimpl->copy(backend, src, dst, graph);
}

int64_t llama_moe_cache::copy_experts(ggml_backend_t backend, const ggml_tensor * w, ggml_tensor * dst, int64_t e, int64_t last) {
    return pimpl->copy_experts(backend, w, dst, e, last);
}

std::map<ggml_backend_buffer_type_t, size_t> llama_moe_cache::memory_breakdown() const {
    std::map<ggml_backend_buffer_type_t, size_t> res;
    for (const auto & d : pimpl->devices) {
        if (d.buf) {
            res[ggml_backend_buffer_get_type(d.buf.get())] += d.buf_size;
        }
    }
    if (pimpl->buf_host) {
        res[ggml_backend_buffer_get_type(pimpl->buf_host.get())] += pimpl->buf_host_size;
    }
    return res;
}
