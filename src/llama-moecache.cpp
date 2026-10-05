#include "llama-moecache.h"

#include "llama-impl.h"
#include "llama-model.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <cinttypes>
#include <condition_variable>
#include <cstring>
#include <deque>
#include <map>
#include <mutex>
#include <set>
#include <string>
#include <system_error>
#include <thread>
#include <vector>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace {

static bool file_exists(const char * path) {
    if (!path || !path[0]) {
        return false;
    }
    FILE * f = fopen(path, "rb");
    if (f) {
        fclose(f);
        return true;
    }
    return false;
}

static bool read_strp_profile(const char * path,
                              int64_t n_layers, int64_t n_expert,
                              std::vector<std::pair<int32_t, int32_t>> & ranked) {
    if (!path || !path[0]) {
        return false;
    }
    FILE * f = fopen(path, "rb");
    if (!f) {
        LLAMA_LOG_WARN("moe-cache: cannot open profile '%s'\n", path);
        return false;
    }
    char magic[4] = {0};
    uint32_t hdr[5] = {0};
    if (fread(magic, 1, 4, f) != 4 || fread(hdr, 4, 5, f) != 5) {
        fclose(f);
        LLAMA_LOG_WARN("moe-cache: truncated profile header in '%s'\n", path);
        return false;
    }
    if (memcmp(magic, "STRP", 4) != 0) {
        fclose(f);
        LLAMA_LOG_WARN("moe-cache: '%s' is not an STRP profile (magic mismatch)\n", path);
        return false;
    }
    const uint32_t nl = hdr[1];
    const uint32_t ne = hdr[2];
    const uint32_t n_ranked = hdr[4];
    if (n_layers > 0 && (int64_t)nl != n_layers) {
        LLAMA_LOG_WARN("moe-cache: profile layer count %u differs from model %lld (continuing with matching layers)\n",
                nl, (long long)n_layers);
    }
    if (n_expert > 0 && (int64_t)ne != n_expert) {
        LLAMA_LOG_WARN("moe-cache: profile expert count %u differs from model %lld\n",
                ne, (long long)n_expert);
    }
    std::vector<uint16_t> raw((size_t)n_ranked * 2);
    if (n_ranked > 0 && fread(raw.data(), 2, (size_t)n_ranked * 2, f) != (size_t)n_ranked * 2) {
        fclose(f);
        LLAMA_LOG_WARN("moe-cache: truncated ranked pairs in '%s'\n", path);
        return false;
    }
    fclose(f);
    ranked.reserve(n_ranked);
    for (uint32_t i = 0; i < n_ranked; ++i) {
        ranked.push_back({(int32_t)raw[i * 2], (int32_t)raw[i * 2 + 1]});
    }
    return true;
}

struct layer_state {
    llama_moe_cache_layer pub;

    ggml_backend_t upload_backend = nullptr;

    // LRU bookkeeping (host side; the tables mirror expert_slot)
    std::vector<int32_t>  slot_expert;   // slot -> expert id, -1 when empty
    std::vector<int32_t>  expert_slot;   // expert id -> slot, -1 when uncached
    std::vector<uint64_t> slot_last_use; // slot -> lamport clock of last hit
    std::vector<int32_t>  pending;       // uncached ids observed since last step (dedup, obs order)
    std::vector<int32_t>  table;

    std::vector<bool> slot_in_flight;   // slot has an upload pending
    std::vector<bool> expert_in_flight; // expert has an upload pending
    bool              table_dirty = false;

    uint64_t n_hit  = 0;
    uint64_t n_miss = 0;
};

struct upload_job {
    size_t  layer_idx;
    int32_t expert;
    int32_t slot;
    bool    done = false;
};

struct moe_cache {
    const llama_model * model = nullptr;
    const llama_context * owner = nullptr;

    int32_t n_slots     = 0;
    int32_t max_inserts = 2;

    uint64_t clock   = 0;
    uint64_t n_steps = 0;

    std::mutex mtx; // guards pending lists + clock (observe runs during graph exec)

    std::vector<layer_state> layers;
    std::map<const ggml_tensor *, size_t> by_up_src;
    std::map<const ggml_tensor *, size_t> by_gate_src;

    std::vector<ggml_context *>         ctxs;
    std::vector<ggml_backend_buffer_t>  bufs;
    std::vector<ggml_backend_t>         backends;

    struct pinned_host_buffer {
        void * base = nullptr;
        void (*unreg_fn)(void *) = nullptr;
    };
    std::vector<pinned_host_buffer> pinned_buffers;

    // async upload worker: slices are copied to the device off the decode
    // thread; the new table mapping is only published at a later step() once
    // the upload has completed, so a running graph never reads a torn slot
    std::thread              worker;
    std::mutex               wmtx;
    std::condition_variable  wcv;
    std::deque<upload_job>   todo;
    std::vector<upload_job>  done;
    bool                     stop = false;
};

moe_cache * g_cache = nullptr;
std::mutex g_init_mtx;

void moe_obs_cb(const struct ggml_tensor * experts, const struct ggml_tensor * ids, void * ud) {
    moe_cache * mc = (moe_cache *) ud;

    const int64_t n_ids    = ids->ne[0];
    const int64_t n_tokens = ids->ne[1];
    if (n_tokens > 4) {
        return; // batch/prefill: the cache graph is not built there, don't pollute the LRU
    }

    const auto it = mc->by_gate_src.find(experts);
    if (it == mc->by_gate_src.end()) {
        return;
    }
    layer_state * ls = &mc->layers[it->second];

    std::lock_guard<std::mutex> lock(mc->mtx);
    for (int64_t t = 0; t < n_tokens; ++t) {
        for (int64_t i = 0; i < n_ids; ++i) {
            const int32_t id = *(const int32_t *) ((const char *) ids->data + t*ids->nb[1] + i*ids->nb[0]);
            if (id < 0 || id >= (int32_t) ls->expert_slot.size()) {
                continue;
            }
            const int32_t slot = ls->expert_slot[id];
            if (slot >= 0) {
                ls->n_hit++;
                ls->slot_last_use[slot] = ++mc->clock;
            } else {
                ls->n_miss++;
                if (ls->expert_in_flight[id]) {
                    continue;
                }
                bool dup = false;
                for (int32_t p : ls->pending) {
                    if (p == id) { dup = true; break; }
                }
                if (!dup) {
                    ls->pending.push_back(id);
                }
            }
        }
    }
}

void upload_slice(ggml_backend_t backend, ggml_tensor * dst_c, const ggml_tensor * src, int32_t expert, int32_t slot) {
    const size_t sz = src->nb[2];
    if ((size_t) slot*dst_c->nb[2] + sz > ggml_nbytes(dst_c) || (size_t) expert*sz + sz > ggml_nbytes(src)) {
        LLAMA_LOG_ERROR("moe-cache: bad upload %s <- %s expert=%d slot=%d sz=%zu dst_nb2=%zu dst_bytes=%zu src_bytes=%zu\n",
                dst_c->name, src->name, expert, slot, sz, dst_c->nb[2], ggml_nbytes(dst_c), ggml_nbytes(src));
        return;
    }
    ggml_backend_tensor_set_async(backend, dst_c, (const char *) src->data + (size_t) expert*sz, (size_t) slot*dst_c->nb[2], sz);
}

void set_table_entry(layer_state & ls, int32_t expert, int32_t slot_or_dummy) {
    ls.table[expert] = slot_or_dummy;
    memcpy((char *) ls.pub.host_table->data + (size_t) expert*sizeof(int32_t), &slot_or_dummy, sizeof(int32_t));
    ls.table_dirty = true;
}

} // namespace

bool llama_moe_cache_init(const llama_model & model, const llama_context & ctx,
                          int32_t n_slots, int32_t max_inserts,
                          const char * profile_path,
                          bool pin_host) {
    std::lock_guard<std::mutex> init_lock(g_init_mtx);
    if (g_cache) {
        return false;
    }
    bool enabled = false;
    [&]() {
        if (n_slots <= 0) {
            return;
        }

        auto * mc = new moe_cache();
        mc->model = &model;
        mc->owner = &ctx;
        mc->n_slots = n_slots;
        if (max_inserts > 0) {
            mc->max_inserts = max_inserts;
        }

        // collect the host-resident expert layers, grouped by the device buffer
        // type of that layer's router (the cache lives next to the router)
        struct cand { int il; const llama_layer * l; };
        std::map<ggml_backend_buffer_type_t, std::vector<cand>> groups;

        for (size_t il = 0; il < model.layers.size(); ++il) {
            const auto & l = model.layers[il];
            if (!l.ffn_up_exps || !l.ffn_gate_exps || !l.ffn_down_exps || !l.ffn_gate_inp) {
                continue;
            }
            if (!l.ffn_up_exps->data || !l.ffn_gate_exps->data || !l.ffn_down_exps->data) {
                continue; // dry-run / memory-estimation model: weights not loaded, don't bind to it
            }
            if (!l.ffn_up_exps->buffer || !ggml_backend_buffer_is_host(l.ffn_up_exps->buffer)) {
                continue; // experts already on a device: nothing to cache
            }
            if (!l.ffn_gate_inp->buffer || ggml_backend_buffer_is_host(l.ffn_gate_inp->buffer)) {
                continue; // no device home for the cache
            }
            groups[ggml_backend_buffer_get_type(l.ffn_gate_inp->buffer)].push_back({(int) il, &l});
        }

        if (groups.empty()) {
            LLAMA_LOG_INFO("%s: LLAMA_MOE_CACHE_SLOTS=%d but no host-resident expert layers found - disabled\n", __func__, n_slots);
            delete mc;
            return;
        }

        // host buffer for the CPU-side tables
        std::vector<cand> all;
        for (auto & g : groups) {
            all.insert(all.end(), g.second.begin(), g.second.end());
        }

        auto alloc_group = [&](ggml_backend_buffer_type_t buft, const std::vector<cand> & cands, bool tables_only) -> bool {
            ggml_backend_t upload_backend = nullptr;
            if (!tables_only) {
                ggml_backend_dev_t dev = ggml_backend_buft_get_device(buft);
                if (!dev || buft != ggml_backend_dev_buffer_type(dev)) {
                    LLAMA_LOG_WARN("%s: no MoE cache upload stream for buffer type %s - cache disabled\n",
                            __func__, ggml_backend_buft_name(buft));
                    return false;
                }
                upload_backend = ggml_backend_dev_init(dev, nullptr);
                if (!upload_backend) {
                    LLAMA_LOG_WARN("%s: failed to create MoE cache upload stream for %s - cache disabled\n",
                            __func__, ggml_backend_buft_name(buft));
                    return false;
                }
            }

            ggml_init_params ip = {
                /*.mem_size  =*/ ggml_tensor_overhead()*(cands.size()*4 + 8),
                /*.mem_buffer=*/ nullptr,
                /*.no_alloc  =*/ true,
            };
            ggml_context * ctx = ggml_init(ip);
            if (!ctx) {
                ggml_backend_free(upload_backend);
                return false;
            }
            mc->ctxs.push_back(ctx);

            for (const auto & c : cands) {
                layer_state * ls = nullptr;
                for (auto & l : mc->layers) {
                    if (l.pub.il == c.il) { ls = &l; break; }
                }
                if (!ls) {
                    mc->layers.push_back({});
                    ls = &mc->layers.back();
                    ls->pub.il       = c.il;
                    ls->pub.n_slots  = n_slots;
                    ls->pub.n_dummy  = model.hparams.n_expert_used;
                    ls->pub.up_src   = c.l->ffn_up_exps;
                    ls->pub.gate_src = c.l->ffn_gate_exps;
                    ls->pub.down_src = c.l->ffn_down_exps;
                }

                if (tables_only) {
                    ls->pub.host_table = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 1, ls->pub.up_src->ne[2]);
                    ggml_format_name(ls->pub.host_table, "moe_cache_htbl.%d", c.il);
                } else {
                    ls->upload_backend = upload_backend;
                    const ggml_tensor * u = c.l->ffn_up_exps;
                    const ggml_tensor * g = c.l->ffn_gate_exps;
                    const ggml_tensor * d = c.l->ffn_down_exps;
                    ls->pub.up_c   = ggml_new_tensor_3d(ctx, u->type, u->ne[0], u->ne[1], n_slots + ls->pub.n_dummy);
                    ls->pub.gate_c = ggml_new_tensor_3d(ctx, g->type, g->ne[0], g->ne[1], n_slots + ls->pub.n_dummy);
                    ls->pub.down_c = ggml_new_tensor_3d(ctx, d->type, d->ne[0], d->ne[1], n_slots + ls->pub.n_dummy);
                    ls->pub.dev_table = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 1, u->ne[2]);
                    ggml_format_name(ls->pub.up_c,      "moe_cache_up.%d",   c.il);
                    ggml_format_name(ls->pub.gate_c,    "moe_cache_gate.%d", c.il);
                    ggml_format_name(ls->pub.down_c,    "moe_cache_down.%d", c.il);
                    ggml_format_name(ls->pub.dev_table, "moe_cache_tbl.%d",  c.il);
                }
            }

            ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);
            if (!buf) {
                LLAMA_LOG_WARN("%s: failed to allocate MoE cache buffer on %s - cache disabled\n",
                        __func__, ggml_backend_buft_name(buft));
                ggml_backend_free(upload_backend);
                return false;
            }
            ggml_backend_buffer_clear(buf, 0);
            mc->bufs.push_back(buf);
            if (upload_backend) {
                mc->backends.push_back(upload_backend);
            }
            return true;
        };

        bool ok = alloc_group(ggml_backend_cpu_buffer_type(), all, /*tables_only=*/true);
        for (auto & g : groups) {
            if (!ok) {
                break;
            }
            ok = alloc_group(g.first, g.second, /*tables_only=*/false);
        }

        if (!ok) {
            for (auto * backend : mc->backends) { ggml_backend_free(backend); }
            for (auto * b : mc->bufs) { ggml_backend_buffer_free(b); }
            for (auto * c : mc->ctxs) { ggml_free(c); }
            delete mc;
            return;
        }

        // init LRU state + tables (everything uncached -> dummy slot n_slots)
        size_t vram = 0;
        for (auto & ls : mc->layers) {
            const int64_t n_expert = ls.pub.up_src->ne[2];
            ls.slot_expert.assign(n_slots, -1);
            ls.expert_slot.assign(n_expert, -1);
            ls.slot_last_use.assign(n_slots, 0);
            ls.slot_in_flight.assign(n_slots, false);
            ls.expert_in_flight.assign(n_expert, false);
            ls.table.assign(n_expert, n_slots);

            ggml_backend_tensor_set(ls.pub.dev_table,  ls.table.data(), 0, n_expert*sizeof(int32_t));
            ggml_backend_tensor_set(ls.pub.host_table, ls.table.data(), 0, n_expert*sizeof(int32_t));

            mc->by_up_src[ls.pub.up_src] = &ls - mc->layers.data();
            mc->by_gate_src[ls.pub.gate_src] = &ls - mc->layers.data();
            vram += ggml_nbytes(ls.pub.up_c) + ggml_nbytes(ls.pub.gate_c) + ggml_nbytes(ls.pub.down_c);
            LLAMA_LOG_DEBUG("moe-cache: init layer %d '%s' %zu bytes/expert\n",
                    ls.pub.il, ls.pub.up_src->name, ls.pub.up_src->nb[2]);
        }

        // pre-seed cache from an expert profile if provided or available
        std::string resolved_profile;
        if (profile_path && profile_path[0]) {
            if (strcmp(profile_path, "none") != 0 && strcmp(profile_path, "0") != 0) {
                resolved_profile = profile_path;
            }
        } else {
            const char * env_prof = getenv("LLAMA_MOE_EXPERT_PROFILE");
            if (env_prof && env_prof[0]) {
                if (strcmp(env_prof, "none") != 0 && strcmp(env_prof, "0") != 0) {
                    resolved_profile = env_prof;
                }
            } else {
                int64_t n_exp = mc->layers.empty() ? 0 : mc->layers[0].pub.up_src->ne[2];
                if (n_exp == 256 && file_exists("data/expert-profile-coder.bin")) {
                    resolved_profile = "data/expert-profile-coder.bin";
                } else if (file_exists("data/expert-profile.bin")) {
                    resolved_profile = "data/expert-profile.bin";
                }
            }
        }

        std::vector<std::pair<int32_t, int32_t>> ranked;
        int64_t model_n_expert = mc->layers.empty() ? 0 : mc->layers[0].pub.up_src->ne[2];
        if (!resolved_profile.empty() && read_strp_profile(resolved_profile.c_str(), (int64_t)mc->layers.size(), model_n_expert, ranked)) {
            std::map<int, size_t> layer_map;
            for (size_t li = 0; li < mc->layers.size(); ++li) {
                layer_map[mc->layers[li].pub.il] = li;
            }
            std::vector<int32_t> slots_filled(mc->layers.size(), 0);
            size_t total_seeded = 0;
            for (const auto & rp : ranked) {
                auto it = layer_map.find(rp.first);
                if (it == layer_map.end()) {
                    continue;
                }
                size_t li = it->second;
                auto & ls = mc->layers[li];
                int32_t exp = rp.second;
                if (exp < 0 || exp >= (int32_t)ls.expert_slot.size()) {
                    continue;
                }
                if (slots_filled[li] < n_slots && ls.expert_slot[exp] < 0) {
                    int32_t slot = slots_filled[li]++;
                    ls.slot_expert[slot] = exp;
                    ls.expert_slot[exp]  = slot;
                    ls.slot_last_use[slot] = ++mc->clock;
                    set_table_entry(ls, exp, slot);
                    total_seeded++;
                }
            }
            // upload all pre-seeded slices to device synchronously
            for (auto & ls : mc->layers) {
                for (int32_t s = 0; s < n_slots; ++s) {
                    int32_t exp = ls.slot_expert[s];
                    if (exp >= 0) {
                        upload_slice(ls.upload_backend, ls.pub.up_c,   ls.pub.up_src,   exp, s);
                        upload_slice(ls.upload_backend, ls.pub.gate_c, ls.pub.gate_src, exp, s);
                        upload_slice(ls.upload_backend, ls.pub.down_c, ls.pub.down_src, exp, s);
                    }
                }
                ggml_backend_synchronize(ls.upload_backend);
                if (ls.table_dirty) {
                    ggml_backend_tensor_set(ls.pub.dev_table, ls.table.data(), 0, ls.table.size()*sizeof(int32_t));
                    ls.table_dirty = false;
                }
            }
            LLAMA_LOG_INFO("%s: pre-seeded MoE cache with %zu hot experts from '%s'\n",
                    __func__, total_seeded, resolved_profile.c_str());
        }

        // register host-resident expert buffers with the device for async DMA
        bool should_pin = pin_host;
        const char * env_pin = getenv("LLAMA_MOE_CACHE_PIN");
        if (env_pin && atoi(env_pin) == 0) {
            should_pin = false;
        }

        std::set<void *> registered_host_bases;
        for (auto & g : groups) {
            if (!should_pin) {
                LLAMA_LOG_INFO("%s: host memory pinning disabled (safe mapped mode)\n", __func__);
                break;
            }
            ggml_backend_dev_t dev = ggml_backend_buft_get_device(g.first);
            if (!dev) {
                continue;
            }
            ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
            if (!reg) {
                continue;
            }
            auto reg_fn = (bool (*)(void *, size_t)) ggml_backend_reg_get_proc_address(reg, "ggml_backend_register_host_buffer");
            auto unreg_fn = (void (*)(void *)) ggml_backend_reg_get_proc_address(reg, "ggml_backend_unregister_host_buffer");
            if (!reg_fn) {
                continue;
            }

            for (const auto & c : g.second) {
                const ggml_tensor * tensors[] = { c.l->ffn_up_exps, c.l->ffn_gate_exps, c.l->ffn_down_exps };
                for (const auto * t : tensors) {
                    if (!t || !t->buffer) {
                        continue;
                    }
                    void * base = ggml_backend_buffer_get_base(t->buffer);
                    size_t size = ggml_backend_buffer_get_size(t->buffer);
                    if (base && size > 0 && registered_host_bases.insert(base).second) {
#ifdef _WIN32
                        MEMORYSTATUSEX ms;
                        ms.dwLength = sizeof(ms);
                        if (GlobalMemoryStatusEx(&ms)) {
                            constexpr uint64_t headroom = 4ULL * 1024 * 1024 * 1024; // 4 GiB headroom
                            if (size + headroom > ms.ullAvailPhys) {
                                LLAMA_LOG_WARN("%s: skipping pinning of %.1f MiB host memory to preserve 4 GiB RAM headroom (%.1f MiB available)\n",
                                        __func__, size/1024.0/1024.0, ms.ullAvailPhys/1024.0/1024.0);
                                continue;
                            }
                        }
#endif
                        if (reg_fn(base, size)) {
                            mc->pinned_buffers.push_back({base, unreg_fn});
                            LLAMA_LOG_INFO("%s: pinned %.1f MiB host memory for async DMA expert uploads\n",
                                    __func__, size/1024.0/1024.0);
                        }
                    }
                }
            }
        }

        try {
            mc->worker = std::thread([mc]() {
                for (;;) {
                    upload_job j;
                    {
                        std::unique_lock<std::mutex> lk(mc->wmtx);
                        mc->wcv.wait(lk, [mc]() { return mc->stop || !mc->todo.empty(); });
                        if (mc->stop) {
                            return;
                        }
                        j = mc->todo.front();
                        mc->todo.pop_front();
                    }
                    auto & ls = mc->layers[j.layer_idx];
                    upload_slice(ls.upload_backend, ls.pub.up_c,   ls.pub.up_src,   j.expert, j.slot);
                    upload_slice(ls.upload_backend, ls.pub.gate_c, ls.pub.gate_src, j.expert, j.slot);
                    upload_slice(ls.upload_backend, ls.pub.down_c, ls.pub.down_src, j.expert, j.slot);
                    ggml_backend_synchronize(ls.upload_backend);
                    {
                        std::lock_guard<std::mutex> lk(mc->wmtx);
                        j.done = true;
                        mc->done.push_back(j);
                    }
                }
            });
        } catch (const std::system_error & e) {
            LLAMA_LOG_WARN("%s: failed to create MoE cache worker: %s - cache disabled\n", __func__, e.what());
            for (const auto & pb : mc->pinned_buffers) {
                if (pb.unreg_fn) {
                    pb.unreg_fn(pb.base);
                }
            }
            for (auto * backend : mc->backends) { ggml_backend_free(backend); }
            for (auto * buffer : mc->bufs) { ggml_backend_buffer_free(buffer); }
            for (auto * ctx : mc->ctxs) { ggml_free(ctx); }
            delete mc;
            return;
        }

        ggml_set_moe_obs_callback(moe_obs_cb, mc);
        g_cache = mc;
        enabled = true;

        LLAMA_LOG_INFO("%s: MoE expert cache enabled: %zu layers x %d slots, %d inserts/step, %.1f MiB device memory\n",
                __func__, mc->layers.size(), n_slots, mc->max_inserts, vram/1024.0/1024.0);
    }();
    return enabled;
}

void free_cache(moe_cache * mc) {
    ggml_set_moe_obs_callback(nullptr, nullptr);
    g_cache = nullptr;
    {
        std::lock_guard<std::mutex> lock(mc->wmtx);
        mc->stop = true;
    }
    mc->wcv.notify_one();
    mc->worker.join();

    for (const auto & pb : mc->pinned_buffers) {
        if (pb.unreg_fn) {
            pb.unreg_fn(pb.base);
        }
    }

    for (auto * backend : mc->backends) { ggml_backend_free(backend); }
    for (auto * buffer : mc->bufs) { ggml_backend_buffer_free(buffer); }
    for (auto * ctx : mc->ctxs) { ggml_free(ctx); }
    delete mc;
}

void llama_moe_cache_free(const llama_context & ctx) {
    std::lock_guard<std::mutex> init_lock(g_init_mtx);
    moe_cache * mc = g_cache;
    if (!mc || mc->owner != &ctx) {
        return;
    }
    free_cache(mc);
}

void llama_moe_cache_free(const llama_model & model) {
    std::lock_guard<std::mutex> init_lock(g_init_mtx);
    moe_cache * mc = g_cache;
    if (!mc || mc->model != &model) {
        return;
    }
    free_cache(mc);
}

const llama_moe_cache_layer * llama_moe_cache_lookup(const ggml_tensor * up_exps) {
    if (!g_cache) {
        return nullptr;
    }
    auto it = g_cache->by_up_src.find(up_exps);
    if (it == g_cache->by_up_src.end()) {
        return nullptr;
    }
    return &g_cache->layers[it->second].pub;
}

void llama_moe_cache_step() {
    moe_cache * mc = g_cache;
    if (!mc) {
        return;
    }

    // 1) publish completed uploads (sync point: no graph is executing)
    std::vector<upload_job> done;
    {
        std::lock_guard<std::mutex> wlk(mc->wmtx);
        done.swap(mc->done);
    }
    {
        std::lock_guard<std::mutex> lk(mc->mtx);
        for (const auto & j : done) {
            auto & ls = mc->layers[j.layer_idx];
            ls.slot_expert[j.slot]     = j.expert;
            ls.expert_slot[j.expert]   = j.slot;
            ls.slot_last_use[j.slot]   = ++mc->clock;
            ls.slot_in_flight[j.slot]  = false;
            ls.expert_in_flight[j.expert] = false;
            set_table_entry(ls, j.expert, j.slot);
        }
    }

    std::lock_guard<std::mutex> lock(mc->mtx);
    mc->n_steps++;

    // 2) schedule new uploads: evict at a sync point (clear the victim's table
    //    entry now), then hand the slice copies to the worker
    for (size_t li = 0; li < mc->layers.size(); ++li) {
        auto & ls = mc->layers[li];
        if (ls.pending.empty()) {
            continue;
        }

        int budget = mc->max_inserts;
        for (auto it = ls.pending.rbegin(); it != ls.pending.rend() && budget > 0; ++it) {
            const int32_t id = *it;
            if (ls.expert_slot[id] >= 0 || ls.expert_in_flight[id]) {
                continue;
            }

            // victim: an empty non-in-flight slot if any, else the LRU non-in-flight slot
            int32_t slot = -1;
            uint64_t best = UINT64_MAX;
            for (int32_t s = 0; s < mc->n_slots; ++s) {
                if (ls.slot_in_flight[s]) {
                    continue;
                }
                if (ls.slot_expert[s] < 0) { slot = s; break; }
                if (ls.slot_last_use[s] < best) { best = ls.slot_last_use[s]; slot = s; }
            }
            if (slot < 0) {
                break; // every slot is in flight; try again next step
            }

            const int32_t victim = ls.slot_expert[slot];
            if (victim >= 0) {
                ls.expert_slot[victim] = -1;
                ls.slot_expert[slot]   = -1;
                set_table_entry(ls, victim, mc->n_slots);
            }
            ls.slot_in_flight[slot] = true;
            ls.expert_in_flight[id] = true;

            std::lock_guard<std::mutex> wlk(mc->wmtx);
            mc->todo.push_back({li, id, slot});
            --budget;
        }
        ls.pending.clear();
    }

    for (auto & ls : mc->layers) {
        if (ls.table_dirty) {
            ggml_backend_tensor_set(ls.pub.dev_table, ls.table.data(), 0, ls.table.size()*sizeof(int32_t));
            ls.table_dirty = false;
        }
    }
    mc->wcv.notify_one();

    if (mc->n_steps % 512 == 0) {
        uint64_t h = 0, m = 0;
        for (auto & ls : mc->layers) { h += ls.n_hit; m += ls.n_miss; }
        LLAMA_LOG_DEBUG("moe-cache: steps=%" PRIu64 " hits=%" PRIu64 " misses=%" PRIu64 " hit-rate=%.1f%%\n",
                mc->n_steps, h, m, h + m ? 100.0*h/(h + m) : 0.0);
    }
}
