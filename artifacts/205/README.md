# Issue 205 — cross-process JIT stampede: Phase 1 measurement

**Phase 1 only: this directory contains a harness, raw data, and conclusions. No
runtime coordination, locking, or cache-semantics change was implemented or is
proposed here.** The upstream decision (no cross-process lock at this stage)
stands; this work measures what the lock would have bought and what it would
have cost.

Upstream issue: `NVlabs/cutile-rs#205` — "the in-process JIT cache compiles each
kernel once per process; the persistent disk cache is shared across processes,
but compilation is not deduplicated across process boundaries."

## 1. What was measured, in one paragraph

`cutile`'s JIT caches compiled cubins in two layers: an in-process L1 map and an
opt-in, content-addressed, on-disk L2 store (`cutile-compiler/src/jit_cache.rs`).
The disk store is off by default and has **no environment variable that enables
it** — a program must construct a `FileSystemJitStore` over its own directory and
call `jit_cache::enable`. When N processes that share one store directory
cold-start at the same instant, each of them independently misses the same L2
key and spawns `tileiras` for the same kernel. Writes are atomic
(temp file + `rename`), so the store converges on exactly one entry and the
result is correct — the question is only what the redundant work costs.

This harness starts N already-built worker processes behind a cross-process
ready/start barrier, timed from `main` and from barrier release, and records for
every rank: the actual L2 key (derived independently up front *and* verified
against the file the runtime actually wrote), real `tileiras` spawn attempts
from a pass-through wrapper (the runtime's own counter only counts *successful*
compiles), disk hits/misses/puts, CPU time including the compiler subprocess,
GPU identity, and the correctness of the kernel result.

## 2. Environment

<!-- ENV-TABLE -->
(filled in from `raw/run_header*.json`)

## 3. Harness layout

| file | role |
|---|---|
| `cutile-examples/examples/jit_stampede_worker.rs` | one rank: bind device, install the store for its mode, derive the L2 key from meta tensors, upload real inputs with host→device copies (no kernel compile), print `STAMPEDE_READY`, block on one stdin byte, then JIT+launch+verify and write its JSON record |
| `scripts/bench_jit_stampede.py` | coordinator: builds nothing inside a timed region, starts the binary directly, runs the ready/start barrier, enforces deadlines, kills process groups, writes raw JSONL/CSV + `summary.json` |
| `scripts/tileiras_wrap.sh` | pass-through `tileiras` wrapper used in **every** arm: logs spawn attempts, spawn class (`version` probe / `probe` capability check / `compile`), child exit status, cubin size and the child's wall time; `--version` is passed through untouched so the key fingerprint is unchanged |
| `scripts/run_jit_stampede_205.sh` | the one command that builds and runs the whole matrix |
| `scripts/jit_stampede_tables.py` | turns `summary.json` into the tables below |

### 3.1 Barrier protocol

The coordinator starts every rank with a pipe on stdin and waits until each rank
prints `STAMPEDE_READY`. A rank reaches READY only after it has bound its
device, created a CUDA context, installed (or deliberately not installed) its
JIT store, derived its L2 key, and uploaded its inputs. The coordinator then
writes one byte to every rank's stdin, in one tight loop: that write is the
**barrier release**, and it is observed by each rank with a microsecond-scale
blocking read (no polling, no file-watching latency).

### 3.2 What the numbers mean

| field | definition |
|---|---|
| `process_to_result_ms` | first line of the rank's `main` → verified-correct result on the host. Includes device init, the frontend-only key derivation, upload, compile, launch, and the host-side check. |
| `barrier_to_result_ms` | barrier release → verified-correct result. Everything before the barrier is identical in every arm, so this is the contention-sensitive number. |
| `barrier_to_launch_ms` | barrier release → kernel result materialized on the device (`sync`), i.e. JIT + module load + launch. |
| `relaunch_min_ms` | the same launch again, after the measurement, when the in-process L1 cache is already populated: the pure launch+verify floor with no JIT at all. |
| `jit_attributable_ms` | `barrier_to_launch_ms − relaunch_min_ms`: the part of the first launch that the JIT (frontend + backend) actually added, separated from GPU/module-load noise. |
| `backend_attempts` | **real `tileiras` stage-2 spawn attempts** in the timed region, from the wrapper log. This is the spawn counter the runtime does not have. |
| `backend_success_delta` | `jit_backend_compile_count()` delta: `tileiras` runs that produced a non-empty cubin. **Not** a spawn count. |
| `wrapper_compile_ms` | wall time inside the wrapper around the `tileiras` child: the pure backend compile cost, independent of the GPU. |
| `disk_hits_delta` / `disk_misses_delta` / `disk_puts_delta` | `jit_cache::stats()` deltas across the timed region. |
| `cpu_user_ms` / `cpu_system_ms` | `getrusage(RUSAGE_SELF) + getrusage(RUSAGE_CHILDREN)` deltas across the timed region, so the `tileiras` subprocess is included, not just the Rust parent. |
| `l2_key` | 64-char SHA-256 derived up front from the compiler frontend over **meta** tensors (no store access, no backend, no GPU). |
| `store_keys_on_disk` | the `<key>.cubin` file names actually present in the rank's store after its run — an independent check that the key used by the runtime is the key declared. |

Group-level metrics (`group_to_all_correct_ms`) use `CLOCK_REALTIME` timestamps
taken in the workers, compared against the coordinator's timestamp captured just
before it spawned the first rank. All per-rank durations use a per-process
`Instant`, so no cross-process clock is used for them.

## 4. Modes

| mode | store | purpose |
|---|---|---|
| `nocache` | none installed (`jit_cache::disable()`, also the library default) | raw concurrent-compile control; must be distinguished from default behaviour, and it is: the default is *also* no disk cache, but with no way to share one |
| `private` | fresh directory **per rank per round** | unshared cold compile cost: no cross-process sharing at all, but the store code path is live |
| `shared` | one fresh directory for all ranks in the round | reproduces the same-key duplicate backend compiles |
| `shared-warm` | one directory per round, populated by a **single measured process first** | upper bound of the documented workaround; warmup cost and post-warm startup cost are reported separately, plus the total |

## 5. Results

<!-- RESULTS -->

## 6. Raw data layout

<!-- LAYOUT -->

## 7. Reproducing

<!-- RERUN -->

## 8. Limitations

<!-- LIMITS -->
