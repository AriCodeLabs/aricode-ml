# aricode threading demos

Multi-core parallelism in aricode uses three primitives that ship in the
compiler:

- `thread_spawn(func)` / `thread_spawn(func, arg)` — issues a raw
  `clone(CLONE_VM|CLONE_FS|CLONE_FILES|CLONE_SIGHAND|CLONE_THREAD, …)`
  syscall and returns the child tid.  The worker runs in the same
  address space as the parent (CLONE_VM), so any `arr_new` / `arr_f64_new`
  allocation is visible to both.  The two-arg form passes `arg` in RDI
  — SysV first integer-arg register — giving the worker a pointer it
  can use to find shared state.
- `atomic_add_i64(base, idx, delta)` — one `lock xadd` instruction.
  Returns the old value; the memory update is uninterruptible.  Used
  for barriers (workers bump a shared "done" counter, parent spin-waits
  on it) and lock-free statistics.
- Plain `arr_get` / `arr_f64_get` on a shared slot — x86 already guarantees
  that aligned 64-bit loads/stores are atomic, so no extra primitive is
  needed for the "read the counter" side of a barrier.

## `parallel_matvec.ari` — near-linear 4× speedup

Compute `y = W · x` for a 512 × 512 f64 matrix `W`, repeated 10 000 times.
Work is split row-wise: four workers each compute 128 output rows,
calling `arr_f64_dot_range` (the AVX2 dot-product builtin) on their
slice of `W`.  No contention on the output (disjoint rows) and W fits
in L2 cache per worker — so the workload is compute-bound rather than
memory-bandwidth-bound.

Measured on a Zen 3 (Ryzen 7 5800X, 4 workers):

| Variant                   | Wall time | Speedup |
|---------------------------|----------:|--------:|
| `serial_matvec.ari`       | 0.53 s    |  1.00×  |
| `parallel_matvec.ari`     | 0.13 s    |  **4.0×** |

Both binaries print the same `-95.856602` checksum — bit-identical
arithmetic independent of worker count.

## Usage

```sh
aric parallel_matvec.ari -o parallel_matvec
aric serial_matvec.ari   -o serial_matvec
time ./serial_matvec
time ./parallel_matvec
```

The `user` / `real` ratio in the parallel run is a direct read on
effective core occupancy.  Expect `user ≈ N_WORKERS × real`.

## Caveats

- The parent spin-waits on the "done" counter.  For short-running
  workers this is fine; long-running workloads should prefer a futex
  or blocking syscall to avoid burning a whole core on the wait.
- Memory-bound workloads (matrices larger than your L3, or dense BLAS
  with poor locality) will scale sub-linearly because all workers
  share the same memory controller.  The demo here is sized to fit
  cleanly in L2 to isolate the compute-parallelism win.
- `thread_wait` uses `wait4` which returns `-ECHILD` for `CLONE_THREAD`
  children; the spin-on-counter barrier is the intended coordination
  pattern until a proper futex wait lands.
