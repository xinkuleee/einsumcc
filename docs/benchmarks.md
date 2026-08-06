# Benchmarks

Benchmark inputs are versioned JSON corpora. Results are JSON documents that
record environment, seed, timings, correctness error, workspace, schedule, and
the analytical selection for every case.

Run the small laptop corpus:

```bash
make benchmark
python3 -B -m einsumcc benchmark benchmarks/cpu-smoke.json \
  --repeats 5 -o cpu-results.json
```

Run the versioned generated-code corpus (this builds and tunes real dylibs):

```bash
make benchmark-native
python3 -B -m einsumcc benchmark-native benchmarks/native-mini-v0.1.json \
  --warmups 2 --repeats 20 --max-schedules 8 \
  --cache .einsumcc-cache/tuning-native-mini-v0.1.json \
  -o native-results.json
```

The CPU corpus is a regression and semantics suite. Its timings are not evidence
for A100 plan quality because Direct is a transparent Python implementation.
For one generated workload use `tune-native`; for stable comparisons across
commits use `benchmark-native` with the versioned native corpus. Both reports
record raw and expanded schedules, samples/median, correctness error, artifact
key, target identity, CPU/toolchain environment, tolerances, timestamp,
baseline, selector result, and measured speedup. The corpus report additionally
records FLOPs and observed GFLOP/s for the best and baseline schedules. The
baseline is the first statically ranked distinct schedule, not an untiled
kernel. A speedup of 1.0 means that candidate won; a larger value means another
measured schedule won. Samples are bound `ctypes` host-call latency, so they
still contain foreign-function call overhead.
v0.1 does not assert that tiling beats an external or untiled baseline.
Future A100 corpora should use realistic dimensions and the same result fields,
plus device/toolchain metadata and external cuTENSOR/course-starter baselines.

Corpus case names are unique and stable. Shapes and strides are element units,
not bytes. Changing a case changes its compiler workload key even if the case
name is preserved.
