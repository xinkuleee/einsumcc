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

The CPU corpus is a regression and semantics suite. Its timings are not evidence
for A100 plan quality because Direct is a transparent Python implementation.
Future A100 corpora should use realistic dimensions and the same result fields,
plus device/toolchain metadata and external cuTENSOR/course-starter baselines.

Corpus case names are unique and stable. Shapes and strides are element units,
not bytes. Changing a case changes its compiler workload key even if the case
name is preserved.
