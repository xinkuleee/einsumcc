# EinsumCC v0.1 技术设计与面试复习

本文记录 v0.1（Nano v1）实际完成的能力、关键算法、工程取舍和可继续扩展的边界。它既是实现说明，也是后续准备编译器、Kernel 和 AI Compiler 岗位面试时的复习提纲。

## 一句话定位

EinsumCC 是一个面向**二元静态 Tensor Contraction** 的小型领域编译器：它把显式 Einstein notation、shape 和 stride 变成经过验证的语义 IR，分析 layout，选择 Direct、GEMM-view 或 Packed-GEMM 计划，并能在 macOS 上把 Direct 路径经自定义 MLIR dialect 编译成可执行的原生 CPU 动态库。

它不是 Triton 的重实现。Triton 是通用 GPU kernel 编程语言和编译器；EinsumCC v0.1 是范围更窄的 contraction compiler，重点展示：

- DSL/frontend 与语义验证；
- layout legality 与算法选择；
- 可解释的 cost model 和 autotuning 接口；
- 自定义 MLIR dialect、lowering pass 和原生运行时；
- ABI、artifact cache 与端到端差分测试。

## v0.1 能力边界

| 能力 | v0.1 状态 | 说明 |
|---|---:|---|
| Einstein parser 与 verifier | 已实现 | 显式输出、两个输入、单字母索引 |
| 静态 shape/stride 语义 IR | 已实现 | FP32、正 input stride、连续 output |
| B/M/N/K 分类 | 已实现 | 支持 batch、自由轴和多个 reduction 轴 |
| 三种执行计划及代价模型 | 已实现 | Direct、GEMM-view、Packed-GEMM |
| 三种计划的可执行语义 | 已实现 | Python/NumPy CPU backend |
| schedule space 与经验调优 | 已实现 | 当前测量 Python CPU backend |
| 持久化 tuning cache | 已实现 | target-aware、文件锁、原子替换 |
| 自定义 `tc.contract` dialect | 已实现 | TableGen op、独立 C++ verifier |
| `tc.contract -> linalg.generic` | 已实现 | 项目自己的 conversion pass |
| Direct 原生 CPU 编译执行 | 已实现 | MLIR 到 LLVM IR、dylib、`ctypes` ABI |
| 原生 artifact cache | 已实现 | 内容寻址、校验和、并发安全 |
| 原生 GEMM-view/Packed-GEMM | 未实现 | v0.1 只原生编译 Direct |
| schedule 驱动原生 loop 变换 | 未实现 | 当前 native lowering 使用通用 loops |
| CUDA/GPU/NVVM backend | 未实现 | 需要在 NVIDIA 机器上继续 |

因此，准确的项目描述是“实现了完整的小型 contraction compiler loop 和 Direct CPU native backend”，而不是“实现了高性能 Tensor Core 编译器”。

## 1. 前端与语义模型

### 1.1 为什么只接受显式二元 einsum

输入形如：

```text
bij,bjk->bik
```

v0.1 有意拒绝隐式输出、ellipsis、对角线、广播、两个以上输入和无 reduction 的 outer product。这样每个合法索引都能落入确定的 B/M/N/K 类别，backend 不必猜测 NumPy 的隐式规则。

解析器不调用 `numpy.einsum` 来定义语义；NumPy 只在测试中充当独立 oracle。这个边界避免 reference library 的行为意外变成 compiler specification。

### 1.2 B/M/N/K 分类

令左、右输入和输出的 label 集合分别为 $L$、$R$、$O$：

$$
\begin{aligned}
B &= L \cap R \cap O \\
M &= (L \cap O) \setminus R \\
N &= (R \cap O) \setminus L \\
K &= (L \cap R) \setminus O
\end{aligned}
$$

- B：batch 轴，同时出现在两个输入和输出；
- M：只属于左输入与输出的自由轴；
- N：只属于右输入与输出的自由轴；
- K：两个输入共享、但不出现在输出中的 reduction 轴。

B/M/N 按输出中的顺序稳定排列，K 按左输入中的顺序排列。多个逻辑轴分别折叠成：

$$
B_s=\prod_{b\in B}d_b,\quad
M_s=\prod_{m\in M}d_m,\quad
N_s=\prod_{n\in N}d_n,\quad
K_s=\prod_{k\in K}d_k
$$

一次 contraction 的计算量按 FMA 计为：

$$
F=2B_sM_sN_sK_s
$$

例如课程 contraction：

```text
aijd,bckd->abcijk
```

分类结果为 $B=\varnothing$、$M=(a,i,j)$、$N=(b,c,k)$、$K=(d)$。虽然它不是物理上已经排好的矩阵，仍可在计划层抽象成 $M_s\times K_s$ 与 $K_s\times N_s$ 的 contraction。

### 1.3 `ContractionProblem` 为什么是核心 IR

`ContractionProblem` 是 frontend、planner 和 backend 共享的不可变语义边界，包含：

- 已验证的 equation；
- 三个 `TensorSpec` 的 shape、element stride 和 dtype；
- B/M/N/K 分组与每个 label 的 extent；
- 折叠后的 B/M/N/K 大小、FLOPs 和稳定 workload key。

它不持有 NumPy array、MLIR value 或设备指针。这样 target-independent 逻辑不会和某个运行时的 buffer ownership 耦合。即使调用者直接构造公开 dataclass，而不走工厂函数，各类也会在 `__post_init__` 重新建立不变量。

## 2. Layout 分析与三种执行计划

### 2.1 GEMM-view 的 legality proof

“数学上等价于 GEMM”不等于“可以零拷贝交给 GEMM”。v0.1 只有满足以下条件时才允许 GEMM-view：

1. 两个输入都是 C-contiguous；
2. 左输入的物理 label 顺序为 `B+M+K` 或 `B+K+M`；
3. 右输入的物理 label 顺序为 `B+K+N` 或 `B+N+K`；
4. 输出已经是连续的规范 `B+M+N` 顺序；
5. 唯一允许的转置是折叠完成后的 whole-matrix transpose。

合法时，backend 先把连续逻辑组 collapse 成 `[B, rows, columns]`，再用 `swapaxes` 表示矩阵转置。实现还用 `shares_memory` 断言 reshape/transpose 没有偷偷 materialize。

这段分析的核心价值不是“会调用 BLAS”，而是给出了 zero-copy 的**可证明条件**，并为非法候选保留可解释原因。

### 2.2 Direct

Direct 保留原始 layout，遍历折叠后的 B/M/N/K，再把 flat index 反解回每个原始 label 的坐标：

```text
for b in B:
  for m_tile in M:
    for n_tile in N:
      for m, n in tile:
        acc = 0
        for k_tile in K:
          for k in tile:
            acc += lhs[index(lhs_labels)] * rhs[index(rhs_labels)]
        out[index(output_labels)] = acc
```

优点是支持所有 v0.1 合法 layout、无需 pack workspace；缺点是高秩 index reconstruction 和通用访存会带来额外开销。Python 实现强调透明语义，并不追求 CPU 峰值性能。

### 2.3 GEMM-view

GEMM-view 在 legality proof 成功后，把输入零拷贝地解释成 batched matrices，然后调用矩阵乘法。它避免 pack/unpack，通常是能用 GEMM 时最便宜的计划。

### 2.4 Packed-GEMM

Packed-GEMM 显式执行：

```text
lhs original -> transpose/copy B+M+K --+
                                        +-> batched GEMM -> optional output unpack
rhs original -> transpose/copy B+K+N --+
```

它为两个输入分配规范连续 workspace；如果输出不是 `B+M+N`，还要把规范 GEMM 结果重新排列。对于经过验证的 v0.1 problem，它始终合法，但必须为 layout traffic 和 workspace 付费。Python backend 当前使用 NumPy 的 transpose/copy 和 `dot` 来表达这些语义；原生 pack kernel 与 BLAS call lowering 尚未实现。

### 2.5 为什么非法计划也保留在候选中

`Planner.enumerate()` 始终返回三种计划，分别携带 `legal`、estimated latency、workspace 和 rationale。选择器只在合法计划中取最优，但 `explain` 仍能回答“为什么没有选 GEMM-view”。这比在优化流程中悄悄丢弃候选更适合调试和面试展示。

## 3. Cost model、schedule 与 autotuning

### 3.1 一阶 cost model

模型是 roofline 风格的候选排序器，不是性能承诺。其结构可概括为：

$$
T_{kernel}=T_{launch}+\max\left(\frac{F}{P},\frac{Q}{BW}\right)+T_{extra}
$$

其中 $F$ 是 FLOPs，$P$ 是目标的有效计算吞吐，$Q$ 是估算字节流量，$BW$ 是有效带宽。

- Direct 使用较低的 direct throughput，并增加高秩 index reconstruction penalty；
- GEMM-view 使用 GEMM throughput，没有 layout materialization；
- Packed-GEMM 额外计入两侧 pack、可选 unpack、更多 launch 和 workspace traffic。

CPU/A100 target model 只是透明的启发式参数。没有真实 A100 测量前，不能把模型输出当作 A100 benchmark。

### 3.2 Direct schedule space

`DirectSchedule` 包含 `block_m`、`block_n`、`block_k`、`threads` 和 `vector_width`。候选先经过静态剪枝：

- threads 不超过 target limit；
- vector width 必须整除折叠后的 K；
- M/K 与 N/K tiles 的估算 shared-memory footprint 不超过预算。

剩余候选按 tile utilization、数据复用和 vector width 的启发式分数排序。CPU target 固定为单线程、标量 vector width；这些字段主要是为后续 GPU backend 保留稳定接口。

需要明确：v0.1 的 Python Direct backend 会消费 B/M/N/K tile；当前 native MLIR loop lowering还没有消费 planner 选出的 schedule。

### 3.3 经验调优

调优器执行以下闭环：

1. 枚举所有合法 plan；
2. 为 Direct 取静态排序后的若干 schedule；
3. 每个候选先与 `numpy.einsum` 做差分正确性检查；
4. warmup 后采集多次 latency；
5. 用 median 而不是 minimum 选择最优候选；
6. 把 plan、schedule、samples 和 median 写入 target-specific cache。

缓存键是 `(target_name, workload_key)`。workload key 覆盖 equation、shape、stride、dtype 和输出 metadata，因此不同 layout 不会错误复用记录；target 单独参与键，CPU 测量也不会覆盖 A100 选择。

## 4. 自定义 MLIR dialect 与 lowering

### 4.1 为什么不从一开始就只生成 `linalg.generic`

`tc.contract` 在一个 op 中保留 contraction 的领域语义：

- 原始 Einstein equation；
- 三个 affine indexing maps；
- parallel/reduction iterator 分类；
- tensor 类型和静态 shape。

如果 frontend 直接展开成 loops，这些信息会过早丢失，后续很难可靠地区分 contraction、验证 B/M/N/K membership 或做计划选择。自定义 dialect 还形成了明确的 Python frontend/C++ compiler 边界。

### 4.2 verifier 检查什么

C++ verifier 不盲目信任 Python frontend，而是独立检查：

- operands/result 都是 ranked、static、FP32 tensor；
- input rank 为 1 到 6，extent 为正，tensor 无 encoding；
- 恰好三个 projected-permutation indexing maps，且不含 symbol；
- map rank、tensor rank、iterator 数量一致；
- result map 恰好覆盖所有 parallel iterators；
- reduction iterator 同时索引两个输入且不索引输出；
- parallel iterator 索引输出及至少一个输入；
- 同一个 loop dimension 在不同 tensor 上 extent 一致；
- 可选 equation 的 label、map 和 iterator semantics 相互一致。

这是 defense in depth：即使另一个 frontend 或手写 MLIR 构造 `tc.contract`，非法 IR 也不能进入 lowering。

### 4.3 `tc.contract -> linalg.generic`

conversion pass 生成：

1. `tensor.empty`；
2. `linalg.fill` 用 FP32 零初始化 reduction identity；
3. 携带原 indexing maps/iterators 的 `linalg.generic`；
4. region 中执行 `mulf` 和 `addf`。

显式 zero fill 非常重要：reduction 的输出不能从未初始化内存开始累加。

### 4.4 原生 Direct pipeline

当前真实可执行路径为：

```text
Einstein frontend
  -> tc.contract
  -> linalg.generic
  -> one-shot bufferization
  -> result-to-out-parameter conversion
  -> linalg/scf/cf loops
  -> LLVM dialect
  -> LLVM IR
  -> clang -O2 shared library
  -> ctypes call
```

`einsumcc-opt` 是项目自己构建的 driver，注册自定义 dialect/pass 与所需 upstream passes；CLI 的 `--stage linalg|llvm|llvm-ir` 展示的是这条真实 pipeline 的中间产物，而不是手工拼接的示例文本。

## 5. Native ABI 与内存所有权

MLIR C wrapper 使用 ranked memref descriptor。对 rank 为 $r$ 的 tensor，Python 侧用 `ctypes.Structure` 构造：

```text
allocated/base pointer
aligned/data pointer
offset
sizes[r]
strides[r]
```

接口选择 out-parameter ownership：Python 分配 lhs、rhs 和 output，generated code 只读输入并写入 output，不把 MLIR/LLVM 分配的 heap object 交回 Python。这避免了跨 ABI allocator ownership 和返回 descriptor 生命周期问题。

调用前 runtime 验证：

- 必须是 NumPy ndarray 和 FP32；
- shape 与编译时 problem 完全一致；
- byte stride 能整除 FP32 item size，element stride 与静态 ABI 一致；
- output 可写且连续；
- output 不与输入 alias，因为 kernel 会先把 output 初始化为零。

输入允许经过编译声明的任意正 stride；scalar output 通过 rank-0 descriptor 覆盖。

## 6. 两类缓存解决的是不同问题

### 6.1 Tuning cache

保存“某个 target 上，这个 workload 测得哪种 plan/schedule 最快”。写入采用稳定 sidecar file lock，在锁内完成 read-modify-write，再通过同目录临时文件、`fsync` 和 `os.replace` 原子发布，避免并发 tuner 丢记录或读到半截 JSON。

### 6.2 Native artifact cache

保存“这段 Direct semantic IR 经这套 pipeline/toolchain 编译出的动态库”。cache identity 覆盖：

- workload key 与生成的 `tc` source hash；
- pipeline revision 和完整 pass 列表；
- optimizer、translator、clang 的路径/metadata；
- host platform 与 machine。

manifest 记录动态库 SHA-256。命中时同时验证 identity 和 binary checksum；损坏条目会在锁内重建。编译先发生在临时目录，完成后再原子安装；两个并发进程针对同一 key 时，只有一个真正构建，另一个随后命中。

面试中可以强调：tuning cache 缓存的是**性能决策**，artifact cache 缓存的是**编译产物**，二者的 invalidation 条件并不相同。

## 7. 测试策略

测试按编译器边界分层：

- parser/problem：合法语法、非法 membership、shape/stride 和不可变式；
- planner/layout：三种 plan、transpose view、非法原因、workspace；
- semantic backend：所有合法 plan 与 NumPy 差分；
- tuning/cache：候选正确性、target 隔离、坏数据拒绝与持久化 round-trip；
- dialect：FileCheck 正向、负向和 `tc -> linalg`；
- native runtime：cache hit/corruption/concurrency、ABI 错误和 alias 检查；
- end-to-end：matrix、batch、permuted layout、scalar output、multiple reductions、课程高秩 contraction 和 positive-stride input 全部走到 arm64 dylib，再与 NumPy 比较。

推荐本地验收：

```bash
make test
make test-mlir
make test-cpu-codegen
make check
```

CPU benchmark 只用于 regression 和验证 benchmark plumbing；Python Direct latency 不能作为优化后 native CPU 性能，更不能作为 A100 性能证据。

## 8. 最值得讲的设计取舍

### 为什么 CPU-first

macOS 没有 CUDA，但 equation legality、layout proof、plan model、IR verifier、lowering semantics、ABI 和缓存一致性都可以先独立验证。这样未来上 A100 时，新增变量主要集中在 GPU mapping 和性能，而不是同时调试整条 compiler stack。

### 为什么同时保留 semantic backend 和 native backend

semantic backend 可读、易差分，覆盖三种 plan；native backend 验证真实 dialect/lowering/ABI。目前两者能力不完全对称是有意暴露的边界，而不是用 Python execution 冒充 native codegen。

### 为什么选择静态 shape

静态 shape 简化了 extent proof、loop lowering、AOT ABI、artifact identity 和测试。动态 shape 很有价值，但会同时引入 runtime shape checks、动态 workspace、specialization policy 和 cache key 设计，不适合 v0.1 一次解决。

### 为什么先实现 Direct native

Direct 不依赖外部 BLAS ABI，能优先打通“自定义 op 到真实 machine code”的完整链路。GEMM lowering 不是简单把名字换成 `matmul`：还需要 call ABI、transpose flags、batched stride、workspace ownership 和 pack/unpack kernel。

## 9. 面试常见追问与回答框架

### “这和 Triton 有什么区别？”

Triton 提供通用 GPU kernel language、编程模型和优化 lowering；EinsumCC 是 contraction-specific compiler。二者都涉及 tile、layout、cost model 和 lower-level codegen，但 v0.1 的抽象输入是 Einstein contraction，优化空间首先是 plan/layout 选择，而不是让用户手写 program ID 和 block kernel。

### “你真正做了哪些编译优化？”

v0.1 已实现的是 plan-level optimization：证明零拷贝 GEMM 是否合法，在 Direct、view 和 materialize-then-GEMM 之间按 compute、traffic、launch、workspace 选择，并提供 empirical override。它还没有宣称 native tiling/vectorization 或 GPU Tensor Core 优化；当前 native Direct 是正确性完整、性能优化仍可继续的 baseline。

### “为什么不能所有 contraction 都 reshape 成 GEMM？”

reshape 只有在物理 stride 允许连续 group collapse 时才是 view。任意 label permutation 可能需要数据搬运；输出顺序不规范也需要 scatter/unpack。忽略 stride 只看 shape 会把隐式 copy 错当成零成本优化，甚至产生错误解释。

### “为什么需要自定义 dialect？”

它保留领域语义、提供独立 verifier、建立 frontend/backend contract，并为以后在语义还清晰时做 plan selection 或 specialized lowering 留出位置；同时仍能降到 upstream `linalg` 复用成熟 pass infrastructure。

### “最难的工程问题是什么？”

可从三点回答：

1. layout legality 必须区分数学等价和物理零拷贝；
2. MLIR 到 Python 的 memref ABI 需要明确 descriptor、stride 和内存所有权；
3. native cache 不只要 hash，还要处理 toolchain invalidation、损坏产物和多进程并发发布。

### “怎么证明正确？”

说明分层验证：Python frontend 与 C++ dialect 双重 verifier；三种独立执行路径逐候选对 NumPy；IR 正负测试；代表性 contraction 从 DSL 走到 native dylib；ABI/cache failure path 也有测试。性能结论则与正确性结论分开，必须在真实 target 上测。

## 10. 简历表述参考

可以写成：

> Built a CPU-first tensor-contraction compiler for explicit Einstein notation, including B/M/N/K analysis, layout-aware selection among Direct/zero-copy GEMM/packed GEMM plans, a verified MLIR dialect and lowering pipeline to native arm64 code, plus ranked-memref runtime and process-safe content-addressed caches.

如果后续没有完成 GPU backend，不要写 CUDA/Tensor Core code generation；可以写“designed extension points for GPU/NVVM lowering”。如果 native schedule 尚未接入，也不要写“implemented MLIR tiling/vectorization”。

## 11. v0.1 之后的合理路线

建议按“先让优化真实影响 machine code，再扩展 target”的顺序推进：

1. 让 `DirectSchedule` 驱动 native loop tiling/interchange，并检查生成 IR；
2. 增加 native CPU benchmark，对比 generic Direct、scheduled Direct 和 BLAS；
3. 实现 native GEMM-view call lowering，再实现 pack/unpack；
4. 将计划选择和 native executable 真正连接起来；
5. 上 NVIDIA 环境实现 `gpu`/`nvvm` Direct baseline；
6. 加入 shared-memory tiling、coalescing、vectorization，再在 A100 上 autotune；
7. 最后扩展 FP16/BF16、mixed accumulation、Tensor Core 和 epilogue fusion。

最有价值的下一个里程碑不是扩大 DSL 语法，而是闭合这条性能链路：

```text
selected plan + selected schedule
          -> observable IR transformation
          -> target executable
          -> reproducible benchmark
          -> measured cache record
```

## 12. 代码复习地图

| 主题 | 入口 |
|---|---|
| Parser 与索引分类 | `src/einsumcc/equation.py` |
| Semantic IR/workload key | `src/einsumcc/problem.py` |
| Zero-copy legality | `src/einsumcc/layout.py` |
| 三种 plan 与 cost model | `src/einsumcc/plans.py` |
| Schedule space | `src/einsumcc/schedule.py` |
| Python 三计划语义 | `src/einsumcc/cpu_backend.py` |
| Tuner 与决策缓存 | `src/einsumcc/tuner.py`, `cache.py` |
| `tc.contract` frontend emission | `src/einsumcc/tc_emitter.py` |
| Dialect verifier | `lib/Dialect/TC/IR/TCOps.cpp` |
| `tc -> linalg` pass | `lib/Conversion/TCToLinalg/TCToLinalg.cpp` |
| Native pipeline/ABI/artifact cache | `src/einsumcc/native_backend.py` |
| 公开 façade | `src/einsumcc/compiler.py` |
| CLI | `src/einsumcc/cli.py` |

复习时应能不看代码解释清楚四条主线：**索引如何分类、layout 为什么合法或非法、语义怎样逐层 lowering、生成代码怎样安全地被 Python 调用和缓存**。
