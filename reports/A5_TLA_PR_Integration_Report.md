# A5 TLA PR 集成报告

2026-11-19，baai-ysh-A5 / wmg_flagtree。工作根目录为 /home/ysh-workspace/.isolated/a5-tla-validation。

**本次 Triton Ascend 原生扩展和配套 AscendNPU-IR 构建均通过。使用配套编译器在 Ascend950DT_9582 实机执行了 54 个用例，全部通过。** 此外，10 个 MMAD 配置通过独立离线二进制编译，#2982 发布包作为对照也通过同规模的实机矩阵。首次 WSL 阶段的 SSH 认证和 VSCode 窗口不可用已通过恢复现有 Remote SSH 终端解决。交付文件及日志在本工作根目录。

## 环境与版本

Ubuntu 22.04.5 aarch64，Python 3.12.16，Git 2.34.1，Clang 15.0.7，CANN 9.2.1，npu-smi 25.1.rc2 识别 8 张 Ascend950DT。独立 venv 通过 system-site-packages 复用 torch 2.10.0+cpu、torch_npu 2.10.0.post7 和 pytest 8.4.2，并安装 CMake 3.31.10。系统 BishengIR 2.0.0 efb7141581c0。最终使用独立安装目录下的 BishengIR 2.0.0 79a36dc7729b，LLVM 19.1.7 57b2ed73d65a；Triton 为 triton_ascend-3.6.0.dev0+git02b93345。源码、venv、缓存、构建产物和临时目录在本隔离目录。设备用例串行运行，每组之前通过 npu-smi 确认明确空闲的设备。环境冒烟在物理卡 2 上，FP32 加法最大绝对误差 0。详见 logs/environment.log、logs/build-deps.log、logs/npu-initial.log 和 logs/torch-npu-smoke-2.log。

## 独立集成分支

| 仓库 | 本次获取的基线 | 本地最终 SHA |
| --- | --- | --- |
| Triton Ascend main-dev | 0227cdebc7940f38cbe67007110a44fe2434ec94 | 02b933457ce1a0a93b3d0625077619b9270a479b |
| AscendNPU-IR upstream/master | 0167d41d7c4b6115de5fb72f17fc2834fe3f3871 | 79a36dc7729b4163cd296098f52992eeda0bcb74 |

两个代码仓分支均为 a5-tla-integration-20261119，未推送。四个 PR 在检查时均为 open。

| PR | 原 head | 本地集成 |
| --- | --- | --- |
| [#2982](https://github.com/triton-lang/triton-ascend/pull/2982) | e53ed579d8efc227a499fcf2c78abe3e45ce2950 | 0029fff74，MMAD L0 |
| [#3267](https://github.com/triton-lang/triton-ascend/pull/3267) | 64bfdc63bc8b690813651db6e682f43a985e2ab8 | ea4ceacd5，保留简化后的 DMA/Fixpipe 参数 |
| [#3284](https://github.com/triton-lang/triton-ascend/pull/3284) | d0bf025df6be8390f8038c5f5ac4baeb58675379 | b18a3d31a，只摘取独有 L1→L0 变更 |
| [#5251](https://gitcode.com/Ascend/AscendNPU-IR/pull/5251) | d04e61b57b7258f0bd97c9f2b7ef035c658dcde3 | 79a36dc7729b，新增简化的 HIVM L12L0 builder |

#3284 父提交与 #2982 的稳定 patch-id 均为 77fc6725b9f0f38306961dcc6fa9d07ca8f87132，MMAD 只合入一次。C++ 冲突保留 main-dev 的动态 GM view/subview 以及 #3284 的 bind/reinterpret。DMA 测试冲突保留 #3267 的参数负例，将已删除的 actual_batch 移到构造失败检查。Triton gitlink 从基线 bd228faec3b574ce116edb54ad8437355d630523 配套到 79a36dc7729b4163cd296098f52992eeda0bcb74，排除 PR 中的旧 1389120d7732cbb333033e9d8b34e1853b6cb8ea。后端 LLVM 精确 pin 是 57b2ed73d65a8f4dbf2f20dfaeaf718b5ec0a17d。#5251 的 PR 来源是 qq_52736072 的 Fork，用户指定的 Estella516 Fork 用作 origin。详见 logs/remote-integration.log。

## 构建及分层验证

| 层次 | 最终结果 | 日志 |
| --- | --- | --- |
| Python | 81 项 Recorder 参数与负例、2 项新增源码回归通过 | logs/remote-python-contract.log、logs/remote-integration.log |
| Native IR | 73 项通过，另直接调用 DMA 和 L1→L0 两组合约，含逻辑布局、子视图、偏移、transpose 和非法参数 | logs/native-ir-direct.log、logs/native-ir-*.py.log、logs/matched-dma-extra-contracts.log |
| 后端 C++ 与 IR | 4 项 #5251 builder C++ 测试，以及布局推导、HIVM lowering、PlanMemory 3 项检查通过 | logs/pr5251-builder-gtest.log、logs/pr5251-native-lld-checks.log |
| 离线 Compile | 10 个 MMAD FP16/BF16 配置生成 TTIR、HIVM 和 A5 ELF，用配套 BishengIR 的 Ascend950 32 AIC / 64 AIV 目标配置 | logs/offline-mmad-inputs.log、logs/offline-mmads/、logs/target-stages-lld.log |
| A5 Runtime / Accuracy | 实际执行并比对参考值，54 项全通过 | 下表 |

旧系统编译器缺少 Ascend950 目标和 --enable-tla-mode，相关控制见 logs/system-compiler-smoke/。源码构建所用 ccec 和 llvm-link 位于 CANN 的 aarch64-linux/bin。GNU ld 最终阶段未解析 setSanitizerAddrArgName，nm 确认其定义在调用方之前的静态归档；用仓库支持的 LLVM_ENABLE_LLD=ON 重链接并安装，后端源码无需额外修改。Triton 原生构建还需要 C++ 可选参数修复及在实际子模块路径检出配套后端。详见 logs/npuir-build-resumed.log、logs/npuir-build-lld.log、logs/triton-build-matched-submodule.log。

## 配套工具链 A5 实测

| 组 | 通过 | 配置与精度 |
| --- | ---: | --- |
| MMAD L0 | 10 | FP16/BF16；(M,N,K,L0_K) 包含 (32,64,32,32)、(32,64,96,32)、(32,64,128,32)、(32,64,144,48)、(16,16,16,16)；单次与 K 分块 |
| 扩展 MMAD L0 | 18 | 单缓冲、ping-pong、64 KiB L0A/B 和 256 KiB L0C 边界、M/N/K 2048、动态 clear、静态和动态子视图。两个 MMAD 组最大绝对误差 3.0517578125e-5，参考为 CPU 上同精度量化输入的 FP32 累加 |
| AIC/AIV 同步 | 12 | manual、auto_v1、auto_v2。每个配置 12 次，两个输出与 CPU 参考精确相等，GM guard 检查通过 |
| Cube DMA | 3 | UB→L1→UB 宽度 16/32/64，输出与参考精确相等。L1→L0 在 MMAD 数据路径覆盖，独立目的轴和布局由 C++ 与 IR 检查覆盖 |
| Fixpipe | 10 | L0C→GM/UB/L1，按 PR 合法样本覆盖 ND/NZ/DN、FP32/FP16/BF16、ReLU。另以 seed 7、42、123 做 30 次参考对比，最大绝对误差 1.9073486328125e-6，非零参考点最大相对误差 1.2130180948588531e-5 |
| SIMD 转换 | 1 | FP16↔FP32 带掩码，与参考精确相等 |

两个 MMAD 组、同步和 Cube DMA 实测在物理卡 3。Fixpipe、SIMD 和 Fixpipe 误差补测在物理卡 4。首轮在两组 MMAD 后检测到卡 3 已占用，脚本主动停止；每组重新确认空闲卡后，剩余组全部通过。详见 logs/a5-runtime-cube-mmad_l0*.py.log、logs/matched-tail-summary.log、logs/matched-tail-*.py.log、logs/matched-sync-tail/ 和 logs/matched-fixpipe-metrics.log。

对照使用 #2982 发布包 0bfb79a16bee，归档 SHA256 为 6ae11823d873c021725751bd1777b4f07d9c80aa8cb8993b5a69993166e54207。4 个离线 ELF 和 54 个实机样本均通过。首次对照选到了 PATH 的系统旧编译器，前置正确 bin 后重跑。详见 logs/pr-package-compile-counts.log、logs/pr-package-runtime-retry.log、logs/pr-matrix-summary.log、logs/pr-matrix-*.py.log 和 logs/pr-fixpipe-metrics.log。

## 本地增量与根因

1. #3267 删除 Fixpipe 配置中的 unit_flag_ctrl 后，迁移三份 MMAD 测试的六处旧调用。保留 MMAD 自身的 UnitFlag 和所有事件、barrier。
2. 最新 main-dev 的 tla.view 是 GM pointer API。将 #3284 定向测试中的两处片上 tla.view(space=...) 迁移为类型化 L1/L0A buffer 和 tla.subview，保留原 stride/offset 验证。添加两项回归，迁移前均失败，迁移后均通过。
3. 最新 HIVM 对 VFTruncFOp、VFExtFOp 的 builder 增加可选 mode 参数，原调用无法通过编译。显式传入缺省属性，由原有 applySimdMode 流程处理。重建后通过 native IR 和实机掩码转换用例。

最终补丁从集成基点 56a065db7f2ba7bdea8511275c25925b2807c48b 导出，只包含上述集成后新增修复，不包含 PR 或 gitlink 配套提交。git apply --check 在从该基点取出的干净文件上通过。SHA256 f531b98afb64aa8f5a82307a7da5c1087547ef13401b8538065fa0ea71617454。验证日志 logs/final-patch-check.log。

## 规范对照与当前范围

审查基于 tla-spec upstream/main 的 9bbe909f27250bf9c169d2bbd8747bbebce15e9a，已含 PR #14 merge d8b835f07f482284d09171fb23416f3d759936b2。读取 spec/cube.md、spec/dma.md、spec/memory.md 和三份对应 tracking。审查聚焦 Python API、C++ 传参、HIVM IR、目标编译和样本结果。

- MMAD L0 当前通过 FP16/BF16、FP32 累加及已列形状，静态负例和实际分块/同步均有覆盖。L0 单块容量校验还需和多缓冲总占用规划配合使用。L1→BT/bias 路径没有公开端到端支持证据。
- UB→L1 和 L1→UB 使用 NZ 物理布局。完整 NZ、各有效起始地址、行距和行宽的 32B 对齐、AIV 共享 L1 写入不重叠仍须按规范满足。见样例覆盖，完整 stride、offset 和多 AIV 组合尚未穷举。
- L1→L0 的公开范围为 FP16/BF16、规范 4D NZ 视图。IR 接受的 rank-2 或 FP32 不计入公开能力。保留内层 (256,16,1)、外层 fractal pitch、对齐和容量约束。任意外层 pitch、偏移和子视图的独立硬件组合仍需扩充。
- Fixpipe 仍受目的连续性、layout、mode、quant、ReLU、dual/sub-block 组合限制，任意目的子视图或额外 pitch 无规范支持。L0C→L1 的 FP32 C0=16 结果不能用 reshape 作为 C0=8 的 Cube 输入。下一轮优先补齐规范内的对齐边界、合法 mode 组合及跨核 buffer 生命周期压力测试。
- 显式事件与自动 SyncSolver v1/v2 在所列矩阵通过。设备覆盖限 Ascend950DT_9582，未扩展到其他 SKU 或未测试的并行调度。

## TLA Spec 修改与推送

现有 DMA 文档已经引用 MMAD L0，Cube 文档缺独立入口。README.md 补索引；spec/cube.md 记录候选签名、返回与 IR 边界；tracking/cube.md 加 CUBE-008 和仍需评审项。文档保留待评审状态和既有 DMA 约束。隔离 clone 的 upstream 为 patchliu/tla-spec，origin 为 oshy516/tla-spec。独立分支 a5-tla-spec-review-20261119-v2 已推至 Fork，本地及远端 SHA 均为 5df7910cfe91f0026ea83b71f1dcc34c2fad56bf，父提交为上述 upstream/main。已用 ls-remote 核对。见 logs/spec-fork-identity-verification.log 和 [Fork 分支](https://github.com/oshy516/tla-spec/tree/a5-tla-spec-review-20261119-v2)。

## 交付路径

- /home/ysh-workspace/.isolated/a5-tla-validation/reports/A5_TLA_PR_Integration_Report.md
- /home/ysh-workspace/.isolated/a5-tla-validation/patches/A5_TLA_Local_Fixes.patch
- /home/ysh-workspace/.isolated/a5-tla-validation/logs/
