# A5 TLA 集成验证结论

四个 PR（Triton #2982、#3267、#3284 和 AscendNPU-IR #5251）的**配套集成版**在 Ascend950DT_9582 上通过 **54/54 项实机测试**。原生构建和 **10/10 项独立离线编译**通过。实机结果包含随附 Patch 的三项兼容修复，合入前需要回填到对应代码。

## 实机覆盖

| 项目 | 通过 | 覆盖和结果 |
| --- | ---: | --- |
| MMAD L0 | 28 | FP16/BF16、单次与 K 分块累加、单缓冲与 ping-pong、L0 容量边界、动态 clear 及子视图。L1→L0 在 MMAD 路径中验证。最大绝对误差 3.05e-5。 |
| AIC/AIV 同步 | 12 | 手动同步、SyncSolver v1/v2。输出与 CPU 参考精确相等，GM guard 通过。 |
| Cube DMA | 3 | UB→L1→UB，宽度 16/32/64，结果与参考精确相等。 |
| Fixpipe | 10 | L0C→GM/UB/L1，覆盖样本中的 ND/NZ/DN、FP32/FP16/BF16 和 ReLU。30 次补测最大绝对误差 1.91e-6。 |
| SIMD | 1 | FP16↔FP32 带掩码转换，与参考精确相等。 |

## PR 需要补齐的改动

- **#2982 / #3267**：#3267 移除 Fixpipe 配置字段后，#2982 的 MMAD 测试仍有六处传入旧字段 unit_flag_ctrl。迁移调用，保留 MMAD 的 UnitFlag 和原有同步。
- **#3284 / #5251**：#3284 与 #2982 的 MMAD 提交只合入一次；L1→L0 builder 与 #5251 配套。更新 #3284 中两处已移除的片上 view 调用，改用类型化 buffer 和 subview，并保留 stride、offset 检查。
- **最新后端兼容**：HIVM 的 VFTruncFOp、VFExtFOp builder 需显式传入缺省 mode；否则 Triton 原生编译失败。此修复已通过带掩码转换实测。

## 待补验证

- L1→L0 需要进一步验证规范内的 pitch、offset、subview 和对齐边界组合。当前实机证据来自 MMAD 路径，另有独立的 IR 和 C++ 检查。
- Fixpipe 的合法模式组合，以及多 AIV 共享 L1 的 buffer 生命周期压力测试尚未覆盖完整。L1→BT/bias 仍缺端到端支持。
- MMAD L0 的规范条目仍待评审。 [规范分支](https://github.com/oshy516/tla-spec/tree/a5-tla-spec-review-20261119-v2)

[详细报告](reports/A5_TLA_PR_Integration_Report.md) · [三项增量修复 Patch](patches/A5_TLA_Local_Fixes.patch)
