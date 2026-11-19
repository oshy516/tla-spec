# Cube接口规格

[返回接口目录](../README.md)

以下 DMA、L0 搬运和填充约束面向 Ascend950（RegBase）。

<a name="api-tla-cube-mmad-l0"></a>

## tla.cube.mmad_l0

**评审中，不转测。** 本节补齐显式 L0 乘累加入口的索引；候选参数和目标约束仍须单独确认，不能由 DMA-004 的结论推定为已确认规格。

PR #2982 中的公开签名：`tla.cube.mmad_l0(a, b, acc, *, m, n, k, bias=None, clear_accum=False, disable_gemv=True, unit_flag_ctrl=None)`。

返回：None。候选编译表示：`hivm.hir.mmadL0`。该操作直接消费 L0A/L0B 并写入 L0C，不隐式执行 L1→L0 搬运或 K 分块循环。

当前只有 PR 源码证据，本条不代表 native IR、编译或 A5 精度验证通过。参数范围、物理布局、容量、offset 和同步的实现证据及待决问题见 [CUBE-008](../tracking/cube.md#cube-008)。既有 [L1→BT 的关闭结论](#api-tla-copy-l1-to-bt)保持有效，PR 中存在 bias 入口不等于本轮已开放 BT 链路。

源码依据：[PR #2982 cube.py](https://github.com/triton-lang/triton-ascend/blob/e53ed579d8efc227a499fcf2c78abe3e45ce2950/third_party/ascend/language/tla/cube.py#L267)。

<a name="api-tla-copy-ub-to-l1"></a>

## tla.copy_ub_to_l1

签名：`tla.copy_ub_to_l1(src, dst)`

返回：None。编译表示：`hivm.hir.copy`。

| 参数名 | 参数含义 | 参数类型 | 合法参数范围 | 其他约束 | 默认值 |
| --- | --- | --- | --- | --- | --- |
| `src` | 源 | tla.buffer | UB；float16 / bfloat16 | 保持准备好的物理布局，不自动 ND2NZ。 | 必填 |
| `dst` | 目的 | tla.buffer | L1；与 src 相同 | 形状与布局须匹配。 | 必填 |

支持 4D NZ 物理拷贝及 2D 物理视图子块拷贝；末维连续，源和目的形状相容。源、目的有效起始地址（含 offset）、相邻搬运行的字节间距及每行搬运字节数均须为 32B 的整数倍；FP16/BF16 对应 16 个元素。NZ 子块还须保持完整的物理分块。偏移、行间距及实际搬运范围不得导致越界；不提供非对齐搬运回退。

该接口不执行 ND2NZ，调用方须先准备目标物理布局。多个 AIV 子块写共享 L1 时必须使用互不重叠的目的范围，或只由一个子块写入；AIV 写入后供 AIC 消费必须建立跨核同步。完成同步不消除多个写者之间的重叠写竞争。

约束：暂不支持手动设置 `pad_mode` 和 `pad_value`。

```python
@tla.jit
def example(src, dst):
    with tla.scope.vector():
        tla.copy_ub_to_l1(src, dst)
```

仓库测试：[tla_ut/dma/copy.py](https://github.com/triton-lang/triton-ascend/blob/074cf62907f815694a01e5cdc4ac78a90610ae82/third_party/ascend/unittest/tla_ut/dma/copy.py#L19)。

<a name="api-tla-copy-l1-to-ub"></a>

## tla.copy_l1_to_ub

签名：`tla.copy_l1_to_ub(src, dst)`

返回：None。编译表示：`hivm.hir.l12ub`。

| 参数名 | 参数含义 | 参数类型 | 合法参数范围 | 其他约束 | 默认值 |
| --- | --- | --- | --- | --- | --- |
| `src` | 源 | tla.buffer | L1；float16 / bfloat16 | 4D NZ 物理布局。 | 必填 |
| `dst` | 目的 | tla.buffer | UB；与 src 相同 | 2D ND；逻辑元素数一致。 | 必填 |

允许末维连续且满足对齐的行间距和偏移子视图。目的 UB 的有效起始地址（含 offset）及行间距须为 32B 的整数倍；FP16/BF16 对应 16 个元素。L1 源子块须保持完整 NZ 分块，源和目的范围均不得越界。不提供非对齐搬运回退。

当前路径写入 sub-block 0 的 UB，由 AIC 发出 MTE1→AIV 的跨核同步后，由对应子块读取。不提供其他 dtype、5D 批量或非连续末维。

```python
@tla.jit
def example(src, dst):
    with tla.scope.cube():
        tla.copy_l1_to_ub(src, dst)
```

仓库测试：[tla_ut/dma/copy.py](https://github.com/triton-lang/triton-ascend/blob/074cf62907f815694a01e5cdc4ac78a90610ae82/third_party/ascend/unittest/tla_ut/dma/copy.py#L30)。

<a name="api-tla-copy-l1-to-bt"></a>

## tla.copy_l1_to_bt

**本轮不需要，暂不开放。** 保留入口供问题追溯：`tla.copy_l1_to_bt(src, dst, params=None)`，IR 为 `hivm.hir.l12bt`。

当前不提供可用参数组合。原生操作要求常量 n∈[1,1024]，源和目的容量不少于 roundUp(n,16)，L1 地址按 32B、BT 地址按 64B 对齐，末维 stride=1；生产和消费之间须建立 MTE1→M 依赖。

仓库 IR 构造用例：[tla_ut/dma/copy.py](https://github.com/triton-lang/triton-ascend/blob/074cf62907f815694a01e5cdc4ac78a90610ae82/third_party/ascend/unittest/tla_ut/dma/copy.py#L45)。

<a name="api-tla-copy-l1-to-fb"></a>

## tla.copy_l1_to_fb

**本轮不需要。** 入口直接抛出 NotImplementedError。保留规格供追溯。

签名：`tla.copy_l1_to_fb(src, dst)`

**预留入口。** 当前调用会抛出 NotImplementedError。

| 参数名 | 参数含义 | 参数类型 | 合法参数范围 | 其他约束 | 默认值 |
| --- | --- | --- | --- | --- | --- |
| `src` | 待搬运的源缓冲区 | tla.buffer | 地址空间：L1<br>数据类型：无可执行签名 | 当前入口抛 NotImplementedError，不生成 DMA op。 | 必填 |
| `dst` | 接收数据的目标缓冲区 | tla.buffer | 地址空间：FB<br>数据类型：无可执行签名 | 当前入口抛 NotImplementedError，不生成 DMA op。 | 必填 |

<a name="api-tla-copy-l0c-to-ub"></a>

## tla.copy_l0c_to_ub

签名：`tla.copy_l0c_to_ub(src, dst, params=None)`

返回：None。

| 参数名 | 参数含义 | 参数类型 | 合法参数范围 | 其他约束 | 默认值 |
| --- | --- | --- | --- | --- | --- |
| `src` | 累加结果 | tla.buffer | L0C；float32 | 4D NZ；NZ2NZ 还须满足下方来源连续性约束；不得读取未初始化累加器。 | 必填 |
| `dst` | 目的 | tla.buffer | UB；float32 / float16 / bfloat16 | 仅限连续物理布局、offset=0 的直接分配或等价 reshape；还须符合本接口的组合限制。 | 必填 |
| `params` | 搬运配置 | tla.FixpipeL0C2UBParams / None | 本接口列出的配置组合 | None 使用下列字段的默认值。 | None |
| `params.dma_mode` | 目的布局 | tla.FixpipeDMAMode | NZ2ND / NZ2NZ / NZ2DN | 与 dst 的 rank、shape 配套。 | NZ2NZ |
| `params.pre_quant` | 类型转换 | tla.FixpipePreQuantMode | NO_QUANT / F322F16 / F322BF16 | 分别对应 FP32 / FP16 / BF16；还须满足下方组合限制。 | NO_QUANT |
| `params.pre_relu` | 激活 | tla.FixpipePreReluMode | NO_RELU / NORMAL_RELU | 可与本接口列出的布局、类型转换组合。 | NO_RELU |
| `params.dual_dst_mode` | AIV 目的切分 | tla.FixpipeDualDstMode | NO_DUAL / ROW_SPLIT / COLUMN_SPLIT | 双目标仅支持 NZ2ND。 | NO_DUAL |
| `params.sub_block_idx` | 单目标子块 | tla.FixpipeSubBlock / None | SUB_BLOCK_0 / SUB_BLOCK_1 / None | 双目标必须显式设为 None，不能保留默认值。 | SUB_BLOCK_0 |

**组合限制（Ascend950）**

`pre_quant` 与目的元素类型必须一一对应：`NO_QUANT→FP32`、`F322F16→FP16`、`F322BF16→BF16`。以下组合均允许 `pre_relu=NO_RELU` 或 `NORMAL_RELU`。

| dma_mode | pre_quant | dual_dst_mode | sub_block_idx | 每个目的 UB 的形状 |
| --- | --- | --- | --- | --- |
| NZ2ND | NO_QUANT / F322F16 / F322BF16 | NO_DUAL | SUB_BLOCK_0 或 SUB_BLOCK_1 | 2D (M,N) |
| NZ2NZ | NO_QUANT / F322F16 / F322BF16 | NO_DUAL | SUB_BLOCK_0 或 SUB_BLOCK_1 | 与源相容的 4D NZ |
| NZ2DN | NO_QUANT / F322F16 / F322BF16 | NO_DUAL | SUB_BLOCK_0 或 SUB_BLOCK_1 | 2D (N,M) |
| NZ2ND | NO_QUANT / F322F16 / F322BF16 | ROW_SPLIT | 必须显式 None | 2D (M/2,N) |
| NZ2ND | NO_QUANT / F322F16 / F322BF16 | COLUMN_SPLIT | 必须显式 None | 2D (M,N/2) |

逻辑 M、N 为 16 的正整数倍，源和目的均不得超过各自存储空间容量。ROW_SPLIT 要求 M 可均分；COLUMN_SPLIT 要求 N 为 32 的倍数。双目标不支持 NZ2NZ/NZ2DN，也不能同时指定单目标子块。

当 `dma_mode=NZ2NZ` 时，来源必须为连续的 4D NZ 物理布局，内层为 `(16,16)`，以元素为单位的 stride 必须为 `(shape[1] × 256, 256, 16, 1)`。暂不支持保留父 buffer 额外间距的来源子视图；不能仅修改 shape 或 stride 描述来替代实际数据重排。此项来源连续性限制仅适用于 NZ2NZ。

目的 UB 仅提供上述形状对应的连续物理布局，使用直接 `tla.alloc` 或保持连续、零偏移的 `tla.reshape` 结果。暂不支持目的 `subview`、非零或动态 offset、带额外行间距的目的视图；满足 32B 对齐也不自动获得这些视图的支持。

AIC 完成写入后必须建立跨核依赖，再由对应 AIV 子块读取。单目标仅由指定子块消费；双目标分别消费对应行分块或列分块。

内部保留 `quant_scale=None`、`unit_flag_ctrl=None`、`unit_flag_group_id=None` 的默认行为；None 表示属性缺省，不等同于 DISABLED。这些字段不作为本轮可配置参数。

约束：暂不支持 `QF322F32_PRE`、手动设置 `quant_scale`、`LEAKY_RELU/P_RELU`、整数/FP8、5D 批量及手动 UnitFlag。

约束：双目标仅支持 NZ2ND，不支持其他目的布局。

仓库测试：[tla_ut/cube/gemm_bias.py](https://github.com/triton-lang/triton-ascend/blob/074cf62907f815694a01e5cdc4ac78a90610ae82/third_party/ascend/unittest/tla_ut/cube/gemm_bias.py#L35)。

<a name="api-tla-copy-l0c-to-l1"></a>

## tla.copy_l0c_to_l1

签名：`tla.copy_l0c_to_l1(src, dst, params=None)`

返回：None。

| 参数名 | 参数含义 | 参数类型 | 合法参数范围 | 其他约束 | 默认值 |
| --- | --- | --- | --- | --- | --- |
| `src` | 累加结果 | tla.buffer | L0C；float32 | 4D NZ；不得读取未初始化累加器。 | 必填 |
| `dst` | 目的 | tla.buffer | L1；float32 / float16 / bfloat16 | 形状、布局与类型须符合本接口的组合限制。 | 必填 |
| `params` | 搬运配置 | tla.FixpipeL0C2L1Params / None | 本接口列出的配置组合 | None 使用下列字段的默认值。 | None |
| `params.dma_mode` | 目的布局 | tla.FixpipeDMAMode | NZ2ND / NZ2NZ / NZ2DN | 与 dst 的 rank、shape 配套。 | NZ2NZ |
| `params.pre_quant` | 类型转换 | tla.FixpipePreQuantMode | NO_QUANT / F322F16 / F322BF16 | 分别对应 FP32 / FP16 / BF16；还须满足下方组合限制。 | NO_QUANT |
| `params.pre_relu` | 激活 | tla.FixpipePreReluMode | NO_RELU / NORMAL_RELU | 可与本接口列出的布局、类型转换组合。 | NO_RELU |

**组合限制（Ascend950）**

`pre_quant` 与目的元素类型必须一一对应：`NO_QUANT→FP32`、`F322F16→FP16`、`F322BF16→BF16`。以下组合均允许 `pre_relu=NO_RELU` 或 `NORMAL_RELU`。

| dma_mode | pre_quant | 目的类型 | 目的形状 |
| --- | --- | --- | --- |
| NZ2ND | NO_QUANT / F322F16 / F322BF16 | FP32 / FP16 / BF16，按转换模式对应 | 2D (M,N) |
| NZ2NZ | NO_QUANT / F322F16 / F322BF16 | FP32 / FP16 / BF16，按转换模式对应 | 与源相容的 4D NZ |
| NZ2DN | F322F16 / F322BF16 | 仅 FP16 / BF16 | 2D (N,M) |

**NZ2DN + NO_QUANT→FP32 不开放。** 本接口不提供 `dual_dst_mode` 或 `sub_block_idx`。

逻辑 M、N 为 16 的正整数倍，源和目的均不得超过各自存储空间容量。ND/DN 输出不能未经布局转换直接作为 NZ 输入。调用方须保证写入完成后再由后续运算读取，并保证缓冲区复用的依赖关系。

内部保留 `quant_scale=None`、`unit_flag_ctrl=None`、`unit_flag_group_id=None` 的默认行为；None 表示属性缺省，不等同于 DISABLED。这些字段不作为本轮可配置参数。

另外固定 `channel_split=False`、`c0_pad_en=True`，不提供配置变体。

约束：`NZ2NZ + NO_QUANT→FP32` 在上述固定配置下输出 C0=16 的 NZ 物理布局，对逻辑 `(M,N)` 使用 `(N/16, M/16, 16, 16)` 描述。该输出不能直接作为要求 C0=8 布局的 FP32 Cube 输入；须先完成实际布局转换。`reshape` 或修改视图元数据不会重排数据，也不能解除此限制。

约束：暂不支持 `QF322F32_PRE`、手动设置 `quant_scale`、`LEAKY_RELU/P_RELU`、整数/FP8、5D 批量及手动 UnitFlag。

约束：暂不支持手动设置 `channel_split` 和 `c0_pad_en`。

仓库测试：[tla_ut/dma/fixpipe.py](https://github.com/triton-lang/triton-ascend/blob/074cf62907f815694a01e5cdc4ac78a90610ae82/third_party/ascend/unittest/tla_ut/dma/fixpipe.py#L69)。

<a name="api-tla-store-l0c-to-gm"></a>

## tla.store_l0c_to_gm

签名：`tla.store_l0c_to_gm(src, dst, params=None)`

返回：None。

| 参数名 | 参数含义 | 参数类型 | 合法参数范围 | 其他约束 | 默认值 |
| --- | --- | --- | --- | --- | --- |
| `src` | 累加结果 | tla.buffer | L0C；float32 | 4D NZ；不得读取未初始化累加器。 | 必填 |
| `dst` | 目的 | tla.buffer | GM；float32 / float16 / bfloat16 | 形状、布局与类型须符合本接口的组合限制。 | 必填 |
| `params` | 搬运配置 | tla.FixpipeL0C2GMParams / None | 本接口列出的配置组合 | None 使用下列字段的默认值。 | None |
| `params.dma_mode` | 目的布局 | tla.FixpipeDMAMode | NZ2ND / NZ2NZ / NZ2DN | 与 dst 的 rank、shape 配套。 | NZ2NZ |
| `params.pre_quant` | 类型转换 | tla.FixpipePreQuantMode | NO_QUANT / F322F16 / F322BF16 | 分别对应 FP32 / FP16 / BF16；还须满足下方组合限制。 | NO_QUANT |
| `params.pre_relu` | 激活 | tla.FixpipePreReluMode | NO_RELU / NORMAL_RELU | 可与本接口列出的布局、类型转换组合。 | NO_RELU |

**组合限制（Ascend950）**

`pre_quant` 与目的元素类型必须一一对应：`NO_QUANT→FP32`、`F322F16→FP16`、`F322BF16→BF16`。以下组合均允许 `pre_relu=NO_RELU` 或 `NORMAL_RELU`。

| dma_mode | pre_quant | 目的类型 | 目的形状 |
| --- | --- | --- | --- |
| NZ2ND | NO_QUANT / F322F16 / F322BF16 | FP32 / FP16 / BF16，按转换模式对应 | 2D (M,N) |
| NZ2NZ | NO_QUANT / F322F16 / F322BF16 | FP32 / FP16 / BF16，按转换模式对应 | 与源相容的 4D NZ |
| NZ2DN | NO_QUANT / F322F16 / F322BF16 | FP32 / FP16 / BF16，按转换模式对应 | 2D (N,M) |

本接口不提供 `dual_dst_mode` 或 `sub_block_idx`。逻辑 M、N 为 16 的正整数倍，源和目的均不得超过各自存储空间容量。目的描述符的布局、形状及元素类型必须与所选转换一致；调用方须保证源累加器在搬运完成前不被改写，且后续消费者在写入完成后读取。

内部保留 `quant_scale=None`、`unit_flag_ctrl=None`、`unit_flag_group_id=None` 的默认行为；None 表示属性缺省，不等同于 DISABLED。这些字段不作为本轮可配置参数。

约束：暂不支持 `QF322F32_PRE`、手动设置 `quant_scale`、`LEAKY_RELU/P_RELU`、整数/FP8、5D 批量及手动 UnitFlag。

仓库测试：[tla_ut/cube/allocation.py](https://github.com/triton-lang/triton-ascend/blob/074cf62907f815694a01e5cdc4ac78a90610ae82/third_party/ascend/unittest/tla_ut/cube/allocation.py#L47)。

<a name="api-tla-cube-mmad-mx"></a>

## tla.cube.mmad_mx

签名：`tla.cube.mmad_mx(a, b, scale_a, scale_b, acc, *, format_a=None, format_b=None, transpose_a=False, transpose_b=False, clear_accum=False, bias=None, real_m=None, real_k=None, real_n=None)`

返回：None。

编译表示：hivm.hir.mmadmxL1。

| 参数名 | 参数含义 | 参数类型 | 合法参数范围 | 其他约束 | 默认值 |
| --- | --- | --- | --- | --- | --- |
| `a` | 第一个操作数 | tla.buffer | 地址空间：L1<br>数据类型：<br>B8：float8e4m3fn / float8e5m2 / int8 / uint8<br>B16：不支持<br>B32：不支持<br>B64：不支持 | 元素按 FP8 格式或 packed E2M1 解读。双边须同为 FP4 或同为 FP8。 采用逻辑 2D 或物理 4D NZ。 | 必填 |
| `b` | 第二个操作数 | tla.buffer | 地址空间：L1<br>数据类型：<br>B8：float8e4m3fn / float8e5m2 / int8 / uint8<br>B16：不支持<br>B32：不支持<br>B64：不支持 | 元素按 FP8 格式或 packed E2M1 解读。双边须同为 FP4 或同为 FP8。 采用逻辑 2D 或物理 4D NZ。 | 必填 |
| `scale_a` | A 的 E8M0 块缩放表 | tla.buffer | 地址空间：L1<br>数据类型：int8 / uint8（E8M0 编码） | 每 32 个逻辑 K 取一个因子，逻辑形状 (M, K/32)。4D zZ/nN 的奇数组数对齐到偶数。 | 必填 |
| `scale_b` | B 的 E8M0 块缩放表 | tla.buffer | 地址空间：L1<br>数据类型：int8 / uint8（E8M0 编码） | 每 32 个逻辑 K 取一个因子，逻辑形状 (N, K/32)。4D zZ/nN 的奇数组数对齐到偶数。 | 必填 |
| `acc` | 累加器 | tla.buffer | 地址空间：L0C<br>数据类型：<br>B8：不支持<br>B16：不支持<br>B32：float32<br>B64：不支持 | 逻辑形状须等于 (real_m, real_n)。clear_accum=False 时读取已有累加值。 | 必填 |
| `format_a` | A 的 MX 编码格式 | tla.MxFormat 或 None | None / E5M2 / E4M3 / E2M1 | None 仅能从带类型的 FP8 推断。E2M1 每字节装两个值。 | None |
| `format_b` | B 的 MX 编码格式 | tla.MxFormat 或 None | None / E5M2 / E4M3 / E2M1 | None 仅能从带类型的 FP8 推断。E2M1 每字节装两个值。 | None |
| `transpose_a` | 是否在乘法的逻辑索引中转置 A | 编译期 bool | False / True | 转置后须满足逻辑形状。 | False |
| `transpose_b` | 是否在乘法的逻辑索引中转置 B | 编译期 bool | False / True | 转置后须满足逻辑形状。 | False |
| `clear_accum` | 是否从零开始计算 | 编译期 bool | False / True | 为 True 时从零开始，False 累加到已有值。 | False |
| `bias` | 可选的逐通道偏置 | tla.buffer 或 None | None 或 L1 buffer<br>数据类型：同 acc | 形状 (N,)、(1,N) 或等价 4D zN。 | None |
| `real_m` | 实际计算的 M 维长度 | 编译期 int 或 None | None 或 16 的正整数倍，不超过对应 tile 维度 | None 使用全长。 | None |
| `real_k` | 实际使用的 K 维长度 | 编译期 int 或 None | None 或 16 的正整数倍，不超过对应 tile 维度 | None 使用全长。MX 的 packed FP4 real_k 用字节计，逻辑 K 按每字节两个值计算。 | None |
| `real_n` | 实际计算的 N 维长度 | 编译期 int 或 None | None 或 16 的正整数倍，不超过对应 tile 维度 | None 使用全长。 | None |

**调用示例**

Ascend950PR。输入为布局相容的 FP8 L1 tile，缩放表为 E8M0，累加缓冲区为 float32 L0C。

```python
@tla.jit
def example(a_l1, b_l1, scale_a, scale_b, acc_l0c):
    with tla.scope.cube():
        tla.cube.mmad_mx(a_l1, b_l1, scale_a, scale_b, acc_l0c, clear_accum=True)
```

<a name="api-tla-scope-cube"></a>

## tla.scope.cube

签名：`tla.scope.cube()`

返回：无独立返回值。通过 with 声明内核区域。

将区域标记为 Cube 核模式。区域之间的跨核数据交接需要显式同步。

**调用示例**

```python
@tla.jit
def example(a_l1, b_l1, acc_l0c):
    with tla.scope.cube():
        tla.cube.mmad(a_l1, b_l1, acc_l0c, clear_accum=True)
```

仓库测试：[tla_ut/dma/ctrl.py](https://github.com/triton-lang/triton-ascend/blob/074cf62907f815694a01e5cdc4ac78a90610ae82/third_party/ascend/unittest/tla_ut/dma/ctrl.py#L19)。

<a name="api-tla-cube-mmad"></a>

## tla.cube.mmad

签名：`tla.cube.mmad(a, b, acc, transpose_a=False, transpose_b=False, clear_accum=False, unit_flag_ctrl=None, *, unit_flag_group_id=None, real_m=None, real_k=None, real_n=None)`

返回：None。

编译表示：hivm.hir.mmadL1。

| 参数名 | 参数含义 | 参数类型 | 合法参数范围 | 其他约束 | 默认值 |
| --- | --- | --- | --- | --- | --- |
| `a` | 第一个操作数 | tla.buffer | 地址空间：L1<br>数据类型：已确认 float16 / bfloat16 / float32 / int8 的常规签名，详见其他约束 | a/b 元素类型相同，转置后的 K 相等。浮点输入累加到 float32，常规 int8 输入累加到 int32。 采用逻辑 2D 或物理 4D NZ。 | 必填 |
| `b` | 第二个操作数 | tla.buffer | 地址空间：L1<br>数据类型：已确认 float16 / bfloat16 / float32 / int8 的常规签名，详见其他约束 | a/b 元素类型相同，转置后的 K 相等。浮点输入累加到 float32，常规 int8 输入累加到 int32。 采用逻辑 2D 或物理 4D NZ。 | 必填 |
| `acc` | 累加器 | tla.buffer | 地址空间：L0C<br>数据类型：<br>B8：不支持<br>B16：不支持<br>B32：int32 / float32<br>B64：不支持 | 逻辑形状须等于 (real_m, real_n)。clear_accum=False 时读取已有累加值。 | 必填 |
| `transpose_a` | 是否在乘法的逻辑索引中转置 A | 编译期 bool | False / True | 转置后须满足逻辑形状。 | False |
| `transpose_b` | 是否在乘法的逻辑索引中转置 B | 编译期 bool | False / True | 转置后须满足逻辑形状。 | False |
| `clear_accum` | 是否从零开始计算 | 编译期 bool | False / True | 为 True 时从零开始，False 累加到已有值。 | False |
| `unit_flag_ctrl` | 选择累加器 UnitFlag 的控制方式 | 编译期 tla.UnitFlagMode / int / None | None / 0（DISABLED）/ 2（ENABLED_WITHOUT_UPDATE）/ 3（ENABLED_WITH_UPDATE） | 1 保留。None 省略属性。手动模式可用 unit_flag=False 关闭自动推断。 | None |
| `unit_flag_group_id` | 指定配对操作的 UnitFlag 组号 | 编译期 int 或 None | None 或 −2⁶³ 至 2⁶³−1（IR 编码） | 与生产者及消费者协议一致。编码域不代表芯片可用 ID 集。 | None |
| `real_m` | 实际计算的 M 维长度 | 编译期 int 或 None | None 或 16 的正整数倍，不超过对应 tile 维度 | None 使用全长。 | None |
| `real_k` | 实际使用的 K 维长度 | 编译期 int 或 None | None 或 16 的正整数倍，不超过对应 tile 维度 | None 使用全长。 | None |
| `real_n` | 实际计算的 N 维长度 | 编译期 int 或 None | None 或 16 的正整数倍，不超过对应 tile 维度 | None 使用全长。 | None |

**调用示例**

A、B 为布局相容的 L1 tile，例如 float16 输入，acc 为 float32 L0C。

```python
@tla.jit
def example(a_l1, b_l1, acc_l0c):
    with tla.scope.cube():
        tla.cube.mmad(a_l1, b_l1, acc_l0c, clear_accum=True)
```

仓库测试：[tla_ut/cube/gemm_bias.py](https://github.com/triton-lang/triton-ascend/blob/074cf62907f815694a01e5cdc4ac78a90610ae82/third_party/ascend/unittest/tla_ut/cube/gemm_bias.py#L34)。

<a name="api-tla-set-ctrl"></a>

## tla.set_ctrl

签名：`tla.set_ctrl(enable, idx)`

返回：None。

编译表示：hivm.hir.set_ctrl。

| 参数名 | 参数含义 | 参数类型 | 合法参数范围 | 其他约束 | 默认值 |
| --- | --- | --- | --- | --- | --- |
| `enable` | 是否置位 | 编译期 bool | False / True | — | 必填 |
| `idx` | CTRL bit 索引 | 编译期 int | 0 至 63（64 位 CTRL 的 bit 索引） | 仅使用目标定义的位。不会自动恢复状态。 | 必填 |

**调用示例**

示例为当前表中记录的 RegBase HF32 位。

```python
@tla.jit
def example():
    with tla.scope.cube():
        tla.set_ctrl(True, 46)
```

仓库测试：[tla_ut/dma/ctrl.py](https://github.com/triton-lang/triton-ascend/blob/074cf62907f815694a01e5cdc4ac78a90610ae82/third_party/ascend/unittest/tla_ut/dma/ctrl.py#L20)。
