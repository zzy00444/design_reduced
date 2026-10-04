# 精简版 INT8 GEMM（100 MHz）

有效源代码位于 `design_reduced/`。Vitis HLS 2023.2 工程为 `gemm_reduced_100mhz/solution1`，目标器件为 `xczu5eg-sfvc784-1-e`，目标时钟周期为 10 ns。原 `gemmhls_gemini` 工程元数据仍保留，可供对照。

## 结构与接口

| 参数 | 原设计 | 精简版 |
|---|---:|---:|
| PE 阵列 | 4x4 | 2x2 |
| N/M 外层分块 | 256x256 | 128x128 |
| 每个 PE 的 M 并行通道 | 8 | 8 |
| 每个 PE 的 K 并行通道 | 4 | 4 |
| AXI 读/写突发长度上限 | 256 | 64 |
| AXI 读/写未完成事务数（outstanding） | 64 | 8 |
| AXI 数据字宽 | 128 位 | 128 位 |

计算为 `C[n,m] = sum_k A[k,n] * B[k,m]`，PE 使用 32 位有符号累加器。标量 DATAFLOW 流的深度显式设为 16。按每次乘加计两次运算，理论峰值为每周期 128 次 INT8 MAC，即 100 MHz 下 0.0256 TOPS；这不是板级实测吞吐率。

顶层新增 AXI-Lite 开关 `output_int32`，导出 IP 的控制寄存器偏移为 `0x5c`。C 端口在两种模式下均保持 128 位字宽：

| `output_int32` | 输出数据 | 每个 128 位字 | C 每行步长 | 后处理 |
|---:|---|---:|---|---|
| `0` | UINT8 | 16 个元素 | `ceil(size_m/16)` 字 | 重定量、饱和；`enable_gelu` 控制 GELU |
| `1` | 有符号 INT32 | 4 个元素 | `ceil(size_m/4)` 字 | 直接输出累加器，跳过重定量和 GELU |

每个字的第 0 个元素位于最低有效位。INT32 模式下 `enable_gelu` 和 `layer_idx` 不参与输出计算；调用方仍需按所选模式分配 C 缓冲区并解释数据。例如 `N=M=129` 的 INT32 输出需要 `129 * ceil(129/4) = 4257` 个 128 位字。A、B 的物理布局分别为 `[K][ceil(N/16)]`、`[K][ceil(M/16)]`，每字各含 16 个 INT8 元素。

HLS 的 A/B AXI 指针 `depth=4096`，C 指针 `depth=8192`，测试平台按这些深度分配内存，以满足 RTL 协同仿真的拷贝要求。此深度注解不是生成 RTL 的运行时矩阵尺寸限制。

## 构建与验证

在 `F:/zky/gemmhls_gemini` 目录运行：

```powershell
& 'D:\Vivado2023\Vitis_HLS\2023.2\bin\vitis_hls.bat' -f build_reduced_100mhz.tcl
& 'D:\Vivado2023\Vitis_HLS\2023.2\bin\vitis_hls.bat' -f verify_reduced_rtl.tcl
& 'D:\Vivado2023\Vivado\2023.2\bin\vivado.bat' -mode batch -source validate_reduced_ip.tcl
```

构建脚本执行 C 仿真、C 综合和 Verilog IP 导出；验证脚本使用 XSIM 进行 Verilog C/RTL 协同仿真；Vivado 脚本将解包后的 IP 目录加入内存工程，并检查 VLNV。

## 本次结果

| 检查项 | 结果 |
|---|---|
| C 仿真 | 通过，7 个用例均无元素不匹配 |
| Verilog C/RTL 协同仿真 | `Pass`，7 个事务均无元素不匹配 |
| HLS 目标周期 | 10 ns（100 MHz） |
| HLS 估计周期 | 7.30 ns（估计 136.99 MHz） |
| BRAM_18K | 138 / 288 |
| URAM | 24 / 64 |
| LUT | 42,574 / 117,120 |
| FF | 43,074 / 234,240 |
| DSP | 121 / 1,248 |
| Vivado 2023.2 IP 目录识别 | 通过，`xilinx.com:hls:MatrixMultiplicationKernelInt8:1.0` |

测试用例覆盖 `(N,M,K)=(16,16,7)、(17,19,5)、(129,129,5)、(129,129,7)`，并交替运行 UINT8 与 INT32 模式。它们检查非 16 对齐的行步长、K 尾部补零、跨越 128 分块、两层参数、GELU 开关，以及连续启动时双向切换输出模式。最终协同仿真报告记录的 Verilog 延迟最小值、平均值和最大值分别为 5,750、15,689、38,030 周期。

报告文件：

- `gemm_reduced_100mhz/solution1/syn/report/csynth.rpt`
- `gemm_reduced_100mhz/solution1/syn/report/csynth.xml`
- `gemm_reduced_100mhz/solution1/sim/report/MatrixMultiplicationKernelInt8_cosim.rpt`

Vivado IP 产物：

- 压缩包：`gemm_reduced_100mhz/solution1/impl/export.zip`
- 解包后的 IP 目录：`gemm_reduced_100mhz/solution1/impl/ip`
- VLNV：`xilinx.com:hls:MatrixMultiplicationKernelInt8:1.0`
- 压缩包 SHA-256：`605B647989645512BACCE42DE62C9F8D89B95C727CA1CBA3BED145B0D44B7C8A`

## 集成注意事项

调用时要求 `size_k > 0`；原有零 K 缓冲区问题尚未处理。UINT8 模式下 `layer_idx` 必须在 `[0,35]`，代码未进行越界检查。INT32 模式输出当前 32 位累加器的结果，未增加防溢出的更宽累加器。每个有效输出元素占用的字节数是 UINT8 模式的 4 倍；按 128 位字逐行补齐后，实际缓冲区比例取决于 `size_m`。

上述时序和资源数字均为 HLS 估算值，尚未进行 Vivado 布局布线或板级性能测试。A/B 读取推断出 AXI 突发传输，条件式 C 写回没有推断出突发传输。Vitis 仍报告非规范嵌套 DATAFLOW 等告警，实际系统集成时需检查时序与 DDR 行为。
