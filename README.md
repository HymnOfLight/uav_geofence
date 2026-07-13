# 无人机电子围栏：INT8 QNN 形式化验证实验代码

本代码包对应《形式化方法在无人机电子围栏中的应用：面向单人研究与单工作站条件的可执行实验设计与评估方案》。核心闭环是：

1. 用安全教师控制器生成状态—动作数据；
2. 训练 2×16 / 2×32 小型 ReLU MLP；
3. 导出具有明确舍入、饱和和整数累加语义的 INT8 QNN；
4. 用区间预筛 + Z3 对整数网络做 E0 语义一致性与 E1 单步安全包络验证；
5. 用区间可达性做 E2 有限时域闭环验证；
6. 用统计仿真比较教师、FP32、INT8、INT8+安全屏蔽；
7. 输出 SAFE / UNSAFE / UNKNOWN、真实反例复核、置信区间和运行成本。

代码不依赖真机、PX4、HIL、大型视觉网络或完整 EKF/UKF 证明。默认 `smoke.yaml` 可在普通 CPU 上完成；`main.yaml` 按 RTX 5090、25 vCPU、92 GB RAM 的单工作站条件配置 16 个 Z3 进程。

## 1. 快速运行

Linux / AutoDL / Ubuntu：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
python -m geofence_qnn.cli all --config configs/smoke.yaml --output runs/smoke
python -m unittest discover -s tests -v
```

Windows PowerShell：

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
python -m geofence_qnn.cli all --config configs/smoke.yaml --output runs/smoke
python -m unittest discover -s tests -v
```

冒烟实验应生成：

```text
runs/smoke/
├── float_model.npz
├── int8_model.npz
├── training_history.csv
├── training_summary.json
├── e0_consistency.csv
├── e0_summary.json
├── e1_cells.csv
├── e1_summary.json
├── e2_reachability.csv
├── e2_summary.json
├── monte_carlo_summary.csv
├── monte_carlo_summary.json
└── all_summary.json
```

## 2. 分阶段运行

```bash
# 训练 FP32 MLP 并导出整数 QNN
python -m geofence_qnn.cli train --config configs/smoke.yaml --output runs/smoke

# E0：NumPy INT8 与 Z3 编码逐输入一致性
python -m geofence_qnn.cli e0 --config configs/smoke.yaml --output runs/smoke

# E1：边界状态单元上的精确 SMT 安全包络验证
python -m geofence_qnn.cli e1 --config configs/smoke.yaml --output runs/smoke

# E2：有限时域闭环区间可达性和自适应分区
python -m geofence_qnn.cli e2 --config configs/smoke.yaml --output runs/smoke

# E3/E4：统计仿真、安全屏蔽、风扰和定位误差
python -m geofence_qnn.cli mc --config configs/smoke.yaml --output runs/smoke
```

完整实验步骤、验收标准、主实验矩阵和论文出图方法见 [EXPERIMENT_STEPS.md](EXPERIMENT_STEPS.md)。

汇总五种子结果并生成基础图：

```bash
python -m pip install -e '.[plots]'
python scripts/aggregate_results.py --runs runs/sweep --output runs/aggregate --plots
```

`smoke.yaml` 的作用是检查整条证据流水线，样本数、训练轮数和轨迹数都被刻意压缩；其安全率、任务成功率和 UNKNOWN 比例不得作为论文结论。正式结果必须使用多种子主配置。

## 3. 研究边界

- `SAFE`：在明确的输入单元或初始集合、扰动界、时域和模型下，过近似集合不触及扩张后的禁飞区。
- `UNSAFE`：E1 中表示 Z3 找到违反动作半空间的整数输入；只有 `replay_found_violation=true` 才是当前连续状态单元中被随机回放复现的反例。E2 只有当整个可达盒已位于禁飞区内时才直接标为 UNSAFE。
- `UNKNOWN`：超时或可达盒与禁飞区相交但无法证明真实穿越。UNKNOWN 绝不计入安全。
- 当前 E2 是声称保守的区间可达性原型，不是 Flow*/dReach 的替代品；它适合形成可运行的第一篇原型和后续接入更强后端的接口。

## 4. 代码结构

```text
src/geofence_qnn/
├── config.py          # YAML → 强类型配置
├── geometry.py        # 禁飞矩形、距离与区间相交
├── dynamics.py        # 二维双积分器与区间传播
├── features.py        # 状态到 6 维仿射特征
├── controller.py      # PID/CBF 风格安全教师
├── data.py            # 数据采样
├── model.py           # NumPy MLP 与 Adam 训练
├── quantization.py    # INT8/INT32、舍入、饱和、区间推理
├── smt.py             # Z3 精确整数网络编码
├── verification.py    # E1 状态单元验证与反例回放
├── reachability.py    # E2 自适应闭环可达性
├── simulation.py      # 蒙特卡洛与运行时安全屏蔽
├── io_utils.py        # JSON/CSV、哈希与可复现产物
└── cli.py             # all/train/e0/e1/e2/mc 命令
```

## 5. 实现约定

- 所有层共享 `qscale=32`，输入、权重和激活使用有符号整数；隐藏层 ReLU 后饱和到 `[0,127]`，输出饱和到 `[-127,127]`。
- 偏置按 `qscale²` 存入 INT32；累加后使用“远离零的半舍入”除以 `qscale`。
- E1 先用声称保守的整数区间界判定显然 SAFE/UNSAFE 的单元，只把未决单元送入 Z3；`method` 字段区分 `interval_prefilter` 与 `z3_exact`。
- Z3 与 NumPy 使用同一舍入公式；E0 必须达到 100% 一致，否则流水线中止。
- 输出动作是归一化值乘 `amax`；SMT 性质直接在整数输出上验证，避免浮点歧义。
- 主实验应保留配置、模型 SHA-256、环境版本、逐单元 CSV 和汇总 JSON。

## 6. 许可与使用方式

这是研究原型，不是适航认证软件。可在论文与课题原型中修改使用；发表时应明确模型、状态范围、扰动界、验证时域和 UNKNOWN 的处理方式。
