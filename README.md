# 无人机电子围栏：INT8 QNN 形式化验证实验代码

本代码包对应《形式化方法在无人机电子围栏中的应用：面向单人研究与单工作站条件的可执行实验设计与评估方案》。核心闭环是：

1. 用安全教师控制器生成状态—动作数据，或直接从 PX4 Autopilot / ArduPilot 飞行日志与 SITL 采集数据；
2. 训练 2×16 / 2×32 小型 ReLU MLP；
3. 导出具有明确舍入、饱和和整数累加语义的 INT8 QNN；
4. 用区间预筛 + Z3 对整数网络做 E0 语义一致性与 E1 单步安全包络验证；
5. 用区间可达性做 E2 有限时域闭环验证；
6. 用统计仿真比较教师（内置 / PX4 / ArduPilot 围栏基线）、FP32、INT8、INT8+安全屏蔽；
7. 输出 SAFE / UNSAFE / UNKNOWN、真实反例复核、置信区间和运行成本。

核心验证流水线不依赖真机、HIL、大型视觉网络或完整 EKF/UKF 证明；PX4/ArduPilot 集成是可选层（数据来源、教师后端、仿真基线与 SITL 采集），见下文第 3 节。默认 `smoke.yaml` 可在普通 CPU 上完成；`main.yaml` 按 RTX 5090、25 vCPU、92 GB RAM 的单工作站条件配置 16 个 Z3 进程。

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

## 3. 开源飞控支持（PX4 Autopilot / ArduPilot）

实验不再仅限于合成数据和自训练网络，提供三条与主流开源飞控对接的路径（依赖可选安装：`python -m pip install -e '.[flightstack]'`，包含 `pyulog` 与 `pymavlink`）：

### 3.1 真实飞行日志作为训练数据

**最短路径（真实开源日志，两条命令）**——自动从 [logs.px4.io](https://logs.px4.io) 公开数据库抓取真实四旋翼飞行并跑完整流水线：

```bash
python scripts/fetch_px4_logs.py --count 5 --dest logs/px4
python -m geofence_qnn.cli all --config configs/px4_public_logs.yaml --output runs/px4_public
```

抓取脚本按机型（四旋翼）、时长和飞行模式（Position）过滤最近的公开日志，逐个下载并实际解析验证（无位置数据或原地不动的日志自动跳过）。`configs/px4_public_logs.yaml` 中的 `data.auto_align: true` 会把每条航迹平移到围栏几何上（只平移位置，速度、加速度、时间不动），免去逐文件手调 `data.offset`。发表时数据来源引用 logs.px4.io。

`data.logs` 也可直接填 http(s) URL（如 Flight Review 的下载链接），首次运行自动下载并缓存到 `logs/_downloads/`，之后离线复跑：

```yaml
data:
  source: px4_ulog
  logs: ["https://logs.px4.io/download?log=<uuid>&type=0"]
  auto_align: true
```

`data.source` 支持四种取值：

| source | 说明 | 依赖 |
|---|---|---|
| `synthetic` | 教师控制器合成数据（默认，兼容旧配置） | 无 |
| `px4_ulog` | PX4 `.ulg` 日志，默认读 `vehicle_local_position` | pyulog |
| `ardupilot_log` | ArduPilot DataFlash `.bin`/`.log`（XKF1/NKF1）或 MAVLink `.tlog`（LOCAL_POSITION_NED） | pymavlink |
| `csv` | 通用 CSV 轨迹 `t,x,y,vx,vy[,ax,ay][,episode]`（含本包 SITL 录制输出） | 无 |

NED 日志自动映射到实验平面坐标（x=东、y=北）；对齐用 `data.auto_align: true`（推荐）或手动 `data.offset`；动作优先取日志加速度，否则对重采样速度差分，再按 `amax` 截断归一化。真实航迹不会绕我们的虚拟围栏飞行，所以 `data.synthetic_fraction`（`px4_public_logs.yaml` 默认 0.5）混入教师样本提供避障行为与边界带覆盖。其他模板：`configs/px4_ulog.yaml`、`configs/ardupilot_log.yaml`（自备日志文件）。

完全离线时，可用行为教师生成演示 CSV 日志走通同一条摄取链路：

```bash
python scripts/make_demo_logs.py --config configs/demo_csv.yaml --backend px4 --output logs/demo
python -m geofence_qnn.cli all --config configs/demo_csv.yaml --output runs/demo_csv
```

演示数据仍是合成数据（行为级模型 rollout），论文中不得写成真机或固件结果。

### 3.2 PX4 / ArduPilot 围栏行为教师与仿真基线

`teacher.backend` 支持 `builtin`（原教师）、`px4`（MPC 位置环级联 + `GF_PREDICT` 预测刹车/外推持稳）、`ardupilot`（AC_Avoid 平方根控制器限速滑移 + `AVOID_BACKUP_SPD` 退避），参数名与相应飞控参数一一对应（`MPC_XY_P`、`MPC_DEC_HOR_MAX`、`FENCE_MARGIN`、`AVOID_ACCEL_MAX` 等），可经 `teacher.params` 覆盖。`simulation.controllers` 可加入 `px4`、`ardupilot` 作为蒙特卡洛对照组：

```bash
python -m geofence_qnn.cli all --config configs/smoke_flightstack.yaml --output runs/smoke_fs
```

注意：这两个教师是对官方文档所述围栏逻辑的行为级建模（在二维双积分器上），不是固件本身；论文中必须如实标注。固件级数据请用 3.1 的日志或 3.3 的 SITL。

### 3.3 从运行中的 SITL 录制轨迹

`sitl-record` 通过 MAVLink 连接正在运行的 PX4 SITL（`make px4_sitl jmavsim`）或 ArduPilot SITL（`sim_vehicle.py -v ArduCopter`），录制 `LOCAL_POSITION_NED` 到 CSV（自动转世界系），随后用 `data.source: csv` 训练：

```bash
python -m geofence_qnn.cli sitl-record --config configs/smoke.yaml --output runs/sitl \
  --url udp:127.0.0.1:14550 --episodes 5 --duration 120 --rate 20
```

录制器不改飞行模式、不解锁；任务、围栏上传与模式切换由操作者（QGroundControl / MAVProxy）负责。`--command-goal` 可选地向配置目标点流式发送位置设定值（需 OFFBOARD/GUIDED 模式）。

## 4. 研究边界

- `SAFE`：在明确的输入单元或初始集合、扰动界、时域和模型下，过近似集合不触及扩张后的禁飞区。
- `UNSAFE`：E1 中表示 Z3 找到违反动作半空间的整数输入；只有 `replay_found_violation=true` 才是当前连续状态单元中被随机回放复现的反例。E2 只有当整个可达盒已位于禁飞区内时才直接标为 UNSAFE。
- `UNKNOWN`：超时或可达盒与禁飞区相交但无法证明真实穿越。UNKNOWN 绝不计入安全。
- 当前 E2 是声称保守的区间可达性原型，不是 Flow*/dReach 的替代品；它适合形成可运行的第一篇原型和后续接入更强后端的接口。

## 5. 代码结构

```text
src/geofence_qnn/
├── config.py          # YAML → 强类型配置（data/teacher/controllers 节可选）
├── geometry.py        # 禁飞矩形、距离与区间相交
├── dynamics.py        # 二维双积分器与区间传播
├── features.py        # 状态到 6 维仿射特征
├── controller.py      # PID/CBF 风格安全教师
├── data.py            # 数据采样与数据源分发（合成/飞行日志）
├── flightstack/       # 开源飞控集成
│   ├── teachers.py    #   PX4 / ArduPilot 围栏行为教师与工厂
│   ├── logs.py        #   ULog / DataFlash / tlog / CSV → 状态-动作数据集
│   └── sitl.py        #   MAVLink SITL 轨迹录制器
├── model.py           # NumPy MLP 与 Adam 训练
├── quantization.py    # INT8/INT32、舍入、饱和、区间推理
├── smt.py             # Z3 精确整数网络编码
├── verification.py    # E1 状态单元验证与反例回放
├── reachability.py    # E2 自适应闭环可达性
├── simulation.py      # 蒙特卡洛与运行时安全屏蔽
├── io_utils.py        # JSON/CSV、哈希与可复现产物
└── cli.py             # all/train/e0/e1/e2/mc/sitl-record 命令
```

## 6. 实现约定

- 所有层共享 `qscale=32`，输入、权重和激活使用有符号整数；隐藏层 ReLU 后饱和到 `[0,127]`，输出饱和到 `[-127,127]`。
- 偏置按 `qscale²` 存入 INT32；累加后使用“远离零的半舍入”除以 `qscale`。
- E1 先用声称保守的整数区间界判定显然 SAFE/UNSAFE 的单元，只把未决单元送入 Z3；`method` 字段区分 `interval_prefilter` 与 `z3_exact`。
- Z3 与 NumPy 使用同一舍入公式；E0 必须达到 100% 一致，否则流水线中止。
- 输出动作是归一化值乘 `amax`；SMT 性质直接在整数输出上验证，避免浮点歧义。
- 主实验应保留配置、模型 SHA-256、环境版本、逐单元 CSV 和汇总 JSON。

## 7. 许可与使用方式

这是研究原型，不是适航认证软件。可在论文与课题原型中修改使用；发表时应明确模型、状态范围、扰动界、验证时域和 UNKNOWN 的处理方式。
