# 详细实验步骤与评估协议

## 一、实验目标与最小完成标准

实验目标不是证明任意真实无人机绝对安全，而是在明确模型边界内完成一条可审计证据链：

```text
教师控制器 → FP32 MLP → INT8 QNN → 整数语义一致性
          → 单步 SMT 包络 → 闭环可达性 → 反例回放/安全屏蔽
```

最小完成标准：

1. E0 的 NumPy INT8 与 Z3 编码在全部固定输入上 100% 一致；
2. 至少完成 2×16 和 2×32 两档网络、S1/S2 两类边界场景；
3. E1 输出逐单元 SAFE/UNSAFE/UNKNOWN，候选反例有独立回放字段；
4. E2 报告按初始集合体积加权的 SAFE/UNSAFE/UNKNOWN；
5. E3 比较 FP32、INT8、INT8+屏蔽，零观测事故只报告 Wilson 区间；
6. 所有表格可由保存的 CSV/JSON 重建。

## 二、阶段 0：环境建立与冒烟测试

### 步骤 0.1：建立独立环境

```bash
cd uav_geofence_qnn_experiments
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

不要在现有大模型环境里直接安装，避免 PyTorch/CUDA/flash-attn 依赖冲突。该代码核心只需要 NumPy、PyYAML 和 z3-solver。

### 步骤 0.2：运行单元测试

```bash
python -m unittest discover -s tests -v
```

通过条件：全部测试为 `OK`。如 `test_smt_fixed_input` 失败，停止所有批量实验。

### 步骤 0.3：运行完整冒烟流水线

```bash
python -m geofence_qnn.cli all \
  --config configs/smoke.yaml \
  --output runs/smoke
```

建议先打开：

- `runs/smoke/e0_summary.json`：`pass` 必须为 `true`；
- `runs/smoke/e1_summary.json`：检查 SAFE、UNSAFE candidate 和 UNKNOWN 是否都有合理数值；
- `runs/smoke/e2_summary.json`：确认体积比例之和约为 1；
- `runs/smoke/monte_carlo_summary.csv`：确认四种控制器均有结果。

## 三、阶段 1：数据生成、教师与 QNN 训练

### 步骤 1.1：固定物理与几何参数

主配置 `configs/main.yaml` 中：

- 控制周期 `dt=0.05 s`，高精度回放积分步长 `0.01 s`；
- 最大速度 `8 m/s`，最大加速度 `4 m/s²`；
- 禁飞区 `[-5,5] × [-15,15] m`；
- 安全裕度 `1 m`；
- 输入特征为目标相对位置、禁飞区中心相对位置和二维速度，共 6 维。

第一轮禁止同时修改全部参数。先固定上述模型，得到可复现主结果，再做误差敏感性。

### 步骤 1.2：生成教师数据

`data.py` 从禁飞区外采样状态；教师由 `teacher.backend` 决定：`builtin`（目标吸引、速度阻尼和边界排斥）、`px4`（PX4 位置环级联 + GF_PREDICT 预测刹车）或 `ardupilot`（AC_Avoid 平方根限速滑移）。训练/测试按固定顺序 80%/20% 切分。

若使用真实飞行数据而非合成数据，把 `data.source` 设为 `px4_ulog`、`ardupilot_log` 或 `csv` 并给出 `data.logs`，详见第十四节。

运行：

```bash
python -m geofence_qnn.cli train \
  --config configs/main.yaml \
  --output runs/main_seed42_2x32
```

检查 `training_summary.json`：

- `float_test_mse`：FP32 模仿误差；
- `int8_test_mse`：量化后误差；
- 两个模型的 SHA-256：论文重现时使用。

### 步骤 1.3：训练 5 个种子与三档网络

主种子：`42, 123, 456, 789, 2024`。

网络档位：

- A：`[16,16]`，必须完成；
- B：`[32,32]`，必须完成；
- C：`[32,32,32]`，只作为扩展性边界。

运行自动扫描：

```bash
python scripts/run_seed_sweep.py \
  --base-config configs/main.yaml \
  --output-root runs/sweep \
  --seeds 42 123 456 789 2024 \
  --networks 16x16 32x32 32x32x32 \
  --stage train
```

若 C 档网络在 E1 中超过 50% 单元超时，停止继续扩大网络，把它保留为可扩展性负结果。

## 四、阶段 2（E0）：精确整数语义一致性

### 步骤 2.1：理解整数语义

对每层：

1. 输入与权重乘 `qscale=32` 后半舍入并截断到 `[-127,127]`；
2. INT8×INT8 在 INT32 中累加；
3. 偏置按 `qscale²` 编码；
4. 累加值除以 32，采用远离零的半舍入；
5. 隐藏层 ReLU 后截断到 `[0,127]`，输出层截断到 `[-127,127]`。

### 步骤 2.2：运行一致性验证

```bash
python -m geofence_qnn.cli e0 \
  --config configs/main.yaml \
  --output runs/main_seed42_2x32
```

程序对随机整数输入同时执行 NumPy 解释器与 Z3 符号网络，并询问“是否存在输出与 NumPy 期望不同”。结果必须为 UNSAT。

### 步骤 2.3：验收

`e0_summary.json` 中：

```json
{"inputs": 1000, "consistent": 1000, "inconsistent": 0, "pass": true}
```

任何不一致都必须先修复编码，不能把不一致样本删除后继续。

## 五、阶段 3（E1）：单步 SMT 安全包络

### 步骤 3.1：构造边界状态单元

`make_boundary_cells` 在禁飞区四个面外侧分层采样状态单元。位置单元宽度主配置为 `0.25 m`，速度区间半宽为 `0.15 m/s`。每个状态单元通过仿射特征映射转为 6 维整数输入盒。

### 步骤 3.2：安全性质

根据所在面选外法向量 `n`，验证：

```text
n · u_q >= ceil((0.20 / amax) × qscale)
```

它表示边界附近控制动作至少具有小幅向外加速度。代码先用整数区间传播预筛：若动作下界已经满足性质则直接 SAFE；若动作上界仍违反性质则直接得到 UNSAFE candidate；只有区间未决单元才交给 Z3 精确寻找违反性质的整数输入：

- UNSAT → SAFE；
- SAT → UNSAFE candidate，并保存整数反例；
- timeout/unknown → UNKNOWN。

逐单元输出中的 `method=interval_prefilter` 或 `method=z3_exact` 记录实际判定后端，论文中应分别统计，不能把预筛结果全部写成 Z3 求解结果。

### 步骤 3.3：运行

```bash
python -m geofence_qnn.cli e1 \
  --config configs/main.yaml \
  --output runs/main_seed42_2x32
```

主配置使用 16 个进程、每单元 60 秒超时。若内存超过 80 GB，把 `workers` 从 16 调到 12 或 8；不要取消超时。

### 步骤 3.4：反例复核

对 SAT 单元，程序在对应连续状态盒中采样 128 个状态并执行真实 INT8 推理。`replay_found_violation=true` 表示至少找到一个真实连续状态违反单步动作性质；否则只称“整数输入盒候选反例”，不能称为真实穿越。

### 步骤 3.5：主表指标

- SAFE / candidate UNSAFE / UNKNOWN 数量与比例；
- `replayed_violations`；
- 求解时间 median、P90；论文正式版再从逐单元 CSV 计算 P99；
- 网络规模与认证覆盖率—CPU 小时曲线。

## 六、阶段 4（E2）：闭环区间可达性

### 步骤 4.1：初始集合

`initial_box` 的顺序是：

```text
[px_lo, px_hi, py_lo, py_hi, vx_lo, vx_hi, vy_lo, vy_hi]
```

主配置默认从禁飞区左侧向目标飞行。初始盒不得与扩张后的禁飞区相交。

### 步骤 4.2：逐周期传播

每一步执行：

1. 状态区间 → 特征区间；
2. 特征区间 → 整数输入区间；
3. INT8 网络区间传播得到动作上下界；
4. 双积分器传播位置与速度区间；
5. 检查可达盒与 `禁飞区 ⊕ 安全裕度` 的关系。

判断规则：

- 可达盒完全在禁飞区内：UNSAFE；
- 可达盒与禁飞区可能相交：UNKNOWN；
- 全时域不相交：SAFE。

UNKNOWN 单元按归一化后最宽的状态维度二分，直到 `max_refinement_depth`。

### 步骤 4.3：运行 1 s / 2 s / 4 s 时域

先把 `horizon_steps` 分别设为 20、40、80，每次使用独立输出目录：

```bash
python -m geofence_qnn.cli e2 --config configs/main_h20.yaml --output runs/e2_h20
python -m geofence_qnn.cli e2 --config configs/main_h40.yaml --output runs/e2_h40
python -m geofence_qnn.cli e2 --config configs/main_h80.yaml --output runs/e2_h80
```

建议先完成 20、40 步。若 80 步 UNKNOWN 仍高于 50%，不要无限细分；将 4 s 结果作为可扩展性边界。

### 步骤 4.4：报告

读取 `e2_summary.json`：

- `safe_volume_ratio`；
- `unsafe_volume_ratio`；
- `unknown_volume_ratio`；
- `max_depth`。

三者必须按初始盒体积加权，而不是简单数叶子节点。

## 七、阶段 5（E3）：统计仿真与运行时安全屏蔽

### 步骤 5.1：四组控制器

1. `teacher`：PID/CBF 风格教师；
2. `float`：FP32 MLP；
3. `int8`：INT8 QNN；
4. `int8_shield`：INT8 QNN + 短时预测屏蔽。

屏蔽器预测未来 `shield_horizon` 个周期；如预计进入扩张禁飞区，则用教师动作替换 QNN 动作。

### 步骤 5.2：运行

```bash
python -m geofence_qnn.cli mc \
  --config configs/main.yaml \
  --output runs/main_seed42_2x32
```

四种方法共享同一组初始状态、风扰种子和定位误差种子，因此是配对实验。

### 步骤 5.3：指标

- `violation_rate` 及 95% Wilson 区间；
- `goal_success_rate`；
- 平均和最小真实 clearance；
- 屏蔽介入轨迹比例；
- 每条轨迹平均介入次数。

零次穿越只写“10⁴ 条轨迹中未观测到穿越，95% Wilson 上界为……”，不能写成穿越概率为零。

## 八、阶段 6（E4）：风扰、定位误差与动态边界敏感性

复制 `main.yaml` 形成独立配置，依次修改一个因素：

| 因素 | 水平 |
|---|---|
| 风扰上界 | 0、0.25、0.5 m/s² |
| 定位误差 | 0、0.25、0.5、1.0 m |
| 安全裕度 | 0.5、1.0、2.0 m |
| 时域 | 20、40、80 周期 |
| 分区深度 | 2、3、4、5 |

每次只改一个因素并使用清晰目录名，例如：

```text
runs/sensitivity/wind_0.25_loc_0.50_margin_1.0_h40/
```

当前原型的动态边界通过在每个控制周期调用 `ForbiddenBox.moved(dx,dy)` 接入；正式动态围栏实验应增加“边界速度”和“地图更新延迟”配置，并保持分段线性运动，避免首阶段引入任意非线性几何。

## 九、五种子主实验与统计汇总

### 步骤 7.1：运行扫描

```bash
python scripts/run_seed_sweep.py \
  --base-config configs/main.yaml \
  --output-root runs/sweep_2x32 \
  --seeds 42 123 456 789 2024 \
  --networks 32x32 \
  --stage all
```

### 步骤 7.2：统计原则

- 逻辑结果 SAFE/UNSAFE 不做 p 值检验；
- 求解时间和仿真连续指标按种子报告均值、标准差和 bootstrap 置信区间；
- 同一初始状态集上的控制器比较使用配对置换检验或 Wilcoxon；
- 多重比较使用 Holm 校正；
- 同时报出 UNKNOWN，禁止将其合并进 SAFE。

## 十、消融实验

按以下顺序进行，每次只关闭一项：

1. 把精确整数 QNN 换成连续量化误差区间；比较漏报反例和求解速度；
2. 把自适应分区换成同数量均匀网格；比较 SAFE 体积/CPU 小时；
3. 关闭反例回放；统计候选反例与真实复现的差距；
4. 比较 FP32、INT8 和修复后 INT8；
5. 关闭屏蔽；比较安全收益与任务完成率；
6. 把定位误差从时变集合简化为常数偏置；检查结论变化。

## 十一、算力与停止条件

按现有工作站建议：

- E1 默认 16 个进程，观察内存后最多 20；
- 每个 Z3 单元 60 秒，困难实例可另行 300 秒复跑；
- 单个主配置不超过 24 CPU 小时；
- 整机峰值内存不超过 80 GB；
- 2×16 在 S1/S2 上至少 80% 单元应在 60 秒内判定；
- 3×32 超时率超过 50% 时停止扩大；
- E2 连续两轮细化后 UNKNOWN 仍高于 50% 时停止细化。

## 十二、与已有 QNNVerifier / Repair 代码对接

最值得替换的是 `smt.py` 与 `quantization.py`：

1. 保留 `Int8MLP.save()` 的权重/偏置/qscale 格式；
2. 用已有 QNNVerifier 编码替换 `encode_int8_network`；
3. 保持 `verify_action_halfspace` 返回 `status / elapsed_s / counterexample` 字段；
4. 用现有 repair 模块读取 `e1_cells.csv` 中可复现反例；
5. 修复后重新运行 E0、E1、E2，检查原 SAFE 单元是否退化。

这样可以复用既有研究积累，而不必重写训练、场景、统计和产物管理。

## 十三、论文结果表建议

主表 1：网络、量化、种子、E1 SAFE/UNSAFE/UNKNOWN、回放反例、median/P90/P99、CPU 小时。

主表 2：时域、风扰、定位误差、安全裕度、E2 三类体积比例、最大细化深度。

主表 3：四种控制器的穿越率及 Wilson 区间、目标成功率、最小距离、屏蔽介入率。

主图：

1. 认证 SAFE 体积—累计 CPU 小时；
2. 定位误差—最小安全裕度/UNKNOWN 率；
3. 屏蔽安全收益—任务性能 Pareto 曲线；
4. 网络规模—求解时间分布。

## 十四、开源飞控（PX4 Autopilot / ArduPilot）实验路径

### 步骤 14.1：安装可选依赖

```bash
python -m pip install -e '.[flightstack]'   # pyulog + pymavlink
```

核心验证流水线（E0—E2）不依赖这些包；只有日志解析和 SITL 录制需要。

### 步骤 14.2：没有自己的飞行日志时的四个选项

按证据强度从高到低：

1. **公开真机日志（最短路径，两条命令）**：

```bash
python scripts/fetch_px4_logs.py --count 5 --dest logs/px4
python -m geofence_qnn.cli all --config configs/px4_public_logs.yaml --output runs/px4_public
```

   脚本从 [logs.px4.io](https://logs.px4.io) 公开数据库按机型/时长/飞行模式过滤并下载真实四旋翼飞行，逐个解析验证；`data.auto_align: true` 自动把每条航迹平移到围栏几何上，无需手调 `data.offset`。`data.logs` 也可直接填 Flight Review 下载 URL，自动下载并缓存到 `logs/_downloads/`；
2. **SITL 录制（固件级证据）**：SITL 跑的是与真机相同的固件代码，按步骤 14.5 录制后走 `data.source: csv`；SITL 自身产生的 `.ulg`/`.bin` 也可直接用步骤 14.3 的加载器读取；
3. **演示 CSV 日志（零外部依赖，验证链路用）**：

```bash
python scripts/make_demo_logs.py --config configs/demo_csv.yaml --backend px4 --output logs/demo
python -m geofence_qnn.cli all --config configs/demo_csv.yaml --output runs/demo_csv
```

   它用 PX4/ArduPilot 行为教师闭环 rollout 生成 CSV 轨迹，走与真实日志完全相同的摄取代码；论文中只能标注为"行为级模型 rollout"；
4. **合成数据 + 飞控行为教师**：`data.source: synthetic` 配 `teacher.backend: px4/ardupilot`（`configs/smoke_flightstack.yaml`），完全不经过日志链路。

建议顺序：直接从 1 开始（真实数据、两条命令）；离线环境先用 3 或 4 打通流程；2 作为固件级证据补充。`training_summary.json` 的 `data_source`/`data_logs`/`teacher_backend` 字段记录每次运行的实际数据来源，论文中据此区分证据等级。

### 步骤 14.3：用自己的飞行日志训练

1. 把 PX4 `.ulg`（真机或 SITL 均可）放入 `logs/px4/`，或把 ArduPilot DataFlash `.bin`/MAVLink `.tlog` 放入 `logs/ardupilot/`（`data.logs` 也接受 http(s) URL，自动下载缓存）；
2. 以 `configs/px4_ulog.yaml` 或 `configs/ardupilot_log.yaml` 为模板，检查：
   - 对齐：优先 `data.auto_align: true`（把每条航迹的包围盒中心平移到围栏中心，只平移位置不动速度/加速度）；需要精确控制时用 `data.offset` 手动平移；
   - `data.synthetic_fraction`：真实航迹不会绕虚拟围栏飞行，按比例混入教师样本提供避障行为与边界带覆盖；
3. 正常运行 `train/e0/e1/e2/mc`。`training_summary.json` 中的 `data_source`、`data_logs` 和 `teacher_backend` 字段记录数据来源，论文中必须报告。

验收标准与合成数据相同（E0 100% 一致等），另加：日志转换后落在边界带内的样本数不得为零，否则 E1 结论对训练分布不具代表性。

### 步骤 14.4：PX4 / ArduPilot 围栏行为基线

`simulation.controllers` 加入 `px4` 和/或 `ardupilot` 后，蒙特卡洛主表会多出对应的行为基线行（与其他控制器共享同一组初始状态与扰动种子，属于配对实验）：

```bash
python -m geofence_qnn.cli mc --config configs/smoke_flightstack.yaml --output runs/smoke_fs
```

论文中这两行必须标注为"文档所述围栏逻辑的行为级模型"，不得写成固件本体结果。

### 步骤 14.5：SITL 固件级数据采集

1. 启动 SITL：PX4 用 `make px4_sitl jmavsim`，ArduPilot 用 `sim_vehicle.py -v ArduCopter --console --map`；
2. 上传围栏并规划任务（QGroundControl / MAVProxy）；
3. 录制：

```bash
python -m geofence_qnn.cli sitl-record --config configs/main.yaml --output runs/sitl \
  --url udp:127.0.0.1:14550 --episodes 10 --duration 120 --rate 20
```

4. 把输出 `runs/sitl/sitl_trajectories.csv` 填入 `data.logs` 并设 `data.source: csv`，回到步骤 14.3。

录制器只读取 `LOCAL_POSITION_NED` 并转换到世界系，不解锁、不切模式；`--command-goal` 仅在 OFFBOARD/GUIDED 下有效。
