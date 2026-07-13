# 本代码包的验证记录

生成 ZIP 前已完成以下检查：

- Python 源码、脚本与测试通过 `compileall`；
- `unittest` 共 7 项全部通过；
- 实际运行 `configs/smoke.yaml` 的完整 `all` 流水线；
- E0：50/50 个输入的 NumPy INT8 与 Z3 编码一致；
- E1：16 个边界单元均产生 SAFE/UNSAFE/UNKNOWN 记录，候选反例可回放；
- E2：成功产生体积加权的闭环可达性汇总；
- 蒙特卡洛：教师、FP32、INT8、INT8+屏蔽四组均产生 Wilson 区间、最小距离和屏蔽延迟；
- 16 单元 smoke 流水线墙钟时间约 30 秒（具体时间依赖 CPU 与 Z3 版本）。

验证环境：Python 3.12、NumPy 2.3.5、z3-solver 4.15.3。`smoke.yaml` 只验证代码链路，不用于论文性能结论。

