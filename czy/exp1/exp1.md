# Holosoma 训练实验笔记

按 [lab-notebook](../skills/lab-notebook/SKILL.md) 维护索引和七节实验结构。本文路径沿用统一笔记约定；首次流程验证按用户要求编号为 `exp0`，下一项新目标默认编号 `exp_1`。时间除附件 JSON 的 UTC 查询时间外，均为北京时间（UTC+8）。

## 实验索引

| 编号 | 日期 | 摘要 | 状态 | Task ID | Flux账号 | checkpoint |
| --- | --- | --- | --- | --- | --- | --- |
| [exp0](#实验-exp0flux-isaaclab-上的-g1-ppo-流程验证) | 2026-10-28 | 修复安装源与版本回溯，完成 128 环境 × 2 次 PPO 迭代；checkpoint 上传/下载及云端 ONNX 校验通过 | 已测试 | `TASK_20261028_224`（含两次失败尝试，见 §7） | `limxmts0elyormeym6@uberip.com` | [model_00001.pt](../data/exp0/model_00001.pt) |
| [exp_1](#实验-exp_1g1-平地-ppo-正式训练) | 2026-10-28 | 安装阶段下载 Open3D 时超过 900 秒，未进入 GPU 训练；2026-10-29 完成终止诊断 | 失败 | `TASK_20261028_251` | `limxmts0elyormeym6@uberip.com` | 无；未开始训练 |
| [exp_1.1](#实验-exp_11精简依赖并重试正式训练) | 2026-10-29 | 完成 5000 轮；末段 episode 均值 887.93 steps；本机 MuJoCo 录像完成，云端录像清理失败 | 已测试 | `TASK_20261029_112`；回放 `TASK_20261029_249` | `limxmts0elyormeym6@uberip.com` | [model_04999.pt](../data/exp_1.1/model_04999.pt) |
| [exp_2](#实验-exp_2g1-带箱体的搬运跟踪训练) | 2026-10-29 | 安装阶段发现入口切换目录后的输出路径问题，主动停止，未开始 PPO | 失败 | `TASK_20261029_261` | `limxmts0elyormeym6@uberip.com` | 无 |
| [exp_2.1](#实验-exp_21修正搬箱子入口的工作目录) | 2026-10-29 | 完成 8000 轮；原生评估完整片段通过 2/128（1.56%），主要为箱体/手脚跟踪失败 | 已测试 | 训练 `TASK_20261029_267`；评估 `TASK_20261030_051` | 训练 `limxmts0elyormeym6@uberip.com`；评估用户 4391，见 §7 | [model_07999.pt](../data/exp_2.1/model_07999.pt) |
| [exp_2.2](#实验-exp_22原生规模下恢复完整状态续训) | 2026-10-30 | 完成 4096 环境 × 新增 6000 轮；本地策略回放已录制，原生线上回放待验收 | 已测试 | `TASK_20261030_073`（061、065 已停止，见 §7） | `limxmts0f9p800yu7g@uberip.com` / 4391 | 源 `model_07999.pt`；目标 `model_13999.pt` |

## 实验 exp0：Flux IsaacLab 上的 G1 PPO 流程验证

### 1. 上一实验结果与教训

本轮为当前 Holosoma 项目的首次云端训练流程验证，无上一轮策略训练或行走评估数据。先前的镜像查询、版本匹配和 CLI dry-run 只能证明存在候选配置。

本轮调试过程中保留了两次失败尝试：第一次运行 24 秒，安装源 DNS 失败；第二次运行 643 秒，依赖安装触发 600 秒超时；两次均未开始 GPU 训练、没有 checkpoint。具体任务、提交和修复见 §7。

**核心教训**：预装 IsaacLab 的镜像还需要补齐 Holosoma 依赖；包索引可达也不代表下载和版本解析可以在限时内完成。应先修复网络源与依赖约束，再启动下一次小规模验证。

### 2. 本轮修改目标

- 在用户指定账号下复用 Flux 官方 IsaacLab 2.3.2 / IsaacSim 5.1 镜像。
- 使用 G1 29 自由度、PPO、平地、128 个环境，从零完成 2 次迭代。
- 验收 CUDA 仿真、可加载 checkpoint、ONNX 导出/推理、TensorBoard 和平台上传，记录可复现的安装与启动方法。
- 本轮验收目标为流程可运行；不设行走效果或收敛阈值。

### 3. 修改内容

### 修改一：增加有时限的云端包装入口

| 项目 | 原状态 | 本轮方案 | 理由 |
| --- | --- | --- | --- |
| 平台入口 | 原训练脚本依赖预先安装的 Holosoma 包 | `gm-run holosama/scripts/train_flux_smoke.py` | 满足平台 `gm-run` 限制，在训练导入前补齐依赖 |
| Python 环境 | 可能误用宿主/其他环境 | 统一使用 `gm-run` 的 `sys.executable` | 复用镜像仿真环境及 CUDA PyTorch |
| 训练规模 | 原预设用于正式训练 | 128 环境、2 次迭代、每轮保存 | 控制验证规模，并获得最终 checkpoint |
| 进程生命周期 | 安装、训练、校验可能混在同一解释器 | 安装后分别启动新子进程做 CUDA、训练和产物校验 | 避免包装进程已导入旧依赖造成干扰 |
| 时间限制 | 无专门 smoke 限制 | 安装 900 秒、训练 600 秒、校验 120 秒，成功后等待上传 60 秒 | 各阶段有明确失败边界 |
| 产物验收 | 仅检查文件存在不足以证明可用 | 加载 checkpoint；ONNX CPU 推理输出 `(1,29)` 且有限；检查 TensorBoard 事件 | 校验实际可读/可执行性 |

### 修改二：修复包索引与依赖版本回溯

| 项目 | 失败时状态 | 成功方案 | 理由 |
| --- | --- | --- | --- |
| pip 索引 | 镜像默认 `mirrors.ivolces.com` 无法解析；PyPI 文件下载有超时 | 清华源 → 阿里云源 → PyPI 探测，显式传入 pip，屏蔽旧配置 | 云端不依赖本机内网代理；本次选中清华源 |
| IsaacSim 约束 | 需要保持其运行时合同 | `flux_smoke_constraints.txt` 引用 `isaacsim_constraints.txt` | 保留 `packaging==23.0`、torch 2.7 等约束 |
| Jupyter 系列 | 最新版本要求更高 packaging，pip 大量回溯 | Notebook 7.3.3、JupyterLab 4.3.8、Server 2.15.0、Lab Server 2.27.3 | 直接选择兼容组合 |
| OpenCV / TensorDict / TensorBoard | 无上界选择触发更多候选版本 | 4.11.0.86 / 0.9.0 / 2.20.0 | 保持 NumPy 1.x 及镜像 TensorDict 兼容性 |

依赖方案在本地通过 Linux / Python 3.11 解析后再推送测试。原奖励、机器人控制参数和 PPO 学习率未为此次流程验证调优。

### 修改三：固定任务账号并验证提交来源

每次操作在子进程环境通过 `FLUX_API_KEY` 选择用户指定账号；`whoami` 核对邮箱/用户 ID，创建后再核对任务 `userId=4390`、项目和状态。避免共享 Keychain 被其他会话切换后操作到别的账号。密钥没有进入实验记录或 Git。

Git 在本机使用用户代理 `http://10.12.202.95:1080`，训练启动前推送并核对远端 `main`。Flux 克隆目录为 `holosama`，包名为 `holosoma`。代理协议不通时只重试只读查询，不能据此重新创建训练任务。

### 4. 修改文件

- [scripts/train_flux_smoke.py](../../scripts/train_flux_smoke.py)：云端依赖安装、CUDA 探针、小规模训练和产物校验。
- [scripts/flux_smoke_constraints.txt](../../scripts/flux_smoke_constraints.txt)：本次成功使用的 Flux 安装约束。
- [scripts/isaacsim_constraints.txt](../../scripts/isaacsim_constraints.txt)：被引用的现有 IsaacSim 运行时约束。
- [tests/test_flux_smoke.py](../../tests/test_flux_smoke.py)：5 项 CPU 测试，覆盖规模边界、索引回退和产物验收；本次修复时已通过 Ruff 检查。
- [flux-cli](../skills/flux-cli/SKILL.md) 及其引用文档：镜像、启动、账号、安装失败定位与实验记录。
- 本记录与 [lab-notebook](../skills/lab-notebook/SKILL.md)：按 Holosoma 实际参数和产物归档；文档更新不改变已完成任务的训练 commit。

### 5. 训练参数

实际配置从下载 checkpoint 的 `experiment_config` 提取，见 [training_config.json](artifacts/exp0/training_config.json)。

| 参数 | 实际值 |
| --- | --- |
| 训练方式 | 从零；`training.checkpoint=null` |
| Flux账号 / 用户 ID | `limxmts0elyormeym6@uberip.com` / `4390` |
| 项目 / 任务 | `PRO_20261028_008` / `TASK_20261028_224` |
| 仿真 / 机器人 / 算法 | IsaacSim / `g1_29dof` / PPO |
| Tyro 组合 | `exp:g1-29dof simulator:isaacsim logger:disabled terrain:terrain-locomotion-plane` |
| `training.num_envs` / `training.seed` | 128 / 1 |
| `training.headless` / 视频 / Wandb | true / 关闭 / 关闭；TensorBoard 仍启用 |
| `algo.config.num_learning_iterations` / `save_interval` | 2 / 1（零基迭代编号为 0、1） |
| `num_steps_per_env` / 总采样数 | 24 / `128 × 24 × 2 = 6,144` |
| actor / critic 学习率 | 配置初值均为 `1e-5`，最终 optimizer 值均为 `1e-5`；日志显示 `0.0000` 是格式精度限制 |
| optimizer / schedule | AdamW，weight decay=0.001 / adaptive，desired KL=0.01 |
| PPO epoch / minibatch | 8 / 4 |
| gamma / lambda / clip | 0.99 / 0.95 / 0.2 |
| 网络 / 观测 / 动作 | MLP `[512,256,128]`，ELU；actor 100 维、critic 103 维；29 维动作 |
| 仿真频率 / 控制降采样 | 200 Hz / 4；`control_dt=0.02 s`，50 Hz 控制 |
| 算力 | `ESKU000001`，后台 `SKUSL000002`，1 GPU；目录标记 4090D 24G，CUDA 实报 NVIDIA GeForce RTX 4090，25,393,692,672 bytes |
| 镜像 / 版本 | `BJX00000178` / `V000420`，`isaac-lab2.3.2-v2` |
| Python / torch | `gm-run` Python 路径 `/workspace/isaaclab/_isaac_sim/kit/python/bin/python3`；torch `2.7.0+cu128` |
| 关键 Python 包 | NumPy 1.26.4、Pydantic 2.11.7、Tyro 1.0.16、TensorDict 0.9.0、warp-lang 1.16.0 |
| 代码仓库 / 分支 | [Lee-Weather/holosama](https://github.com/Lee-Weather/holosama) / `main` |
| 实际训练 commit | `b438b4e23a7a3a1d52136a2b0b8a687b0ba9d154` |
| 平台启动命令 | `gm-run holosama/scripts/train_flux_smoke.py` |
| 原始实验目录 | `logs/flux-smoke-20261028_095658/20261028_095921-g1_ppo_smoke-locomotion/`（UTC 命名） |

复现时先核查镜像、资源、账号与 Git 提交，再使用包装入口默认参数。它会设置 EULA 环境变量、探测安装源，并用上述 Tyro 组合及训练参数启动原训练脚本。`hparamsPath` 留空；本项目没有可供平台直接修改的统一训练 YAML。

### 6. 预期与验收

| 指标 | 上一轮 | 本轮目标 | 异常信号 |
| --- | --- | --- | --- |
| CUDA / 仿真 | 无历史策略训练 | 可识别 GPU，G1 环境完成采样 | 无 CUDA、扩展/资产启动失败 |
| 迭代 / 步数 | 两次安装失败均为 0 | 2 次 / 6,144 | 提前退出或训练超时 |
| 最终 checkpoint | 无 | `iter>=1`、actor 权重有限、成功下载 | 仅有中间文件、加载失败或非有限权重 |
| ONNX | 无 | 导出且 CPU 输出 `(1,29)`，所有值有限 | 缺失、维度错误或 NaN/Inf |
| 日志 / 平台 | 未完成端到端验收 | TensorBoard 事件、可读曲线、最终状态 5 | 无事件、无产物、异常终态 |
| Mean reward / episode length | 无历史可比基线 | 记录实测，不设收敛门槛 | 不用两轮结果判断行走能力 |

### 7. 实验结果

#### 三次任务尝试及失败根因

| 尝试 | Task ID | commit | 开始–结束（北京时间） | 运行秒数 | 结果与处理 |
| --- | --- | --- | --- | --- | --- |
| A | `TASK_20261028_067` | `d4480c6` | 14:56:10–14:56:34 | 24 | 状态 6；默认 pip 源 DNS 失败，build backend 无法安装；无训练/模型；增加可达源探测 |
| B | `TASK_20261028_113` | `4601422` | 17:20:14–17:30:57 | 643 | 状态 6；PyPI 下载有超时，Jupyter/packaging 回溯导致 pip 安装超过 600 秒；无训练/模型；增加版本约束、国内源优先 |
| C | `TASK_20261028_224` | `b438b4e` | 17:56:26–18:00:44 | 258 | 状态 5（Completed）；依赖安装、CUDA、2 次 PPO 迭代、ONNX 校验与 `.pt` 上传全部完成 |

三次均属指定账号 `4390`。时长来自各任务 API，合计 925 秒；不是纯训练耗时，也不含任务之间的本地排查间隔。证据：[A](artifacts/exp0/failed_index.json)、[B](artifacts/exp0/failed_resolver.json)、[B 超时日志摘录](artifacts/exp0/attempt_b_excerpt.txt)、[C](artifacts/exp0/task.json)、[账号](artifacts/exp0/identity.json)。未启动的其他账号草稿不属于这三次训练。

#### 最终结果（iter 1，总计 2 次迭代）

| 指标 | 上一轮 | 目标 | 实测 | 判定 |
| --- | --- | --- | --- | --- |
| 训练迭代 / 采样数 | 安装失败，无训练 | 2 / 6,144 | 2 / 6,144 | ✅ |
| Mean reward | 无可比训练 | 记录，不设门槛 | -3.955620765686035（日志 -3.96） | 观测值 |
| Mean episode length | 无可比训练 | 记录，不设门槛 | 41.181819915771484 steps（日志 41.18） | 观测值 |
| 时间换算 | 无 | 按实际 control_dt | `41.1818199 × 0.02 ≈ 0.824 s`；非独立站立测试 | 仅换算 |
| 训练速度 / 最后一轮耗时 | 无 | 正常推进 | 3,570 steps/s / 约 0.86 秒；两轮累计约 1.86 秒 | ✅ |
| checkpoint | 无 | 最终可读、有限权重 | `model_00001.pt`，5,300,224 bytes，iter=1，9 个 actor 张量全部有限 | ✅ |
| ONNX | 无 | CPU 输出 29 维且有限 | 958,244 bytes；全零输入，输出 `(1,29)` 且有限 | ✅ 云端验收 |
| TensorBoard / 平台指标 | 无 | 事件存在、曲线可读 | 1 个事件文件；补查 reward、episode length、FPS 均有 steps `[0,1]` | ✅ |
| 终态 / 全程耗时 | 状态 6 | 正常完成 | 状态 5，258 秒（含安装、仿真启动及 60 秒上传等待） | ✅ |

#### 训练趋势

下表来自平台 `precise` 实测，展示六位小数；原始精度见 [reward](artifacts/exp0/reward.json)、[episode length](artifacts/exp0/episode_length.json)、[FPS](artifacts/exp0/fps.json)。

| iter | Mean reward | Mean episode length | steps/s |
| --- | --- | --- | --- |
| 0 | -0.792372 | 12.250000 | 3,084 |
| 1 | -3.955621 | 41.181820 | 3,570 |

刚完成时的 `accelerate` 奖励查询只返回 step 0；本次归档时 `precise` 已返回两点。两次查询时间和采样模式不同，不能单凭此差异断定具体原因。收尾应复查，并记录时间、模式与实际 steps。两点不足以分析收敛趋势。

#### 各奖励项最终值

权重来自 checkpoint 内实际配置，数值取云日志 iter 1 的 `rew_*`（不是 `raw_rew_*`）；带惩罚课程的项还受动态缩放影响，日志 `Env/penalty_scale=0.0993`。这些日志项不直接等同总 episode reward。

| 奖励项 | 配置权重 | 最终日志值 | 说明 |
| --- | --- | --- | --- |
| tracking_lin_vel | 2.0 | 0.0136 | 线速度跟踪 |
| tracking_ang_vel | 1.5 | 0.0150 | 角速度跟踪 |
| penalty_ang_vel_xy | -1.0 | -0.0298 | 惩罚课程 |
| penalty_orientation | -10.0 | -0.0113 | 惩罚课程 |
| penalty_action_rate | -2.0 | -0.3166 | 惩罚课程 |
| feet_phase | 5.0 | 0.1346 | 步态相位 |
| pose | -0.5 | -0.0306 | 惩罚课程 |
| penalty_close_feet_xy | -10.0 | -0.0104 | 惩罚课程 |
| penalty_feet_ori | -5.0 | -0.0138 | 惩罚课程 |
| alive | 1.0 | 0.0444 | 存活项 |

#### 产物与证据归档

- 最终模型：[czy/data/exp0/model_00001.pt](../data/exp0/model_00001.pt)。SHA-256：`8d60055838bec656e9bf5bf7e691b6642d28b7b36f45df0c4d5f4f83d4baa1c2`。
- [模型本地校验](artifacts/exp0/checkpoint_validation.json)、[云端 smoke 结果](artifacts/exp0/smoke_result.json)、[实际配置](artifacts/exp0/training_config.json)。
- [完整脱敏云日志](artifacts/exp0/cloud.txt)（去除签名 URL、ANSI 颜色及行末空白）、[日志指标摘录](artifacts/exp0/log_metrics.json)、[平台模型列表](artifacts/exp0/models.json)、[指标 keys](artifacts/exp0/metric_keys.json)。
- ONNX 已在云端导出并通过 CPU 推理；SDK 发现文件，但平台模型列表只有两个 `.pt`，未提供 ONNX 下载项，因此本地未归档 ONNX。
- 本次关闭视频、未执行回放/CSV 诊断，也未下载 TensorBoard 原始事件文件；不创建空的占位产物。

**结论：✅ exp0 流程验收达标。** Flux 的该 IsaacLab 镜像经包装入口和依赖适配，可以完成 Holosoma G1 PPO 小规模训练、最终 checkpoint 上传/下载及云端 ONNX 推理。长时间训练、策略行走效果、视频与部署链路仍需独立验收。

**跑通经验与保留限制**：

1. 账号必须在操作时核对，不能只依赖共享登录状态；任务创建后再核对所属用户、项目和状态。
2. 复用镜像解释器，使用 `python -m pip`；本镜像没有独立 `pip` 命令。不要默认重装 IsaacSim 或新建隔离于 SDK 的 Conda 环境。
3. 国内索引优先加明确依赖约束解决了本次冷启动。遇到同一故障先修改并验证，再复制新任务；保留失败任务，不删除历史。
4. 新任务创建不等于开训，状态 0 后仍需 `task run`；平台入口始终使用 `gm-run`，不是 `flux-run` 或 shell 命令串。
5. 发布名 IsaacLab 2.3.2 与发行包元数据 `isaaclab=0.54.2` 不是同一版本字段；`isaacsim` 也不一定以 Python distribution 形式可见。
6. 安装仍提示 `nvidia-srl-usd` / usd-core 元数据、docstring-parser 以及 FastAPI/Starlette 依赖声明冲突。实际路径通过不等于全局 `pip check` 通过；不要为消除元数据告警直接向 IsaacSim 安装另一套 USD。
7. CUDA 名称与商品名分别记录；日志学习率的四位小数显示会把 `1e-5` 打印成 `0.0000`，应核对 checkpoint。
8. `.pt` 已验证自动上传；ONNX、完整 TensorBoard 点集和视频按实际 API/产物另验，不根据文件扫描日志直接承诺下载能力。

**下一轮方向**：

- 新目标记录为 `exp_1`：在另行确定的训练规模下评估 G1 平地 PPO 的长期奖励、episode 长度和行走表现，保留 exp0 作为环境基线。
- 准备可保存并下载 ONNX 的产物通道，随后独立验证回放/视频及推理部署；本次归档不启动新计费任务。

## 实验 exp_1：G1 平地 PPO 正式训练

### 1. 上一实验结果与教训

exp0 已在同账号的 `TASK_20261028_224` 正常完成，128 环境、2 次迭代、6,144 步；最后 Mean reward=-3.955621、Mean episode length=41.181820 steps（约 0.824 秒），训练速度约 3,570 steps/s。checkpoint 下载后权重有限，云端 ONNX 推理通过。全程 258 秒主要包含冷启动，不能用两轮短测的速度准确估算大规模训练。

核心教训：依赖安装必须使用已验证的索引/约束；账号逐次核对；日志学习率 `0.0000` 实为格式化后的 `1e-5`。128 环境测试未证明大规模显存容量或收敛，因此本轮先选择 2048 环境，保留 6 小时训练时限和每 100 轮 checkpoint。

### 2. 本轮修改目标

- 用户已授权推送实验产物并开启一次正式训练。本轮从零训练 G1 平地 PPO，不从 exp0 两轮模型续训。
- 完成 5000 次迭代，共计划采样 `2048 × 24 × 5000 = 245,760,000` 步。
- 目标为获得可继续评估的平地策略，记录奖励、episode 长度、速度及 checkpoint；训练曲线不能替代行走回放验收。

### 3. 修改内容

### 修改一：提高规模并建立正式入口

| 参数 | exp0 | exp_1 | 理由 |
| --- | --- | --- | --- |
| 包装入口 | `scripts/train_flux_smoke.py` | `scripts/train_flux.py` | 正式入口独立配置，保留 smoke 规模边界 |
| 环境数 | 128 | 2048 | 在 24GB GPU 上开始规模训练；容量待实际启动验证 |
| PPO 迭代数 | 2 | 5000 | 形成训练曲线与策略优化；首次正式运行不直接使用预设 25000 轮 |
| 保存间隔 | 1 | 100（并保存最终模型） | 保留阶段性恢复点及最终 checkpoint |
| 训练超时 | 600 秒 | 21600 秒（6 小时） | 明确本次运行边界；超时退出，不自动重试/续费续训 |

### 修改二：复用已验证的依赖与验收

提取 exp0 的环境准备、依赖安装和 CUDA 检查函数供两个入口共用；不修改 PPO、奖励、机器人和随机化实现。正式入口记录云端实际 Git SHA 与启动参数，完成后在新解释器加载最终 checkpoint、执行 ONNX CPU 推理并检查 TensorBoard。

### 4. 修改文件

- [scripts/train_flux.py](../../scripts/train_flux.py)：正式训练入口及规模/运行时限。
- [scripts/train_flux_smoke.py](../../scripts/train_flux_smoke.py)：将既有 bootstrap 提取为共享函数；保留原 smoke 参数和验收。
- [tests/test_flux_training.py](../../tests/test_flux_training.py)：验证无效规模不会启动、安装失败不会进入 GPU、训练超时不会重试。
- 本笔记和 Flux 参考文档：记录开训计划、实际任务 ID 与后续进度。

### 5. 训练参数

| 参数 | 计划值 |
| --- | --- |
| 训练方式 / 实验编号 | 从零 / `exp_1` |
| Flux账号 / 用户 ID / 项目 | `limxmts0elyormeym6@uberip.com` / `4390` / `PRO_20261028_008` |
| 组合 | `exp:g1-29dof simulator:isaacsim logger:disabled terrain:terrain-locomotion-plane` |
| 环境数 / seed | 2048 / 1 |
| 迭代数 / 每环境采样步数 / 保存间隔 | 5000 / 24 / 100 |
| actor / critic 学习率 | 初始均 `1e-5`，adaptive，desired KL=0.01；沿用预设 |
| optimizer / 网络 / PPO 参数 | 与 exp0 相同：AdamW；MLP `[512,256,128]`；epoch=8，minibatch=4，gamma=0.99，lambda=0.95 |
| 物理 / 控制频率 | 200 Hz / 50 Hz，control_dt=0.02 s |
| GPU / 镜像 | `ESKU000001`（1×4090D 24G）/ `BJX00000178`，`V000420` |
| 仓库 / 分支 | [Lee-Weather/holosama](https://github.com/Lee-Weather/holosama) / `main`；启动前推送并核对，云端计划记录实际 SHA |
| 启动命令 | `gm-run holosama/scripts/train_flux.py --experiment exp_1 --num-envs 2048 --iterations 5000 --save-interval 100 --training-timeout 21600` |
| 安装 / 训练 / 上传等待 | 900 秒 / 21600 秒 / 成功后 60 秒 |
| 记录方式 | TensorBoard；关闭 Wandb、视频；本轮不包含回放任务 |

### 6. 预期与验收

| 指标 | exp0 | 本轮目标 | 异常信号 |
| --- | --- | --- | --- |
| 迭代 / 最终 checkpoint | 2 / `model_00001.pt` | 完成 5000；`model_04999.pt` 的 iter>=4999 | 提前结束、超时或权重非有限 |
| GPU / 运行状态 | 128 环境可用 | 2048 环境正常采样，无 OOM/持续异常 | 显存不足、NaN/Inf、仿真崩溃 |
| Mean episode length | 41.181820 steps | 目标末段 100 轮均值>=800 steps（16 秒）；待测 | 长期停留在短 episode，需分析终止原因 |
| Mean reward | -3.955621 | 目标末段 100 轮均值高于 exp0；仅作方向参考 | 持续退化或非有限 |
| 产物 | `.pt` 上传、ONNX 云端校验通过 | checkpoint 上传且可加载，最终 ONNX 校验通过 | 上传缺失、模型不可读 |

以上是本轮预期，不是已验证效果。到达迭代上限与达到策略质量目标分别判定；行走表现留待后续回放。

### 7. 实验结果

#### 最终状态与诊断（2026-10-29 核查）

**结论：❌ exp_1 本次未达开训验收目标，失败于依赖安装阶段。**

| 项目 | 实测结果 |
| --- | --- |
| 账号 / 用户 ID | `limxmts0elyormeym6@uberip.com` / `4390`，本次直连 `whoami` 已复核 |
| 最终状态 | `6`（Terminated） |
| 开始 / 结束 | 2026-10-28 18:32:09 / 18:47:24（北京时间） |
| 任务全程 | 915 秒（15 分 15 秒） |
| 异常 | `subprocess.TimeoutExpired`，pip 安装进程超过约 900 秒 |
| 最后下载项 | 清华源的 Open3D wheel，400.8 MB；无下载完成记录 |
| 相邻包下载速度 | Notebook 257.6 kB/s；ONNX Runtime 250.4 kB/s |
| CUDA / PPO | 未出现 `FLUX_SMOKE_GPU`、`FLUX_TRAIN_START` 或 `Learning iteration` |
| 完成迭代 / checkpoint | 0 / 平台模型列表 `total=0` |
| 训练曲线 | 平台指标 keys 为空，Mean reward / episode length 均无本次数据 |

**根因分析**：

- `scripts/train_flux.py` 在调用 `bootstrap.install_training_package()` 时退出；该函数对 `pip install -c scripts/flux_smoke_constraints.txt -e ./src/holosoma` 设置 900 秒上限。异常和平台结束时间吻合。
- 最终日志停在 `Downloading ... (400.8 MB)` 的 Open3D 下载。按相邻 ONNX Runtime 的 250.4 kB/s 粗估，Open3D 单包约需 1601 秒，已超过整个安装阶段 900 秒预算；这是估算，并非对 Open3D 下载速度的直接测量。
- 包源选择仅验证索引 HTTP 可达，没有验证大文件吞吐。exp0 同配置的安装成功不保证另一节点或时段具备相同速度。
- Open3D、Notebook、mypy 当前都是训练包的强制依赖；静态检索 `src/holosoma/holosoma/` 和 `scripts/` 未发现对三者的 import，造成此次 G1 无界面训练仍需下载大型工具包。下一次需核实依赖用途并拆分可选工具依赖，不能用 `--no-deps` 跳过依赖检查。
- 此次未进入 GPU 仿真，因此无法据此判断 2048 环境的显存容量、训练速度或收敛。6 小时训练超时尚未开始计时。

**证据归档**：[完整脱敏终止日志](artifacts/exp_1/terminated_cloud.txt)、[终态 API](artifacts/exp_1/task_final.json)、[空模型清单](artifacts/exp_1/models_final.json)、[空指标列表](artifacts/exp_1/metric_keys_final.json)、[诊断摘要](artifacts/exp_1/termination_diagnosis.json)。保留 [本地启动计划](artifacts/exp_1/launch_plan.json)、[云端实际计划](artifacts/exp_1/cloud_launch_plan.json) 和 [初始启动摘录](artifacts/exp_1/startup_excerpt.txt)。

**下一轮方向**：先拆分非训练必需工具依赖，增加下载吞吐/备用源验证，或使用已补齐依赖的匹配镜像；本次只完成终止诊断与归档，不创建新的计费任务。保持原任务失败记录，不能将状态 3 的历史启动记录视为训练已经实际推进。

#### 原始开训记录（保留历史）

开训记录：`TASK_20261028_251`，2026-10-28 18:32:09 开始，所属用户 `4390`，平台状态 3。云端 `FLUX_TRAIN_PLAN` 已确认实际 Git SHA 为 `853ef2a29e164281d5f4422924e0fae11241721e`，2048 环境、5000 次迭代、保存间隔 100、训练超时 21600 秒。计划及身份/资源见 [launch_plan.json](artifacts/exp_1/launch_plan.json)。

18:32:21 启动 bootstrap，选中清华源，开始安装。原始运行项目名为 `flux-exp_1-20261028_103221`。6 小时时限从训练子进程启动时计算，安装另有 900 秒上限。正式入口和 bootstrap 的 8 项 CPU 测试及 Ruff 检查通过；这些本地测试不验证云端下载带宽或真实训练容量。

## 实验 exp_1.1：精简依赖并重试正式训练

### 1. 上一实验结果与教训

exp_1 的 `TASK_20261028_251` 运行 915 秒，在 pip 安装阶段触发 900 秒超时。最后下载项 Open3D 400.8 MB，相邻 ONNX Runtime 下载为 250.4 kB/s；完成训练迭代 0，模型数 0，指标 keys 为空。索引可达不能替代文件下载速度验证，无界面训练也不应默认下载大型交互分析工具。

### 2. 本轮修改目标

- 用户已同意公开推送 exp_1 诊断，修复后重训，Git 代理改为 `10.12.201.122:39009`。
- 在完整运行时依赖解析下去除纯训练不需要的 Open3D/Jupyter/mypy 下载，按小段实际文件吞吐选择源。
- 保留 exp_1 的 PPO、奖励、2048 环境、5000 次迭代、seed=1 和每 100 轮保存设置，验证正式训练实际推进。

### 3. 修改内容

### 修改三：将分析与开发工具改为可选依赖（继承 exp_1 修改编号）

| 项目 | 旧值 | 新值 | 依据 |
| --- | --- | --- | --- |
| Open3D / Notebook | 默认强制安装 | `[analysis]` extras | 静态检索训练源码未找到对应 import；这是前次大型下载来源 |
| mypy / 类型 stub | 默认强制安装 | `[dev]` extras | 静态检查不属于 GPU 训练运行时 |
| 运行时依赖检查 | 完整 pip 解析 | 仍完整解析，无 `--no-deps` | 本地 Linux/Python 3.11 完整解析后确认四个工具包未重新引入 |
| 原工具安装方式 | 训练安装时捆绑 | 显式 `-e './src/holosoma[analysis,dev]'` | 保留分析和开发工具的可用性 |

### 修改四：测速选择包源并支持子实验编号

清华、阿里云、PyPI 分别探测索引和同一个公开 wheel 的前至多 1 MiB，比较吞吐后选择可达源。仅采样下载，不安装测速 wheel；单请求设置超时，全测速不可用时保留可达源回退。修复备用源使用相对链接时的解析，测试验证采样字节上限。正式入口允许 `exp_1.1` 这样的安全分段编号，仍拒绝 `../exp_1` 等路径。

### 4. 修改文件

- [训练包依赖声明](../../src/holosoma/pyproject.toml)、[安装说明](../../src/holosoma/README.md)：可选 extras。
- [Flux bootstrap](../../scripts/train_flux_smoke.py)：文件测速及源选择；安装/训练时限不变。
- [正式入口](../../scripts/train_flux.py)：支持点分子实验编号。
- [bootstrap 测试](../../tests/test_flux_smoke.py)、[正式入口测试](../../tests/test_flux_training.py)：测速选择、采样上限、编号和失败边界。
- [依赖约束](../../scripts/flux_smoke_constraints.txt)、Flux skill 和本笔记：保留原 IsaacSim 合同并记录实测范围。

### 5. 训练参数

| 参数 | 值 |
| --- | --- |
| 编号 / 训练方式 | `exp_1.1` / 从零；上次没有可续训模型 |
| 账号 / 用户 ID / 项目 | `limxmts0elyormeym6@uberip.com` / `4390` / `PRO_20261028_008` |
| 预设 / 仿真 / 地形 | `exp:g1-29dof` / `simulator:isaacsim` / `terrain:terrain-locomotion-plane` |
| 环境 / 迭代 / 保存 / seed | 2048 / 5000 / 100 / 1 |
| actor / critic 学习率 | 初始均 `1e-5`，adaptive；其余 PPO、奖励、机器人参数继承 exp_1 |
| 镜像 / GPU | `BJX00000178`，`V000420` / `ESKU000001`，1×4090D 24G |
| 安装 / 训练时限 | 900 秒 / 21600 秒；无自动重试 |
| 启动命令 | `gm-run holosama/scripts/train_flux.py --experiment exp_1.1 --num-envs 2048 --iterations 5000 --save-interval 100 --training-timeout 21600` |
| 代码 / 日志 | GitHub `main`，启动前核对 commit；TensorBoard 开启，Wandb/视频关闭 |

### 6. 预期与验收

| 指标 | exp_1 | 本轮目标 | 异常信号 |
| --- | --- | --- | --- |
| 安装 | Open3D 下载超时 | 900 秒内完成，默认无 Open3D/Notebook/mypy | 仍下载这些工具或超时 |
| GPU / 首轮训练 | 未进入 | 2048 环境完成实际 PPO 迭代，上传 checkpoint | OOM、资产/扩展异常、NaN/Inf |
| 完整训练 | 0/5000 | 5000 轮，最终 `model_04999.pt` 可加载且 ONNX 验收通过 | 提前终止、权重非有限或产物缺失 |
| 策略质量 | 无数据 | 沿用 exp_1 末段 episode 长度与奖励目标，待训练完成后判断 | 不以成功启动代替收敛 |

### 7. 实验结果

待训练完成后补充最终结果。开训前已完成纯运行时依赖的 Linux/Python 3.11 解析：未包含 Open3D、Notebook、JupyterLab、mypy；12 项 CPU 测试及 Ruff/格式检查通过。

新任务 `TASK_20261029_112`，2026-10-29 14:30:53 开始，指定账号用户 `4390`，状态 3。云端实际 Git commit=`ed6c3c5120b349fdeb99ac2e6950068cf9136d81`，参数为 2048 环境、5000 轮、保存间隔 100、安装 900 秒/训练 21600 秒上限。证据：[启动计划](artifacts/exp_1.1/launch_plan.json)、[云端实际计划](artifacts/exp_1.1/cloud_launch_plan.json)、[初始日志](artifacts/exp_1.1/startup_excerpt.txt)、[本地依赖验证](artifacts/exp_1.1/bootstrap_validation.json)。运行时包解析见 [linux_runtime_resolution.txt](artifacts/exp_1.1/linux_runtime_resolution.txt)，它不是对镜像已有包的安装锁文件。

#### 安装与实际开训验证

14:31:05 开始 bootstrap。文件测速记录：清华 408 KiB/s、阿里云 478 KiB/s、PyPI 文件源 2577 KiB/s，因此选择 PyPI。默认运行时安装不再拉取 Open3D、Notebook、mypy，保留镜像 PyTorch `2.7.0+cu128`。14:37:16 的训练运行目录已创建，随后产生实际 PPO 迭代和 checkpoint；安装在 900 秒上限内完成。

首次模型于 14:37:56 上传。下载 `model_00000.pt` 后验证：iter=0，5,300,224 bytes，actor/critic 张量均有限；内嵌实际配置为 2048 环境、5000 轮、seed=1、保存间隔 100，项目名 `flux-exp_1.1-20261029_063105`。SHA-256=`deb80277030404a20037dfef71b196742f5a1d53acf13353e9c453ded069a490`。该中间模型只缓存于 Git 忽略的 `logs/flux-validation/TASK_20261029_112/`，最终归档待训练完成。验证见 [initial_checkpoint_validation.json](artifacts/exp_1.1/initial_checkpoint_validation.json)。

#### 2026-10-29 14:41:54 进度快照

| 指标 | 当前观测 |
| --- | --- |
| 状态 / 迭代 | 3（Running）/ iter 158，计划 5000 轮，约完成 3.18% |
| Mean reward | 19.38（日志显示精度） |
| Mean episode length | 250.51 steps，按 0.02 s 换算约 5.01 s；不是独立行走回放验收 |
| 速度 / 每轮耗时 | 31,487 steps/s / 约 1.56 s |
| 预计剩余 | 日志 ETA=7825.6 秒，约 2 小时 10 分；早期估算会变化 |
| 平台 checkpoint | `model_00000.pt`、`model_00100.pt`；后者 14:40:33 上传 |
| 训练配置 | 下载模型内实际配置与计划一致；2048 环境已实际采样，无启动 OOM |

证据：[进度快照](artifacts/exp_1.1/progress_snapshot.json)、[近期训练日志片段](artifacts/exp_1.1/early_training_excerpt.txt)、[平台 precise 早期曲线](artifacts/exp_1.1/early_curves.json)、[已上传模型清单](artifacts/exp_1.1/models_snapshot.json)。平台 `task logs` 可只返回近期片段，启动/CUDA 探针行不在片段里不能解释为探针失败；实际迭代和可读模型证明已通过先行检查。

**阶段结论：依赖冷启动修复已通过实际安装、GPU 仿真、持续 PPO 迭代及周期保存验证。** 任务保持运行；5000 轮完成、最终 ONNX 验收和策略质量尚未判定。本任务没有自动重试或监控计划，6 小时训练时限仍有效。

#### 2026-10-29 15:39 训练中检查

`TASK_20261029_112` 仍为状态 3，用户 ID `4390`。日志和平台曲线均持续推进，约 iter 2440/5000（49%），累计约 1.19 亿采样步。日志速度约 32,319 steps/s，每轮 1.52 秒，ETA 约 3891 秒（65 分钟），按当时估算约北京时间 16:44–16:45 完成；该时间会变化。

| 指标 | 观测 |
| --- | --- |
| Mean reward | 平台最近 100 个点均值 4.2052；该次查询末点 iter 2435 为 4.9595 |
| Mean episode length | 平台最近 100 个点均值 691.3709 steps（约 13.83 秒）；末点 iter 2437 为 704.35 steps |
| 惩罚课程系数 | 最近 100 个点均为 1.0，已达完整惩罚强度 |
| 最新 checkpoint | `model_02400.pt`，15:38:41 上传；平台共列出 25 个周期 checkpoint |

不同指标串行查询，末点相差数轮，不能拼成同一轮的精确观测。早期 iter 158 的惩罚系数较小，当前奖励不宜直接与当时 19.38 的值比较。当前 episode 长度较早期明显增加，仍低于本轮末段 100 轮均值 800 steps 的目标；训练未完成，不提前判定达标或收敛。此次检查没有停止、重启或修改任务。

本次独立快照：[日志片段](artifacts/exp_1.1/progress_20261029_153936/log_excerpt.txt)、[奖励曲线](artifacts/exp_1.1/progress_20261029_153936/Train_mean_reward.json)、[episode 曲线](artifacts/exp_1.1/progress_20261029_153936/Train_mean_episode_length.json)、[惩罚系数曲线](artifacts/exp_1.1/progress_20261029_153936/Env_penalty_scale.json)。

#### 最终模型回放与录像安排（2026-10-29）

用户要求训练完成后进行回放并录制视频。入口 [replay_flux.py](../../scripts/replay_flux.py) 已准备：只使用本轮最终 `model_04999.pt`，先验证训练终态、iter、有限权重及下载 SHA-256，再用原生确定性评估单环境回放 30 秒（1280×720，50 Hz / 1500 步）。评估 timeout 显式为 60 秒，跌倒仍重置；录像不用于冒充训练 20 秒回合上限下的存活指标。

入口沿用安装 bootstrap，安装 900 秒、回放 1200 秒时限；片段拼接为浏览器兼容 H.264，完整解码检查时长、尺寸、空帧/静止/截断，成功后预留 60 秒上传。19 项 CPU 测试通过，包含下载完整性、签名位置不进入错误信息，以及正常/黑屏/静止/截断视频验收；Ruff 通过。云端回放尚未开始，任务号与实际视频验证待补充；目标产物为 `czy/data/exp_1.1/play_output.mp4`，不新增训练实验编号。

回放入口已推送至 `80121a9`。已启用本聊天心跳「Holosoma 训练完成后回放录像」（ID `holosoma`），每 5 分钟检查并接续；正常运行无变化时保持安静，完成/失败/需用户处理时通知，交付后停止。接续器在忽略的本地运行目录持久化任务状态，创建结果不明确时不重复发起；源任务失败不自动重训。此安排取代前述 14:41 快照时的“无监控计划”状态。

#### 最终训练验收（2026-10-29 16:48）

**5000 轮正式训练完成，流程及预设末段指标目标通过，独立回放录像仍在进行。** 任务 `TASK_20261029_112` 状态 5，14:30:53 开始、16:45:04 结束，平台运行 8051 秒（2 小时 14 分 11 秒，含安装/上传）。PPO 本体累计 7509.13 秒、245,760,000 采样步，最后迭代索引 4999。

| 指标 | 最终值 / 验收 |
| --- | --- |
| Mean reward | 末段 100 轮均值 15.073078；末点 16.407948；高于 exp0 的 -3.955621，但规模与训练阶段不同，不作直接同条件比较 |
| Mean episode length | 末段 100 轮均值 887.925098 steps ≈ 17.7585 秒，高于目标 800 steps / 16 秒；末点 892.409973 steps |
| 训练速度 | 末段 100 轮均值 32,790.79 steps/s，末点 32,074 steps/s |
| 曲线覆盖 | reward / episode / fps 均获取 steps 0..4999 共 5000 个 precise 实测点；末段统计取 4900..4999 |
| checkpoint | `model_04999.pt`，iter=4999，5,300,224 bytes，actor/critic 张量均有限，配置来自本轮实际 checkpoint |
| SHA-256 | `bca3da441b1a417c1b76e82cd548852a5de54faa920f4034e6c54db80bb11105` |
| 云端 ONNX | 958,249 bytes，CPU 推理输出 `[1,29]` 且有限，TensorBoard 文件 1 个；日志存在 `FLUX_TRAIN_SUCCESS` |
| 平台上传 | 最终模型 16:44:10 上传；共 51 个 checkpoint。ONNX 云端校验通过，未作为本地 ONNX 文件交付 |

证据：[最终任务快照](artifacts/exp_1.1/final_progress_snapshot.json)、[最终日志](artifacts/exp_1.1/final_training_excerpt.txt)、[完整平台曲线](artifacts/exp_1.1/final_curves.json)、[末段统计](artifacts/exp_1.1/final_curve_statistics.json)、[checkpoint 校验](artifacts/exp_1.1/final_checkpoint_validation.json)、[上传清单](artifacts/exp_1.1/final_models_snapshot.json)。最终模型已归档至 [model_04999.pt](../data/exp_1.1/model_04999.pt)。曲线及权重验收不能替代独立回放中对步态、跌倒和命令跟踪的观察。

#### 回放任务启动

指定账号用户 `4390` 下已创建并启动唯一回放任务 `TASK_20261029_249`，使用入口 commit `ab53ac3832d4096d17ca217dbabb91b707689521`、同一 IsaacLab 镜像/单 GPU，源模型 SHA-256 如上。16:48 检查为状态 3，正在安装运行时依赖；此时尚未产生经验证的视频。后续继续核查原生评估、编码、平台 MP4 上传和本地解码，不以任务启动作为录像成功。

#### 回放收尾与任务澄清

云端回放 `TASK_20261029_249` 于 17:12:08 以状态 6 结束，运行 1467 秒。完整日志显示完成片段保存后，在 IsaacSim 录像资源清理时出现 `HydraTexture has no attribute split`，应用未正常关闭，最终达到 1200 秒回放子进程上限；未形成经验证的合并 `play_output.mp4`，不自动重试。摘要证据见 [replay_failure_summary.json](artifacts/exp_1.1/replay_failure_summary.json)。

本机使用最终 checkpoint 完成 MuJoCo 策略闭环及 30 秒录像（[play_mujoco.mp4](../data/exp_1.1/play_mujoco.mp4)）；另一次固定前进命令 0.3 m/s、30 秒无渲染验证未触发本地跌倒检测，1500 控制步状态有限，20 秒时按本地回合时限重置。此判据不是 IsaacSim 的接触终止统计，不能称为不间断站立 30 秒。记录见 [local_mujoco_simulation.json](artifacts/exp_1.1/local_mujoco_simulation.json)。本机版本 MuJoCo 3.15.0，与项目 setup 的 3.10.0 不同；只记录本机实测范围。

用户明确不需要不同输入的误差对比，已移除本地导出入口中的随机输入误差对比。用户同时确认目标应为搬箱子：exp_1.1 实际是平地行走，其模型不具备箱体交互训练目标。保留该历史结果，搬箱子切换任务并新建 exp_2。

## 实验 exp_2：G1 带箱体的搬运跟踪训练

### 1. 上一实验结果与教训

exp_1.1 已完成 2048 环境、5000 轮平地 PPO，末段 100 轮 reward 均值 15.073078、episode 长度均值 887.925098 steps（17.7585 秒），最终模型与 ONNX 验收通过。但它使用行走速度指令，没有箱体和搬运奖励，不能作为搬箱子策略。云端录像清理异常与本机行走回放均不改变这个任务范围。

**核心教训：先核对任务目标、场景对象、动作数据和奖励项，不能以行走流程跑通替代搬箱子训练。**

### 2. 本轮修改目标

- 使用 `exp:g1-29dof-wbt-w-object`，训练 G1 跟踪仓库中的抬箱/搬运参考动作，同时驱动物理箱体。
- 开训前验证 NPZ 中机器人与箱体轨迹、关节名、哈希，以及自由箱体、手部碰撞 URDF 和箱体奖励。
- 首阶段为 1024 环境、8000 轮 PPO；保持原生 WBT 奖励和优化参数。8000 轮是首阶段训练预算，原预设完整默认为 30000 轮，不能预先宣称策略收敛。
- 验收分开判定：先确认有箱体的 GPU 训练实际推进、checkpoint/ONNX 可用；训练后再评估完整搬运片段中的箱体跟踪误差、抬升和跌落，不能只看行走 episode 长度。

### 3. 修改内容

### 修改一：将任务由平地行走切换到带物体 WBT

| 项目 | exp_1.1 | exp_2 |
| --- | --- | --- |
| 预设 / 环境 | `g1-29dof` / locomotion | `g1-29dof-wbt-w-object` / WholeBodyTrackingManager |
| 指令 | 随机线速度/角速度 | `sub3_largebox_003_mj_w_obj.npz` 的机器人与箱体参考轨迹 |
| 场景 | 机器人、平地 | 机器人、平地及 `object` 自由箱体 |
| 手部碰撞资产 | `g1_29dof.urdf` | `main_mesh_collision_halfspherehand.urdf` |
| 箱体奖励 | 无 | 位置/姿态跟踪指数奖励，原生权重各 1.0 |
| 初始化 | 从零 | 从零；不续训行走 checkpoint |

### 修改二：为箱体任务添加独立启动与验收入口

新增 `scripts/train_flux_box.py`，复用已验证的 Flux 依赖 bootstrap。云端安装后先检查任务配置和 motion 哈希，再运行 128 环境 × 2 轮；仅在箱体 checkpoint 配置、有限权重/归一化统计、ONNX 和 TensorBoard 验收通过后，自动进入正式阶段。任一阶段失败立即退出，不自动重试，也不静默退回行走任务。

### 4. 修改文件

- [train_flux_box.py](../../scripts/train_flux_box.py)：固定带箱体 WBT 预设，数据/任务检查、分阶段训练及成功验收。
- [test_flux_box.py](../../tests/test_flux_box.py)：防止误用机器人单体动作或行走预设，验证 smoke 失败阻断正式训练。
- [Flux 参考文档](../skills/flux-cli/references/holosoma-isaaclab.md) 和本笔记：记录任务边界与实际验证范围。

### 5. 训练参数

| 参数 | 值 |
| --- | --- |
| 账号 / 用户 / 项目 | `limxmts0elyormeym6@uberip.com` / `4390` / `PRO_20261028_008` |
| 镜像 / 资源 | `BJX00000178` / `V000420`；`ESKU000001`，1×4090D 24G |
| 预设 / 算法 / seed | `exp:g1-29dof-wbt-w-object` / PPO / 1 |
| 小规模门槛 | 128 环境，2 轮，每轮保存；不作为策略质量验收 |
| 正式环境 / 轮数 / 保存 | 1024 / 8000 / 每 200 轮，最终 `model_07999.pt` |
| 参考动作 | 325 帧、50 Hz，名义 6.5 秒；箱体参考高度范围 0.1831–0.8449 m |
| motion SHA-256 | `76077a5035726aabe25b084de5414f3113646c43ac1bfa6ccc0742e46b1a4cea` |
| actor / critic 学习率 | 原生 WBT 初始均 `1e-3`，adaptive；启用经验观测归一化、关闭 symmetry |
| 物理 / 控制 / 单回合上限 | 200 Hz / 50 Hz / 原生 WBT 10 秒；参考动作结束及跟踪失败可提前终止 |
| 动作缩放 | `0.25 × effort_limit / kp`；启用自碰撞和手部交互模型 |
| 箱体 | `large_box.urdf` 自由刚体；URDF 基准质量 0.1 kg，startup mass randomizer 为加性 `[1.0,4.0]` kg，因此配置理论范围 1.1–4.1 kg；沿用摩擦/惯量随机化，实际采样仍以云端为准 |
| 安装 / smoke / 正式时限 | 900 / 900 / 21600 秒，失败不自动重试 |
| 记录 / 录像 | TensorBoard 开启，Wandb/训练视频关闭；视频评估放到训练后 |
| 启动命令 | `gm-run holosama/scripts/train_flux_box.py --experiment exp_2 --num-envs 1024 --iterations 8000 --save-interval 200 --training-timeout 21600` |

### 6. 预期与验收

| 指标 | 本轮目标 / 判定方式 |
| --- | --- |
| 任务身份 | 实际 checkpoint 有箱体 scene、带物体 motion、手部碰撞 URDF、箱体位置/姿态奖励 |
| 初期启动 | 有箱体的 GPU 仿真完成 128 环境 smoke，随后 1024 环境持续迭代，无 OOM / 非有限状态 |
| 首阶段完成 | 8000 轮，最终 iter=7999，actor/critic/观测归一化状态有限，ONNX 输出 29 个有限动作 |
| 箱体跟踪 | 记录 object position/orientation reward 及真实位置误差；不套用行走 reward 阈值 |
| 搬运质量 | 训练后独立回放完整参考片段，报告箱体是否抬起、是否脱手/跌落、轨迹误差；完成训练不等于通过此项 |

### 7. 实验结果

待创建训练任务。开训前已核实仓库确有带物体的 WBT 预设、自由箱体场景和 325 帧/29 关节的 motion 数据，全部数值数组有限；启动入口与相关 CPU 检查 15 项通过。本地在隔离环境实际解析与云端相同的 CLI，确认箱体 fixed=False、1024 环境、8000 轮、经验归一化开启、录像关闭，并执行 motion/资产预检查：[计划配置](artifacts/exp_2/planned_config.json)、[本地预检查](artifacts/exp_2/box_preflight.json)。云端 box preflight、GPU smoke 和正式迭代尚待实际运行，不能以本地配置检查替代云端验证。

已创建并启动 `TASK_20261029_261`，账号用户 `4390`，17:22:52 开始；17:23:50 检查为状态 3，云端 `FLUX_BOX_PLAN` 记录实际 commit=`b482369b4a75dae4796ae183c06dd4cc94d88c5e`，正式计划为 1024 环境、8000 轮、每 200 轮保存，smoke 为 128 环境 × 2 轮。此时处于依赖安装阶段，尚不能称为正式 PPO 已开始。[提交计划](artifacts/exp_2/launch_plan.json)、[云端实际计划](artifacts/exp_2/cloud_launch_plan.json) 和 [进度快照](artifacts/exp_2/progress_snapshot.json) 保留身份与阶段证据。

17:26 在审查启动路径时发现：`bootstrap.prepare_environment()` 会切换 cwd 到 clone 根目录，而新增入口此前先在 SDK 工作目录创建相对 `logs/<project>`，之后 preflight 会向 clone 下尚不存在的项目目录写入结果。该问题尚未在 worker 触发，但已能由源码和切换 cwd 的回归场景确定。为避免继续安装后必然失败，主动停止当前仍在安装阶段的任务；17:26:38 终态 6，运行 226 秒，日志没有 `FLUX_BOX_PREFLIGHT`、GPU smoke 或 PPO 迭代。按失败启动保留本编号，修复记为 exp_2.1，不虚构已出现的云端异常栈。

## 实验 exp_2.1：修正搬箱子入口的工作目录

### 1. 上一实验结果与教训

exp_2 的带箱体 WBT 配置、CLI 和 motion 在本地验证通过，但任务在安装阶段被主动停止，运行 226 秒，0 轮 PPO、无 checkpoint。原因是新增入口在 bootstrap 切换 cwd 前创建相对结果目录；原有使用空字典替代 bootstrap 的测试未覆盖真实的目录切换行为。

**核心教训：包装入口的本地检查必须覆盖 SDK 初始目录与 checkout 目录不同的情况。**

### 2. 本轮修改目标

- 修正 clone 根目录下输出位置，确保 preflight、训练和验收使用同一项目目录。
- 保持 exp_2 的箱体场景、motion、奖励、PPO、1024 环境和 8000 轮首阶段预算。
- 重新启动后确认真正完成带箱体 smoke，并观察正式阶段实际迭代。

### 3. 修改内容

### 修改三：先建立 bootstrap 环境，再创建绝对输出目录

`prepare_environment()` 在创建日志和计划前执行，项目根目录转为绝对路径传给新解释器。新增回归场景从模拟 SDK 目录启动，bootstrap 切换至另一 checkout，检查 preflight 目标目录及 `launch_plan.json` 确实在 checkout 下；4 项搬箱子入口测试与 Ruff 均通过。

### 4. 修改文件

- [train_flux_box.py](../../scripts/train_flux_box.py)：调整环境准备顺序，使用绝对项目目录。
- [test_flux_box.py](../../tests/test_flux_box.py)：覆盖 bootstrap 改变工作目录的真实边界。
- 本笔记：保留 exp_2 主动终止证据和子实验重启记录。

### 5. 训练参数

| 参数 | 值 |
| --- | --- |
| 编号 / 初始化 | `exp_2.1` / 从零，父实验尚无模型 |
| 账号 / 项目 | `limxmts0elyormeym6@uberip.com`，用户 4390 / `PRO_20261028_008` |
| 预设 / motion / 箱体 | 与 exp_2 相同：`g1-29dof-wbt-w-object`，带箱体轨迹与自由箱体 |
| smoke / 正式 | 128 环境 × 2 轮 / 1024 环境 × 8000 轮，每 200 轮保存 |
| 镜像 / GPU | `BJX00000178` / `V000420` / 单 4090D 24G |
| 时限 | 安装 900 秒、smoke 900 秒、正式训练 21600 秒，无自动重试 |
| 其他 PPO / 物理 / 随机化 | 继承 exp_2，不修改奖励、数据或算法参数 |
| 启动命令 | `gm-run holosama/scripts/train_flux_box.py --experiment exp_2.1 --num-envs 1024 --iterations 8000 --save-interval 200 --training-timeout 21600` |

### 6. 预期与验收

先验收实际云端 `FLUX_BOX_PREFLIGHT`、`FLUX_BOX_SMOKE_SUCCESS` 及 `FLUX_BOX_TRAIN_START`，并确认 1024 环境的正式迭代和 checkpoint。后续完整首阶段、箱体姿态/位置跟踪及独立搬运视频标准与 exp_2 §6 相同。通过路径回归测试不代表已开训或学会搬运。

### 7. 实验结果

工作目录回归和三项原保护测试共 4 项通过；父实验已确认终止，避免与重启任务并行消耗算力。修复已推送 `9d2845701df5a644ebfc1e36d3911ea36475f4c6`，重启任务 `TASK_20261029_267` 于 17:29:06 开始，指定账号用户 `4390`；17:29:56 为状态 3。云端实际 `FLUX_BOX_PLAN` 的 commit 与修复一致，处于依赖安装阶段，GPU smoke 和正式迭代尚未判定。[重启计划](artifacts/exp_2.1/launch_plan.json)、[云端实际计划](artifacts/exp_2.1/cloud_launch_plan.json)、[进度快照](artifacts/exp_2.1/progress_snapshot.json)。

#### 带箱体 smoke 和正式开训验收（2026-10-29 17:36）

**带箱体 GPU smoke 已通过，1024 环境的正式 PPO 已实际开始。** 云端约 17:33:58 创建 smoke 运行目录，17:34:14 完成 2 轮，17:34:16 打印 `FLUX_BOX_SMOKE_SUCCESS`；其后自动进入 `FLUX_BOX_TRAIN_START`。smoke 最终 iter=1，checkpoint 6,839,751 bytes，ONNX 1,157,380 bytes，CPU 推理输出 `[1,29]` 有限，TensorBoard 文件 1 个，object task contract 验收为 true。[云端 smoke 验收](artifacts/exp_2.1/cloud_smoke_validation.json)、[成功日志片段](artifacts/exp_2.1/smoke_success_excerpt.txt)。

正式首个 checkpoint `model_00000.pt` 已上传并下载核对：实际项目 `flux-exp_2.1-box-20261029_092927`（无 smoke 后缀），num_envs=1024，计划 8000 轮，iter=0；actor/critic/actor 观测归一化张量有限。actor/critic 输入分别 154/298 维，与行走的 100/103 维不同；嵌入配置含自由箱体 fixed=False、`main_mesh_collision_halfspherehand.urdf`、带物体 motion 和两项箱体奖励。大小 6,839,751 bytes，SHA-256=`6080c2493b5fa69ef909500e5fdaf7595689bf04c61751979395c0e246a4883d`。中间 checkpoint 只缓存于 Git 忽略的 `logs/flux-validation/TASK_20261029_267/`，不冒充最终模型。[正式首个模型验收](artifacts/exp_2.1/formal_initial_checkpoint_validation.json)、[smoke 首个模型核对](artifacts/exp_2.1/smoke_initial_checkpoint_validation.json)、[已上传模型清单](artifacts/exp_2.1/models_snapshot.json)。

| 指标 | iter 48/8000 的完整日志块 |
| --- | --- |
| 平台状态 / 账号用户 | 3 / 4390 |
| Mean reward | 0.07（日志精度） |
| Mean episode length | 10.32 steps，约 0.2064 秒；初始策略频繁跟踪失败，不是完整搬运表现 |
| 箱体位置 / 姿态奖励 | 0.0167 / 0.0131（日志中 episode 奖励项，不反推位置 RMSE） |
| 速度 / ETA | 11,810 steps/s / 17,184.1 秒，约 4 小时 46 分；早期估计会变化 |

平台实际 keys 已包含箱体位置/姿态奖励及机器人跟踪误差，已取得 `precise` 初期曲线。[早期正式指标](artifacts/exp_2.1/early_formal_metrics.json)、[平台曲线](artifacts/exp_2.1/early_curves.json)。同一个任务含 smoke 和正式两套事件文件，0..1 step 可能重叠；曲线默认 session 不直接用来辨别阶段，阶段身份以 checkpoint 配置和正式日志为准。日志目录仍含旧的 `-locomotion` 后缀，此名称不是任务判据，实际 env_class/scene/motion/reward 均已核对为带箱体 WBT。

**阶段结论：✅ 搬箱子任务身份、GPU smoke、1024 环境开训和首个模型上传/下载通过；8000 轮完成与搬运质量均待后续验收。** 无自动重试，不进行不同输入的误差对比。训练保持运行，正式阶段 21600 秒上限不变。

#### 2026-10-29 18:41 起的训练进度检查

任务 `TASK_20261029_267` 保持状态 3，所属用户仍为 4390。日志读到 iter 1831/8000；稍后的 precise 曲线已到 1859–1860 轮，约完成 23.3%。近期日志未发现异常栈、超时或 OOM，周期 checkpoint 持续上传，最新 `model_01800.pt` 于 18:40:22 上传。

| 指标 | 最近连续 100 轮 / 最新日志 |
| --- | --- |
| Mean reward | 平台 steps 1760..1859 均值 3.901075；开训 iter 48 日志为 0.07 |
| Mean episode length | 同窗口均值 68.431500 steps，控制步长 0.02 秒，约 1.3686 秒；开训日志 10.32 steps |
| 箱体位置跟踪奖励 | steps 1760..1859 均值 0.116620 |
| 箱体姿态跟踪奖励 | steps 1761..1860 均值 0.101844；查询时间略晚，窗口与前三项差一轮 |
| 训练速度 | steps 1761..1860 平均 10,355.47 steps/s；iter 1831 日志为 11,212 steps/s |
| 日志 ETA | iter 1831 为 13,359.3 秒，约 3 小时 43 分；按该次查询时间估算约 22:25 完成，会随速度变化 |

**判断：训练持续推进，奖励和回合长度较初期有所提高，但当前平均回合约 1.37 秒，仍不足以证明能稳定完成整段 6.5 秒参考搬运。** 箱体项是 episode 奖励，不直接反推位置 RMSE；完整抬升、搬运和脱手/跌落情况仍待独立回放。任务保持运行。

本次单独保存：[日志片段](artifacts/exp_2.1/progress_20261029_184123/startup_excerpt.txt)、[日志指标](artifacts/exp_2.1/progress_20261029_184123/early_formal_metrics.json)、[平台 precise 曲线](artifacts/exp_2.1/progress_20261029_184123/early_curves.json)、[连续窗口统计](artifacts/exp_2.1/progress_20261029_184123/curve_statistics.json)。日志与 API 查询时间不同，不能把两者末点视为同一轮。

#### 最终训练与模型验收（2026-10-30 收尾）

**训练流程成功完成，搬运质量部分达标，尚不能交付稳定搬箱策略。** `TASK_20261029_267` 于 2026-10-29 **22:45:11** 进入成功终态 5，归属仍为指定账号用户 4390；从 17:29:06 起平台总运行 18,965 秒（5 小时 16 分 5 秒，包含安装、smoke 和上传，不能当作纯 PPO 耗时）。实际训练 commit 为 `9d2845701df5a644ebfc1e36d3911ea36475f4c6`。完整 8000 轮、最终 iter=7999，完整日志含 `FLUX_BOX_TRAIN_SUCCESS`。[最终汇总](artifacts/exp_2.1/final_summary.json)、[脱敏完整日志（gzip）](artifacts/exp_2.1/final_training_log.txt.gz)、[云端最终验收](artifacts/exp_2.1/final_flux_box_train_success.json)。

最终 [model_07999.pt](../data/exp_2.1/model_07999.pt) 已下载，6,839,751 bytes，SHA-256=`e2d120660bbc199d0dc03d0bce17a4989ef6117877c2f97f9da72d2c4dbc2c97`；iter、正式项目、1024 环境和 8000 轮计划一致，actor/critic 与两套观测归一化状态均有限，动态箱体任务配置通过。云端 ONNX 1,157,383 bytes，CPU 输出 `[1,29]` 有限，TensorBoard 文件 1 个；未取得 ONNX 本地下载件，不把云端导出验收写成已下载。[模型校验](artifacts/exp_2.1/final_checkpoint_validation.json)、[模型嵌入的实际配置](artifacts/exp_2.1/actual_final_config.json)。

| 指标 | precise 曲线末段 100 轮（7900..7999）均值 | 最后一轮 7999 |
| --- | --- | --- |
| Mean reward | 6.877422 | 6.208186 |
| Mean episode length | 128.145200 steps，约 2.562904 秒 | 115.320000 steps，约 2.306400 秒 |
| 箱体位置奖励 | 0.215164 | 0.214359 |
| 箱体姿态奖励 | 0.198361 | 0.193227 |
| 训练速度 | 9,094.81 steps/s | 8,284 steps/s |

五项曲线均有 steps 0..7999 共 8000 点：[平台最终曲线](artifacts/exp_2.1/final_curves.json)、[窗口统计](artifacts/exp_2.1/final_curve_statistics.json)。相较早期回合约 0.2064 秒，跟踪持续时间提高，但 2.56 秒仍不足以说明稳定覆盖 6.5 秒整段参考片段；原生 WBT 在 motion 片段末尾可重置机器人和箱体而不结束 episode，因此也不能用 episode 长度反推出连续完整搬运次数。箱体奖励不反推位置 RMSE。

#### 最终策略的本机 MuJoCo 物理回放

**后续更正：本节及下一节的初版回放存在根部初始速度适配错误，旧录像保留为历史证据；修复版结果见末尾「失败诊断与根部速度修正」。初版数据不能作为严格复现训练初态的评估。**

新增 [replay_g1_box_mujoco.py](../../scripts/replay_g1_box_mujoco.py)，直接加载上述 checkpoint 的确定性 actor、154 维观测及保存的经验归一化（`(obs-mean)/(std+0.01)`），按训练关节名/参考动作顺序重排，保留保存的 PD、力矩限幅及 `0.25 × effort_limit / kp` 动作缩放。使用训练的手部碰撞 URDF、同一箱体碰撞网格；机器人/箱体仅在起始帧赋值，此后由力矩、自由刚体接触与 `mj_step` 推进，无轨迹强制写入、无自动复位、无物体位姿修正。此回放不是原始参考动作动画，也不进行不同输入的误差对比。

本机 MuJoCo **3.15.0**，200 Hz 物理 / 50 Hz 控制，单次名义质量 **2.6 kg**，箱体 geom 滑动摩擦系数 0.35。初始根位置与参考一致，torso 的 URDF FK 与 motion 位置有约 0.01 m 差异；碰撞已启用、29 个电机使用单位 gear。MuJoCo 接触求解、摩擦混合与 PhysX 不同，未逐项复现训练域随机化；这是跨仿真单次物理回放，不能替代原生 IsaacLab 评估或归因所有失败于策略训练。

| 实测结果 | 数值 / 观察 |
| --- | --- |
| 回放视频 | [play_box_mujoco.mp4](../data/exp_2.1/play_box_mujoco.mp4)，1280×720 / 50 fps / 325 帧 / 6.5 秒，H.264/yuv420p |
| 箱体首次越过跟踪阈值 | 0.66 秒（位置 >0.25 m 或姿态 >0.8 rad） |
| 箱体位置 RMSE / 最大误差 | 1.208848 / 1.629077 m，单次完整无复位回放的轨迹统计 |
| 箱体最高中心高度 | 0.240614 m；参考片段最高 0.8449 m，未完成预期抬升 |
| 机器人最低根高度 | 0.062949 m；代表帧约 2 秒已跌倒，箱体留在地面 |
| 媒体验证 | OpenCV 全 325 帧解码及 FFmpeg `-xerror` 全片解码通过；非空画面，0/0.5/1/2/4/6 秒代表帧已人工查看 |
| 视频大小 / SHA-256 | 447,470 bytes / `774e8f6384b38910839bd22fbd9ef7ca1079a505fce852ea09b062b1c2d281fd` |

[物理回放与逐帧轨迹](artifacts/exp_2.1/local_box_replay_validation.json)、[视频解码校验](artifacts/exp_2.1/local_box_video_validation.json)、[代表帧](artifacts/exp_2.1/local_box_replay_frames.jpg)。复现：`python scripts/replay_g1_box_mujoco.py`，默认读取本轮最终模型并保存上述视频和证据；脚本 Ruff 检查与实际物理运行通过。

**分级结论：✅ 8000 轮训练、最终 checkpoint/云端 ONNX/上传下载通过；❌ 本次本机 MuJoCo 完整搬运未通过；⚠️ 原生 IsaacLab 的最终策略搬运质量尚无独立回放验收。** 没有自动续训、切换账号或创建新计费回放任务。后续应先在原生环境评估箱体/机器人跟踪与终止原因，再决定是否增加训练预算或修正跨仿真适配，不能只凭本次视频自动重训。

#### 按用户要求去掉箱体，仅观察策略动作（2026-10-30）

在同一最终模型 `model_07999.pt` 上增加 `--no-box` 回放选项：从 MuJoCo 场景中完整移除箱体刚体、碰撞与显示，实际箱体 body 数为 0，保留原来的参考片段、actor/观测归一化、机器人碰撞及 PD 力矩控制，不重新训练、不强制写入参考关节动作。无箱体版本单独保存，不覆盖前述带箱回放。

复现命令：`python scripts/replay_g1_box_mujoco.py --no-box`。[无箱体视频](../data/exp_2.1/play_robot_only_mujoco.mp4) 为 1280×720、50 fps、325 帧、6.5 秒，499,320 bytes，SHA-256=`c216ce42d0480dfddd4e20ca24ebd0fb73346f19c404779dc6f854d85402e466`。OpenCV 和 FFmpeg 全片解码通过，0/0.5/1/2/4/6 秒代表帧已人工检查，画面确认没有箱子。机器人约 1 秒坐倒、2 秒侧倒，最低根高度 0.066108 m；移除箱体后仍未保持动作跟踪，不能据此认定之前的失败仅由箱体尺寸引起，也不能将无负载表现当成搬运能力验收。

[无箱体运行证据](artifacts/exp_2.1/local_robot_only_replay_validation.json)、[视频校验](artifacts/exp_2.1/local_robot_only_video_validation.json)、[代表帧](artifacts/exp_2.1/local_robot_only_replay_frames.jpg)。本次仅补充同一实验 §7，不新增训练编号。

#### 失败诊断与根部速度修正（2026-10-30）

按用户要求排查失败原因，确认新增 MuJoCo 适配器有一处初态错误：`body_lin_vel_w` 是质心线速度，不能直接写入自由根关节的平移速度。根据 `v_origin = v_COM - omega_world × (p_COM-p_origin)` 转换后，与 motion 的 `joint_vel[0,:3]` 仅相差约 `1.02e-7 m/s`，证实原实现造成约 **0.19454 m/s** 初始速度偏差。现在直接读取 NPZ 保存的原生 MuJoCo `joint_vel[0,:6]`（世界系原点线速度 + 局部系角速度）；此修复不改变模型、观测归一化、PD、参考片段或训练权重。

修复后分别重跑一次有箱和无箱物理回放，均为 325 帧 / 6.5 秒 / 720p，完整解码与代表帧人工检查通过：

| 条件 | 修正后的实际结果 | 视频 / 证据 |
| --- | --- | --- |
| 动态箱体 2.6 kg | 箱体首次越阈值 1.12 秒，最高中心 0.262192 m，位置 RMSE 1.228499 m；约 2 秒机器人跌倒，最低根高度 0.062081 m | [录像](../data/exp_2.1/play_box_velocity_fixed.mp4) / [轨迹](artifacts/exp_2.1/local_box_velocity_fixed.json) / [视频校验](artifacts/exp_2.1/local_box_velocity_fixed_video.json) |
| 完全移除箱体 | 箱体 body 数 0，约 2 秒机器人仍跌倒，最低根高度 0.071294 m | [录像](../data/exp_2.1/play_robot_only_velocity_fixed.mp4) / [轨迹](artifacts/exp_2.1/local_robot_only_velocity_fixed.json) / [视频校验](artifacts/exp_2.1/local_robot_only_velocity_fixed_video.json) |

因此根部速度适配错误是已确认的问题，**不是全部失败的解释**。初始接触检查为足部与地面约毫米级穿入，电机 gain/gear 为 1、实际关节力矩非零；没有发现「未施加控制」或严重初始自穿透的证据。观测项顺序、154 维归一化公式、动作缩放、PD、控制频率及关节名映射已按源码核对，但尚不能据此宣称 PhysX 与 MuJoCo 完全等价。

训练侧同样没有达到稳定完整跟踪的证据：末段平均回合 2.56 秒，低于 10 秒 timeout；本配置的另一终止项为 bad_tracking（机器人姿态、手脚高度或箱体跟踪）。最后两个千轮窗口的奖励均值仍从 6.459955 升至 6.767822，episode 均值从 120.078130 升至 127.319900 steps。仅完成预算 8000 轮不等于收敛；原生预设 30000 轮是预算对照，不能保证延长即可成功。训练从自适应采样的片段相位重置，因此也不能用平均回合时长直接计算整段成功率。

**诊断边界：已确认训练稳定跟踪未达标、初版回放有速度适配错误；修复后两种场景仍失败。其他接触/碰撞、惯量与求解器差异对失败的贡献，以及原生环境究竟是哪类 bad_tracking 终止，仍需原生 IsaacLab 无视频评估和终止分项统计。** 无箱体仍跌倒不能支持「箱体尺寸是唯一原因」，且去掉箱体改变了训练时的交互条件，不构成完整因果隔离。未自动创建计费任务或重训。[诊断数值与限制](artifacts/exp_2.1/failure_diagnosis.json)。

#### 用户授权的原生 IsaacLab 评估与账号切换（2026-10-30）

用户明确指定切换到 **`limxmts0f9p800yu7g@uberip.com`**，已通过 `auth whoami` 核对用户 ID **4391**，每次 CLI 子进程均绑定该账号凭据，评估不得回退到原账号。新建专用项目 `PRO_20261030_002`（Holosoma native IsaacLab evaluation）。原训练任务与源模型仍属于原账号 4390，本次从已授权推送的仓库读取同一个最终模型，核对 SHA-256，不查询旧账号私有任务或复用其签名链接，不重训、不更新权重。

新账号目录查询提供的兼容镜像为 `BJX00000178` / **`V000446`**（`isaac-lab2.3.2-v3`，IsaacSim 5.1 / IsaacLab 2.3.2 / Python 3.11.13 / PyTorch 2.7.0），算力 `ESKU000001` / `SKUSL000002`，单 4090D 24G。旧训练使用的 `V000420` 不在本次版本列表中；本次应记录实际 v3 运行环境，不能假称镜像完全一致。[账号与平台预检查](artifacts/exp_2.1/native_isaaclab/platform_preflight.json)。

新增 [eval_flux_box.py](../../scripts/eval_flux_box.py)：原生 `setup_simulation_environment` + 保存配置的 PPO 确定性策略和经验归一化，**32 环境 × 每环境 4 回合 = 128 回合**，最多 2500 控制步。原生 eval 从片段起点复位；保留训练的动态箱体、机器人、PD、奖励、终止阈值及 startup/reset 随机化，关闭观测噪声与录像，原生 eval 模式关闭推搡，timeout 恢复为训练的 10 秒。安装和评估子进程各上限 900 秒，无自动重试。

在 `termination_manager.check()` 返回后、环境复位前只读捕获机器人参考高度、姿态、手脚高度、箱体位置/姿态五类 native bad_tracking 原因及 timeout；多个原因可同时命中，次数不可直接相加当作回合数。逐环境固定 4 次避免快失败环境占用样本；片段成功定义为无失败到达参考最后一帧，单独记录，不能用「原生片段结束后自动复位」后的存活时间当作连续搬运。轨迹误差按复位前采样，不能直接和本机无复位整段 RMSE 比大小。

CPU 检查已用真实最终 checkpoint 重建 typed config，确认物理/随机化/命令/终止配置未改；3 项测试覆盖配置、终止前快照/重叠原因/每环境配额、样本不足拒绝验收，Ruff 通过。当前仅为本地与平台准备验收，原生 GPU 评估结果待实际任务返回，不新增训练实验编号。

#### 原生评估完成：流程通过，策略质量未达标（2026-10-30 09:14）

新账号 **`limxmts0f9p800yu7g@uberip.com` / 4391** 下的任务 **`TASK_20261030_051`** 已进入成功终态 **5**：09:10:14 开始，09:14:05 结束，平台总耗时 **231 秒**（包含调度、安装、仿真启动与等待，不是纯评估耗时）。评估实际 commit 为 **`cef2d3cbcc2127035f05567a7f199695740b68cf`**，源 checkpoint SHA-256 与本轮最终模型一致；128 条回合记录均已核对，32 个 env 各 4 条，无遗漏/重复。worker 正常退出并打印 `FLUX_NATIVE_BOX_EVAL_SUCCESS`。[启动计划](artifacts/exp_2.1/native_isaaclab/launch_plan.json)、[云端实际计划](artifacts/exp_2.1/native_isaaclab/cloud_plan.json)、[最终验收](artifacts/exp_2.1/native_isaaclab/final_validation.json)、[脱敏完整日志 gzip](artifacts/exp_2.1/native_isaaclab/native_evaluation_log.txt.gz)。

| 原生实测指标 | 结果 |
| --- | --- |
| 完成首个完整参考片段 | **2/128 = 1.5625%**；另 126 回合未走完首个片段 |
| 平均 / 中位回合时长 | **0.88625 / 0.54 秒**，89/128 回合不足 1 秒 |
| 最短 / 最长回合 | 0.12 / 7.04 秒 |
| 片段通过回合的后续情况 | env 25 第 1 回合、env 13 第 1 回合完成首片段；原生 clip 自动复位后又分别于总长 6.82 / 7.04 秒跟踪失败，不能声称连续多轮搬运成功 |
| 复位前箱体位置 RMSE / 平均姿态误差 | 0.120684 m / 0.217229 rad；失败即复位，不能和 MuJoCo 无复位 6.5 秒整段 RMSE 直接比较 |
| 箱体最高中心 / 最低机器人根高度 | 0.848528 / 0.124029 m；最高箱体高度说明部分试次可抬起，不等于可靠完成搬运 |
| 实际执行量 | 462 个并行控制步，5,672 个有效状态采样；2500 步上限未触发 |

只针对 **126 个首片段未完成的回合**，终止原因如下。同一回合可同时命中多个原因，不能将次数相加计算失败回合数；35 个手脚高度失败内部也可能同时命中多个肢体。

| 原生终止条件 | 回合数 | 本轮阈值 / 解释 |
| --- | --- | --- |
| 箱体位置偏离 | **55** | 箱体世界位置距参考 >0.25 m |
| 箱体姿态偏离 | **35** | 姿态角误差 >0.8 rad |
| 手脚高度跟踪失败 | **35** | 脚踝/手腕 Z 误差 >0.25 m；左/右手腕各命中 15 次，左/右脚踝 5/1 次（内部重叠） |
| 机身参考姿态失败 | **2** | 使用原生 projected-gravity Z 差阈值 0.8，不是直接的四元数角阈值 |
| 机身参考高度 / timeout | **0 / 0** | 本轮未触发 |

其中 **90/126 = 71.43%** 的首片段失败涉及箱体位置或姿态。若统计全部 128 个回合结束事件（包含两次首片段通过后在第二片段失败的事件），对应计数是箱体位置 56、姿态 35、手脚高度 36、机身姿态 2；与上表的统计口径不同。[128 条逐回合数据](artifacts/exp_2.1/native_isaaclab/episodes.json)、[原生汇总](artifacts/exp_2.1/native_isaaclab/summary.json)、[首片段失败分析](artifacts/exp_2.1/native_isaaclab/outcome_analysis.json)。

实际协议与版本已归档于 [protocol.json](artifacts/exp_2.1/native_isaaclab/protocol.json)。新任务平台镜像为 `V000446`，与训练的 `V000420` 有版本差异；但训练和评估日志中的 `isaaclab` distribution 元数据均为 **0.54.2**，Torch 均为 **2.7.0+cu128**，GPU 探针均报告 NVIDIA GeForce RTX 4090。平台镜像标签仍为 IsaacLab 2.3.2 / IsaacSim 5.1，不能把 Python distribution 元数据直接当作该镜像标签。原生评估从片段起点开始，训练曲线包含自适应采样相位，0.886 秒与训练 2.56 秒不是同口径，不能由此判断训练退化。

**更新后的判断：✅ 原生 IsaacLab 评估流程通过；❌ 现有模型在本次保留随机化的起点评估中仍无法稳定完成搬运。主要观测到的失败模式是箱体与手部协同跟踪，MuJoCo 适配不能解释全部失败；箱体尺寸也未被证实为主要原因。** 这次终止统计定位了失败表现，尚不能单独证明是训练预算、奖励设计或初始化分布中的哪一项造成学习不足；后续应围绕抓握起始阶段做原生诊断，再决定如何续训。没有启动新训练、修改权重或录制 RTX 视频。


## 实验 exp_2.2：原生规模下恢复完整状态续训

### 1. 上一实验结果与教训

exp_2.1 在 1024 环境完成 8000 轮，最终 `model_07999.pt`；末段 100 轮平均奖励 6.877422、回合约 2.562904 秒。原生 IsaacLab 起点评估 128 回合仅 2 回合完成首个参考片段（1.5625%），126 次首片段失败中 90 次涉及箱体位置/姿态，手脚高度失败 35 次。训练最后两个千轮窗口的奖励仍从 6.459955 升至 6.767822，尚无稳定收敛证据。不能把失败全归因于 MuJoCo，也不能保证只增加预算即可解决。

旧模型包含完整 actor/critic、两套经验观测归一化和两个 AdamW 优化器，actor 优化器 step=160000，actor/critic 实际学习率均为 `3.375e-5`。因此用户明确选择续训，保留学习成果；此前讨论的从零方案未创建任务，不另占失败实验编号。

### 2. 本轮修改目标

- 按用户要求使用原生 4096 环境规模，从现有最终模型**额外训练 6000 轮**。
- 完整恢复优化器和观测归一化，正式编号从 8000 接到 13999，不重复旧的 7999。
- seed=42，保存间隔按用户最新要求从原生 4000 改为 **100**；保留任务/奖励/数据/PD/随机化，恢复的学习率和探索 std 沿用模型状态，不重新初始化。
- 4096 环境完整恢复容量已在前两次任务验证，本次明确复用容量结果，直接从原始 exp_2.1 模型正式续训，不重复首个 checkpoint 的下载校验。

### 3. 修改内容

### 修改四：独立完整状态续训入口与同规模容量门槛

新增 `scripts/resume_flux_box.py`，固定读取已授权推送的源 checkpoint 并校验 SHA-256。用原生预设构建配置，核对任务模块及 PPO 参数与源配置一致；处理保存元数据中 actor 输出维度已由符号解析成 29 的表示差异。只显式允许续训所需的轮数、保存间隔和 optimizer load 字段差异。

原生 `PPO.learn()` 按“当前 iteration 起，执行 num_learning_iterations 次”工作。包装器在 `PPO.load()` 后将“最后完成索引 7999”推进到“下一次索引 8000”，仅影响本次实例；再执行 6000 次得到最终 13999。显示用总轮数设为 14000，不改变 learn 的新增次数。原生 ETA 用绝对 iteration 计算，续训初期会偏低，监控应按新增轮数和本次耗时重算，不能直接沿用其 ETA。

恢复后逐项比较 actor、critic、两套 normalizer、两个 optimizer 的完整 state_dict，任何缺失/不一致立即失败。容量测试从源模型更新 2 次至 8001、校验 checkpoint/ONNX/TensorBoard 并退出释放 GPU；正式阶段从相同源重新恢复，容量测试权重不进入正式训练，失败不自动重试、不降低环境数。

源 checkpoint 没有 `env_state`，因此旧仿真状态、RNG 与自适应采样器历史未恢复；扩大环境数后按 seed=42 重新建立。这里完整恢复指 checkpoint 中已有的网络、优化器与归一化状态，不代表逐帧续接原来的仿真轨迹。

### 修改五：每 100 轮保存，复用已经验证的容量

新增 `--save-interval`（默认 100）传入正式 worker 配置；预检、两个阶段及正式训练之间显式传递保存间隔。新增 `--skip-capacity-test`，默认仍保留容量门槛，仅在本次同规模重提时明确启用。保留源模型哈希、任务身份与内存中加载状态一致性检查，以及训练结束时的最终产物验收；按用户指示不再下载/校验首个新 checkpoint，不重复两轮容量测试。源仍为 `model_07999.pt`，未改为中间模型。相关配置与控制流共 8 项 CPU 测试、Ruff 通过。

### 4. 修改文件

- [resume_flux_box.py](../../scripts/resume_flux_box.py)：完整恢复、编号衔接、4096 容量门槛及正式续训。
- [train_flux_box.py](../../scripts/train_flux_box.py)：任务身份校验增加显式 `allow_resume` 参数，原从零入口默认限制不变。
- [test_flux_box_resume.py](../../tests/test_flux_box_resume.py)：真实配置、优化器恢复/编号、容量失败门槛。
- [isaacsim.py](../../src/holosoma/holosoma/simulator/isaacsim/isaacsim.py)：用户要求删除 `write_state_updates()` 每步输出的 3 条 DEBUG 日志，仿真同步行为不变。
- 本笔记与 Flux 参考文档：记录续训语义、账号和实际验证范围。

### 5. 训练参数

| 参数 | 值 |
| --- | --- |
| 账号 / 用户 / 项目 | `limxmts0f9p800yu7g@uberip.com` / 4391 / `PRO_20261030_003` |
| 镜像 / 资源 | `BJX00000178` / `V000446`；`ESKU000001` / `SKUSL000002`，单 GPU 24G |
| 源模型 | `exp_2.1/model_07999.pt`，iter=7999，源任务 `TASK_20261029_267` |
| 源 SHA-256 | `e2d120660bbc199d0dc03d0bce17a4989ef6117877c2f97f9da72d2c4dbc2c97` |
| 原生预设 | `g1-29dof-wbt-w-object`；同一自由箱体、手部碰撞模型和 325 帧 motion |
| 环境数 / 新增轮数 | **4096 / 6000**，正式全局 iter 8000..13999 |
| rollout / epochs / minibatches | 24 / 5 / 4；每 PPO 轮 98,304 个控制样本 |
| 正式新增样本 | 589,824,000 个控制步；加旧模型累计约 786,432,000，容量测试不计入 |
| 经验归一化 / optimizer | 两套归一化和两个 AdamW 完整恢复；初始实际 LR `3.375e-5`，adaptive 继续调整 |
| seed / 保存间隔 | 原生 42 / 用户指定 **100**；保存 8000、8100…13900，结束另存 13999；前两次提交为 4000 |
| 本次容量 / 首模型重复验收 | `--skip-capacity-test`；复用已验证的同规模配置，不重复下载校验首个 checkpoint |
| 安装 / 容量 / 正式上限 | 900 / 900 / **86400 秒**；扩大规模后不受旧 6 小时包装器上限截断，完成指定轮数即退出 |
| 记录 / 视频 | TensorBoard 开启；Wandb、RTX 视频关闭 |
| 平台恢复方式 | 新任务 `trainType=1`，Python worker 执行原生完整 `PPO.load`；模型来自已授权仓库，不跨账号访问私有旧任务或挂载 checkpoint |
| 启动入口 | `gm-run holosama/scripts/resume_flux_box.py --experiment exp_2.2 --additional-iterations 6000 --save-interval 100 --skip-capacity-test` |

### 6. 预期与验收

前两次提交已验收 4096 环境容量、完整恢复与首个正式模型。本次只核对账号归属、实际 commit/源哈希、save_interval=100 和正式恢复标记，不重复容量或首模型下载校验。正式阶段从原始源恢复，起始 optimizer step=160000、iter=8000。最终 model_13999.pt 应完成额外 6000 轮，checkpoint、ONNX、TensorBoard 验收通过。最终策略仍按同一原生 32×4 起点评估统计成功率及终止原因，不以训练完成替代搬运达标。

### 7. 实验结果

账号 4391、兼容镜像 V000446 与单 GPU 资源已重新查询，训练项目 `PRO_20261030_003` 已创建。真实源 checkpoint 构建的计划配置与任务模块核对通过：[计划配置](artifacts/exp_2.2/planned_config.json)、[平台预检查](artifacts/exp_2.2/platform_preflight.json)。3 项续训测试与 4 项原从零保护测试共 **7 项通过**，Ruff 通过。

#### 第一次提交：已进入正式阶段，用户主动停止

`TASK_20261030_061` 使用 commit `168aa17e9f9d469dfd757cc6bb9921b4971b7f2b`，09:46:45 开始，09:54:32 停止，平台耗时 467 秒。先前脱敏日志尾部已读到正式 iter **8009/14000**，说明同规模容量门槛后已进入正式阶段；该尾部未包含容量/恢复验收标记，不能冒充完整验收报告。用户发现 `write_state_updates()` 每步刷出状态同步 DEBUG 日志，明确要求删除，并自行停止云端任务。平台状态 6 在此次表示主动停止，不作为训练代码崩溃的证据。[停止状态及归属](artifacts/exp_2.2/previous_task_stopped.json)、[首次进度](artifacts/exp_2.2/attempt_TASK_20261030_061/progress_snapshot.json)、[首次云端计划](artifacts/exp_2.2/attempt_TASK_20261030_061/cloud_launch_plan.json)。

#### 清理日志后按用户授权重新提交（同一 exp_2.2）

已推送 commit **`681c06ec487f1917b32bc9399026edc353c6979c`**，仅删除每步同步前、同步后及无变化时的 3 条 DEBUG，Python 语法与 diff 检查通过；不改变同步调用、dirty flag、PPO 或训练参数。重新查询旧任务终态后，在相同账号/项目创建唯一的新任务 **`TASK_20261030_065`**，09:57:47 开始；重提之前已确认 10:11:10 结束、终态 6，总耗时 803 秒。worker 启动计划已确认实际 commit 为 `681c06e`，源哈希、4096 环境及新增 6000 轮与上表一致。[本次启动记录](artifacts/exp_2.2/attempt_TASK_20261030_065/launch_plan.json)、[实际云端计划](artifacts/exp_2.2/attempt_TASK_20261030_065/cloud_launch_plan.json)、[最新进度](artifacts/exp_2.2/attempt_TASK_20261030_065/progress_snapshot.json)。

本次重新从原始 `model_07999.pt` 恢复，目标仍是 8000..13999；未从第一次提交的中间模型接续，不将其少量更新计入本次 6000 轮。仍执行独立的 4096 环境 × 2 轮容量门槛，无自动重试。

**本次开训验收已通过**：容量阶段和正式阶段的完整恢复标记均已捕获，两个阶段均从 source iter=7999 / optimizer step=160000 开始，实际学习率 `3.3750000000000014e-5`。容量 checkpoint `model_08001.pt` 已上传，正式阶段进入 iter **8003/14000**。容量成功标记本身已不在 API 截断尾部，但包装器必须验收容量 checkpoint/ONNX/TensorBoard 后才会启动正式阶段，正式恢复与正式模型共同证明容量门槛已通过；不补造缺失的原始标记。[容量恢复](artifacts/exp_2.2/attempt_TASK_20261030_065/capacity_restored.json)、[正式恢复](artifacts/exp_2.2/attempt_TASK_20261030_065/formal_restored.json)、[模型列表安全字段](artifacts/exp_2.2/attempt_TASK_20261030_065/model_list.json)。

首个正式 `model_08000.pt`（不是容量阶段的同名文件）已下载至忽略的验证缓存，6,840,071 字节，SHA-256=`c17d1ac9a122ffa2c94d128af5ce2c0da30128ee811844c4a6609c71341aac6d`。其保存配置为 4096 环境 / 新增 6000 轮 / load_optimizer=true；actor、critic 两个 AdamW 的 step 均为 **160020**，两套 normalizer count 均从 **196608000** 增至 **196706304**（增加 4096×24），网络与归一化张量均有限，验证了原模型完整状态延续。[首次正式模型验收](artifacts/exp_2.2/attempt_TASK_20261030_065/first_formal_checkpoint_validation.json)。本次已捕获的日志中未再出现删除的 3 条同步 DEBUG。[脱敏日志尾部 gzip](artifacts/exp_2.2/attempt_TASK_20261030_065/startup_excerpt.txt.gz)。最终 6000 轮完成情况及搬运质量仍待训练结束后验收。


#### 按用户要求改为每 100 轮保存并重提

已确认前次任务 `TASK_20261030_065` 停止，仍使用账号 4391、4096 环境、源 `model_07999.pt`，新增 6000 轮至 13999。本次显式设置 `--save-interval 100 --skip-capacity-test`，复用上述容量验证，不重复首个 checkpoint 下载校验。8 项本地测试与 Ruff 通过。新任务 **`TASK_20261030_073`** 已在同一账号/项目启动，10:18:21 开始、平台状态 3，提交版本 `f6566f80dff2c3f8d1549464a2bd0ea38c32cbca`；实际 worker 计划已核对为相同 commit、4096 环境、新增 6000 轮、save_interval=100、skip_capacity_test=true、capacity_iterations=0；正式恢复与迭代进度以本次日志为准。[前次终态](artifacts/exp_2.2/save100_previous_task_stop.json)。

[本次启动记录](artifacts/exp_2.2/launch_plan.json)、[本地参数验收](artifacts/exp_2.2/save100_local_validation.json)、[本次进度](artifacts/exp_2.2/progress_snapshot.json)。未重复下载校验首个新模型，未读取前次任务的私有模型来替换原始续训源。


#### 续训中期检查（2026-10-30 13:21，仍在训练）

任务 `TASK_20261030_073` 仍为状态 3，账号用户 4391，最新日志 iter **10435/14000**，本次新增 **2436/6000（40.60%）**。模型列表已出现 `model_10400.pt`，读取的仅是模型元数据，未下载或重复校验 checkpoint。当前日志未发现 traceback。最近约 4.18 秒/轮，按本次剩余轮数重算预计 17:30 左右完成，最终验收/上传及速度变化另计。

本次使用平台 `precise` 曲线，查询截止 13:20:18；各指标均包含 iter 8000..10417 的 2418 个实测点。比较本次最初 500 轮（8000..8499）与最近 500 轮（9918..10417）：

| 指标 | 最初 500 轮均值 | 最近 500 轮均值 | 变化 |
| --- | --- | --- | --- |
| Mean reward | 6.172041 | 9.563667 | +54.95% |
| 平均回合 steps / 秒（50 Hz） | 115.221400 / 2.304428 | 181.520380 / 3.630408 | +57.54% |
| 箱体位置跟踪奖励 | 0.200503 | 0.317276 | +58.24% |
| 箱体姿态跟踪奖励 | 0.184125 | 0.296041 | +60.78% |

连续完整的 500 轮窗口奖励均值为 6.172041 → 7.899924 → 8.584123 → 9.129868；当前未满的 10000..10417 窗口为 9.581582。说明提升不只来自某个单轮峰值。最近 100 轮（10318..10417）奖励 9.610668、episode 183.107900 steps（3.662158 秒），相较 exp_2.1 最后 100 轮的 6.877422 / 128.145200，分别高 39.74% / 42.89%；跨任务对照改变了环境数、seed 和自适应相位分布，不属于成对策略评估。

**中期结论：训练奖励与平均持续跟踪时长有持续提升，尚不能判定已稳定完成整段搬运。** 箱体奖励升高也不能单独解释为同一相位下位置/姿态误差按同比下降。身体位置误差在本次最初/最近 500 轮均值约 0.068153 / 0.067293，而 joint-position error 约 1.532556 / 1.543990，动作精度没有全面改善；指标还受自适应相位采样和失败复位影响。最终仍需相同原生 32×4 起点评估核对完整片段成功率及终止原因，不将平均 episode 或训练完成当作成功率。

[实测曲线与统计](artifacts/exp_2.2/progress_20261030_132018/trend_summary.json)、[曲线原始数据 gzip](artifacts/exp_2.2/progress_20261030_132018/curves.json.gz)、[旧模型对照与当前 ETA](artifacts/exp_2.2/progress_20261030_132018/comparison.json)、[模型元数据](artifacts/exp_2.2/progress_20261030_132018/model_list.json)、[当前状态](artifacts/exp_2.2/progress_20261030_132018/current_status.json)。


#### 续训后段检查（2026-10-30 15:46，仍在训练）

任务 `TASK_20261030_073` 仍为状态 3、账号用户 4391；精细曲线最新 iter **12469/14000**，本次新增 **4470/6000（74.5%）**。最新模型元数据为 `model_12400.pt`，未下载或重复校验 checkpoint。当前日志未发现 traceback；最近约 4.25 秒/轮，估计剩余 1 小时 48 分钟，完成时间约 17:35，最终验收/上传及速度变化另计。

本次 `precise` 曲线覆盖 iter 8000..12469，共 4470 个实测点。对比上次检查的最近 500 轮（9918..10417）与当前最近 500 轮（11970..12469）：

| 指标 | 上次 500 轮均值 | 当前 500 轮均值 | 变化 |
| --- | --- | --- | --- |
| Mean reward | 9.563667 | 11.085848 | +15.92% |
| 平均回合 steps / 秒（50 Hz） | 181.520380 / 3.630408 | 208.904820 / 4.178096 | +15.09% |
| 箱体位置跟踪奖励 | 0.317276 | 0.365579 | +15.23% |
| 箱体姿态跟踪奖励 | 0.296041 | 0.345156 | +16.59% |

最近几个窗口的奖励均值继续从 10.488042（11000..11499）升至 10.890642（11500..11999），当前未满的 12000..12469 为 11.092596。身体位置误差从上次 0.067293 降至 0.064159，约下降 4.66%；joint-position error 从 1.543990 增至 1.562816，约升高 1.22%。**结论：训练指标继续改善，增幅较初期放缓；动作精度尚未全面改善，不能由训练奖励或回合时长判定完整搬运成功率。** 仍应在最终训练后按相同原生起点评估核对完整片段通过率与终止原因。

[本次曲线与统计](artifacts/exp_2.2/progress_20261030_154644/trend_summary.json)、[与上次检查对照](artifacts/exp_2.2/progress_20261030_154644/comparison_previous_check.json)、[原始曲线 gzip](artifacts/exp_2.2/progress_20261030_154644/curves.json.gz)、[模型元数据](artifacts/exp_2.2/progress_20261030_154644/model_list.json)。


#### 续训后段检查（2026-10-30 16:25，仍在训练）

任务 `TASK_20261030_073` 仍为状态 3、账号用户 4391；最新精细曲线 iter **13010/14000**，本次新增 **5011/6000（83.52%）**，余 989 次更新。最新模型元数据 `model_13000.pt` 已出现，未下载或重复校验 checkpoint。当前日志未发现 traceback。最近约 4.31 秒/轮，估计剩余约 71 分钟，完成时间约 17:36，最终验收/上传及速度波动另计。

比较当前最近 500 轮（12511..13010）与上次检查最近 500 轮（11970..12469）：

| 指标 | 上次均值 | 当前均值 | 变化 |
| --- | --- | --- | --- |
| Mean reward | 11.085848 | 11.468039 | +3.45% |
| 平均回合 steps / 秒（50 Hz） | 208.904820 / 4.178096 | 215.426000 / 4.308520 | +3.12% |
| 箱体位置跟踪奖励 | 0.365579 | 0.375676 | +2.76% |
| 箱体姿态跟踪奖励 | 0.345156 | 0.354487 | +2.70% |
| 身体位置误差 | 0.064159 | 0.062987 | 下降 1.83% |
| joint-position error | 1.562816 | 1.523227 | 下降 2.53% |

**结论：继续小幅提升，奖励与回合时长增速较前段放缓；本次身体位置与关节误差均有所下降。** 当前 13000..13010 的新窗口仅 11 点，不能拿它与完整 500 轮均值比较后宣称退化或收敛；仍以完整窗口统计为主。上述均为训练采样统计，不能代替完整片段搬运成功率，最终仍需同协议原生评估。

[本次统计](artifacts/exp_2.2/progress_20261030_162524/trend_summary.json)、[与前次检查对照](artifacts/exp_2.2/progress_20261030_162524/comparison_previous_check.json)、[原始精细曲线 gzip](artifacts/exp_2.2/progress_20261030_162524/curves.json.gz)、[模型元数据](artifacts/exp_2.2/progress_20261030_162524/model_list.json)。


#### 训练完成与最终模型（2026-10-30 17:37）

`TASK_20261030_073` 已进入成功终态 **5**，10:18:21–17:37:08，平台耗时 **26327 秒（7小时18分47秒）**，包含安装、初始化、学习、最终验收及上传等待。实际训练 commit `f6566f80dff2c3f8d1549464a2bd0ea38c32cbca`；完整完成新增 6000 次更新至 iter=13999，4096 环境、每100轮保存。云端最终 checkpoint、ONNX CPU 推理 `[1,29]` 有限和 TensorBoard 验收通过，ONNX 未下载。

最终模型 [model_13999.pt](../data/exp_2.2/model_13999.pt) 已下载，6,840,071 字节，SHA-256=`7546092cda21371a8c37a6fac3eb92952a560b9e0bf2484c256bec3a92d05120`；actor/critic/两套归一化有限。两个 AdamW step 均为 **280000**，两套归一化 count 均为 **786432000**，符合源状态加6000次更新。按用户要求未重复首个checkpoint下载验收，本次下载的是最终模型。[最终模型校验](artifacts/exp_2.2/final_checkpoint_validation.json)、[云端最终验收](artifacts/exp_2.2/final_cloud_validation.json)、[实际最终配置](artifacts/exp_2.2/actual_final_config.json)。

| 最后100轮（13900..13999）指标 | 实测 |
| --- | --- |
| Mean reward | 12.079987 |
| Mean episode length | 227.007700 steps / 4.540154 秒 |
| 箱体位置 / 姿态跟踪奖励 | 0.395234 / 0.373823 |
| FPS均值 | 22991.36 |

[最终精细指标](artifacts/exp_2.2/final_metrics.json)、[完整精细曲线 gzip](artifacts/exp_2.2/progress_20261030_174602/curves.json.gz)。**训练流程通过，完整搬运质量仍以回放和原生起点评估为准。**

#### 按用户顺序先完成本地回放

使用最终模型、修复后的根关节速度初始化和保存的154维观测/归一化，单个325帧参考片段（6.5秒、50Hz、720p），自由箱体名义质量2.6kg，初始写状态后真实MuJoCo物理推进，无自动复位或后续位姿纠正。由于最终模型来自续训，本地任务身份检查显式允许resume；未改变物理或策略。

视频 [play_box_mujoco.mp4](../data/exp_2.2/play_box_mujoco.mp4) 已生成并完整解码325帧，代表帧0.5/2/4/6秒人工检查可见抬箱、移动与末段放下，机器人最低根高度0.546773m，未出现旧模型约2秒即跌倒的现象。但箱体姿态最大误差1.298346rad，1.78秒首次超过训练跟踪阈值；箱体位置最大误差0.221097m、全段RMSE0.110489m，最高中心0.842888m。**本机跨仿真结果明显改善，姿态跟踪仍未完全达标；此单次名义条件回放不代表原生随机化成功率。** [本地轨迹与数值](artifacts/exp_2.2/local_box_replay_validation.json)、[视频解码](artifacts/exp_2.2/local_video_decode.json)、[代表帧](artifacts/exp_2.2/local_box_representative_frames.jpg)、[人工检查](artifacts/exp_2.2/local_video_human_review.json)。

#### 随后准备线上原生回放

按用户授权，在同一账号4391、评估项目 `PRO_20261030_002` 下，计划以最终模型先做与旧模型相同32环境×4回合的原生起点评估，再做1环境、30秒720p原生录像。视频保留10秒timeout、bad_tracking和clip自动复位，录像时长不当作连续搬运成功率。固定模型digest并从授权仓库读取，无训练、不自动重试、不切换账号。

新增 `scripts/replay_flux_box.py` 复用bootstrap，更新原生评估worker允许续训模型并显式核对13999/4096身份。修复上次HydraTexture清理错误：RGB annotator使用render product的字符串path挂接/卸载，随后销毁持有的texture；依据[NVIDIA Replicator接口](https://docs.omniverse.nvidia.com/kit/docs/omni_replicator/1.12.27/source/extensions/omni.replicator.core/docs/API.html)及[返回值变更记录](https://docs.omniverse.nvidia.com/kit/docs/omni_replicator/1.12.27/source/extensions/omni.replicator.core/docs/CHANGELOG.html)。相关CPU检查共12项通过（包括新模型录像配置与HydraTexture清理回归）；本机没有IsaacSim，实际GPU录像效果须由线上任务验收。
