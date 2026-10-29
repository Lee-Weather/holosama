# Holosoma 训练实验笔记

按 [lab-notebook](../skills/lab-notebook/SKILL.md) 维护索引和七节实验结构。本文路径沿用统一笔记约定；首次流程验证按用户要求编号为 `exp0`，下一项新目标默认编号 `exp_1`。时间除附件 JSON 的 UTC 查询时间外，均为北京时间（UTC+8）。

## 实验索引

| 编号 | 日期 | 摘要 | 状态 | Task ID | Flux账号 | checkpoint |
| --- | --- | --- | --- | --- | --- | --- |
| [exp0](#实验-exp0flux-isaaclab-上的-g1-ppo-流程验证) | 2026-10-28 | 修复安装源与版本回溯，完成 128 环境 × 2 次 PPO 迭代；checkpoint 上传/下载及云端 ONNX 校验通过 | 已测试 | `TASK_20261028_224`（含两次失败尝试，见 §7） | `limxmts0elyormeym6@uberip.com` | [model_00001.pt](../data/exp0/model_00001.pt) |
| [exp_1](#实验-exp_1g1-平地-ppo-正式训练) | 2026-10-28 | 安装阶段下载 Open3D 时超过 900 秒，未进入 GPU 训练；2026-10-29 完成终止诊断 | 失败 | `TASK_20261028_251` | `limxmts0elyormeym6@uberip.com` | 无；未开始训练 |

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
