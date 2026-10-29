<a id="holosoma-training-framework"></a>

# Holosoma 训练框架

面向人形机器人强化学习的核心训练框架，支持运动控制（速度跟踪）和全身动作跟踪任务。

| **类别** | **支持选项** |
|-------------|----------------------|
| **仿真器** | IsaacSim、MJWarp（训练）\| MuJoCo（评估） |
| **算法** | PPO、FastSAC |
| **机器人** | Unitree G1、Booster T1 |

<a id="training"></a>

## 训练

纯训练安装使用 `python -m pip install -e ./src/holosoma`。交互式分析工具 Open3D、Notebook 已拆为可选依赖：需要时使用 `python -m pip install -e './src/holosoma[analysis]'`；静态检查工具和类型 stub 使用 `[dev]`。无界面云端训练不安装这些 extras，避免冷启动下载大型工具包。IsaacSim 环境仍需遵守 `scripts/isaacsim_constraints.txt`；Flux 安装入口使用包含该文件的 `scripts/flux_smoke_constraints.txt`。

所有训练和评估脚本均支持通过 `--help` 查看可用参数，例如 `python src/holosoma/holosoma/train_agent.py --help`。

> **提示：** 使用 `logger:wandb` 时，默认启用视频录制。在无显示界面的服务器上，可能需要关闭视频录制或配置渲染环境。详见下文的[视频录制](#video-recording)。

<a id="locomotion-velocity-tracking"></a>

### 运动控制（速度跟踪）

训练机器人跟踪速度指令。

```bash
# 在 IsaacSim 中使用 FastSAC 训练 G1
source scripts/source_isaacsim_setup.sh
python src/holosoma/holosoma/train_agent.py \
    exp:g1-29dof-fast-sac \
    simulator:isaacsim \
    logger:wandb \
    --training.seed 1

# 在 IsaacSim 中使用 PPO 训练 T1
source scripts/source_isaacsim_setup.sh
python src/holosoma/holosoma/train_agent.py \
    exp:t1-29dof \
    simulator:isaacsim \
    logger:wandb \
    --training.seed 1
```

保存检查点后，可以使用[训练配置下的评估](#in-training-evaluation)（与训练使用相同的仿真器），也可以在 MuJoCo 中进行跨仿真器评估（参阅 [holosoma_inference 文档](../holosoma_inference/README.md)）。

<a id="mjwarp-training-for-locomotion-velocity-tracking"></a>

### 使用 MJWarp 训练运动控制（速度跟踪）

使用 MJWarp 仿真器（GPU 加速的 MuJoCo）进行训练。**注意：MJWarp 支持目前处于测试阶段。**

```bash
# 使用 FastSAC 训练 G1
source scripts/source_mujoco_setup.sh
python src/holosoma/holosoma/train_agent.py \
    exp:g1-29dof-fast-sac \
    simulator:mjwarp \
    logger:wandb

# 使用 PPO 训练 G1
source scripts/source_mujoco_setup.sh
python src/holosoma/holosoma/train_agent.py \
    exp:g1-29dof \
    simulator:mjwarp \
    logger:wandb

# 使用 FastSAC 训练 T1
source scripts/source_mujoco_setup.sh
python src/holosoma/holosoma/train_agent.py \
    exp:t1-29dof-fast-sac \
    simulator:mjwarp \
    logger:wandb

# 使用 PPO 训练 T1
source scripts/source_mujoco_setup.sh
python src/holosoma/holosoma/train_agent.py \
    exp:t1-29dof \
    simulator:mjwarp \
    logger:wandb \
    --terrain.terrain-term.scale-factor=0.5  # 必须设置此参数，以避免训练不稳定

```

> **提示：**
>
> - MJWarp 默认使用 `nconmax=96`（每个环境的最大接触数量）。如有需要，可通过 `--simulator.config.mujoco-warp.nconmax-per-env=96` 调整。
> - 这些示例使用 `--training.num-envs=4096`，实际使用时可能需要根据硬件配置调整此值。
> - 在混合地形上使用 PPO 训练 T1 时，请设置 `--terrain.terrain-term.scale-factor=0.5`，以避免训练不稳定。


<a id="whole-body-tracking"></a>

### 全身动作跟踪

训练机器人跟踪全身动作序列。

**注意：** 目前仅支持 Unitree G1 与 IsaacSim 的组合。

```bash
# 使用 FastSAC 训练 G1
source scripts/source_isaacsim_setup.sh
python src/holosoma/holosoma/train_agent.py \
    exp:g1-29dof-wbt-fast-sac \
    logger:wandb

# 使用 PPO 训练 G1
source scripts/source_isaacsim_setup.sh
python src/holosoma/holosoma/train_agent.py \
    exp:g1-29dof-wbt \
    logger:wandb

# 使用自定义动作文件
source scripts/source_isaacsim_setup.sh
python src/holosoma/holosoma/train_agent.py \
    exp:g1-29dof-wbt \
    logger:wandb \
    --command.setup_terms.motion_command.params.motion_config.motion_file="holosoma/data/motions/g1_29dof/whole_body_tracking/<your file>.npz"

# 训练前在 IsaacSim 中可视化回放动作文件
source scripts/source_isaacsim_setup.sh
python src/holosoma/holosoma/replay.py \
    exp:g1-29dof-wbt \
    --training.headless=False \
    --training.num_envs=1
```

保存检查点后，可以使用[训练配置下的评估](#in-training-evaluation)（与训练使用相同的仿真器），也可以在 MuJoCo 中进行跨仿真器评估（参阅 [holosoma_inference 文档](../holosoma_inference/README.md)）。

---

<a id="evaluation"></a>

## 评估

<a id="in-training-evaluation"></a>

### 训练配置下的评估

使用与训练时完全相同的配置（包括仿真器、环境设置等）评估策略：

```bash
# 评估 Wandb 上的检查点
python src/holosoma/holosoma/eval_agent.py \
    --checkpoint=wandb://<ENTITY>/<PROJECT>/<RUN_ID>/<CHECKPOINT_NAME>
# 例如：--checkpoint=wandb://username/fastsac-t1-locomotion/abcdefgh/model_0010000.pt

# 评估本地检查点
python src/holosoma/holosoma/eval_agent.py \
    --checkpoint=<CHECKPOINT_PATH>
# 例如：--checkpoint=/home/username/checkpoints/fastsac-t1-locomotion/model_0010000.pt
```

此评估模式支持：

- 自动从检查点加载训练配置。
- 在与训练相同的仿真器和环境中运行评估。
- 将策略导出为 ONNX 格式（通过 `--training.export_onnx=True`）。
- 在运动控制评估中，通过键盘交互式输入速度指令（需使仿真器窗口处于活动状态）：`w`/`a`/`s`/`d` 输入线速度指令，`q`/`e` 输入角速度指令，`z` 将速度指令归零。

<a id="cross-simulator-evaluation-mujoco"></a>

### 跨仿真器评估（MuJoCo）

如需在 MuJoCo 仿真中测试训练好的策略，或将策略部署到真实机器人，请参阅 [holosoma_inference 文档](../holosoma_inference/README.md)，其中包括：

- 跨仿真器评估（IsaacSim/MJWarp → MuJoCo）。
- 真实机器人部署（运动控制和全身动作跟踪 WBT）。

**提示：** 训练过程中通常会在保存 `.pt` 检查点的同时导出 ONNX 策略，也可以使用上述训练配置评估脚本生成 ONNX 策略。

<a id="advanced-configuration"></a>

## 高级配置

训练系统采用分层配置。`exp` 配置是主要入口，提供针对各类算法和机器人调优后的默认配置。可以通过命令行覆盖参数，自定义训练过程。

> **提示：** 组合 Tyro 配置时，请将 `exp:<name>` 预设放在其他配置片段（例如 `logger:wandb`）之前。Tyro 要求先声明基础实验，顺序颠倒可能导致难以理解的配置解析错误。

<a id="logging-with-weights--biases"></a>

### 使用 Weights & Biases 记录日志

```bash
source scripts/source_isaacsim_setup.sh
python src/holosoma/holosoma/train_agent.py \
    exp:g1-29dof \
    simulator:isaacsim \
    --training.seed 1 \
    --algo.config.use-symmetry=False \
    logger:wandb \
    --logger.project locomotion-g1-29dof-ppo \
    --logger.name ppo-without-symmetry-seed1
```

<a id="video-recording"></a>

### 视频录制

使用 `logger:wandb` 时，**默认启用视频录制**。系统会定期录制视频，并上传到 Weights & Biases。

**配置方式：**

```bash
# 关闭视频录制
--logger.video.enabled False

# 调整录制间隔（以回合为单位）
--logger.video.interval 10

# 修改分辨率
--logger.video.width 640 --logger.video.height 360
```

**无显示界面环境的问题排查：**

如果在无显示界面的服务器上训练时，出现显示或渲染错误（例如 `GLXBadFBConfig`、`eglInitialize failed`、`GLFW initialization failed`），可采用以下处理方式：

- **IsaacSim：** 使用 `--logger.video.enabled False` 关闭视频录制，或通过 `DISPLAY= python ...` 强制使用 EGL，也可以通过 `xvfb-run -a python ...` 使用虚拟显示器。
- **MJWarp/MuJoCo：** 训练前设置环境变量 `export MUJOCO_GL=egl`。参阅 [MuJoCo 文档](https://mujoco.readthedocs.io/en/stable/programming/index.html#using-opengl)。

<a id="terrain"></a>

### 地形

```bash
# 使用平面地形替代混合地形
source scripts/source_isaacsim_setup.sh
python src/holosoma/holosoma/train_agent.py \
    exp:g1-29dof-fast-sac \
    simulator:isaacsim \
    terrain:terrain-locomotion-plane
```

<a id="multi-gpu-training"></a>

### 多 GPU 训练

```bash
source scripts/source_isaacsim_setup.sh
torchrun --nproc_per_node=4 src/holosoma/holosoma/train_agent.py \
    exp:t1-29dof-fast-sac \
    simulator:isaacsim \
    --training.num-envs 16384  # 所有 GPU 合计的环境总数
```

<a id="custom-reward-weights"></a>

### 自定义奖励权重

```bash
source scripts/source_isaacsim_setup.sh
python src/holosoma/holosoma/train_agent.py \
    exp:g1-29dof-fast-sac \
    simulator:isaacsim \
    --reward.terms.tracking-lin-vel.weight=2.5 \
    --reward.terms.feet-phase.params.swing-height=0.12
```

<a id="observation-noise"></a>

### 观测噪声

```bash
# 关闭观测噪声
source scripts/source_isaacsim_setup.sh
python src/holosoma/holosoma/train_agent.py \
    exp:g1-29dof-fast-sac \
    simulator:isaacsim \
    --observation.groups.actor-obs.enable-noise=False
```

<a id="observation-history-length"></a>

### 观测历史长度

部分策略可以通过堆叠多个时间步的观测获得更好的效果。可以通过以下方式增加训练时使用的观测历史长度：

```bash
source scripts/source_isaacsim_setup.sh
python src/holosoma/holosoma/train_agent.py \
    exp:g1-29dof-fast-sac \
    simulator:isaacsim \
    --observation.groups.actor_obs.history-length 4
```

运行推理时，请传入与训练相同的历史长度，确保导出的 ONNX 策略接收到形状正确的输入。

<a id="curriculum-learning"></a>

### 课程学习

```bash
# 关闭课程学习
source scripts/source_isaacsim_setup.sh
python src/holosoma/holosoma/train_agent.py \
    exp:g1-29dof-fast-sac \
    simulator:isaacsim \
    --curriculum.setup-terms.penalty-curriculum.params.enabled=False

# 自定义课程学习阈值（适用于较短的回合）
source scripts/source_isaacsim_setup.sh
python src/holosoma/holosoma/train_agent.py \
    exp:g1-29dof-fast-sac \
    simulator:isaacsim \
    --simulator.config.sim.max-episode-length-s=10.0 \
    --curriculum.setup-terms.penalty-curriculum.params.level-up-threshold=350
```

<a id="domain-randomization"></a>

### 域随机化

```bash
source scripts/source_isaacsim_setup.sh
python src/holosoma/holosoma/train_agent.py \
    exp:g1-29dof-fast-sac \
    simulator:isaacsim \
    --randomization.setup-terms.push-randomizer-state.params.enabled=False \
    --randomization.setup-terms.randomize-base-com-startup.params.enabled=True \
    --randomization.setup-terms.mass-randomizer.params.added-mass-range=[-1.0,3.0]
```
