<a id="sim-to-sim-whole-body-tracking-workflow"></a>

# 跨仿真器全身动作跟踪工作流程

> **另请参阅：** [推理与部署指南](../../README.md)，了解所有部署选项。

本指南介绍在 MuJoCo 仿真中运行全身动作跟踪（WBT）策略的完整流程。

<a id="overview"></a>

## 概述

跨仿真器工作流程可在 MuJoCo 中运行 IsaacSim 训练的 WBT 检查点，以便评估和测试。

<a id="prerequisites"></a>

## 前提条件

- 已配置 MuJoCo 环境（`scripts/source_mujoco_setup.sh`）
- 已配置 Holosoma 推理环境（`scripts/source_inference_setup.sh`）
- ONNX 模型检查点
- 用于控制的键盘

**注意：** 推理程序和 MuJoCo 在同一台机器运行时，始终使用 `--task.interface lo`（回环接口）。

---

<a id="unitree-g1-29-dof"></a>

## Unitree G1（29 自由度）

<a id="1-start-mujoco-environment"></a>

### 1. 启动 MuJoCo 环境

在一个终端中启动 MuJoCo 环境：

```bash
source scripts/source_mujoco_setup.sh
python src/holosoma/holosoma/run_sim.py robot:g1-29dof
```

机器人将在仿真器中生成，并吊挂在保护架上。

<a id="2-launch-the-policy"></a>

### 2. 启动策略程序

在另一个终端中运行策略推理：

```bash
source scripts/source_inference_setup.sh
python3 src/holosoma_inference/holosoma_inference/run_policy.py inference:g1-29dof-wbt \
    --task.model-path src/holosoma_inference/holosoma_inference/models/wbt/fastsac_g1_29dof_dancing.onnx \
    --task.no-use-joystick \
    --task.use-sim-time \
    --task.rl-rate 50 \
    --task.interface lo
```

<a id="3-initialize-stiff-control-mode"></a>

### 3. 初始化刚性控制模式

在策略终端出现提示时按 `Enter`。机器人进入刚性控制模式并保持初始姿态。

<a id="4-deploy-the-robot"></a>

### 4. 让机器人接触地面

- 在 MuJoCo 窗口中按 `8` 降低吊架，直到机器人接触地面。
- 在 MuJoCo 窗口中按 `9` 移除吊架。
- 等待数秒，让刚性控制器稳定机器人。

<a id="5-start-the-policy"></a>

### 5. 启用策略

在策略终端中按 `]` 启用策略。

<a id="6-start-motion-clip"></a>

### 6. 开始动作片段

在策略终端中按 `m` 开始动作片段，机器人将开始跟踪全身动作。

---

<a id="mujoco-controls-reference"></a>

## MuJoCo 控制参考

**在 MuJoCo 窗口中输入这些命令**（而不是策略终端）：

<a id="gantry-controls"></a>

### 吊架控制

- `7`：升高吊架。
- `8`：降低吊架。
- `9`：禁用或移除吊架。

<a id="general-controls"></a>

### 通用控制

- `Backspace`：重置仿真。

---

<a id="policy-controls-reference"></a>

## 策略控制参考

**在策略终端中输入这些命令**（即运行 `run_policy.py` 的终端）：

<a id="general-controls-1"></a>

### 通用控制

| 操作 | 键盘 | 手柄 |
|--------|----------|----------|
| 启用策略 | `]` | A 键 |
| 停止策略 | `o` | B 键 |
| 将机器人设为默认姿态 | `i` | Y 键 |

<a id="whole-body-tracking-controls"></a>

### 全身动作跟踪控制

| 操作 | 键盘 | 手柄 |
|--------|----------|----------|
| 开始动作片段 | `m` | Select+A |

**默认姿态：** 站立并抬起双臂。

---

<a id="tips-and-troubleshooting"></a>

## 提示与故障排查

- **随时重置：** 在 MuJoCo 窗口中按 `Backspace` 重置仿真。
- **接口：** 在同一台机器进行跨仿真器测试时始终使用 `lo`（回环接口）。
- **刚性模式：** 在提示时按 `Enter` 初始化刚性控制模式；WBT 策略启动前需要该模式保持平衡。
- **稳定姿态：** 移除吊架（步骤 3）后，等待数秒让刚性控制器稳定机器人，再启用策略。
- **强化学习控制频率：** WBT 策略使用 `--task.rl-rate 50`（50 Hz 控制频率）。
- **仿真时间：** 使用 `--task.use-sim-time` 与 MuJoCo 的仿真时间同步。
