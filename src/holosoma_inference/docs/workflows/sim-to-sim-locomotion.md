<a id="sim-to-sim-locomotion-workflow"></a>

# 跨仿真器运动控制工作流程

> **另请参阅：** [推理与部署指南](../../README.md)，了解所有部署选项。

本指南介绍在 MuJoCo 仿真中运行运动控制策略的完整流程。

<a id="overview"></a>

## 概述

跨仿真器工作流程可在 MuJoCo 中运行 IsaacSim 训练的运动控制检查点，以便评估和测试。

<a id="prerequisites"></a>

## 前提条件

- 已配置 MuJoCo 环境（`scripts/source_mujoco_setup.sh`）
- 已配置 Holosoma 推理环境（`scripts/source_inference_setup.sh`）
- ONNX 模型检查点
- 用于控制的键盘或手柄

**注意：** 推理程序和 MuJoCo 在同一台机器运行时，始终使用 `--task.interface lo`（回环接口）。

---

<a id="robot-workflows"></a>

## 机器人工作流程

<a id="unitree-g1-29-dof"></a>

### Unitree G1（29 自由度）

<a id="1-start-mujoco-environment"></a>

#### 1. 启动 MuJoCo 环境

在一个终端中启动 MuJoCo 环境：

**使用键盘控制：**
```bash
source scripts/source_mujoco_setup.sh
python src/holosoma/holosoma/run_sim.py robot:g1-29dof
```

**使用手柄控制：**
```bash
source scripts/source_mujoco_setup.sh
python src/holosoma/holosoma/run_sim.py robot:g1-29dof \
    --simulator.config.bridge.enabled=True \
    --simulator.config.bridge.use-joystick=True
```

机器人将在仿真器中生成，并吊挂在保护架上。

<a id="2-launch-the-policy"></a>

#### 2. 启动策略程序

在另一个终端中运行策略推理：

**使用键盘控制：**
```bash
source scripts/source_inference_setup.sh
python3 src/holosoma_inference/holosoma_inference/run_policy.py inference:g1-29dof-loco \
    --task.model-path src/holosoma_inference/holosoma_inference/models/loco/g1_29dof/fastsac_g1_29dof.onnx \
    --task.no-use-joystick \
    --task.interface lo
```

**使用手柄控制：**
```bash
source scripts/source_inference_setup.sh
python3 src/holosoma_inference/holosoma_inference/run_policy.py inference:g1-29dof-loco \
    --task.model-path src/holosoma_inference/holosoma_inference/models/loco/g1_29dof/fastsac_g1_29dof.onnx \
    --task.use-joystick \
    --task.interface lo
```

<a id="3-deploy-the-robot"></a>

#### 3. 让机器人接触地面

- 在 MuJoCo 窗口中按 `8` 降低吊架，直到机器人接触地面。
- 在 MuJoCo 窗口中按 `9` 移除吊架。

<a id="4-start-the-policy"></a>

#### 4. 启用策略

在策略终端中按 `]`（或按手柄上的 `A` 键）启用策略。

<a id="5-enter-walking-mode"></a>

#### 5. 进入行走模式

在策略终端中按 `=`（或按手柄上的 `Start` 键）进入行走模式。

<a id="6-control-the-robot"></a>

#### 6. 控制机器人

- 使用键盘（`w` `a` `s` `d` 控制线速度，`q` `e` 控制角速度）或手柄控制速度。
- 左摇杆：前进、后退、左移和右移。
- 右摇杆：左转和右转。

---

<a id="booster-t1-29-dof"></a>

### Booster T1（29 自由度）

<a id="1-start-mujoco-environment-1"></a>

#### 1. 启动 MuJoCo 环境

在一个终端中启动 MuJoCo 环境：

**使用键盘控制：**
```bash
source scripts/source_mujoco_setup.sh
python src/holosoma/holosoma/run_sim.py robot:t1-29dof-waist-wrist
```

**使用手柄控制：**
```bash
source scripts/source_mujoco_setup.sh
python src/holosoma/holosoma/run_sim.py robot:t1-29dof-waist-wrist \
    --simulator.config.bridge.enabled=True \
    --simulator.config.bridge.use-joystick=True
```

机器人将在仿真器中生成，并吊挂在保护架上。

<a id="2-launch-the-policy-1"></a>

#### 2. 启动策略程序

在另一个终端中运行策略推理：

**使用键盘控制：**
```bash
source scripts/source_inference_setup.sh
python3 src/holosoma_inference/holosoma_inference/run_policy.py inference:t1-29dof-loco \
    --task.model-path src/holosoma_inference/holosoma_inference/models/loco/t1_29dof/ppo_t1_29dof.onnx \
    --task.no-use-joystick \
    --task.interface lo
```

**使用手柄控制：**
```bash
source scripts/source_inference_setup.sh
python3 src/holosoma_inference/holosoma_inference/run_policy.py inference:t1-29dof-loco \
    --task.model-path src/holosoma_inference/holosoma_inference/models/loco/t1_29dof/ppo_t1_29dof.onnx \
    --task.use-joystick \
    --task.interface lo
```

<a id="3-deploy-the-robot-1"></a>

#### 3. 让机器人接触地面

- 在 MuJoCo 窗口中按 `8` 降低吊架，直到机器人接触地面。
- 在 MuJoCo 窗口中按 `9` 移除吊架。

<a id="4-start-the-policy-1"></a>

#### 4. 启用策略

在策略终端中按 `]` 启用策略。

<a id="5-enter-walking-mode-1"></a>

#### 5. 进入行走模式

在策略终端中按 `=` 进入行走模式。

<a id="6-control-the-robot-1"></a>

#### 6. 控制机器人

使用键盘控制速度：
- `w` `a` `s` `d` 控制线性运动（前进、后退、左移、右移）。
- `q` `e` 控制角运动（左转、右转）。

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

<a id="locomotion-controls"></a>

### 运动控制

| 操作 | 键盘 | 手柄 |
|--------|----------|----------|
| 切换行走/站立模式 | `=` | Start 键 |
| 调整线速度 | `w` `a` `s` `d` | 左摇杆 |
| 调整角速度 | `q` `e` | 右摇杆 |

---

<a id="tips-and-troubleshooting"></a>

## 提示与故障排查

- **随时重置：** 在 MuJoCo 窗口中按 `Backspace` 重置仿真。
- **接口：** 在同一台机器进行跨仿真器测试时始终使用 `lo`（回环接口）。
- **站立模式：** 机器人初始处于站立模式，按 `=` 或 `Start` 切换到行走模式。
- **默认姿态：** 按 `i` 或 `Y` 键使机器人恢复站立姿态。
