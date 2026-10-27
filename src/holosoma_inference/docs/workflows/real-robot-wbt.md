<a id="real-robot-whole-body-tracking-workflow"></a>

# 真实机器人全身动作跟踪工作流程

> **另请参阅：** [推理与部署指南](../../README.md)，了解所有部署选项。

本指南介绍在真实机器人硬件上运行全身动作跟踪（WBT）策略的完整流程。

<a id="overview"></a>

## 概述

在真实 Unitree G1 机器人上部署和运行 WBT 策略，执行动作跟踪和动态运动。

**注意：** WBT 目前尚不支持 Booster T1。

<a id="prerequisites"></a>

## 前提条件

- 真实机器人硬件（Unitree G1）
- 以太网线
- Unitree G1 遥控器
- 已配置 Holosoma 推理环境的笔记本电脑

<a id="hardware-setup"></a>

## 硬件配置

<a id="1-prepare-the-robot"></a>

### 1. 准备机器人

- 将机器人吊挂在保护架上。
- 启动机器人和遥控器。
- 使用网线连接机器人和笔记本电脑。
- 将机器人切换到阻尼模式。
- 按遥控器上的 `L2+R2` 进入开发模式。

详细的硬件配置说明请参阅 [Unitree 快速开始页面](https://support.unitree.com/home/en/G1_developer/quick_start)。

<a id="2-configure-network"></a>

### 2. 配置网络

配置笔记本电脑的网络接口：
- IP 地址：`192.168.123.224`
- 子网掩码：`255.255.255.0`

<a id="3-find-your-network-interface"></a>

### 3. 查找网络接口

查找与机器人连接的网络接口：

```bash
ifconfig
```

查找 IP 地址为 `192.168.123.224` 的接口。常见名称包括：
- `eth0`：常见的以太网接口名称。
- `enp0s31f6`：现代 Linux 以太网命名方式。

<a id="running-the-policy"></a>

## 运行策略

<a id="1-launch-the-policy"></a>

### 1. 启动策略程序

```bash
source scripts/source_inference_setup.sh
python3 src/holosoma_inference/holosoma_inference/run_policy.py inference:g1-29dof-wbt \
    --task.model-path src/holosoma_inference/holosoma_inference/models/wbt/fastsac_g1_29dof_dancing.onnx \
    --task.use-joystick \
    --task.rl-rate 50 \
    --task.interface eth0
```

**注意：** 将 `eth0` 替换为实际网络接口名称（例如 `enp0s31f6`），可使用 `ifconfig` 查看。

<a id="2-initialize-stiff-control-mode"></a>

### 2. 初始化刚性控制模式

策略终端出现提示时按 `Enter`。机器人进入刚性控制模式并保持初始姿态。

<a id="3-start-the-policy"></a>

### 3. 启用策略

按手柄上的 `A` 键启用策略。

<a id="4-start-motion-clip"></a>

### 4. 开始动作片段

按手柄上的 `Select+A` 开始动作片段，机器人将开始跟踪全身动作。

---

<a id="policy-controls-reference"></a>

## 策略控制参考

**在策略终端中输入这些命令**（即运行 `run_policy.py` 的终端）：

<a id="general-controls"></a>

### 通用控制

| 操作 | 键盘 | 手柄 |
|--------|----------|----------|
| 启用策略 | `]` | A 键 |
| 停止策略 | `o` | B 键 |
| 将机器人设为默认姿态 | `i` | Y 键 |
| 结束控制程序 | - | L1（LB）+ R1（RB） |

<a id="whole-body-tracking-controls"></a>

### 全身动作跟踪控制

| 操作 | 键盘 | 手柄 |
|--------|----------|----------|
| 开始动作片段 | `m` | Select+A |

**默认姿态：** 站立并抬起双臂。

---

<a id="deployment-options"></a>

## 部署选项

<a id="option-1-run-offboard-laptop"></a>

### 选项 1：在外部电脑运行（笔记本电脑）

这是上文介绍的默认部署方式：策略在笔记本电脑上运行，并通过以太网与机器人通信。

**优点：**
- 便于调试和监控。
- 无需修改机器人软件。
- 便于快速迭代。

**要求：**
- 稳定的以太网连接。
- 具有足够算力的笔记本电脑。

<a id="option-2-run-onboard-jetson"></a>

### 选项 2：在机载电脑运行（Jetson）

在机器人机载 Jetson 电脑上直接运行策略，以降低延迟。

<a id="setup-steps"></a>

#### 配置步骤

1. 完成硬件配置（见上文）。

2. 通过 SSH 连接机载 Jetson：
   ```bash
   ssh unitree@192.168.123.164
   # 默认密码：'123'
   ```

3. 将 Jetson 设置为最高性能模式：
   ```bash
   sudo jetson_clocks
   ```

4. 在 Jetson 上配置环境：
   ```bash
   cd ~/holosoma
   bash scripts/setup_inference.sh
   source scripts/source_inference_setup.sh
   ```

5. 使用 `eth0` 接口运行策略：
   ```bash
   python3 src/holosoma_inference/holosoma_inference/run_policy.py inference:g1-29dof-wbt \
       --task.model-path <path-to-onnx> \
       --task.use-joystick \
       --task.rl-rate 50 \
       --task.interface eth0
   ```

**优点：**
- 延迟更低。
- 配置完成后无需外接笔记本电脑。
- 便于移动使用。

<a id="option-3-run-inside-docker"></a>

### 选项 3：在 Docker 中运行

在 Docker 容器中运行策略，机载和外部电脑部署均可使用。

<a id="setup-steps-1"></a>

#### 配置步骤

1. 完成硬件配置（见上文）。

2. 构建 Docker 镜像：
   ```bash
   ECR_REGISTRY="account-id.dkr.ecr.region.amazonaws.com" \
     bash holosoma/src/holosoma_inference/docker/build.sh
   ```

3. 创建并进入 Docker 容器：
   ```bash
   bash holosoma/src/holosoma_inference/docker/run.sh
   ```

4. 在容器中运行策略：
   - 在 Jetson 上使用 `eth0`。
   - 在笔记本电脑上使用实际接口名称（通过 `ifconfig` 查看）。

**优点：**
- 环境一致。
- 便于管理依赖。
- 配置可复现。

---

<a id="tips-and-troubleshooting"></a>

## 提示与故障排查

<a id="network-issues"></a>

### 网络问题

- **无法连接机器人：** 确认 IP 为 `192.168.123.224`，子网掩码为 `255.255.255.0`。
- **接口错误：** 使用 `ifconfig` 确认与机器人连接的接口。
- **连接中断：** 确认网线连接正常且未损坏。

<a id="robot-behavior"></a>

### 机器人行为

- **刚性模式：** 在提示时按 `Enter` 初始化刚性控制模式；WBT 策略启动前需要该模式保持平衡。
- **紧急停止：** 按手柄上的 `L1 + R1`（LB + RB）结束控制程序。
- **默认姿态：** 按 `Y` 键使机器人恢复双臂抬起的站立姿态。
- **动作未开始：** 确认已先按 `A` 启用策略，再按 `Select+A`。

<a id="control-issues"></a>

### 控制问题

- **手柄无响应：** 确认已设置 `--task.use-joystick` 参数。
- **键盘控制：** 去掉 `--task.use-joystick` 参数，并使用 `m` 键开始动作片段。
- **混合输入：** 分别使用 `--task.velocity-input` 和 `--task.state-input` 配置混合输入（例如 ROS2 速度指令和键盘命令）。参阅[输入来源](../../README.md#input-sources)。
- **响应迟缓：** 在外部电脑运行时检查网络延迟。
- **强化学习控制频率：** WBT 策略始终使用 `--task.rl-rate 50`（50 Hz 控制频率）。

<a id="safety"></a>

### 安全

- 初始测试时**始终**将机器人保持在保护吊架上。
- 准备好紧急停止操作（`L1 + R1`）。
- WBT 动作可能幅度较大，确保机器人周围留有足够空间。
- 尝试复杂编排动作之前，先测试简单动作。

<a id="motion-clip-issues"></a>

### 动作片段问题

- **动作片段未播放：** 确认启用策略后已按 `Select+A`。
- **跟踪不稳定：** 确认刚性控制模式已成功初始化。
