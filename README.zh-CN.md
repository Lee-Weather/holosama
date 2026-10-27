# Holosoma

[English](README.md) | [简体中文](README.zh-CN.md)

Holosoma（希腊语，意为“全身”）是一个综合性人形机器人框架，用于训练和部署人形机器人的强化学习策略，以及进行动作重定向。它支持运动控制（速度跟踪）和全身动作跟踪任务，可在多个仿真器（IsaacSim、MJWarp、MuJoCo）中使用，并提供 PPO、FastSAC 等算法。

## 功能特性

- **多仿真器支持**：IsaacGym（已停止支持）、IsaacSim、MuJoCo Warp（MJWarp）和 MuJoCo（仅用于推理）。
- **多种强化学习算法**：PPO 和 FastSAC。
- **机器人支持**：Unitree G1 和 Booster T1 人形机器人。
- **任务类型**：运动控制（速度跟踪）和全身动作跟踪。
- **仿真到仿真与仿真到现实部署**：仿真与真实机器人控制共用推理流程。
- **动作重定向**：将人体动作捕捉数据转换为机器人动作，同时保留与物体和地形的交互关系。
- **Wandb 集成**：支持视频记录、自动上传 ONNX 检查点，以及直接从 Wandb 加载检查点。

## 仓库结构

```text
src/
├── holosoma/              # 核心训练框架（运动控制与全身动作跟踪）
├── holosoma_inference/    # 推理与部署流程
└── holosoma_retargeting/  # 将人体动作数据重定向到机器人
```

## 文档

- **[训练指南](src/holosoma/README.md)**：在 IsaacSim/MJWarp 中训练运动控制和全身动作跟踪策略。
- **[推理与部署指南](src/holosoma_inference/README.md)**：将策略部署到真实机器人，或在 MuJoCo 仿真中进行评估。
- **[动作重定向指南](src/holosoma_retargeting/README.md)**：将人体动作捕捉数据转换为机器人动作。

## 快速开始

### 环境配置

根据使用场景选择相应的环境配置脚本：

```bash
# 用于 IsaacSim 训练
# 由于 IsaacSim 的依赖要求，需要 Ubuntu 22.04 或更新版本
bash scripts/setup_isaacsim.sh

# 用于 MJWarp 训练和 MuJoCo 仿真（推理）——使用 conda
bash scripts/setup_mujoco.sh

# 用于 MJWarp 训练和 MuJoCo 仿真（推理）——使用 uv（替代方案）
bash scripts/setup_mujoco_via_uv.sh

# 用于推理与部署
bash scripts/setup_inference.sh

# 用于动作重定向
bash scripts/setup_retargeting.sh
```

### 训练

在 IsaacSim 中使用 FastSAC 训练 G1 机器人：

```bash
source scripts/source_isaacsim_setup.sh
python src/holosoma/holosoma/train_agent.py \
    exp:g1-29dof-fast-sac \
    simulator:isaacsim \
    logger:wandb \
    --training.seed 1
```

> **提示：** 对于无显示界面的服务器，请参阅[训练指南](src/holosoma/README.md#video-recording)中的视频录制配置说明。

更多示例和配置选项，请参阅[训练指南](src/holosoma/README.md)。

### 快速演示

项目提供了运行完整流程的脚本，涵盖数据下载与处理（针对 LAFAN）、动作重定向、数据转换，以及全身动作跟踪策略训练。

```bash
# 使用 OMOMO 数据进行动作重定向和全身动作跟踪策略训练
bash demo_scripts/demo_omomo_wb_tracking.sh

# 使用 LAFAN 数据进行动作重定向和全身动作跟踪策略训练
bash demo_scripts/demo_lafan_wb_tracking.sh
```

### 部署与评估

训练完成后，可以通过以下方式部署策略：

- **真实机器人**：参阅[真实机器人运动控制](src/holosoma_inference/docs/workflows/real-robot-locomotion.md)或[真实机器人全身动作跟踪（WBT）](src/holosoma_inference/docs/workflows/real-robot-wbt.md)。
- **MuJoCo 仿真**：参阅[跨仿真器运动控制](src/holosoma_inference/docs/workflows/sim-to-sim-locomotion.md)或[跨仿真器全身动作跟踪](src/holosoma_inference/docs/workflows/sim-to-sim-wbt.md)。

也可以在[推理与部署指南](src/holosoma_inference/README.md)中查看所有部署选项。

### 演示视频

观看 Holosoma 策略在真实机器人上的部署效果（*点击缩略图播放*）。

<table>
  <tr>
    <th>G1 运动控制</th>
    <th>T1 运动控制</th>
    <th>G1 舞蹈</th>
  </tr>
  <tr>
    <td width="33%">
      <a href="https://youtu.be/YYMgj5BDIMI">
        <img src="https://img.youtube.com/vi/YYMgj5BDIMI/hqdefault.jpg" width="100%" alt="▶ G1 运动控制">
      </a>
    </td>
    <td width="33%">
      <a href="https://youtu.be/Q6rNHJZ2a6Y">
        <img src="https://img.youtube.com/vi/Q6rNHJZ2a6Y/hqdefault.jpg" width="100%" alt="▶ T1 运动控制">
      </a>
    </td>
    <td width="33%">
      <a href="https://youtu.be/ouPk69_eFfE">
        <img src="https://img.youtube.com/vi/ouPk69_eFfE/hqdefault.jpg" width="100%" alt="▶ G1 舞蹈">
      </a>
    </td>
  </tr>
</table>

## 旧版 Isaac Gym

> **当前版本已不再支持 NVIDIA Isaac Gym。**
>
> 如需继续使用现有的 Isaac Gym 工作流程，请使用最后一个兼容版本 [`holosoma-v0.0.3`][isaacgym-release]，并参考该版本的[原始 README][isaacgym-readme]。
>
> **支持的仿真器：** NVIDIA Isaac Sim、MuJoCo Warp 和 MuJoCo。

请单独检出该版本，使仿真后端、环境配置脚本、依赖约束和文档保持配套：

```bash
git clone --branch holosoma-v0.0.3 https://github.com/amazon-far/holosoma.git holosoma-isaacgym
cd holosoma-isaacgym
bash scripts/setup_isaacgym.sh
source scripts/source_isaacgym_setup.sh
```

有关 Isaac Gym 的训练与评估，请遵循该版本的[原始 README][isaacgym-readme]和[训练指南][isaacgym-training]。

基于该提交发布的 PyPI 包版本如下：

| 软件包 | 版本 | 标签 |
|---|---|---|
| `holosoma` | `0.0.3`（最后一个兼容 Isaac Gym 的版本） | `holosoma-v0.0.3` |
| `holosoma-inference` | `0.1.3`（最后一个支持 Python 3.8 和 3.9 的版本） | `holosoma-inference-v0.1.3` |

[isaacgym-release]: https://github.com/amazon-far/holosoma/releases/tag/holosoma-v0.0.3
[isaacgym-readme]: https://github.com/amazon-far/holosoma/blob/holosoma-v0.0.3/README.md
[isaacgym-training]: https://github.com/amazon-far/holosoma/blob/holosoma-v0.0.3/src/holosoma/README.md

## 问题反馈

欢迎提供反馈和问题报告，帮助改进 Holosoma。请通过 Issue：

- 报告缺陷和技术问题。
- 提出新功能需求。

## 获取支持

如果需要 Issue 之外的帮助，欢迎加入我们的 [Discord 社区](https://discord.gg/TPupMvpqHc)。

可以在 Discord 中讨论较大规模的计划及其他需要深入交流的问题。

## 安全

更多信息请参阅[贡献指南](CONTRIBUTING.md#security-issue-notifications)。

## 引用

如果在研究中使用 Holosoma，请按照 GitHub 仓库右侧栏“Cite this repository”（引用此仓库）面板中的信息进行引用。

## 许可证

本项目采用 Apache-2.0 许可证。
