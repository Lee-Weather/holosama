# Flux RL 镜像与算力选择

## 先识别仓库使用的仿真栈

| 工作流 | 镜像选择依据 | 详细流程 |
|---|---|---|
| Holosoma IsaacSim 训练 | 仓库 `scripts/versions.sh` 中的 IsaacSim/IsaacLab 版本及 PyTorch、Python 约束 | [holosoma-isaaclab.md](holosoma-isaaclab.md) |
| F1/X1 Isaac Gym 训练与回放 | Isaac Gym Preview 4 及对应 Python/PyTorch 组合 | 本文 F1/X1 配置与 [replay.md](replay.md) |
| Holosoma MJWarp 训练 | 仓库 MuJoCo、mujoco-warp、warp-lang 和 CUDA 约束 | 查询兼容镜像后单独适配；IsaacLab 镜像不代表已经包含 MJWarp |

不要按「版本最新」直接选择镜像，也不要把一个项目验证过的镜像作为所有 RL 任务的默认值。

## 选择原则

创建任务前先查询当前平台资源，不要只依赖硬编码推荐：

```text
flux task resource list --goods-back-category 3 --page 1 --limit 20
flux task image official
flux task image versions --image-id <image>
```

flux-cli `1.1.0` 已验证 RL 与 VLA 共用资源类别 `3`；类别 `6` 当前返回空列表，不要使用。

字段含义：

- `goodsId`：填写 `taskBaseInfo.goodsId`；
- `goodsBackId`：后端 SKU 标识，只有接口要求时填写对应字段；
- `goodsBackSku.gpuNum`：填写 GPU 数量；
- 镜像版本返回中的 `id`：填写 `taskBaseInfo.imageVersion`；
- `versionCode`：仅供展示，不能代替版本 `id`。

## Holosoma IsaacLab 镜像查询记录

2026-10-27 使用 Flux CLI 1.1.0 实际查询得到：

| 项目 | 值 |
|---|---|
| 官方镜像 ID | `BJX00000178` |
| 镜像版本 ID（`imageVersion`） | `V000420` |
| 可读版本名（不填入 `imageVersion`） | `isaac-lab2.3.2-v2` |
| 运行环境描述 | IsaacSim 5.1 / IsaacLab 2.3.2 / Python 3.11.13 / PyTorch 2.7.0 / Ubuntu 22.04.5 |
| 当前账号可见算力 | `ESKU000001`，`goodsBackId=SKUSL000002`，1×4090D 24G，`gpuNum=1` |

该组合与当时 Holosoma 的 `ISAACSIM_VERSION=5.1.0`、`ISAACLAB_REF=v2.3.2`、PyTorch 2.7.0 要求匹配。2026-10-28 重新查询仍可用，并完成 `TASK_20261028_224` 的 128 环境、2 次 PPO 迭代；依赖修复、产物与验证范围见 [Holosoma 实测记录](holosoma-isaaclab.md#已完成的云端验证)。这不保证后续实时库存，创建前重新查询。

## F1/X1 历史已验证配置

F1/X1 的 Isaac Gym RL 训练与回放已验证：

| 配置 | 值 |
|---|---|
| 算力 | `ESKU000001`（1×4090D 24G） |
| 镜像 ID | `BJX00000001` |
| 镜像版本 | `V000124` |
| GPU 数量 | 1 |
| 运行环境 | Isaac Gym Preview 4 / Python 3.8 / PyTorch 2.4.x |

推荐值只在平台查询仍返回该资源和镜像时使用。若已下架或版本变化，选择兼容 Isaac Gym Preview 4、CUDA、Python 和 PyTorch 的最新可用镜像，并记录实际选择。

## 创建与恢复规则

- 新建 RL 任务使用 `trainType=1`；Holosoma 先按对应参考文档准备依赖与小规模训练入口。
- 恢复训练通常使用 `trainType=2`，并复用源任务的算力、镜像、代码和 checkpoint 元数据。
- Isaac Gym 回放的 checkpoint 挂载已验证不可靠，回放仍使用 `trainType=1` + 运行时下载。
- RL `startScript` 必须以 `gm-run` 开头，即使客户端命令已经改为 `flux`。
- VLA 任务通常不传 `startScript`，由底模配置自动生成运行命令。

## 提交前检查

提交 payload 前确认：

- `projectId`、`goodsId`、`imageId`、`imageVersion` 存在；
- Git 分支已推送且平台有权限读取；
- `mainCodeUri` 与 `startScript` 使用 clone 后的仓库路径；
- `hparamsPath` 包含仓库目录前缀；
- 恢复训练的 checkpoint 来自 `model list`，不要猜测对象路径；
- 先执行 `--dry-run`，确认后再正式提交。
