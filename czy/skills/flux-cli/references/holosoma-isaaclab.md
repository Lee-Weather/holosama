# Holosoma 在 Flux IsaacLab 镜像上的训练适配

## 适用范围与证据

用于核查、准备和运行 Holosoma 的 IsaacSim 训练。F1/X1 Isaac Gym 和 VLA 使用各自流程。

2026-10-27 查询到版本匹配的官方镜像与算力，实际 ID 和环境描述统一记录在 [resources.md 的 Holosoma 查询记录](resources.md#holosoma-isaaclab-镜像查询记录)。2026-10-28 经两次安装修复后，`TASK_20261028_224` 已完成 G1 PPO 小规模云端训练、checkpoint 上传及 ONNX CPU 推理校验。

结论：当前镜像加仓库包装入口已跑通小规模训练流程；尚未验证长时间收敛、视频、WBT 或其他环境规模。`--dry-run` 仅预览请求，不能替代运行证据。

### 已完成的云端验证

| 项目 | 实测结果 |
|---|---|
| 账号 / 用户 ID | `limxmts0elyormeym6@uberip.com` / `4390`（用户明确指定） |
| 项目 / 任务 | `PRO_20261028_008` / `TASK_20261028_224` |
| 训练代码 commit | `b438b4e23a7a3a1d52136a2b0b8a687b0ba9d154` |
| 镜像 / 版本 | `BJX00000178` / `V000420` |
| 算力 | `ESKU000001`；CUDA 报告 NVIDIA GeForce RTX 4090，显存 25,393,692,672 bytes |
| 配置 | `exp:g1-29dof`、PPO、IsaacSim、平地、128 环境、2 迭代、seed=1、无视频/Wandb |
| 结束状态 / 时长 | `5`（Completed），2026-10-28 17:56:26–18:00:44，258 秒（含安装和上传等待） |
| 训练步数 / 速度 | 6,144；最后一轮约 3,570 steps/s，迭代本身约 0.86 秒 |
| 最后一轮日志 | mean reward=-3.96，mean episode length=41.18；仅流程验证，不能判断收敛 |
| checkpoint | `model_00001.pt`，5,300,224 bytes，iter=1；平台上传并下载后确认 actor 权重均有限 |
| ONNX | 云端生成 958,244 bytes，CPU 输入全零，输出 shape=(1,29) 且有限；模型列表未返回 ONNX 下载项 |
| TensorBoard / 平台曲线 | 1 个事件文件；首次奖励查询只有 step 0，归档补查 reward/episode length/FPS 均有 steps `[0,1]` |

本地 checkpoint SHA-256：`8d60055838bec656e9bf5bf7e691b6642d28b7b36f45df0c4d5f4f83d4baa1c2`。

平台刚完成时的 `accelerate` 奖励查询只有 step 0；本次 exp0 归档补查时，`precise` 已得到两轮 reward、episode length 和 FPS。最终 reward=-3.955620765686035、episode length=41.181819915771484；表内数值保留日志显示精度。时间和采样模式同时改变，不能据此判断缺点原因，也不证明其他指标完整。

完整实验、三次尝试、验收和经验统一见 [exp0 实验笔记](../../../exp1/exp1.md)。最终 checkpoint 已归档至 [czy/data/exp0/model_00001.pt](../../../data/exp0/model_00001.pt)，实际配置、脱敏日志、API 曲线和校验 JSON 位于 [exp0 附件目录](../../../exp1/artifacts/exp0/)。`logs/flux-validation/TASK_20261028_224/` 仅为 Git 忽略的工作副本。不要把此账号设为技能的通用默认账号；后续按用户当次指定确认身份。

## 每次使用时重新核查

按主技能要求先检查 CLI 帮助，再使用默认 JSON 输出查询：

```text
flux --yes task image official
flux --yes task image versions --image-id BJX00000178
flux --yes task resource list --goods-back-category 3 --page 1 --limit 50
```

只投影镜像 ID、说明、版本及资源字段，不输出完整带凭据的任务详情。账号、仓库或平台目录改变后，以上 ID 仅作查询线索。

对照当前 checkout 的这些文件，而不是只看镜像名称：

- `scripts/versions.sh`：本次为 IsaacSim 5.1.0、IsaacLab v2.3.2。
- `scripts/setup_isaacsim.sh`：本次安装 PyTorch 2.7.0 / torchvision 0.22.0。
- `scripts/isaacsim_constraints.txt`：保留 IsaacSim 运行时的依赖约束。
- `src/holosoma/pyproject.toml`：训练包要求 Python >=3.10、NumPy <2，并有额外 Python 依赖。

不要为了「更新」选 IsaacSim 6 / IsaacLab 3 的镜像，也不要退回 Isaac Gym。MJWarp 使用另一组 MuJoCo/Warp 约束，需另行核查。

## 运行环境与依赖

镜像预装 IsaacLab 不代表预装了当前 Holosoma 包。云端运行前需要确认：

1. `gm-run` 实际使用的 Python 解释器及其 Python、torch、IsaacSim、IsaacLab 版本。
2. CUDA 可用、实际 GPU/显存与驱动满足当前 IsaacSim 镜像要求。
3. 同一解释器能找到 `holosoma` 及其 `tensordict`、Tyro、ONNX 导出等依赖。
4. 机器人资产可读，URDF 导入扩展、扩展缓存和必要网络访问可用。
5. 依赖补齐后，平台 SDK 仍能正常加载和扫描日志/模型。

优先复用镜像现有的 IsaacLab/Python 环境，按仓库约束补齐训练包。例如下列命令须在云端仓库根目录、且 Python 与 `gm-run` 一致时执行：

```bash
python -m pip install -c scripts/flux_smoke_constraints.txt -e ./src/holosoma
python -m pip check
```

这组约束已在上述任务完成安装和训练。Flux 专用约束包含原 IsaacSim 约束，并固定与 `packaging==23.0`、NumPy 1.x 兼容的 Jupyter/OpenCV 组合，复用镜像 TensorDict 0.9.0；修改后先在本地解析 Linux/Python 3.11 依赖。不要用 `--no-deps` 隐藏缺失项，也不要无依据整体升级镜像中的 torch、IsaacSim 或平台 SDK。仅做训练不必同时安装 inference/retargeting；后者 NumPy 2.3.5 与训练包 NumPy <2 冲突，应分环境。

上述成功任务仍有 pip 全局依赖告警：`nvidia-srl-usd` 未检测到 `usd-core` 发行包、`nvidia-srl-base` 要求 docstring-parser 0.16 而 Tyro 安装 0.18，以及镜像 FastAPI 0.115.7 与 Starlette 0.49.1 的声明不符。本次仿真、训练、SDK 上传均实际通过，但不能宣称全环境 `pip check` 通过。按使用范围分别诊断，不要直接往 IsaacSim 环境安装另一套 USD 来消除元数据告警。

实际遇到的安装故障：

- `TASK_20261028_067`：镜像默认源 `mirrors.ivolces.com` 在 worker 中无法解析，安装 build backend 失败。包装入口应探测可达源并显式传给 `python -m pip`。
- `TASK_20261028_113`：PyPI 元数据下载有超时，且无上界的 Notebook/JupyterLab 与 IsaacSim 的 packaging 约束导致大量版本回溯，600 秒安装超时；没有 checkpoint。先修正依赖解析，再重试，不应仅反复增加计费任务。
- `gm-run` 使用 `/workspace/isaaclab/_isaac_sim/kit/python/bin/python3`；镜像可能没有名为 `pip` 的独立可执行文件，但 `python -m pip` 可用。IsaacLab 发行包元数据版本为 `0.54.2`，不能直接据此判定镜像不符 2.3.2 发布版本。
- 内网代理仅用于本机 Git/CLI；云端安装源由 worker 自己探测。入口优先探测国内镜像，再回退 PyPI，避免索引可达但其文件 CDN 持续超时。

不要在已有匹配镜像中默认执行完整 `setup_isaacsim.sh`：它可能另建 Conda 环境、重复下载运行时，并使训练进程脱离 SDK 所在环境。`setup_isaacsim_delta.sh` 也有运行时路径和 Conda 布局前提，不能只凭镜像名称就直接套用。

若平台没有已验证的依赖安装阶段，应在用户要求准备/运行训练时补充仓库内的 Python 包装入口：在训练导入前完成必要安装或环境检查，再进入原训练入口，并保留原参数和退出码。包装入口须实际存在、验证并推送后才能填入任务。不要把 `pip install ... && python ...` 或 `source ...` 填入 RL `startScript`；该字段仍必须是 `gm-run ...`。

## 仓库、路径与配置

本次用户仓库为 `https://github.com/Lee-Weather/holosama.git`，默认分支 `main`。clone 目录名是 **holosama**，Python 包名是 **holosoma**，不要混淆。

对应原始训练入口：

```text
holosama/src/holosoma/holosoma/train_agent.py
```

训练前按 [git-mirror.md](git-mirror.md) 核对 `origin` 上的 commit；Git 操作可使用本会话用户指定的单次代理：

```powershell
git -c http.proxy=http://10.12.202.95:1080 ls-remote origin refs/heads/main
```

该代理地址是用户于 2026-10-28 指定的本地网络配置；不要把本机或内网代理写入云端任务当成可访问的代理。

Holosoma 使用 Tyro/dataclass 组合配置：

- 实验预设如 `exp:g1-29dof`、`exp:g1-29dof-fast-sac`、`exp:g1-29dof-wbt`。
- 仿真选择使用 `simulator:isaacsim`，不是 `--task=...`。
- 参数使用 `--training.num-envs`、`--algo.config.num-learning-iterations` 等实际字段；不要套用 F1/X1 的 `--num_envs`、`--max_iterations`。
- 没有统一的 `configs/train.yaml`。不要为了填写 `hparamsPath` 编造文件路径，也不要假定平台可以自动修改组合式 Python 配置。没有验证过的超参文件时先留空，用显式 CLI 参数并记录实验预设；如需平台超参编辑，再单独验证该功能。

## 小规模验证与正式训练

获得训练运行授权后，先确认安装阶段和入口就绪，再执行小规模训练。

当前仓库提供 `scripts/train_flux_smoke.py`：在 `gm-run` 的解释器中补齐训练包依赖，使用新 Python 子进程运行 G1 PPO，再加载 checkpoint、执行 ONNX CPU 推理并检查 TensorBoard 文件。默认 128 环境、2 次迭代，安装限时 900 秒、训练限时 600 秒，最后预留 60 秒供平台扫描上传。平台启动命令为：

```text
gm-run holosama/scripts/train_flux_smoke.py
```

这个包装入口的 5 项 CPU 测试及上述云端任务已通过；后续运行仍以当次日志和产物为准。

下面是**依赖已经安装且 `gm-run` 使用正确解释器时**的候选启动命令；若需要 Python 包装入口，用已提交的实际路径替换脚本部分：

```text
gm-run holosama/src/holosoma/holosoma/train_agent.py exp:g1-29dof simulator:isaacsim logger:disabled terrain:terrain-locomotion-plane --training.num-envs=128 --training.headless=True --algo.config.num-learning-iterations=2 --algo.config.save-interval=1 --logger.video.enabled=False
```

这些入口参数已由包装脚本在上述 Flux 任务使用。`logger:disabled` 关闭 Wandb，TensorBoard 仍启用。首次验证关闭视频，避免渲染和 Wandb 认证干扰；24GB 显存下从小环境数起步，不能把默认 4096 环境视为已验证容量。

构造任务时：

- 使用用户指定或确认的项目，`taskType=1`、`trainType=1`、`codeType=2`。
- 填实际查询到的 `goodsId`、`goodsBackId`、`gpuNum`、`imageId` 和版本 **ID**。
- `codeUrl` 使用包含仓库地址、版本类型及分支的 JSON 字符串；分支应包含需要的 commit。
- `mainCodeUri`、`startScript` 及非空的 `hparamsPath` 与实际 clone 目录一致。
- `personalDataPath` 仅在确认资源支持相应存储后设置；不要照搬其他任务的挂载。
- 先 `flux --yes task create --file ./create-holosoma.json --dry-run`；退出码 `10` 只说明预览通过。
- 正式创建和启动必须在运行授权范围内；创建成功后立即清理临时 payload，状态 `0` 表示仍需 `task run`。

本次 `codeUrl` 的值为字符串 `[{"codeUrl":"https://github.com/Lee-Weather/holosama.git","versionType":"1","versionName":"main"}]`，不是嵌套 JSON 数组；`mainCodeUri=holosama/scripts/train_flux_smoke.py`，`hparamsPath` 为空。安装故障修复后可复制同账号任务重试；运行前核对新任务为状态 0、所属用户及镜像资源正确，且远端分支包含修复 commit。不要用不完整 `task edit` 请求覆盖任务（该接口为全量更新）。

小规模验证成功需看到 GPU 仿真正常启动、完成指定迭代、生成可加载的 `.pt`、完成 `.onnx` 导出并正常结束。出错时按依赖/导入、资产/扩展、显存、训练异常和 SDK 上传分别定位；同一未解决故障不要连续创建新任务重试。

正式训练再按 PPO/FastSAC、locomotion/WBT、视频需求和实际显存调整环境数及迭代数。WBT 使用真实 motion NPZ；仓库自带参考动作可作初始验证，不需要 F1/X1 的轨迹参数。

## 日志与产物

Holosoma 默认写入 `logs/<project>/<timestamp>-<name>-<task>/`，包含 TensorBoard、配置、`.pt` 和导出的 `.onnx`。平台 SDK 是否发现、上传这些文件，需要实际日志和 `model list` 共同验证。

- 运行中先查实际指标 keys，再用 `precise`；不能凭 CLI 请求成功断言平台曲线已接入。
- `logger:disabled` 仍有 TensorBoard，但平台是否解析对应事件文件需实测。
- `.pt` 下载从 `model list` 获取 `policUrlDown`。上述成功任务的 SDK 发现了 ONNX 文件，但模型列表仅有两个 `.pt`，没有 ONNX 下载项；部署前需另行验证 ONNX 的保存/下载通道。不要把 ONNX 改成 `.pt` 扩展名来伪装上传。
- 视频按 Holosoma 的 `logger.video` 配置另行验证；不默认要求 `play_output.mp4`、`model_isaac_csv.pt` 或 `isaac_diag.csv`。
- 默认交付 checkpoint、需要部署的 ONNX，以及用户要求的日志/视频。记录任务 ID、实际镜像与版本、commit、实验预设、环境数、迭代数和验证范围；不记录密钥或签名 URL。
