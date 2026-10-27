<a id="native-inference-library"></a>
# 原生推理库

C++ 库负责模型执行、观测历史、动作缩放和策略状态。Python 绑定的是同一套实现。构造库中的策略时，不会连接机器人、打开套接字、启动键盘监听器或打开 SDK 接口。

<a id="python"></a>
## Python

统一入口是 `Policy(preset_path, checkpoint_path)`。对于运动控制、内嵌动作的全身跟踪（WBT）或由调用方提供参考的 WBT，其 `step(RobotState)` 都会返回一个 `JointCommand`。`set_reference(ReferenceWindow)` 为后一种方式提供参考数据。下面的预设和配置工厂均基于同一个类进行组合。

使用 C++17 编译器和 CMake 构建扩展：

```sh
uv pip install ./src/holosoma_inference
```

CMake 从已安装的软件包或固定版本的下载资源中解析依赖。如需使用已有 SDK（包括支持 CUDA 的 SDK），请设置 `ONNXRUNTIME_ROOT`。默认的源码构建使用 CPU 推理。机器人 SDK 是显式启动的命令行运行程序的可选依赖，无设备 API 不依赖它们。

```python
from dataclasses import replace
import numpy as np
import holosoma_inference as inference

config = inference.get_config("g1-29dof-loco")
config = replace(config, task=replace(config.task, model_path="policy.onnx"))
policy = inference.get_policy_from_preset(config)
policy.reset()

# One already-packed actor observation, in the model's configured term order.
obs_config = config.observation
width = sum(obs_config.obs_dims[t] for t in obs_config.obs_dict["actor_obs"])
width *= obs_config.history_length_dict.get("actor_obs", 1)
targets = policy.step(np.zeros((1, width), dtype=np.float32))
```

`get_config` 返回独立副本，并包含已注册的扩展预设。机器人配置子类中的 SDK 专用字段仍可供 Python 通信层使用；原生引擎接收通用机器人约定。还导出了 `InferenceConfig` 和 `LocomotionPolicy(config)`，以便不使用命名预设进行组合。模型路径可以是本地路径或受支持的 W&B URL，路径会在构造原生对象之前解析。打包后的观测必须具有 `(N,)` 或 `(1, N)` 形状，且数值必须有限。返回的 `(1, num_joints)` 数组包含绝对关节位置目标，其中已包含默认值和标定偏移，而不是神经网络的原始动作。

使用原生观测构造时，`policy.step_state(state)` 接受以下按关节顺序排列的扁平状态：
`[base_pos(3), quaternion_wxyz(4), joint_pos(N), base_lin_vel(3),
base_ang_vel(3), joint_vel(N)]`，其后可以附加修正后的重力向量（3 个值）。返回的命令包含 `q`、`dq`、`tau`、`kp` 和 `kd`。若要通过 `policy.command("START")` / `"STOP"` / `"INIT"` 显式控制，应以 `autostart=False` 启动。`velocity(x, y, yaw)` 用于提供运动请求。`step_state` 每调用一次会推进一次步态相位；如果外部循环已负责推进相位，应传入 `advance_phase=False`。`infer_offsets(state)` 用于预热原生模型和历史，不会改变启用状态或发布命令。

<a id="live-reference-whole-body-tracking"></a>
## 实时参考全身跟踪

内嵌动作策略从模型中读取参考输出。实时参考策略则从调用方接收参考帧，并根据 ONNX `experiment_config` 元数据确定观测布局。两种约定均是显式定义的，不会仅根据宽度猜测另一种布局。

与 SONIC 对齐的布局名称指的是 NVIDIA 公开的 [GEAR-SONIC 模型系列](https://github.com/NVlabs/GR00T-WholeBodyControl)。为保持兼容，解析器保留了相应的导出器元数据键。

```python
from holosoma_inference import Policy, RobotState, ReferenceWindow

policy = Policy.from_live_model("live-policy.onnx")
state = RobotState()
state.dof_pos = state.dof_vel = [0.0] * policy.num_dofs()
reference = ReferenceWindow()
slots = max(1, len(policy.lookahead_offsets))
# Replace these illustrative values with caller-buffered frames at the trained offsets.
reference.q = reference.dq = [0.0] * (slots * policy.num_dofs())
reference.quat = [[1.0, 0.0, 0.0, 0.0]] * slots
policy.set_reference(reference)
policy.set_elapsed_seconds(0.0)
command = policy.step(state)
```

上述代码仅展示数据类型，并非机器人动作序列。实时帧使用检查点元数据中的关节顺序、wxyz 四元数方向、机体局部坐标系角速度和国际单位制。`reset_heading()` 重新锚定航向和动作相位，同时保留历史；`reset()` 开始新的 episode，并清除历史与动作反馈。

历史初始化是运行时设置 `task.wbt_history_initialization`（`run_policy` 和 C++ 运行程序使用 `--task.wbt-history-initialization`，`Policy.from_live_model` 和 `LiveWbtPolicy` 使用 `history_initialization=`，C++ 使用 `WbtPolicy::SetHistoryInitialization`）。默认值 `repeat_first_valid_obs` 使用每个观测项首次有效且完整构建的帧初始化该项历史，之后按从旧到新的顺序滚动更新。`zero_pad` 用零填充缺失帧。完整调用 `reset()` 后，会根据下一次观测重新初始化；仅重置航向或替换参考缓冲区则保留历史。更改该设置会重置策略，因此同一个 episode 中不会混用两种方式。导出的 ONNX 元数据仍可能将 `wbt_history_initialization` 声明为 `repeat_first` 或 `zero`。该声明会被校验并记录在加载日志中，但最终由运行时设置决定；覆盖检查点设置时，日志会作出说明。元数据和运行时设置中的其他拼写都会导致失败。该设置仅适用于元数据驱动的实时 WBT 检查点；内嵌动作 WBT 和运动控制预设接受但忽略它，因为它们的历史始终使用零填充。

原始动作在动作缩放和动作历史反馈之前，会被对称裁剪到最大幅值 20（`kWbtActionCeiling`）。这是对策略原始输出的限制，而非关节角度限制：缩放后的目标仍受常规关节和驱动器限制。若声明的 `experiment_config.robot.control.action_clip_value` 小于 20，则会进一步收紧范围。未提供该值、该值大于 20 或设置 `clip_actions=false` 时，限制仍为 20；未提供标志时按 `true` 处理。即使标志为 false，也会校验已声明的值。非布尔类型的标志和无效的裁剪值会导致初始化失败。加载日志会报告实际裁剪范围，以及它来自上限还是检查点。该规则适用于所有元数据驱动的实时 WBT 模型。观测裁剪（`clip_observations`）以及独立的内嵌动作和运动控制策略仍遵循各自原有的约定。

便捷实时预设默认使用标准 G1 关节名称。对于其他 29 关节人形机器人，在构造 `Policy` 之前，应将 `config.robot.dof_names` 设置为其检查点中按顺序排列的 `dof_names`。库要求名称和顺序完全一致，并据此计算默认位置和锚点正向运动学（FK）。状态、参考、命令和增益数组均遵循相同顺序；通信层必须在接口边界对带名称的消息重新排序。
未显式提供经过时间时钟时，实时 API 每次调用推进一个模型时间步。每个控制周期前更新 `set_elapsed_seconds(seconds)`，可在周期丢失时保持动作相位；传入 `None` 可恢复按调用次数计时。`set_control_interval(seconds)` 设置经过时间时钟的换算方式。Python 只保留当前提供的参考，并返回独立的命令快照。`set_reference(None)` 会清除该参考。为兼容旧代码，仍保留原来的 `LiveWbtPolicy` 入口。

参考数据以时间偏移为主序排列，每帧有一个四元数。窗口维度必须与检查点匹配；格式错误或包含非有限值的输入会在推理前被拒绝。除非 `ReferenceWindow.current` 显式提供 `PolicyReferenceFrame`，否则第一个未来采样点也用作当前参考。默认的 0p2s 预设使用五个偏移量，但元数据完整的检查点可以声明其他布局。

`set_reference` 更改的是借用的缓冲区，不会开始新的 episode。在连续流式传输过程中替换窗口时，航向锚点、历史和动作时间均会保留。切换到起始航向不同的独立动作片段时，应在该片段的第一次 step 之前调用 `reset()`。这会清除历史和动作反馈，并在新 episode 的第一个状态重新记录机器人航向和参考航向，同时保留已设置的参考。仅靠指针是否相同无法识别片段边界，因为流式调用方可能在每个控制周期都替换缓冲区。

```python
policy.set_reference(next_clip_window)
policy.reset()
command = policy.step(state)
```

等价的 C++ 调用顺序为 `policy.set_reference(&next_clip_window); policy.reset(); policy.step(state);`。仅设置新指针会按设计继续使用旧的航向锚点。

<a id="c"></a>
## C++

```sh
cmake -S src/holosoma_inference/holosoma_cpp -B build/native \
  -DHOLOSOMA_BUILD_RUNNER=OFF -DHOLOSOMA_BUILD_TESTS=OFF
cmake --build build/native -j
cmake --install build/native --prefix "$HOME/.local"
```

使用方通过 `find_package(holosomaInference CONFIG REQUIRED)` 查找包，并链接 `holosoma::inference`。使用方无需机器人 SDK 或 YAML 头文件：

```cpp
#include <holosoma_cpp/api.hpp>

holosoma::Policy policy("preset.yaml", "checkpoint.onnx");
holosoma::RobotState state;  // fill joint positions, velocities and base orientation
const holosoma::JointCommand& command = policy.step(state);
```

预设中的相对模型路径以预设所在目录为基准解析；显式传入的检查点参数以调用方工作目录为基准解析。对于元数据完整的实时模型，使用 `Policy(live_policy_config(path, use_gpu, threads))` 构造策略；在执行 step 之前提供一个借用的 `ReferenceWindow`，并在其被替换之前保持对象有效。三类策略使用相同的 `num_dofs()`、`dof_names()`、`config()` 和用于直接访问底层策略的 `policy()` API。还导出了 `OfflineInterface`，用于影子模式集成。

`GeneralWBTPolicy` 适配元数据驱动的核心，不另外维护一套观测实现。它共享策略槽中的 ONNX 会话，包括 CUDA 和线程选项。`InferenceEngine` 是兼容性别名。配置工厂、打包观测和显式启用控制仍可作为组合扩展使用。每个实例只有一个所有者；调用必须串行化，包括对参考数据的修改。高层 Python 预设接口提供锁；直接使用 `Policy` 绑定时，推理期间会释放 GIL，需要由调用方保证串行访问。

<a id="command-line-compatibility"></a>
## 命令行兼容性

Python 入口 `python -m holosoma_inference.run_policy` 保留了 Tyro 参数、预设、次级策略、W&B 路径解析、SDK 插件和输入提供器。内置策略使用原生引擎。Python 通信层支持显式注入 `interface`、`velocity_input`、`command_provider` 和 `shared_hardware`，无需先构造硬件再将其替换掉。SDK 命令覆盖仍遵循关节顺序；电机排列转换由 SDK 负责。配置了仿真时钟时，仍使用 Python 时钟订阅器，而时间步换算由 C++ 引擎负责。

机器人专用适配器可以覆盖 `transform_command(command, state)`，调整最终按关节顺序排列的目标或增益，例如实现平滑的启动保持，而无需重新实现推理。形状无效、数值非有限或增益为负的命令会在传输前被拒绝。内嵌动作策略提供 `initial_motion`，用于此类初始化；原生模型切换后，适配器会收到模型切换钩子。

进行可编辑开发时，修改 C++ 源码后需要重新安装。容器的依赖构建阶段会在最后覆盖 Python 源码之前编译 C++ 源码，因此仅修改 Python 可以复用原生构建，而修改 C++ 会使该构建缓存失效。

旧版 Python 策略实现仅保留在 `holosoma_cpp/tests/python_reference` 中，作为独立的回放验证基准，不包含在安装包中。自定义策略应组合使用原生库，并将通信相关逻辑放在推理引擎之外。

<a id="verification"></a>
## 验证

```sh
uv pip install './src/holosoma_inference[dev]'
cmake -S src/holosoma_inference/holosoma_cpp -B build/native-tests \
  -DHOLOSOMA_WITH_UNITREE_SDK=OFF -DHOLOSOMA_BUILD_TESTS=ON \
  -DPython_EXECUTABLE="$(command -v python)"
cmake --build build/native-tests -j
ctest --test-dir build/native-tests --output-on-failure
python -m pytest --import-mode=importlib src/holosoma_inference/tests -q
```

测试覆盖两种模型约定、观测历史、航向与正向运动学、前瞻、时钟、重新加载失败、原生 Python 加载以及 SDK 边界的关节顺序。合成模型和记录的基准向量不能证明实体机器人运行稳定；应在目标机器人上单独评估所选检查点。
