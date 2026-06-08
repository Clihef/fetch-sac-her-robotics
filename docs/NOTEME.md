# Fetch 中四个任务实现笔记

## 说明

本文解释 Gymnasium-Robotics 中四个 Fetch 任务文件：

- `pick_and_place.py`
- `push.py`
- `reach.py`
- `slide.py`

同时解释它们如何通过 `gym.make(...)` 应用到当前 SAC + HER 训练脚本中，并说明 `fetch_env.py` 与 `robot_env.py` 的区别。

说明：

- 本文按源码中的实际行号解释。
- 各任务文件中大段三引号文档字符串和注释说明内容不逐句解释，只保留它们对代码理解有用的结论。
- 当前项目使用的是 `v4` 环境，即新 MuJoCo Python bindings，对应 `MujocoFetch...Env` 类；`MujocoPyFetch...Env` 是旧 `mujoco_py` 版本，主要用于历史兼容。

## 数据流向

核心调用链如下：

```text
train_fetch_sac_her.py
  -> gym.register_envs(gymnasium_robotics)
  -> gym.make(args.env_id)
  -> Gymnasium 根据 env_id 找到 entry_point
  -> 实例化 MujocoFetchReachEnv / MujocoFetchPushEnv / MujocoFetchSlideEnv / MujocoFetchPickAndPlaceEnv
  -> 这些任务类调用 MujocoFetchEnv.__init__(...)
  -> MujocoFetchEnv 继承 BaseFetchEnv + MujocoRobotEnv
  -> BaseRobotEnv.step(action) 推进 MuJoCo 仿真
  -> BaseFetchEnv._get_obs() 返回 observation / achieved_goal / desired_goal
  -> BaseFetchEnv.compute_reward(...) 为环境和 HER 重算 reward
  -> Stable-Baselines3 SAC 从 Dict observation 中学习策略
```

环境注册关系在：

`candidate-projects/Gymnasium-Robotics/gymnasium_robotics/__init__.py`

关键映射：

| env id                 | entry point                                                  | 实际类                       |
| ---------------------- | ------------------------------------------------------------ | ---------------------------- |
| `FetchReach-v4`        | `gymnasium_robotics.envs.fetch.reach:MujocoFetchReachEnv`    | `MujocoFetchReachEnv`        |
| `FetchPush-v4`         | `gymnasium_robotics.envs.fetch.push:MujocoFetchPushEnv`      | `MujocoFetchPushEnv`         |
| `FetchSlide-v4`        | `gymnasium_robotics.envs.fetch.slide:MujocoFetchSlideEnv`    | `MujocoFetchSlideEnv`        |
| `FetchPickAndPlace-v4` | `gymnasium_robotics.envs.fetch.pick_and_place:MujocoFetchPickAndPlaceEnv` | `MujocoFetchPickAndPlaceEnv` |

所以训练脚本中这一行：

```python
env = gym.make(env_id)
```

会根据 `env_id` 自动进入对应任务文件。例如：

```powershell
--env-id FetchPickAndPlace-v4
```

会实例化：

```python
MujocoFetchPickAndPlaceEnv
```

## 文件功能对比说明

| 文件                                                      | 抽象层级         | 主要职责                                                     |         是否只服务 Fetch | 与训练脚本的关系                         |
| --------------------------------------------------------- | ---------------- | ------------------------------------------------------------ | -----------------------: | ---------------------------------------- |
| `robot_env.py`                                            | 更底层           | MuJoCo XML 加载、仿真初始化、Gymnasium step/reset/render/action space/observation space | 否，也服务其他机器人环境 | `env.step` 和 `env.reset` 的底层主流程   |
| `fetch_env.py`                                            | Fetch 任务通用层 | Fetch 的动作转换、目标采样、reward、success、observation 组装、物体随机化 |      是，服务 Fetch 系列 | 决定 SAC 看到什么 obs、reward 和 success |
| `reach.py` / `push.py` / `slide.py` / `pick_and_place.py` | 具体任务配置层   | 选择 XML，设置 has_object、block_gripper、target_range 等任务参数 |             是，单个任务 | 由 `gym.make(env_id)` 实例化             |

更直观地说：

```text
robot_env.py
  解决：一个 MuJoCo 机器人环境如何 reset / step / render？

fetch_env.py
  解决：Fetch 机器人任务的 action / observation / goal / reward 怎么定义？

reach.py 等任务文件
  解决：这个具体任务是 Reach、Push、Slide 还是 PickAndPlace？
```

### 具体任务参数配置层——四个任务文件

四个任务文件的结构高度相似：

```text
import
MODEL_XML_PATH
MujocoFetchXXXEnv
  __init__
    initial_qpos
    MujocoFetchEnv.__init__(...)
    EzPickle.__init__(...)
MujocoPyFetchXXXEnv
  __init__
    initial_qpos
    MujocoPyFetchEnv.__init__(...)
    EzPickle.__init__(...)
```

任务文件本身不直接实现 `step`、`reset`、`compute_reward`、`_get_obs`。它们的主要作用是：

- 选择 MuJoCo XML 模型文件。
- 指定初始关节位置和物体初始位姿。
- 指定任务是否有物体。
- 指定夹爪是否锁死。
- 指定目标是否可以出现在空中。
- 指定物体和目标的采样范围。
- 指定成功距离阈值。
- 把这些配置传给 `MujocoFetchEnv`。

真正的环境通用逻辑在：

```text
fetch_env.py
robot_env.py
```

#### 四个任务关键参数对比

| 任务         | XML                  | has_object | block_gripper | target_in_the_air | gripper_extra_height | obj_range | target_range | target_offset     |
| ------------ | -------------------- | ---------: | ------------: | ----------------: | -------------------: | --------: | -----------: | ----------------- |
| Reach        | `reach.xml`          |      False |          True |              True |                  0.2 |      0.15 |         0.15 | 0.0               |
| Push         | `push.xml`           |       True |          True |             False |                  0.0 |      0.15 |         0.15 | 0.0               |
| Slide        | `slide.xml`          |       True |          True |             False |                -0.02 |       0.1 |          0.3 | `[0.4, 0.0, 0.0]` |
| PickAndPlace | `pick_and_place.xml` |       True |         False |              True |                  0.2 |      0.15 |         0.15 | 0.0               |

这些参数直接决定训练难度：

- `has_object=False` 时，`achieved_goal` 是夹爪位置，Reach 最简单。
- `has_object=True` 时，`achieved_goal` 是物体位置，策略必须间接控制物体。
- `block_gripper=True` 时，第四维动作会被置零，不能抓取。
- `block_gripper=False` 时，第四维动作控制夹爪开合，可以抓取，但探索更难。
- `target_in_the_air=True` 会让目标 z 坐标可能高于桌面，需要抬升。
- `target_offset` 和 `target_range` 越大，目标越远或变化越大，任务越难。

### Fetch 任务通用层——将参数变为动作、观测、奖励、采样

`fetch_env.py` 是 Fetch 系列任务的通用逻辑层。四个任务文件只传参数，真正把这些参数变成动作、观测、reward、目标采样的是这里。

#### 关键结构

```python
def get_base_fetch_env(RobotEnvClass):
```

这是一个工厂函数。它接收一个机器人基类：

- `MujocoRobotEnv`

然后动态生成一个 `BaseFetchEnv` 类。

这样做的原因是：Fetch 任务逻辑相同，但底层 MuJoCo 绑定可以不同。

#### `BaseFetchEnv.__init__`

`BaseFetchEnv.__init__` 接收四个任务文件传入的参数：

```python
gripper_extra_height
block_gripper
has_object
target_in_the_air
target_offset
obj_range
target_range
distance_threshold
reward_type
```

它把这些参数保存到 `self` 上，后续 `_set_action`、`_get_obs`、`_sample_goal`、`compute_reward` 都会使用它们。

最后：

```python
super().__init__(n_actions=4, **kwargs)
```

这会进入 `robot_env.py` 中的 `BaseRobotEnv.__init__`，创建 action space、observation space，并初始化 MuJoCo 仿真。

#### `compute_reward`

```python
d = goal_distance(achieved_goal, goal)
if self.reward_type == "sparse":
    return -(d > self.distance_threshold).astype(np.float32)
else:
    return -d
```

含义：

- 先计算 achieved goal 与 desired goal 的距离。
- sparse reward 下，距离大于阈值返回 `-1`，否则返回 `0`。
- dense reward 下，直接返回负距离。

这也是 HER 能工作的关键。HER 会把 replay buffer 中某些未来 achieved goal 当成新的 desired goal，然后调用环境的 `compute_reward` 重算 reward。

#### `_set_action`

```python
assert action.shape == (4,)
action = action.copy()
pos_ctrl, gripper_ctrl = action[:3], action[3]
pos_ctrl *= 0.05
rot_ctrl = [1.0, 0.0, 1.0, 0.0]
gripper_ctrl = np.array([gripper_ctrl, gripper_ctrl])
if self.block_gripper:
    gripper_ctrl = np.zeros_like(gripper_ctrl)
action = np.concatenate([pos_ctrl, rot_ctrl, gripper_ctrl])
return action
```

训练脚本中的 SAC 输出 4 维动作：

```text
[dx, dy, dz, gripper]
```

`_set_action` 会把它转换成 MuJoCo 控制向量：

- 前 3 维控制末端执行器位置变化。
- 乘以 `0.05` 限制单步移动幅度。
- 姿态固定为四元数 `[1, 0, 1, 0]`。
- 第 4 维复制成左右两个夹爪控制量。
- 如果 `block_gripper=True`，夹爪控制被置零。

所以：

- Reach / Push / Slide 的第 4 维动作基本无效。
- PickAndPlace 的第 4 维动作有效。

#### `_get_obs`

`_get_obs` 调用：

```python
generate_mujoco_observations()
```

拿到：

```text
grip_pos
object_pos
object_rel_pos
gripper_state
object_rot
object_velp
object_velr
grip_velp
gripper_vel
```

然后根据 `has_object` 决定 `achieved_goal`：

```python
if not self.has_object:
    achieved_goal = grip_pos.copy()
else:
    achieved_goal = np.squeeze(object_pos.copy())
```

这就是 Reach 与另外三个任务最核心的数据差异：

- Reach：目标是末端执行器位置。
- Push / Slide / PickAndPlace：目标是物体位置。

最后返回 GoalEnv 格式：

```python
{
    "observation": obs.copy(),
    "achieved_goal": achieved_goal.copy(),
    "desired_goal": self.goal.copy(),
}
```

这正是训练脚本里使用 `MultiInputPolicy` 的原因：普通 MLP policy 不能直接处理 Dict observation，`MultiInputPolicy` 可以处理这种字典输入。

#### `_sample_goal`

如果有物体：

```python
goal = self.initial_gripper_xpos[:3] + random_offset
goal += self.target_offset
goal[2] = self.height_offset
if self.target_in_the_air and random < 0.5:
    goal[2] += random_height
```

如果没有物体：

```python
goal = self.initial_gripper_xpos[:3] + random_offset
```

含义：

- Reach 没有物体，目标围绕夹爪初始位置采样。
- Push / Slide / PickAndPlace 有物体，目标通常围绕夹爪初始位置采样，并把 z 高度设为物体桌面高度。
- PickAndPlace 因为 `target_in_the_air=True`，一部分目标会升到空中。
- Slide 因为 `target_offset=[0.4, 0, 0]`，目标会整体偏远。

#### `_is_success`

```python
d = goal_distance(achieved_goal, desired_goal)
return (d < self.distance_threshold).astype(np.float32)
```

它返回 `0.0` 或 `1.0`，训练评估中的 `success_rate` 就来自 `info["is_success"]`。

#### `MujocoFetchEnv`

`MujocoFetchEnv` 是当前 v4 使用的类，继承：

```python
get_base_fetch_env(MujocoRobotEnv)
```

也就是说：

- Fetch 任务逻辑来自 `BaseFetchEnv`
- MuJoCo 新版底层仿真来自 `MujocoRobotEnv`

它实现了：

- `_step_callback`：夹爪锁死时强制左右夹爪 joint 为 0。
- `_set_action`：把 BaseFetchEnv 转换后的 action 传给 MuJoCo 控制器和 mocap。
- `generate_mujoco_observations`：从 MuJoCo model/data 中读取夹爪、物体、速度、旋转等状态。
- `_get_gripper_xpos`：读取夹爪位置。
- `_render_callback`：把可视化目标 site 移到当前 goal 位置。
- `_reset_sim`：重置仿真，并随机化物体初始位置。
- `_env_setup`：设置初始关节，移动末端执行器到初始位置，并记录初始夹爪位置和物体高度。

### robot 机器人底层

`robot_env.py` 是更底层的机器人环境基类。它不只服务 Fetch，也服务 Hand 等机器人任务。

它负责：

- 加载 MuJoCo XML 文件。
- 初始化 MuJoCo `model` 和 `data`。
- 定义 Gymnasium action space。
- 根据 `_get_obs()` 推断 observation space。
- 实现 `step(action)` 的标准 Gymnasium 流程。
- 实现 `reset(seed=...)` 的标准流程。
- 实现 render / close / dt / MuJoCo step。

#### `BaseRobotEnv.__init__`

它接收：

```python
model_path
initial_qpos
n_actions
n_substeps
render_mode
width
height
```

关键步骤：

1. 拼出 XML 完整路径。
2. 检查 XML 文件是否存在。
3. 保存 `n_substeps`、`initial_qpos`、渲染宽高。
4. 调用 `_initialize_simulation()` 初始化 MuJoCo。
5. 调用 `_get_obs()` 获取一个初始 observation。
6. 创建 action space：

```python
spaces.Box(-1.0, 1.0, shape=(n_actions,), dtype="float32")
```

Fetch 中 `n_actions=4`，所以动作空间是 4 维连续控制。

7. 创建 observation space：

```python
spaces.Dict(
    desired_goal=Box(...),
    achieved_goal=Box(...),
    observation=Box(...)
)
```

这就是 SAC 训练脚本必须使用：

```python
"MultiInputPolicy"
```

的原因。

#### `BaseRobotEnv.step`

`step(action)` 是训练时最重要的函数。

流程：

1. 检查动作维度是否匹配。
2. 把动作 clip 到 action space 范围。
3. 调用 `_set_action(action)`，由 Fetch 层把 4 维动作转换为 MuJoCo 控制。
4. 调用 `_mujoco_step(action)` 推进 MuJoCo 仿真。
5. 调用 `_step_callback()`，例如夹爪锁死。
6. 如果是 human render mode，则渲染。
7. 调用 `_get_obs()` 获取新 observation。
8. 调用 `_is_success(...)` 写入 `info["is_success"]`。
9. 调用 `compute_terminated(...)`。
10. 调用 `compute_truncated(...)`。
11. 调用 `compute_reward(...)`。
12. 返回：

```python
obs, reward, terminated, truncated, info
```

训练脚本中：

```python
model.learn(...)
```

内部反复调用的就是这个 `env.step(action)`。

#### `BaseRobotEnv.reset`

`reset` 流程：

1. 设置随机种子。
2. 循环调用 `_reset_sim()`，直到仿真重置成功。
3. 调用 `_sample_goal()` 采样新目标。
4. 调用 `_get_obs()` 返回初始 observation。
5. 返回：

```python
obs, {}
```

训练脚本中：

```python
env.reset(seed=seed)
```

最终进入这个流程。

#### `MujocoRobotEnv`

这是 v4 Fetch 使用的新 MuJoCo 版本。

关键点：

- 如果没有安装 `mujoco`，会抛出依赖错误。
- 用 `mujoco.MjModel.from_xml_path(...)` 加载 XML。
- 用 `mujoco.MjData(self.model)` 创建仿真数据。
- 用 `MujocoRenderer` 做渲染。
- `_mujoco_step` 调用：

```python
mujoco.mj_step(self.model, self.data, nstep=self.n_substeps)
```

这就是 MuJoCo 在项目里的实际作用：每个环境 step 内部都由 MuJoCo 推进物理仿真。

#### `MujocoPyRobotEnv`

这是旧版 `mujoco_py` 兼容类，对应 v1 环境。

当前训练使用的是 v4，所以不走这里。

## MDP建模梳理

```text
fetch_env.py 负责 Fetch 任务 MDP 的主要建模；
robot_env.py 负责 Gymnasium/MuJoCo 环境运行框架；
```

更具体地说：

```text
MDP = (S, A, P, R, gamma)
```

在 Fetch 系列里，各部分大致对应：

| MDP 部分        | 主要发生位置                                        | 说明                                                         |
| --------------- | --------------------------------------------------- | ------------------------------------------------------------ |
| 状态 / 观测 `S` | `fetch_env.py`                                      | `_get_obs()` 定义 `observation / achieved_goal / desired_goal` |
| 动作 `A`        | `robot_env.py` + `fetch_env.py`                     | `robot_env.py` 创建 4 维 Box action space；`fetch_env.py` 解释 4 维动作含义 |
| 状态转移 `P`    | `robot_env.py` + MuJoCo XML + MuJoCo 引擎           | `robot_env.py.step()` 推进仿真；实际物理转移由 MuJoCo 和 XML 模型决定 |
| 奖励 `R`        | `fetch_env.py`                                      | `compute_reward()` 定义 sparse/dense reward                  |
| 初始状态分布    | `fetch_env.py` + 任务文件                           | `_reset_sim()` 随机化物体初始位置；任务文件给 `initial_qpos / obj_range` |
| 目标分布        | `fetch_env.py` + 任务文件                           | `_sample_goal()` 采样目标；任务文件给 `target_range / target_offset / target_in_the_air` |
| 终止 / 截断     | `robot_env.py` + Gymnasium TimeLimit                | Fetch 自身 continuing task 不主动 terminated；默认 50 step 被 TimeLimit 截断 |
| 具体任务差异    | `reach.py / push.py / slide.py / pick_and_place.py` | 通过 `has_object / block_gripper / target_in_the_air / target_offset` 等参数改变 MDP |

###  状态转移 `P` 到底在哪里？

MDP 里最难说的是状态转移：

```text
P(s' | s, a)
```

在 Fetch 里，它不是纯 Python 手写的，而是由三部分共同决定：

```text
robot_env.py 的 step 框架
+ fetch_env.py 的动作映射
+ MuJoCo XML 物理模型和 MuJoCo 引擎
```

具体路径：

```text
BaseRobotEnv.step(action)
  -> fetch_env.py::_set_action(action)
  -> MujocoFetchEnv._set_action(action)
  -> mujoco_utils.ctrl_set_action(...)
  -> mujoco_utils.mocap_set_action(...)
  -> robot_env.py::_mujoco_step(action)
  -> mujoco.mj_step(...)
```

所以状态转移建模应该这样说：

> 状态转移框架在 `robot_env.py` 的 `step()` 中，动作到 MuJoCo 控制的映射在 `fetch_env.py` 中，真实物理转移由 MuJoCo XML 模型和 MuJoCo 引擎计算。

更准确是：

> 这里的 MDP transition 是仿真型 MDP，不显式写状态转移概率，而是由 MuJoCo 物理仿真器根据 XML 模型、当前状态和控制输入数值积分得到下一个状态。

### 训练脚本如何对应？

```python
env = gym.make(env_id)
```

如果：

```powershell
--env-id FetchPickAndPlace-v4
```

就会实例化：

```python
MujocoFetchPickAndPlaceEnv
```

然后该类传入：

```python
has_object=True
block_gripper=False
target_in_the_air=True
```

进入 `fetch_env.py`。

SAC 训练时：

```python
model.learn(...)
```

内部反复做：

```text
obs -> policy -> action -> env.step(action) -> next_obs, reward, done, info
```

其中：

- `action` 的 4 维含义来自 `fetch_env.py::_set_action`
- `next_obs` 来自 `fetch_env.py::_get_obs`
- `reward` 来自 `fetch_env.py::compute_reward`
- `success_rate` 来自 `fetch_env.py::_is_success`
- 物理仿真推进来自 `robot_env.py::_mujoco_step` 和 MuJoCo

HER 训练时：

```python
HerReplayBuffer
```

会使用：

```text
achieved_goal
desired_goal
compute_reward
```

这些都来自 `fetch_env.py` 定义的 GoalEnv 结构。

### 一句话结论

> `fetch_env.py` 定义 Fetch MDP 的状态/观测、动作语义、目标采样、reward 和 success；`robot_env.py` 定义 Gymnasium + MuJoCo 的通用 step/reset 框架和仿真推进；四个任务文件通过参数和 XML 模型把通用 Fetch MDP 实例化为 Reach、Push、Slide、PickAndPlace。

## 总结

来自gymnasium_robotics：

- 四个任务文件决定“做什么任务”
- `fetch_env.py` 决定“Fetch 任务如何计算动作、观测、目标和奖励”，相当于MDP建模，定义了观测空间、动作空间、奖励
- `robot_env.py` 决定“MuJoCo 机器人环境如何 reset/step/render”，相当于根据MDP的动作输出机器人控制指令

编写脚本：

- 编写的训练脚本负责接入这些环境 到 由SB3调用的 SAC + HER，让策略通过与 MuJoCo 仿真的交互学习连续控制行为。
