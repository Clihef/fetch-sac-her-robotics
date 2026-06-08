# Fetch 四个任务源码逐行解释与训练代码关联

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

## 一、整体调用链

当前训练脚本是：

`scripts/train_fetch_sac_her.py`

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

| env id | entry point | 实际类 |
|---|---|---|
| `FetchReach-v4` | `gymnasium_robotics.envs.fetch.reach:MujocoFetchReachEnv` | `MujocoFetchReachEnv` |
| `FetchPush-v4` | `gymnasium_robotics.envs.fetch.push:MujocoFetchPushEnv` | `MujocoFetchPushEnv` |
| `FetchSlide-v4` | `gymnasium_robotics.envs.fetch.slide:MujocoFetchSlideEnv` | `MujocoFetchSlideEnv` |
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

## 二、四个任务文件共同结构

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

## 三、`reach.py` 逐行解释

文件路径：

`candidate-projects/Gymnasium-Robotics/gymnasium_robotics/envs/fetch/reach.py`

### 有效代码逐行解释

| 行号 | 代码含义 |
|---:|---|
| 1 | `import os`，用于拼接 XML 模型路径，保证 Windows / Linux 路径分隔符兼容。 |
| 3 | 导入 `EzPickle`，用于 Gymnasium 环境序列化和复现环境构造参数。 |
| 5 | 从 Fetch 包导入新 MuJoCo 版本 `MujocoFetchEnv` 和旧 `mujoco_py` 版本 `MujocoPyFetchEnv`。 |
| 8 | `MODEL_XML_PATH = os.path.join("fetch", "reach.xml")`，指定 Reach 任务使用 `assets/fetch/reach.xml` 这个 MuJoCo 场景模型。 |
| 11 | 定义 `MujocoFetchReachEnv`，这是 `FetchReach-v4` 实际注册和使用的环境类。它继承 `MujocoFetchEnv` 和 `EzPickle`。 |
| 12-123 | 三引号文档字符串，说明任务、动作空间、观测空间、reward、初始状态和版本历史。本文按要求省略其逐句解释。 |
| 125 | 定义构造函数，默认 `reward_type="sparse"`。如果注册的是 `FetchReachDense-v4`，注册逻辑会传入 `reward_type="dense"`。 |
| 126 | 创建 `initial_qpos` 字典，用于设置机器人初始关节位置。 |
| 127 | 设置 `robot0:slide0`，对应 Fetch 机器人底座滑轨/基座相关初始位置。 |
| 128 | 设置 `robot0:slide1`。 |
| 129 | 设置 `robot0:slide2`。 |
| 130 | 结束 `initial_qpos` 字典。Reach 没有物体，因此没有 `object0:joint`。 |
| 131 | 调用 `MujocoFetchEnv.__init__`，把 Reach 的任务配置交给 Fetch 通用环境。 |
| 132 | 传入 `self`。 |
| 133 | `model_path=MODEL_XML_PATH`，加载 `fetch/reach.xml`。 |
| 134 | `has_object=False`，Reach 任务没有物块，目标是末端执行器位置。 |
| 135 | `block_gripper=True`，夹爪锁死。Reach 不需要开合夹爪。 |
| 136 | `n_substeps=20`，每个 Gym step 内部执行 20 个 MuJoCo 子步。 |
| 137 | `gripper_extra_height=0.2`，初始化时末端执行器相对桌面额外抬高。 |
| 138 | `target_in_the_air=True`，目标点可以在三维空间中，包括空中。 |
| 139 | `target_offset=0.0`，目标采样后不额外平移。 |
| 140 | `obj_range=0.15`，虽然 Reach 没有物体，但通用构造参数仍然保留该字段。 |
| 141 | `target_range=0.15`，目标点围绕初始夹爪位置在一定范围内随机采样。 |
| 142 | `distance_threshold=0.05`，末端执行器距离目标小于 5cm 视为成功。 |
| 143 | `initial_qpos=initial_qpos`，传入初始关节配置。 |
| 144 | `reward_type=reward_type`，传入 sparse 或 dense 奖励类型。 |
| 145 | `**kwargs`，接收 Gymnasium 额外参数，例如 `render_mode`、`max_episode_steps` 包装参数等。 |
| 146 | 结束 `MujocoFetchEnv.__init__` 调用。 |
| 147 | 调用 `EzPickle.__init__`，记录构造参数，便于环境被 pickle。 |
| 150 | 定义旧版 `MujocoPyFetchReachEnv`，对应 `FetchReach-v1`，不是当前训练的 v4。 |
| 151-173 | 旧版环境构造逻辑与 v4 基本一致，只是父类换成 `MujocoPyFetchEnv`。当前训练不用它。 |

### Reach 任务本质

Reach 是四个任务里最简单的：

- `has_object=False`
- `achieved_goal = grip_pos`
- `desired_goal = 目标末端位置`
- 只要求机械臂末端到达目标点
- 不涉及接触、推动、抓取

这解释了为什么 Reach 在短训练中更容易成功。

## 四、`push.py` 逐行解释

文件路径：

`candidate-projects/Gymnasium-Robotics/gymnasium_robotics/envs/fetch/push.py`

### 有效代码逐行解释

| 行号 | 代码含义 |
|---:|---|
| 1 | 导入 `os`，用于拼接 XML 模型路径。 |
| 3 | 导入 `EzPickle`。 |
| 5 | 导入新 MuJoCo 版本和旧 `mujoco_py` 版本的 Fetch 通用环境类。 |
| 8 | `MODEL_XML_PATH = os.path.join("fetch", "push.xml")`，指定 Push 任务使用 `fetch/push.xml`。 |
| 11 | 定义旧版 `MujocoPyFetchPushEnv`，对应 `FetchPush-v1`。 |
| 12 | 旧版 Push 的构造函数。 |
| 13 | 创建初始关节位置字典。 |
| 14-16 | 设置机器人初始位置。 |
| 17 | 设置 `object0:joint`，即物块的初始位置和四元数姿态。Push 有物体，所以需要这一项。 |
| 18 | 结束字典。 |
| 19-35 | 调用旧版 `MujocoPyFetchEnv.__init__` 并初始化 pickle。当前 v4 训练不用它。 |
| 38 | 定义 `MujocoFetchPushEnv`，这是 `FetchPush-v4` 实际使用的类。 |
| 39-165 | 三引号文档字符串，本文省略逐句解释。 |
| 167 | 定义 v4 Push 构造函数，默认 sparse reward。 |
| 168 | 创建 `initial_qpos`。 |
| 169 | 设置 `robot0:slide0` 初始位置。 |
| 170 | 设置 `robot0:slide1` 初始位置。 |
| 171 | 设置 `robot0:slide2` 初始位置。 |
| 172 | 设置物体 `object0:joint` 初始位置和姿态。列表前 3 个值是物体位置，后 4 个值是四元数。 |
| 173 | 结束 `initial_qpos`。 |
| 174 | 调用 `MujocoFetchEnv.__init__`。 |
| 175 | 传入 `self`。 |
| 176 | `model_path=MODEL_XML_PATH`，加载 `fetch/push.xml`。 |
| 177 | `has_object=True`，Push 任务有物块。 |
| 178 | `block_gripper=True`，夹爪锁死，不允许用夹取完成任务，只能推动。 |
| 179 | `n_substeps=20`，每个 Gym step 推进 20 个 MuJoCo 子步。 |
| 180 | `gripper_extra_height=0.0`，初始化时夹爪不额外抬高，便于接触桌面物块。 |
| 181 | `target_in_the_air=False`，目标只在桌面上，不在空中。 |
| 182 | `target_offset=0.0`，目标不额外整体偏移。 |
| 183 | `obj_range=0.15`，物体初始位置在夹爪附近一定范围内随机采样。 |
| 184 | `target_range=0.15`，目标位置采样范围。 |
| 185 | `distance_threshold=0.05`，物块距离目标小于 5cm 视为成功。 |
| 186 | 传入初始关节和物体配置。 |
| 187 | 传入奖励类型。 |
| 188 | 传入额外参数。 |
| 189 | 结束 `MujocoFetchEnv.__init__` 调用。 |
| 190 | 调用 `EzPickle.__init__`。 |

### Push 任务本质

Push 与 Reach 的关键区别：

- `has_object=True`
- `achieved_goal = object_pos`
- `desired_goal = 目标物块位置`
- `block_gripper=True`
- `target_in_the_air=False`

也就是说，策略不能直接把目标点当末端执行器目标，而必须先移动夹爪去接触物块，再通过接触动力学改变物块位置。

## 五、`slide.py` 逐行解释

文件路径：

`candidate-projects/Gymnasium-Robotics/gymnasium_robotics/envs/fetch/slide.py`

### 有效代码逐行解释

| 行号 | 代码含义 |
|---:|---|
| 1 | 导入 `os`。 |
| 3 | 导入 `numpy as np`，Slide 任务需要用 `np.array([0.4, 0.0, 0.0])` 设置目标偏移。 |
| 4 | 导入 `EzPickle`。 |
| 6 | 导入 `MujocoFetchEnv` 和 `MujocoPyFetchEnv`。 |
| 9 | `MODEL_XML_PATH = os.path.join("fetch", "slide.xml")`，指定 Slide 使用 `fetch/slide.xml`。 |
| 12 | 定义旧版 `MujocoPyFetchSlideEnv`。 |
| 13 | 旧版构造函数。 |
| 14 | 创建初始关节/物体位置字典。 |
| 15 | 设置 `robot0:slide0=0.05`，Slide 的机器人初始位置明显不同。 |
| 16 | 设置 `robot0:slide1=0.48`。 |
| 17 | 设置 `robot0:slide2=0.0`。 |
| 18 | 设置物体 `object0:joint` 初始位置为 `[1.7, 1.1, 0.41, ...]`，位置比 Push 更远，符合滑动任务长桌场景。 |
| 19 | 结束字典。 |
| 20-36 | 旧版 `MujocoPyFetchEnv` 初始化逻辑，当前 v4 训练不用。 |
| 39 | 定义 `MujocoFetchSlideEnv`，这是 `FetchSlide-v4` 使用的类。 |
| 40-164 | 三引号文档字符串，本文省略逐句解释。 |
| 166 | 定义 v4 Slide 构造函数。 |
| 167 | 创建初始状态字典。 |
| 168 | 设置机器人 `slide0` 初始值为 `0.05`。 |
| 169 | 设置机器人 `slide1` 初始值。 |
| 170 | 设置机器人 `slide2` 初始值。 |
| 171 | 设置物体初始位置和姿态。Slide 中物体更像 puck，需要被击打后滑行。 |
| 172 | 结束 `initial_qpos`。 |
| 173 | 调用 `MujocoFetchEnv.__init__`。 |
| 174 | 传入 `self`。 |
| 175 | `model_path=MODEL_XML_PATH`，加载 `fetch/slide.xml`。 |
| 176 | `has_object=True`，Slide 有滑块/圆盘。 |
| 177 | `block_gripper=True`，夹爪锁死，不能抓取。 |
| 178 | `n_substeps=20`。 |
| 179 | `gripper_extra_height=-0.02`，末端执行器初始化更低，更贴近滑块，利于击打。 |
| 180 | `target_in_the_air=False`，目标在桌面上。 |
| 181 | `target_offset=np.array([0.4, 0.0, 0.0])`，目标整体向 x 方向偏移 0.4m，使目标更远，强化“滑过去”的任务属性。 |
| 182 | `obj_range=0.1`，物体初始随机范围比 Push 小。 |
| 183 | `target_range=0.3`，目标随机范围更大。 |
| 184 | `distance_threshold=0.05`，5cm 内成功。 |
| 185 | 传入初始状态。 |
| 186 | 传入奖励类型。 |
| 187 | 传入额外参数。 |
| 188 | 结束父类初始化调用。 |
| 189 | 调用 `EzPickle.__init__`。 |

### Slide 任务本质

Slide 比 Push 更难，关键在：

- 目标更远：`target_offset=[0.4, 0, 0]`
- 目标采样范围更大：`target_range=0.3`
- 物体需要靠速度和摩擦滑行，而不是一直推着走
- 策略需要学会击打方向和力度

因此 10k 或几十 k 步内 success rate 波动较大是正常现象。

## 六、`pick_and_place.py` 逐行解释

文件路径：

`candidate-projects/Gymnasium-Robotics/gymnasium_robotics/envs/fetch/pick_and_place.py`

### 有效代码逐行解释

| 行号 | 代码含义 |
|---:|---|
| 1 | 导入 `os`。 |
| 3 | 导入 `EzPickle`。 |
| 5 | 导入 `MujocoFetchEnv` 和 `MujocoPyFetchEnv`。 |
| 7 | `MODEL_XML_PATH = os.path.join("fetch", "pick_and_place.xml")`，指定 PickAndPlace 使用的 MuJoCo XML。 |
| 10 | 定义 `MujocoFetchPickAndPlaceEnv`，这是 `FetchPickAndPlace-v4` 实际使用的类。 |
| 11-137 | 三引号文档字符串，本文省略逐句解释。 |
| 139 | 定义构造函数，默认 sparse reward。 |
| 140 | 创建 `initial_qpos` 字典。 |
| 141 | 设置机器人 `slide0` 初始位置。 |
| 142 | 设置机器人 `slide1` 初始位置。 |
| 143 | 设置机器人 `slide2` 初始位置。 |
| 144 | 设置物体 `object0:joint` 初始位置和姿态。PickAndPlace 有物块。 |
| 145 | 结束字典。 |
| 146 | 调用 `MujocoFetchEnv.__init__`。 |
| 147 | 传入 `self`。 |
| 148 | `model_path=MODEL_XML_PATH`，加载 `fetch/pick_and_place.xml`。 |
| 149 | `has_object=True`，任务中有物块。 |
| 150 | `block_gripper=False`，夹爪不锁死，可以开合，这是 PickAndPlace 与 Push/Slide 的核心区别。 |
| 151 | `n_substeps=20`。 |
| 152 | `gripper_extra_height=0.2`，初始化时夹爪抬高，便于从上方接近物体。 |
| 153 | `target_in_the_air=True`，目标可以在空中，因此任务可能要求把物体抓起并抬升。 |
| 154 | `target_offset=0.0`。 |
| 155 | `obj_range=0.15`，物体初始位置采样范围。 |
| 156 | `target_range=0.15`，目标采样范围。 |
| 157 | `distance_threshold=0.05`，物块距离目标小于 5cm 视为成功。 |
| 158 | 传入初始状态。 |
| 159 | 传入奖励类型。 |
| 160 | 传入额外参数。 |
| 161 | 结束父类初始化调用。 |
| 162 | 调用 `EzPickle.__init__`。 |
| 165 | 定义旧版 `MujocoPyFetchPickAndPlaceEnv`。 |
| 166-189 | 旧版构造逻辑与 v4 基本相同，只是父类换为 `MujocoPyFetchEnv`。当前训练不用。 |

### PickAndPlace 任务本质

PickAndPlace 的核心难点来自：

- `has_object=True`
- `block_gripper=False`
- `target_in_the_air=True`
- 第四维动作会影响夹爪开合
- 策略需要学会靠近、闭合夹爪、抓取、抬升、移动、放置

这比 Push 和 Slide 的动作链更长，也比 Reach 难得多。

## 七、四个任务关键参数对比

| 任务 | XML | has_object | block_gripper | target_in_the_air | gripper_extra_height | obj_range | target_range | target_offset |
|---|---|---:|---:|---:|---:|---:|---:|---|
| Reach | `reach.xml` | False | True | True | 0.2 | 0.15 | 0.15 | 0.0 |
| Push | `push.xml` | True | True | False | 0.0 | 0.15 | 0.15 | 0.0 |
| Slide | `slide.xml` | True | True | False | -0.02 | 0.1 | 0.3 | `[0.4, 0.0, 0.0]` |
| PickAndPlace | `pick_and_place.xml` | True | False | True | 0.2 | 0.15 | 0.15 | 0.0 |

这些参数直接决定训练难度：

- `has_object=False` 时，`achieved_goal` 是夹爪位置，Reach 最简单。
- `has_object=True` 时，`achieved_goal` 是物体位置，策略必须间接控制物体。
- `block_gripper=True` 时，第四维动作会被置零，不能抓取。
- `block_gripper=False` 时，第四维动作控制夹爪开合，可以抓取，但探索更难。
- `target_in_the_air=True` 会让目标 z 坐标可能高于桌面，需要抬升。
- `target_offset` 和 `target_range` 越大，目标越远或变化越大，任务越难。

## 八、`fetch_env.py` 的作用

文件路径：

`candidate-projects/Gymnasium-Robotics/gymnasium_robotics/envs/fetch/fetch_env.py`

`fetch_env.py` 是 Fetch 系列任务的通用逻辑层。四个任务文件只传参数，真正把这些参数变成动作、观测、reward、目标采样的是这里。

### 关键结构

```python
def goal_distance(goal_a, goal_b):
    assert goal_a.shape == goal_b.shape
    return np.linalg.norm(goal_a - goal_b, axis=-1)
```

它计算 achieved goal 和 desired goal 的欧氏距离。这个距离同时用于：

- sparse reward 判断是否成功
- dense reward 返回负距离
- `_is_success` 成功率统计

```python
def get_base_fetch_env(RobotEnvClass):
```

这是一个工厂函数。它接收一个机器人基类：

- `MujocoRobotEnv`
- 或 `MujocoPyRobotEnv`

然后动态生成一个 `BaseFetchEnv` 类。

这样做的原因是：Fetch 任务逻辑相同，但底层 MuJoCo 绑定可以不同。

### `BaseFetchEnv.__init__`

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

### `compute_reward`

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

### `_set_action`

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

### `_get_obs`

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

### `_sample_goal`

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

### `_is_success`

```python
d = goal_distance(achieved_goal, desired_goal)
return (d < self.distance_threshold).astype(np.float32)
```

它返回 `0.0` 或 `1.0`，训练评估中的 `success_rate` 就来自 `info["is_success"]`。

### `MujocoFetchEnv`

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

## 九、`robot_env.py` 的作用

文件路径：

`candidate-projects/Gymnasium-Robotics/gymnasium_robotics/envs/robot_env.py`

`robot_env.py` 是更底层的机器人环境基类。它不只服务 Fetch，也服务 Hand 等机器人任务。

它负责：

- 加载 MuJoCo XML 文件。
- 初始化 MuJoCo `model` 和 `data`。
- 定义 Gymnasium action space。
- 根据 `_get_obs()` 推断 observation space。
- 实现 `step(action)` 的标准 Gymnasium 流程。
- 实现 `reset(seed=...)` 的标准流程。
- 实现 render / close / dt / MuJoCo step。

### `BaseRobotEnv.__init__`

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

### `BaseRobotEnv.step`

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

### `BaseRobotEnv.reset`

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

### `MujocoRobotEnv`

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

### `MujocoPyRobotEnv`

这是旧版 `mujoco_py` 兼容类，对应 v1 环境。

当前训练使用的是 v4，所以不走这里。

## 十、`fetch_env.py` 和 `robot_env.py` 的区别

| 文件 | 抽象层级 | 主要职责 | 是否只服务 Fetch | 与训练脚本的关系 |
|---|---|---|---:|---|
| `robot_env.py` | 更底层 | MuJoCo XML 加载、仿真初始化、Gymnasium step/reset/render/action space/observation space | 否，也服务其他机器人环境 | `env.step` 和 `env.reset` 的底层主流程 |
| `fetch_env.py` | Fetch 任务通用层 | Fetch 的动作转换、目标采样、reward、success、observation 组装、物体随机化 | 是，服务 Fetch 系列 | 决定 SAC 看到什么 obs、reward 和 success |
| `reach.py` / `push.py` / `slide.py` / `pick_and_place.py` | 具体任务配置层 | 选择 XML，设置 has_object、block_gripper、target_range 等任务参数 | 是，单个任务 | 由 `gym.make(env_id)` 实例化 |

更直观地说：

```text
robot_env.py
  解决：一个 MuJoCo 机器人环境如何 reset / step / render？

fetch_env.py
  解决：Fetch 机器人任务的 action / observation / goal / reward 怎么定义？

reach.py 等任务文件
  解决：这个具体任务是 Reach、Push、Slide 还是 PickAndPlace？
```

## 十一、这些代码如何应用到训练脚本

训练脚本中：

```python
gym.register_envs(gymnasium_robotics)
env = gym.make(env_id)
return Monitor(env)
```

这里发生了三件事：

1. `gym.register_envs(gymnasium_robotics)` 调用 Gymnasium-Robotics 的注册逻辑。
2. `gym.make(env_id)` 根据 `FetchPush-v4` 等 ID 找到对应 entry point。
3. 实例化对应任务类，并套上 Gymnasium 的 `TimeLimit` wrapper，默认 `max_episode_steps=50`。

然后训练脚本创建 SAC：

```python
model = SAC(
    "MultiInputPolicy",
    env,
    replay_buffer_class=HerReplayBuffer,
    replay_buffer_kwargs={
        "n_sampled_goal": 4,
        "goal_selection_strategy": "future",
    },
    ...
)
```

这里和环境源码的关系是：

- `MultiInputPolicy` 对应 `BaseRobotEnv` 创建的 `spaces.Dict` observation space。
- `HerReplayBuffer` 依赖 GoalEnv 的三个键：
  - `observation`
  - `achieved_goal`
  - `desired_goal`
- HER 重标目标时依赖环境的 `compute_reward(...)`。
- SAC 每次输出 4 维连续动作，传入 `BaseRobotEnv.step(action)`。
- `BaseRobotEnv.step` 再调用 Fetch 层 `_set_action`，把动作变成 MuJoCo 控制。
- MuJoCo 推进仿真后，Fetch 层 `_get_obs` 返回新的 GoalEnv observation。
- reward 和 success 由 `compute_reward` / `_is_success` 基于 achieved goal 与 desired goal 的距离计算。

评估函数中：

```python
final_success = float(info.get("is_success", 0.0))
```

这个 `info["is_success"]` 来自：

```python
BaseRobotEnv.step
  -> self._is_success(obs["achieved_goal"], self.goal)
```

而 `_is_success` 定义在 `fetch_env.py` 的 `BaseFetchEnv` 中。

## 十二、面试讲解口径

可以这样讲：

> 四个 Fetch 任务文件本身不是算法实现，而是任务配置文件。它们分别指定不同 XML 模型、是否有物体、夹爪是否锁定、目标是否在空中、目标和物体采样范围等参数。真正的通用 Fetch 逻辑在 `fetch_env.py`，包括 4 维动作如何映射到 MuJoCo 控制、observation / achieved_goal / desired_goal 如何构造、sparse reward 如何计算、success 如何判断。更底层的 `robot_env.py` 负责 MuJoCo XML 加载、reset、step、render 和 Gymnasium action/observation space。我的训练脚本通过 `gym.register_envs` 和 `gym.make(env_id)` 创建对应环境，再用 SAC + HER 处理这些 GoalEnv 数据。

如果继续追问为什么四个任务难度不同，可以这样回答：

> Reach 的 `has_object=False`，achieved_goal 就是末端执行器位置，所以 SAC 直接学末端到目标点。Push 和 Slide 的 `has_object=True`，achieved_goal 变成物体位置，策略必须通过接触间接控制物体。Slide 额外有目标偏移和更大目标范围，需要控制速度和摩擦。PickAndPlace 的 `block_gripper=False` 且 `target_in_the_air=True`，策略还要学会夹爪开合、抓取、抬升和放置，所以探索难度最高。

## 十三、一句话总结

四个任务文件决定“做什么任务”，`fetch_env.py` 决定“Fetch 任务如何计算动作、观测、目标和奖励”，`robot_env.py` 决定“MuJoCo 机器人环境如何 reset/step/render”，训练脚本则把这些环境接入 SAC + HER，让策略通过与 MuJoCo 仿真的交互学习连续控制行为。
