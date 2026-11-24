# 训练侧实际生效的源码（cat40/lee2019 环境）

> 来源：WSL `~/mujoco_playground/mujoco_playground/_src/locomotion/cat40/lee2019/`，
> 于 2026-11-23 18:55 原样复制。**这是 v13–v15 训练时真正跑的那份代码**，
> 不是 11-05 的 `scripts/_impl_standup.py` 补丁（那个只是当时的改造清单，内容早已被覆盖）。
> 已排除：`*.bak_*`（各轮改造的历史备份，共 10 余份）、`__pycache__`。

## 文件

| 文件 | 字节 | 最后修改 | 作用 |
|---|---|---|---|
| `task.py` | 16,597 | 11-20 09:07 | 环境主体：`reset()` 起点注入、`termination()` 三种终止模式 |
| `rewards.py` | 62,081 | 11-21 06:05 | 全部奖励/成本项实现（`TERM_SPECS` / `COST_TERMS` / `_term_*` 函数） |
| `defaults.py` | 42,556 | 11-21 11:46 | 配置默认值与全部调参键（v14 门控、v15 五段课程都在这里的 reward_config） |
| `obs.py` | 8,035 | 11-19 22:49 | 观测拼装（9 段 = 90 维） |
| `action.py` | 3,906 | 11-21 11:40 | 动作映射（`delta_current` / `per_side`） |
| `networks.py` | 2,220 | 11-04 07:57 | 128×128 网络定义 |
| `dr.py` | 4,177 | 11-11 02:18 | 域随机化 |
| `__init__.py` | 254 | 11-08 02:47 | 包导出 |

## 读代码时先看这三处

1. **`task.py:164 termination()`**：`terminate_mode` 三选一（`fall` / `timeout` / `stand_fall`）。
   `stand_fall` 是 v13 加的**锁存式**：只有 `info["stand_latch"]`（本回合曾站正且 bz>0.10）为真之后，
   掉到 `up_z<0.65` 或 `bz<0.06` 才终止；**没站起来过的回合永不提前终止**。
2. **`rewards.py:15 TERM_SPECS` / `:88 COST_TERMS`**：奖励项注册表；每轮的开关与权重来自
   `defaults.py` 的 `reward_config`（v15 实际值见 `../config/exp15_s2stand_v15_运行时配置.json`）。
3. **`action.py`**：`delta_current` 的 `m_t = clamp(q_t + 0.25·a_t)` 就实现在这里，
   与 `../scripts/pd_baseline.py` 的 delta 模式一一对应（后者是它的零动作特例）。

## 与仓库其它部分的关系

- 训练配置：`../config/exp15_s2stand_v15_运行时配置.json`（运行时 dump，权威）
- 起点库：`../data/curriculum_mix.npz`（300 条，`group` 分 near_stand / transition / original）
- 模型：`../model/scene_selfright_full_v2.xml`
- 检查点：`../checkpoints/000000327680` / `000000655360`（Orbax 格式）
- 训练入口：WSL `~/mujoco_playground/learning/train_jax_ppo.py`（未随包复制；如需可另取）
