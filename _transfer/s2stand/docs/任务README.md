# 任务：Lee 2019 standing-up（趴姿 → 站起来）

> **一个任务一个文件夹**：本目录是第二个行为复现任务的唯一权威副本。
> 文档按整合后的约定只保留两份：**本文件（入口/索引）** + **`实验记录.md`（逐项对照/自定参数/验收/待办）**。
> 上一个行为在 `../lee2019_selfright/`，两者共用 WSL 的 `Cat40Lee2019` 环境副本，
> **主线 `joystick.py` 零改动**。

## 当前状态（2026-11-05 12:5x）

🟡 **配置就绪，代码补丁未执行** —— EXP-01 训练正占着 GPU，执行补丁后必须跑等价性回归
（会抢显存），所以只做了静态校验，没有改任何 WSL 代码、没有启动训练。

| 项 | 值 |
|---|---|
| 配方 | `configs/standup.json`（37 键，obs **93 维**） |
| 实验编号 | `STANDUP-01` / tag `lee2019_standup_exp01` |
| 规模 | 1,000,000 步 / **512 envs**（8 GB 卡上全碰撞模型的实证上限）/ 128×128 / 5 s 回合 |
| 模型 | `scene_selfright_full.xml`（65 碰撞 geom）+ `njmax=64` / `naconmax=1024` |
| 前置 | `scripts/_impl_standup.py --apply` → 等价性 30/30 → CPU 验证 sit 初始姿态 |

## 论文口径（p5 II-D，逐字核对见 `实验记录.md`）

> "Standing up behavior is to stand up from **arbitrary sitting configurations**.
> The cost function is similar to the cost of the self-righting but **a height cost is
> additionally introduced**: `kjpcjp + koco + khch + kaca + kadcad + kjslimcjslim + kτcτ`.
> **The target joint configuration is the standing configuration.** We use the same strategy
> for sampling initial states as the self-righting task but **with a near-upright pose**
> and do not specify any termination condition except for the time limit."

| 与 self-righting 的差异 | 本任务落地 |
|---|---|
| 成本 9 项 → **7 项**：去掉 `c_bi`/`c_bs`/`c_c,in`，**新增 `c_h`** | `reward_config.terms` 7 项 |
| 目标关节：坐姿 `q_sit` → **站姿**（相对零位全 0） | `reward_config.q_target=[0]*12` |
| 初始：随机姿态投落 → **near-upright 任意坐姿** | `task_config.init_mode="sit"` |
| 观测：+ 机体系线速度（base height **不进观测**） | `obs_config.blocks` 10 段 / 93 维 |
| 动作 / 网络 / 终止 | **与 self-righting 完全相同**（`delta_current` / 128×128 / `timeout`） |

⚠️ **高度项的语义**：上一轮证过"二值高度分类判据不可用"；这里 `c_h` 是**连续成本项**
（仿真真值驱动梯度，不进网络输入），两者不矛盾。论文 Table I 明确 base height 只进
Locomotion 的观测。详见 `实验记录.md` §3。

## 使用前必须按顺序做

```bash
# 1) 等 EXP-01 训练结束（或 kill）后，执行补丁
python3 tasks/lee2019_standup/scripts/_impl_standup.py --apply
#   （先跑不带 --apply 的 --check 可复核锚点；已静态校验 3 处唯一命中）

# 2) 等价性回归：默认行为必须与主线逐位等价
python play/_check_lee2019_equiv.py --check play/_baseline_lee2019.npz

# 3) CPU 验证 sit 初始姿态（四足/机身是否稳定着地、不穿地）
#    见 实验记录.md §6 待办

# 4) 结构自检 + 只解析不训练
python play/_inspect_env_cfg.py tasks/lee2019_standup/configs/standup.json
DRY_RUN=1 bash tools/train_cat40_speed.sh tasks/lee2019_standup/configs/standup.json

# 5) 训练
bash tools/train_cat40_speed.sh tasks/lee2019_standup/configs/standup.json

# 6) 验收（跑完把实验目录传进去）
python tasks/lee2019_standup/scripts/_eval_standup.py <实验目录>
```

## 目录

| 路径 | 内容 |
|---|---|
| `configs/standup.json` | 训练配方（本任务唯一权威配置） |
| `configs/sit_pose_touch.json` | 定稿趴姿目标副本：z=0.0416、12 关节角（从 selfright 任务复制） |
| `scripts/_impl_standup.py` | WSL 副本补丁（`sit` 初始模式 + `c_height` 成本项 + 配置键） |
| `scripts/_eval_standup.py` | 验收脚本草案（判据见 `实验记录.md` §5） |
| `evidence/` | 预留：初始姿态截图、验收数据、显存监控 |

## 已知风险（未验证，不能写成结论）

1. **保持力矩**：趴姿腿几乎水平、力臂最大，站起瞬时力矩可能超出 7.4 V 舵机能力。
2. **几何可达性**：上一个行为的 gate 已证 cat4.0 从四脚朝天翻不动（膝行程 180°→50°）；
   趴姿→站姿是否物理可行同样没验过，建议训练前先做一次纯 FK / 静力矩检查。
3. **吞吐**：全碰撞 + 512 envs 在 8 GB 卡上编译约 17 分钟且显存余量仅 1.4 GB，
   EXP-01 已出现长时间无进度输出；本配方已把步数压到 1M，仍可能要数小时。
