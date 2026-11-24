# 第二网络（s2stand / 站立起身）—— 中转包

> 这个文件夹是**一次性中转包**：把第二个网络（趴姿 → 起身站稳）的全部配置、脚本、模型、
> 训练检查点和说明文档集中放在一起，供网页版智能体整包阅读。
> 生成时间：2026-11-23 18:10；源工作区 `C:\Users\dth15\Desktop\mujoco` + WSL `~/mujoco_playground`。
> 路径用 ASCII（`_transfer/s2stand`）是刻意的：MuJoCo 在 Windows 上打开**含中文的路径会报
> `Error opening file`**（C++ 侧用窄字符 fopen）。模型放在中文目录里会加载失败。

---

## 1. 这包是干什么的

四足机器猫（12 自由度、五连杆腿、STM32F407 + MuJoCo/MJX）用三个网络分工：
`p1` 从四脚朝天翻到侧躺 → **`s2stand`（本包主角）从各种倒地/蹲姿起身并站稳** → `walk` 行走。

本包只装第二个网络。**它目前没跑通**，所以这包里没有"能用的权重"，
有的是"完整的问题现场"：配置、代码补丁、训练曲线来源、检查点、以及一份写给外部模型的交接文档。

## 2. 建议阅读顺序

| 顺序 | 文件 | 为什么先读它 |
|---|---|---|
| 0 | `ALL_IN_ONE.md` | **只想给网页版智能体喂一次**：11 份文本按固定顺序合并成 117 KB 单文件，读完即全貌 |
| 1 | `README.md`（本文件） | 目录与参数索引 |
| 2 | `docs/交接文档_2026-11-22.md` | **最关键**：一句话现状、7 版奖励都没突破的证据、已排除的假设 |
| 3 | `config/exp15_s2stand_v15_运行时配置.json` | 最近一次训练（v15）真正生效的完整配置（26 个顶层键） |
| 4 | `docs/实验记录.md` | 逐项对照 Lee 2019 论文、自定参数、验收口径 |
| 5 | `scripts/_impl_standup.py` | 训练侧代码补丁（奖励/终止/起点注入） |
| 6 | `docs/任务README.md` | 任务目录入口，含旧配方 `standup_exp01` 的口径 |

## 3. 文件清单

```
s2stand/
├── README.md                                  本文件
├── ALL_IN_ONE.md                              ★ 单文件合订本（117 KB，喂网页智能体用）
├── docs/
│   ├── 交接文档_2026-11-22.md                  ★ 问题现状与证据（35 KB）
│   ├── 任务README.md                           任务入口 + 论文 p5 II-D 原文口径
│   └── 实验记录.md                             逐项对照 / 自定参数 / 验收 / 待办
├── config/
│   ├── exp15_s2stand_v15_运行时配置.json       ★ 权威：v15 训练时实际生效的配置
│   ├── ppo_network_config.json                 PPO 网络结构（128×128）
│   ├── standup_exp01_20261105.json             旧配方（obs 93 维，含 linvel 段）
│   └── sit_pose_touch.json                     坐姿触地起点生成配置
├── scripts/
│   ├── _impl_standup.py                        训练侧补丁（--apply 才生效）
│   └── _eval_standup.py                        CPU 评估/验收脚本
├── checkpoints/
│   ├── 000000327680/                           Orbax 检查点（327,680 步）
│   └── 000000655360/                           Orbax 检查点（655,360 步）
└── model/
    ├── scene_selfright_full_v2.xml             ★ 训练用的场景（入口）
    ├── robot_selfright_full_v2.xml             对应的机器人（含碰撞体）
    ├── 其余 robot_*.xml / scene_*.xml          同系列变体，便于对照
    ├── textures/desert.png                     场景地面贴图（缺了会报 Error opening file）
    ├── cat4.0_live.mjz                         打包好的实时模型（可直接 mj_load）
    ├── meshes/                                 35 个 STL 网格（20 MB）
    └── 模型说明.md                             模型结构与质量/限位说明
```

> 已实测：`model/scene_selfright_full_v2.xml` 在本目录内可直接
> `mujoco.MjModel.from_xml_path(...)` 加载成功（`nq=31, ngeom=78, nmesh=35, 1.959 kg`），
> 不需要回到原工作区。

## 4. 关键参数（取自 v15 运行时配置，逐项可查）

| 项 | 值 |
|---|---|
| 模型 | `scene_selfright_full_v2.xml`（cat4.0，总质量 1.959 kg） |
| 观测 | 9 段拼接 = `gravity`(3) `gyro`(3) `jpos`(12) `jvel`(12) `jpos_err`(12) `last_act` `last2_act` `last3_act`(各 12) `motor_targets`(12) → **90 维** |
| 动作 | 12 维，`delta_current`：`m_t = clamp(q_t + 0.25·a_t, 限位)`，`apply_velocity_limit=false` |
| 控制/物理 | `ctrl_dt=0.02`（50 Hz）、`sim_dt=0.002`（500 Hz，每拍 10 子步）、`episode_length=250`（5 s） |
| 求解器 | `impl=jax`、`njmax=64`、`naconmax=1024` |
| 终止 | `terminate_mode=stand_fall`：曾满足 `up_z>0.85 且 bz>0.10` 之后掉到 `up_z<0.65` 或 `bz<0.06` 即结束；另有 `min_base_height=0.1`、`up_vec_z=0.4` |
| 起点 | `init_mode=pose_bank`、`init_up_z_max=-0.85`、`joint_range_scale=0.8`（近站 60 / 过渡 180 / 趴侧仰 60） |
| 奖励 | `cost_mode=true`，147 个键，5 段课程（`c_stage_*`）；站高参考 `0.1473 m` / 四足面积 `0.040915 m²` |
| 域随机化 | `pert_config.enable=false`（v13–v15 关闭，为了隔离变量）；`noise_config.level=1.0` |
| 网络 | 128 × 128，PPO `lr=3e-4 / entropy 0.002 / γ=0.9988 / clip 0.2 / 16 epochs` |
| 检查点 | 327,680 步 / 655,360 步（Orbax 目录格式，需 JAX 才能读） |

## 5. 现状（截至 2026-11-22）

- 从**接近站姿**的起点能站住约 1–3 s（最好一条连续 2.9 s）；
- 从**趴姿**基本起不来：22 条趴姿起点里 **21 条**最长站稳 **0 拍**；
- 站上去时姿态很怪（三足支撑、一条腿翘起，关节角离站姿 30–60°）；
- **任何一轮训练都无法把机身高度维持在起始高度之上**，高度曲线全程单调下沉；
- 换了 **7 版奖励结构**（v9–v15）没有突破，怀疑问题在"目标函数结构 + 动作参数化"，不在超参。

> 结论性判断都在 `docs/交接文档_2026-11-22.md` 的 §6，请以那里为准；本文件只做导航。

## 6. 读这份包时要注意的几处坑

1. **观测维度有两个口径**：旧配方 `standup_exp01_20261105.json` 是 **93 维**（含 `linvel` 段），
   v15 运行时配置是 **90 维**（无 `linvel`）。引用数字前先确认看的是哪一份。
2. **`act_mapping=per_side` 与 `action_config.mode=delta_current` 同时存在**：前者是动作映射约定，
   后者是增量参数化；固件里两条路径都有实现（`obs_act_to_counts` / `obs_act_to_counts_delta`），
   第二网络该走哪条**尚未在真机验证**。
3. **`ablate_contact=true`、`ablate_tau=true`**，且 `obs_config.blocks` 里确实没有 `contact`/`tau` 段 ——
   说明当前网络看不到足端接触和关节力矩，这两个信息被刻意关掉了（历史原因见交接文档）。
4. **检查点是 Orbax 目录**，不是 `.npz`。要在 Windows 侧直接加载，得先在 WSL 里用
   `orbax.checkpoint` 转成 numpy；本包没有附带转换脚本。
5. **模型是 cat4.0**，而仓库 `catcat/sim/models/` 目前只到 cat3.4 —— 本包里的 `model/` 是 cat4.0 的第一份。

## 7. 来源

| 内容 | 原始位置 | 时间 |
|---|---|---|
| 任务配置与脚本 | `Desktop/mujoco/tasks/lee2019_standup/` | 2026-11-05 ~ 11-22 |
| 运行时配置与检查点 | WSL `~/mujoco_playground/logs/Cat40Lee2019-20261121-225616-exp15_s2stand_v15/` | 2026-11-21 22:56 |
| 模型（cat4.0） | `Desktop/mujoco/models/cat4.0/` | 2026-11-18 更新 |
| 交接文档 | `Desktop/s2站立网络_问题交接_2026-11-22.md` | 2026-11-22 03:20 |
