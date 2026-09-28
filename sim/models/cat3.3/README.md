# cat3.3 模型目录(2026-08-21 重构:menagerie 风格 + linkage 模块化)

## 结构

```
models/cat3.3/
+-- README.md                 <- 本文件
+-- scene.xml                 <- 入口(正式版 = joint equality,推荐)
+-- scene_connect.xml         <- 入口(对照版 = connect,仅供对比实验)
+-- robot.xml                 <- 机器人本体(自包含,可独立加载)
+-- linkage_joint.xml         <- 四连杆闭环:8 个 <joint> equality(正式)
+-- linkage_connect.xml       <- 四连杆闭环:4 个 <connect>(对照)
+-- fourbar_map_4legs_pure.json  <- 4 腿纯运动学映射表(生成物)
+-- meshes/  textures/        <- STL 与纹理(ROS 导出物也引用,勿改名)
+-- config/ urdf/ launch/ package.xml CMakeLists.txt  <- ROS/SolidWorks 生态,原样保留
+-- _archive_20260821/        <- 旧版 3 文件 + 迁移脚本(回退用)
+-- 交接文档_2026-08-21.md 落地抖动诊断报告_2026-08-21.md 模型参数总表.txt <- 历史文档
```

## 加载方式

```bash
# 正式版(推荐):joint equality 四次多项式闭环,无约束修正力
python -m mujoco.viewer models/cat3.3/scene.xml

# 对照版:connect 软约束(有 40N 级约束力,仅用于对比抖动/滑移现象)
python -m mujoco.viewer models/cat3.3/scene_connect.xml

# 仅看本体:无闭环约束,被动关节自由(查看几何用)
python -m mujoco.viewer models/cat3.3/robot.xml
```

## 文件职责

| 文件 | 内容 | 可否独立加载 |
|---|---|---|
| scene.xml | include robot + linkage_joint + 环境(option/visual/floor/light) | 是(正式) |
| scene_connect.xml | include robot + linkage_connect + 环境 | 是(对照) |
| robot.xml | 本体:41 mesh(含 base_hull) + 21 body + 20 joint + site + contact + 13 actuator | 是(neq=0) |
| linkage_joint.xml | 8 个 <joint> equality(polycoef) | 否(需 robot) |
| linkage_connect.xml | 4 个 <connect> equality | 否(需 robot) |

注意:linkage_*.xml 与 fourbar_map JSON 是生成物,勿手改;改完模型后重跑:

```bash
python scripts/planE_4legs_official.py   # 扫描 -> 拟合 -> 重生成 linkage 两文件 + 加载检查
```

## 参数位置速查

| 想改什么 | 去哪改 |
|---|---|
| 舵机 kp/kv/forcerange | robot.xml 的 <default class="servo">(一处生效 12 个) |
| **注意: nu=13 的 slot0 是占位死电机**(joint1, ctrl/frc=[0,0]),仅为让 viewer 的 control 编号 1~12 对应 12 个舵机。真实舵机只有 12 个,写代码跳过 slot0 |
| 舵机转子惯量 armature | robot.xml 的 <default class="servo_joint">(12 主动关节,标定 0.033) |
| 关节摩擦 frictionloss | servo_joint=0.125 / bear6804=0.002(小腿) / bear688=0.001(连杆),damping 0.05/0.02(详见 docs/摩擦与关节阻尼.md) |
| 舵机限位 ctrlrange | robot.xml 的 <actuator> 各 position 行 |
| 关节限位/阻尼 | robot.xml 各 <joint> |
| 四连杆 polycoef | 勿手改,跑 planE 脚本重生成 |
| 时间步/积分器/重力 | scene.xml 的 <option> |
| 地面/灯光/天空盒 | scene.xml 的 asset/worldbody |
| 碰撞分组/排除 | robot.xml 的 geom contype/conaffinity 与 <contact> |
| 质量/惯性 | robot.xml 各 <inertial>(数据源自 SolidWorks/URDF) |

## 版本对照(重构前后行为一致,已验证)

| 验证项 | scene.xml | scene_connect.xml |
|---|---|---|
| 加载统计 | nbody=22 njnt=21 nu=13 neq=8 | nbody=22 njnt=21 nu=13 neq=4 |
| 10s 落地滑移 | +0.002 m | -2.695 m |
| 稳态 pitch | +7.50 deg | +7.82 deg |
| 约束力峰值 | 2.2 N | 65.7 N |
| 物理一致性 | 与旧 cat3.3_E2_world 全等(mass/range/eq_data/gain/bias) | 同左 |

旧版三个文件在 _archive_20260821/(路径已修正为 ../meshes ../textures,可独立加载)。

## 遗留问题(见 交接文档 与 落地抖动诊断报告)

1. 前后腿不等长 43.4mm(两版都有 7.5 deg 前倾,E2 版不滑移) —— 修挂载点 z 可根治
2. polycoef 拟合区 ±120 deg 限制,步态规划必须遵守
3. 四连杆零件互不碰撞(contype 分组设计),腿部交叉会穿模