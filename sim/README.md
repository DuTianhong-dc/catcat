# sim — MuJoCo 模型

## 结构

| 目录 | 内容 |
|---|---|
| `models/cat3.3` | 现役机型:robot / scene XML + meshes + 四连杆映射表(`fourbar_map_4legs_pure.json`)+ URDF 导出包 |
| `models/cat3.4` | 在研机型:同结构(meshes 在 `build/` 下),含机型参数与碰撞补丁 |

## 提示

- 模型为 MuJoCo(XML)格式;**当前工作基于 cat3.4**
- 其它仿真组件(播放 / 评估脚本、软件在环 SIL、训练补丁)**暂未开放**
