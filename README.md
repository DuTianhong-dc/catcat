# catcat:一只基于强化学习的机器猫

> 「猫包」四足机器猫(cat3.3 → cat3.4)——基于 STM32F407 + MuJoCo/MJX,
> 强化学习策略直接部署在 STM32 上,跑 50Hz 本地闭环。
> 本仓库开放:**机械设计、MuJoCo 模型、STM32 固件** 三部分。

## 仓库结构

| 目录 | 内容 |
|---|---|
| `cad/` | SolidWorks 设计(0.0 ~ 3.3 全系列)+ 复刻 BOM |
| `sim/models/` | MuJoCo 模型(cat3.3 现役 / cat3.4 在研) |
| `firmware/` | STM32F407 固件(CAT_APP2.0)+ 协议与说明文档 |
| `docs/` | 接线说明(定稿,含总图) |

## 快速指路

- 接线与接头线材 → [`docs/接线说明.md`](docs/接线说明.md)
- 复刻物料 → [`cad/BOM_复刻物料清单.md`](cad/BOM_复刻物料清单.md)
- 模型结构 → [`sim/README.md`](sim/README.md)

## 许可

- 代码(`firmware/`)→ **MIT**(见 [LICENSE](LICENSE))
- CAD 与文档(`cad/` `sim/` `docs/`)→ **CC BY 4.0**
