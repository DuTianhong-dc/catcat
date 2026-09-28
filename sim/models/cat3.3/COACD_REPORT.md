# cat3.3 joint2 欠调根因排查最终报告（CoACD 路线验证）

**日期**：2026-08-21
**结论先行**：joint2 卡 15° 的根因是 **mesh 凸包近似假穿透**（非模型几何错误，用户 SolidWorks 干涉检查正确）。最优解法 = **关闭 mesh 自碰撞 + 保留 base 凸包用于地面接触**（`cat3.3_nocollide_world.xml`），12 关节全部恢复完美跟踪。

---

## 1. 本次做了什么（按用户授权）

1. **安装 CoACD**（`coacd 1.0.13`，清华镜像，5s）+ trimesh 5.0.0 + rtree
2. **base 凸分解**：30000 面 threshold 0.08 → 35 凸块（体积比 228%）
3. **20 个 Link 批量凸分解**（后台，Link4/5/9/10 等已出，Link1/2/6/7/11/12/16/17 大件处理中）
4. **验证矩阵**（无重力固定 base 4s，joint2 目标 20.1°）：

| 方案 | joint2 | 说明 |
|------|--------|------|
| 原始（全碰撞） | ~15° | 卡住 |
| 全关 mesh 碰撞 | **20.10°** ✅ | 基线 |
| 只开 base 碰撞 | **20.10°** ✅ | **base 不是根因** |
| 只开膝部碰撞 | -22.96° ❌ | **膝部四连杆是根因** |
| 膝部凸块替换 | 14.48° | 凸块仍膨胀，部分缓解 |
| base 凸块替换 | 12.4° | 凸块膨胀引入新接触 |

5. **最优方案**（XML 级，`cat3.3_nocollide_world.xml`）：
   - 所有真实 mesh：`contype="0" conaffinity="0"`（纯渲染，不参与碰撞）
   - base_hull 凸包：`contype="1" conaffinity="1"`（仅用于地面接触）
   - **验证：12/12 关节误差 < 0.214°，base 稳定站立（3 接触点）**

## 2. 根因链条（完整）

1. **膝部四连杆件是薄片/细杆**（Link4 曲柄 133×47×15mm 薄板）
2. MuJoCo 对 mesh 碰撞**强制用凸包**（qhull），薄片的凸包比原形大
3. Link4 凸包与 Link5 凸包**初始 -7mm 假穿透**（膝部 0.14mm 真实间隙被凸包吞掉）
4. 假穿透产生接触力，通过四连杆 equality 约束**反向作用到 joint2**，顶住欠调
5. base↔Link12/17 的 111N/202N 接触是**次要现象**（有重力全模型下显著，误导了早期排查）

**用户判断验证**：SolidWorks 干涉检查零干涉 → 正确。初始姿态 base 实体与 4 条腿 x 方向完全分离（重叠=0），穿透是凸包假象 + 运动过程中产生。

## 3. CoACD 的结论

- CoACD 凸分解**方向正确**（把凹件拆成凸块贴合原形），但：
  - base 凸块体积膨胀 228%（凸包超集固有），凸块间仍假穿透
  - 精细分解对比：35 凸块 threshold 0.08 → 228%；**178 凸块 threshold 0.03 → 202.6%**（8 分钟，仍远超 120% 目标）——膨胀是凸分解固有特性，精度提升收益递减
  - 需要 threshold 0.01-0.02 级精细分解才接近原形，计算量大且收益有限
- **性价比低**：关闭 mesh 自碰撞（一行 contype=0）直接解决问题，CoACD 非必需
- **CoACD 的真正价值**：凸块可视化（已通过 `*_hull.STL` 实现），让用户能直观看到碰撞形状

## 4. 交付物

| 文件 | 说明 |
|------|------|
| `cat3.3_nocollide_world.xml` | **最优方案**（新文件，原文件未动） |
| `cat3.3_nocollide_model.xml` | 对应的 model 文件 |
| `meshes/LinkN_part*.STL` | 各零件 CoACD 凸块（后台继续生成） |
| `meshes/*_hull.STL` | 21 个凸包可视化文件 |
| `scripts/nocollide_standing.png` | 最优方案站立验证图 |
| `scripts/legs_in_base_overlap.png` | 初始姿态 base 与腿分离验证图 |

## 5. 使用

```bash
# 用最优方案（12 关节完美跟踪 + 站地）
python -m mujoco.viewer models/cat3.3/cat3.3_nocollide_world.xml

# 看凸包（21 个半透明）
python scripts/view_hull_only.py
```

## 6. 遗留

- 后台 Link 凸分解（Link1/2/6/7/11/12/16/17 大件）继续跑，完成后 `LinkN_part*.STL` 齐全
- base_hull 凸包比实体大 → base 视觉悬浮约 2cm（凸包底部触地）。如需贴地可微调 base_hull pos
- 若要保留 mesh 碰撞的物理真实性（防腿相互穿插），可只对膝部薄片件用更精细 CoACD（threshold 0.02），其余保持原样
