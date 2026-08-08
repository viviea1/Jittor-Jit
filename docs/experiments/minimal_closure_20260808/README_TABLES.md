# JiT–Jittor 最小实验闭环（2026-08-08）

## 正式 FID-50K

| 实现 | FID | IS | 总耗时 | GPU 峰值 | 退出状态 |
|---|---:|---:|---:|---:|---:|
| Torch 官方预训练权重 | 3.6580 | 269.4181 | 8009 s | 5647 MiB | 0 |
| Jittor 生成 + 同口径 Torch-Fidelity | 3.6808 | 269.0536 ± 3.7150 | 18570 s | 3345 MiB | 生成 0 / 评估 0 |

说明：两者均使用现有 `jib-b-16.pth`、50,000 张、BF16、Heun-50、CFG 3.0、区间 [0.1,1.0]、batch 64。Jittor 的 FID 评估阶段复用仓库固定版本的 Torch-Fidelity。短时显存来自 1 秒设备采样，不等同于严格同实现的训练显存比较。

## 本次 FP32 公平短训

| 实现 | 10-step 状态 | 首步 loss | 第10步 loss | step 2–10 吞吐 | GPU 峰值 | 进程 RSS 峰值 | 参数更新 L2 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Torch 2.5.1 | exit 0 | 0.649360 | 0.276542 | 74.84 img/s | 4725 MiB | 13.40 GiB | 0.940651 |
| Jittor 1.3.11.0 | exit 0 | 0.649439 | 0.276411 | 38.36 img/s | 7055 MiB | 5.31 GiB | 0.940594 |

公平条件：同一 SHA-256 的 183-tensor Torch seed-0 FP32 checkpoint；同一 8 张 ImageNet 固定小批和顺序；同一预生成 noise、t、label-drop；effective batch 8；AdamW LR 2e-4、betas (0.9,0.95)、eps 1e-8、WD 0；双 EMA 0.9999/0.9996。首步 loss 绝对差 `0.00007915`，10步最大绝对差 `0.00050414`；两边均确认非零更新。

边界：这是为可复现/数值路径对齐设计的 FP32 小批短训，不是论文 BF16 吞吐 benchmark，也不能证明长期收敛或最终 FID。Jittor 稳态约为 Torch 的 `51.3%`，只适用于本 harness。

## 事实边界

- FID 4.7355 / IS 221.5594 属于 **Torch 200-epoch** 模型；Jittor 完整 200-epoch 尚未完成。
- 既有 Jittor 双卡 3-round 是 checkpoint 初始化且 microbatch/累积不同，只是受控性能短测，不是严格公平收敛对比。
- 既有 Jittor random-init 1-step 首步 LR=0，只是 smoke/probe。
