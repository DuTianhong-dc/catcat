# -*- coding: utf-8 -*-
"""【策略大小】网络规格单一事实源。

网络不在环境里 —— brax PPO 的规模由 learning/train_jax_ppo.py 的 flag 决定：
    --policy_hidden_layer_sizes / --value_hidden_layer_sizes（默认 [64,64,64]）
Windows 侧 tools/train_cat40_speed.sh 把配方 _meta.ppo 转成这些 flag。
本模块给出论文 Lee 2019 四个网络的规格，并提供校验，避免配方手抄错。

论文：self-righting / standing-up / behavior selector = 128,128；
      locomotion = 128,256；selector 输出 3 类 softmax。
四个网络分别训练 => 复制四份配方，各写自己的 _meta.net_spec / _meta.ppo。
"""
NET_SPECS = {
    "default": {
        "policy_hidden_layer_sizes": [64, 64, 64],
        "value_hidden_layer_sizes": [64, 64, 64],
    },
    "lee2019_behavior": {
        "policy_hidden_layer_sizes": [128, 128],
        "value_hidden_layer_sizes": [128, 128],
    },
    "lee2019_locomotion": {
        "policy_hidden_layer_sizes": [128, 256],
        "value_hidden_layer_sizes": [128, 256],
    },
    "lee2019_selector": {
        "policy_hidden_layer_sizes": [128, 128],
        "value_hidden_layer_sizes": [128, 128],
    },
}

PPO_FLAG_WHITELIST = {
    "policy_hidden_layer_sizes", "value_hidden_layer_sizes", "num_envs",
    "num_eval_envs", "batch_size", "learning_rate", "entropy_cost",
    "discounting", "clipping_epsilon", "max_grad_norm",
    "policy_obs_key", "value_obs_key",
    # 2026-11-04: 训练侧 --episode_length 默认 1000(20s)，与 env 配方里的
    # episode_length=250(5s) 是两套独立参数。不加这个键，论文「5 s 内恢复」
    # 的回合口径不会生效（实测日志 PPO episode_length=1000）。
    "episode_length",
}


def spec(name):
    if name not in NET_SPECS:
        raise ValueError("未知网络规格: %s（可选 %s）"
                         % (name, ", ".join(sorted(NET_SPECS))))
    return dict(NET_SPECS[name])


def validate_ppo(ppo):
    """校验配方里的 _meta.ppo；未知键训练脚本也会报错，这里提前拦住。"""
    bad = set(ppo) - PPO_FLAG_WHITELIST
    if bad:
        raise ValueError("未知 ppo 键: %s" % ", ".join(sorted(bad)))
    return True
