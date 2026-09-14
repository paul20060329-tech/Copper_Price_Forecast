# -*- coding: utf-8 -*-
"""训练日志文本生成（仅依赖标准库 + numpy，可在无 torch 环境下使用）。

展示快路径（读取缓存）也会调用本模块，因此不得引入 torch。
"""
import numpy as np


def build_logs(out, torch_ver='(cached)'):
    m = out['meta']
    E = m['epochs']
    marks = [1, 3, 8, 15, 25, 40, 60, 85, 110, 140, 170, 200]
    marks = [x for x in marks if x <= E]
    L = [f'[init ] torch {torch_ver} · device=cpu',
         f'[init ] LSTM(hidden=64,layers=2)+AR(w={m["win"]}) · Adam(8e-3) · MSELoss',
         f'[data ] train={m["trainDays"]}d  val={m["valDays"]}d  future={m["futDays"]}d  regions=5']
    for k in marks:
        for rg in out['regions']:
            e = min(k, len(rg['loss'])) - 1
            L.append(f'[epoch {k:03d}/{E}] {rg["name"]}  train_loss={rg["loss"][e]:.6f}'
                     f'  lr={8e-3 * 0.55 ** (k // 70):.2e}')
    for rg in out['regions']:
        g = rg['garch']
        L.append(f'[garch ] {rg["name"]}  omega={g["omega"]:.3e} alpha={g["alpha"]:.4f} '
                 f'beta={g["beta"]:.4f} persist={g["persist"]:.4f}')
    for rg in out['regions']:
        L.append(f'[eval  ] {rg["name"]}  R2={rg["r2"]:.4f}  RMSE={rg["rmse"]:.1f}  '
                 f'MAE={rg["mae"]:.1f}  MAPE={rg["mape"]:.2f}%')
    L.append(f'[done  ] 五大核心地区平均 R2 = {m["avgR2"]:.4f}')
    return L
