# -*- coding: utf-8 -*-
"""训练流水线（仅 --retrain 重训练时使用；依赖 torch / pandas / scipy）。

演示快路径直接读取 forecast_cache.json，不导入本模块；
因此打包后的 exe 不会携带 torch / pandas / scipy，体积仅数十 MB。
"""
import time
import datetime as dt
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from garch import fit as garch_fit, ci_band
import paths
import logs

REGIONS = ['江苏', '浙江', '佛山', '安徽', '上海']
WIN = 30
EPOCHS = 200
EPOCHS_F = 90
NTR = 305
FUT = 90
H_FUT = 30
SRC = paths.SRC

torch.set_num_threads(4)
torch.manual_seed(2024)
np.random.seed(2024)


class LSTM_AR(nn.Module):
    """LSTM 条件均值网络 + 线性自回归跳连"""

    def __init__(self, w, hid=64, layers=2):
        super().__init__()
        self.lstm = nn.LSTM(1, hid, layers, batch_first=True,
                            dropout=0.05 if layers > 1 else 0.0)
        self.fc = nn.Linear(hid, 1)
        self.ar = nn.Linear(w, 1)

    def step(self, buf):
        o, _ = self.lstm(buf.unsqueeze(-1))
        return self.fc(o[:, -1, :]).squeeze(-1) + self.ar(buf).squeeze(-1)

    def unroll(self, w0, H):
        buf = w0
        out = []
        for _ in range(H):
            v = self.step(buf)
            out.append(v)
            buf = torch.cat([buf[:, 1:], v.unsqueeze(1)], 1)
        return torch.stack(out, 1)


def train_one_step(z, epochs=EPOCHS, lr=8e-3):
    """单步目标：用于验证集滚动外推"""
    tr = z[:NTR]
    ms = len(tr) - WIN
    W0 = torch.tensor(np.array([tr[j:j + WIN] for j in range(ms)]), dtype=torch.float32)
    Y = torch.tensor(np.array([tr[j + WIN] for j in range(ms)]), dtype=torch.float32)
    m = LSTM_AR(WIN, 64, 2)
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    sch = torch.optim.lr_scheduler.StepLR(opt, 70, 0.55)
    lf = nn.MSELoss()
    losses = []
    m.train()
    for _ in range(epochs):
        perm = torch.randperm(ms)
        tot = 0.0
        for i in range(0, ms, 64):
            b = perm[i:i + 64]
            opt.zero_grad()
            l = lf(m.step(W0[b]), Y[b])
            l.backward(); opt.step(); sch.step()
            tot += l.item() * len(b)
        losses.append(tot / ms)
    return m.eval(), losses


def train_multi_step(z, epochs=EPOCHS_F, lr=8e-3):
    """单步 + 多步展开联合目标：增强长程递归稳定性，用于未来外推"""
    H = H_FUT
    ms = len(z) - WIN - H
    W0 = torch.tensor(np.array([z[j:j + WIN] for j in range(ms)]), dtype=torch.float32)
    Y1 = torch.tensor(np.array([z[j + WIN] for j in range(ms)]), dtype=torch.float32)
    YH = torch.tensor(np.array([z[j + WIN:j + WIN + H] for j in range(ms)]), dtype=torch.float32)
    m = LSTM_AR(WIN, 48, 1)
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    sch = torch.optim.lr_scheduler.StepLR(opt, 45, 0.6)
    lf = nn.MSELoss()
    losses = []
    m.train()
    for _ in range(epochs):
        perm = torch.randperm(ms)
        tot = 0.0
        for i in range(0, ms, 64):
            b = perm[i:i + 64]
            pr = m.unroll(W0[b], H)
            l = lf(pr[:, 0], Y1[b]) + lf(pr, YH[b])
            opt.zero_grad(); l.backward(); opt.step(); sch.step()
            tot += l.item() * len(b)
        losses.append(tot / ms)
    return m.eval(), losses


# ============================ 计算流水线 ============================
def run_pipeline(state, gates):
    def prog(p, msg=''):
        state['prog'] = float(p)
        if msg:
            state['msg'] = msg

    def wait_gate(i):
        while not gates[i]:
            time.sleep(0.05)

    # ---------------- Stage 0 : 数据 ----------------
    state['stage'] = 0
    prog(0.05, '打开 workbook 句柄 ...')
    xl = pd.ExcelFile(SRC)
    data = {}
    for i, r in enumerate(REGIONS):
        df = xl.parse(r)
        d = pd.to_datetime(df['时间'])
        p = pd.to_numeric(df['均价'], errors='coerce').interpolate().bfill().values.astype(float)
        data[r] = {'dates': d, 'price': p}
        prog(0.05 + 0.9 * (i + 1) / len(REGIONS),
             f'载入 {r} · {len(p)} 条日频记录 · {p.min():.0f}~{p.max():.0f} 元/吨')
    wait_gate(0)

    # ---------------- Stage 1 : LSTM 条件均值 ----------------
    state['stage'] = 1
    prog(0.0, '初始化权重 · Adam(8e-3) · MSE')
    res = {}
    for i, r in enumerate(REGIONS):
        p = data[r]['price']
        lp = np.log(p)
        mu_tr, sd_tr = lp[:NTR].mean(), lp[:NTR].std()
        z = ((lp - mu_tr) / sd_tr).astype(np.float32)
        prog(i / len(REGIONS) + 0.01, f'训练 {r} · LSTM(2×64)+AR · {EPOCHS} epochs')
        m, loss = train_one_step(z)
        res[r] = {'z': z, 'mu': mu_tr, 'sd': sd_tr, 'price': p, 'model': m, 'loss': loss}
        prog((i + 1) / len(REGIONS), f'{r} 收敛 · loss={loss[-1]:.6f}')

    # 验证集滚动单步外推（参数仅由 1–10 月拟合，11–12 月仅作输入不参与拟合）
    for i, r in enumerate(REGIONS):
        R = res[r]
        z, mu, sd = R['z'], R['mu'], R['sd']
        nv = len(z) - NTR
        preds = []
        with torch.no_grad():
            for k in range(nv):
                w = torch.tensor(z[NTR - WIN + k:NTR + k], dtype=torch.float32).unsqueeze(0)
                preds.append(R['model'].step(w).item())
        pv = np.exp(np.array(preds) * sd + mu)
        real = R['price'][NTR:]
        r2 = 1 - np.sum((real - pv) ** 2) / np.sum((real - real.mean()) ** 2)
        R['val'] = pv
        R['r2'] = float(r2)
        R['rmse'] = float(np.sqrt(np.mean((real - pv) ** 2)))
        R['mae'] = float(np.mean(np.abs(real - pv)))
        R['mape'] = float(np.mean(np.abs((real - pv) / real)) * 100)
        prog(0.85 + 0.15 * (i + 1) / len(REGIONS), f'{r} · 样本外 R²={r2:.4f}')
    wait_gate(1)

    # ---------------- Stage 2 : 外推模型 + GARCH ----------------
    state['stage'] = 2
    prog(0.0, '重拟合全年数据 · 提取 LSTM 残差')
    for i, r in enumerate(REGIONS):
        R = res[r]
        p = R['price']
        lp = np.log(p)
        mu_a, sd_a = lp.mean(), lp.std()
        zf = ((lp - mu_a) / sd_a).astype(np.float32)
        prog(i / len(REGIONS) + 0.02, f'{r} · 多步展开训练 (H={H_FUT})')
        mf, _ = train_multi_step(zf)
        with torch.no_grad():
            W = torch.tensor(np.array([zf[j - WIN:j] for j in range(WIN, len(zf))]),
                             dtype=torch.float32)
            pred = mf.step(W).numpy()
        resid = (zf[WIN:] - pred) * sd_a
        g = garch_fit(resid)
        R.update(zf=zf, mu_a=mu_a, sd_a=sd_a, mf=mf, garch=g, resid=resid)
        prog((i + 1) / len(REGIONS),
             f'{r} · ω={g["omega"]:.2e} α={g["alpha"]:.3f} β={g["beta"]:.3f} '
             f'α+β={g["persist"]:.3f}')
    wait_gate(2)

    # ---------------- Stage 3 : 外推 + 渲染 ----------------
    state['stage'] = 3
    prog(0.05, f'递归外推 {FUT} 步 · 2025-01-01 → 2025-03-31')
    out = {'regions': []}
    for i, r in enumerate(REGIONS):
        R = res[r]
        with torch.no_grad():
            fz = R['mf'].unroll(torch.tensor(R['zf'][-WIN:], dtype=torch.float32).unsqueeze(0),
                                FUT).squeeze(0).numpy()
        flog = fz * R['sd_a'] + R['mu_a']
        fpred = np.exp(flog)
        hw = ci_band(R['garch'], FUT)
        flo, fhi = np.exp(flog - hw), np.exp(flog + hw)

        fd = [(dt.date(2025, 1, 1) + dt.timedelta(days=k)).strftime('%Y-%m-%d') for k in range(FUT)]
        pred, lo, hi = np.array(fpred), np.array(flo), np.array(fhi)
        mth = []
        for mm in (1, 2, 3):
            k = np.array([int(s[5:7]) == mm for s in fd])
            mth.append({'month': f'2025-{mm:02d}', 'pred': float(pred[k].mean()),
                        'lo': float(lo[k].mean()), 'hi': float(hi[k].mean())})
        out['regions'].append({
            'name': r,
            'dates': data[r]['dates'].dt.strftime('%Y-%m-%d').tolist(),
            'real': [float(v) for v in R['price']],
            'valDates': data[r]['dates'].dt.strftime('%Y-%m-%d').tolist()[NTR:],
            'valPred': [float(v) for v in R['val']],
            'futDates': fd,
            'futPred': [float(v) for v in fpred],
            'futLo': [float(v) for v in flo],
            'futHi': [float(v) for v in fhi],
            'loss': [float(v) for v in R['loss']],
            'monthly': mth,
            'r2': R['r2'], 'rmse': R['rmse'], 'mae': R['mae'], 'mape': R['mape'],
            'garch': {k: R['garch'][k] for k in ('omega', 'alpha', 'beta', 'persist', 'long_run_vol')},
        })
        prog(0.05 + 0.85 * (i + 1) / len(REGIONS),
             f'{r} · Q1 起点 {fpred[0]:.0f} → 终点 {fpred[-1]:.0f}')

    avg = float(np.mean([x['r2'] for x in out['regions']]))
    out['meta'] = {'epochs': EPOCHS, 'win': WIN, 'trainDays': NTR,
                   'valDays': 366 - NTR, 'futDays': FUT, 'avgR2': round(avg, 4)}
    state['raw'] = [(r['name'], r['r2']) for r in out['regions']]
    out['meta']['logs'] = logs.build_logs(out, torch.__version__)
    state['result'] = out
    state['stage'] = 4
    prog(1.0, '面板渲染完成 · 正在启动 Dashboard')
    wait_gate(3)
