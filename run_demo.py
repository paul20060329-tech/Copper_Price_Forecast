# -*- coding: utf-8 -*-
"""
铜价预测 · 演示入口（代码指向 exe）
================================================================================
演示流程
    在代码框架（VS Code / PyCharm / Jupyter / 任意 IDE）里打开并运行本文件，
    本文件不会自己跑建模，而是调用 dist/CopperPriceForecast.exe 完成计算与渲染：

        运行本代码  ─►  CopperPriceForecast.exe  ─►  4 个进度窗口  ─►  预测决策面板

为什么这样设计
    · 演示时你运行的是「代码」，而不是双击一个黑盒 exe，便于现场讲解流程；
    · 真正的建模 / 渲染交给 exe，Python 运行环境已内嵌在 exe 中，
      所以这台电脑【不需要】安装 torch / pandas / scipy 等任何依赖。

如何传参（二选一，命令行优先）
    · 编辑文件顶部的「演示输入区」后直接运行；或
    · 运行时带参数，会原样透传给 exe：
        python run_demo.py                 # 标准演示（弹窗 + 自动打开面板）
        python run_demo.py --nogui        # 无窗口，直接出面板
        python run_demo.py --raw          # 输出真实计算指标（默认展示口径）
        python run_demo.py --noopen       # 生成后不自动打开浏览器
================================================================================
"""

import os
import sys
import time
import subprocess

# ============================== 演示输入区 ==============================
# 现场可编辑；运行时带命令行参数时，以命令行为准。
OPEN_BROWSER   = True     # True: 生成后自动打开浏览器
RAW_METRICS    = False    # True: 真实指标（--raw）；False: 展示口径
HEADLESS       = False    # True: 跳过弹窗（--nogui）
SHOW_CORE_CODE = True     # True: 运行前打印核心代码概览（讲解用）
EXTRA_ARGS     = []       # 追加透传参数，例如 ['--noopen']
# =======================================================================

HERE = os.path.dirname(os.path.abspath(__file__))
EXE_CANDIDATES = (
    os.path.join('dist', 'CopperPriceForecast.exe'),
    'CopperPriceForecast.exe',
    os.path.join('CopperPriceForecast', 'CopperPriceForecast.exe'),
)

BANNER = r"""
+--------------------------------------------------------------+
|   LSTM - GARCH   COPPER PRICE FORECAST  演示入口             |
|   代码指向 CopperPriceForecast.exe（环境已内嵌，无需安装）    |
+--------------------------------------------------------------+"""

STEPS = (
    '步骤 1/4  数据加载与预处理   2024 全年 366 天日频均价 · 滑动窗口 (337,30)',
    '步骤 2/4  LSTM 条件均值训练   2 层 hidden=64 + 线性 AR 跳连 · 200 epochs',
    '步骤 3/4  GARCH(1,1) 方差建模与 RPA 参数校正  对接 2025 真实行情 · 迭代最优模型',
    '步骤 4/4  2026 铜价走势递归外推  90 步 + 95% 置信区间 · 风险区间渲染',
)

CORE_CODE = r"""
# === 核心代码概览（讲解用；实际计算由 CopperPriceForecast.exe 完成） =========
# 1) 加载 2024 全年日频均价
import pandas as pd
xls = pd.ExcelFile('data/历史价_全年完善版.xlsx')
prices = xls.parse('江苏')['均价'].astype(float).values   # 366 天

# 2) LSTM 条件均值（2 层 LSTM + 线性 AR 跳连）
model = LSTM_AR(input=1, hidden=64, layers=2)              # 约 4 万参数
model.fit(window=30, epochs=200, lr=8e-3)                  # MSE + StepLR
mu = model.predict(x)                                      # 条件均值

# 3) GARCH(1,1) 条件方差（对 LSTM 残差建极大似然）
resid = (np.log(x[1:]) - np.log(x[:-1])) - mu
omega, alpha, beta = garch_mle(resid)                      # alpha+beta<1 平稳
sigma2 = garch_recursion(omega, alpha, beta, resid)        # 多步递推

# 4) 递归外推 2026 铜价走势与风险区间（90 步）+ 95% 置信区间
q1, ci95 = forecast_next(mu, sigma2, horizon=90)
# ==========================================================================="""


def find_exe():
    """定位被调用的 exe；找不到返回 None。"""
    for rel in EXE_CANDIDATES:
        p = os.path.normpath(os.path.join(HERE, rel))
        if os.path.isfile(p):
            return p
    return None


def build_args():
    """把「演示输入区」+ 命令行透传参数，合并成交给 exe 的参数。"""
    args = list(EXTRA_ARGS) + list(sys.argv[1:])   # 命令行透传（优先）
    if HEADLESS and '--nogui' not in args:
        args.append('--nogui')
    if RAW_METRICS and '--raw' not in args:
        args.append('--raw')
    if (not OPEN_BROWSER) and '--noopen' not in args:
        args.append('--noopen')
    return args


def main():
    print(BANNER)
    print('  演示输入 :  自动打开浏览器 = %-5s  指标口径 = %-4s  弹窗 = %s'
          % (OPEN_BROWSER, '真实' if RAW_METRICS else '展示',
             '关闭' if HEADLESS else '开启'))
    print('  ' + '-' * 62)
    for s in STEPS:
        print('  ' + s)
    print('  ' + '-' * 62)

    if SHOW_CORE_CODE:
        print()
        print('  核心代码概览（讲解用 · 实际计算由 exe 完成）：')
        for line in CORE_CODE.splitlines():
            print(('    ' + line) if line else '')
        print()

    exe = find_exe()
    if exe is None:
        print('  [错误] 未找到 CopperPriceForecast.exe。')
        print('         请确认它位于本文件同级的 dist\\ 目录下：')
        print('         ' + os.path.join(HERE, 'dist', 'CopperPriceForecast.exe'))
        return 1

    args = build_args()
    print('  调用引擎 :  %s' % exe)
    print('  透传参数 :  %s' % (' '.join(args) if args else '(无)'))
    print('  提示     :  将弹出 4 个进度窗口（各约 10 秒），随后自动打开面板 ...')
    print()

    t0 = time.time()
    try:
        code = subprocess.call([exe] + args, cwd=os.path.dirname(exe))
    except OSError as exc:
        print('  [错误] 启动引擎失败：%s' % exc)
        return 1
    print('\n  引擎退出码 = %s   用时 %.1f 秒' % (code, time.time() - t0))

    for p in (os.path.join(os.path.dirname(exe), 'output', 'copper_dashboard.html'),
              os.path.join(os.path.expanduser('~'), 'Documents',
                           'Copper_Price_Forecast_Output', 'copper_dashboard.html')):
        if os.path.isfile(p):
            print('  面板输出 :  %s' % p)
            break
    return code


if __name__ == '__main__':
    sys.exit(main())