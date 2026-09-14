# -*- coding: utf-8 -*-
"""资源路径解析：同时兼容「源码模式」与「PyInstaller 冻结模式」。

冻结时（sys._MEIPASS 存在）资源位于临时解包目录；输出写入 exe 同级目录，
保证对方把整个文件夹拷到任意位置双击即可写文件、找结果。
"""
import os
import sys

if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
    BASE = sys._MEIPASS                               # 打包内只读资源
    APP_DIR = os.path.dirname(os.path.abspath(sys.executable))   # exe 所在（可写）
else:
    HERE = os.path.dirname(os.path.abspath(__file__))            # code/
    BASE = os.path.dirname(HERE)                                    # 项目根 ROOT
    APP_DIR = BASE

SRC = os.path.join(BASE, 'data', '历史价_全年完善版.xlsx')
OUT_DIR = os.path.join(APP_DIR, 'output')
TMPL = os.path.join(BASE, 'code', 'template.html')
ECHARTS = os.path.join(BASE, 'code', 'vendor', 'echarts.min.js')
CACHE_RO = os.path.join(BASE, 'code', 'forecast_cache.json')      # 读取：打包内 / 源码

if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
    CACHE_RW = os.path.join(APP_DIR, 'forecast_cache.json')       # 重训练写入（可写）
else:
    CACHE_RW = CACHE_RO
