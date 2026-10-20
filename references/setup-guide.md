# 环境配置指南

本 Skill 只依赖本机 Python 运行时与两个纯计算库，不涉及账号、凭据、网络与云端服务。
所有安装都在用户级完成，不需要管理员权限，也不会修改系统级配置。

核验日期：2026-10-06。

## 依赖总览

| 依赖 id | 类型 | 必需 | 官方主页 | 官方文档 | 安装来源 |
|---|---|---|---|---|---|
| python | 运行时 | 是 | https://www.python.org/ | https://docs.python.org/3/ | https://www.python.org/downloads/ |
| pillow | 本地库 | 是 | https://python-pillow.org/ | https://pillow.readthedocs.io/ | https://pypi.org/project/pillow/ |
| numpy | 本地库 | 是 | https://numpy.org/ | https://numpy.org/doc/ | https://pypi.org/project/numpy/ |

三个依赖都没有登录入口，没有 Token、Key 或 OAuth 流程，因此不需要凭据存储，也不存在权限 scope、
额度或过期轮换问题。

## python

- 用途：作为本地脚本的运行载体，支撑全部四项功能。
- 支持版本：3.10 及以上；Windows、macOS、Linux 均可。
- 官方主页：https://www.python.org/
- 官方文档：https://docs.python.org/3/
- 下载地址：https://www.python.org/downloads/
- 登录网址：不需要
- Token/Key/OAuth 入口：不需要
- 权限与 scope：无，纯本地执行
- 配置步骤：
  1. 先确认现有解释器：执行 `python --version`；若命令不存在，再试 `python3 --version`。
  2. 版本低于 3.10 或命令不存在时，打开下载地址，安装 3.10 或更高版本。
  3. 安装向导里勾选把 Python 加入 PATH。
  4. 重新打开终端让 PATH 生效，再执行一次 `python --version` 核对。
- 凭据存储：不适用
- 验证方式：`python -c "import sys; print(sys.version_info[:3])"`
  输出的前两位应不低于 (3, 10)。
- 过期、权限、额度处理：均不适用
- 轮换和撤销：需要卸载时走操作系统自带的程序卸载入口即可
- 核验日期：2026-10-06

## pillow

- 用途：图像解码与编码、双线性与双三次重采样、EXIF 方向校正；
  支撑配准、去除与质检三项功能。
- 支持版本：Pillow 9.0 及以上。
- 官方主页：https://python-pillow.org/
- 官方文档：https://pillow.readthedocs.io/
- 下载地址：https://pypi.org/project/pillow/
- 登录网址：不需要
- Token/Key/OAuth 入口：不需要
- 权限与 scope：无
- 配置步骤：
  1. 确定要安装到哪个解释器：`python -c "import sys; print(sys.executable)"`。
  2. 执行安装：`python -m pip install --upgrade pillow`。
  3. 若使用虚拟环境，先激活环境再执行安装，避免装到系统解释器上。
- 凭据存储：不适用
- 验证方式：`python -c "import PIL; print(PIL.__version__)"`
  能打印版本号即安装成功；报 `ModuleNotFoundError` 说明当前解释器里还没有这个库。
- 过期、权限、额度处理：均不适用
- 轮换和撤销：`python -m pip uninstall pillow`
- 核验日期：2026-10-06

## numpy

- 用途：数组运算、傅里叶变换、卷积与统计；支撑条纹辨识、去除与质检三项功能。
- 支持版本：NumPy 1.24 及以上。
- 官方主页：https://numpy.org/
- 官方文档：https://numpy.org/doc/
- 下载地址：https://pypi.org/project/numpy/
- 登录网址：不需要
- Token/Key/OAuth 入口：不需要
- 权限与 scope：无
- 配置步骤：
  1. 确定目标解释器，与安装 Pillow 时保持同一个。
  2. 执行安装：`python -m pip install --upgrade numpy`。
  3. 若下载缓慢，可临时指定镜像源，但这属于可选优化，不影响功能。
- 凭据存储：不适用
- 验证方式：`python -c "import numpy; print(numpy.__version__)"`
- 过期、权限、额度处理：均不适用
- 轮换和撤销：`python -m pip uninstall numpy`
- 核验日期：2026-10-06

## 一次跑通的自检

在 Skill 目录下依次执行：

```
python scripts/check_environment.py --capability watermark-removal
python -c "import PIL, numpy; print(PIL.__version__, numpy.__version__)"
```

第一条给出结构化的依赖状态：

- `ready`：全部就绪，直接进入正常流程。
- `partial`：可选检查有异常，按输出里的 degradation 降级执行。
- `needs_setup`：有必需依赖缺失，按输出里的 setup 步骤处理后再重检。
- `unavailable`：必需依赖存在但不可用（例如版本不满足），按 reason_code 排查。

第二条是本 Skill 的真实包级验证。由于依赖清单的机器检查只能确认运行时存在，
包是否真的可导入以这一条为准；`scripts/remove_stripe_watermark.py` 与
`scripts/rect_overlay.py` 在启动时都会做一次同样的导入自检，
失败时返回退出码 3 与 `missing_dependency` 状态，不会静默降级。

**不依赖 scipy / numba。** 全部计算只用 numpy + Pillow，纯 numpy 实现。
不要为了提速引入 scipy/ndimage —— 本机实测没有装，且引入后部署门槛会抬高。

## 目录结构

```
gi-watermark-removal/
├── SKILL.md
├── scripts/
│   ├── remove_stripe_watermark.py   A 分支：条纹水印
│   ├── rect_overlay.py              B 分支：矩形蒙版水印
│   ├── inspect_watermark.py         B 分支只读体检：判类型、标定框、复核接缝
│   └── check_environment.py         依赖自检
├── references/
│   ├── pipeline.md                  三类水印 / 两条处理分支的算法链路（含 §5b 体检算法）
│   ├── qc-and-troubleshooting.md    报告字段与故障排查
│   └── setup-guide.md               本文件
├── skill-dependencies.json          能力与降级说明
└── cache/                运行时自动创建：配准结果缓存（可安全删除）
```

`inspect_watermark.py` 不引入任何新依赖：它 `import rect_overlay` 与
`remove_stripe_watermark`，复用它们已校验过的运行环境。所以
`check_environment.py` 通过、第 2 步的解释器探测通过，它就一定能跑。

`cache/` 存放配准结果 JSON，键是参考图与目标图的「文件名 + 尺寸」组合。
命中可省约 2.4s；删掉只是慢一点，不影响正确性。缓存命中时会在缩略图上校验 NCC，
低于记录值 0.03 就自动回落到完整搜索，所以留着它不会带来错误结果。

## 多个解释器共存时怎么选

同一台机器上常常同时存在多个 Python：系统自带、安装包装的、托管运行时、虚拟环境。
Pillow 与 numpy 装在哪一个里，只有那一个能跑本 Skill。PATH 里的 `python` 不保证就是装了依赖的那个。

选定步骤：

1. 先探测 PATH 里的 `python`：`python -c "import PIL, numpy; print('ok')"`。
2. 失败就依次试其它候选，常见的有 `py -3.12`、`python3`，以及虚拟环境里的解释器。
3. 找到能打印 `ok` 的那一个，记下它的调用名，后续所有命令统一使用。
4. 全部候选都缺依赖时，挑一个按上面的步骤安装，然后重新探测。

核对当前解释器到底是谁：

```
python -c "import sys; print(sys.executable, sys.version)"
```

注意事项：

- 不要把依赖装到不打算使用的解释器上，装完却调用另一个是最常见的坑。
- 一次只用一个解释器跑完全流程，中途不要切换。
- 用虚拟环境时，先激活再安装、再运行，三步在同一个终端会话里完成。

## 常见安装问题

| 现象 | 原因 | 处理 |
|---|---|---|
| `python` 不是内部或外部命令 | 未加入 PATH | 重装并勾选加入 PATH，或改用完整解释器路径调用 |
| 安装了但脚本仍报缺库 | 装到了另一个解释器或另一个虚拟环境 | 用 `python -c "import sys; print(sys.executable)"` 确认路径，对着同一个解释器重装 |
| `pip` 提示权限不足 | 试图写入系统目录 | 改用 `python -m pip install --user`，或使用虚拟环境 |
| 导入 Pillow 报二进制不兼容 | 解释器版本与预编译轮子不匹配 | 升级到 3.10 以上的主流版本，或改用源码安装 |
| 多个 Python 共存导致混乱 | 系统里同时有多个发行版 | 固定一个解释器，全流程用它的绝对路径调用 |
