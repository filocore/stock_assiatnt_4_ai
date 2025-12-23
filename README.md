# 股票助手（命令行版）

本项目提供一个纯 Python 脚本 `stock_assistant.py`，用于：

- 拉取股票实时快照（最新价/昨收/今开/最高/最低/成交量/成交额）
- 拉取股票历史日 K（开高低收量）并计算一组常用技术指标
- 以人类可读或 JSON Lines（`--jsonl`）方式输出，方便你后续接入策略或可视化

## 环境要求

- macOS / Linux / Windows 均可
- Python 3.9+（推荐 3.10+）
- 无第三方依赖

## 快速开始

进入项目目录后直接运行：

```bash
python3 stock_assistant.py --once --codes 601138
```

实时刷新（每 2 秒一次，刷新 10 次后退出）：

```bash
python3 stock_assistant.py --codes 601138,600000 --interval 2 --ticks 10 --clear
```

JSON Lines 输出（便于程序接入）：

```bash
python3 stock_assistant.py --codes 601138 --jsonl --ticks 1
```

## 股票代码格式

`--codes` 支持以下输入（逗号分隔多个）：

- 纯数字：`600000`、`000001`、`601138`
- 带交易所前缀：`sh600000`、`sz000001`、`bj830001`

脚本会自动归一化（`600000 -> sh600000`，`000001 -> sz000001`）。

## 输出说明

脚本输出分三类指标：

- `intraday_metrics`：基于程序运行时持续采样的实时价格序列（tick 级）计算
- `m5_metrics`：基于 5 分钟 K 线（近 6 个交易日）计算
- `daily_metrics`：基于历史日 K（长度由 `--daily-len` 决定）计算

### 人类可读输出

运行 `--once` 或不加 `--jsonl` 时，会打印：

- 基础行情：现价、昨收、今开、最高、最低、成交量、成交额
- 实时指标（JSON 格式缩进）
- 5 分钟 K 线指标（JSON 格式缩进，涵盖今日及近期走势）
- 日 K 指标（JSON 格式缩进，涵盖中长期趋势，默认开启）

### JSONL 输出

加 `--jsonl` 后，每行输出一个 JSON 对象（方便 `jq` / 日志采集 / 下游程序处理）：

- `intraday_metrics`、`m5_metrics`、`daily_metrics` 会分别作为字段输出
- 所有 `NaN/inf` 会被转换为 `null`（JSON 规范不支持 NaN）

示例：

```bash
python3 stock_assistant.py --codes 601138 --jsonl --ticks 1 | head -n 1
```

## 参数说明

```bash
python3 stock_assistant.py -h
```

常用参数：

- `--codes`：股票代码，逗号分隔；为空则会提示你在命令行输入
- `--once`：只拉取一次并输出后退出
- `--interval`：刷新间隔秒数（默认 5，最小 0.5）
- `--ticks`：运行 N 次刷新后退出（默认 0，表示持续运行）
- `--timeout`：网络超时秒数（默认 5）
- `--clear`：每次刷新清屏（更像“仪表盘”）
- `--jsonl`：按 JSON Lines 输出

日 K 相关：

- `--daily` / `--no-daily`：是否抓取日 K 并计算日 K 指标（默认开启 `--daily`）
- `--daily-len`：日 K 拉取长度（默认 320）
- `--bench`：基准指数（默认 `sh000001`），用于计算 `beta/corr/alpha/rs` 等相对强弱指标

## 指标字段说明（概览）

### intraday_metrics（实时采样）

用于“程序启动后、实时更新”的短周期观察。常见字段：

- 涨跌/波动：`pct_change`、`open_change`、`high_change`、`low_change`、`intraday_range`
- 多周期收益（按采样点）：`ret_1/3/5/10/20/60`
- 均线：`sma_5/10/20/60`、`ema_12/26`
- 动量：`rsi_14`、`macd/macd_signal/macd_hist`
- 布林带：`boll_mid/boll_upper/boll_lower/boll_bw/boll_pos`
- 风险：`realized_vol`、`vol_annualized`、`zscore_20`、`mdd_60`
- 量价：`vwap`
- 样本数：`samples`

### m5_metrics（5 分钟 K 线）

用于日内波段分析，基于近 6 个交易日的 5 分钟 K 线计算。
指标字段与下方 `daily_metrics` 基本一致（如 `sma_20` 代表 20 个 5 分钟周期的均线，即 100 分钟均线）。

### daily_metrics（日 K 序列）

用于更“严肃/全面”的分析。常见字段：

- 收益：`pct_1d/5d/20d/60d/252d`
- 趋势：`sma_20/sma_60`、`ema_20`
- 动量：`rsi_14`、`mfi_14`、`macd/macd_signal/macd_hist`
- 波动/风险：`atr_14`、`realized_vol_20`、`vol_annualized_20`、`mdd_252`
- 位置：`hi_252/lo_252`、`dist_hi_252/dist_lo_252`、`pos_252`
- 量能：`avg_vol_20`、`vol_ratio_20`、`obv`
- 相对强弱（对基准指数，默认 60 日窗口）：`beta_60`、`corr_60`、`alpha_60`、`rs_20`
- 样本数与最新日：`samples`、`last_date`

## 为什么有很多 null？

这是正常现象，常见原因：

1. **样本数不足**  
   很多指标需要足够长的窗口才能计算。例如：

   - `sma_20` 需要至少 20 个样本
   - `rsi_14` 需要至少 15 个样本
   - `macd` 至少需要 26 个样本
   - `pct_252d` 至少需要 253 根日 K

   启动脚本的前几十秒，`intraday_metrics.samples` 还很小，因此大量字段会是 `null`。

2. **JSON 不支持 NaN**  
   指标算不出来时，代码内部会用 `NaN` 表示；输出为 JSON 时会转成 `null`，避免生成非法 JSON。

如何减少 `null`：

- 让实时指标有值：多运行一会儿，例如 `--interval 1 --ticks 60`
- 让日 K 更完整：提高 `--daily-len`，例如使用默认 `320` 或更大
- 让相对强弱/Beta 有值：确保 `--bench` 正确，并且日 K 长度足够（至少 80+，更推荐 320）

## 数据来源与注意事项

- 实时快照：新浪行情接口
- 5 分钟 K 线：新浪行情接口
- 日 K：腾讯行情接口（前复权/不复权取决于接口返回）

这些接口为公开网络服务，可能存在：

- 频率限制/封禁、字段变化、短时不可用
- 盘中与盘后数据差异（尤其是 `close` 与“最新价”的含义）

建议你将其作为研究/学习与数据探索工具使用，并根据需要替换为稳定的付费数据源。

## 免责声明

本项目输出的任何指标与信息仅用于技术研究与学习交流，不构成投资建议。
