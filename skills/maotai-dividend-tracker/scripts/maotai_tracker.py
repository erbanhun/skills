#!/usr/bin/env python3
"""
茅台红利持有逻辑周度追踪器
按周收集茅台（600519）数据，基于《红利持有逻辑》框架评估持有得分。

依赖: pip install akshare pandas numpy
用法: python maotai_tracker.py [--output-dir ./reports]
"""

import argparse
import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

import akshare as ak


SYMBOL = "600519"
SYMBOL_SINA = "sh600519"
STOCK_NAME = "贵州茅台"


def fetch_price_data(years: int = 5) -> pd.DataFrame:
    """获取历史日线行情（前复权），用于计算估值分位和技术指标。"""
    start_date = (datetime.now() - timedelta(days=years * 365)).strftime("%Y%m%d")
    df = ak.stock_zh_a_daily(symbol=SYMBOL_SINA, start_date=start_date, adjust="qfq")
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df


def fetch_financial_data(start_year: int = 2020) -> pd.DataFrame:
    """获取财务分析指标（季度）。"""
    df = ak.stock_financial_analysis_indicator(symbol=SYMBOL, start_year=str(start_year))
    # 列名在 akshare 中以 gbk 字节流传输，这里按位置索引硬编码关键列
    # 0:报告期, 2:加权每股收益(元), 7:每股经营现金流(元), 25:股息支付率(%),
    # 32:净利润增长率(%), 61:资产负债率(%), 65:经营现金净流量与净利润的比率(%)
    key_cols = [0, 2, 7, 25, 32, 61, 65]
    col_names = [
        "报告期",
        "每股收益",
        "每股经营现金流",
        "股息支付率",
        "净利润增长率",
        "资产负债率",
        "现金流净利润比",
    ]
    result = df.iloc[:, key_cols].copy()
    result.columns = col_names
    result["报告期"] = pd.to_datetime(result["报告期"], errors="coerce")
    # 剔除空行并转为数值
    for c in col_names[1:]:
        result[c] = pd.to_numeric(result[c], errors="coerce")
    result = result.dropna(subset=["报告期"]).sort_values("报告期").reset_index(drop=True)
    return result


def fetch_dividend_data() -> pd.DataFrame:
    """获取历史分红数据，计算最近12个月每股分红。"""
    df = ak.stock_dividend_cninfo(symbol=SYMBOL)
    # 关键列位置: 0实施公告日, 1分红类型, 4派息金额(每10股)
    # 注意：该接口第8列"每股派息"通常为NaN，需用第4列/10推算
    result = df.iloc[:, [0, 1, 4]].copy()
    result.columns = ["公告日", "分红类型", "每10股派息"]
    result["公告日"] = pd.to_datetime(result["公告日"], errors="coerce")
    result["每10股派息"] = pd.to_numeric(result["每10股派息"], errors="coerce")
    result = result.dropna(subset=["公告日", "每10股派息"])
    # 只保留现金分红（每10股派息>0）
    result = result[result["每10股派息"] > 0].copy()
    result["每股派息"] = result["每10股派息"] / 10.0
    result = result.sort_values("公告日").reset_index(drop=True)
    return result


def compute_dividend_yield(price_df: pd.DataFrame, div_df: pd.DataFrame) -> float:
    """计算当前股息率（最近12个月总分红 / 最新收盘价）。"""
    latest_price = float(price_df["close"].iloc[-1])
    if latest_price <= 0 or div_df.empty:
        return 0.0

    latest_date = price_df["date"].iloc[-1]
    one_year_ago = latest_date - timedelta(days=365)
    recent_div = div_df[div_df["公告日"] >= one_year_ago]["每股派息"].sum()
    return recent_div / latest_price * 100


def compute_pe_pb_percentile(price_df: pd.DataFrame) -> tuple:
    """基于历史收盘价和财务数据，估算 PE/PB 历史分位。
    由于新浪日线无 PE/PB，这里用‘价格/每股收益’近似 PE。
    """
    # 取最近 4 个季度的每股收益之和作为 TTM 每股收益
    fin = fetch_financial_data(start_year=datetime.now().year - 3)
    if fin.empty or len(fin) < 4:
        return None, None

    prices = price_df["close"].values
    if len(prices) < 60:
        return None, None

    current_price = prices[-1]

    # 计算 5 年价格分位
    price_pct = np.sum(prices <= current_price) / len(prices) * 100

    # 用历史每股收益序列构建近似 PE 序列
    # 将季度财务数据的报告期转为对应季度末的 datetime，再与日线合并
    fin_copy = fin.copy()
    fin_copy["quarter_end"] = fin_copy["报告期"].dt.to_period("Q").dt.to_timestamp(how="end")
    fin_quarterly = fin_copy.groupby("quarter_end").last()[["每股收益"]].rename(columns={"每股收益": "eps"})

    price_df_copy = price_df.copy()
    # 将每个交易日的日期映射到对应的季度末
    price_df_copy["quarter_end"] = price_df_copy["date"].dt.to_period("Q").dt.to_timestamp(how="end")
    merged = price_df_copy.merge(fin_quarterly, on="quarter_end", how="left")
    merged["eps"] = merged["eps"].ffill()
    merged = merged[merged["eps"] > 0].copy()
    merged["pe"] = merged["close"] / merged["eps"]

    pe_pct = None
    if not merged.empty and len(merged) > 60:
        current_pe = merged["pe"].iloc[-1]
        pe_pct = np.sum(merged["pe"].values <= current_pe) / len(merged) * 100

    return pe_pct, price_pct


def compute_technical_score(price_df: pd.DataFrame) -> dict:
    """计算技术面得分代理指标。"""
    df = price_df.copy()
    df["ma20w"] = df["close"].rolling(window=100, min_periods=60).mean()  # 约 20 周均线
    latest = df.iloc[-1]
    close = latest["close"]
    ma20w = latest["ma20w"]

    # 52 周高点位置
    year_high = df["close"].iloc[-252:].max() if len(df) >= 252 else df["close"].max()
    high_ratio = close / year_high if year_high and year_high > 0 else 1.0

    # 20 周均线趋势（斜率）
    ma_slope = 0
    if len(df) >= 110:
        ma_slope = (df["ma20w"].iloc[-1] - df["ma20w"].iloc[-10]) / df["ma20w"].iloc[-10]

    score_high = 8
    if high_ratio > 0.95:
        score_high = 2
    elif high_ratio > 0.85:
        score_high = 5
    elif high_ratio > 0.70:
        score_high = 6
    else:
        score_high = 8

    score_ma = 7
    if close > ma20w and ma_slope > 0.02:
        score_ma = 7
    elif close > ma20w and ma_slope > -0.02:
        score_ma = 5
    elif close > ma20w:
        score_ma = 4
    else:
        score_ma = 2

    return {
        "52周高点比": round(high_ratio * 100, 2),
        "20周均线": round(ma20w, 2) if pd.notna(ma20w) else None,
        "均线斜率": round(ma_slope * 100, 2) if pd.notna(ma_slope) else None,
        "位置得分": score_high,
        "趋势得分": score_ma,
    }


def evaluate(
    price_df: pd.DataFrame,
    fin_df: pd.DataFrame,
    div_df: pd.DataFrame,
) -> dict:
    """基于《红利持有逻辑》框架对茅台进行评分。"""
    latest_price = float(price_df["close"].iloc[-1])
    latest_date_dt = price_df["date"].iloc[-1]
    latest_date = latest_date_dt.strftime("%Y-%m-%d")

    # 1. 股息率评分 (25分)
    dy = compute_dividend_yield(price_df, div_df)
    if dy >= 4.0:
        score_dy = 25
        dy_level = "极高（需深究）"
    elif dy >= 3.0:
        score_dy = 22
        dy_level = "低估区间"
    elif dy >= 1.5:
        score_dy = 15
        dy_level = "合理区间"
    else:
        score_dy = 5
        dy_level = "偏贵"

    # 2. 分红健康度 (40分)
    # 取最近 4 个季度数据
    recent_fin = fin_df.tail(4)
    if recent_fin.empty:
        recent_fin = fin_df.tail(1)

    # 净利润增长率 - 取最新季度
    np_growth = recent_fin["净利润增长率"].iloc[-1]
    if pd.isna(np_growth):
        score_growth = 5
    elif np_growth > 10:
        score_growth = 10
    elif np_growth > 0:
        score_growth = 7
    elif np_growth > -10:
        score_growth = 4
    else:
        score_growth = 0

    # 经营现金流/净利润比
    # 该接口此列数据统一为小数比例（如0.72=72%），需统一乘以100
    cf_ratio_raw = recent_fin["现金流净利润比"].iloc[-1]
    if pd.isna(cf_ratio_raw):
        cf_ratio = None
        score_cf = 5
    else:
        cf_ratio = cf_ratio_raw * 100
        # 金融类（银行/保险）现金流特征不同，放宽评分上限
        if cf_ratio > 200:
            score_cf = 10
        elif cf_ratio > 80:
            score_cf = 10
        elif cf_ratio > 50:
            score_cf = 6
        else:
            score_cf = 2

    # 资产负债率
    debt_ratio = recent_fin["资产负债率"].iloc[-1]
    if pd.isna(debt_ratio):
        score_debt = 5
    elif debt_ratio < 20:
        score_debt = 10
    elif debt_ratio < 40:
        score_debt = 7
    elif debt_ratio < 60:
        score_debt = 4
    else:
        score_debt = 0

    # 分红率（股息支付率）
    # 接口中股息支付率数据常缺失，改用「年度每股分红 / 年度每股收益」自行计算
    # 注意：财务数据中的每股收益为累计值，需取最近年报数据作为年度 EPS
    annual_eps = None
    annual_div = 0.0
    if not fin_df.empty:
        # 找最近年报（12-31）
        annual_rows = fin_df[fin_df["报告期"].dt.month == 12]
        if not annual_rows.empty:
            annual_eps = annual_rows["每股收益"].iloc[-1]
            annual_year = annual_rows["报告期"].iloc[-1].year
            # 对应该年度的分红：公告日在该年内的分红
            year_divs = div_df[
                (div_df["公告日"] >= pd.Timestamp(f"{annual_year}-01-01"))
                & (div_df["公告日"] <= pd.Timestamp(f"{annual_year}-12-31"))
            ]["每股派息"].sum()
            annual_div = year_divs

    payout = None
    if pd.notna(annual_eps) and annual_eps > 0 and annual_div > 0:
        payout = (annual_div / annual_eps) * 100

    if payout is None:
        score_payout = 5
    elif 30 <= payout <= 60:
        score_payout = 10
    elif 20 <= payout < 30 or 60 < payout <= 80:
        score_payout = 7
    else:
        score_payout = 3

    score_health = score_growth + score_cf + score_debt + score_payout

    # 3. 估值安全垫 (20分)
    pe_pct, price_pct = compute_pe_pb_percentile(price_df)

    def pct_score(pct):
        if pct is None:
            return 5
        if pct < 30:
            return 10
        elif pct < 50:
            return 7
        elif pct < 70:
            return 4
        else:
            return 1

    score_pe = pct_score(pe_pct)
    # 价格分位作为 PB 分位的代理（茅台盈利稳定，价格分位≈估值分位）
    score_pb = pct_score(price_pct)
    score_valuation = score_pe + score_pb

    # 4. 趋势与技术面 (15分)
    tech = compute_technical_score(price_df)
    score_tech = tech["位置得分"] + tech["趋势得分"]

    total = score_dy + score_health + score_valuation + score_tech

    return {
        "评估日期": latest_date,
        "最新收盘价": round(latest_price, 2),
        "股息率(%)": round(dy, 2),
        "股息率评级": dy_level,
        "股息率得分": score_dy,
        "净利润增长率(%)": round(np_growth, 2) if pd.notna(np_growth) else None,
        "净利润增长得分": score_growth,
        "现金流净利润比(%)": round(cf_ratio, 2) if pd.notna(cf_ratio) else None,
        "现金流得分": score_cf,
        "资产负债率(%)": round(debt_ratio, 2) if pd.notna(debt_ratio) else None,
        "资产负债得分": score_debt,
        "分红率(%)": round(payout, 2) if pd.notna(payout) else None,
        "年度每股分红": round(annual_div, 2) if annual_div > 0 else None,
        "年度每股收益": round(annual_eps, 2) if pd.notna(annual_eps) else None,
        "分红率得分": score_payout,
        "分红健康度总分": score_health,
        "PE历史分位(%)": round(pe_pct, 2) if pe_pct is not None else None,
        "PE分位得分": score_pe,
        "价格历史分位(%)": round(price_pct, 2) if price_pct is not None else None,
        "价格分位得分": score_pb,
        "估值安全垫总分": score_valuation,
        "技术面": tech,
        "趋势技术面总分": score_tech,
        "总得分": total,
    }


def generate_report(result: dict, output_path: Path) -> str:
    """生成 Markdown 报告。"""
    tech = result["技术面"]
    md = f"""# {STOCK_NAME}（{SYMBOL}）红利持有逻辑周度评估

> 评估日期：**{result['评估日期']}**  
> 数据来源：AKShare（新浪财经/巨潮资讯）  
> 评估框架：基于《股息类股票的持有与浮亏处理——实战手册》

---

## 一、核心指标速览

| 指标 | 数值 | 说明 |
|------|------|------|
| 最新收盘价 | **{result['最新收盘价']} 元** | 前复权 |
| 股息率（TTM） | **{result['股息率(%)']}%** | {result['股息率评级']} |
| 净利润增长率 | **{result['净利润增长率(%)']}%** | 最新季度同比 |
| 经营现金流/净利润 | **{result['现金流净利润比(%)']}%** | 盈余质量 |
| 资产负债率 | **{result['资产负债率(%)']}%** | 财务安全 |
| 分红率（派息比例） | **{result['分红率(%)']}%** | 分红可持续性 |
| PE 历史分位 | **{result['PE历史分位(%)']}%** | 5 年区间 |
| 价格历史分位 | **{result['价格历史分位(%)']}%** | 5 年区间 |
| 52 周高点比 | **{tech['52周高点比']}%** | 位置感 |

---

## 二、持有逻辑评分（满分 100）

### 1. 股息率定价（25 分）→ **{result['股息率得分']} 分**

红利投资的买入信号是「高股息率」，而非低 PE。

| 股息率区间 | 得分 | 信号 |
|-----------|------|------|
| < 1.5% | 5 | 偏贵，非红利买点 |
| 1.5% - 3% | 15 | 合理区间，可分批建仓 |
| 3% - 4% | 22 | 明显低估，重点加仓 |
| ≥ 4% | 25 | 极度低估或基本面异常，需深究 |

**当前：{result['股息率(%)']}% → {result['股息率评级']}**

---

### 2. 分红健康度（40 分）→ **{result['分红健康度总分']} 分**

> 持有阶段只跟踪这 6 件事：净利润、经营现金流、资产负债率、分红率、行业景气度、政策监管。
> 只要没有红灯，股价跌多少都不重要。

| 指标 | 数值 | 得分 | 警戒标准 |
|------|------|------|---------|
| 净利润同比增长 | {result['净利润增长率(%)']}% | {result['净利润增长得分']}/10 | 连续 2 季度下滑为红灯 |
| 经营现金流/净利润 | {result['现金流净利润比(%)']}% | {result['现金流得分']}/10 | < 50% 盈余质量差 |
| 资产负债率 | {result['资产负债率(%)']}% | {result['资产负债得分']}/10 | 显著攀升为红灯 |
| 分红率（派息比例） | {result['分红率(%)']}% | {result['分红率得分']}/10 | 突然 > 90% 不可持续 |

**小计：{result['分红健康度总分']} / 40**

---

### 3. 估值安全垫（20 分）→ **{result['估值安全垫总分']} 分**

| 指标 | 历史分位 | 得分 | 说明 |
|------|---------|------|------|
| PE 近似分位 | {result['PE历史分位(%)']}% | {result['PE分位得分']}/10 | < 30% 为安全区 |
| 价格历史分位 | {result['价格历史分位(%)']}% | {result['价格分位得分']}/10 | < 30% 为安全区 |

**小计：{result['估值安全垫总分']} / 20**

---

### 4. 趋势与技术面（15 分）→ **{result['趋势技术面总分']} 分**

| 指标 | 数值 | 得分 |
|------|------|------|
| 距 52 周高点 | {tech['52周高点比']}% | {tech['位置得分']}/8 |
| 20 周均线 | {tech['20周均线']} 元 | {tech['趋势得分']}/7 |
| 均线斜率 | {tech['均线斜率']}% | — |

**小计：{result['趋势技术面总分']} / 15**

---

## 三、综合得分与持有建议

# **{result['总得分']} / 100**

"""
    if result["总得分"] >= 80:
        md += """
**评级：🟢 强烈持有 / 积极加仓**

- 股息率与分红健康度均处于舒适区
- 估值具备安全垫，时间站在你这边
- 动作：维持仓位，分红到账后按「股息率最高原则」再投资
"""
    elif result["总得分"] >= 60:
        md += """
**评级：🟡 持有观察**

- 核心分红逻辑未变，但某项指标出现黄灯
- 动作：暂停新增加仓，持续跟踪季度财报；若浮亏 < 20%，安心收息
"""
    elif result["总得分"] >= 40:
        md += """
**评级：🟠 谨慎观察**

- 多项指标趋弱，需强制复盘买入逻辑
- 动作：逐项核对当初买入逻辑是否仍然成立；若逻辑已变，按新逻辑重新定仓
"""
    else:
        md += """
**评级：🔴 警惕 / 考虑减仓**

- 基本面或估值出现红灯信号
- 动作：默认减仓一半，然后再研究；原则：先保命，再研究
"""

    md += f"""
---

## 四、决策树自检（浮亏处理参考）

```
浮亏发生
  ↓
① 分红是否受影响？
  ├─ 否 → ② 基本面是否恶化？
  │         ├─ 否 → ③ 是系统性下跌吗？
  │         │         ├─ 是 → 【加仓】（仓位允许的话）
  │         │         └─ 否 → 【持有】
  │         └─ 是 → 【减仓 30-50%】
  └─ 是 → 分红下降幅度？
            ├─ < 20% → 【持有观察 1-2 季度】
            └─ > 20% → 【清仓】
```

**当前自检：**
- 分红健康度得分 {result['分红健康度总分']}/40 → {'基本面未恶化，股价波动只是浮动估值' if result['分红健康度总分'] >= 28 else '需关注部分指标'}
- 股息率 {result['股息率(%)']}% → {'租金照收，持有心态' if result['股息率(%)'] >= 2.0 else '股息偏低，成长属性>红利属性'}

---

> **红利投资的本质：你是「债主」，不是「股东」。**  
> 每年收租（分红），商铺估价每天在变（股价）。只要租约不违约、租金不降，估价跌多少都不重要。  
> **时间 × 分红 × 复利**，才是底层信仰。

---

*报告生成时间：{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}*
"""

    output_path.write_text(md, encoding="utf-8")
    return str(output_path)


def save_json(result: dict, output_path: Path) -> str:
    """保存原始 JSON 数据，方便后续做时间序列对比。"""
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(output_path)


def main():
    parser = argparse.ArgumentParser(description="茅台红利持有逻辑周度追踪器")
    parser.add_argument("--output-dir", default=".", help="报告输出目录")
    parser.add_argument("--json", action="store_true", help="同时输出 JSON 原始数据")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    today_str = datetime.now().strftime("%Y%m%d")
    md_path = out_dir / f"maotai_dividend_report_{today_str}.md"
    json_path = out_dir / f"maotai_dividend_data_{today_str}.json"

    print(f"[{datetime.now().strftime('%H:%M:%S')}] 开始获取数据...")
    price_df = fetch_price_data(years=5)
    print(f"  → 行情数据: {len(price_df)} 条, 最新价 {price_df['close'].iloc[-1]}")

    fin_df = fetch_financial_data(start_year=datetime.now().year - 5)
    print(f"  → 财务数据: {len(fin_df)} 条季度记录")

    div_df = fetch_dividend_data()
    print(f"  → 分红数据: {len(div_df)} 条记录")

    print(f"[{datetime.now().strftime('%H:%M:%S')}] 评估中...")
    result = evaluate(price_df, fin_df, div_df)

    md_file = generate_report(result, md_path)
    print(f"  → Markdown 报告: {md_file}")

    if args.json:
        json_file = save_json(result, json_path)
        print(f"  → JSON 数据: {json_file}")

    print(f"[{datetime.now().strftime('%H:%M:%S')}] 完成。总得分: {result['总得分']}/100")
    return result


if __name__ == "__main__":
    main()
