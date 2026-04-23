#!/usr/bin/env python3
"""
红利组合持有逻辑周度追踪器（批量版）
基于《股息类股票的持有与浮亏处理——实战手册》框架，对红利组合中的多只股票
按周收集数据并评估持有逻辑得分，输出列表式汇总清单。

依赖: pip install akshare pandas numpy
用法: python dividend_portfolio_tracker.py [--output-dir ./reports]
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

import akshare as ak


# ==================== 红利组合默认配置 ====================
# 基于《红利持有逻辑.md》中提到的类别与示例标的
DEFAULT_PORTFOLIO = {
    "600519": "贵州茅台",
    "601398": "工商银行",
    "601939": "建设银行",
    "601288": "农业银行",
    "601988": "中国银行",
    "601318": "中国平安",
    "601857": "中国石油",
    "600028": "中国石化",
    "600938": "中国海油",
    "601088": "中国神华",
    "600900": "长江电力",
    "600941": "中国移动",
    "601728": "中国电信",
    "600332": "白云山",
}


# ==================== 数据获取层 ====================

def get_sina_symbol(stock_code: str) -> str:
    """将6位A股代码转换为新浪格式。"""
    code = str(stock_code).strip()
    if code.startswith(("6", "5", "9")):
        return f"sh{code}"
    else:
        return f"sz{code}"


def fetch_price_data(stock_code: str, years: int = 5) -> pd.DataFrame:
    """获取历史日线行情（前复权）。"""
    sina_sym = get_sina_symbol(stock_code)
    start_date = (datetime.now() - timedelta(days=years * 365)).strftime("%Y%m%d")
    df = ak.stock_zh_a_daily(symbol=sina_sym, start_date=start_date, adjust="qfq")
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df


def fetch_financial_data(stock_code: str, start_year: int = 2020) -> pd.DataFrame:
    """获取财务分析指标（季度）。"""
    df = ak.stock_financial_analysis_indicator(symbol=stock_code, start_year=str(start_year))
    # 硬编码列位置（akshare 列名编码问题）
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
    for c in col_names[1:]:
        result[c] = pd.to_numeric(result[c], errors="coerce")
    result = result.dropna(subset=["报告期"]).sort_values("报告期").reset_index(drop=True)
    return result


def fetch_dividend_data(stock_code: str) -> pd.DataFrame:
    """获取历史分红数据。"""
    df = ak.stock_dividend_cninfo(symbol=stock_code)
    # 第4列 = 每10股派息（元）
    result = df.iloc[:, [0, 1, 4]].copy()
    result.columns = ["公告日", "分红类型", "每10股派息"]
    result["公告日"] = pd.to_datetime(result["公告日"], errors="coerce")
    result["每10股派息"] = pd.to_numeric(result["每10股派息"], errors="coerce")
    result = result.dropna(subset=["公告日", "每10股派息"])
    result = result[result["每10股派息"] > 0].copy()
    result["每股派息"] = result["每10股派息"] / 10.0
    result = result.sort_values("公告日").reset_index(drop=True)
    return result


# ==================== 指标计算层 ====================

def compute_dividend_yield(price_df: pd.DataFrame, div_df: pd.DataFrame) -> float:
    """计算当前股息率（TTM）。"""
    latest_price = float(price_df["close"].iloc[-1])
    if latest_price <= 0 or div_df.empty:
        return 0.0
    latest_date = price_df["date"].iloc[-1]
    one_year_ago = latest_date - timedelta(days=365)
    recent_div = div_df[div_df["公告日"] >= one_year_ago]["每股派息"].sum()
    return recent_div / latest_price * 100


def compute_pe_percentile(price_df: pd.DataFrame, fin_df: pd.DataFrame) -> tuple:
    """估算 PE 历史分位与价格历史分位。"""
    prices = price_df["close"].values
    if len(prices) < 60:
        return None, None
    current_price = prices[-1]
    price_pct = np.sum(prices <= current_price) / len(prices) * 100

    fin_copy = fin_df.copy()
    fin_copy["quarter_end"] = fin_copy["报告期"].dt.to_period("Q").dt.to_timestamp(how="end")
    fin_quarterly = fin_copy.groupby("quarter_end").last()[["每股收益"]].rename(columns={"每股收益": "eps"})

    price_df_copy = price_df.copy()
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
    """计算技术面代理指标。"""
    df = price_df.copy()
    df["ma20w"] = df["close"].rolling(window=100, min_periods=60).mean()
    latest = df.iloc[-1]
    close = latest["close"]
    ma20w = latest["ma20w"]

    year_high = df["close"].iloc[-252:].max() if len(df) >= 252 else df["close"].max()
    high_ratio = close / year_high if year_high and year_high > 0 else 1.0

    ma_slope = 0
    if len(df) >= 110 and pd.notna(ma20w):
        prev_ma = df["ma20w"].iloc[-10]
        if pd.notna(prev_ma) and prev_ma != 0:
            ma_slope = (ma20w - prev_ma) / prev_ma

    score_high = 8
    if high_ratio > 0.95:
        score_high = 2
    elif high_ratio > 0.85:
        score_high = 5
    elif high_ratio > 0.70:
        score_high = 6

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


# ==================== 评分层 ====================

def evaluate_stock(stock_code: str, stock_name: str) -> dict:
    """对单只股票进行红利持有逻辑评分。"""
    try:
        price_df = fetch_price_data(stock_code)
        fin_df = fetch_financial_data(stock_code)
        div_df = fetch_dividend_data(stock_code)
    except Exception as e:
        return {
            "代码": stock_code,
            "名称": stock_name,
            "错误": str(e),
            "总得分": 0,
        }

    latest_price = float(price_df["close"].iloc[-1])
    latest_date_dt = price_df["date"].iloc[-1]

    # 1. 股息率评分 (25分)
    dy = compute_dividend_yield(price_df, div_df)
    if dy >= 4.0:
        score_dy = 25
    elif dy >= 3.0:
        score_dy = 22
    elif dy >= 1.5:
        score_dy = 15
    else:
        score_dy = 5

    # 2. 分红健康度 (40分)
    recent_fin = fin_df.tail(4)
    if recent_fin.empty:
        recent_fin = fin_df.tail(1)

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

    cf_ratio_raw = recent_fin["现金流净利润比"].iloc[-1]
    if pd.isna(cf_ratio_raw):
        score_cf = 5
        cf_ratio = None
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

    # 分红率自行计算
    annual_eps = None
    annual_div = 0.0
    if not fin_df.empty:
        annual_rows = fin_df[fin_df["报告期"].dt.month == 12]
        if not annual_rows.empty:
            annual_eps = annual_rows["每股收益"].iloc[-1]
            annual_year = annual_rows["报告期"].iloc[-1].year
            annual_div = div_df[
                (div_df["公告日"] >= pd.Timestamp(f"{annual_year}-01-01"))
                & (div_df["公告日"] <= pd.Timestamp(f"{annual_year}-12-31"))
            ]["每股派息"].sum()

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
    pe_pct, price_pct = compute_pe_percentile(price_df, fin_df)

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

    score_valuation = pct_score(pe_pct) + pct_score(price_pct)

    # 4. 趋势技术面 (15分)
    tech = compute_technical_score(price_df)
    score_tech = tech["位置得分"] + tech["趋势得分"]

    total = score_dy + score_health + score_valuation + score_tech

    # 评级
    if total >= 80:
        rating = "[强]强烈持有"
    elif total >= 60:
        rating = "[观]持有观察"
    elif total >= 40:
        rating = "[慎]谨慎观察"
    else:
        rating = "[警]警惕"

    return {
        "代码": stock_code,
        "名称": stock_name,
        "最新价": round(latest_price, 2),
        "股息率": round(dy, 2),
        "净利润增长": round(np_growth, 2) if pd.notna(np_growth) else None,
        "现金流净利润比": round(cf_ratio, 2) if cf_ratio is not None else None,
        "资产负债率": round(debt_ratio, 2) if pd.notna(debt_ratio) else None,
        "分红率": round(payout, 2) if payout is not None else None,
        "PE分位": round(pe_pct, 2) if pe_pct is not None else None,
        "价格分位": round(price_pct, 2) if price_pct is not None else None,
        "52周高点比": tech["52周高点比"],
        "股息率得分": score_dy,
        "健康度得分": score_health,
        "估值得分": score_valuation,
        "技术面得分": score_tech,
        "总得分": total,
        "评级": rating,
    }


# ==================== 报告生成层 ====================

def generate_summary_report(results: list, output_path: Path) -> str:
    """生成列表式汇总 Markdown 报告。"""
    valid_results = [r for r in results if "错误" not in r]
    error_results = [r for r in results if "错误" in r]

    md = f"""# 红利组合持有逻辑周度评估清单

> 评估日期：**{datetime.now().strftime('%Y-%m-%d')}**  
> 数据来源：AKShare（新浪财经/巨潮资讯）  
> 评估框架：基于《股息类股票的持有与浮亏处理——实战手册》

---

## 一、组合总览

| 维度 | 数值 |
|------|------|
| 组合股票数 | **{len(results)}** 只 |
| 成功评分 | **{len(valid_results)}** 只 |
| 评分失败 | **{len(error_results)}** 只 |
| 平均得分 | **{sum(r['总得分'] for r in valid_results) / len(valid_results) if valid_results else 0:.1f}** 分 |

---

## 二、评分汇总表（按总得分降序）

"""

    sorted_results = sorted(valid_results, key=lambda x: x["总得分"], reverse=True)

    md += "| 排名 | 股票 | 最新价 | 股息率 | 净利润增长 | 现金流/净利润 | 资产负债率 | 分红率 | PE分位 | 总得分 | 评级 |\n"
    md += "|------|------|--------|--------|-----------|--------------|-----------|--------|--------|--------|------|\n"

    for i, r in enumerate(sorted_results, 1):
        np_str = f"{r['净利润增长']}%" if r['净利润增长'] is not None else "-"
        cf_str = f"{r['现金流净利润比']}%" if r['现金流净利润比'] is not None else "-"
        debt_str = f"{r['资产负债率']}%" if r['资产负债率'] is not None else "-"
        payout_str = f"{r['分红率']}%" if r['分红率'] is not None else "-"
        pe_str = f"{r['PE分位']}%" if r['PE分位'] is not None else "-"

        md += f"| {i} | {r['名称']}<br>{r['代码']} | {r['最新价']} | {r['股息率']}% | {np_str} | {cf_str} | {debt_str} | {payout_str} | {pe_str} | **{r['总得分']}** | {r['评级']} |\n"

    md += "\n---\n\n## 三、分项得分明细\n\n"
    md += "| 股票 | 股息率(25) | 健康度(40) | 估值(20) | 技术面(15) | 总分 |\n"
    md += "|------|-----------|-----------|---------|-----------|------|\n"
    for r in sorted_results:
        md += f"| {r['名称']} | {r['股息率得分']} | {r['健康度得分']} | {r['估值得分']} | {r['技术面得分']} | **{r['总得分']}** |\n"

    md += "\n---\n\n## 四、按类别分组（参考《红利持有逻辑》配置）\n\n"
    md += """
| 类别 | 建议只数 | 当前组合中的标的 |
|------|---------|-----------------|
| 金融（银行/保险） | 2-3只 | 工商银行、建设银行、农业银行、中国银行、中国平安 |
| 能源/资源 | 2-3只 | 中国石油、中国石化、中国海油、中国神华 |
| 公用事业 | 1-2只 | 长江电力 |
| 电信/运营商 | 1只 | 中国移动、中国电信 |
| 消费/医药红利 | 1-2只 | 贵州茅台、白云山 |

> **组合纪律**：单只不超过 15%，单一行业不超过 30%，合计 8-12 只。

"""

    if error_results:
        md += "\n---\n\n## 五、数据获取失败的标的\n\n"
        md += "| 代码 | 名称 | 错误信息 |\n"
        md += "|------|------|---------|\n"
        for r in error_results:
            md += f"| {r['代码']} | {r['名称']} | {r['错误']} |\n"

    md += f"""
---

## 六、持有建议速查

| 得分区间 | 建议动作 |
|---------|---------|
| ≥ 80 分 | 🟢 强烈持有 / 积极加仓，分红再投资 |
| 60-79 分 | 🟡 持有观察，暂停新增仓，安心收息 |
| 40-59 分 | 🟠 谨慎观察，强制复盘买入逻辑 |
| < 40 分 | 🔴 警惕，默认减仓一半再研究 |

---

> **红利投资的本质：你是「债主」，不是「股东」。**  
> 时间 × 分红 × 复利，才是底层信仰。

---

*报告生成时间：{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}*
"""

    output_path.write_text(md, encoding="utf-8")
    return str(output_path)


def save_json(results: list, output_path: Path) -> str:
    output_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(output_path)


# ==================== 主入口 ====================

def main():
    parser = argparse.ArgumentParser(description="红利组合持有逻辑周度追踪器")
    parser.add_argument("--output-dir", default=".", help="报告输出目录")
    parser.add_argument("--json", action="store_true", help="同时输出 JSON 原始数据")
    parser.add_argument("--codes", nargs="+", help="自定义股票代码列表（空格分隔，如 600519 601398）")
    parser.add_argument("--names", nargs="+", help="自定义股票名称列表（与 --codes 一一对应）")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    today_str = datetime.now().strftime("%Y%m%d")
    md_path = out_dir / f"dividend_portfolio_report_{today_str}.md"
    json_path = out_dir / f"dividend_portfolio_data_{today_str}.json"

    # 确定股票列表
    if args.codes and args.names and len(args.codes) == len(args.names):
        portfolio = dict(zip(args.codes, args.names))
    elif args.codes:
        portfolio = {c: c for c in args.codes}
    else:
        portfolio = DEFAULT_PORTFOLIO

    print(f"[{datetime.now().strftime('%H:%M:%S')}] 开始评估红利组合，共 {len(portfolio)} 只标的...")
    results = []
    for idx, (code, name) in enumerate(portfolio.items(), 1):
        print(f"  [{idx}/{len(portfolio)}] 正在分析 {name} ({code})...")
        start = time.time()
        result = evaluate_stock(code, name)
        elapsed = time.time() - start
        if "错误" in result:
            print(f"    [WARN] 失败: {result['错误'][:60]} ({elapsed:.1f}s)")
        else:
            print(f"    [OK] 得分: {result['总得分']}/100 | 股息率: {result['股息率']}% | 评级: {result['评级']} ({elapsed:.1f}s)")
        results.append(result)

    print(f"[{datetime.now().strftime('%H:%M:%S')}] 生成报告中...")
    md_file = generate_summary_report(results, md_path)
    print(f"  → Markdown 报告: {md_file}")

    if args.json:
        json_file = save_json(results, json_path)
        print(f"  → JSON 数据: {json_file}")

    valid = [r for r in results if "错误" not in r]
    avg_score = sum(r["总得分"] for r in valid) / len(valid) if valid else 0
    print(f"[{datetime.now().strftime('%H:%M:%S')}] 完成。成功 {len(valid)}/{len(results)}，平均得分: {avg_score:.1f}")


if __name__ == "__main__":
    main()
