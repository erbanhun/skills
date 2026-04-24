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
    # 列5=每股净资产(调整后), 用于周期股PB计算
    key_cols = [0, 2, 5, 7, 25, 32, 61, 65]
    col_names = [
        "报告期",
        "每股收益",
        "每股净资产",
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
    # 第0列=公告日, 第4列=每10股派息（元）, 第6列=除权除息日
    result = df.iloc[:, [0, 1, 4, 6]].copy()
    result.columns = ["公告日", "分红类型", "每10股派息", "除权除息日"]
    result["公告日"] = pd.to_datetime(result["公告日"], errors="coerce")
    result["除权除息日"] = pd.to_datetime(result["除权除息日"], errors="coerce")
    result["每10股派息"] = pd.to_numeric(result["每10股派息"], errors="coerce")
    result = result.dropna(subset=["公告日", "每10股派息"])
    result = result[result["每10股派息"] > 0].copy()
    result["每股派息"] = result["每10股派息"] / 10.0
    result = result.sort_values("公告日").reset_index(drop=True)
    return result


def compute_ttm_eps_series(fin_df: pd.DataFrame) -> pd.DataFrame:
    """将季度累计 EPS 转为单季度 EPS，并计算 TTM EPS 序列。

    财务数据中的每股收益为累计值（Q1=1-3月累计，Q2=1-6月累计，
    Q3=1-9月累计，Q4=全年累计）。本函数先拆分为单季度 EPS，
    再滚动求和得到 TTM EPS，用于准确的 PE 计算。
    """
    df = fin_df.copy()
    df = df.sort_values("报告期").reset_index(drop=True)
    df["year"] = df["报告期"].dt.year
    df["month"] = df["报告期"].dt.month

    def month_to_quarter(m):
        if m == 3:
            return 1
        elif m == 6:
            return 2
        elif m == 9:
            return 3
        elif m == 12:
            return 4
        else:
            return None

    df["quarter"] = df["month"].apply(month_to_quarter)
    df = df.dropna(subset=["quarter"])

    df["eps_single"] = None
    for idx, row in df.iterrows():
        y, q = row["year"], row["quarter"]
        if q == 1:
            df.at[idx, "eps_single"] = row["每股收益"]
        else:
            prev_rows = df[(df["year"] == y) & (df["quarter"] == q - 1)]
            if not prev_rows.empty:
                prev_eps = prev_rows["每股收益"].iloc[-1]
                df.at[idx, "eps_single"] = row["每股收益"] - prev_eps
            else:
                df.at[idx, "eps_single"] = None

    df["eps_single"] = pd.to_numeric(df["eps_single"], errors="coerce")

    df = df.sort_values("报告期").reset_index(drop=True)
    ttm_values = []
    for i in range(len(df)):
        valid_single = df.iloc[: i + 1]["eps_single"].dropna()
        if len(valid_single) >= 4:
            ttm = valid_single.iloc[-4:].sum()
        else:
            ttm = None
        ttm_values.append(ttm)
    df["eps_ttm"] = ttm_values
    return df


# ==================== 指标计算层 ====================

def compute_dividend_yield(price_df: pd.DataFrame, div_df: pd.DataFrame) -> float:
    """计算当前股息率（TTM）。使用除权除息日筛选，避免公告日边界错配。"""
    latest_price = float(price_df["close"].iloc[-1])
    if latest_price <= 0 or div_df.empty:
        return 0.0
    latest_date = price_df["date"].iloc[-1]
    one_year_ago = latest_date - timedelta(days=365)

    # 优先使用除权除息日筛选；若缺失，回退到公告日
    if "除权除息日" in div_df.columns and div_df["除权除息日"].notna().any():
        recent_div = div_df[
            (div_df["除权除息日"].notna()) &
            (div_df["除权除息日"] >= one_year_ago)
        ]["每股派息"].sum()
    else:
        recent_div = div_df[div_df["公告日"] >= one_year_ago]["每股派息"].sum()
    return recent_div / latest_price * 100


def compute_pe_percentile(price_df: pd.DataFrame, fin_df: pd.DataFrame) -> tuple:
    """估算 PE 历史分位与价格历史分位。使用 TTM EPS 计算 PE，避免季度累计值偏差。"""
    prices = price_df["close"].values
    if len(prices) < 60:
        return None, None
    current_price = prices[-1]
    price_pct = np.sum(prices <= current_price) / len(prices) * 100

    # 使用 TTM EPS 计算历史 PE 序列
    fin_ttm = compute_ttm_eps_series(fin_df)
    fin_ttm = fin_ttm[fin_ttm["eps_ttm"].notna() & (fin_ttm["eps_ttm"] > 0)].copy()

    if fin_ttm.empty:
        return None, price_pct

    fin_ttm["quarter_end"] = fin_ttm["报告期"].dt.to_period("Q").dt.to_timestamp(how="end")
    fin_quarterly = fin_ttm.groupby("quarter_end").last()[["eps_ttm"]].rename(columns={"eps_ttm": "eps"})

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


def compute_pb_percentile(price_df: pd.DataFrame, fin_df: pd.DataFrame) -> float:
    """估算 PB 历史分位。周期股不适用 PE，改用 PB 分位 + 价格分位。"""
    prices = price_df["close"].values
    if len(prices) < 60:
        return None
    current_price = prices[-1]

    fin_pb = fin_df[fin_df["每股净资产"].notna() & (fin_df["每股净资产"] > 0)].copy()
    if fin_pb.empty:
        return None

    fin_pb["quarter_end"] = fin_pb["报告期"].dt.to_period("Q").dt.to_timestamp(how="end")
    fin_quarterly = fin_pb.groupby("quarter_end").last()[["每股净资产"]].rename(columns={"每股净资产": "bps"})

    price_df_copy = price_df.copy()
    price_df_copy["quarter_end"] = price_df_copy["date"].dt.to_period("Q").dt.to_timestamp(how="end")
    merged = price_df_copy.merge(fin_quarterly, on="quarter_end", how="left")
    merged["bps"] = merged["bps"].ffill()
    merged = merged[merged["bps"] > 0].copy()
    merged["pb"] = merged["close"] / merged["bps"]

    pb_pct = None
    if not merged.empty and len(merged) > 60:
        current_pb = merged["pb"].iloc[-1]
        pb_pct = np.sum(merged["pb"].values <= current_pb) / len(merged) * 100

    return pb_pct


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

    # 金融股现金流/净利润比不适用（银行经营现金流含负债端科目，方法论错误）
    financial_stocks = {"601398", "601939", "601288", "601988", "601318"}
    is_financial = stock_code in financial_stocks

    cf_ratio_raw = recent_fin["现金流净利润比"].iloc[-1]
    if is_financial:
        score_cf = 5  # 中性分，不加分不减分
        cf_ratio = None  # 不在报告中显示异常数值
    elif pd.isna(cf_ratio_raw):
        score_cf = 5
        cf_ratio = None
    else:
        cf_ratio = cf_ratio_raw * 100
        # 负现金流直接得 0 分（红线）
        if cf_ratio < 0:
            score_cf = 0
        elif cf_ratio > 200:
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

    # 周期股（石油、煤炭）不适用 PE，改用 PB 分位 + 价格分位
    cyclical_stocks = {"601857", "600028", "600938", "601088"}
    is_cyclical = stock_code in cyclical_stocks

    if is_cyclical:
        pb_pct = compute_pb_percentile(price_df, fin_df)
        pe_pct = None
        # 价格分位单独计算
        prices = price_df["close"].values
        price_pct = None
        if len(prices) >= 60:
            current_price = prices[-1]
            price_pct = np.sum(prices <= current_price) / len(prices) * 100
        score_valuation = pct_score(pb_pct) + pct_score(price_pct)
    else:
        pe_pct, price_pct = compute_pe_percentile(price_df, fin_df)
        pb_pct = None
        score_valuation = pct_score(pe_pct) + pct_score(price_pct)

    # 4. 趋势技术面 (15分)
    tech = compute_technical_score(price_df)
    score_tech = tech["位置得分"] + tech["趋势得分"]

    # 盈利趋势检测（连续季度净利润增长率为负）
    growth_series = recent_fin["净利润增长率"].dropna().iloc[-3:]
    earning_trend = "正常"
    if len(growth_series) >= 2:
        negative_count = (growth_series < 0).sum()
        if negative_count >= 3:
            earning_trend = "红灯（连续3季度负增长）"
        elif negative_count >= 2:
            earning_trend = "黄灯（连续2季度负增长）"

    # 股息率陷阱检测：价格分位高 + 净利润负增长
    div_trap = False
    if price_pct is not None and price_pct > 90 and np_growth is not None and np_growth < 0:
        div_trap = True

    # 分红率超额检测（优化版）
    payout_excess = False
    payout_risk = False
    if payout is not None and payout > 100:
        payout_excess = True
        # 结合现金流与负债判断是否为真正风险（借债分红）
        if cf_ratio is not None and cf_ratio < 0:
            payout_risk = True  # 经营现金流为负仍超额分红，借债分红嫌疑高
        elif debt_ratio is not None and debt_ratio > 60:
            payout_risk = True  # 高负债+超额分红，需警惕

    # 价格高位风险提示
    price_high_alert = False
    if price_pct is not None and price_pct > 95:
        price_high_alert = True

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

    # 建议动作（增强40-59分区分度）
    if total >= 80:
        action = "强烈持有 / 积极加仓，分红再投资"
    elif total >= 60:
        action = "持有观察，暂停新增仓，安心收息"
    elif total >= 50:
        action = "谨慎观察：暂停新增仓，1个月内强制复盘买入逻辑"
    elif total >= 40:
        action = "警惕：建议减仓至半仓，2周内复盘或考虑调出"
    else:
        action = "默认减仓一半，暂停一切新增仓，深入研究后再决策"

    # 数据异常检测（三角验证第一层）
    data_anomaly = []
    if is_financial and dy is not None and (dy < 2.0 or dy > 7.0):
        data_anomaly.append(f"股息率{dy}%离群，银行股正常区间3.5%-5.5%，建议Wind/交易所交叉验证")
    if not is_financial and cf_ratio is not None and (cf_ratio > 500 or cf_ratio < -50):
        data_anomaly.append(f"现金流/净利润比{cf_ratio}%异常，建议核实AKShare列映射或数据源")

    # 自由现金流覆盖分红（年度）— 金融股不适用
    fcf_coverage = None
    if not is_financial and not fin_df.empty:
        annual_ocf_rows = fin_df[fin_df["报告期"].dt.month == 12]
        if not annual_ocf_rows.empty:
            annual_ocf = annual_ocf_rows["每股经营现金流"].iloc[-1]
            if pd.notna(annual_ocf) and annual_div > 0:
                fcf_coverage = annual_ocf / annual_div

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
        "PB分位": round(pb_pct, 2) if pb_pct is not None else None,
        "价格分位": round(price_pct, 2) if price_pct is not None else None,
        "52周高点比": tech["52周高点比"],
        "股息率得分": score_dy,
        "健康度得分": score_health,
        "估值得分": score_valuation,
        "技术面得分": score_tech,
        "总得分": total,
        "评级": rating,
        "建议动作": action,
        "盈利趋势": earning_trend,
        "股息率陷阱": div_trap,
        "分红超额": payout_excess,
        "分红超额风险": payout_risk,
        "价格高位预警": price_high_alert,
        "金融股": is_financial,
        "周期股": is_cyclical,
        "数据异常": data_anomaly,
        "自由现金流覆盖分红": round(fcf_coverage, 2) if fcf_coverage is not None else None,
    }


# ==================== 报告生成层 ====================

def generate_summary_report(results: list, output_path: Path) -> str:
    """生成列表式汇总 Markdown 报告。"""
    valid_results = [r for r in results if "错误" not in r]
    error_results = [r for r in results if "错误" in r]

    # 组合层指标计算
    avg_dy = sum(r['股息率'] for r in valid_results) / len(valid_results) if valid_results else 0
    pe_values = [r['PE分位'] for r in valid_results if r.get('PE分位') is not None]
    pb_values = [r['PB分位'] for r in valid_results if r.get('PB分位') is not None]
    avg_pe = sum(pe_values) / len(pe_values) if pe_values else None
    avg_pb = sum(pb_values) / len(pb_values) if pb_values else None

    # 10年期国债收益率
    bond_10y = None
    try:
        bond_start = (datetime.now() - timedelta(days=7)).strftime('%Y%m%d')
        bond_end = datetime.now().strftime('%Y%m%d')
        bond_df = ak.bond_china_yield(start_date=bond_start, end_date=bond_end)
        gov_rows = bond_df[bond_df.iloc[:, 0].str.contains('国债', na=False)]
        bond_10y = float(gov_rows.iloc[-1, 8]) if not gov_rows.empty else None
    except Exception:
        bond_10y = None

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

## 一（附）. 组合层指标

| 指标 | 数值 | 说明 |
|------|------|------|
| 组合等权股息率 | **{avg_dy:.2f}%** | 各股票股息率的等权平均 |
| 组合等权PE分位 | {f'{avg_pe:.1f}%' if avg_pe is not None else 'N/A'} | 非周期股PE分位等权平均 |
| 组合等权PB分位 | {f'{avg_pb:.1f}%' if avg_pb is not None else 'N/A'} | 周期股PB分位等权平均 |
| 10年期国债收益率 | {f'{bond_10y:.2f}%' if bond_10y is not None else 'N/A'} | 中债国债到期收益率 |
| 利差（股息率-国债） | {f'{(avg_dy - bond_10y):.2f}%' if bond_10y is not None else 'N/A'} | 红利相对无风险利率的溢价 |

> **利率敏感度说明**：红利股对10年期国债收益率最敏感。当前国债{bond_10y if bond_10y else '—'}%，若利率上行，高股息吸引力下降；若利率下行，红利配置价值提升。

---

## 二、评分汇总表（按总得分降序）

"""

    sorted_results = sorted(valid_results, key=lambda x: x["总得分"], reverse=True)

    md += "| 排名 | 股票 | 最新价 | 股息率 | 净利润增长 | 现金流/净利润 | 资产负债率 | 分红率 | 估值分位(PB/PE) | 总得分 | 评级 | 建议动作 |\n"
    md += "|------|------|--------|--------|-----------|--------------|-----------|--------|----------------|--------|------|---------|\n"

    for i, r in enumerate(sorted_results, 1):
        np_str = f"{r['净利润增长']}%" if r['净利润增长'] is not None else "-"
        if r.get("金融股"):
            cf_str = "N/A（金融股）"
        elif r['现金流净利润比'] is not None:
            cf_str = f"{r['现金流净利润比']}%"
        else:
            cf_str = "-"
        debt_str = f"{r['资产负债率']}%" if r['资产负债率'] is not None else "-"
        payout_str = f"{r['分红率']}%" if r['分红率'] is not None else "-"
        if r.get("周期股"):
            val_str = f"PB{r['PB分位']}%" if r['PB分位'] is not None else "-"
        else:
            val_str = f"PE{r['PE分位']}%" if r['PE分位'] is not None else "-"
        action_str = r.get('建议动作', '-')

        md += f"| {i} | {r['名称']}<br>{r['代码']} | {r['最新价']} | {r['股息率']}% | {np_str} | {cf_str} | {debt_str} | {payout_str} | {val_str} | **{r['总得分']}** | {r['评级']} | {action_str} |\n"

    md += "\n---\n\n## 三、分项得分明细\n\n"
    md += "| 股票 | 股息率(25) | 健康度(40) | 估值(20) | 技术面(15) | 总分 |\n"
    md += "|------|-----------|-----------|---------|-----------|------|\n"
    for r in sorted_results:
        md += f"| {r['名称']} | {r['股息率得分']} | {r['健康度得分']} | {r['估值得分']} | {r['技术面得分']} | **{r['总得分']}** |\n"

    # 风险预警与特殊标注
    alerts = []
    warnings = []
    notes = []
    for r in sorted_results:
        if r.get("股息率陷阱"):
            alerts.append(f"| {r['名称']} | 🚨 股息率陷阱 | 价格分位 {r['价格分位']}% + 净利润增长 {r['净利润增长']}% |")
        if r.get("价格高位预警"):
            warnings.append(f"| {r['名称']} | ⚠️ 价格高位 | 价格分位 {r['价格分位']}%，接近 5 年最高 |")
        trend = r.get("盈利趋势", "正常")
        if "红灯" in trend:
            alerts.append(f"| {r['名称']} | 🚨 {trend} | 盈利趋势恶化，PE 可能被动升高 |")
        elif "黄灯" in trend:
            warnings.append(f"| {r['名称']} | ⚠️ {trend} | 盈利趋势恶化，PE 可能被动升高 |")
        if r.get("分红超额风险"):
            alerts.append(f"| {r['名称']} | 🚨 超额分红（高风险） | 分红率 {r['分红率']}% + 现金流为负/负债高企，有借债分红嫌疑 |")
        elif r.get("分红超额"):
            notes.append(f"| {r['名称']} | ℹ️ 超额分红（中性） | 分红率 {r['分红率']}%，现金流充沛/负债低，属资本开支周期尾端回报 |")
        if r.get("金融股"):
            notes.append(f"| {r['名称']} | ℹ️ 金融股现金流 | 银行/保险现金流结构特殊，该指标已剔除，不参与健康度评分 |")
        if r.get("现金流净利润比") is not None and r["现金流净利润比"] < 0:
            alerts.append(f"| {r['名称']} | 🚨 经营现金流为负 | 现金流 {r['现金流净利润比']}%，红利投资生命线告急 |")
        # 数据异常
        for anomaly in r.get("数据异常", []):
            alerts.append(f"| {r['名称']} | ⚠️ 数据异常 | {anomaly} |")

    md += "\n---\n\n## 四、风险预警与特殊标注\n\n"
    if alerts:
        md += "### 🔴 红灯预警\n\n"
        md += "| 股票 | 类型 | 说明 |\n"
        md += "|------|------|------|\n"
        for a in alerts:
            md += a + "\n"
        md += "\n"
    if warnings:
        md += "### 🟡 黄灯提示\n\n"
        md += "| 股票 | 类型 | 说明 |\n"
        md += "|------|------|------|\n"
        for w in warnings:
            md += w + "\n"
        md += "\n"
    if notes:
        md += "### 🟢 备注说明\n\n"
        md += "| 股票 | 类型 | 说明 |\n"
        md += "|------|------|------|\n"
        for n in notes:
            md += n + "\n"
        md += "\n"
    # 行业共振过滤器
    sector_map = {
        "601398": "金融", "601939": "金融", "601288": "金融", "601988": "金融", "601318": "金融",
        "601857": "能源", "600028": "能源", "600938": "能源", "601088": "能源",
        "600900": "公用事业",
        "600941": "电信", "601728": "电信",
        "600519": "消费", "600332": "消费",
    }
    sector_risks = {}
    for r in sorted_results:
        sector = sector_map.get(r["代码"], "其他")
        if "红灯" in r.get("盈利趋势", ""):
            sector_risks.setdefault((sector, "连续负增长"), []).append(r["名称"])
        if r.get("股息率陷阱"):
            sector_risks.setdefault((sector, "股息率陷阱"), []).append(r["名称"])

    resonance = []
    for (sector, risk_type), stocks in sector_risks.items():
        if len(stocks) >= 2:
            resonance.append(f"| {sector} | {', '.join(stocks)} | {risk_type}（{len(stocks)}只同时触发）|")

    if resonance:
        md += "\n### ⚠️ 行业共振风险（同行业多股同时触发）\n\n"
        md += "| 行业 | 涉及股票 | 共振风险 |\n"
        md += "|------|---------|---------|\n"
        for row in resonance:
            md += row + "\n"
        md += "\n> **说明**：同行业多股同时触发同类风险，大概率是宏观/行业周期因素（如油价、煤价回落），不应视为多个独立微观风险。\n\n"

    if not alerts and not warnings and not notes and not resonance:
        md += "> 当前组合无特殊预警。\n\n"

    # 组合纪律检查
    sector_counts = {}
    for r in sorted_results:
        sector = sector_map.get(r["代码"], "其他")
        sector_counts[sector] = sector_counts.get(sector, 0) + 1
    total_valid = len(sorted_results)

    discipline_rows = []
    for sector, count in sector_counts.items():
        pct = count / total_valid * 100
        if pct > 30:
            discipline_rows.append(f"| {sector} | {count}/{total_valid} = **{pct:.1f}%** | ❌ 已超标（应≤30%） |")
        elif pct > 25:
            discipline_rows.append(f"| {sector} | {count}/{total_valid} = {pct:.1f}% | ⚠️ 接近上限 |")

    md += "\n---\n\n## 五、组合纪律与类别配置\n\n"

    if discipline_rows:
        md += "### 🔴 组合纪律报警\n\n"
        md += "| 行业 | 当前占比 | 状态 |\n"
        md += "|------|---------|------|\n"
        for row in discipline_rows:
            md += row + "\n"
        md += "\n"

    md += """
| 类别 | 建议只数 | 当前组合中的标的 |
|------|---------|-----------------|
| 金融（银行/保险） | 2-3只 | 工商银行、建设银行、农业银行、中国银行、中国平安 |
| 能源/资源 | 2-3只 | 中国石油、中国石化、中国海油、中国神华 |
| 公用事业 | 1-2只 | 长江电力 |
| 电信/运营商 | 1只 | 中国移动、中国电信 |
| 消费/医药红利 | 1-2只 | 白云山 |
| 类红利成长（不占红利配额） | — | 贵州茅台 |

> **组合纪律**：单只不超过 15%，单一行业不超过 30%，合计 8-12 只。
> **注意**：贵州茅台为「类红利成长」标的，投资逻辑以品牌护城河+成长为主，与纯红利股波动特征不同，**不占红利配额**。

"""

    if error_results:
        md += "\n---\n\n## 六、数据获取失败的标的\n\n"
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

## 七、前瞻判断框架（非量化，需人工跟踪）

> 以下因素不进入评分模型，但决定红利组合**长期回报**的真正方向。建议每月更新一次定性判断。

| 行业/主题 | 关键前瞻问题 | 当前观察窗口 | 数据来源建议 |
|-----------|-------------|-------------|-------------|
| 银行 | 净息差是否见底？ | 季度财报：净息差(NIM)环比变化 | 银保监会统计、各行IR |
| 保险 | 新业务价值(NBV)拐点？ | 月度保费收入增速、Q1/Q3 NBV | 上市险企月度保费公告 |
| 煤炭 | 煤价明年走势？ | 动力煤长协价、港口库存、进口政策 | 秦皇岛动力煤价、CCTD指数 |
| 石油 | 油价中枢判断？ | 布伦特/WTI走势、OPEC+产量政策 | EIA库存、IEA月报、OPEC会议 |
| 利率 | 10年期国债方向？ | 央行MLF/LPR操作、CPI/PPI数据 | 中债估值、央行货币政策报告 |
| 宏观 | 红利风格是否拥挤？ | 红利ETF份额变化、主动基金配置比例 | 基金季报、ETF申赎数据 |

> **使用方式**：若某行业前瞻判断恶化（如银行净息差继续下行、煤价跌破长协下限），即使当前评分仍在"持有观察"区间，也应提前启动减仓预案。

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
