#!/usr/bin/env python3
"""
A股板块主线追踪器（Sector Leader Tracker）
基于用户提供的指南针活跃市值数据判断多头区间，扫描行业+概念板块，
通过四维评分识别主线板块，挖掘龙头候选，并预警主线见顶。

依赖: pip install akshare pandas numpy
用法: python sector_leader_tracker.py --compass-csv ./data/compass.csv --output-dir ./reports
"""

import argparse
import json
import os
import sqlite3
import sys
import time
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# 尝试导入 akshare，未安装则提示
try:
    import akshare as ak
except ImportError:
    print("[ERROR] 请先安装 akshare: pip install akshare")
    sys.exit(1)

# ==================== 配置常量 ====================

# 多头区间判定规则（用户指定）
BULL_TRIGGER_SINGLE = 4.0       # 单日涨幅≥4% 开启多头
BULL_TRIGGER_2DAY = 4.0         # 连续2日累计涨幅≥4% 开启多头
BULL_END_THRESHOLD = -2.3       # 单日跌幅≤-2.3% 结束多头

# 板块评分权重
WEIGHT_MOMENTUM = 30
WEIGHT_FUND = 30
WEIGHT_SENTIMENT = 25
WEIGHT_STRUCTURE = 15

# 主线确认阈值
LEADER_CANDIDATE_SCORE = 75
LEADER_CONFIRM_DAYS = 5         # 近10个交易日中出现≥5次
LEADER_CONFIRM_WINDOW = 10

# 概念板块过滤黑名单（技术性概念，非题材性）
CONCEPT_BLACKLIST_KEYWORDS = [
    "涨停", "连板", "炸板", "跌停", "次新股", "新股", "破净",
    "ST", "退市", "预亏", "预盈", "高送转", "低价股", "高价股",
    "昨日", "近日", "今日", "本周", "本月", "昨日涨停", "昨日连板",
    "融资融券", "质押", "回购", "增持", "减持", "员工持股",
    "国资", "社保", "QFII", "基金重仓", "信托", "保险",
    "MSCI", "标普", "富时", "深股通", "沪股通", "港股通",
    "转融通", "债转股", "垃圾分类", "兜底增持",
]

# 行业板块强制保留（即使评分不高也显示）
CORE_INDUSTRIES = [
    "半导体", "银行", "证券", "保险", "房地产开发", "电力",
    "煤炭开采", "石油开采", "有色金属", "汽车整车", "白酒",
    "医疗器械", "生物制品", "中药", "化学制药", "消费电子",
    "通信设备", "计算机设备", "光伏设备", "电池", "风电设备",
    "军工", "钢铁", "水泥", "建筑装饰", "航运港口", "物流",
    "农产品", "食品加工", "饮料", "家电", "纺织", "造纸",
]


# ==================== SQLite 数据库层 ====================

def init_db(db_path: Path):
    """初始化 SQLite 数据库。"""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    c = conn.cursor()

    c.execute("""
        CREATE TABLE IF NOT EXISTS market_regime (
            date TEXT PRIMARY KEY,
            regime TEXT,
            compass_value REAL,
            compass_change_pct REAL,
            bull_trigger_reason TEXT,
            hs300_price REAL,
            hs300_ma20 REAL,
            hs300_above_ma20 INTEGER,
            created_at TEXT
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS sector_scores (
            date TEXT,
            sector_name TEXT,
            sector_type TEXT,
            change_20d REAL,
            change_5d REAL,
            fund_flow_5d REAL,
            zt_count INTEGER,
            score_momentum INTEGER,
            score_fund INTEGER,
            score_sentiment INTEGER,
            score_structure INTEGER,
            total_score INTEGER,
            is_leader INTEGER,
            PRIMARY KEY (date, sector_name)
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS sector_leaders (
            date TEXT,
            sector_name TEXT,
            stock_code TEXT,
            stock_name TEXT,
            leader_type TEXT,
            change_20d REAL,
            market_cap REAL,
            tags TEXT,
            PRIMARY KEY (date, sector_name, stock_code)
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS leading_sectors_history (
            date TEXT,
            sector_name TEXT,
            status TEXT,
            score INTEGER,
            PRIMARY KEY (date, sector_name)
        )
    """)

    conn.commit()
    conn.close()


def save_market_regime(db_path: Path, data: dict):
    conn = sqlite3.connect(str(db_path))
    c = conn.cursor()
    c.execute("""
        INSERT OR REPLACE INTO market_regime
        (date, regime, compass_value, compass_change_pct, bull_trigger_reason,
         hs300_price, hs300_ma20, hs300_above_ma20, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        data["date"], data["regime"], data["compass_value"],
        data["compass_change_pct"], data["bull_trigger_reason"],
        data["hs300_price"], data["hs300_ma20"],
        data["hs300_above_ma20"], datetime.now().isoformat()
    ))
    conn.commit()
    conn.close()


def save_sector_scores(db_path: Path, date_str: str, sectors: list):
    conn = sqlite3.connect(str(db_path))
    c = conn.cursor()
    for s in sectors:
        c.execute("""
            INSERT OR REPLACE INTO sector_scores
            (date, sector_name, sector_type, change_20d, change_5d,
             fund_flow_5d, zt_count, score_momentum, score_fund,
             score_sentiment, score_structure, total_score, is_leader)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            date_str, s["name"], s["type"], s.get("change_20d"),
            s.get("change_5d"), s.get("fund_flow_5d"), s.get("zt_count", 0),
            s.get("score_momentum", 0), s.get("score_fund", 0),
            s.get("score_sentiment", 0), s.get("score_structure", 0),
            s.get("total_score", 0), 1 if s.get("is_leader") else 0
        ))
    conn.commit()
    conn.close()


def get_recent_regime(db_path: Path, days: int = 30) -> pd.DataFrame:
    """获取最近N天的市场状态历史。"""
    conn = sqlite3.connect(str(db_path))
    df = pd.read_sql_query(f"""
        SELECT * FROM market_regime
        ORDER BY date DESC LIMIT {days}
    """, conn)
    conn.close()
    return df


def get_sector_history(db_path: Path, sector_name: str, days: int = 30) -> pd.DataFrame:
    """获取某板块最近N天的评分历史。"""
    conn = sqlite3.connect(str(db_path))
    df = pd.read_sql_query(f"""
        SELECT * FROM sector_scores
        WHERE sector_name = ?
        ORDER BY date DESC LIMIT {days}
    """, conn, params=(sector_name,))
    conn.close()
    return df


# ==================== 指南针活跃市值处理 ====================

def load_compass_data(csv_path: Path) -> pd.DataFrame:
    """读取指南针活跃市值 CSV。"""
    if not csv_path.exists():
        raise FileNotFoundError(f"指南针活跃市值 CSV 不存在: {csv_path}")

    df = pd.read_csv(csv_path)
    # 支持多种列名
    date_cols = [c for c in df.columns if "date" in c.lower() or "日期" in c]
    value_cols = [c for c in df.columns if "active" in c.lower() or "market" in c.lower()
                  or "市值" in c or "value" in c.lower()]

    if not date_cols or not value_cols:
        raise ValueError(
            f"CSV 格式错误。需要包含日期列和活跃市值列，当前列名: {df.columns.tolist()}"
        )

    df = df.rename(columns={date_cols[0]: "date", value_cols[0]: "active_market_cap"})
    df["date"] = pd.to_datetime(df["date"])
    df["active_market_cap"] = pd.to_numeric(df["active_market_cap"], errors="coerce")
    df = df.dropna().sort_values("date").reset_index(drop=True)
    return df


def detect_bull_regime(compass_df: pd.DataFrame, target_date: datetime) -> dict:
    """
    多头区间判定（用户指定规则）：
    - 开启：单日涨幅>=4% 或 连续2日累计>=4%
    - 维持：已开启后，单日跌幅 > -2.3%
    - 结束：单日跌幅 <= -2.3%
    """
    target_date = pd.Timestamp(target_date).normalize()

    # 只取 target_date 及之前的数据
    df = compass_df[compass_df["date"] <= target_date].copy()
    if len(df) < 2:
        return {
            "is_bull": False,
            "regime": "数据不足",
            "compass_value": None,
            "compass_change_pct": None,
            "bull_days": 0,
            "trigger_reason": None,
        }

    df["change_pct"] = df["active_market_cap"].pct_change() * 100
    df["change_2d_pct"] = df["active_market_cap"].pct_change(2) * 100

    # 从最近往前回溯，找到当前的状态
    latest = df.iloc[-1]
    latest_change = latest["change_pct"]
    latest_2d_change = latest["change_2d_pct"]
    latest_value = latest["active_market_cap"]

    # 逐日回溯判定状态
    regime_list = []
    in_bull = False
    bull_start_idx = None

    for i, row in df.iterrows():
        change = row["change_pct"]
        change_2d = row["change_2d_pct"]

        if not in_bull:
            # 尝试开启多头
            if (change >= BULL_TRIGGER_SINGLE or
                (not pd.isna(change_2d) and change_2d >= BULL_TRIGGER_2DAY)):
                in_bull = True
                bull_start_idx = i
                reason = "单日涨幅≥4%" if change >= BULL_TRIGGER_SINGLE else "连续2日累计涨幅≥4%"
                regime_list.append({
                    "date": row["date"], "regime": "bull", "reason": reason,
                    "change": change, "change_2d": change_2d
                })
            else:
                regime_list.append({
                    "date": row["date"], "regime": "bear",
                    "change": change, "change_2d": change_2d
                })
        else:
            # 维持或结束多头
            if change <= BULL_END_THRESHOLD:
                in_bull = False
                bull_start_idx = None
                regime_list.append({
                    "date": row["date"], "regime": "bear_end",
                    "reason": f"单日跌幅≤{BULL_END_THRESHOLD}%",
                    "change": change, "change_2d": change_2d
                })
            else:
                regime_list.append({
                    "date": row["date"], "regime": "bull",
                    "change": change, "change_2d": change_2d
                })

    # 取 target_date 当天的状态
    target_regime = [r for r in regime_list if r["date"] == target_date]
    if not target_regime:
        # target_date 不在数据中，取最近一天的状态
        target_regime = [regime_list[-1]]

    current = target_regime[0]
    is_bull = current["regime"] == "bull"

    # 计算多头持续天数
    bull_days = 0
    if is_bull:
        # 从 target_date 往前数，连续多少天是 bull
        for r in reversed(regime_list):
            if r["date"] > target_date:
                continue
            if r["regime"] == "bull":
                bull_days += 1
            else:
                break

    return {
        "is_bull": is_bull,
        "regime": "多头" if is_bull else "空头/观望",
        "compass_value": latest_value,
        "compass_change_pct": round(latest_change, 2) if not pd.isna(latest_change) else None,
        "compass_change_2d_pct": round(latest_2d_change, 2) if not pd.isna(latest_2d_change) else None,
        "bull_days": bull_days,
        "trigger_reason": current.get("reason") if is_bull else None,
    }


# ==================== 大盘辅助数据 ====================

def fetch_hs300_data() -> pd.DataFrame:
    """获取沪深300指数日线数据（新浪财经接口，近90天）。"""
    try:
        df = ak.stock_zh_index_daily(symbol="sh000300")
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)
        df = df.tail(90).copy()
        df["ma20"] = df["close"].rolling(window=20, min_periods=10).mean()
        df["ma20_slope"] = df["ma20"].pct_change(5) * 100
        return df
    except Exception as e:
        print(f"[WARN] 沪深300数据获取失败: {e}")
        return pd.DataFrame()


def get_hs300_status(hs300_df: pd.DataFrame, target_date: datetime) -> dict:
    """获取 target_date 当天的沪深300状态。"""
    if hs300_df.empty:
        return {"price": None, "ma20": None, "above_ma20": None, "ma20_slope": None}

    target_date = pd.Timestamp(target_date).normalize()
    row = hs300_df[hs300_df["date"] == target_date]
    if row.empty:
        row = hs300_df.iloc[-1:]

    price = row["close"].values[0] if not row.empty else None
    ma20 = row["ma20"].values[0] if not row.empty else None
    above = price > ma20 if price and ma20 and not pd.isna(ma20) else None
    slope = row["ma20_slope"].values[0] if not row.empty and "ma20_slope" in row.columns else None

    return {
        "price": round(price, 2) if price is not None else None,
        "ma20": round(ma20, 2) if ma20 is not None else None,
        "above_ma20": above,
        "ma20_slope": round(slope, 2) if slope is not None else None,
    }


# ==================== 板块数据获取 ====================

def get_all_sectors() -> tuple:
    """获取行业板块和概念板块列表（同花顺接口）。"""
    industries = []
    concepts = []

    try:
        df_ind = ak.stock_board_industry_name_ths()
        industries = df_ind["name"].tolist()
        print(f"[INFO] 行业板块: {len(industries)} 个")
    except Exception as e:
        print(f"[WARN] 行业板块列表获取失败: {e}")

    try:
        df_con = ak.stock_board_concept_name_ths()
        concepts = df_con["name"].tolist()
        print(f"[INFO] 概念板块: {len(concepts)} 个")
    except Exception as e:
        print(f"[WARN] 概念板块列表获取失败: {e}")

    return industries, concepts


def filter_concepts(concepts: list) -> list:
    """过滤技术性概念，保留题材性概念。"""
    filtered = []
    for c in concepts:
        if any(kw in c for kw in CONCEPT_BLACKLIST_KEYWORDS):
            continue
        filtered.append(c)
    return filtered


def fetch_sector_hist_ths(sector_name: str, sector_type: str = "industry",
                          days: int = 30) -> pd.DataFrame:
    """获取板块历史行情（同花顺接口）。"""
    try:
        start = (datetime.now() - timedelta(days=days+20)).strftime("%Y%m%d")
        end = datetime.now().strftime("%Y%m%d")

        if sector_type == "industry":
            df = ak.stock_board_industry_index_ths(symbol=sector_name, start_date=start, end_date=end)
        else:
            df = ak.stock_board_concept_index_ths(symbol=sector_name, start_date=start, end_date=end)

        if df is None or df.empty:
            return pd.DataFrame()

        # 同花顺接口列名是中文
        df.columns = [str(c).strip() for c in df.columns]
        date_col = next((c for c in df.columns if "日期" in c), None)
        close_col = next((c for c in df.columns if "收盘" in c), None)
        vol_col = next((c for c in df.columns if "成交量" in c or "成交额" in c), None)

        if not date_col or not close_col:
            return pd.DataFrame()

        df = df.rename(columns={date_col: "date", close_col: "close"})
        if vol_col:
            df = df.rename(columns={vol_col: "volume"})

        df["date"] = pd.to_datetime(df["date"])
        df["close"] = pd.to_numeric(df["close"], errors="coerce")
        if "volume" in df.columns:
            df["volume"] = pd.to_numeric(df["volume"], errors="coerce")
        df = df.dropna(subset=["date", "close"]).sort_values("date").reset_index(drop=True)
        return df
    except Exception:
        return pd.DataFrame()


def calc_sector_metrics(hist_df: pd.DataFrame) -> dict:
    """计算板块动量和结构指标。"""
    if hist_df.empty or len(hist_df) < 5:
        return {}

    closes = hist_df["close"].values
    latest = closes[-1]

    # 各周期涨幅
    change_5d = (latest / closes[-6] - 1) * 100 if len(closes) >= 6 else None
    change_20d = (latest / closes[-21] - 1) * 100 if len(closes) >= 21 else None

    # 20日均线
    ma20 = np.mean(closes[-20:]) if len(closes) >= 20 else np.mean(closes)
    above_ma20 = latest > ma20

    # 量价配合
    volume_trend = None
    if "volume" in hist_df.columns and len(hist_df) >= 10:
        vols = hist_df["volume"].values
        vol_recent = np.mean(vols[-5:])
        vol_old = np.mean(vols[-10:-5])
        volume_trend = vol_recent / vol_old if vol_old > 0 else None

    return {
        "latest_close": latest,
        "change_5d": round(change_5d, 2) if change_5d is not None else None,
        "change_20d": round(change_20d, 2) if change_20d is not None else None,
        "ma20": round(ma20, 2),
        "above_ma20": above_ma20,
        "volume_trend": round(volume_trend, 2) if volume_trend is not None else None,
    }


# ==================== 资金流向与涨停数据 ====================

def fetch_fund_flow_industry() -> pd.DataFrame:
    """获取行业资金流向（当日）。"""
    try:
        df = ak.stock_fund_flow_industry()
        return df
    except Exception as e:
        print(f"[WARN] 行业资金流向获取失败: {e}")
        return pd.DataFrame()


def fetch_fund_flow_concept() -> pd.DataFrame:
    """获取概念资金流向（当日）。"""
    try:
        df = ak.stock_fund_flow_concept()
        return df
    except Exception as e:
        print(f"[WARN] 概念资金流向获取失败: {e}")
        return pd.DataFrame()


def fetch_zt_pool(date_str: str = None) -> pd.DataFrame:
    """获取涨停池。"""
    if date_str is None:
        date_str = datetime.now().strftime("%Y%m%d")
    try:
        df = ak.stock_zt_pool_em(date=date_str)
        return df
    except Exception as e:
        print(f"[WARN] 涨停池获取失败({date_str}): {e}")
        return pd.DataFrame()


def map_zt_to_sectors(zt_df: pd.DataFrame, sectors_data: dict) -> dict:
    """
    将涨停股映射到板块。
    sectors_data: {sector_name: {"type": "industry"/"concept", "cons": [code1, code2, ...]}}
    返回: {sector_name: [{"code": ..., "name": ..., "limit_times": ...}, ...]}
    """
    if zt_df.empty or not sectors_data:
        return {}

    # 涨停股代码统一为6位数字
    zt_df = zt_df.copy()
    if "代码" in zt_df.columns:
        zt_df["code"] = zt_df["代码"].astype(str).str.replace(r"\D", "", regex=True).str[-6:]
    if "连板数" in zt_df.columns:
        zt_df["limit_times"] = pd.to_numeric(zt_df["连板数"], errors="coerce").fillna(1).astype(int)
    elif "涨停天数" in zt_df.columns:
        zt_df["limit_times"] = pd.to_numeric(zt_df["涨停天数"], errors="coerce").fillna(1).astype(int)
    else:
        zt_df["limit_times"] = 1

    zt_codes = set(zt_df["code"].tolist())

    sector_zt = {}
    for sector_name, data in sectors_data.items():
        cons_codes = set([str(c).strip()[-6:] for c in data.get("cons", [])])
        matched = zt_codes & cons_codes
        if matched:
            stocks = []
            for code in matched:
                row = zt_df[zt_df["code"] == code]
                if not row.empty:
                    name = row["名称"].values[0] if "名称" in row.columns else code
                    limit = int(row["limit_times"].values[0]) if "limit_times" in row.columns else 1
                    stocks.append({"code": code, "name": name, "limit_times": limit})
            if stocks:
                sector_zt[sector_name] = stocks

    return sector_zt


def get_sector_cons_em(sector_name: str, sector_type: str = "industry") -> list:
    """获取板块成分股列表（仅代码）。"""
    try:
        if sector_type == "industry":
            df = ak.stock_board_industry_cons_em(symbol=sector_name)
        else:
            df = ak.stock_board_concept_cons_em(symbol=sector_name)

        if df is None or df.empty:
            return []

        code_col = next((c for c in df.columns if "代码" in c or "code" in c.lower()), None)
        if code_col:
            return df[code_col].astype(str).tolist()
        return []
    except Exception:
        return []


# ==================== 评分层 ====================

def score_momentum(metrics: dict, hs300_change_20d: float = None) -> int:
    """动量维度评分(30分)。"""
    score = 0
    change_20d = metrics.get("change_20d")
    change_5d = metrics.get("change_5d")

    if change_20d is None:
        return 15

    # 20日涨幅 (15分)
    if change_20d > 20:
        score += 15
    elif change_20d > 10:
        score += 12
    elif change_20d > 5:
        score += 9
    elif change_20d > 0:
        score += 5
    else:
        score += 0

    # 5日加速 (10分)
    if change_5d is not None:
        if change_5d > 10:
            score += 10
        elif change_5d > 5:
            score += 7
        elif change_5d > 2:
            score += 4
        elif change_5d > 0:
            score += 2

    # 相对大盘超额 (5分)
    if hs300_change_20d is not None and change_20d is not None:
        excess = change_20d - hs300_change_20d
        if excess > 15:
            score += 5
        elif excess > 8:
            score += 3
        elif excess > 0:
            score += 1

    return min(score, WEIGHT_MOMENTUM)


def score_fund(fund_flow: float = None) -> int:
    """资金维度评分(30分)。"""
    if fund_flow is None:
        return 15

    score = 0
    # 近5日净流入 (30分)
    if fund_flow > 50:    # 50亿
        score = 30
    elif fund_flow > 20:
        score = 24
    elif fund_flow > 10:
        score = 18
    elif fund_flow > 5:
        score = 12
    elif fund_flow > 0:
        score = 6
    else:
        score = 0

    return score


def score_sentiment(zt_count: int = 0, max_limit_times: int = 0, total_cons: int = 0) -> int:
    """情绪维度评分(25分)。"""
    score = 0

    # 涨停股数量 (15分)
    if zt_count >= 5:
        score += 15
    elif zt_count >= 3:
        score += 12
    elif zt_count >= 2:
        score += 8
    elif zt_count >= 1:
        score += 4

    # 连板高度 (10分)
    if max_limit_times >= 5:
        score += 10
    elif max_limit_times >= 3:
        score += 7
    elif max_limit_times >= 2:
        score += 4
    elif max_limit_times >= 1:
        score += 2

    return min(score, WEIGHT_SENTIMENT)


def score_structure(metrics: dict) -> int:
    """结构维度评分(15分)。"""
    score = 0

    # 20日均线上方 (8分)
    if metrics.get("above_ma20"):
        score += 8
    else:
        score += 2

    # 量价配合 (7分)
    vt = metrics.get("volume_trend")
    if vt is not None:
        if vt > 1.5:
            score += 7
        elif vt > 1.2:
            score += 5
        elif vt > 1.0:
            score += 3
        else:
            score += 1
    else:
        score += 3

    return min(score, WEIGHT_STRUCTURE)


def evaluate_sector(sector_name: str, sector_type: str, hist_df: pd.DataFrame,
                    fund_flow: float = None, zt_stocks: list = None,
                    hs300_change_20d: float = None) -> dict:
    """对单个板块进行综合评分。"""
    metrics = calc_sector_metrics(hist_df)
    if not metrics:
        return None

    zt_count = len(zt_stocks) if zt_stocks else 0
    max_limit = max([s["limit_times"] for s in zt_stocks]) if zt_stocks else 0

    s_momentum = score_momentum(metrics, hs300_change_20d)
    s_fund = score_fund(fund_flow)
    s_sentiment = score_sentiment(zt_count, max_limit)
    s_structure = score_structure(metrics)

    total = s_momentum + s_fund + s_sentiment + s_structure

    return {
        "name": sector_name,
        "type": sector_type,
        "change_20d": metrics.get("change_20d"),
        "change_5d": metrics.get("change_5d"),
        "fund_flow_5d": fund_flow,
        "zt_count": zt_count,
        "max_limit_times": max_limit,
        "zt_stocks": zt_stocks or [],
        "score_momentum": s_momentum,
        "score_fund": s_fund,
        "score_sentiment": s_sentiment,
        "score_structure": s_structure,
        "total_score": total,
        "is_leader": total >= LEADER_CANDIDATE_SCORE,
        "above_ma20": metrics.get("above_ma20"),
    }


# ==================== 龙头挖掘 ====================

def fetch_stock_hist(stock_code: str, days: int = 30) -> pd.DataFrame:
    """获取个股历史行情。"""
    try:
        prefix = "sh" if str(stock_code).startswith(("6", "5", "9")) else "sz"
        df = ak.stock_zh_a_daily(symbol=f"{prefix}{stock_code}",
                                  start_date=(datetime.now() - timedelta(days=days+10)).strftime("%Y%m%d"),
                                  end_date=datetime.now().strftime("%Y%m%d"))
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)
        return df
    except Exception:
        return pd.DataFrame()


def find_first_zt_stock(sector_name: str, sector_type: str, zt_stocks: list,
                        target_date: datetime) -> dict:
    """
    找出板块内首个放量涨停的票。
    简化版：从涨停池匹配的成分股中，取连板天数最多且市值较大的。
    """
    if not zt_stocks:
        return None

    # 按连板天数降序，取Top3
    sorted_zt = sorted(zt_stocks, key=lambda x: x.get("limit_times", 1), reverse=True)
    best = sorted_zt[0]

    return {
        "code": best["code"],
        "name": best["name"],
        "limit_times": best.get("limit_times", 1),
        "leader_type": "first_zt",
    }


def find_leader_candidates(sector_name: str, sector_type: str,
                           cons_codes: list, top_n: int = 5) -> list:
    """
    找出板块内龙头候选 Top N。
    按近20日涨幅排序，优先取有涨停、成交额放大的。
    """
    candidates = []
    for code in cons_codes[:50]:  # 限制数量避免太慢
        try:
            df = fetch_stock_hist(code, days=25)
            if df.empty or len(df) < 5:
                continue

            latest = df.iloc[-1]
            price = latest["close"]
            change_20d = (price / df.iloc[0]["close"] - 1) * 100 if len(df) >= 20 else None
            vol_recent = df["volume"].iloc[-5:].mean()
            vol_old = df["volume"].iloc[-10:-5].mean() if len(df) >= 10 else vol_recent
            vol_ratio = vol_recent / vol_old if vol_old > 0 else 1.0

            candidates.append({
                "code": code,
                "name": code,  # 稍后查名称
                "change_20d": round(change_20d, 2) if change_20d else 0,
                "price": round(price, 2),
                "vol_ratio": round(vol_ratio, 2),
            })
        except Exception:
            continue

    if not candidates:
        return []

    # 按20日涨幅降序
    candidates.sort(key=lambda x: x["change_20d"], reverse=True)

    # 取TopN，加标签
    result = []
    for i, c in enumerate(candidates[:top_n]):
        tags = []
        if c["vol_ratio"] > 1.5:
            tags.append("放量")
        if c["change_20d"] > 20:
            tags.append("强势")
        c["tags"] = ", ".join(tags) if tags else "-"
        c["leader_type"] = "candidate"
        result.append(c)

    return result


# ==================== 主线见顶预警 ====================

def check_peak_warning(sector: dict, sector_history: pd.DataFrame) -> list:
    """检测主线见顶预警信号。"""
    warnings = []

    # 1. 连板股数量连续下降（需要历史数据，简化版：当前无连板股但之前评分高）
    if sector.get("max_limit_times", 0) == 0 and sector.get("total_score", 0) >= 80:
        warnings.append("连板股消失，情绪退潮")

    # 2. 板块20日涨幅>40%且5日涨幅<0
    c20 = sector.get("change_20d")
    c5 = sector.get("change_5d")
    if c20 is not None and c5 is not None and c20 > 40 and c5 < 0:
        warnings.append("20日涨幅>40%但近期回调，可能见顶")

    # 3. 资金净流出（简化版：fund_flow_5d < 0）
    ff = sector.get("fund_flow_5d")
    if ff is not None and ff < 0:
        warnings.append("近5日资金净流出")

    # 4. 跌破20日均线
    if not sector.get("above_ma20", True):
        warnings.append("板块指数跌破20日均线")

    return warnings


# ==================== 报告生成层 ====================

def generate_report(date_str: str, regime: dict, hs300: dict,
                    sectors: list, db_path: Path, output_path: Path):
    """生成 Markdown 报告。"""

    md = f"""# A股板块主线追踪报告

> 报告日期：**{date_str}**  
> 数据来源：AKShare（东方财富/同花顺）+ 用户提供的指南针活跃市值  
> 评估框架：基于板块主线追踪器 v1.0

---

## 一、大盘环境

| 指标 | 数值 | 说明 |
|------|------|------|
| 指南针活跃市值 | {regime.get('compass_value', 'N/A')} | 当日涨跌幅: {regime.get('compass_change_pct', 'N/A')}% |
| 多头状态 | **{regime['regime']}** | 持续天数: {regime.get('bull_days', 0)} 天 |
| 开启原因 | {regime.get('trigger_reason', 'N/A')} | — |
| 沪深300 | {hs300.get('price', 'N/A')} | 20日均线: {hs300.get('ma20', 'N/A')} ({'上方' if hs300.get('above_ma20') else '下方'}) |

> **多头判定规则**：指南针活跃市值单日涨幅≥4% 或 连续2日累计≥4% → 开启多头；开启后单日跌幅≤-2.3% → 结束多头。

---

"""

    if not regime.get("is_bull"):
        md += """## 二、板块扫描结果

> **当前不处于多头区间，暂停板块主线扫描。**

> 建议：维持观望，等待指南针活跃市值出现≥4%的上涨信号后再启动主线追踪。

---

"""
        md += f"""> **报告生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}**
"""
        output_path.write_text(md, encoding="utf-8")
        return str(output_path)

    # 排序并筛选
    sorted_sectors = sorted(sectors, key=lambda x: x["total_score"], reverse=True)
    leader_candidates = [s for s in sorted_sectors if s.get("is_leader")]
    top_sectors = sorted_sectors[:20]

    md += "## 二、主线板块排名（Top 20）\n\n"
    md += "| 排名 | 板块 | 类型 | 20日涨幅 | 5日涨幅 | 涨停股 | 总得分 | 状态 |\n"
    md += "|:----:|------|:----:|:--------:|:-------:|:------:|:------:|:----:|\n"

    for i, s in enumerate(top_sectors, 1):
        type_label = "行业" if s["type"] == "industry" else "概念"
        c20 = f"{s['change_20d']}%" if s['change_20d'] is not None else "-"
        c5 = f"{s['change_5d']}%" if s['change_5d'] is not None else "-"
        zt = s.get("zt_count", 0)
        status = "🟢 主线候选" if s.get("is_leader") else "⚪ 跟踪"
        md += f"| {i} | {s['name']} | {type_label} | {c20} | {c5} | {zt} | **{s['total_score']}** | {status} |\n"

    md += "\n---\n\n"

    # 主线深度分析
    md += "## 三、主线深度分析\n\n"

    if not leader_candidates:
        md += "> 当前无板块达到主线候选标准（≥75分）。市场可能处于多头初期，板块轮动较快。\n\n"
    else:
        for i, s in enumerate(leader_candidates[:5], 1):
            md += f"### {'🥇' if i == 1 else '🥈' if i == 2 else '🥉' if i == 3 else '📌'} 主线{i}：{s['name']}\n\n"

            # 首个放量涨停
            first_zt = s.get("first_zt")
            if first_zt:
                md += f"- **首个放量涨停**：{first_zt['name']}({first_zt['code']})，连板 {first_zt.get('limit_times', 1)} 天\n"
            else:
                md += "- **首个放量涨停**：暂无（板块内无涨停股）\n"

            # 龙头候选
            candidates = s.get("leader_candidates", [])
            if candidates:
                md += "- **龙头候选 Top5**：\n"
                for c in candidates:
                    md += f"  - {c['name']}({c['code']}) | 20日涨幅 {c['change_20d']}% | 量比 {c['vol_ratio']} | {c['tags']}\n"
            else:
                md += "- **龙头候选**：数据获取中...\n"

            # 分项得分
            md += f"- **四维得分**：动量({s['score_momentum']}) + 资金({s['score_fund']}) + 情绪({s['score_sentiment']}) + 结构({s['score_structure']}) = **{s['total_score']}**\n"

            # 见顶预警
            warnings = s.get("peak_warnings", [])
            if warnings:
                md += f"- **⚠️ 见顶预警**：{'；'.join(warnings)}\n"
            else:
                md += "- **见顶预警**：暂无\n"

            md += "\n"

    md += "---\n\n"

    # 历史持续性
    md += "## 四、主线历史持续性\n\n"
    md += "> （需要积累历史数据后显示）\n\n"
    md += "| 板块 | 首次出现 | 持续天数 | 当前状态 |\n"
    md += "|------|---------|---------|---------|\n"
    md += "| *待积累* | — | — | — |\n"

    md += f"""
---

> **报告生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}**
"""

    output_path.write_text(md, encoding="utf-8")
    return str(output_path)


def save_market_regime_json(date_str: str, regime: dict, leading_sectors: list,
                            output_dir: Path):
    """生成 market_regime.json 供红利 skill 读取。"""
    data = {
        "date": date_str,
        "regime": "bull" if regime.get("is_bull") else "bear",
        "bull_days": regime.get("bull_days", 0),
        "compass_change_pct": regime.get("compass_change_pct"),
        "leading_sectors": [s["name"] for s in leading_sectors],
        "defensive_weight_recommend": 0.3 if regime.get("is_bull") else 0.7,
    }
    path = output_dir / "market_regime.json"
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(path)


# ==================== 主入口 ====================

def main():
    parser = argparse.ArgumentParser(description="A股板块主线追踪器")
    parser.add_argument("--compass-csv", required=True, help="指南针活跃市值 CSV 文件路径")
    parser.add_argument("--output-dir", default=".", help="报告输出目录")
    parser.add_argument("--db", default="./data/sector_leader.db", help="SQLite 数据库路径")
    parser.add_argument("--date", help="指定运行日期 (YYYY-MM-DD)，默认今天")
    parser.add_argument("--json", action="store_true", help="同时输出 JSON 原始数据")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    db_path = Path(args.db)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    target_date = datetime.strptime(args.date, "%Y-%m-%d") if args.date else datetime.now()
    date_str = target_date.strftime("%Y-%m-%d")

    print(f"[{datetime.now().strftime('%H:%M:%S')}] 板块主线追踪器启动 — 日期: {date_str}")

    # 初始化数据库
    init_db(db_path)

    # 1. 读取指南针活跃市值
    print("  [1/6] 读取指南针活跃市值...")
    try:
        compass_df = load_compass_data(Path(args.compass_csv))
        print(f"        读取 {len(compass_df)} 条记录")
    except Exception as e:
        print(f"[ERROR] {e}")
        sys.exit(1)

    # 2. 判断多头区间
    print("  [2/6] 判断多头区间...")
    regime = detect_bull_regime(compass_df, target_date)
    print(f"        状态: {regime['regime']} | 活跃市值: {regime['compass_value']} | 日涨幅: {regime['compass_change_pct']}%")

    # 3. 获取大盘辅助数据
    print("  [3/6] 获取沪深300数据...")
    hs300_df = fetch_hs300_data()
    hs300_status = get_hs300_status(hs300_df, target_date)
    print(f"        沪深300: {hs300_status['price']} | 20日线: {hs300_status['ma20']} | {'上方' if hs300_status['above_ma20'] else '下方'}")

    # 保存大盘状态
    save_market_regime(db_path, {
        "date": date_str,
        "regime": regime["regime"],
        "compass_value": regime["compass_value"],
        "compass_change_pct": regime["compass_change_pct"],
        "bull_trigger_reason": regime.get("trigger_reason"),
        "hs300_price": hs300_status["price"],
        "hs300_ma20": hs300_status["ma20"],
        "hs300_above_ma20": 1 if hs300_status["above_ma20"] else 0,
    })

    sectors_results = []

    if not regime["is_bull"]:
        print("  [INFO] 非多头区间，跳过板块扫描")
    else:
        # 4. 获取板块列表
        print("  [4/6] 获取板块列表...")
        industries, concepts = get_all_sectors()
        filtered_concepts = filter_concepts(concepts)
        print(f"        行业: {len(industries)} | 概念(过滤后): {len(filtered_concepts)}")

        # 5. 获取资金流向和涨停池
        print("  [5/6] 获取资金流向与涨停池...")
        fund_ind = fetch_fund_flow_industry()
        fund_con = fetch_fund_flow_concept()
        zt_df = fetch_zt_pool(target_date.strftime("%Y%m%d"))
        print(f"        涨停池: {len(zt_df)} 只")

        # 计算沪深300的20日涨幅作为基准
        hs300_change_20d = None
        if not hs300_df.empty and len(hs300_df) >= 20:
            hs300_change_20d = (hs300_df["close"].iloc[-1] / hs300_df["close"].iloc[-20] - 1) * 100

        # 6. 扫描板块
        print("  [6/6] 扫描板块并评分...")
        all_sectors = [(name, "industry") for name in industries] + [(name, "concept") for name in filtered_concepts]

        # 先快速扫描所有板块的历史行情，计算20日涨幅，对概念板块只保留Top50
        sector_changes = []
        for name, stype in all_sectors:
            try:
                hist = fetch_sector_hist_ths(name, stype, days=25)
                metrics = calc_sector_metrics(hist)
                if metrics and metrics.get("change_20d") is not None:
                    sector_changes.append((name, stype, metrics["change_20d"], hist))
            except Exception:
                continue

        # 概念板块只保留涨幅Top50
        concept_changes = [s for s in sector_changes if s[1] == "concept"]
        concept_changes.sort(key=lambda x: x[2], reverse=True)
        top_concepts = {s[0] for s in concept_changes[:50]}

        # 最终扫描列表：全部行业 + Top50概念
        final_sectors = [(n, t, h) for n, t, _, h in sector_changes if t == "industry" or (t == "concept" and n in top_concepts)]
        print(f"        最终扫描: {len(final_sectors)} 个板块")

        # 预获取板块成分股（用于涨停映射）
        sector_cons = {}
        for name, stype, hist in final_sectors[:30]:  # 只给Top30板块获取成分股
            cons = get_sector_cons_em(name, stype)
            if cons:
                sector_cons[name] = {"type": stype, "cons": cons}

        # 涨停股映射到板块
        sector_zt = map_zt_to_sectors(zt_df, sector_cons)

        # 评分
        for name, stype, hist in final_sectors:
            # 资金流向
            fund_flow = None
            if stype == "industry" and not fund_ind.empty:
                row = fund_ind[fund_ind.iloc[:, 0] == name]
                if not row.empty:
                    try:
                        fund_flow = float(row.iloc[0, 5])  # 净流入列位置可能变化
                    except (ValueError, IndexError):
                        pass
            elif stype == "concept" and not fund_con.empty:
                row = fund_con[fund_con.iloc[:, 0] == name]
                if not row.empty:
                    try:
                        fund_flow = float(row.iloc[0, 5])
                    except (ValueError, IndexError):
                        pass

            zt_stocks = sector_zt.get(name, [])
            result = evaluate_sector(name, stype, hist, fund_flow, zt_stocks, hs300_change_20d)
            if result:
                # 首个放量涨停
                if zt_stocks:
                    result["first_zt"] = find_first_zt_stock(name, stype, zt_stocks, target_date)

                # 龙头候选
                if name in sector_cons:
                    result["leader_candidates"] = find_leader_candidates(
                        name, stype, sector_cons[name]["cons"], top_n=5
                    )

                # 见顶预警
                hist_db = get_sector_history(db_path, name, days=10)
                result["peak_warnings"] = check_peak_warning(result, hist_db)

                sectors_results.append(result)

        # 保存到数据库
        save_sector_scores(db_path, date_str, sectors_results)
        print(f"        评分完成: {len(sectors_results)} 个板块")

    # 生成报告
    print(f"[{datetime.now().strftime('%H:%M:%S')}] 生成报告中...")
    md_path = out_dir / f"sector_leader_report_{date_str.replace('-', '')}.md"
    generate_report(date_str, regime, hs300_status, sectors_results, db_path, md_path)
    print(f"  → Markdown 报告: {md_path}")

    # 生成 market_regime.json
    leader_sectors = [s for s in sectors_results if s.get("is_leader")]
    regime_json_path = save_market_regime_json(date_str, regime, leader_sectors, out_dir)
    print(f"  → 市场状态 JSON: {regime_json_path}")

    if args.json:
        json_path = out_dir / f"sector_leader_data_{date_str.replace('-', '')}.json"
        json_path.write_text(json.dumps({
            "date": date_str,
            "regime": regime,
            "hs300": hs300_status,
            "sectors": sectors_results,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  → 原始 JSON: {json_path}")

    print(f"[{datetime.now().strftime('%H:%M:%S')}] 完成。多头状态: {regime['regime']} | 主线候选: {len([s for s in sectors_results if s.get('is_leader')])} 个")


if __name__ == "__main__":
    main()
