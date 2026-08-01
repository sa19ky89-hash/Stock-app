import datetime
import lightgbm as lgb
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from ta.momentum import RSIIndicator
from ta.trend import MACD, SMAIndicator
import yfinance as yf

# --------------------------------------------------
# ページ基本設定
# --------------------------------------------------
st.set_page_config(
    page_title="株価予測ダッシュボード", page_icon="📈", layout="wide"
)

st.title("📈 株価予測 & テクニカル分析ダッシュボード")

# --------------------------------------------------
# 未上場企業のリスト（案内用）
# --------------------------------------------------
UNLISTED_COMPANIES = {}

# --------------------------------------------------
# 企業名・指数 ➔ (正式表示名, ティッカーコード) 変換辞書
# --------------------------------------------------
COMPANY_MAP = {
    # 主要指数・市場データ
    "日経平均": ("日経平均株価", "^N225"),
    "日経225": ("日経平均株価", "^N225"),
    "N225": ("日経平均株価", "^N225"),
    "S&P500": ("S&P 500", "^GSPC"),
    "SP500": ("S&P 500", "^GSPC"),
    "ナスダック": ("NASDAQ Composite", "^IXIC"),
    "NASDAQ": ("NASDAQ Composite", "^IXIC"),
    # 宇宙関連銘柄
    "アクセルスペース": ("アクセルスペースホールディングス", "402A.T"),
    "アクセルスペースホールディングス": (
        "アクセルスペースホールディングス",
        "402A.T",
    ),
    "402A": ("アクセルスペースホールディングス", "402A.T"),
    "アストロスケール": (
        "アストロスケールホールディングス",
        "186A.T",
    ),
    "アストロスケールホールディングス": (
        "アストロスケールホールディングス",
        "186A.T",
    ),
    "186A": ("アストロスケールホールディングス", "186A.T"),
    "ispace": ("ispace", "9348.T"),
    "アイスペース": ("ispace", "9348.T"),
    # 新規・注目銘柄
    "キオクシア": ("キオクシアホールディングス", "285A.T"),
    "285A": ("キオクシアホールディングス", "285A.T"),
    # 日本株（主要銘柄）
    "トヨタ": ("トヨタ自動車", "7203.T"),
    "トヨタ自動車": ("トヨタ自動車", "7203.T"),
    "ソニー": ("ソニーグループ", "6758.T"),
    "ソニーグループ": ("ソニーグループ", "6758.T"),
    "ソフトバンク": ("ソフトバンクグループ", "9984.T"),
    "ソフトバンクグループ": ("ソフトバンクグループ", "9984.T"),
    "キーエンス": ("キーエンス", "6861.T"),
    "ファーストリテイリング": ("ファーストリテイリング", "9983.T"),
    "ユニクロ": ("ファーストリテイリング", "9983.T"),
    "任天堂": ("任天堂", "7974.T"),
    "三菱UFJ": ("三菱UFJフィナンシャル・グループ", "8306.T"),
    "三菱UFJフィナンシャル・グループ": (
        "三菱UFJフィナンシャル・グループ",
        "8306.T",
    ),
    "レーザーテック": ("レーザーテック", "6920.T"),
    "東京エレクトロン": ("東京エレクトロン", "8035.T"),
    "NTT": ("日本電信電話", "9432.T"),
    "日本電信電話": ("日本電信電話", "9432.T"),
    "楽天": ("楽天グループ", "4755.T"),
    "楽天グループ": ("楽天グループ", "4755.T"),
    # 米国株（主要銘柄）
    "アップル": ("Apple", "AAPL"),
    "Apple": ("Apple", "AAPL"),
    "エヌビディア": ("NVIDIA", "NVDA"),
    "Nvidia": ("NVIDIA", "NVDA"),
    "NVIDIA": ("NVIDIA", "NVDA"),
    "マイクロソフト": ("Microsoft", "MSFT"),
    "Microsoft": ("Microsoft", "MSFT"),
    "アマゾン": ("Amazon.com", "AMZN"),
    "Amazon": ("Amazon.com", "AMZN"),
    "アルファベット": ("Alphabet (Google)", "GOOGL"),
    "グーグル": ("Alphabet (Google)", "GOOGL"),
    "Google": ("Alphabet (Google)", "GOOGL"),
    "メタ": ("Meta Platforms", "META"),
    "Meta": ("Meta Platforms", "META"),
    "フェイスブック": ("Meta Platforms", "META"),
    "テスラ": ("Tesla", "TSLA"),
    "Tesla": ("Tesla", "TSLA"),
}

# --------------------------------------------------
# サイドバー（設定パラメータ）
# --------------------------------------------------
st.sidebar.header("設定")

user_input = st.sidebar.text_input(
    "企業名・指数 または 銘柄コードを入力", value="アクセルスペース"
).strip()

# 未上場企業チェック
if user_input in UNLISTED_COMPANIES:
    st.warning(f"⚠️ {UNLISTED_COMPANIES[user_input]}")
    st.stop()

# 辞書から検索
if user_input in COMPANY_MAP:
    display_name, ticker = COMPANY_MAP[user_input]
else:
    ticker = user_input
    if (
        not ticker.startswith("^")
        and not ticker.endswith(".T")
        and len(ticker) == 4
        and not ticker.isalpha()
    ):
        ticker = f"{ticker}.T"
    display_name = user_input

# 取得期間の選択
period_options = {"1年": "1y", "2年": "2y", "5年": "5y"}
selected_period_label = st.sidebar.selectbox(
    "学習用データ取得期間", list(period_options.keys())
)
period = period_options[selected_period_label]


# --------------------------------------------------
# データ取得 & 特徴量作成処理
# --------------------------------------------------
@st.cache_data(ttl=3600)
def load_and_prep_data(symbol, time_period):
    df = yf.download(symbol, period=time_period)
    if df.empty:
        return None

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # テクニカル指標の作成
    df['SMA_20'] = SMAIndicator(close=df['Close'], window=20).sma_indicator()
    df['RSI'] = RSIIndicator(close=df['Close'], window=14).rsi()
    macd = MACD(close=df['Close'])
    df['MACD'] = macd.macd()
    df['Return'] = df['Close'].pct_change()

    # 目的変数：翌日のリターン（株価変動率）
    df['Target_Return'] = df['Return'].shift(-1)

    df.dropna(inplace=True)
    return df


df = load_and_prep_data(ticker, period)

if df is None or len(df) < 20:
    st.error(
        f"「{user_input}」（コード: {ticker}）のデータ取得に失敗しました。"
    )
    st.stop()

# --------------------------------------------------
# メイン画面表示
# --------------------------------------------------

st.markdown(f"## 🏢 {display_name}  `({ticker})`")
st.markdown("---")

# 1. AIモデル構築 ＆ 今後1ヶ月（20営業日）の再帰予測
st.subheader("🤖 AIによる今後1ヶ月（20営業日）の日別株価予測")

features = ['Close', 'Volume', 'SMA_20', 'RSI', 'MACD', 'Return']
X = df[features]
y = df['Target_Return']

# 回帰モデル（LGBMRegressor）で予測
model = lgb.LGBMRegressor(random_state=42, verbose=-1, n_estimators=100)
model.fit(X, y)

# 未来20営業日の日付作成
last_date = df.index[-1]
future_dates = pd.date_range(
    start=last_date + pd.Timedelta(days=1), periods=35, freq="B"
)
future_dates = [d for d in future_dates if d > last_date][:20]

# 再帰的シミュレーション予測
sim_df = df.copy()
future_records = []

for f_date in future_dates:
    latest_feat = sim_df[features].iloc[[-1]]
    pred_return = model.predict(latest_feat)[0]

    last_close = float(sim_df['Close'].iloc[-1])
    next_close = last_close * (1 + pred_return)

    # 予測データをダミー行として追加し、特徴量を再計算
    new_row = pd.DataFrame(
        {
            "Open": next_close,
            "High": next_close,
            "Low": next_close,
            "Close": next_close,
            "Volume": sim_df["Volume"].mean(),
            "Return": pred_return,
        },
        index=[f_date],
    )

    sim_df = pd.concat([sim_df, new_row])

    # テクニカル指標の更新
    sim_df["SMA_20"] = SMAIndicator(
        close=sim_df["Close"], window=20
    ).sma_indicator()
    sim_df["RSI"] = RSIIndicator(
        close=sim_df["Close"], window=14
    ).rsi()
    macd_obj = MACD(close=sim_df["Close"])
    sim_df["MACD"] = macd_obj.macd()

    diff = next_close - last_close
    future_records.append({
        "日付": f_date.strftime("%Y-%m-%d (%a)"),
        "予想株価": round(next_close, 1),
        "前日比": round(diff, 1),
        "予想騰落率(%)": round(pred_return * 100, 2),
    })

pred_df = pd.DataFrame(future_records)

# 概要指標の表示
start_p = float(df['Close'].iloc[-1])
end_p = pred_df["予想株価"].iloc[-1]
total_change = ((end_p - start_p) / start_p) * 100

col1, col2, col3 = st.columns(3)
col1.metric("現在（最新）の株価", f"¥{start_p:,.1f}")
col2.metric(
    "1ヶ月後の予想株価",
    f"¥{end_p:,.1f}",
    f"{end_p - start_p:+,.1f} ({total_change:+.1f}%)",
)
col3.metric("予測対象日数", "20 営業日 (約1ヶ月)")

# 2. 実績＋今後1ヶ月予測チャート
st.subheader("📈 1ヶ月予測株価チャート")

fig_pred = go.Figure()

# 直近60日分の実績価格
recent_df = df.tail(60)
fig_pred.add_trace(
    go.Scatter(
        x=recent_df.index,
        y=recent_df['Close'],
        mode="lines",
        name="過去の実績株価",
        line=dict(color="#1f77b4", width=2),
    )
)

# 予測データライン（実績の最後から繋げる）
pred_x = [df.index[-1]] + [
    datetime.datetime.strptime(d.split()[0], "%Y-%m-%d")
    for d in pred_df["日付"]
]
pred_y = [float(df['Close'].iloc[-1])] + pred_df["予想株価"].tolist()

fig_pred.add_trace(
    go.Scatter(
        x=pred_x,
        y=pred_y,
        mode="lines+markers",
        name="AI今後1ヶ月予測",
        line=dict(color="#ff7f0e", width=2.5, dash="dash"),
        marker=dict(size=5),
    )
)

fig_pred.update_layout(
    height=400,
    margin=dict(l=20, r=20, t=20, b=20),
    xaxis_title="日付",
    yaxis_title="株価 (円)",
    hovermode="x unified",
)

st.plotly_chart(fig_pred, use_container_width=True)

# 3. 日別予測価格テーブル
st.subheader("📅 今後1ヶ月の日別予想株価一覧")
st.dataframe(
    pred_df.style.format({
        "予想株価": "¥{:,.1f}",
        "前日比": "{:+,.1f}",
        "予想騰落率(%)": "{:+.2f}%",
    }),
    use_container_width=True,
    height=350,
)

st.caption(
    "※表示される株価推移はAI機械学習モデルによるシミュレーション結果です。実際の投資判断には使用しないでください。"
)

st.markdown("---")

# 4. 過去の実績ローソク足チャート
with st.expander("📊 過去のテクニカル分析チャート（ローソク足）を表示"):
    fig = go.Figure()
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name="ローソク足",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df['SMA_20'],
            line=dict(color="orange", width=1.5),
            name="SMA 20日",
        )
    )
    fig.update_layout(
        xaxis_rangeslider_visible=False,
        height=400,
        margin=dict(l=20, r=20, t=20, b=20),
    )
    st.plotly_chart(fig, use_container_width=True)
