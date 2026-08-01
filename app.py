import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from lightgbm import LGBMClassifier
from ta.trend import SMAIndicator, MACD
from ta.momentum import RSIIndicator

# --------------------------------------------------
# ページ基本設定
# --------------------------------------------------
st.set_page_config(
    page_title="株価予測ダッシュボード",
    page_icon="📈",
    layout="wide"
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
    "日経平均": ("日経平均株価", "^N225"), "日経225": ("日経平均株価", "^N225"), "N225": ("日経平均株価", "^N225"),
    "S&P500": ("S&P 500", "^GSPC"), "SP500": ("S&P 500", "^GSPC"),
    "ナスダック": ("NASDAQ Composite", "^IXIC"), "NASDAQ": ("NASDAQ Composite", "^IXIC"),
    
    # 宇宙関連銘柄
    "アクセルスペース": ("アクセルスペースホールディングス", "402A.T"),
    "アクセルスペースホールディングス": ("アクセルスペースホールディングス", "402A.T"),
    "402A": ("アクセルスペースホールディングス", "402A.T"),
    "アストロスケール": ("アストロスケールホールディングス", "186A.T"),
    "アストロスケールホールディングス": ("アストロスケールホールディングス", "186A.T"),
    "186A": ("アストロスケールホールディングス", "186A.T"),
    "ispace": ("ispace", "9348.T"), "アイスペース": ("ispace", "9348.T"),
    
    # 新規・注目銘柄
    "キオクシア": ("キオクシアホールディングス", "285A.T"), "285A": ("キオクシアホールディングス", "285A.T"),
    
    # 日本株（主要銘柄）
    "トヨタ": ("トヨタ自動車", "7203.T"), "トヨタ自動車": ("トヨタ自動車", "7203.T"),
    "ソニー": ("ソニーグループ", "6758.T"), "ソニーグループ": ("ソニーグループ", "6758.T"),
    "ソフトバンク": ("ソフトバンクグループ", "9984.T"), "ソフトバンクグループ": ("ソフトバンクグループ", "9984.T"),
    "キーエンス": ("キーエンス", "6861.T"),
    "ファーストリテイリング": ("ファーストリテイリング", "9983.T"), "ユニクロ": ("ファーストリテイリング", "9983.T"),
    "任天堂": ("任天堂", "7974.T"),
    "三菱UFJ": ("三菱UFJフィナンシャル・グループ", "8306.T"), "三菱UFJフィナンシャル・グループ": ("三菱UFJフィナンシャル・グループ", "8306.T"),
    "レーザーテック": ("レーザーテック", "6920.T"),
    "東京エレクトロン": ("東京エレクトロン", "8035.T"),
    "NTT": ("日本電信電話", "9432.T"), "日本電信電話": ("日本電信電話", "9432.T"),
    "楽天": ("楽天グループ", "4755.T"), "楽天グループ": ("楽天グループ", "4755.T"),
    
    # 米国株（主要銘柄）
    "アップル": ("Apple", "AAPL"), "Apple": ("Apple", "AAPL"),
    "エヌビディア": ("NVIDIA", "NVDA"), "Nvidia": ("NVIDIA", "NVDA"), "NVIDIA": ("NVIDIA", "NVDA"),
    "マイクロソフト": ("Microsoft", "MSFT"), "Microsoft": ("Microsoft", "MSFT"),
    "アマゾン": ("Amazon.com", "AMZN"), "Amazon": ("Amazon.com", "AMZN"),
    "アルファベット": ("Alphabet (Google)", "GOOGL"), "グーグル": ("Alphabet (Google)", "GOOGL"), "Google": ("Alphabet (Google)", "GOOGL"),
    "メタ": ("Meta Platforms", "META"), "Meta": ("Meta Platforms", "META"), "フェイスブック": ("Meta Platforms", "META"),
    "テスラ": ("Tesla", "TSLA"), "Tesla": ("Tesla", "TSLA")
}

# --------------------------------------------------
# サイドバー（設定パラメータ）
# --------------------------------------------------
st.sidebar.header("設定")

user_input = st.sidebar.text_input(
    "企業名・指数 または 銘柄コードを入力", 
    value="アクセルスペース"
).strip()

# 未上場企業チェック
if user_input in UNLISTED_COMPANIES:
    st.warning(f"⚠️ {UNLISTED_COMPANIES[user_input]}")
    st.stop()

# 辞書から検索（該当がなければ入力値をそのまま表示・補正）
if user_input in COMPANY_MAP:
    display_name, ticker = COMPANY_MAP[user_input]
else:
    ticker = user_input
    # 日本株の自動補正（例: "7203" ➔ "7203.T", "402A" ➔ "402A.T"）
    if not ticker.startswith("^") and not ticker.endswith(".T") and len(ticker) == 4 and not ticker.isalpha():
        ticker = f"{ticker}.T"
    display_name = user_input

# 取得期間の選択
period_options = {"1年": "1y", "2年": "2y", "5年": "5y"}
selected_period_label = st.sidebar.selectbox("データ取得期間", list(period_options.keys()))
period = period_options[selected_period_label]

# --------------------------------------------------
# データ取得 & 特徴量作成処理
# --------------------------------------------------
@st.cache_data(ttl=3600)  # 1時間キャッシュ
def load_and_prep_data(symbol, time_period):
    df = yf.download(symbol, period=time_period)
    if df.empty:
        return None
    
    # マルチインデックス対策
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # 1. テクニカル指標の追加
    df['SMA_20'] = SMAIndicator(close=df['Close'], window=20).sma_indicator()
    df['RSI'] = RSIIndicator(close=df['Close'], window=14).rsi()
    macd = MACD(close=df['Close'])
    df['MACD'] = macd.macd()
    
    # 2. リターンの作成
    df['Return'] = df['Close'].pct_change()
    
    # 3. 目的変数（翌日上がれば1、下がれば0）
    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    
    # 欠損値処理
    df.dropna(inplace=True)
    return df

# データの読み込み
df = load_and_prep_data(ticker, period)

if df is None or len(df) < 20:
    st.error(f"「{user_input}」（コード: {ticker}）のデータ取得に失敗しました。正しい名称またはコード（例: 402A.T, アクセルスペース, 7203.T）を入力してください。")
    st.stop()

# --------------------------------------------------
# メイン画面表示
# --------------------------------------------------

# 企業名・指数名とティッカーコードを表示
st.markdown(f"## 🏢 {display_name}  `({ticker})`")

st.markdown("---")

# 1. 簡易AI予測セクション
st.subheader("🤖 AIによる翌日株価予測 (LightGBM)")

features = ['Close', 'Volume', 'SMA_20', 'RSI', 'MACD', 'Return']
X = df[features]
y = df['Target']

# 直近データ以外で学習
X_train, y_train = X.iloc[:-1], y.iloc[:-1]
latest_data = X.iloc[[-1]]

# モデル学習
model = LGBMClassifier(random_state=42, verbose=-1)
model.fit(X_train, y_train)

# 翌日の予測確率
pred_proba = model.predict_proba(latest_data)[0]
up_probability = pred_proba[1] * 100

# 結果の表示
col1, col2, col3 = st.columns(3)

latest_close = float(df['Close'].iloc[-1])
prev_close = float(df['Close'].iloc[-2])
price_diff = latest_close - prev_close

col1.metric("最新終値", f"¥{latest_close:,.1f}", f"{price_diff:+,.1f}")

if up_probability >= 50:
    col2.metric("翌日の株価予測", "上昇予想 🟢", f"上昇確率 {up_probability:.1f}%")
else:
    col2.metric("翌日の株価予測", "下落予想 🔴", f"下落確率 {100 - up_probability:.1f}%")

col3.metric("直近RSI (14日)", f"{float(df['RSI'].iloc[-1]):.1f}")

st.caption("※予測モデルは過去データに基づいたデモです。実際の投資判断には使用しないでください。")

st.markdown("---")

# 2. チャート表示セクション
st.subheader(f"📊 チャート & テクニカル分析: {display_name}")

fig = go.Figure()

# ローソク足
fig.add_trace(go.Candlestick(
    x=df.index,
    open=df['Open'],
    high=df['High'],
    low=df['Low'],
    close=df['Close'],
    name="ローソク足"
))

# 移動平均線
fig.add_trace(go.Scatter(
    x=df.index,
    y=df['SMA_20'],
    line=dict(color='orange', width=1.5),
    name="SMA 20日"
))

fig.update_layout(
    xaxis_rangeslider_visible=False,
    height=450,
    margin=dict(l=20, r=20, t=20, b=20)
)

st.plotly_chart(fig, use_container_width=True)

# 3. データテーブル表示
with st.expander("📄 過去データテーブルを表示"):
    st.dataframe(df.sort_index(ascending=False))
