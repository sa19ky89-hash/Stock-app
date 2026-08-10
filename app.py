import datetime
import lightgbm as lgb
import matplotlib
matplotlib.use('Agg')  # バックエンドの競合防止
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import shap
import streamlit as st
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from ta.momentum import RSIIndicator
from ta.trend import MACD, SMAIndicator
import yfinance as yf

# 日本語描画ライブラリの読み込み試行（文字化け対策）
HAS_JAPANIZE = False
try:
    import japanize_matplotlib
    HAS_JAPANIZE = True
except ImportError:
    HAS_JAPANIZE = False

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
    "日経平均": ("日経平均株価", "^N225"),
    "日経225": ("日経平均株価", "^N225"),
    "N225": ("日経平均株価", "^N225"),
    "S&P500": ("S&P 500", "^GSPC"),
    "SP500": ("S&P 500", "^GSPC"),
    "ナスダック": ("NASDAQ Composite", "^IXIC"),
    "NASDAQ": ("NASDAQ Composite", "^IXIC"),
    "三菱重工": ("三菱重工業", "7011.T"),
    "三菱重工業": ("三菱重工業", "7011.T"),
    "7011": ("三菱重工業", "7011.T"),
    "川崎重工": ("川崎重工業", "7012.T"),
    "川崎重工業": ("川崎重工業", "7012.T"),
    "IHI": ("IHI", "7013.T"),
    "アクセルスペース": ("アクセルスペースホールディングス", "402A.T"),
    "アクセルスペースホールディングス": ("アクセルスペースホールディングス", "402A.T"),
    "402A": ("アクセルスペースホールディングス", "402A.T"),
    "アストロスケール": ("アストロスケールホールディングス", "186A.T"),
    "アストロスケールホールディングス": ("アストロスケールホールディングス", "186A.T"),
    "186A": ("アストロスケールホールディングス", "186A.T"),
    "ispace": ("ispace", "9348.T"),
    "アイスペース": ("ispace", "9348.T"),
    "キオクシア": ("キオクシアホールディングス", "285A.T"),
    "285A": ("キオクシアホールディングス", "285A.T"),
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
    "三菱UFJフィナンシャル・グループ": ("三菱UFJフィナンシャル・グループ", "8306.T"),
    "レーザーテック": ("レーザーテック", "6920.T"),
    "東京エレクトロン": ("東京エレクトロン", "8035.T"),
    "NTT": ("日本電信電話", "9432.T"),
    "日本電信電話": ("日本電信電話", "9432.T"),
    "楽天": ("楽天グループ", "4755.T"),
    "楽天グループ": ("楽天グループ", "4755.T"),
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

FEATURES = ['Close', 'Volume', 'SMA_20', 'RSI', 'MACD', 'Return']

FEATURE_LABELS_JP = {
    'Close': '終値 (Close)',
    'Volume': '出来高 (Volume)',
    'SMA_20': '20日移動平均 (SMA_20)',
    'RSI': 'RSI (相対力指数)',
    'MACD': 'MACD',
    'Return': '前日比リターン (Return)',
}

st.sidebar.link_button(
    "📖 このダッシュボードの解説を見る",
    "https://sa19ky89-hash.github.io/Stock-app/",
    use_container_width=True,
)
st.sidebar.markdown("---")

st.sidebar.header("設定")

user_input = st.sidebar.text_input(
    "企業名・指数 または 銘柄コードを入力", value="三菱重工"
).strip()

if user_input in UNLISTED_COMPANIES:
    st.warning(f"⚠️ {UNLISTED_COMPANIES[user_input]}")
    st.stop()

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

period_options = {"1年": "1y", "2年": "2y", "5年": "5y"}
selected_period_label = st.sidebar.selectbox(
    "学習用データ取得期間", list(period_options.keys())
)
period = period_options[selected_period_label]

st.sidebar.markdown("---")
st.sidebar.subheader("検証・シミュレーション設定")

n_splits = st.sidebar.slider(
    "時系列分割（CV）の分割数", min_value=3, max_value=10, value=5,
    help="過去→未来の順序を守ったまま、検証用に何分割するか",
)

n_simulations = st.sidebar.slider(
    "モンテカルロ シミュレーション回数", min_value=100, max_value=1000,
    value=300, step=100,
    help="回数が多いほど信頼区間は安定するが、計算に時間がかかる",
)

random_seed = 42


@st.cache_data(ttl=3600)
def load_and_prep_data(symbol, time_period):
    try:
        df = yf.download(symbol, period=time_period, progress=False)
        if df.empty:
            return None

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df = df.loc[:, ~df.columns.duplicated()].copy()

        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in required_cols:
            if col not in df.columns:
                return None
            if isinstance(df[col], pd.DataFrame):
                df[col] = df[col].iloc[:, 0]
            df[col] = pd.to_numeric(df[col], errors='coerce')

        close_s = df['Close'].astype(float)

        df['SMA_20'] = SMAIndicator(close=close_s, window=20).sma_indicator()
        df['RSI'] = RSIIndicator(close=close_s, window=14).rsi()
        macd = MACD(close=close_s)
        df['MACD'] = macd.macd()
        df['Return'] = close_s.pct_change()

        df['Target_Return'] = df['Return'].shift(-1)

        df.dropna(inplace=True)
        return df
    except Exception as e:
        st.error(f"データの前処理中にエラーが発生しました: {e}")
        return None


df = load_and_prep_data(ticker, period)

if df is None or len(df) < 20:
    st.error(
        f"「{user_input}」（コード: {ticker}）のデータ取得に失敗しました。正しい名称またはコード（例: 三菱重工, 7011.T, AAPL）を入力してください。"
    )
    st.stop()

st.markdown(f"## 🏢 {display_name}  `({ticker})`")
st.markdown("---")

X = df[FEATURES].astype(float)
y = df['Target_Return'].astype(float)

st.subheader("🧪 モデル精度の検証（時系列分割 / Time Series CV）")

st.caption(
    "通常のランダム分割ではなく、「過去のデータだけで学習 → その先の未来データで検証」"
    "を守った時系列分割（TimeSeriesSplit）で検証しています。"
    "これにより、未来のデータが学習に混入する「リーク」を防ぎます。"
)


@st.cache_data(ttl=3600)
def run_timeseries_cv(_X, _y, n_splits, seed):
    tscv = TimeSeriesSplit(n_splits=n_splits)

    fold_results = []
    all_residuals = []
    last_fold_actual = None
    last_fold_pred = None
    last_fold_dates = None

    for fold_idx, (train_idx, test_idx) in enumerate(tscv.split(_X), start=1):
        X_train, X_test = _X.iloc[train_idx], _X.iloc[test_idx]
        y_train, y_test = _y.iloc[train_idx], _y.iloc[test_idx]

        fold_model = lgb.LGBMRegressor(
            random_state=seed, verbose=-1, n_estimators=100
        )
        fold_model.fit(X_train, y_train)
        preds = fold_model.predict(X_test)

        mae = float(mean_absolute_error(y_test, preds))
        rmse = float(np.sqrt(mean_squared_error(y_test, preds)))

        residuals = (y_test.values - preds).astype(float)
        all_residuals.extend(residuals.tolist())

        fold_results.append({
            "フォールド": fold_idx,
            "学習データ数": len(train_idx),
            "検証データ数": len(test_idx),
            "MAE": mae,
            "RMSE": rmse,
        })

        last_fold_actual = y_test.values
        last_fold_pred = preds
        last_fold_dates = _X.index[test_idx]

    cv_df = pd.DataFrame(fold_results)
    residual_pool = np.array(all_residuals, dtype=float)

    return cv_df, residual_pool, last_fold_actual, last_fold_pred, last_fold_dates


cv_df, residual_pool, last_actual, last_pred, last_dates = run_timeseries_cv(
    X, y, n_splits, random_seed
)

avg_mae = float(cv_df["MAE"].mean())
avg_rmse = float(cv_df["RMSE"].mean())

cv_col1, cv_col2, cv_col3 = st.columns(3)
cv_col1.metric("平均 MAE（平均絶対誤差）", f"{avg_mae * 100:.3f} %")
cv_col2.metric("平均 RMSE（二乗平均平方根誤差）", f"{avg_rmse * 100:.3f} %")
cv_col3.metric("残差サンプル数（不確実性の源）", f"{len(residual_pool):,} 件")

st.dataframe(
    cv_df.style.format({"MAE": "{:.4%}", "RMSE": "{:.4%}"}),
    use_container_width=True,
)

with st.expander("📉 最終フォールド：実測 vs 予測（翌日リターン）を表示"):
    fig_cv = go.Figure()
    fig_cv.add_trace(
        go.Scatter(
            x=last_dates, y=last_actual, mode="lines", name="実測リターン",
            line=dict(color="#1f77b4", width=1.8),
        )
    )
    fig_cv.add_trace(
        go.Scatter(
            x=last_dates, y=last_pred, mode="lines", name="予測リターン",
            line=dict(color="#ff7f0e", width=1.8, dash="dot"),
        )
    )
    fig_cv.update_layout(
        height=320,
        margin=dict(l=20, r=20, t=20, b=20),
        yaxis_title="翌日リターン",
        hovermode="x unified",
    )
    st.plotly_chart(fig_cv, use_container_width=True)
    st.caption(
        "この2本の線が近いほど精度が高いことを意味します。ズレが大きい場合、"
        "そのズレ（残差）がモンテカルロシミュレーションのブレ幅として使われます。"
    )

st.markdown("---")

model = lgb.LGBMRegressor(random_state=random_seed, verbose=-1, n_estimators=100)
model.fit(X, y)

last_date = df.index[-1]
future_dates = pd.date_range(
    start=last_date + pd.Timedelta(days=1), periods=35, freq="B"
)
future_dates = [d for d in future_dates if d > last_date][:20]

st.subheader("🤖 AIによる今後1ヶ月（20営業日）の株価予測（信頼区間つき）")

st.caption(
    "1本の予測線だけでは「どれだけ当たるか」が分からないため、"
    "時系列CVで得られた予測誤差（残差）を毎回ランダムに上乗せしながら、"
    f"{n_simulations}回のシミュレーションを行い、その分布から信頼区間を求めています。"
)


def _extract_indicator_state(close_series):
    """過去の終値から、RSI（Wilder平滑）とMACD（12/26日EMA）の
    「内部の平滑値」を、最終営業日時点の状態として復元する。

    ta ライブラリはRSI/MACDの最終出力値しか公開していないが、
    RSI・MACDはいずれも指数平滑移動平均（EMA）なので、
    ここで同じ計算式を使い、最終日時点のemaup/emadown（RSI用）と
    ema_fast/ema_slow（MACD用）を再現する。これにより、以降の
    シミュレーションでは「全履歴を再計算」せず「1ステップ分だけ
    式で更新」する差分計算に切り替えられる（数学的には同じ結果）。"""
    diff = close_series.diff(1)
    up = diff.where(diff > 0, 0.0)
    down = -diff.where(diff < 0, 0.0)

    alpha_rsi = 1 / 14
    emaup = up.ewm(alpha=alpha_rsi, adjust=False).mean()
    emadown = down.ewm(alpha=alpha_rsi, adjust=False).mean()

    ema_fast = close_series.ewm(span=12, adjust=False).mean()
    ema_slow = close_series.ewm(span=26, adjust=False).mean()

    return {
        "emaup": float(emaup.iloc[-1]),
        "emadown": float(emadown.iloc[-1]),
        "ema_fast": float(ema_fast.iloc[-1]),
        "ema_slow": float(ema_slow.iloc[-1]),
        "last_20_closes": close_series.iloc[-20:].to_numpy(dtype=float),
    }


@st.cache_data(ttl=3600, show_spinner=False)
def run_monte_carlo_forecast(_model, base_df, residual_pool, features,
                              future_dates, n_sims, seed):
    """全シミュレーションをベクトル化して一括計算する高速版。

    以前は「シミュレーション回数 × 日数」回、1行ずつpredictし、
    かつSMA/RSI/MACDを全履歴に対して再計算していたが、
    ここでは「日数」回のループに圧縮し、各ステップで
    全シミュレーション分を配列演算・バッチpredictでまとめて処理する。
    """
    rng = np.random.default_rng(seed)
    n_days = len(future_dates)

    close_series = base_df["Close"].astype(float)
    state = _extract_indicator_state(close_series)

    mean_volume = float(base_df["Volume"].mean())
    alpha_rsi = 1 / 14
    alpha_fast = 2 / (12 + 1)
    alpha_slow = 2 / (26 + 1)

    # 各シミュレーションの「現在の状態」をn_sims本のベクトルとして保持
    last_close = np.full(n_sims, float(close_series.iloc[-1]))
    last_return = np.full(n_sims, float(base_df["Return"].iloc[-1]))
    emaup = np.full(n_sims, state["emaup"])
    emadown = np.full(n_sims, state["emadown"])
    ema_fast = np.full(n_sims, state["ema_fast"])
    ema_slow = np.full(n_sims, state["ema_slow"])
    # SMA_20用：直近20日終値のローリングウィンドウ（シミュレーションごとに保持）
    window20 = np.tile(state["last_20_closes"], (n_sims, 1))

    price_paths = np.zeros((n_sims, n_days), dtype=float)

    for i in range(n_days):
        sma20 = window20.mean(axis=1)
        rsi = np.where(
            emadown == 0, 100.0, 100.0 - 100.0 / (1.0 + emaup / emadown)
        )
        macd = ema_fast - ema_slow

        # その日について、全シミュレーション分の特徴量を1つのDataFrameにまとめる
        feat_df = pd.DataFrame({
            "Close": last_close,
            "Volume": np.full(n_sims, mean_volume),
            "SMA_20": sma20,
            "RSI": rsi,
            "MACD": macd,
            "Return": last_return,
        })[features]

        # ここが最大の高速化ポイント：n_sims回ではなく1回のpredictで済ませる
        pred_returns = _model.predict(feat_df)
        noise = rng.choice(residual_pool, size=n_sims)
        adj_returns = pred_returns + noise

        next_close = last_close * (1.0 + adj_returns)

        # RSI・MACDの状態を「全履歴再計算」ではなく「1ステップ分の式」で更新
        gain = np.clip(next_close - last_close, 0, None)
        loss = np.clip(last_close - next_close, 0, None)
        emaup = alpha_rsi * gain + (1 - alpha_rsi) * emaup
        emadown = alpha_rsi * loss + (1 - alpha_rsi) * emadown
        ema_fast = alpha_fast * next_close + (1 - alpha_fast) * ema_fast
        ema_slow = alpha_slow * next_close + (1 - alpha_slow) * ema_slow

        # SMA_20用のローリングウィンドウを1日分スライド
        window20 = np.concatenate([window20[:, 1:], next_close[:, None]], axis=1)

        last_return = adj_returns
        last_close = next_close
        price_paths[:, i] = next_close

    return price_paths


with st.spinner(f"{n_simulations}回のシミュレーションを実行中..."):
    price_paths = run_monte_carlo_forecast(
        model, df, residual_pool, FEATURES, future_dates,
        n_simulations, random_seed,
    )

p05 = np.percentile(price_paths, 5, axis=0)
p25 = np.percentile(price_paths, 25, axis=0)
p50 = np.percentile(price_paths, 50, axis=0)
p75 = np.percentile(price_paths, 75, axis=0)
p95 = np.percentile(price_paths, 95, axis=0)

start_p = float(df['Close'].iloc[-1])
end_median = float(p50[-1])
end_p05 = float(p05[-1])
end_p95 = float(p95[-1])
total_change = float(((end_median - start_p) / start_p) * 100)

col1, col2, col3 = st.columns(3)
col1.metric("現在（最新）の株価", f"¥{start_p:,.1f}")
col2.metric(
    "1ヶ月後の予想株価（中央値）",
    f"¥{end_median:,.1f}",
    f"{end_median - start_p:+,.1f} ({total_change:+.1f}%)",
)
col3.metric(
    "90%信頼区間（5%〜95%タイル）",
    f"¥{end_p05:,.1f} 〜 ¥{end_p95:,.1f}",
)

st.subheader("📈 1ヶ月予測株価チャート（信頼区間つき）")

fig_pred = go.Figure()

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

x_future = [df.index[-1]] + list(future_dates)


def _with_anchor(arr):
    return [start_p] + list(arr)


p05_full = _with_anchor(p05)
p25_full = _with_anchor(p25)
p50_full = _with_anchor(p50)
p75_full = _with_anchor(p75)
p95_full = _with_anchor(p95)

fig_pred.add_trace(
    go.Scatter(
        x=x_future, y=p95_full, mode="lines",
        line=dict(width=0), showlegend=False, hoverinfo="skip",
    )
)
fig_pred.add_trace(
    go.Scatter(
        x=x_future, y=p05_full, mode="lines",
        line=dict(width=0), fill="tonexty",
        fillcolor="rgba(255,127,14,0.15)",
        name="90%信頼区間 (5〜95%)",
    )
)

fig_pred.add_trace(
    go.Scatter(
        x=x_future, y=p75_full, mode="lines",
        line=dict(width=0), showlegend=False, hoverinfo="skip",
    )
)
fig_pred.add_trace(
    go.Scatter(
        x=x_future, y=p25_full, mode="lines",
        line=dict(width=0), fill="tonexty",
        fillcolor="rgba(255,127,14,0.35)",
        name="50%信頼区間 (25〜75%)",
    )
)

fig_pred.add_trace(
    go.Scatter(
        x=x_future, y=p50_full, mode="lines+markers",
        name="AI予測（中央値）",
        line=dict(color="#ff7f0e", width=2.5, dash="dash"),
        marker=dict(size=5),
    )
)

fig_pred.update_layout(
    height=430,
    margin=dict(l=20, r=20, t=20, b=20),
    xaxis_title="日付",
    yaxis_title="株価 (円)",
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
)

st.plotly_chart(fig_pred, use_container_width=True)

st.caption(
    "濃い帯（50%信頼区間）は「シミュレーションの半分がこの範囲に収まった」ことを、"
    "薄い帯（90%信頼区間）は「9割がこの範囲に収まった」ことを示します。"
    "帯が広いほど、その先の予測の不確実性が高いことを意味します。"
)

st.subheader("📅 今後1ヶ月の日別予想株価一覧（信頼区間つき）")

pred_records = []
prev_close = start_p
for i, f_date in enumerate(future_dates):
    median_price = float(p50[i])
    diff = median_price - prev_close
    pred_records.append({
        "日付": f_date.strftime("%Y-%m-%d (%a)"),
        "予想株価（中央値）": round(median_price, 1),
        "前日比": round(diff, 1),
        "5%タイル（弱気）": round(float(p05[i]), 1),
        "95%タイル（強気）": round(float(p95[i]), 1),
    })
    prev_close = median_price

pred_df = pd.DataFrame(pred_records)

st.dataframe(
    pred_df.style.format({
        "予想株価（中央値）": "¥{:,.1f}",
        "前日比": "{:+,.1f}",
        "5%タイル（弱気）": "¥{:,.1f}",
        "95%タイル（強気）": "¥{:,.1f}",
    }),
    use_container_width=True,
    height=350,
)

st.caption(
    "※表示される株価推移はAI機械学習モデルによるシミュレーション結果です。"
    "信頼区間はあくまで過去の検証誤差から推定した統計的な目安であり、"
    "将来の値動きを保証するものではありません。実際の投資判断には使用しないでください。"
)

st.markdown("---")

st.subheader("💡 AIがこの銘柄の予測で重視した指標（特徴量重要度 & SHAP分析）")

st.caption(
    "LightGBMモデルが予測するにあたり、どの指標を重視したか（Gain）、および「その指標が高いと株価を上げる（＋）／下げる（−）どちらに作用したか（SHAP値）」を可視化しています。"
    "なお、この重要度は学習データにおける傾向であり、将来も同じ指標が有効である保証はありません。"
)


@st.cache_data(ttl=3600, show_spinner=False)
def compute_shap_values(_model, _X):
    """SHAP値を計算してキャッシュする。
    銘柄・期間が変わらない限り（サイドバーの他のスライダー操作では）
    再計算しないようにするための切り出し。"""
    explainer = shap.TreeExplainer(_model)
    shap_res = explainer(_X)

    if hasattr(shap_res, "values"):
        shap_array = np.array(shap_res.values, dtype=float)
    else:
        shap_array = np.array(shap_res, dtype=float)

    if shap_array.ndim == 3:
        shap_array = shap_array[:, :, 0]

    return shap_array

col_imp1, col_imp2 = st.columns(2)

with col_imp1:
    st.markdown("##### 1. 重要度の大きさ (Gainスコア)")
    importances = model.booster_.feature_importance(importance_type="gain")
    importance_df = pd.DataFrame({
        'Feature': [FEATURE_LABELS_JP.get(f, f) for f in FEATURES],
        'Importance': importances
    }).sort_values(by='Importance', ascending=True)

    fig_imp = go.Figure(
        go.Bar(
            x=importance_df['Importance'],
            y=importance_df['Feature'],
            orientation='h',
            marker=dict(color='#0284c7'),
        )
    )

    fig_imp.update_layout(
        height=320,
        margin=dict(l=20, r=20, t=20, b=20),
        xaxis_title="重要度スコア (Gain)",
        yaxis_title="テクニカル指標",
    )
    st.plotly_chart(fig_imp, use_container_width=True)

with col_imp2:
    st.markdown("##### 2. プラス/マイナス影響の方向性 (SHAP Summary)")
    try:
        shap_array = compute_shap_values(model, X)

        if HAS_JAPANIZE:
            X_shap = X.rename(columns=FEATURE_LABELS_JP)
        else:
            X_shap = X.copy()

        fig_shap, ax = plt.subplots(figsize=(7, 4.5))
        shap.summary_plot(
            shap_array, X_shap, show=False, plot_size=None
        )
        plt.tight_layout()
        st.pyplot(fig_shap)
        plt.close(fig_shap)

        if not HAS_JAPANIZE:
            st.caption("※日本語フォントライブラリ未導入のためY軸は英語表記です（`requirements.txt` に `japanize-matplotlib` を追加すると日本語化されます）。")
    except Exception as e:
        st.info("※SHAP分析グラフの表示をスキップしました。左側のGainスコアをご参照ください。")

with st.expander("📖 SHAPグラフの見方ガイド"):
    st.markdown("""
    * **横軸（SHAP value）**: 0より右（プラス）は株価を**押し上げる**影響、0より左（マイナス）は株価を**押し下げる**影響を意味します。
    * **ドットの色**:
      * <span style="color:red; font-weight:bold;">赤色 (High)</span> : そのテクニカル指標の値が高い状態（例: RSIが高い、出来高が多いなど）
      * <span style="color:blue; font-weight:bold;">青色 (Low)</span> : そのテクニカル指標の値が低い状態（例: RSIが低い、出来高が少ないなど）
    * **読み解き例**: 「赤色のドットが右側（プラス領域）に多い」場合 ➔ **「その指標が高くなると、AIは株価が上がると判断しやすい」** ことを表します。
    """, unsafe_allow_html=True)

st.markdown("---")

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
