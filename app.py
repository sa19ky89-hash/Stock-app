import datetime
import lightgbm as lgb
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

    # 重工・防衛・造船銘柄
    "三菱重工": ("三菱重工業", "7011.T"),
    "三菱重工業": ("三菱重工業", "7011.T"),
    "7011": ("三菱重工業", "7011.T"),
    "川崎重工": ("川崎重工業", "7012.T"),
    "川崎重工業": ("川崎重工業", "7012.T"),
    "IHI": ("IHI", "7013.T"),

    # 宇宙関連銘柄
    "アクセルスペース": ("アクセルスペースホールディングス", "402A.T"),
    "アクセルスペースホールディングス": ("アクセルスペースホールディングス", "402A.T"),
    "402A": ("アクセルスペースホールディングス", "402A.T"),
    "アストロスケール": ("アストロスケールホールディングス", "186A.T"),
    "アストロスケールホールディングス": ("アストロスケールホールディングス", "186A.T"),
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
    "三菱UFJフィナンシャル・グループ": ("三菱UFJフィナンシャル・グループ", "8306.T"),
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

FEATURES = ['Close', 'Volume', 'SMA_20', 'RSI', 'MACD', 'Return']

# 日本語表示用ラベル辞書
FEATURE_LABELS_JP = {
    'Close': '終値 (Close)',
    'Volume': '出来高 (Volume)',
    'SMA_20': '20日移動平均 (SMA_20)',
    'RSI': 'RSI (相対力指数)',
    'MACD': 'MACD',
    'Return': '前日比リターン (Return)',
}

# --------------------------------------------------
# サイドバー（設定パラメータ）
# --------------------------------------------------
st.sidebar.header("設定")

user_input = st.sidebar.text_input(
    "企業名・指数 または 銘柄コードを入力", value="三菱重工"
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
        f"「{user_input}」（コード: {ticker}）のデータ取得に失敗しました。正しい名称またはコード（例: 三菱重工, 7011.T, AAPL）を入力してください。"
    )
    st.stop()

# --------------------------------------------------
# メイン画面表示
# --------------------------------------------------

st.markdown(f"## 🏢 {display_name}  `({ticker})`")
st.markdown("---")

X = df[FEATURES]
y = df['Target_Return']

# --------------------------------------------------
# 1. 時系列分割による精度検証（Walk-forward validation）
# --------------------------------------------------
st.subheader("🧪 モデル精度の検証（時系列分割 / Time Series CV）")

st.caption(
    "通常のランダム分割ではなく、「過去のデータだけで学習 → その先の未来データで検証」"
    "を守った時系列分割（TimeSeriesSplit）で検証しています。"
    "により、未来のデータが学習に混入する「リーク」を防ぎます。"
)


@st.cache_data(ttl=3600)
def run_timeseries_cv(_X, _y, n_splits, seed):
    """時系列分割でLightGBMを検証し、各フォールドの誤差指標と
    予測残差（実測 - 予測）を集める。残差はのちのモンテカルロ
    シミュレーションで「現実的なブレ幅」として再利用する。"""
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

        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))

        residuals = y_test.values - preds
        all_residuals.extend(residuals.tolist())

        fold_results.append({
            "フォールド": fold_idx,
            "学習データ数": len(train_idx),
            "検証データ数": len(test_idx),
            "MAE": mae,
            "RMSE": rmse,
        })

        # 最後のフォールドを可視化用に保持
        last_fold_actual = y_test.values
        last_fold_pred = preds
        last_fold_dates = _X.index[test_idx]

    cv_df = pd.DataFrame(fold_results)
    residual_pool = np.array(all_residuals)

    return cv_df, residual_pool, last_fold_actual, last_fold_pred, last_fold_dates


cv_df, residual_pool, last_actual, last_pred, last_dates = run_timeseries_cv(
    X, y, n_splits, random_seed
)

avg_mae = cv_df["MAE"].mean()
avg_rmse = cv_df["RMSE"].mean()

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

# --------------------------------------------------
# 2. 本番モデルの学習（全データで学習し直す）
# --------------------------------------------------
model = lgb.LGBMRegressor(random_state=random_seed, verbose=-1, n_estimators=100)
model.fit(X, y)

# 未来20営業日の日付作成
last_date = df.index[-1]
future_dates = pd.date_range(
    start=last_date + pd.Timedelta(days=1), periods=35, freq="B"
)
future_dates = [d for d in future_dates if d > last_date][:20]


# --------------------------------------------------
# 3. モンテカルロ・シミュレーションによる再帰予測 + 信頼区間
# --------------------------------------------------
st.subheader("🤖 AIによる今後1ヶ月（20営業日）の株価予測（信頼区間つき）")

st.caption(
    "1本の予測線だけでは「どれだけ当たるか」が分からないため、"
    "時系列CVで得られた予測誤差（残差）を毎回ランダムに上乗せしながら、"
    f"{n_simulations}回のシミュレーションを行い、その分布から信頼区間を求めています。"
)


@st.cache_data(ttl=3600, show_spinner=False)
def run_monte_carlo_forecast(_model, base_df, residual_pool, features,
                              future_dates, n_sims, seed):
    """再帰予測をn_sims回繰り返す。各ステップで
    予測リターンにブートストラップした残差を加算することで、
    モデルが説明しきれない「現実のブレ」を反映する。"""
    rng = np.random.default_rng(seed)
    n_days = len(future_dates)
    price_paths = np.zeros((n_sims, n_days))

    for s in range(n_sims):
        sim_df = base_df.copy()
        for i, f_date in enumerate(future_dates):
            latest_feat = sim_df[features].iloc[[-1]]
            pred_return = _model.predict(latest_feat)[0]

            # 検証で得た残差からランダムに1つ抽出し、予測に上乗せする
            noise = rng.choice(residual_pool)
            adj_return = pred_return + noise

            last_close = float(sim_df['Close'].iloc[-1])
            next_close = last_close * (1 + adj_return)

            new_row = pd.DataFrame(
                {
                    "Open": next_close,
                    "High": next_close,
                    "Low": next_close,
                    "Close": next_close,
                    "Volume": sim_df["Volume"].mean(),
                    "Return": adj_return,
                },
                index=[f_date],
            )
            sim_df = pd.concat([sim_df, new_row])

            sim_df["SMA_20"] = SMAIndicator(
                close=sim_df["Close"], window=20
            ).sma_indicator()
            sim_df["RSI"] = RSIIndicator(
                close=sim_df["Close"], window=14
            ).rsi()
            sim_df["MACD"] = MACD(close=sim_df["Close"]).macd()

            price_paths[s, i] = next_close

    return price_paths


with st.spinner(f"{n_simulations}回のシミュレーションを実行中..."):
    price_paths = run_monte_carlo_forecast(
        model, df, residual_pool, FEATURES, future_dates,
        n_simulations, random_seed,
    )

# パーセンタイルの計算（5%, 25%, 50%(中央値), 75%, 95%）
p05 = np.percentile(price_paths, 5, axis=0)
p25 = np.percentile(price_paths, 25, axis=0)
p50 = np.percentile(price_paths, 50, axis=0)
p75 = np.percentile(price_paths, 75, axis=0)
p95 = np.percentile(price_paths, 95, axis=0)

start_p = float(df['Close'].iloc[-1])
end_median = p50[-1]
end_p05 = p05[-1]
end_p95 = p95[-1]
total_change = ((end_median - start_p) / start_p) * 100

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

# --------------------------------------------------
# 4. 実績＋予測ファンチャート（信頼区間の帯グラフ）
# --------------------------------------------------
st.subheader("📈 1ヶ月予測株価チャート（信頼区間つき）")

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

# 起点（最終実績日）を各系列の先頭に接続
x_future = [df.index[-1]] + list(future_dates)


def _with_anchor(arr):
    return [start_p] + list(arr)


p05_full = _with_anchor(p05)
p25_full = _with_anchor(p25)
p50_full = _with_anchor(p50)
p75_full = _with_anchor(p75)
p95_full = _with_anchor(p95)

# 90%信頼区間（5%〜95%）の帯
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

# 50%信頼区間（25%〜75%）の帯
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

# 中央値ライン
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

# --------------------------------------------------
# 5. 日別予測価格テーブル（信頼区間つき）
# --------------------------------------------------
st.subheader("📅 今後1ヶ月の日別予想株価一覧（信頼区間つき）")

pred_records = []
prev_close = start_p
for i, f_date in enumerate(future_dates):
    median_price = p50[i]
    diff = median_price - prev_close
    pred_records.append({
        "日付": f_date.strftime("%Y-%m-%d (%a)"),
        "予想株価（中央値）": round(median_price, 1),
        "前日比": round(diff, 1),
        "5%タイル（弱気）": round(p05[i], 1),
        "95%タイル（強気）": round(p95[i], 1),
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

# --------------------------------------------------
# 6. AIが重視した指標（特徴量重要度 & SHAP方向性分析）
# --------------------------------------------------
st.subheader("💡 AIがこの銘柄の予測で重視した指標（特徴量重要度 & SHAP分析）")

st.caption(
    "LightGBMモデルが予測するにあたり、どの指標を重視したか（Gain）、および「その指標が高いと株価を上げる（＋）／下げる（−）どちらに作用したか（SHAP値）」を可視化しています。"
)

col_imp1, col_imp2 = st.columns(2)

with col_imp1:
    st.markdown("##### 1. 重要度の大きさ (Gainスコア)")
    # Gain（予測精度向上の貢献度）に基づく重要度の抽出
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
    # SHAP値の計算
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X)

    # 日本語名ラベルを適用したデータフレームの作成
    X_jp = X.rename(columns=FEATURE_LABELS_JP)

    # SHAP Bee Swarm プロットの作成
    fig_shap, ax = plt.subplots(figsize=(7, 4.5))
    shap.summary_plot(
        shap_values.values, X_jp, show=False, plot_size=None
    )
    plt.tight_layout()
    st.pyplot(fig_shap)

with st.expander("📖 SHAPグラフの見方ガイド"):
    st.markdown("""
    * **横軸（SHAP value）**: 0より右（プラス）は株価を**押し上げる**影響、0より左（マイナス）は株価を**押し下げる**影響を意味します。
    * **ドットの色**:
      * <span style="color:red; font-weight:bold;">赤色 (High)</span> : そのテクニカル指標の値が高い状態（例: RSIが高い、出来高が多いなど）
      * <span style="color:blue; font-weight:bold;">青色 (Low)</span> : そのテクニカル指標の値が低い状態（例: RSIが低い、出来高が少ないなど）
    * **読み解き例**: 「赤色のドットが右側（プラス領域）に多い」場合 ➔ **「その指標が高くなると、AIは株価が上がると判断しやすい」** ことを表します。
    """, unsafe_allow_unsafe_scale=True)

st.markdown("---")

# --------------------------------------------------
# 7. 過去の実績ローソク足チャート
# --------------------------------------------------
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
