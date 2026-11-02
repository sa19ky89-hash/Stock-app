st.sidebar.markdown("---")
st.sidebar.subheader("🔍 全銘柄トレンド＆過熱感 スクリーニング")

if st.sidebar.button("全銘柄の現在のトレンドを判定する"):
    with st.spinner("各銘柄の最新データを取得し、テクニカル分析を実行中...（数分かかる場合があります）"):
        trend_records = []
        
        # 重複を避けるためティッカーのセットを作成
        unique_tickers = {}
        for name, ticker in COMPANY_MAP.items():
            if ticker not in unique_tickers.values():
                unique_tickers[name] = ticker

        for name, ticker in unique_tickers.items():
            try:
                # 分析には直近6ヶ月のデータがあれば十分
                df_scan = yf.download(ticker, period="6mo", progress=False)
                if df_scan.empty or len(df_scan) < 30:
                    continue
                
                # 終値の取得
                close_s = df_scan['Close']
                if isinstance(close_s, pd.DataFrame):
                    close_s = close_s.iloc[:, 0]
                close_s = pd.to_numeric(close_s, errors='coerce')
                
                # --- 指標の計算 ---
                # 1. 移動平均線 (トレンド系: 相場の方向性を把握) 
                sma_20 = SMAIndicator(close=close_s, window=20).sma_indicator()
                # 2. MACD (複合: トレンドの転換期を捉える)
                macd_obj = MACD(close=close_s)
                macd_line = macd_obj.macd()
                macd_sig = macd_obj.macd_signal()
                # 3. RSI (オシレーター系: 過熱感を測る)
                rsi_14 = RSIIndicator(close=close_s, window=14).rsi()
                
                last_close = float(close_s.iloc[-1])
                last_sma20 = float(sma_20.iloc[-1])
                last_macd = float(macd_line.iloc[-1])
                last_macdsig = float(macd_sig.iloc[-1])
                last_rsi = float(rsi_14.iloc[-1])
                
                # ▼ トレンド判定ロジック ▼
                # 上昇トレンド：株価が20日移動平均を上回り、かつMACDがシグナル線を上回っている
                is_uptrend = (last_close > last_sma20) and (last_macd > last_macdsig)
                # 下降トレンド：株価が20日移動平均を下回り、かつMACDがシグナル線を下回っている
                is_downtrend = (last_close < last_sma20) and (last_macd < last_macdsig)
                
                trend_status = "🔄 もみ合い・レンジ"
                if is_uptrend:
                    trend_status = "↗️ 上昇トレンド"
                elif is_downtrend:
                    trend_status = "↘️ 下降トレンド"
                    
                # ▼ 過熱感（RSI）判定ロジック ▼
                overheat_status = "-"
                if last_rsi >= 70:
                    overheat_status = "🔥 買われすぎ"
                elif last_rsi <= 30:
                    overheat_status = "🧊 売られすぎ"
                    
                trend_records.append({
                    "銘柄名": name,
                    "現在値": last_close,
                    "トレンド状態": trend_status,
                    "SMA20乖離率(%)": (last_close - last_sma20) / last_sma20 * 100,
                    "RSI": last_rsi,
                    "過熱感": overheat_status
                })
            except Exception:
                pass # 取得失敗した銘柄はスキップ
                
        if trend_records:
            trend_df = pd.DataFrame(trend_records)
            
            # 上昇・下降トレンドそれぞれでフィルター
            up_df = trend_df[trend_df["トレンド状態"] == "↗️ 上昇トレンド"].sort_values(by="SMA20乖離率(%)", ascending=False)
            down_df = trend_df[trend_df["トレンド状態"] == "↘️ 下降トレンド"].sort_values(by="SMA20乖離率(%)", ascending=True)
            
            st.markdown("### 📊 本日のトレンド＆過熱感判定結果")
            
            col_up, col_down = st.columns(2)
            with col_up:
                st.success(f"↗️ 上昇トレンドの銘柄 ({len(up_df)}件)")
                st.dataframe(up_df[["銘柄名", "現在値", "SMA20乖離率(%)", "RSI", "過熱感"]].style.format({
                    "現在値": "{:,.1f}", 
                    "SMA20乖離率(%)": "{:+.2f}%",
                    "RSI": "{:.1f}"
                }), use_container_width=True)
                
            with col_down:
                st.error(f"↘️ 下降トレンドの銘柄 ({len(down_df)}件)")
                st.dataframe(down_df[["銘柄名", "現在値", "SMA20乖離率(%)", "RSI", "過熱感"]].style.format({
                    "現在値": "{:,.1f}", 
                    "SMA20乖離率(%)": "{:+.2f}%",
                    "RSI": "{:.1f}"
                }), use_container_width=True)

