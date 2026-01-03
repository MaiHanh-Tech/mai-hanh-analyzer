import streamlit as st
import pandas as pd
from services.blocks.cfo_data_manager import tao_data_full_kpi, validate_uploaded_data, tinh_chi_so, phat_hien_gian_lan
from ai_core import AI_Core

# ✅ THÊM DICTIONARY DỊCH CHO CFO
TRANS_CFO = {
    "vi": {
        "header": "💰 CFO Controller Dashboard",
        "data_source": "📊 **Nguồn dữ liệu**",
        "demo": "Demo (Giả)",
        "upload": "Upload Excel",
        "upload_label": "Upload file Excel",
        "create_demo": "🔄 Tạo data demo mới",
        "kpi_title": "Sức khỏe Tài chính Tháng gần nhất",
        "doanh_thu": "Doanh Thu",
        "loi_nhuan": "Lợi Nhuận ST",
        "ros": "ROS",
        "dong_tien": "Dòng Tiền",
        "cost_title": "🤖 **Trợ lý Phân tích:**",
        "cost_input": "Hỏi về chi phí...",
        "risk_title": "Quét Gian Lận (ML)",
        "risk_btn": "🔍 Quét ngay",
        "risk_clean": "Dữ liệu sạch.",
        "check_title": "Cross-Check (Đối chiếu)",
        "check_tax": "Số liệu Thuế (Tờ khai):",
        "check_erp": "Số liệu Sổ cái (ERP):",
        "check_btn": "So khớp",
        "check_match": "Khớp!",
        "whatif_title": "🎛️ What-If Analysis",
        "price_slider": "Tăng/Giảm Giá Bán (%)",
        "cost_slider": "Tăng/Giảm Chi Phí (%)",
        "profit_old": "Lợi Nhuận Gốc",
        "profit_new": "Lợi Nhuận Mới"
    },
    "en": {
        "header": "💰 CFO Controller Dashboard",
        "data_source": "📊 **Data Source**",
        "demo": "Demo (Mock)",
        "upload": "Upload Excel",
        "upload_label": "Upload Excel file",
        "create_demo": "🔄 Generate new demo data",
        "kpi_title": "Latest Month Financial Health",
        "doanh_thu": "Revenue",
        "loi_nhuan": "Gross Profit",
        "ros": "ROS",
        "dong_tien": "Cash Flow",
        "cost_title": "🤖 **Cost Analyst:**",
        "cost_input": "Ask about costs...",
        "risk_title": "Fraud Detection (ML)",
        "risk_btn": "🔍 Scan now",
        "risk_clean": "Data is clean.",
        "check_title": "Cross-Check",
        "check_tax": "Tax Declaration:",
        "check_erp": "ERP Ledger:",
        "check_btn": "Compare",
        "check_match": "Matched!",
        "whatif_title": "🎛️ What-If Analysis",
        "price_slider": "Price Change (%)",
        "cost_slider": "Cost Change (%)",
        "profit_old": "Original Profit",
        "profit_new": "New Profit"
    },
    "zh": {
        "header": "💰 CFO 控制器仪表板",
        "data_source": "📊 **数据来源**",
        "demo": "演示（模拟）",
        "upload": "上传 Excel",
        "upload_label": "上传 Excel 文件",
        "create_demo": "🔄 生成新演示数据",
        "kpi_title": "最近月份财务健康",
        "doanh_thu": "收入",
        "loi_nhuan": "毛利润",
        "ros": "ROS",
        "dong_tien": "现金流",
        "cost_title": "🤖 **成本分析师:**",
        "cost_input": "询问成本...",
        "risk_title": "欺诈检测 (ML)",
        "risk_btn": "🔍 立即扫描",
        "risk_clean": "数据干净。",
        "check_title": "交叉检查",
        "check_tax": "税务申报:",
        "check_erp": "ERP 账本:",
        "check_btn": "比较",
        "check_match": "匹配！",
        "whatif_title": "🎛️ 假设分析",
        "price_slider": "价格变动 (%)",
        "cost_slider": "成本变动 (%)",
        "profit_old": "原始利润",
        "profit_new": "新利润"
    }
}

def T(key):
    lang = st.session_state.get('cfo_lang', 'vi')
    return TRANS_CFO.get(lang, TRANS_CFO['vi']).get(key, key)

def run():
    ai = AI_Core()

    # ✅ THÊM CHỌN NGÔN NGỮ CHO CFO (sidebar riêng)
    with st.sidebar:
        st.markdown("---")
        st.selectbox(
            "🌐 Ngôn ngữ / Language / 语言",
            ["Tiếng Việt", "English", "中文"],
            key="cfo_lang"
        )

    st.header(T("header"))

    with st.sidebar:
        st.markdown(T("data_source"))
        data_source = st.radio("Chọn nguồn:", [T("demo"), T("upload")])
        if data_source == T("upload"):
            uploaded = st.file_uploader(T("upload_label"), type="xlsx")
            if uploaded:
                try:
                    df_raw = pd.read_excel(uploaded)
                    is_valid, msg = validate_uploaded_data(df_raw)
                    if is_valid:
                        st.session_state.df_fin = df_raw
                        st.success("✅ Tải data thành công!")
                    else:
                        st.error(f"❌ Lỗi data: {msg}")
                except Exception as e:
                    st.error(f"Lỗi đọc file: {e}")
        if st.button(T("create_demo")):
            st.session_state.df_fin = tao_data_full_kpi(seed=int(time.time()))
            st.rerun()

    if 'df_fin' not in st.session_state:
        st.session_state.df_fin = tao_data_full_kpi(seed=42)

    df = tinh_chi_so(st.session_state.df_fin.copy())
    last = df.iloc[-1]

    t1, t2, t3, t4 = st.tabs(["📊 KPIs & Sức Khỏe", "📉 Phân Tích Chi Phí", "🕵️ Rủi Ro & Check", "🔮 Dự Báo & What-If"])

    with t1:
        st.subheader(T("kpi_title"))
        k1, k2, k3, k4 = st.columns(4)
        k1.metric(T("doanh_thu"), f"{last['Doanh Thu']/1e9:.1f} tỷ")
        k2.metric(T("loi_nhuan"), f"{last['Lợi Nhuận ST']/1e9:.1f} tỷ")
        k3.metric("ROS", f"{last.get('ROS',0):.1f}%")
        k4.metric(T("dong_tien"), f"{last['Dòng Tiền Thực']/1e9:.1f} tỷ")
        st.line_chart(df.set_index("Tháng")[["Doanh Thu", "Lợi Nhuận ST"]])

    with t2:
        c1, c2 = st.columns([2,1])
        with c1:
            if "Giá Vốn" in df.columns and "Chi Phí VH" in df.columns:
                st.bar_chart(df.set_index("Tháng")[["Giá Vốn", "Chi Phí VH"]])
            else:
                st.info("Chưa có đủ cột dữ liệu chi phí để vẽ biểu đồ.")
        with c2:
            st.write(T("cost_title"))
            q = st.text_input(T("cost_input"))
            if q:
                with st.spinner("AI đang soi số liệu..."):
                    context = f"Dữ liệu tháng cuối: Doanh thu {last['Doanh Thu']}, Lợi nhuận {last['Lợi Nhuận ST']}."
                    res = ai.generate(q, system_instruction=f"Bạn là Kế toán trưởng. Phân tích dựa trên: {context}")
                    st.write(res)

    with t3:
        c_risk, c_check = st.columns(2)
        with c_risk:
            st.subheader(T("risk_title"))
            if st.button(T("risk_btn")):
                bad = phat_hien_gian_lan(df)
                if not bad.empty:
                    st.error(f"Phát hiện {len(bad)} tháng bất thường!")
                    st.dataframe(bad)
                else:
                    st.success(T("risk_clean"))
        with c_check:
            st.subheader(T("check_title"))
            val_a = st.number_input(T("check_tax"), value=100.0)
            val_b = st.number_input(T("check_erp"), value=105.0)
            if st.button(T("check_btn")):
                diff = val_b - val_a
                if diff != 0:
                    st.warning(f"Lệch: {diff}. Rủi ro truy thu thuế!")
                else:
                    st.success(T("check_match"))

    with t4:
        st.subheader(T("whatif_title"))
        base_rev = last['Doanh Thu']
        base_profit = last['Lợi Nhuận ST']
        c_s1, c_s2 = st.columns(2)
        with c_s1:
            delta_price = st.slider(T("price_slider"), -20, 20, 0)
        with c_s2:
            delta_cost = st.slider(T("cost_slider"), -20, 20, 0)
        new_rev = base_rev * (1 + delta_price/100)
        base_fixed_cost = last.get('Chi Phí VH', 0)
        new_profit = base_profit + (new_rev - base_rev) - (base_fixed_cost * delta_cost/100)
        col_res1, col_res2 = st.columns(2)
        col_res1.metric(T("profit_old"), f"{base_profit/1e9:.2f} tỷ")
        col_res2.metric(T("profit_new"), f"{new_profit/1e9:.2f} tỷ", delta=f"{(new_profit - base_profit)/1e9:.2f} tỷ")
