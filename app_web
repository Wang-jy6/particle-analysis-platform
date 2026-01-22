import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import cv2
import io

# ================= 1. 页面配置 =================
st.set_page_config(page_title="微粒云分析平台", page_icon="☁️", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    div[data-testid="stFileUploader"] {
        border: 2px dashed #4CAF50;
        padding: 20px;
        border-radius: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

DEFAULT_COLORS = {
    'Si': '#FF0000', 'O': '#00FF00', 'C': '#0000FF',
    'Ca': '#FFFF00', 'Al': '#FF00FF', 'Fe': '#00FFFF',
    'S': '#FFA500', 'Cl': '#808080', 'K': '#800080',
    'Mg': '#8B4513', 'Na': '#000080', 'Ti': '#A52A2A'
}

# ================= 2. 内存数据处理函数 =================
# 注意：网页版直接读取内存中的文件流 (BytesIO)，而不是硬盘路径

@st.cache_data
def process_uploaded_files(uploaded_files):
    """处理用户上传的文件列表"""
    data_map = {}
    spectrum_data = {'x': [], 'y': [], 'meta': {}}
    error_log = []

    for uploaded_file in uploaded_files:
        fname = uploaded_file.name
        
        # --- A. 处理 CSV ---
        if fname.endswith(".csv") and "电子图像" not in fname:
            # 提取元素名
            el_name = fname.split(" ")[0].split(".")[0]
            if "_" in el_name: el_name = el_name.split("_")[-1]
            try:
                # 直接从内存读取
                df = pd.read_csv(uploaded_file, header=None)
                mat = df.apply(pd.to_numeric, errors='coerce').fillna(0).values
                data_map[el_name] = mat
            except: error_log.append(f"解析失败: {fname}")

        # --- B. 处理 Excel ---
        elif fname.endswith((".xls", ".xlsx")):
            try:
                xls = pd.ExcelFile(uploaded_file)
                for sheet in xls.sheet_names:
                    # 简单逻辑：Sheet名或文件名
                    clean_sheet = sheet.strip()
                    target_name = clean_sheet if len(clean_sheet) < 5 else fname.split(".")[0]
                    
                    df = pd.read_excel(xls, sheet_name=sheet, header=None)
                    mat = df.apply(pd.to_numeric, errors='coerce').fillna(0).values
                    if mat.size > 100:
                        data_map[target_name] = mat
            except: error_log.append(f"Excel错误: {fname}")

        # --- C. 处理 TXT (能谱) ---
        elif fname.endswith(".txt"):
            try:
                # 需将 bytes 解码为 string
                stringio = io.StringIO(uploaded_file.getvalue().decode("utf-8", errors='ignore'))
                lines = stringio.readlines()
                is_data = False
                for line in lines:
                    line = line.strip()
                    if line.startswith("#"):
                        parts = line.split(":")
                        if len(parts) > 1:
                            spectrum_data['meta'][parts[0].replace("#", "").strip()] = parts[1].strip()
                    if "SPECTRUM" in line: is_data = True; continue
                    if is_data and "," in line:
                        try:
                            x, y = map(float, line.split(","))
                            spectrum_data['x'].append(x)
                            spectrum_data['y'].append(y)
                        except: pass
            except: error_log.append(f"能谱解析失败: {fname}")

    return data_map, spectrum_data, error_log

def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16)/255.0 for i in (0, 2, 4))

def create_composite(data_map, selected_elements, colors, threshold):
    if not data_map: return None
    shape = next(iter(data_map.values())).shape
    rgb = np.zeros((shape[0], shape[1], 3))
    
    for el in selected_elements:
        if el in data_map:
            mat = data_map[el].copy()
            if mat.shape != shape: mat = cv2.resize(mat, (shape[1], shape[0]))
            mat[mat < threshold] = 0
            if mat.max() > 0: mat = mat / mat.max()
            
            r, g, b = hex_to_rgb(colors.get(el, '#FFFFFF'))
            rgb[:, :, 0] += mat * r
            rgb[:, :, 1] += mat * g
            rgb[:, :, 2] += mat * b
    return np.clip(rgb, 0, 1)

# ================= 3. 网页布局 =================

st.title("☁️ 微粒分析云平台")
st.markdown("### 请上传单个微粒的所有数据文件 (CSV/Excel/TXT)")

# --- 文件上传区 ---
with st.container():
    uploaded_files = st.file_uploader(
        "拖拽文件到这里 (支持多选)", 
        accept_multiple_files=True,
        type=['csv', 'txt', 'xls', 'xlsx']
    )

if uploaded_files:
    # 处理数据
    data_map, spectrum, errors = process_uploaded_files(uploaded_files)
    
    if errors:
        for e in errors: st.warning(e)
        
    if not data_map:
        st.info("请上传包含元素 Mapping 数据的 CSV 或 Excel 文件。")
    else:
        # --- 侧边栏控制 ---
        st.sidebar.header("🕹️ 交互控制")
        noise_threshold = st.sidebar.slider("背景降噪", 0.0, 10.0, 2.0)
        
        all_elements = sorted(list(data_map.keys()))
        selected_elements = st.sidebar.multiselect(
            "合成元素选择", all_elements, 
            default=[e for e in ['Si', 'O', 'C', 'Ca', 'Fe'] if e in all_elements]
        )
        
        current_colors = {}
        for el in selected_elements:
            current_colors[el] = st.sidebar.color_picker(f"{el} 颜色", DEFAULT_COLORS.get(el, '#FFFFFF'))

        # --- 结果展示区 ---
        st.success(f"成功加载 {len(data_map)} 个元素通道")
        
        col1, col2 = st.columns([1.5, 1])
        
        # 1. 合成图
        with col1:
            st.subheader("🖼️ 微粒总样貌")
            if selected_elements:
                comp_img = create_composite(data_map, selected_elements, current_colors, noise_threshold)
                st.image(comp_img, use_container_width=True, clamp=True)
            else:
                st.info("请在左侧选择元素")

        # 2. 成分饼图
        with col2:
            st.subheader("📊 信号组成")
            sums = {k: v.sum() for k, v in data_map.items()}
            total = sum(sums.values()) if sums else 1
            pie_data = {k: v for k, v in sums.items() if v/total > 0.005}
            
            fig_pie = go.Figure(data=[go.Pie(labels=list(pie_data.keys()), values=list(pie_data.values()), hole=.4)])
            fig_pie.update_layout(margin=dict(t=0, b=0, l=0, r=0), height=300)
            st.plotly_chart(fig_pie, use_container_width=True)

        # 3. 能谱
        st.subheader("📈 能谱分析")
        if spectrum and spectrum['x']:
            fig_spec = go.Figure()
            fig_spec.add_trace(go.Scatter(x=spectrum['x'], y=spectrum['y'], fill='tozeroy', line=dict(color='#333')))
            fig_spec.update_layout(height=300, margin=dict(l=0, r=0, t=20, b=0), xaxis_title="keV", yaxis_title="Counts")
            st.plotly_chart(fig_spec, use_container_width=True)
            
            # 显示元数据
            if spectrum['meta']:
                st.json(spectrum['meta'], expanded=False)

        # 4. 单元素图
        st.subheader("🧩 元素分布图")
        cols = st.columns(6)
        for i, el in enumerate(all_elements):
            with cols[i % 6]:
                fig, ax = plt.subplots()
                ax.imshow(data_map[el], cmap='magma')
                ax.axis('off')
                ax.set_title(el, fontsize=8)
                st.pyplot(fig)
                plt.close(fig)

else:
    # 引导页
    st.info("👋 欢迎！这是一个在线微粒分析工具。请在上方上传文件开始使用。")
