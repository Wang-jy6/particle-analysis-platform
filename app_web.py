import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from streamlit_drawable_canvas import st_canvas
import cv2
import io
import os
from scipy.signal import find_peaks

# ================= 1. 基础配置与颜色库 =================
st.set_page_config(page_title="微粒交互分析云平台", layout="wide")

# 元素特征能量库 (keV)
ELEMENT_ENERGIES = {
    'C': 0.277, 'N': 0.392, 'O': 0.525, 'Na': 1.041, 'Mg': 1.253, 
    'Al': 1.486, 'Si': 1.739, 'S': 2.307, 'Cl': 2.621, 'K': 3.312, 
    'Ca': 3.690, 'Fe': 6.398, 'Cu': 8.040, 'Zn': 8.630
}

# ================= 2. 数据处理逻辑 =================

@st.cache_data
def load_uploaded_data(uploaded_files):
    data_map = {}
    spectrum = {'x': [], 'y': [], 'meta': {}}
    
    # --- 第一步：读取原始数据 ---
    for f in uploaded_files:
        fname = f.name
        # CSV 处理
        if fname.endswith(".csv"):
            el = fname.split(" ")[0].split(".")[0].split("_")[-1]
            if "电子图像" in fname: el = "SE"
            try:
                df = pd.read_csv(f, header=None)
                # 转换为 numpy 数组
                mat = df.apply(pd.to_numeric, errors='coerce').fillna(0).values
                data_map[el] = mat
            except: pass
            
        # Excel 处理
        elif fname.endswith((".xls", ".xlsx")):
            try:
                xls = pd.ExcelFile(f)
                for sheet in xls.sheet_names:
                    df = pd.read_excel(xls, sheet_name=sheet, header=None)
                    mat = df.apply(pd.to_numeric, errors='coerce').fillna(0).values
                    data_map[sheet] = mat
            except: pass
            
        # 能谱 TXT 处理
        elif fname.endswith(".txt"):
            try:
                stringio = io.StringIO(f.getvalue().decode("utf-8", errors='ignore'))
                lines = stringio.readlines()
                is_data = False
                for line in lines:
                    if "SPECTRUM" in line: is_data = True; continue
                    if is_data and "," in line:
                        x, y = map(float, line.strip().split(","))
                        spectrum['x'].append(x); spectrum['y'].append(y)
            except: pass

    # --- 第二步：【关键修复】强制对齐尺寸 ---
    if data_map:
        # 1. 找到最大的宽和高 (通常以 SE 图或最大 Mapping 为准)
        max_h, max_w = 0, 0
        for mat in data_map.values():
            h, w = mat.shape
            if h * w > max_h * max_w:
                max_h, max_w = h, w
        
        # 2. 将所有矩阵 Resize 到最大尺寸
        aligned_map = {}
        for k, v in data_map.items():
            # cv2.resize 接收 (width, height)，而 shape 是 (height, width)
            if v.shape != (max_h, max_w):
                # 使用线性插值放大，保持平滑
                aligned_map[k] = cv2.resize(v, (max_w, max_h), interpolation=cv2.INTER_LINEAR)
            else:
                aligned_map[k] = v
        
        return aligned_map, spectrum

    return data_map, spectrum
def auto_identify_peaks(x, y):
    x, y = np.array(x), np.array(y)
    if len(y) == 0: return []
    peaks, _ = find_peaks(y, height=np.max(y)*0.05, distance=20)
    results = []
    for p in peaks:
        energy = x[p]
        best_el = None
        min_diff = 0.06
        for el, e_val in ELEMENT_ENERGIES.items():
            if abs(energy - e_val) < min_diff:
                min_diff = abs(energy - e_val); best_el = el
        if best_el:
            results.append({'x': energy, 'y': y[p], 'text': best_el})
    return results

# ================= 3. UI 布局 =================

st.title("🔬 微粒交互式综合分析平台")
st.markdown("上传数据后，可**手动在图像上划定区域**查看局部元素占比及粒径。")

with st.sidebar:
    st.header("📂 数据上传")
    uploaded_files = st.file_uploader("支持 CSV/Excel/TXT", accept_multiple_files=True)
    st.markdown("---")
    st.header("🎨 交互设置")
    draw_mode = st.radio("圈选工具", ("circle", "rect", "transform"), format_func=lambda x: "圆形" if x=="circle" else "矩形" if x=="rect" else "调整位置")
    bg_threshold = st.slider("背景显示阈值", 0, 10, 2)

if uploaded_files:
    data_map, spec = load_uploaded_data(uploaded_files)
    
    if data_map:
        col_canvas, col_result = st.columns([1.2, 1])
        
        # 获取基础尺寸
        first_mat = next(iter(data_map.values()))
        h, w = first_mat.shape
        
        with col_canvas:
            st.subheader("🎯 区域圈选分析")
            # 合成一个底图供人眼识别
            base_rgb = np.zeros((h, w, 3))
            for i, el in enumerate(['Si', 'O', 'C']):
                if el in data_map:
                    m = data_map[el].copy()
                    m[m < bg_threshold] = 0
                    if m.max() > 0: base_rgb[:,:,i] = m / m.max()
            
            # 转换为 8bit 供画布显示
            bg_img = (np.clip(base_rgb * 1.5, 0, 1) * 255).astype(np.uint8)
            bg_img_resized = cv2.resize(bg_img, (w*4, h*4)) # 放大4倍方便手机/电脑精细操作

            # --- 交互式画布组件 ---
            canvas_result = st_canvas(
                fill_color="rgba(255, 165, 0, 0.3)",
                stroke_width=2,
                stroke_color="#fff",
                background_image=None,
                background_color="#000",
                update_streamlit=True,
                height=h * 4,
                width=w * 4,
                drawing_mode=draw_mode,
                key="particle_canvas",
            )
            st.caption("提示：使用左侧工具在图上画圈。橙色区域即为当前的分析范围。")

        with col_result:
            if canvas_result.json_data and canvas_result.json_data["objects"]:
                st.subheader("📊 局部选区报告")
                # 取最后一个绘制的对象
                obj = canvas_result.json_data["objects"][-1]
                
                # 生成 Mask (坐标需从放大4倍还原)
                mask = np.zeros((h, w), dtype=np.uint8)
                if obj["type"] == "circle":
                    cx = int((obj["left"] + obj["radius"]) / 4)
                    cy = int((obj["top"] + obj["radius"]) / 4)
                    r = int(obj["radius"] / 4)
                    cv2.circle(mask, (cx, cy), r, 1, -1)
                elif obj["type"] == "rect":
                    x1, y1 = int(obj["left"]/4), int(obj["top"]/4)
                    x2, y2 = x1 + int(obj["width"]/4), y1 + int(obj["height"]/4)
                    cv2.rectangle(mask, (x1, y1), (x2, y2), 1, -1)
                
                # 计算选区统计
                roi_vals = {}
                for el, mat in data_map.items():
                    if el != "SE": roi_vals[el] = np.sum(mat * mask)
                
                total = sum(roi_vals.values()) + 1e-9
                pie_data = pd.DataFrame({"Element": list(roi_vals.keys()), "Value": list(roi_stats.values())})
                
                # 显示饼图
                fig_pie = go.Figure(data=[go.Pie(labels=list(roi_vals.keys()), values=list(roi_vals.values()), hole=.4)])
                fig_pie.update_layout(margin=dict(t=30, b=0, l=0, r=0), height=300)
                st.plotly_chart(fig_pie, use_container_width=True)
                
                # 物理尺寸估算
                px_area = np.sum(mask)
                dia = np.sqrt(4 * px_area / np.pi) * 0.05 # 假设每像素 0.05 微米
                st.metric("选区等效直径", f"{dia:.2f} μm")
            else:
                st.info("👈 请在左侧图像上划定区域。")

        # --- 能谱自动标峰 ---
        st.markdown("---")
        st.subheader("📈 能谱自动标峰 (Auto-Peak Identification)")
        if spec['x']:
            peaks = auto_identify_peaks(spec['x'], spec['y'])
            
            fig_spec = go.Figure()
            fig_spec.add_trace(go.Scatter(x=spec['x'], y=spec['y'], fill='tozeroy', line=dict(color='#2c3e50', width=1.5), name="Counts"))
            
            # 在图表上添加标注
            for p in peaks:
                fig_spec.add_annotation(x=p['x'], y=p['y'], text=f"<b>{p['text']}</b>", showarrow=True, arrowhead=2, arrowcolor="#e74c3c", ax=0, ay=-30)
            
            fig_spec.update_layout(xaxis_title="Energy (keV)", yaxis_title="Counts", height=400, hovermode="x unified")
            st.plotly_chart(fig_spec, use_container_width=True)
            
            detected = ", ".join(sorted(list(set([p['text'] for p in peaks]))))
            st.success(f"🔍 自动识别到的元素特征峰: {detected}")
        else:
            st.caption("未上传能谱文件 (.txt)")

        # --- 底部详情 ---
        with st.expander("查看所有原始分图"):
            els = sorted(list(data_map.keys()))
            c = st.columns(6)
            for i, el in enumerate(els):
                with c[i%6]:
                    st.image(data_map[el]/ (data_map[el].max()+1e-6), caption=el)

else:
    st.info("👋 欢迎！请上传包含 CSV/Excel Mapping 数据和 TXT 能谱的微粒文件夹。")
