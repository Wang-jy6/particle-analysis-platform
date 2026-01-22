import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from streamlit_drawable_canvas import st_canvas
import cv2
import io
from scipy.signal import find_peaks

# ================= 1. 配置与常量 =================
st.set_page_config(page_title="微粒交互分析平台", layout="wide")

ELEMENT_ENERGIES = {
    'C': 0.277, 'N': 0.392, 'O': 0.525, 'Na': 1.041, 'Mg': 1.253, 
    'Al': 1.486, 'Si': 1.739, 'S': 2.307, 'Cl': 2.621, 'K': 3.312, 
    'Ca': 3.690, 'Fe': 6.398, 'Cu': 8.040, 'Zn': 8.630
}

# ================= 2. 数据处理函数 =================

def process_files(uploaded_files):
    data_map = {}
    spectrum = {'x': [], 'y': []}
    for f in uploaded_files:
        if f.name.endswith(".csv"):
            el = f.name.split(" ")[0].split(".")[0].split("_")[-1]
            df = pd.read_csv(f, header=None)
            data_map[el] = df.apply(pd.to_numeric, errors='coerce').fillna(0).values
        elif f.name.endswith(".txt"):
            lines = io.StringIO(f.getvalue().decode("utf-8", errors='ignore')).readlines()
            is_data = False
            for line in lines:
                if "SPECTRUM" in line: is_data = True; continue
                if is_data and "," in line:
                    x, y = map(float, line.strip().split(","))
                    spectrum['x'].append(x); spectrum['y'].append(y)
    return data_map, spectrum

def find_labels(x, y):
    x, y = np.array(x), np.array(y)
    peaks, _ = find_peaks(y, height=np.max(y)*0.05, distance=15)
    labels = []
    for p in peaks:
        energy = x[p]
        best_el = None
        min_diff = 0.06
        for el, e_val in ELEMENT_ENERGIES.items():
            if abs(energy - e_val) < min_diff:
                min_diff = abs(energy - e_val); best_el = el
        if best_el:
            labels.append({'x': energy, 'y': y[p], 'text': best_el})
    return labels

# ================= 3. 主界面 =================

st.title("🔬 微粒交互式分析系统")
st.markdown("上传数据后，可**手动在图像上划定区域**查看局部元素占比。")

files = st.file_uploader("上传 CSV 和 TXT 文件", accept_multiple_files=True)

if files:
    data_map, spec = process_files(files)
    
    if data_map:
        col_img, col_info = st.columns([1, 1])
        
        with col_img:
            st.subheader("🖱️ 手动圈选分析区")
            st.caption("请选择左侧工具栏的【圆形】或【矩形】工具在微粒上画图")
            
            # 准备底图（用Si和O合成）
            shape = next(iter(data_map.values())).shape
            base_img = np.zeros((shape[0], shape[1], 3))
            for i, el in enumerate(['Si', 'O', 'C']):
                if el in data_map:
                    m = data_map[el]
                    base_img[:,:,i] = m / (m.max() + 1e-6)
            base_img = (np.clip(base_img * 1.5, 0, 1) * 255).astype(np.uint8)

            # --- 交互式画布 ---
            canvas_result = st_canvas(
                fill_color="rgba(255, 165, 0, 0.3)",  # 填充半透明橙色
                stroke_width=2,
                stroke_color="#fff",
                background_image=None,
                background_color="#000",
                update_streamlit=True,
                height=shape[0] * 2, # 放大显示方便操作
                width=shape[1] * 2,
                drawing_mode="rect" if st.checkbox("切换为矩形模式", False) else "circle",
                key="canvas",
            )

        with col_info:
            if canvas_result.json_data is not None:
                objects = canvas_result.json_data["objects"]
                if objects:
                    st.subheader("🎯 选区分析结果")
                    # 取最后一个画的图形
                    obj = objects[-1]
                    
                    # 计算 Mask (由于画布放大了2倍，坐标要除以2)
                    mask = np.zeros(shape, dtype=np.uint8)
                    if obj["type"] == "circle":
                        center = (int(obj["left"] + obj["radius"]), int(obj["top"] + obj["radius"]))
                        cv2.circle(mask, (center[0]//2, center[1]//2), int(obj["radius"])//2, 1, -1)
                    elif obj["type"] == "rect":
                        pt1 = (int(obj["left"])//2, int(obj["top"])//2)
                        pt2 = (pt1[0] + int(obj["width"])//2, pt1[1] + int(obj["height"])//2)
                        cv2.rectangle(mask, pt1, pt2, 1, -1)
                    
                    # 计算选区内元素强度
                    roi_stats = {}
                    for el, mat in data_map.items():
                        roi_stats[el] = np.sum(mat * mask)
                    
                    # 饼图显示
                    total = sum(roi_stats.values()) + 1e-6
                    pie_df = pd.DataFrame({
                        "元素": list(roi_stats.keys()), 
                        "占比": [v/total for v in roi_stats.values()]
                    }).query("占比 > 0.01")
                    
                    st.plotly_chart(go.Figure(data=[go.Pie(labels=pie_df["元素"], values=pie_df["占比"], hole=.4)]), use_container_width=True)
                    
                    # 估算选区直径
                    pixel_count = np.sum(mask)
                    est_dia = np.sqrt(4 * pixel_count / np.pi) * 0.05 # 假设 0.05um/px
                    st.metric("选区估算直径", f"{est_dia:.2f} μm")
                else:
                    st.info("请在左侧图像上画圈以查看局部成分。")

        # --- 能谱部分 ---
        st.markdown("---")
        st.subheader("📈 能谱自动标峰")
        if spec['x']:
            labels = find_labels(spec['x'], spec['y'])
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=spec['x'], y=spec['y'], fill='tozeroy', line=dict(color='#333')))
            
            # 添加标注
            for lbl in labels:
                fig.add_annotation(x=lbl['x'], y=lbl['y'], text=lbl['text'], showarrow=True, arrowhead=1)
                
            fig.update_layout(xaxis_title="Energy (keV)", yaxis_title="Counts", height=400)
            st.plotly_chart(fig, use_container_width=True)
            
        # 4. 单元素图集 (点击可看全图)
        with st.expander("查看所有元素 Mapping 分图"):
            cols = st.columns(6)
            for i, (el, mat) in enumerate(data_map.items()):
                with cols[i%6]:
                    st.image(mat/mat.max(), caption=el, use_container_width=True)

else:
    st.info("👋 请上传微粒文件开始交互分析。")
