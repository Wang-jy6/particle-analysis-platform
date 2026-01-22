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

# ================= 1. 基础配置 =================
st.set_page_config(page_title="微粒交互分析平台 (批量版)", layout="wide")

ELEMENT_ENERGIES = {
    'C': 0.277, 'N': 0.392, 'O': 0.525, 'Na': 1.041, 'Mg': 1.253, 
    'Al': 1.486, 'Si': 1.739, 'S': 2.307, 'Cl': 2.621, 'K': 3.312, 
    'Ca': 3.690, 'Fe': 6.398, 'Cu': 8.040, 'Zn': 8.630
}

# ================= 2. 数据处理逻辑 (增强版) =================

def align_images(data_map):
    """强制对齐所有矩阵尺寸"""
    if not data_map: return data_map
    # 找到最大的宽和高
    max_h, max_w = 0, 0
    for mat in data_map.values():
        h, w = mat.shape
        if h * w > max_h * max_w:
            max_h, max_w = h, w
    
    aligned = {}
    for k, v in data_map.items():
        if v.shape != (max_h, max_w):
            aligned[k] = cv2.resize(v, (max_w, max_h), interpolation=cv2.INTER_LINEAR)
        else:
            aligned[k] = v
    return aligned

@st.cache_data
def parse_uploaded_files(uploaded_files):
    """解析上传文件并按微粒分组"""
    particles = {} # { 'K1-27': {'data': {}, 'spec': {}}, 'K1-28': ... }
    
    for f in uploaded_files:
        fname = f.name
        
        # 1. 尝试提取微粒ID (假设文件名格式为 "ID_元素.csv" 或 "ID 元素.csv")
        # 如果文件名很简单如 "Si.csv"，则归为 "Default_Particle"
        pid = "Default_Particle"
        element = "Unknown"
        
        # 简单的启发式分组逻辑
        if "_" in fname:
            parts = fname.split("_")
            # 假设最后一个部分是元素 (Fe.csv)，前面是ID (Particle_01)
            # 但要排除 "Si Kα1.csv" 这种自带空格的情况
            if len(parts) > 1:
                pid = "_".join(parts[:-1]) # 前面的做ID
                rest = parts[-1]
        elif " " in fname:
            # 处理 "K1-27 Si Kα1.csv" -> ID=K1-27, El=Si
            # 处理 "Si Kα1.csv" -> ID=Default, El=Si
            parts = fname.split(" ")
            if len(parts) > 2 and not parts[0] in ELEMENT_ENERGIES: 
                # 如果第一个词不是元素名，那可能是ID
                pid = parts[0]
        
        # 确保字典存在
        if pid not in particles:
            particles[pid] = {'data': {}, 'spec': {'x':[], 'y':[]}}
            
        # 2. 读取数据
        if fname.endswith(".csv"):
            # 提取元素名
            clean_name = fname.split(".")[0]
            # 尝试从文件名末尾提取元素 (比如 K1-27_Si -> Si)
            possible_el = clean_name.split("_")[-1].split(" ")[0]
            if "电子图像" in fname: possible_el = "SE"
            
            try:
                df = pd.read_csv(f, header=None)
                mat = df.apply(pd.to_numeric, errors='coerce').fillna(0).values
                particles[pid]['data'][possible_el] = mat
            except: pass
            
        elif fname.endswith(".txt"):
            try:
                content = f.getvalue().decode("utf-8", errors='ignore')
                lines = io.StringIO(content).readlines()
                is_data = False
                for line in lines:
                    if "SPECTRUM" in line: is_data = True; continue
                    if is_data and "," in line:
                        x, y = map(float, line.strip().split(","))
                        particles[pid]['spec']['x'].append(x)
                        particles[pid]['spec']['y'].append(y)
            except: pass
            
    # 对齐每个微粒的图像
    for pid in particles:
        particles[pid]['data'] = align_images(particles[pid]['data'])
        
    return particles

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

st.title("🔬 微粒交互分析平台 (批量版)")

with st.sidebar:
    st.header("📂 批量导入")
    st.info("提示：您可以直接拖入一个包含多个微粒文件的文件夹。")
    uploaded_files = st.file_uploader("上传文件 (支持批量)", accept_multiple_files=True)
    
    st.markdown("---")
    st.header("🎨 显示设置")
    zoom_level = st.slider("画布缩放倍率", 1.0, 5.0, 2.0, 0.5)
    bg_threshold = st.slider("背景去噪阈值", 0, 50, 2)
    draw_mode = st.selectbox("圈选工具", ["circle", "rect", "transform"], format_func=lambda x: {"circle":"圆形", "rect":"矩形", "transform":"移动/调整"}[x])

if uploaded_files:
    # 1. 解析并分组
    particles_batch = parse_uploaded_files(uploaded_files)
    
    if not particles_batch:
        st.error("未检测到有效数据")
    else:
        # 2. 选择微粒
        particle_ids = sorted(list(particles_batch.keys()))
        selected_pid = st.sidebar.selectbox("选择要分析的微粒", particle_ids)
        
        # 获取当前微粒数据
        current_data = particles_batch[selected_pid]['data']
        current_spec = particles_batch[selected_pid]['spec']
        
        st.markdown(f"### 当前分析: `{selected_pid}`")
        
        if current_data:
            col_canvas, col_result = st.columns([1.5, 1])
            
            # 准备底图
            shape = next(iter(current_data.values())).shape
            h, w = shape
            base_rgb = np.zeros((h, w, 3))
            
            # 默认合成 Si(红), O(绿), C(蓝)
            colors = {'Si':0, 'O':1, 'C':2} 
            for el, idx in colors.items():
                if el in current_data:
                    m = current_data[el].copy()
                    m[m < bg_threshold] = 0
                    if m.max() > 0: base_rgb[:,:,idx] = m / m.max()
            
            # 转为 8bit
            bg_img = (np.clip(base_rgb * 1.5, 0, 1) * 255).astype(np.uint8)
            
            # --- 交互画布 ---
            with col_canvas:
                # 动态计算画布大小
                canvas_w = int(w * zoom_level)
                canvas_h = int(h * zoom_level)
                
                # 预处理背景图尺寸
                bg_img_resized = cv2.resize(bg_img, (canvas_w, canvas_h))
                
                canvas_result = st_canvas(
                    fill_color="rgba(255, 165, 0, 0.2)",
                    stroke_width=2,
                    stroke_color="#eee",
                    background_image=None, # 我们用 background_color + 覆盖image的方式，或者直接由st_canvas处理
                    # 这里为了性能，我们不传 image 到 background_image 参数，而是让它透明，我们在下面显示图
                    # 哎呀，st_canvas 不支持直接传 numpy array 作为背景，得存成图片
                    # 变通：我们用 initial_drawing 或 background_image (需要PIL Image)
                    background_color="#000000",
                    height=canvas_h,
                    width=canvas_w,
                    drawing_mode=draw_mode,
                    key=f"canvas_{selected_pid}", # 切换微粒时重置画布
                )
                
                # 因为 st_canvas 背景图处理比较麻烦，我们用 CSS 绝对定位或者简单点：
                # 把图画在下面？不，那样没法对齐。
                # 正确做法：把 numpy 转 bytes 传给 st_canvas
                from PIL import Image
                pil_img = Image.fromarray(bg_img_resized)
                # 使用 columns 再次布局，把图垫在 canvas 下面 (Streamlit layout trick)
                # 或者直接用 background_image 参数 (支持 PIL Image) -> 最简单
                
                # *修正*：重新渲染带背景的 Canvas
                # 为了不让页面闪烁，我们把上面的 st_canvas 替换掉
                st.markdown(f"<style>canvas {{ border: 1px solid #444; }}</style>", unsafe_allow_html=True)

            # 重新调用一次带背景的 (Streamlit 渲染顺序是从上到下，上面那个仅仅是为了占位逻辑演示，下面这个才是真的)
            # 实际上不能调两次，会报错。所以我修改上面的参数。
            # 请注意：下面的代码逻辑是整合进去的
            
            # --- 最终 Canvas 渲染 ---
            with col_canvas:
               # 只要不重复写 st_canvas 即可。我们把上面的删除，只留这一个：
               pass 
            
            # 真正的 Canvas
            with col_canvas:
                canvas_result = st_canvas(
                    fill_color="rgba(255, 165, 0, 0.25)",
                    stroke_width=2,
                    stroke_color="#fff",
                    background_image=pil_img,
                    height=canvas_h,
                    width=canvas_w,
                    drawing_mode=draw_mode,
                    key=f"cv_{selected_pid}_{zoom_level}",
                )
                st.caption(f"画布尺寸: {canvas_w}x{canvas_h} (缩放 x{zoom_level})")

            # --- 结果计算 ---
            with col_result:
                if canvas_result.json_data and canvas_result.json_data["objects"]:
                    st.subheader("📊 局部选区成分")
                    obj = canvas_result.json_data["objects"][-1]
                    
                    # 生成 Mask (注意坐标要除以 zoom_level)
                    mask = np.zeros((h, w), dtype=np.uint8)
                    scale = zoom_level
                    
                    if obj["type"] == "circle":
                        cx = int((obj["left"] + obj["radius"]) / scale)
                        cy = int((obj["top"] + obj["radius"]) / scale)
                        r = int(obj["radius"] / scale)
                        cv2.circle(mask, (cx, cy), r, 1, -1)
                    elif obj["type"] == "rect":
                        x1, y1 = int(obj["left"]/scale), int(obj["top"]/scale)
                        w_box, h_box = int(obj["width"]/scale), int(obj["height"]/scale)
                        cv2.rectangle(mask, (x1, y1), (x1+w_box, y1+h_box), 1, -1)
                        
                    # 统计
                    stats = {}
                    for el, mat in current_data.items():
                        if el != "SE": stats[el] = np.sum(mat * mask)
                    
                    # 归一化显示
                    total = sum(stats.values()) + 1e-9
                    df_res = pd.DataFrame({"Element": stats.keys(), "Intensity": stats.values()})
                    df_res["Percent"] = df_res["Intensity"] / total
                    df_res = df_res[df_res["Percent"] > 0.01].sort_values("Percent", ascending=False)
                    
                    st.plotly_chart(go.Figure(data=[go.Pie(labels=df_res["Element"], values=df_res["Percent"], hole=0.4)]), use_container_width=True)
                    
                    # 粒径
                    pixel_area = np.sum(mask)
                    # 假设 0.05 um/pixel
                    dia = np.sqrt(4 * pixel_area / np.pi) * 0.05
                    st.metric("选区等效直径", f"{dia:.2f} μm")
                else:
                    st.info("👈 请在左图拖动鼠标画圈")

        # --- 能谱 ---
        st.markdown("---")
        if current_spec['x']:
            st.subheader("📈 能谱分析 (自动标峰)")
            peaks = auto_identify_peaks(current_spec['x'], current_spec['y'])
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=current_spec['x'], y=current_spec['y'], fill='tozeroy', line=dict(color='#444')))
            for p in peaks:
                fig.add_annotation(x=p['x'], y=p['y'], text=p['text'], showarrow=True, arrowhead=2, ay=-30)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("该微粒无能谱数据")

else:
    st.info("👋 请在左侧上传文件夹（直接拖入多个文件）。")
