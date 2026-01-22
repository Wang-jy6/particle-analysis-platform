import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from streamlit_drawable_canvas import st_canvas
import cv2
import io
import os
import zipfile
import shutil
from PIL import Image
from scipy.signal import find_peaks

# ================= 1. 全局配置 =================
st.set_page_config(page_title="微粒全能分析平台", layout="wide", page_icon="🔬")

# 元素特征能量表
ELEMENT_ENERGIES = {
    'C': 0.277, 'N': 0.392, 'O': 0.525, 'F': 0.677, 'Na': 1.041, 'Mg': 1.253, 
    'Al': 1.486, 'Si': 1.739, 'P': 2.013, 'S': 2.307, 'Cl': 2.621, 'K': 3.312, 
    'Ca': 3.690, 'Ti': 4.508, 'Cr': 5.411, 'Mn': 5.894, 'Fe': 6.398, 'Ni': 7.471, 
    'Cu': 8.040, 'Zn': 8.630, 'Au': 2.120, 'Ag': 2.980
}

# ================= 2. 核心处理逻辑 =================

def align_images(data_map):
    """强制对齐所有图像尺寸"""
    if not data_map: return data_map
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

def parse_element_name(filename):
    name = filename.rsplit('.', 1)[0]
    if "电子图像" in name or "SE" in name.upper(): return "SE"
    parts = name.replace("_", " ").split(" ")
    for p in reversed(parts):
        clean_p = ''.join(filter(str.isalpha, p)) 
        if clean_p in ELEMENT_ENERGIES: return clean_p
    return parts[0]

def read_file_content(file_obj, filename):
    res_type = None
    content = None
    fname_lower = filename.lower()
    try:
        if fname_lower.endswith('.csv'):
            df = pd.read_csv(file_obj, header=None)
            content = df.apply(pd.to_numeric, errors='coerce').fillna(0).values
            res_type = 'map'
        elif fname_lower.endswith(('.xls', '.xlsx')):
            xls = pd.ExcelFile(file_obj)
            content = {}
            for sheet in xls.sheet_names:
                df = pd.read_excel(xls, sheet_name=sheet, header=None)
                content[sheet] = df.apply(pd.to_numeric, errors='coerce').fillna(0).values
            res_type = 'excel_map'
        elif fname_lower.endswith('.txt'):
            if isinstance(file_obj, io.StringIO): text = file_obj.getvalue()
            elif hasattr(file_obj, 'read'): text = file_obj.read().decode('utf-8', errors='ignore')
            else: text = str(file_obj)
            lines = text.splitlines()
            x, y = [], []
            is_data = False
            for line in lines:
                if "SPECTRUM" in line: is_data = True; continue
                if is_data and "," in line:
                    parts = line.strip().split(",")
                    if len(parts)>=2: x.append(float(parts[0])); y.append(float(parts[1]))
            content = {'x': x, 'y': y}
            res_type = 'spec'
    except: pass
    return res_type, content

def auto_identify_peaks(x, y):
    x, y = np.array(x), np.array(y)
    if len(y) == 0: return []
    peaks, _ = find_peaks(y, height=np.max(y)*0.05, distance=15)
    results = []
    for p in peaks:
        energy = x[p]
        best_el = None
        min_diff = 0.06
        for el, e_val in ELEMENT_ENERGIES.items():
            if abs(energy - e_val) < min_diff:
                min_diff = abs(energy - e_val); best_el = el
        if best_el: results.append({'x': energy, 'y': y[p], 'text': best_el})
    return results

def parse_single_mode(uploaded_files):
    data_map = {}
    spec = {'x': [], 'y': []}
    for f in uploaded_files:
        f.seek(0)
        res_type, content = read_file_content(f, f.name)
        if res_type == 'map': data_map[parse_element_name(f.name)] = content
        elif res_type == 'excel_map': 
            for s, m in content.items(): data_map[s] = m
        elif res_type == 'spec': spec = content
    return {'Single_Particle': {'data': align_images(data_map), 'spec': spec}}

def parse_batch_mode(zip_file_obj):
    particles = {}
    temp_dir = "temp_zip_extract"
    if os.path.exists(temp_dir): shutil.rmtree(temp_dir)
    os.makedirs(temp_dir)
    try:
        with zipfile.ZipFile(zip_file_obj, "r") as z: z.extractall(temp_dir)
    except: return {}

    for root, dirs, files in os.walk(temp_dir):
        valid_files = [f for f in files if f.lower().endswith(('.csv', '.xls', '.xlsx', '.txt'))]
        if valid_files:
            pid = os.path.basename(root)
            if pid == temp_dir: pid = "Root"
            if pid in particles: pid = f"{pid}_{len(particles)}"
            particles[pid] = {'data': {}, 'spec': {'x':[], 'y':[]}}
            for f in valid_files:
                with open(os.path.join(root, f), 'rb') as fo:
                    res_type, content = read_file_content(fo, f)
                    if res_type == 'map': particles[pid]['data'][parse_element_name(f)] = content
                    elif res_type == 'excel_map':
                        for s, m in content.items(): particles[pid]['data'][s] = m
                    elif res_type == 'spec': particles[pid]['spec'] = content
            particles[pid]['data'] = align_images(particles[pid]['data'])
    return particles

# ================= 3. 用户界面 =================

st.title("🔬 微粒全能分析平台")

with st.sidebar:
    st.header("📂 数据导入")
    uploaded_files = st.file_uploader("上传文件 (支持 ZIP 批量 或 单文件)", accept_multiple_files=True)
    st.markdown("---")
    st.header("🎨 交互设置")
    zoom_level = st.slider("画布缩放", 0.5, 4.0, 1.5, 0.1)
    bg_threshold = st.slider("背景去噪", 0, 50, 2)
    draw_mode = st.selectbox("圈选工具", ["circle", "rect"], format_func=lambda x: "圆形" if x=="circle" else "矩形")

particles_db = {}
if uploaded_files:
    is_zip = any(f.name.lower().endswith('.zip') for f in uploaded_files)
    if is_zip:
        zip_file = next(f for f in uploaded_files if f.name.lower().endswith('.zip'))
        particles_db = parse_batch_mode(zip_file)
        if particles_db: st.success(f"📦 批量模式: {len(particles_db)} 个微粒")
    else:
        particles_db = parse_single_mode(uploaded_files)
        if particles_db: st.success("📄 单微粒模式")

    if particles_db:
        p_ids = sorted(list(particles_db.keys()))
        selected_pid = p_ids[0]
        if len(p_ids) > 1:
            st.sidebar.markdown("---")
            selected_pid = st.sidebar.selectbox("当前分析对象:", p_ids)
            
        current_data = particles_db[selected_pid]['data']
        current_spec = particles_db[selected_pid]['spec']
        
        st.markdown(f"### 🧪 分析: `{selected_pid}`")
        
        if not current_data:
            st.error("无 Mapping 数据")
        else:
            col_canvas, col_result = st.columns([1.5, 1])
            
            # --- 合成底图 ---
            shape = next(iter(current_data.values())).shape
            h, w = shape
            base_rgb = np.zeros((h, w, 3))
            colors = {'Si': 0, 'O': 1, 'C': 2}
            legend = []
            for el, idx in colors.items():
                if el in current_data:
                    mat = current_data[el].copy()
                    mat[mat < bg_threshold] = 0
                    if mat.max() > 0: base_rgb[:, :, idx] = mat / mat.max()
                    legend.append(el)
            bg_uint8 = (np.clip(base_rgb * 1.5, 0, 1) * 255).astype(np.uint8)
            
            with col_canvas:
                cw, ch = int(w * zoom_level), int(h * zoom_level)
                
                # 【回归正统】使用 PIL Image
                # 注意：必须 convert('RGB')，否则部分 PNG 格式可能触发问题
                bg_pil = Image.fromarray(bg_uint8).convert("RGB").resize((cw, ch))
                
                st.caption(f"合成预览 ({', '.join(legend)})")
                canvas_result = st_canvas(
                    fill_color="rgba(255, 165, 0, 0.2)",
                    stroke_width=2,
                    stroke_color="#fff",
                    background_image=bg_pil, 
                    update_streamlit=True,
                    height=ch,
                    width=cw,
                    drawing_mode=draw_mode,
                    key=f"cv_{selected_pid}_{zoom_level}",
                )

            with col_result:
                if canvas_result.json_data and canvas_result.json_data["objects"]:
                    st.subheader("📊 选区分析")
                    obj = canvas_result.json_data["objects"][-1]
                    mask = np.zeros((h, w), dtype=np.uint8)
                    scale = zoom_level
                    
                    if obj["type"] == "circle":
                        cx = int((obj["left"] + obj["radius"]) / scale)
                        cy = int((obj["top"] + obj["radius"]) / scale)
                        r = int(obj["radius"] / scale)
                        cv2.circle(mask, (cx, cy), r, 1, -1)
                    elif obj["type"] == "rect":
                        x, y = int(obj["left"]/scale), int(obj["top"]/scale)
                        wb, hb = int(obj["width"]/scale), int(obj["height"]/scale)
                        cv2.rectangle(mask, (x, y), (x+wb, y+hb), 1, -1)
                    
                    stats = {}
                    for el, mat in current_data.items():
                        if el != "SE": stats[el] = np.sum(mat * mask)
                    
                    tot = sum(stats.values()) + 1e-9
                    df = pd.DataFrame({"El": stats.keys(), "Val": stats.values()})
                    df["Pct"] = df["Val"] / tot
                    df = df[df["Pct"] > 0.01].sort_values("Pct", ascending=False)
                    
                    st.plotly_chart(go.Figure(data=[go.Pie(labels=df["El"], values=df["Pct"], hole=0.4)]), use_container_width=True)
                    dia = np.sqrt(4 * np.sum(mask) / np.pi) * 0.05
                    st.metric("等效直径", f"{dia:.2f} μm")
                else:
                    st.info("👈 请在左图画圈分析")

            st.markdown("---")
            if current_spec['x']:
                st.subheader("📈 能谱分析")
                peaks = auto_identify_peaks(current_spec['x'], current_spec['y'])
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=current_spec['x'], y=current_spec['y'], fill='tozeroy', line=dict(color='#333')))
                for p in peaks:
                    fig.add_annotation(x=p['x'], y=p['y'], text=p['text'], showarrow=True, arrowhead=2, ay=-30)
                st.plotly_chart(fig, use_container_width=True)
            
            with st.expander("查看分图"):
                cols = st.columns(6)
                for i, (el, mat) in enumerate(current_data.items()):
                    with cols[i%6]:
                        st.image(mat/(mat.max()+1e-6), caption=el)
else:
    st.info("等待上传...")
