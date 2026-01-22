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

# ================= 1. 基础配置 =================
st.set_page_config(page_title="微粒全能分析平台", layout="wide", page_icon="🔬")

# 元素特征能量表
ELEMENT_ENERGIES = {
    'C': 0.277, 'N': 0.392, 'O': 0.525, 'Na': 1.041, 'Mg': 1.253, 
    'Al': 1.486, 'Si': 1.739, 'S': 2.307, 'Cl': 2.621, 'K': 3.312, 
    'Ca': 3.690, 'Fe': 6.398, 'Cu': 8.040, 'Zn': 8.630, 'Au': 2.120
}

# ================= 2. 核心处理函数 =================

def align_images(data_map):
    """强制对齐图像尺寸，防止报错"""
    if not data_map: return data_map
    max_h, max_w = 0, 0
    # 1. 找最大尺寸
    for mat in data_map.values():
        h, w = mat.shape
        if h * w > max_h * max_w: max_h, max_w = h, w
    
    # 2. 统一缩放
    aligned = {}
    for k, v in data_map.items():
        if v.shape != (max_h, max_w):
            aligned[k] = cv2.resize(v, (max_w, max_h), interpolation=cv2.INTER_LINEAR)
        else:
            aligned[k] = v
    return aligned

def parse_filename(fname):
    """从文件名提取元素名"""
    # 移除扩展名
    name = fname.rsplit('.', 1)[0]
    # 处理 "Si Kα1" 或 "01_Si"
    if "电子图像" in name: return "SE"
    parts = name.replace("_", " ").split(" ")
    # 倒序查找，找到第一个在元素表里的词，或者直接用第一个词
    for p in reversed(parts):
        if p in ELEMENT_ENERGIES: return p
    return parts[0] # 兜底

def read_file_content(file_obj, filename):
    """读取单个文件内容返回矩阵或能谱"""
    res_type = None # 'map' or 'spec'
    content = None
    
    if filename.lower().endswith(('.csv')):
        df = pd.read_csv(file_obj, header=None)
        content = df.apply(pd.to_numeric, errors='coerce').fillna(0).values
        res_type = 'map'
        
    elif filename.lower().endswith(('.xls', '.xlsx')):
        # Excel 特殊处理，返回字典
        xls = pd.ExcelFile(file_obj)
        content = {}
        for sheet in xls.sheet_names:
            df = pd.read_excel(xls, sheet_name=sheet, header=None)
            content[sheet] = df.apply(pd.to_numeric, errors='coerce').fillna(0).values
        res_type = 'excel_map'
        
    elif filename.lower().endswith('.txt'):
        # 能谱
        try:
            # 如果是 bytes (ZipExtFile) 需要 decode，如果是 StringIO (UploadedFile) 不需要
            if isinstance(file_obj, io.StringIO): 
                text = file_obj.getvalue()
            elif hasattr(file_obj, 'read'):
                text = file_obj.read().decode('utf-8', errors='ignore')
            else:
                text = str(file_obj)
                
            lines = text.splitlines()
            x, y = [], []
            is_data = False
            for line in lines:
                if "SPECTRUM" in line: is_data = True; continue
                if is_data and "," in line:
                    parts = line.strip().split(",")
                    x.append(float(parts[0]))
                    y.append(float(parts[1]))
            content = {'x': x, 'y': y}
            res_type = 'spec'
        except: pass
        
    return res_type, content

# --- 模式 A: 单微粒解析器 ---
def parse_single_mode(uploaded_files):
    data_map = {}
    spec = {'x': [], 'y': []}
    
    for f in uploaded_files:
        res_type, content = read_file_content(f, f.name)
        
        if res_type == 'map':
            el = parse_filename(f.name)
            data_map[el] = content
        elif res_type == 'excel_map':
            for sheet_name, mat in content.items():
                data_map[sheet_name] = mat
        elif res_type == 'spec':
            spec = content
            
    return {'Single_Particle': {'data': align_images(data_map), 'spec': spec}}

# --- 模式 B: ZIP 批量解析器 ---
def parse_batch_mode(zip_file_obj):
    particles = {}
    temp_dir = "temp_zip_extract"
    if os.path.exists(temp_dir): shutil.rmtree(temp_dir)
    os.makedirs(temp_dir)
    
    try:
        with zipfile.ZipFile(zip_file_obj, "r") as z:
            z.extractall(temp_dir)
    except: return {}

    for root, dirs, files in os.walk(temp_dir):
        valid_files = [f for f in files if f.lower().endswith(('.csv', '.xls', '.xlsx', '.txt'))]
        if valid_files:
            pid = os.path.basename(root)
            if pid == temp_dir: pid = "Root"
            # 避免重名
            if pid in particles: pid = f"{pid}_{len(particles)}"
            
            particles[pid] = {'data': {}, 'spec': {'x':[], 'y':[]}}
            
            for f in valid_files:
                f_path = os.path.join(root, f)
                with open(f_path, 'rb') as fo: # 二进制读取供 pandas 解析
                    # 针对 pandas 读取本地文件，直接传路径即可
                    if f.lower().endswith('.txt'):
                        res_type, content = read_file_content(fo, f)
                    else:
                        # Pandas read functions work better with paths for local files
                        res_type, content = None, None
                        if f.lower().endswith('.csv'):
                            df = pd.read_csv(f_path, header=None)
                            content = df.apply(pd.to_numeric, errors='coerce').fillna(0).values
                            res_type = 'map'
                        elif f.lower().endswith(('.xls', '.xlsx')):
                            # 复用逻辑
                            with open(f_path, 'rb') as excel_fo:
                                res_type, content = read_file_content(excel_fo, f)

                if res_type == 'map':
                    el = parse_filename(f)
                    particles[pid]['data'][el] = content
                elif res_type == 'excel_map':
                    for sheet, mat in content.items():
                        particles[pid]['data'][sheet] = mat
                elif res_type == 'spec':
                    particles[pid]['spec'] = content
            
            particles[pid]['data'] = align_images(particles[pid]['data'])
            
    # shutil.rmtree(temp_dir) # 调试时可注释
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
        if best_el: results.append({'x': energy, 'y': y[p], 'text': best_el})
    return results

# ================= 3. UI 布局 =================

st.title("🔬 微粒全能分析平台")

# --- 侧边栏 ---
with st.sidebar:
    st.header("📂 数据导入")
    st.info("支持两种模式：\n1. **单微粒**：直接拖入多个 CSV/TXT 文件。\n2. **批量**：拖入一个 ZIP 压缩包（包含多个文件夹）。")
    
    uploaded_files = st.file_uploader("拖拽文件到这里", accept_multiple_files=True)
    
    st.markdown("---")
    st.header("🎨 交互设置")
    zoom = st.slider("画布缩放", 0.5, 3.0, 1.5, 0.1)
    threshold = st.slider("背景降噪", 0, 50, 2)
    tool = st.selectbox("圈选工具", ["circle", "rect"], format_func=lambda x: "圆形" if x=="circle" else "矩形")

# --- 主逻辑 ---
particles_db = {}

if uploaded_files:
    # 智能判断模式
    is_zip = any(f.name.endswith('.zip') for f in uploaded_files)
    
    if is_zip:
        st.success("检测到 ZIP 压缩包，已切换至 **批量分析模式**")
        # 找到那个 zip 文件
        zip_file = next(f for f in uploaded_files if f.name.endswith('.zip'))
        particles_db = parse_batch_mode(zip_file)
    else:
        st.success("检测到散乱文件，已切换至 **单微粒模式**")
        particles_db = parse_single_mode(uploaded_files)

    if not particles_db:
        st.error("无法解析数据，请检查文件格式。")
    else:
        # --- 选择微粒 ---
        p_ids = sorted(list(particles_db.keys()))
        
        # 如果是批量模式，在侧边栏显示切换器
        if len(p_ids) > 1:
            st.sidebar.markdown("---")
            st.sidebar.subheader(f"微粒列表 ({len(p_ids)})")
            selected_id = st.sidebar.selectbox("选择微粒", p_ids)
        else:
            selected_id = p_ids[0]
            
        current_data = particles_db[selected_id]['data']
        current_spec = particles_db[selected_id]['spec']
        
        st.markdown(f"## 🧪 分析对象: `{selected_id}`")
        
        # --- 渲染分析界面 ---
        if not current_data:
            st.warning("该微粒没有 Mapping 数据")
        else:
            c1, c2 = st.columns([1.5, 1])
            
            # 1. 准备底图
            shape = next(iter(current_data.values())).shape
            h, w = shape
            base_rgb = np.zeros((h, w, 3))
            
            # 合成逻辑 Si(R) O(G) C(B)
            legend = []
            colors = {'Si':0, 'O':1, 'C':2}
            for el, idx in colors.items():
                if el in current_data:
                    m = current_data[el].copy()
                    m[m < threshold] = 0
                    if m.max() > 0: base_rgb[:,:,idx] = m / m.max()
                    legend.append(f"{el}")
            
            bg_uint8 = (np.clip(base_rgb * 1.5, 0, 1) * 255).astype(np.uint8)
            
            # 2. 画布区域
            with c1:
                cw, ch = int(w*zoom), int(h*zoom)
                bg_pil = Image.fromarray(bg_uint8).resize((cw, ch))
                
                st.caption(f"合成预览 ({', '.join(legend)}) - 尺寸 {cw}x{ch}")
                canvas = st_canvas(
                    fill_color="rgba(255, 165, 0, 0.2)",
                    stroke_width=2,
                    stroke_color="#fff",
                    background_image=bg_pil,
                    height=ch, width=cw,
                    drawing_mode=tool,
                    key=f"cv_{selected_id}_{zoom}" # ID变了画布自动重置
                )
                
            # 3. 统计结果
            with c2:
                if canvas.json_data and canvas.json_data["objects"]:
                    st.subheader("📊 局部选区成分")
                    obj = canvas.json_data["objects"][-1]
                    
                    # 生成Mask
                    mask = np.zeros((h, w), dtype=np.uint8)
                    scale = zoom
                    if obj["type"] == "circle":
                        cx, cy = int((obj["left"]+obj["radius"])/scale), int((obj["top"]+obj["radius"])/scale)
                        r = int(obj["radius"]/scale)
                        cv2.circle(mask, (cx, cy), r, 1, -1)
                    elif obj["type"] == "rect":
                        x, y = int(obj["left"]/scale), int(obj["top"]/scale)
                        wb, hb = int(obj["width"]/scale), int(obj["height"]/scale)
                        cv2.rectangle(mask, (x, y), (x+wb, y+hb), 1, -1)
                        
                    # 统计
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
                    st.info("👈 请在左图进行圈选分析")
                    
            # 4. 能谱
            st.markdown("---")
            if current_spec['x']:
                st.subheader("📈 能谱分析")
                peaks = auto_identify_peaks(current_spec['x'], current_spec['y'])
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=current_spec['x'], y=current_spec['y'], fill='tozeroy', line=dict(color='#333')))
                for p in peaks:
                    fig.add_annotation(x=p['x'], y=p['y'], text=p['text'], showarrow=True, arrowhead=2, ay=-30)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.caption("无能谱数据")
                
            # 5. 图集
            with st.expander("查看全部分图"):
                cols = st.columns(6)
                for i, (el, mat) in enumerate(current_data.items()):
                    with cols[i%6]:
                        st.image(mat/(mat.max()+1e-6), caption=el)

else:
    st.info("👋 等待数据上传...")
