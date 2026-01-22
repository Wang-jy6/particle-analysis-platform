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

# 常见元素特征能量表 (keV) - 用于自动标峰
ELEMENT_ENERGIES = {
    'C': 0.277, 'N': 0.392, 'O': 0.525, 'F': 0.677,
    'Na': 1.041, 'Mg': 1.253, 'Al': 1.486, 'Si': 1.739,
    'P': 2.013, 'S': 2.307, 'Cl': 2.621, 'K': 3.312, 
    'Ca': 3.690, 'Ti': 4.508, 'Cr': 5.411, 'Mn': 5.894,
    'Fe': 6.398, 'Ni': 7.471, 'Cu': 8.040, 'Zn': 8.630, 
    'Au': 2.120, 'Ag': 2.980, 'Ba': 4.465
}

# ================= 2. 核心处理逻辑 =================

def align_images(data_map):
    """
    强制对齐所有图像尺寸，解决 SE 图与 Mapping 图分辨率不一致导致的 ValueError
    """
    if not data_map: return data_map
    
    # 1. 寻找最大尺寸
    max_h, max_w = 0, 0
    for mat in data_map.values():
        h, w = mat.shape
        if h * w > max_h * max_w:
            max_h, max_w = h, w
    
    # 2. 统一缩放到最大尺寸
    aligned = {}
    for k, v in data_map.items():
        if v.shape != (max_h, max_w):
            # 注意 cv2.resize 接受 (width, height)
            aligned[k] = cv2.resize(v, (max_w, max_h), interpolation=cv2.INTER_LINEAR)
        else:
            aligned[k] = v
    return aligned

def parse_element_name(filename):
    """
    智能解析文件名中的元素名
    例如: "Si Kα1.csv" -> "Si", "01_Fe.xls" -> "Fe"
    """
    # 移除后缀
    name = filename.rsplit('.', 1)[0]
    
    # 特殊标记
    if "电子图像" in name or "SE" in name.upper(): 
        return "SE"
    
    # 分割字符串，寻找元素表中的关键字
    parts = name.replace("_", " ").split(" ")
    # 优先匹配末尾的词（通常元素名在最后）
    for p in reversed(parts):
        # 去除可能附带的数字或符号
        clean_p = ''.join(filter(str.isalpha, p)) 
        if clean_p in ELEMENT_ENERGIES:
            return clean_p
            
    # 如果没找到，返回第一个词作为默认
    return parts[0]

def read_file_content(file_obj, filename):
    """
    统一读取器：支持 CSV, Excel, TXT
    返回: type ('map'/'spec'/'excel_map'), content
    """
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
                mat = df.apply(pd.to_numeric, errors='coerce').fillna(0).values
                content[sheet] = mat
            res_type = 'excel_map'
            
        elif fname_lower.endswith('.txt'):
            # 处理编码和读取方式差异
            if isinstance(file_obj, io.StringIO):
                text = file_obj.getvalue()
            elif hasattr(file_obj, 'read'):
                # 二进制流需要解码
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
                    if len(parts) >= 2:
                        x.append(float(parts[0]))
                        y.append(float(parts[1]))
            content = {'x': x, 'y': y}
            res_type = 'spec'
            
    except Exception as e:
        # 这里的 print 只有在后台终端能看到，网页上不会报错中断
        print(f"Error reading {filename}: {e}")
        pass
        
    return res_type, content

def auto_identify_peaks(x, y):
    """能谱自动找峰"""
    x, y = np.array(x), np.array(y)
    if len(y) == 0: return []
    
    # 寻找波峰，高度至少为最大值的 5%
    peaks, _ = find_peaks(y, height=np.max(y)*0.05, distance=15)
    
    results = []
    found_elements = set()
    
    for p in peaks:
        energy = x[p]
        peak_height = y[p]
        
        best_el = None
        min_diff = 0.05 # 容差 50eV
        
        for el, e_val in ELEMENT_ENERGIES.items():
            if abs(energy - e_val) < min_diff:
                min_diff = abs(energy - e_val)
                best_el = el
        
        if best_el and best_el not in found_elements:
            results.append({'x': energy, 'y': peak_height, 'text': best_el})
            # 简单的防重机制，防止相近峰标两遍 (可选)
            # found_elements.add(best_el) 
            
    return results

# --- 模式 A: 单微粒 (直接解析 UploadedFile 列表) ---
def parse_single_mode(uploaded_files):
    data_map = {}
    spec = {'x': [], 'y': []}
    
    for f in uploaded_files:
        # 重置指针，防止读取空内容
        f.seek(0)
        res_type, content = read_file_content(f, f.name)
        
        if res_type == 'map':
            el = parse_element_name(f.name)
            data_map[el] = content
        elif res_type == 'excel_map':
            # Excel 可能包含多个 Sheet (多个元素)
            for sheet_name, mat in content.items():
                data_map[sheet_name] = mat
        elif res_type == 'spec':
            spec = content
            
    # 对齐并返回结构
    return {'Single_Particle': {'data': align_images(data_map), 'spec': spec}}

# --- 模式 B: ZIP 批量 (解压后遍历) ---
def parse_batch_mode(zip_file_obj):
    particles = {}
    
    # 创建临时目录
    temp_dir = "temp_zip_extract"
    if os.path.exists(temp_dir): shutil.rmtree(temp_dir)
    os.makedirs(temp_dir)
    
    try:
        with zipfile.ZipFile(zip_file_obj, "r") as z:
            z.extractall(temp_dir)
    except:
        return {}
        
    # 遍历目录结构
    for root, dirs, files in os.walk(temp_dir):
        # 筛选有效文件
        valid_files = [f for f in files if f.lower().endswith(('.csv', '.xls', '.xlsx', '.txt'))]
        
        if valid_files:
            # 以前一文件夹名作为微粒 ID
            pid = os.path.basename(root)
            if pid == temp_dir: pid = "Root_Folder"
            if pid in particles: pid = f"{pid}_{len(particles)}" # 防重名
            
            particles[pid] = {'data': {}, 'spec': {'x':[], 'y':[]}}
            
            for f in valid_files:
                f_path = os.path.join(root, f)
                with open(f_path, 'rb') as fo:
                    res_type, content = read_file_content(fo, f)
                    
                    if res_type == 'map':
                        el = parse_element_name(f)
                        particles[pid]['data'][el] = content
                    elif res_type == 'excel_map':
                        for sheet, mat in content.items():
                            particles[pid]['data'][sheet] = mat
                    elif res_type == 'spec':
                        particles[pid]['spec'] = content
            
            # 对齐该微粒的所有图像
            particles[pid]['data'] = align_images(particles[pid]['data'])
            
    # 清理临时文件 (建议在 Web 服务中启用，防止磁盘占满)
    # shutil.rmtree(temp_dir)
    return particles

# ================= 3. 用户界面 (UI) =================

st.title("🔬 微粒全能分析平台")

# --- 侧边栏 ---
with st.sidebar:
    st.header("📂 数据导入")
    st.info("""
    **智能双模式：**
    1. **单微粒**：直接拖入多个 .csv/.xlsx/.txt 文件。
    2. **批量**：拖入一个 .zip 压缩包（内含多个微粒文件夹）。
    """)
    uploaded_files = st.file_uploader("请上传文件", accept_multiple_files=True)
    
    st.markdown("---")
    st.header("🎨 交互设置")
    zoom_level = st.slider("画布缩放 (Zoom)", 0.5, 4.0, 1.5, 0.1)
    bg_threshold = st.slider("背景降噪 (Threshold)", 0, 50, 2)
    draw_mode = st.selectbox("圈选工具", ["circle", "rect"], format_func=lambda x: "圆形" if x=="circle" else "矩形")

# --- 主逻辑区 ---
particles_db = {}

if uploaded_files:
    # 1. 检测文件类型，决定模式
    is_zip = any(f.name.lower().endswith('.zip') for f in uploaded_files)
    
    if is_zip:
        # 批量模式：只处理第一个 zip
        zip_file = next(f for f in uploaded_files if f.name.lower().endswith('.zip'))
        with st.spinner(f"正在解压分析 {zip_file.name}..."):
            particles_db = parse_batch_mode(zip_file)
        if particles_db:
            st.success(f"📦 已切换至批量模式，检测到 {len(particles_db)} 个微粒")
    else:
        # 单微粒模式
        particles_db = parse_single_mode(uploaded_files)
        if particles_db:
            st.success("📄 已切换至单微粒模式")

    if not particles_db:
        st.warning("未检测到有效数据，请检查文件格式。")
        
    else:
        # 2. 微粒选择器
        p_ids = sorted(list(particles_db.keys()))
        selected_pid = p_ids[0]
        
        # 如果微粒数量大于1，显示选择框
        if len(p_ids) > 1:
            st.sidebar.markdown("---")
            st.sidebar.subheader(f"选择微粒 ({len(p_ids)})")
            selected_pid = st.sidebar.selectbox("当前分析对象:", p_ids)
            
        # 获取当前数据
        current_data = particles_db[selected_pid]['data']
        current_spec = particles_db[selected_pid]['spec']
        
        st.markdown(f"### 🧪 当前分析: `{selected_pid}`")
        
        # 3. 渲染分析区
        if not current_data:
            st.error("该微粒没有有效的元素分布图 (Mapping) 数据。")
        else:
            col_canvas, col_result = st.columns([1.5, 1])
            
            # --- A. 图像合成与画布 ---
            # 获取尺寸
            shape = next(iter(current_data.values())).shape
            h, w = shape
            
            # 动态合成底图 (默认显示 Si, O, C)
            base_rgb = np.zeros((h, w, 3))
            colors = {'Si': 0, 'O': 1, 'C': 2} # R, G, B
            legend = []
            
            for el, ch_idx in colors.items():
                if el in current_data:
                    mat = current_data[el].copy()
                    # 简单降噪
                    mat[mat < bg_threshold] = 0
                    # 归一化
                    if mat.max() > 0:
                        base_rgb[:, :, ch_idx] = mat / mat.max()
                    legend.append(f"{el}")
            
            # 增强亮度并转为 8-bit 图片
            bg_uint8 = (np.clip(base_rgb * 1.5, 0, 1) * 255).astype(np.uint8)
            
            with col_canvas:
                # 计算缩放后的画布尺寸
                cw, ch = int(w * zoom_level), int(h * zoom_level)
                
                # 将 numpy array 转为 PIL Image 以供画布背景使用
                bg_pil = Image.fromarray(bg_uint8).resize((cw, ch))
                
                st.caption(f"合成视图 ({', '.join(legend)}) - 尺寸: {w}x{h} -> {cw}x{ch}")
                
                # 交互式画布
                canvas_result = st_canvas(
                    fill_color="rgba(255, 165, 0, 0.2)",  # 半透明橙色填充
                    stroke_width=2,
                    stroke_color="#ffffff",
                    background_image=bg_pil,
                    update_streamlit=True,
                    height=ch,
                    width=cw,
                    drawing_mode=draw_mode,
                    key=f"canvas_{selected_pid}_{zoom_level}", # ID变化时重置画布
                )
                
            # --- B. 选区统计结果 ---
            with col_result:
                if canvas_result.json_data and canvas_result.json_data["objects"]:
                    st.subheader("📊 选区成分分析")
                    obj = canvas_result.json_data["objects"][-1]
                    
                    # 创建 Mask (需要还原缩放比例)
                    mask = np.zeros((h, w), dtype=np.uint8)
                    scale = zoom_level
                    
                    if obj["type"] == "circle":
                        cx = int((obj["left"] + obj["radius"]) / scale)
                        cy = int((obj["top"] + obj["radius"]) / scale)
                        r = int(obj["radius"] / scale)
                        cv2.circle(mask, (cx, cy), r, 1, -1)
                        
                    elif obj["type"] == "rect":
                        x = int(obj["left"] / scale)
                        y = int(obj["top"] / scale)
                        wb = int(obj["width"] / scale)
                        hb = int(obj["height"] / scale)
                        cv2.rectangle(mask, (x, y), (x + wb, y + hb), 1, -1)
                    
                    # 计算选区内的元素总量
                    stats = {}
                    for el, mat in current_data.items():
                        if el == "SE": continue # 跳过电子图像
                        stats[el] = np.sum(mat * mask)
                    
                    # 归一化并绘图
                    total_intensity = sum(stats.values()) + 1e-9
                    df_res = pd.DataFrame({"Element": stats.keys(), "Intensity": stats.values()})
                    df_res["Percentage"] = df_res["Intensity"] / total_intensity
                    # 只显示占比 > 1% 的元素
                    df_res = df_res[df_res["Percentage"] > 0.01].sort_values("Percentage", ascending=False)
                    
                    st.plotly_chart(go.Figure(data=[go.Pie(
                        labels=df_res["Element"], 
                        values=df_res["Percentage"],
                        hole=0.4
                    )]), use_container_width=True)
                    
                    # 估算粒径 (假设 0.05 um/pixel)
                    pixel_area = np.sum(mask)
                    est_diameter = np.sqrt(4 * pixel_area / np.pi) * 0.05
                    st.metric("选区等效直径", f"{est_diameter:.2f} μm")
                    
                else:
                    st.info("👈 请在左侧图像上画圈，查看局部成分占比。")
                    
            # --- C. 能谱分析 ---
            st.markdown("---")
            if current_spec['x']:
                st.subheader("📈 EDS 能谱 (自动标峰)")
                
                # 自动找峰
                peaks = auto_identify_peaks(current_spec['x'], current_spec['y'])
                
                fig = go.Figure()
                # 绘制波形
                fig.add_trace(go.Scatter(
                    x=current_spec['x'], y=current_spec['y'],
                    mode='lines', fill='tozeroy', line=dict(color='#444'), name='Spectrum'
                ))
                # 添加标注
                for p in peaks:
                    fig.add_annotation(
                        x=p['x'], y=p['y'],
                        text=p['text'],
                        showarrow=True, arrowhead=2, ay=-30
                    )
                
                fig.update_layout(
                    xaxis_title="Energy (keV)", 
                    yaxis_title="Counts",
                    height=400,
                    hovermode="x unified"
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.caption("该微粒未检测到能谱 (.txt) 数据")
                
            # --- D. 图集概览 ---
            with st.expander("查看所有元素分图 (点击展开)"):
                # 获取所有元素名并排序
                elements = sorted(current_data.keys())
                cols = st.columns(6) # 每行6个
                for i, el in enumerate(elements):
                    with cols[i % 6]:
                        # 显示缩略图 (归一化)
                        mat = current_data[el]
                        norm_mat = mat / (mat.max() + 1e-6)
                        st.image(norm_mat, caption=el, use_container_width=True)

else:
    # 引导页
    st.info("👋 欢迎使用微粒分析平台！请在左侧上传数据开始。")
