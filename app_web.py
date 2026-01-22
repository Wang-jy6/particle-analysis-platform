import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import cv2
import io
import os
import torch
import torch.nn as nn
from scipy.signal import find_peaks

# ================= 1. 配置与常量 =================
st.set_page_config(page_title="微粒分析云平台 Pro", page_icon="🔬", layout="wide")

# 常见元素的 EDS 特征能量 (Kα 线, 单位 keV)
ELEMENT_ENERGIES = {
    'C': 0.277, 'N': 0.392, 'O': 0.525, 'F': 0.677,
    'Na': 1.041, 'Mg': 1.253, 'Al': 1.486, 'Si': 1.739,
    'P': 2.013, 'S': 2.307, 'Cl': 2.621, 'K': 3.312,
    'Ca': 3.690, 'Ti': 4.508, 'Cr': 5.411, 'Mn': 5.894,
    'Fe': 6.398, 'Co': 6.924, 'Ni': 7.471, 'Cu': 8.040,
    'Zn': 8.630, 'Au': 2.120, 'Mo': 2.290 
}

# 13个目标元素 (模型输入顺序)
TARGET_ELEMENTS = ['C', 'O', 'Si', 'Al', 'Ca', 'Fe', 'K', 'S', 'Cl', 'Cu', 'Zn', 'P', 'N']
IMG_SIZE = (128, 128)
MICRONS_PER_PIXEL = 0.05 # 默认假设

# ================= 2. AI 模型定义 (必须与训练时一致) =================
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        return self.sigmoid(self.fc2(self.relu1(self.fc1(self.avg_pool(x)))) + 
                            self.fc2(self.relu1(self.fc1(self.max_pool(x)))))

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        return self.sigmoid(self.conv1(torch.cat([avg_out, max_out], dim=1)))

class CBAM(nn.Module):
    def __init__(self, in_planes):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes)
        self.sa = SpatialAttention()
    def forward(self, x):
        return x * self.ca(x) * self.sa(x)

class DualStreamCNN(nn.Module):
    def __init__(self, num_classes):
        super(DualStreamCNN, self).__init__()
        self.stream_phys = nn.Sequential(
            nn.Conv2d(2, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2, 2)
        )
        self.stream_chem = nn.Sequential(
            nn.Conv2d(14, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2, 2)
        )
        self.fusion_conv = nn.Conv2d(96, 128, 3, padding=1)
        self.cbam = CBAM(128)
        self.pool = nn.MaxPool2d(2, 2)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 16 * 16, 256), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    def forward(self, x):
        f_phys = self.stream_phys(x[:, 0:2])
        f_chem = self.stream_chem(x[:, 2:])
        f_cat = torch.cat([f_phys, f_chem], dim=1)
        return self.classifier(self.pool(self.cbam(self.fusion_conv(f_cat))))

# ================= 3. 功能函数 =================

@st.cache_resource
def load_model():
    """加载模型权重"""
    # 尝试在不同路径寻找模型
    paths = ["best_model_16ch.pth", "./Processed_Data/best_model_16ch.pth"]
    model_path = next((p for p in paths if os.path.exists(p)), None)
    
    # 尝试寻找类别名文件
    cls_paths = ["class_names.npy", "./Processed_Data/class_names.npy"]
    cls_path = next((p for p in cls_paths if os.path.exists(p)), None)
    
    if model_path and cls_path:
        try:
            class_names = np.load(cls_path)
            model = DualStreamCNN(num_classes=len(class_names))
            model.load_state_dict(torch.load(model_path, map_location='cpu'))
            model.eval()
            return model, class_names
        except Exception as e:
            st.error(f"模型加载失败: {e}")
            return None, None
    return None, None

def process_uploaded_files(uploaded_files):
    """内存文件读取"""
    data_map = {}
    spectrum_data = {'x': [], 'y': [], 'meta': {}}
    
    for uploaded_file in uploaded_files:
        fname = uploaded_file.name
        
        # 1. 读取 CSV Mapping
        if fname.endswith(".csv"):
            is_sem = "电子图像" in fname
            # 提取元素名
            el_name = fname.split(" ")[0].split(".")[0]
            if "_" in el_name: el_name = el_name.split("_")[-1]
            if is_sem: el_name = "SE" # 标记为 SE 图像
            
            try:
                df = pd.read_csv(uploaded_file, header=None)
                mat = df.apply(pd.to_numeric, errors='coerce').fillna(0).values
                data_map[el_name] = mat
            except: pass

        # 2. 读取 Excel
        elif fname.endswith((".xls", ".xlsx")):
            try:
                xls = pd.ExcelFile(uploaded_file)
                for sheet in xls.sheet_names:
                    clean_sheet = sheet.strip()
                    target_name = clean_sheet if len(clean_sheet) < 5 else fname.split(".")[0]
                    df = pd.read_excel(xls, sheet_name=sheet, header=None)
                    mat = df.apply(pd.to_numeric, errors='coerce').fillna(0).values
                    if mat.size > 100: data_map[target_name] = mat
            except: pass

        # 3. 读取 能谱 TXT
        elif fname.endswith(".txt"):
            try:
                stringio = io.StringIO(uploaded_file.getvalue().decode("utf-8", errors='ignore'))
                lines = stringio.readlines()
                is_data = False
                for line in lines:
                    line = line.strip()
                    if line.startswith("#"):
                        parts = line.split(":")
                        if len(parts) > 1: spectrum_data['meta'][parts[0].replace("#", "").strip()] = parts[1].strip()
                    if "SPECTRUM" in line: is_data = True; continue
                    if is_data and "," in line:
                        x, y = map(float, line.split(","))
                        spectrum_data['x'].append(x)
                        spectrum_data['y'].append(y)
            except: pass

    return data_map, spectrum_data

def build_tensor_for_ai(data_map):
    """构建 16通道 Tensor"""
    # 1. 统一尺寸
    if not data_map: return None
    shape = IMG_SIZE
    
    # 获取或生成 SE 图
    if "SE" in data_map:
        sem_raw = data_map["SE"]
    else:
        # 如果没有SE图，用所有元素的平均值代替 (权宜之计)
        sem_raw = np.mean(list(data_map.values()), axis=0)
    
    sem_layer = cv2.resize(sem_raw, shape).astype(np.float32)
    if sem_layer.max() > 0: sem_layer /= sem_layer.max()
    
    # 2. 计算纹理
    img_uint8 = (sem_layer * 255).astype(np.uint8)
    gx = cv2.Scharr(img_uint8, cv2.CV_64F, 1, 0)
    gy = cv2.Scharr(img_uint8, cv2.CV_64F, 0, 1)
    tex_layer = cv2.addWeighted(cv2.convertScaleAbs(gx), 0.5, cv2.convertScaleAbs(gy), 0.5, 0)
    tex_layer = tex_layer.astype(np.float32) / 255.0
    
    # 3. 处理化学元素
    chem_layers = {}
    for el in TARGET_ELEMENTS:
        if el in data_map:
            mat = cv2.resize(data_map[el], shape).astype(np.float32)
            if mat.max() > 0: mat /= mat.max()
            chem_layers[el] = mat
        else:
            chem_layers[el] = np.zeros(shape, dtype=np.float32)
            
    # 4. Si 修正
    si_raw = chem_layers['Si']
    al_raw = chem_layers['Al']
    ca_raw = chem_layers['Ca']
    mineral_factor = np.maximum(al_raw, ca_raw)
    trust_mask = np.tanh(mineral_factor * 5.0)
    si_corr = si_raw * trust_mask
    
    # 5. 堆叠 (顺序必须严格: SE, Tex, Si_raw, Si_corr, Others...)
    layers = [sem_layer, tex_layer, si_raw, si_corr]
    for el in TARGET_ELEMENTS:
        if el != 'Si':
            layers.append(chem_layers[el])
            
    return torch.FloatTensor(np.stack(layers, axis=0)).unsqueeze(0), sem_raw

def find_spectrum_peaks(x, y):
    """能谱自动标峰"""
    x = np.array(x)
    y = np.array(y)
    if len(y) == 0: return []
    
    # 找峰值 (高度至少是最大值的 5%)
    peaks, _ = find_peaks(y, height=np.max(y)*0.05, distance=10)
    
    annotations = []
    found_elements = set()
    
    for p in peaks:
        energy = x[p]
        intensity = y[p]
        
        # 匹配元素
        best_el = None
        min_diff = 0.05 # 容差 50 eV
        
        for el, el_energy in ELEMENT_ENERGIES.items():
            diff = abs(energy - el_energy)
            if diff < min_diff:
                min_diff = diff
                best_el = el
        
        if best_el:
            annotations.append({
                'x': energy, 'y': intensity, 
                'text': best_el, 
                'showarrow': True, 'arrowhead': 2
            })
            found_elements.add(best_el)
            
    return annotations

# ================= 4. 界面逻辑 =================
st.title("☁️ 微粒分析云平台 Pro (AI + EDS)")
st.caption("集成双流CNN预测与能谱自动识别功能")

# 模型加载状态
model, class_names = load_model()
if model:
    st.sidebar.success("✅ AI 模型已加载")
else:
    st.sidebar.warning("⚠️ 未检测到模型文件 (best_model_16ch.pth)，AI功能暂不可用")

with st.container():
    uploaded_files = st.file_uploader("上传单个微粒的所有数据", accept_multiple_files=True)

if uploaded_files:
    data_map, spectrum = process_uploaded_files(uploaded_files)
    
    if not data_map:
        st.error("未找到有效的 CSV/Excel 数据")
    else:
        # --- 布局 ---
        col_main, col_info = st.columns([1.5, 1])
        
        # 1. 图像合成 (主图)
        with col_main:
            st.subheader("🖼️ 微粒总样貌")
            all_els = sorted(list(data_map.keys()))
            sel_els = st.multiselect("合成通道", all_els, default=[e for e in ['Si','O','C','Ca'] if e in all_els])
            
            # 简单的合成逻辑
            if sel_els:
                shape = next(iter(data_map.values())).shape
                rgb = np.zeros((shape[0], shape[1], 3))
                colors = {'Si':(1,0,0), 'O':(0,1,0), 'C':(0,0,1), 'Ca':(1,1,0), 'Al':(1,0,1)}
                for el in sel_els:
                    mat = data_map[el]
                    if mat.shape != shape: mat = cv2.resize(mat, (shape[1],shape[0]))
                    if mat.max()>0: mat = mat/mat.max()
                    c = colors.get(el, (0.5,0.5,0.5))
                    for i in range(3): rgb[:,:,i] += mat * c[i]
                st.image(np.clip(rgb,0,1), use_container_width=True, clamp=True)

        # 2. AI 诊断 & 成分
        with col_info:
            st.subheader("🤖 AI 智能诊断")
            
            if model and st.button("开始 AI 识别"):
                tensor, sem_raw = build_tensor_for_ai(data_map)
                with st.spinner("神经网络计算中..."):
                    with torch.no_grad():
                        probs = torch.nn.functional.softmax(model(tensor), dim=1).numpy()[0]
                    
                    pred_idx = np.argmax(probs)
                    pred_class = class_names[pred_idx]
                    conf = probs[pred_idx]
                    
                    # 规则修正逻辑 (简化版)
                    final_pred = pred_class
                    note = "符合特征"
                    # 这里可以加入您的 AR/Diameter 计算逻辑进行修正
                    
                    # 显示结果
                    st.metric("预测类别", final_pred, delta=f"置信度 {conf:.1%}")
                    
                    # 概率分布图
                    chart_data = pd.DataFrame({"Class": class_names, "Prob": probs})
                    st.bar_chart(chart_data, x="Class", y="Prob", height=200)

            st.markdown("---")
            st.subheader("📊 元素占比")
            sums = {k: v.sum() for k,v in data_map.items()}
            tot = sum(sums.values())
            pie_d = {k:v for k,v in sums.items() if v/tot > 0.01}
            st.plotly_chart(go.Figure(data=[go.Pie(labels=list(pie_d.keys()), values=list(pie_d.values()))]), use_container_width=True)

        # 3. 能谱 (带自动标峰)
        st.subheader("📈 EDS 能谱自动分析")
        if spectrum['x']:
            # 自动找峰
            annotations = find_spectrum_peaks(spectrum['x'], spectrum['y'])
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=spectrum['x'], y=spectrum['y'], fill='tozeroy', line=dict(color='#2E86C1')))
            
            # 添加标峰
            fig.update_layout(
                annotations=annotations,
                height=350,
                xaxis_title="Energy (keV)",
                yaxis_title="Counts",
                title="自动识别谱峰元素"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # 显示找到的元素
            found_els = sorted(list(set([a['text'] for a in annotations])))
            st.success(f"🔍 谱图中检测到的特征峰: {', '.join(found_els)}")
            
        else:
            st.info("未上传能谱 TXT 文件")
            
        # 4. 单元素图
        st.subheader("🧩 元素分布")
        cols = st.columns(6)
        for i, el in enumerate(all_els):
            with cols[i%6]:
                plt.figure()
                plt.imshow(data_map[el], cmap='magma')
                plt.axis('off')
                plt.title(el)
                st.pyplot(plt)
                plt.close()
