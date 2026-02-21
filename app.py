import streamlit as st
import google.generativeai as genai
from PIL import Image, ImageOps
import pypdf
import time
import os
import tempfile
from fpdf import FPDF
import datetime
import re
import cv2
import numpy as np
import extra_streamlit_components as stx
import pandas as pd
import urllib.request
import json
import xml.etree.ElementTree as ET

# --- CONFIGURACIÓN ---
st.set_page_config(page_title="LabMind", page_icon="🧬", layout="wide")

# --- ESTILOS CSS REVISADOS ---
st.markdown("""
<style>
    .block-container { padding-top: 0.5rem !important; padding-bottom: 120px !important; }
    div[data-testid="stSelectbox"] { margin-bottom: -15px !important; }
    .stButton>button { width: 100%; border-radius: 8px; font-weight: bold; margin-top: 10px; }
    button[data-testid="baseButton-primary"] { background-color: #0066cc !important; color: white !important; border: none !important; }
    
    details { padding: 15px; border-radius: 8px; margin-bottom: 10px; }
    summary { cursor: pointer; font-size: 1.05rem; outline: none; font-weight: bold; list-style: none; }
    summary::-webkit-details-marker { display: none; }
    details > summary::before { content: '🔽 '; font-size: 0.9em; }
    details[open] > summary::before { content: '🔼 '; }
    details[open] summary { border-bottom: 1px dashed currentcolor; padding-bottom: 8px; margin-bottom: 8px; }
    details p { margin-top: 5px; margin-bottom: 0; line-height: 1.5; }
    
    details.diagnosis-box { background-color: #e3f2fd; border-left: 6px solid #2196f3; color: #0d47a1; }
    details.action-box { background-color: #ffebee; border-left: 6px solid #f44336; color: #b71c1c; }
    details.material-box { background-color: #e8f5e9; border-left: 6px solid #4caf50; color: #1b5e20; }
    details.radiomics-box { background-color: #f3e5f5; border-left: 6px solid #9c27b0; color: #4a148c; }
    details.pocus-box { background-color: #e0f2f1; border-left: 6px solid #00897b; color: #004d40; }
    details.longevity-box { background-color: #fff8e1; border-left: 6px solid #ffc107; color: #ff6f00; }
    details.pubmed-box { background-color: #e8eaf6; border-left: 6px solid #3f51b5; color: #1a237e; }
    
    div[data-testid="stChatInput"] {
        position: fixed !important; bottom: 0px !important; left: 0px !important;
        width: 100% !important; background-color: white !important;
        padding: 10px 20px 25px 20px !important; z-index: 9999 !important;
        box-shadow: 0px -4px 10px rgba(0,0,0,0.1) !important;
    }
    .pull-up { margin-top: -25px !important; margin-bottom: 5px !important; height: 1px !important; display: block !important; }
</style>
""", unsafe_allow_html=True)

# --- GESTOR DE ESTADO ---
cookie_manager = stx.CookieManager()
time.sleep(0.1)

for key in ["autenticado", "api_key", "resultado_analisis", "pdf_bytes", "chat_messages", "img_marcada", "video_bytes", "modelos_disponibles"]:
    if key not in st.session_state: st.session_state[key] = [] if key in ["chat_messages", "modelos_disponibles"] else ("" if key == "api_key" else None)
if "autenticado" not in st.session_state or not st.session_state.autenticado:
    st.session_state.autenticado = False
if "pocus_metrics" not in st.session_state: st.session_state.pocus_metrics = {}

cookie_api_key = cookie_manager.get(cookie="labmind_secret_key")
if not st.session_state.autenticado:
    if cookie_api_key:
        st.session_state.api_key = cookie_api_key; st.session_state.autenticado = True; st.rerun()
    else:
        st.title("LabMind Acceso")
        k = st.text_input("API Key:", type="password")
        if st.button("Entrar", type="primary"):
            expires = datetime.datetime.now() + datetime.timedelta(days=30)
            cookie_manager.set("labmind_secret_key", k, expires_at=expires)
            st.session_state.api_key = k; st.session_state.autenticado = True; st.rerun()
        st.stop()

# ==========================================
#      MOTOR POCUS GOD MODE V6 (SINGULARIDAD)
# ==========================================

def procesar_pocus_v6_singularidad(video_path):
    """Límite matemático: Speckle Tracking (Lucas-Kanade), Índice VCI, Termografía M-Mode"""
    try:
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0 or np.isnan(fps): fps = 30.0

        ret, frame1 = cap.read()
        if not ret: return None, None, None, {}

        # 1. OPTIMIZACIÓN ESPACIAL EXTREMA
        h_orig, w_orig = frame1.shape[:2]
        scale = 320.0 / w_orig 
        new_w, new_h = 320, int(h_orig * scale)
        frame1 = cv2.resize(frame1, (new_w, new_h))

        # 2. AUTO-CROP & MÁSCARAS
        gray_init = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        _, beam_thresh = cv2.threshold(gray_init, 15, 255, cv2.THRESH_BINARY)
        beam_contours, _ = cv2.findContours(beam_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        roi_mask = np.zeros_like(gray_init)
        if beam_contours:
            largest_beam = max(beam_contours, key=cv2.contourArea)
            cv2.drawContours(roi_mask, [largest_beam], -1, 255, -1)
        else:
            cv2.ellipse(roi_mask, (new_w//2, new_h//2), (int(new_w*0.4), int(new_h*0.4)), 0, 0, 360, 255, -1)

        # 3. SPECKLE TRACKING INIT
        tissue_mask = cv2.bitwise_and(gray_init, gray_init, mask=roi_mask)
        _, tissue_only = cv2.threshold(tissue_mask, 50, 255, cv2.THRESH_BINARY)
        p0 = cv2.goodFeaturesToTrack(gray_init, maxCorners=50, qualityLevel=0.1, minDistance=10, mask=tissue_only)
        old_gray = gray_init.copy()
        speckle_displacements = []

        volumes_simpson = []
        vessel_widths = []
        m_mode_columns = []
        b_lines_score = 0
        has_doppler = False
        frame_count = 0
        
        static_black_map = np.ones_like(gray_init, dtype=np.uint8) * 255
        best_endo_frame = frame1.copy()
        doppler_frame = np.zeros_like(frame1)
        lower_color, upper_color = np.array([0, 50, 50]), np.array([179, 255, 255])

        lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

        while True:
            ret, frame2 = cap.read()
            if not ret: break
            
            frame_count += 1
            skip_rate = max(1, int(fps // 15))
            if frame_count % skip_rate != 0: continue

            frame2 = cv2.resize(frame2, (new_w, new_h))
            next_gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
            next_gray_smooth = cv2.medianBlur(next_gray, 5) 

            if p0 is not None:
                p1, st_lk, err = cv2.calcOpticalFlowPyrLK(old_gray, next_gray_smooth, p0, None, **lk_params)
                if p1 is not None:
                    good_new = p1[st_lk==1]
                    good_old = p0[st_lk==1]
                    if len(good_new) > 0:
                        distances = np.linalg.norm(good_new - good_old, axis=1)
                        speckle_displacements.append(np.mean(distances))
                    p0 = good_new.reshape(-1, 1, 2)
            old_gray = next_gray_smooth.copy()

            masked_gray = cv2.bitwise_and(next_gray_smooth, next_gray_smooth, mask=roi_mask)
            _, thresh_black = cv2.threshold(masked_gray, 35, 255, cv2.THRESH_BINARY_INV)
            thresh_black = cv2.bitwise_and(thresh_black, thresh_black, mask=roi_mask)
            static_black_map = cv2.bitwise_and(static_black_map, thresh_black)

            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
            thresh_black_clean = cv2.morphologyEx(thresh_black, cv2.MORPH_OPEN, kernel)
            contours, _ = cv2.findContours(thresh_black_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            max_area = 0; best_c = None; dynamic_x = new_w // 2

            for c in contours:
                area = cv2.contourArea(c)
                if area > max_area and area > (new_w * new_h * 0.03): 
                    max_area = area; best_c = c

            if best_c is not None:
                x, y, w_bb, h_bb = cv2.boundingRect(best_c)
                dynamic_x = x + (w_bb // 2)
                vessel_widths.append(w_bb)
                
                vol_proxy = (max_area ** 2) / float(h_bb) if h_bb > 0 else 0
                volumes_simpson.append(vol_proxy)

                if vol_proxy >= max(volumes_simpson):
                    best_endo_frame = frame2.copy()
                    cv2.drawContours(best_endo_frame, [best_c], -1, (0, 255, 100), 2)
                    if p0 is not None:
                        for pt in p0:
                            a, b = pt.ravel()
                            cv2.circle(best_endo_frame, (int(a), int(b)), 2, (255, 0, 0), -1)

            m_mode_columns.append(next_gray_smooth[:, dynamic_x])

            hsv = cv2.cvtColor(frame2, cv2.COLOR_BGR2HSV)
            mask_color = cv2.inRange(hsv, lower_color, upper_color)
            if cv2.countNonZero(mask_color) > (new_w * new_h * 0.005):
                has_doppler = True
                doppler_frame = cv2.bitwise_or(doppler_frame, cv2.bitwise_and(frame2, frame2, mask=mask_color))

            lower_half = next_gray_smooth[int(new_h*0.5):, :]
            sobelx = cv2.Sobel(lower_half, cv2.CV_64F, 1, 0, ksize=3)
            _, thresh_b = cv2.threshold(np.absolute(sobelx), 160, 255, cv2.THRESH_BINARY)
            if cv2.countNonZero(thresh_b) > (new_w * new_h * 0.015):
                b_lines_score += 1

        cap.release()

        # 4. MÉTRICAS FINALES
        metrics = {}
        frames_procesados = frame_count // skip_rate if skip_rate > 0 else 1
        
        if len(volumes_simpson) > 5:
            vols = np.array(volumes_simpson)
            smoothed_vols = np.convolve(vols, np.ones(3)/3, mode='valid')
            edv, esv = np.percentile(smoothed_vols, 95), np.percentile(smoothed_vols, 5)
            metrics['FEVI'] = round(max(10.0, min(((edv - esv) / edv) * 100 if edv > 0 else 0, 85.0)), 1) 
            
            crossings = np.where(np.diff(np.sign(smoothed_vols - np.mean(smoothed_vols))) > 0)[0]
            if len(crossings) > 1:
                intervals = np.diff(crossings) * skip_rate
                bpm = 60.0 / ((np.mean(intervals)) / fps)
                metrics['BPM'] = round(max(30, min(bpm, 200)))
                metrics['Ritmo'] = "Irregular ⚠️" if np.std(intervals) > (np.mean(intervals) * 0.15) else "Regular ✅"
            else: metrics['BPM'] = "N/A"; metrics['Ritmo'] = "N/A"
        else: metrics['FEVI'] = "N/A"; metrics['BPM'] = "N/A"; metrics['Ritmo'] = "N/A"

        metrics['Strain_Proxy'] = round(np.mean(speckle_displacements)*10, 1) if speckle_displacements else "N/A"
        if len(vessel_widths) > 5:
            max_w = np.percentile(vessel_widths, 95)
            min_w = np.percentile(vessel_widths, 5)
            metrics['Colapso_VCI'] = round(((max_w - min_w) / max_w) * 100, 1) if max_w > 0 else "N/A"
        else: metrics['Colapso_VCI'] = "N/A"

        m_mode_pil = None
        if len(m_mode_columns) > 0:
            m_mode_img = np.transpose(np.array(m_mode_columns))
            _, m_thresh = cv2.threshold(m_mode_img, 200, 255, cv2.THRESH_BINARY)
            y_coords = np.where(m_thresh > 0)[0]
            metrics['TAPSE_Proxy'] = round((np.percentile(y_coords, 95) - np.percentile(y_coords, 5)) * (150.0 / new_h), 1) if len(y_coords) > 0 else "N/A"
                
            m_mode_img = cv2.resize(m_mode_img, (new_w, new_h))
            m_mode_color = cv2.applyColorMap(np.uint8(m_mode_img), cv2.COLORMAP_OCEAN) 
            m_mode_pil = Image.fromarray(cv2.cvtColor(m_mode_color, cv2.COLOR_BGR2RGB))

        cv2.ellipse(static_black_map, (new_w//2, new_h//2), (int(new_w*0.25), int(new_h*0.25)), 0, 0, 360, 0, -1)
        metrics['Derrame'] = cv2.countNonZero(static_black_map) > (new_w * new_h * 0.05)
        metrics['B_Lines'] = b_lines_score > (frames_procesados * 0.1)
        metrics['Doppler'] = has_doppler

        endo_pil = Image.fromarray(cv2.cvtColor(best_endo_frame, cv2.COLOR_BGR2RGB))
        doppler_pil = Image.fromarray(cv2.cvtColor(doppler_frame, cv2.COLOR_BGR2RGB)) if has_doppler else None
            
        return m_mode_pil, endo_pil, doppler_pil, metrics
    except Exception as e:
        print(f"Error God Mode V6: {e}")
        return None, None, None, {}

# ==========================================
#      OTRAS FUNCIONES IA
# ==========================================

def buscar_en_pubmed(query, max_results=4):
    try:
        base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
        search_url = f"{base_url}esearch.fcgi?db=pubmed&term={urllib.parse.quote(query)}&retmode=json&retmax={max_results}&sort=relevance"
        req = urllib.request.Request(search_url, headers={'User-Agent': 'Mozilla/5.0'})
        data = json.loads(urllib.request.urlopen(req).read().decode('utf-8'))
        ids = data.get('esearchresult', {}).get('idlist', [])
        if not ids: return "No se encontraron artículos en PubMed."
        
        fetch_url = f"{base_url}efetch.fcgi?db=pubmed&id={','.join(ids)}&retmode=xml"
        root = ET.fromstring(urllib.request.urlopen(urllib.request.Request(fetch_url, headers={'User-Agent': 'Mozilla/5.0'})).read())
        
        resultados = ""
        for article in root.findall('.//PubmedArticle'):
            pmid = article.find('.//PMID').text
            title = article.find('.//ArticleTitle').text if article.find('.//ArticleTitle') is not None else "Sin título"
            abs_node = article.find('.//AbstractText')
            abstract = "".join(abs_node.itertext()) if abs_node is not None else "Sin abstract."
            resultados += f"PMID: {pmid}\nTÍTULO: {title}\nABSTRACT: {abstract}\n\n"
        return resultados
    except Exception as e: return f"Error PubMed: {e}"

def aislar_trazado_ecg(pil_image):
    try:
        img_cv = cv2.cvtColor(np.array(pil_image.convert('RGB')), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 15)
        smooth = cv2.GaussianBlur(thresh, (3, 3), 0)
        return Image.fromarray(cv2.cvtColor(smooth, cv2.COLOR_GRAY2RGB))
    except: return pil_image

def extraer_y_dibujar_bboxes(texto, img_pil=None):
    patron = r'BBOX:\s*\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]\s*LABEL:\s*([^\n<]+)'
    matches = re.findall(patron, texto)
    if not matches or img_pil is None: return None, texto, False
    
    img_cv = cv2.cvtColor(np.array(img_pil.convert('RGB')), cv2.COLOR_RGB2BGR)
    h, w = img_cv.shape[:2]
    
    for match in matches:
        ymin, xmin, ymax, xmax, label = match
        try:
            x1, y1 = max(0, int(int(xmin) * w / 1000)), max(0, int(int(ymin) * h / 1000))
            x2, y2 = min(w, int(int(xmax) * w / 1000)), min(h, int(int(ymax) * h / 1000))
            cv2.rectangle(img_cv, (x1, y1), (x2, y2), (0, 0, 255), 4) 
            texto_label = label.strip().upper()
            escala = max(0.6, w/1000); grosor = max(2, int(w/500))
            (w_txt, h_txt), _ = cv2.getTextSize(texto_label, cv2.FONT_HERSHEY_SIMPLEX, escala, grosor)
            cv2.rectangle(img_cv, (x1, max(0, y1-h_txt-10)), (x1 + w_txt, max(0, y1)), (0,0,0), -1)
            cv2.putText(img_cv, texto_label, (x1, max(0, y1-5)), cv2.FONT_HERSHEY_SIMPLEX, escala, (255, 255, 255), grosor, cv2.LINE_AA)
        except: pass

    return Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)), re.sub(patron, '', texto).strip(), True

def create_pdf(texto):
    class PDF(FPDF):
        def header(self): 
            self.set_font('helvetica', 'B', 12)
            self.cell(0, 10, 'LabMind - Informe IA', align='C', new_x="LMARGIN", new_y="NEXT")
            self.ln(5)
        def footer(self): 
            self.set_y(-15); self.set_font('helvetica', 'I', 8)
            self.cell(0, 10, f'Pag {self.page_no()}', align='C')
            
    pdf = PDF(); pdf.add_page(); pdf.set_font("helvetica", size=10)
    pdf.cell(0, 10, f"Fecha: {datetime.datetime.now().strftime('%d/%m/%Y %H:%M')}", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(5)
    clean = re.sub(r'<[^>]+>', '', texto).replace('€','EUR').replace('’',"'").replace('“','"').replace('”','"')
    pdf.multi_cell(0, 5, txt="".join(c for c in clean if ord(c) < 256))
    return bytes(pdf.output())

# ==========================================
#      INTERFAZ DE USUARIO PRINCIPAL
# ==========================================

st.title("🩺 LabMind")
col_left, col_center, col_right = st.columns([1, 2, 1])

with col_left:
    st.subheader("⚙️ Motor de IA")
    if st.session_state.autenticado and not st.session_state.modelos_disponibles:
        try:
            genai.configure(api_key=st.session_state.api_key)
            modelos_encontrados = [m.name.replace('models/', '') for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
            st.session_state.modelos_disponibles = sorted(modelos_encontrados, reverse=True) if modelos_encontrados else ["gemini-1.5-pro"]
        except: st.session_state.modelos_disponibles = ["gemini-1.5-pro"]

    lista_para_mostrar = st.session_state.modelos_disponibles if st.session_state.modelos_disponibles else ["Inicia sesión..."]
    
    idx_defecto = 0
    for i, modelo in enumerate(lista_para_mostrar):
        if "3" in modelo.lower() and "flash" in modelo.lower() and "preview" in modelo.lower():
            idx_defecto = i
            break

    st.session_state.modelo_seleccionado = st.selectbox("Versión de Gemini:", lista_para_mostrar, index=idx_defecto)
    
    # --- MENÚ ZONA ANATÓMICA RESTAURADO (AUTODETECT POR DEFECTO) ---
    st.session_state.punto_cuerpo = st.selectbox("Zona anatómica:", ["✨ Autodetectar", "Cara", "Pecho", "Abdomen", "Sacro/Glúteo", "Pierna", "Talón", "Pie", "No aplicable"])

with col_center:
    st.subheader("1. Selección de Modo")
    # --- MENÚ ESPECIALIDAD CON AUTODETECT POR DEFECTO ---
    modo = st.selectbox("Especialidad:", 
                 ["✨ Autodetectar", "🩹 Heridas / Úlceras", "🦇 Ecografía / POCUS (God Mode)", "📚 Agente Investigador (PubMed)", "📈 ECG", "💀 RX/TAC/Resonancia", "🧴 Dermatología", "🩸 Analítica Funcional", "🧩 Integral"])
    contexto = st.selectbox("🏥 Contexto:", ["Hospitalización", "Residencia", "Urgencias", "UCI", "Domicilio"])
    
    archivos = []; audio_val = None 
    
    if modo == "📚 Agente Investigador (PubMed)":
        st.info("🤖 **Agente Activo:** Conectado a la API oficial del NCBI.")
        query_pubmed = st.text_input("🔍 ¿Qué duda clínica quieres investigar en PubMed?")
        with st.expander("📝 Notas Clínicas / Contexto", expanded=False):
            notas = st.text_area("Contexto", height=70, label_visibility="collapsed")
    else:
        fs = st.file_uploader("Subir Archivos Clínicos", type=['jpg','png','pdf', 'mp4', 'mov'], accept_multiple_files=True)
        if fs:
            for f in fs:
                if f.type.startswith('video') or f.name.lower().endswith(('mp4', 'mov')): archivos.append(("video", f))
                elif "pdf" in f.type: archivos.append(("doc", f))
                else: archivos.append(("img", f))
            
        with st.expander("📝 Notas Clínicas / Preguntas Específicas", expanded=False):
            notas = st.text_area("Notas", height=70, label_visibility="collapsed")
        
        with st.expander("🎙️ Adjuntar Nota de Voz", expanded=False):
            if hasattr(st, "audio_input"): audio_val = st.audio_input("Dictar notas", key="mic", label_visibility="collapsed")

    c1, c2 = st.columns([3, 1])
    with c1: btn_analizar = st.button("🚀 ANALIZAR / INVESTIGAR", type="primary", use_container_width=True)
    with c2:
        if st.button("🔄 NUEVO", type="secondary", use_container_width=True):
            for k in ["resultado_analisis", "img_marcada", "video_bytes", "chat_messages"]: st.session_state[k] = None if "img" in k or "video" in k or "resultado" in k else []
            st.session_state.pocus_metrics = {}
            st.rerun()

    if btn_analizar:
        st.session_state.resultado_analisis = None
        st.session_state.video_bytes = None
        st.session_state.img_marcada = None
        st.session_state.pocus_metrics = {}
        st.session_state.chat_messages = []
        
        with st.spinner("🧠 Analizando e identificando contexto..."):
            try:
                genai.configure(api_key=st.session_state.api_key)
                model = genai.GenerativeModel(st.session_state.modelo_seleccionado) 
                con = []; txt_docs = ""; datos_pubmed = ""
                
                if modo == "📚 Agente Investigador (PubMed)" and query_pubmed:
                    datos_pubmed = buscar_en_pubmed(query_pubmed)
                
                imagen_para_visor = None
                
                if modo != "📚 Agente Investigador (PubMed)":
                    if audio_val:
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tf_audio:
                            tf_audio.write(audio_val.read())
                        con.append(genai.upload_file(path=tf_audio.name))
                        
                    for tipo, file in archivos:
                        if tipo == "doc":
                            file.seek(0)
                            txt_docs += "".join([p.extract_text() for p in pypdf.PdfReader(file).pages])
                        
                        elif tipo == "video":
                            file.seek(0)
                            st.session_state.video_bytes = file.read()
                            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tf_video:
                                tf_video.write(st.session_state.video_bytes)
                            
                            # --- LÓGICA AUTOPILOT POCUS ---
                            # Se activa si el usuario elige "POCUS" manualmente, O si sube un vídeo y está en "Autodetectar"
                            if "POCUS" in modo or modo == "✨ Autodetectar":
                                st.toast("🧮 Vídeo detectado: Activando God Mode Ultrasonido...")
                                m_mode, endo_map, doppler_map, p_metrics = procesar_pocus_v6_singularidad(tf_video.name)
                                st.session_state.pocus_metrics = p_metrics
                                st.session_state.img_marcada = m_mode if m_mode else endo_map
                                
                                if p_metrics:
                                    txt_docs += f"\n[TELEMETRÍA POCUS V6]\n- FEVI Estimada (Vol): {p_metrics.get('FEVI', 'N/A')}%\n- FC: {p_metrics.get('BPM', 'N/A')} lpm\n- Ritmo: {p_metrics.get('Ritmo', 'N/A')}\n- Strain Tisular Proxy: {p_metrics.get('Strain_Proxy', 'N/A')}\n- Colapso VCI Proxy: {p_metrics.get('Colapso_VCI', 'N/A')}%\n- Líneas B Pleurales: {'Positivo' if p_metrics.get('B_Lines') else 'Negativo'}\n- Doppler Activo: {'Sí' if p_metrics.get('Doppler') else 'No'}\n- Derrame Pericárdico Sugerido: {'Sí' if p_metrics.get('Derrame') else 'No'}\n"
                                
                            st.toast("🎥 Subiendo archivo de vídeo a Gemini Vision...")
                            v_file = genai.upload_file(path=tf_video.name)
                            while v_file.state.name == "PROCESSING":
                                time.sleep(2); v_file = genai.get_file(v_file.name)
                            con.append(v_file)

                        else: 
                            file.seek(0)
                            i_pil = ImageOps.exif_transpose(Image.open(file)).convert("RGB")
                            # Si es modo Autodetectar, NO aplicamos el filtro destructivo de ECG por si es una herida.
                            if "ECG" in modo:
                                img_aislada = aislar_trazado_ecg(i_pil)
                                con.extend([i_pil, img_aislada])
                            else: 
                                con.append(i_pil)
                            
                            if not imagen_para_visor: imagen_para_visor = i_pil

                # --- REGLAS DE AUTOPILOT (V111) ---
                instruccion_bbox = "INSTRUCCIÓN OBLIGATORIA: Si detectas lesión/fractura/isquemia/anomalía indica coordenadas: BBOX: [ymin, xmin, ymax, xmax] LABEL: Nombre."
                
                if modo == "✨ Autodetectar":
                    titulo_caja = "🛠️ DIAGNÓSTICO Y PLAN DE ACCIÓN"
                    instruccion_modo = 'Identifica visualmente de qué tipo de imagen médica se trata (ECG, Radiografía, Herida, Lesión dermatológica, Analítica, Ecografía, etc.). Asume automáticamente el rol del médico hiper-especialista en esa área exacta y analiza la imagen con máxima profundidad clínica.'
                elif "ECG" in modo:
                    titulo_caja = "💡 LECTURA ECG Y MANEJO"
                    instruccion_modo = 'ERES UN CARDIÓLOGO CLÍNICO. Analiza ritmo, eje, ondas y segmentos.'
                elif "POCUS" in modo:
                    titulo_caja = "🦇 ANÁLISIS POCUS AVANZADO"
                    instruccion_modo = 'ERES UN EXPERTO EN ECOGRAFÍA CLÍNICA. Analiza el vídeo y los resultados matemáticos extremos.'
                elif modo in ["🧴 Dermatología", "💀 RX/TAC/Resonancia", "🩹 Heridas / Úlceras"]:
                    titulo_caja = "🛠️ PLAN DE ACCIÓN"
                    instruccion_modo = 'Analiza el caso clínico y la imagen.'
                else:
                    titulo_caja = "🛠️ PLAN DE ACCIÓN"
                    instruccion_modo = 'Analiza el caso.'

                # --- REGLA DE ZONA ANATÓMICA ---
                if st.session_state.punto_cuerpo == "✨ Autodetectar":
                    instruccion_anatomia = "Deduce visualmente la zona anatómica o el encuadre de la imagen e indícalo en tu análisis."
                else:
                    instruccion_anatomia = f"ATENCIÓN: El usuario ha especificado manualmente que la zona anatómica es: {st.session_state.punto_cuerpo}. OBRIGA ESTA INDICACIÓN Y BASA TU ANÁLISIS EN ELLO."

                caja_enfermeria = '\n<details class="pocus-box" open><summary>👩‍⚕️ CUIDADOS DE ENFERMERÍA</summary><p>[Cuidados específicos]</p></details>' if contexto in ["Hospitalización", "Urgencias", "UCI"] else ""

                prompt = f"""
                Rol: Médico IA Avanzado (Autopilot). Contexto: {contexto}. Modo Seleccionado: {modo}.
                Usuario: "{notas}"
                Docs/Datos: {txt_docs[:10000]}
                
                REGLAS:
                1. EMPIEZA DIRECTAMENTE con <details>. Cero saludos o muletillas introductorias.
                2. {instruccion_modo}
                3. {instruccion_anatomia}
                4. {instruccion_bbox}
                5. Omite datos faltantes (no digas que faltan datos, simplemente analiza lo que tienes).
                6. DIAGNÓSTICO EN NEGRITA: En la primera frase usa <b>...</b>.

                FORMATO HTML REQUERIDO:
                <details class="diagnosis-box" open><summary>🚨 HALLAZGOS PRINCIPALES</summary><p><b>[Diagnóstico en negrita y zona anatómica detectada/confirmada]</b>. [Descripción detallada]</p></details>
                <details class="action-box" open><summary>⚡ ACCIÓN INMEDIATA</summary><p>[Explicación de manejo clínico]</p></details>
                <details class="{'pubmed-box' if 'PubMed' in modo else 'material-box'}" open><summary>{titulo_caja}</summary><p>[Desarrollo o Bibliografía]</p></details>{caja_enfermeria}
                """

                resp = model.generate_content([prompt, *con] if con else prompt)
                texto_generado = resp.text[resp.text.find("<details"):] if "<details" in resp.text else resp.text

                if imagen_para_visor:
                    img_marcada, texto_generado, detectado = extraer_y_dibujar_bboxes(texto_generado, imagen_para_visor)
                    st.session_state.img_marcada = img_marcada if detectado else imagen_para_visor

                st.session_state.resultado_analisis = texto_generado.strip()
                st.session_state.pdf_bytes = create_pdf(st.session_state.resultado_analisis)

            except Exception as e: st.error(f"Error: {e}")

    if st.session_state.resultado_analisis:
        st.markdown(st.session_state.resultado_analisis, unsafe_allow_html=True)
        if st.session_state.pdf_bytes: st.download_button("📥 Descargar PDF", st.session_state.pdf_bytes, "informe.pdf")
            
        st.markdown("---")
        st.markdown("### 💬 Chat Clínico")
        st.caption("Resuelve dudas específicas sobre este caso con la IA.")
        for msg in st.session_state.chat_messages:
            with st.chat_message(msg["role"]): st.markdown(msg["content"])

with col_right:
    visor_abierto = True if (st.session_state.get("img_marcada") or st.session_state.get("video_bytes")) else False
    with st.expander("👁️ Visor Visual / IA", expanded=visor_abierto):
        metrics = st.session_state.get("pocus_metrics", {})
        if metrics:
            st.markdown("### 🧮 Telemetría POCUS (V6)")
            
            c1, c2, c3 = st.columns(3)
            c1.metric("FEVI", f"{metrics.get('FEVI', 'N/A')}%")
            c2.metric("FC", f"{metrics.get('BPM', 'N/A')} bpm")
            c3.metric("Ritmo", metrics.get('Ritmo', 'N/A'))
            
            st.divider()
            
            c4, c5 = st.columns(2)
            c4.metric("Strain T. (Proxy)", f"{metrics.get('Strain_Proxy', 'N/A')}")
            c5.metric("Colapso VCI (Proxy)", f"{metrics.get('Colapso_VCI', 'N/A')}%")
            
            st.divider()
            
            c6, c7 = st.columns(2)
            c6.metric("Líneas B", "⚠️ Detectadas" if metrics.get('B_Lines') else "✅ Negativo")
            c7.metric("Derrame", "⚠️ Sí" if metrics.get('Derrame') else "✅ No")
            
            st.divider()

        if st.session_state.get("video_bytes"): st.video(st.session_state.video_bytes)
        
        if st.session_state.get("img_marcada"):
            st.markdown("#### 🎯 Visión / Detección IA")
            st.image(st.session_state.img_marcada, use_container_width=True)
        elif not st.session_state.get("video_bytes"):
            st.caption("Aquí aparecerá la imagen o el análisis visual.")

# ==========================================
# --- CHAT FLOTANTE GLOBAL ---
# ==========================================
if st.session_state.resultado_analisis:
    if user_query := st.chat_input("💬 Escribe tu duda clínica aquí..."):
        st.session_state.chat_messages.append({"role": "user", "content": user_query})
        try:
            genai.configure(api_key=st.session_state.api_key)
            chat_model = genai.GenerativeModel(st.session_state.modelo_seleccionado)
            ctx_chat = f"Eres el médico experto de guardia. Has generado este informe:\n{st.session_state.resultado_analisis}\nResponde a la duda:\n{user_query}"
            st.session_state.chat_messages.append({"role": "assistant", "content": chat_model.generate_content(ctx_chat).text})
        except Exception as e: st.session_state.chat_messages.append({"role": "assistant", "content": f"⚠️ Error: {e}"})
        st.rerun()
