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
import plotly.express as px

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
    details.pubmed-box { background-color: #e8eaf6; border-left: 6px solid #3f51b5; color: #1a237e; }
    
    div[data-testid="stChatInput"] {
        position: fixed !important; bottom: 0px !important; left: 0px !important;
        width: 100% !important; background-color: white !important;
        padding: 10px 20px 25px 20px !important; z-index: 9999 !important;
        box-shadow: 0px -4px 10px rgba(0,0,0,0.1) !important;
    }
</style>
""", unsafe_allow_html=True)

# --- CONFIGURACIÓN SEGURIDAD GOOGLE (ANTIBLOQUEO) ---
MEDICAL_SAFETY_SETTINGS = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"}
]

# --- GESTOR DE ESTADO ---
cookie_manager = stx.CookieManager()
time.sleep(0.1)

for key in ["autenticado", "api_key", "resultado_analisis", "pdf_bytes", "chat_messages", "img_marcada", "video_bytes", "modelos_disponibles", "last_video_path"]:
    if key not in st.session_state: st.session_state[key] = [] if key in ["chat_messages", "modelos_disponibles"] else None
if "autenticado" not in st.session_state or not st.session_state.autenticado:
    st.session_state.autenticado = False
if "pocus_metrics" not in st.session_state: st.session_state.pocus_metrics = {}
if "sam_metrics" not in st.session_state: st.session_state.sam_metrics = {}

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
#      ANONIMIZACIÓN RGPD (SELECTIVA)
# ==========================================
def anonimizar_imagen(img_pil, modo):
    try:
        img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        
        if modo not in ["🩹 Heridas / Úlceras", "🧴 Dermatología"]:
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            rostros = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)
            for (x, y, w, h) in rostros:
                roi = img_cv[y:y+h, x:x+w]
                roi_blur = cv2.GaussianBlur(roi, (99, 99), 30)
                img_cv[y:y+h, x:x+w] = roi_blur
        
        try:
            import pytesseract
            datos = pytesseract.image_to_data(gray, output_type=pytesseract.Output.DICT)
            for i in range(len(datos['text'])):
                if int(datos['conf'][i]) > 60: 
                    texto = datos['text'][i].strip()
                    if len(texto) >= 4:
                        x, y, w, h = datos['left'][i], datos['top'][i], datos['width'][i], datos['height'][i]
                        roi = img_cv[y:y+h, x:x+w]
                        roi_blur = cv2.GaussianBlur(roi, (15, 15), 10)
                        img_cv[y:y+h, x:x+w] = roi_blur
        except ImportError:
            pass
            
        return Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
    except Exception as e:
        print(f"Error RGPD Anonymizer: {e}")
        return img_pil

# ==========================================
#      CACHÉ PARA MOTOR MOBILE SAM 1
# ==========================================
@st.cache_resource
def load_sam_model():
    try:
        from ultralytics import SAM
        if os.path.exists('mobile_sam.pt'):
            return SAM('mobile_sam.pt')
    except Exception as e:
        print(f"Fallo al cargar SAM en caché: {e}")
    return None

def segmentar_herida_sam_v2(img_pil):
    try:
        model_sam = load_sam_model()
        if model_sam is None: return None, 0
        
        img_cv = cv2.cvtColor(np.array(img_pil.convert('RGB')), cv2.COLOR_RGB2BGR)
        h, w = img_cv.shape[:2]
        puntos = [[w//2, h//2]]
        results = model_sam.predict(img_cv, points=puntos, labels=[1], verbose=False)
        
        if results and len(results)>0 and results[0].masks is not None:
            mask = results[0].masks.data[0].cpu().numpy()
            mask = cv2.resize(mask, (w, h))
            overlay = img_cv.copy()
            overlay[mask > 0.5] = overlay[mask > 0.5] * 0.5 + np.array([255, 150, 0]) * 0.5 
            contours, _ = cv2.findContours((mask > 0.5).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay, contours, -1, (0, 255, 0), 3)
            return Image.fromarray(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)), int(np.sum(mask > 0.5))
    except Exception as e: print(f"Error SAM Segmentación: {e}")
    return None, 0

# ==========================================
#      MOTOR POCUS GOD MODE V6
# ==========================================
def procesar_pocus_v6_singularidad(video_path):
    try:
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        ret, frame1 = cap.read()
        if not ret: return None, None, None, {}

        h_orig, w_orig = frame1.shape[:2]
        scale = 320.0 / w_orig 
        new_w, new_h = 320, int(h_orig * scale)
        frame1 = cv2.resize(frame1, (new_w, new_h))
        gray_init = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        
        _, beam_t = cv2.threshold(gray_init, 15, 255, cv2.THRESH_BINARY)
        cnts, _ = cv2.findContours(beam_t, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        roi_mask = np.zeros_like(gray_init)
        if cnts: cv2.drawContours(roi_mask, [max(cnts, key=cv2.contourArea)], -1, 255, -1)
        else: cv2.ellipse(roi_mask, (new_w//2, new_h//2), (int(new_w*0.4), int(new_h*0.4)), 0, 0, 360, 255, -1)

        volumes = []; m_mode = []; frame_count = 0
        best_frame = frame1.copy()
        
        while True:
            ret, f = cap.read()
            if not ret: break
            frame_count += 1
            if frame_count % max(1, int(fps // 15)) != 0: continue
            f = cv2.resize(f, (new_w, new_h))
            g = cv2.medianBlur(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), 5)
            masked = cv2.bitwise_and(g, g, mask=roi_mask)
            _, th = cv2.threshold(masked, 35, 255, cv2.THRESH_BINARY_INV)
            th = cv2.bitwise_and(th, th, mask=roi_mask)
            c_pocus, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if c_pocus:
                best_c = max(c_pocus, key=cv2.contourArea)
                if cv2.contourArea(best_c) > (new_w*new_h*0.03):
                    x, y, wb, hb = cv2.boundingRect(best_c)
                    vol = (cv2.contourArea(best_c)**2)/hb if hb>0 else 0
                    volumes.append(vol)
                    m_mode.append(g[:, x+(wb//2)])
                    if vol >= max(volumes):
                        best_frame = f.copy()
                        cv2.drawContours(best_frame, [best_c], -1, (0, 255, 0), 2)
        
        cap.release()
        metrics = {}
        if len(volumes) > 5:
            v_arr = np.array(volumes)
            edv, esv = np.percentile(v_arr, 95), np.percentile(v_arr, 5)
            metrics['FEVI'] = round(max(10.0, min(((edv-esv)/edv)*100, 85.0)), 1) if edv>0 else "N/A"
        
        m_mode_pil = None
        if m_mode:
            mm_img = cv2.resize(np.transpose(np.array(m_mode)), (new_w, new_h))
            m_mode_pil = Image.fromarray(cv2.applyColorMap(mm_img, cv2.COLORMAP_OCEAN))
            
        return m_mode_pil, Image.fromarray(cv2.cvtColor(best_frame, cv2.COLOR_BGR2RGB)), metrics
    except: return None, None, {}

# ==========================================
#      EXTRACTOR DE FOTOGRAMAS (VÍDEO)
# ==========================================
def extraer_frame_video(video_path, texto):
    patron_frame = r'FRAME:\s*\[([\d\.]+)\]'
    match = re.search(patron_frame, texto)
    if not match or not video_path: return None, texto
    
    try:
        segundo = float(match.group(1))
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_MSEC, segundo * 1000)
        ret, frame = cap.read()
        cap.release()
        
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(frame_rgb)
            texto_limpio = re.sub(patron_frame, '', texto).strip()
            return img_pil, texto_limpio
    except Exception as e: print(f"Error Extrayendo Frame: {e}")
    return None, texto

# ==========================================
#      FUNCIONES IA Y AUXILIARES
# ==========================================
def buscar_en_pubmed(query, max_results=10):
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

def extraer_y_dibujar_bboxes(texto, img_pil):
    patron = r'BBOX:\s*\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]\s*LABEL:\s*([^\n<]+)'
    matches = re.findall(patron, texto)
    if not matches: return None, texto, False
    
    m = matches[0] 
    img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    h, w = img_cv.shape[:2]
    
    try:
        y1, x1, y2, x2 = [int(int(c)*dim/1000) for c, dim in zip(m[:4], [h, w, h, w])]
        cv2.rectangle(img_cv, (x1, y1), (x2, y2), (0, 0, 255), 3)
        cv2.putText(img_cv, m[4].upper(), (x1, max(0, y1-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
    except: pass
        
    texto_limpio = re.sub(patron, '', texto).strip()
    return Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)), texto_limpio, True

def aislar_trazado_ecg(pil_image):
    try:
        img_cv = cv2.cvtColor(np.array(pil_image.convert('RGB')), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 15)
        smooth = cv2.GaussianBlur(thresh, (3, 3), 0)
        return Image.fromarray(cv2.cvtColor(smooth, cv2.COLOR_GRAY2RGB))
    except: return pil_image

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

# --- INTERFAZ PRINCIPAL ---
st.title("🩺 LabMind")
col_l, col_c, col_r = st.columns([1, 2, 1])

with col_l:
    st.subheader("⚙️ Motor IA")
    try:
        genai.configure(api_key=st.session_state.api_key)
        if not st.session_state.modelos_disponibles:
            st.session_state.modelos_disponibles = [m.name.replace('models/','') for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
    except: st.session_state.modelos_disponibles = ["gemini-1.5-flash"]
    
    idx_defecto = 0
    if st.session_state.modelos_disponibles:
        for i, modelo in enumerate(st.session_state.modelos_disponibles):
            if "gemini-3" in modelo.lower() and "flash" in modelo.lower() and "preview" in modelo.lower():
                idx_defecto = i
                break
        else:
            for i, modelo in enumerate(st.session_state.modelos_disponibles):
                if "gemini-3" in modelo.lower() and "flash" in modelo.lower():
                    idx_defecto = i
                    break
                
    st.session_state.modelo_seleccionado = st.selectbox("Versión de Gemini:", st.session_state.modelos_disponibles, index=idx_defecto)
    
    lista_anatomia = [
        "✨ Autodetectar", "Cabeza", "Cara", "Cuello", "Tórax", "Espalda", 
        "Abdomen", "Genitales", "Glúteo", "Sacro", "Brazo", "Mano", 
        "Muslo", "Pierna", "talón", "Pie"
    ]
    st.session_state.punto_cuerpo = st.selectbox("Anatomía:", lista_anatomia)

with col_c:
    st.subheader("1. Selección de Modo")
    lista_modos = [
        "✨ Autodetectar", 
        "🧠 Medicina Interna (Holístico)", 
        "🩸 Analíticas (God Mode)", 
        "🩹 Heridas / Úlceras", 
        "🦇 Ecografía / POCUS", 
        "📚 Agente Investigador (PubMed)", 
        "📈 ECG", 
        "💀 RX/TAC", 
        "🧴 Dermatología"
    ]
    modo = st.selectbox("Especialidad:", lista_modos)
    contexto = st.selectbox("🏥 Contexto:", ["Urgencias", "Hospitalización", "UCI", "Residencia", "Domicilio"], index=1)
    
    archivos = []; audio_val = None; fs = None; cam_pic = None
    notas = ""
    query_pubmed = ""
    
    if modo == "📚 Agente Investigador (PubMed)":
        st.info("🤖 **Agente Clínico y Farmacológico**")
        
        if hasattr(st, "audio_input"):
            audio_val = st.audio_input("🎙️ Dictar duda clínica")
        else:
            st.warning("⚠️ Tu servidor necesita actualizar Streamlit para usar la grabadora de voz.")
            
        with st.expander("📝 Búsqueda Avanzada (Opcional)", expanded=False):
            query_pubmed = st.text_input("🔍 Buscar en PubMed (Ej: collagenase silver):")
            notas = st.text_area("Notas extra:", height=70, placeholder="Contexto del paciente...", label_visibility="collapsed")
    else:
        metodo_captura = st.radio("Método de entrada", ["📁 Subir Archivos", "📸 Tomar Foto"], horizontal=True, label_visibility="collapsed")
        
        if metodo_captura == "📁 Subir Archivos":
            fs = st.file_uploader("Archivos Clínicos:", type=['jpg','png','pdf','mp4','mov'], accept_multiple_files=True)
            st.caption("📱 *En móviles, presiona arriba para grabar vídeo directamente.*")
        elif metodo_captura == "📸 Tomar Foto":
            cam_pic = st.camera_input("Cámara")
            
        if fs:
            for f in fs:
                if f.type.startswith('video') or f.name.lower().endswith(('mp4', 'mov')): archivos.append(("video", f))
                elif "pdf" in f.type: archivos.append(("doc", f))
                else: archivos.append(("img", f))
        
        if cam_pic:
            archivos.append(("img", cam_pic))
                
        with st.expander("📝 Notas Clínicas / Preguntas", expanded=False):
            notas = st.text_area("Notas", height=70, placeholder="Escribe el contexto del paciente...", label_visibility="collapsed")
            
        with st.expander("🎙️ Adjuntar Nota de Voz", expanded=False):
            if hasattr(st, "audio_input"):
                audio_val = st.audio_input("Dictar notas", key="mic", label_visibility="collapsed")
            else:
                st.warning("⚠️ Tu servidor necesita actualizar Streamlit para usar la nueva grabadora de voz.")
    
    c1, c2 = st.columns([3, 1])
    with c1: 
        btn_analizar = st.button("🚀 ANALIZAR", type="primary", use_container_width=True)
    with c2:
        if st.button("🔄 NUEVO", type="secondary", use_container_width=True):
            for k in ["resultado_analisis", "img_marcada", "video_bytes", "chat_messages", "last_video_path"]: 
                st.session_state[k] = None if ("img" in k or "video" in k or "resultado" in k) else []
            st.session_state.pocus_metrics = {}
            st.session_state.sam_metrics = {}
            st.rerun()
    
    if btn_analizar:
        st.session_state.resultado_analisis = None
        st.session_state.img_marcada = None
        st.session_state.video_bytes = None
        st.session_state.last_video_path = None
        st.session_state.pocus_metrics = {}
        st.session_state.sam_metrics = {}
        st.session_state.chat_messages = []
        
        with st.spinner("Procesando datos (Verificación cruzada activada)..."):
            try:
                model = genai.GenerativeModel(st.session_state.modelo_seleccionado)
                con = []
                txt_docs = ""
                sam_utilizado = False
                imagen_para_visor = None
                video_presente = False
                
                if modo == "📚 Agente Investigador (PubMed)":
                    q_val = locals().get('query_pubmed', '')
                    if q_val:
                        txt_docs += buscar_en_pubmed(q_val)
                    
                if audio_val:
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tf_audio:
                        tf_audio.write(audio_val.read())
                    con.append(genai.upload_file(path=tf_audio.name))
                
                for tipo, f in archivos:
                    if tipo == "video":
                        video_presente = True
                        st.session_state.video_bytes = f.read()
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tf:
                            tf.write(st.session_state.video_bytes)
                            vp = tf.name
                            st.session_state.last_video_path = vp
                            
                        if "POCUS" in modo:
                            st.toast("Activando God Mode POCUS...")
                            mm, eb, met = procesar_pocus_v6_singularidad(vp)
                            st.session_state.pocus_metrics = met
                            if not st.session_state.img_marcada:
                                st.session_state.img_marcada = mm if mm else eb
                            txt_docs += f"\n[POCUS] FEVI Estimada: {met.get('FEVI','N/A')}%\n"
                            
                        st.toast("Subiendo vídeo a la IA...")
                        v_file = genai.upload_file(vp)
                        while v_file.state.name == "PROCESSING": 
                            time.sleep(2)
                            v_file = genai.get_file(v_file.name)
                        con.append(v_file)
                        
                    elif tipo == "doc":
                        txt_docs += "".join([p.extract_text() for p in pypdf.PdfReader(f).pages])
                    elif tipo == "img":
                        img_raw = ImageOps.exif_transpose(Image.open(f)).convert("RGB")
                        img = anonimizar_imagen(img_raw, modo)
                        
                        if modo in ["🩹 Heridas / Úlceras", "🧴 Dermatología"]:
                            sam_res, a_px = segmentar_herida_sam_v2(img)
                            if a_px > 100: 
                                st.session_state.img_marcada = sam_res
                                st.session_state.sam_metrics = {'Area_PX': a_px}
                                sam_utilizado = True
                                txt_docs += f"\n[SAM] Área de lesión aislada: {a_px} píxeles.\n"
                        
                        if "ECG" in modo:
                            img_aislada = aislar_trazado_ecg(img)
                            con.extend([img, img_aislada])
                        else:
                            con.append(img)
                        if not imagen_para_visor: imagen_para_visor = img

                # --- CONFIGURACIÓN DE BBOX ---
                if sam_utilizado:
                    instruccion_bbox = "La imagen ya ha sido segmentada milimétricamente. NO devuelvas BBOX."
                elif modo in ["🩸 Analíticas (God Mode)", "📚 Agente Investigador (PubMed)"]:
                    instruccion_bbox = "INSTRUCCIÓN: Estás en modo de análisis de texto o investigación. NO devuelvas ni calcules coordenadas BBOX bajo ningún concepto."
                else:
                    if video_presente:
                        instruccion_bbox = "INSTRUCCIÓN DE RADIOLOGÍA DINÁMICA: Estás analizando un VÍDEO. Si detectas patología, busca el fotograma donde se vea más clara. Al final de tu texto, imprime UNA SOLA VEZ el segundo exacto y sus coordenadas así: FRAME: [segundos] BBOX: [ymin, xmin, ymax, xmax] LABEL: TuTexto."
                    else:
                        instruccion_bbox = "INSTRUCCIÓN DE ANCLAJE ESPACIAL: Si detectas patología, usa tu anclaje visual nativo para marcarla. Imprime esto UNA SOLA VEZ al final: BBOX: [ymin, xmin, ymax, xmax] LABEL: TuTexto."

                if st.session_state.punto_cuerpo == "✨ Autodetectar":
                    instruccion_anatomia = "Deduce visualmente la zona anatómica."
                else:
                    instruccion_anatomia = f"El usuario especifica que la zona es: {st.session_state.punto_cuerpo}. Basa tu análisis en ello."

                # --- INSTRUCCIONES ESPECÍFICAS Y HTML V147 ---
                instrucciones_especificas = ""
                html_requerido = ""
                
                if modo == "🩹 Heridas / Úlceras":
                    try:
                        if os.path.exists("protocolo.jpg"):
                            img_protocolo = Image.open("protocolo.jpg").convert("RGB")
                            con.append(img_protocolo) 
                            st.toast("📜 Protocolo de San Eloy adjuntado a la IA.")
                            instrucciones_especificas = "- INSTRUCCIÓN DE CURAS: Te he adjuntado una foto del 'Protocolo de San Eloy'. PRIORIDAD MÁXIMA: Si vas a recomendar un apósito, usa EXACTAMENTE los nombres de esa imagen. FLEXIBILIDAD: Si consideras que necesita algo que no está ahí, tienes libertad para añadirlo."
                    except: pass
                    
                    html_requerido = """
<details class="diagnosis-box" open>
<summary>🚨 HALLAZGOS Y RAZONAMIENTO</summary>
<p><b>[Diagnóstico]</b> [Certeza: XX%]. [Tu análisis]</p>
</details>

<details class="action-box" open>
<summary>⚡ ACCIÓN INMEDIATA</summary>
<p>[Plan médico]</p>
</details>

<details class="material-box" open>
<summary>🩹 CURA Y CUIDADOS</summary>
<p>[1. PAUTA DE CURA: Prioriza los materiales del Protocolo de San Eloy adjunto. 2. CUIDADOS DE ENFERMERÍA: Posición, monitorización, etc.]</p>
</details>
"""
                elif modo == "🩸 Analíticas (God Mode)":
                    instrucciones_especificas = "- INSTRUCCIÓN ULTRA GOD MODE ANALÍTICAS: Eres un experto intensivista y bioquímico clínico. 1. Busca patrones ocultos. 2. Si dispones de los datos numéricos, CALCULA OBLIGATORIAMENTE y muestra: Anion Gap, Gap Osmolar, Calcio corregido por albúmina y filtrado glomerular estimado (CKD-EPI). 3. Identifica el trastorno ácido-base primario y las compensaciones esperadas. 4. Advierte explícitamente sobre posibles errores pre-analíticos."
                    html_requerido = """
<details class="diagnosis-box" open>
<summary>🩸 ANÁLISIS BIOQUÍMICO Y PATRÓN</summary>
<p><b>[Patrón Principal Detectado]</b> [Certeza: XX%]. [Análisis detallado incluyendo cálculos obligatorios]</p>
</details>

<details class="action-box" open>
<summary>⚡ RIESGOS INMINENTES Y ACCIÓN</summary>
<p>[Estratificación del riesgo vital, alertas de errores preanalíticos y manejo inmediato]</p>
</details>

<details class="material-box" open>
<summary>🛠️ TRATAMIENTO MÉDICO</summary>
<p>[Manejo farmacológico específico o ajustes sugeridos (fluidoterapia, electrolitos)]</p>
</details>

<details class="pocus-box" open>
<summary>👩‍⚕️ CUIDADOS Y MONITORIZACIÓN</summary>
<p>[Plan de cuidados específicos para enfermería y monitorización hemodinámica]</p>
</details>
"""
                elif modo == "🧠 Medicina Interna (Holístico)":
                    instrucciones_especificas = "- INSTRUCCIÓN ULTRA GOD MODE INTERNA: Actúa como Jefe de Servicio de Medicina Interna de un hospital terciario. Tienes una visión holística. 1. SÍNTESIS: Cruza TODOS los datos. 2. NAVAJA DE OCKHAM: Busca y prioriza un diagnóstico principal y unificador. 3. DICTUM DE HICKAM: Propón un diagnóstico diferencial riguroso. 4. ESTRATIFICACIÓN VITAL: Define el nivel de gravedad."
                    html_requerido = """
<details class="diagnosis-box" open>
<summary>🧠 DIAGNÓSTICO SINDRÓMICO INTEGRAL</summary>
<p><b>[Diagnóstico Unificador Principal]</b> [Certeza: XX%]. [Síntesis holística cruzando todas las pruebas aportadas]</p>
</details>

<details class="action-box" open>
<summary>⚡ DIAGNÓSTICO DIFERENCIAL Y TRIAGE</summary>
<p>[Dictum de Hickam: Patologías concurrentes a descartar. Estratificación de gravedad y destino ideal]</p>
</details>

<details class="material-box" open>
<summary>🛠️ MANEJO TERAPÉUTICO GLOBAL</summary>
<p>[Tratamiento médico integral abordando la causa raíz]</p>
</details>

<details class="pocus-box" open>
<summary>👩‍⚕️ PLAN DE CUIDADOS (ENFERMERÍA)</summary>
<p>[Cuidados a pie de cama, monitorización holística y prevención de complicaciones]</p>
</details>
"""
                elif modo == "📚 Agente Investigador (PubMed)":
                    instrucciones_especificas = """- INSTRUCCIÓN AGENTE CLÍNICO: Eres un experto farmacólogo e investigador. ESCUCHA ATENTAMENTE EL AUDIO ADJUNTO (si lo hay) y lee los datos de PubMed.
REGLA DE ORO DE TRANSPARENCIA Y ENLACES HTML:
1. Si usas artículos con números PMID, OBLIGATORIAMENTE cítalos usando esta estructura HTML: <a href="https://pubmed.ncbi.nlm.nih.gov/AQUI_EL_NUMERO_PMID/" target="_blank">PMID: AQUI_EL_NUMERO_PMID</a>.
2. Si los "Datos" están vacíos, busca en tu memoria interna evidencia de OTRAS fuentes (Cochrane, UpToDate, guías). Inicia la respuesta con: "⚠️ <b>Búsqueda en PubMed sin resultados. Evidencia rescatada de otras fuentes.</b>" e incluye un enlace HTML clicable a la web de la organización.
3. Como ÚLTIMO RECURSO, inicia con: "⚠️ <b>No existe evidencia indexada clara. Respuesta basada en principios fisiopatológicos.</b>" """
                    html_requerido = """
<details class="pubmed-box" open>
<summary>📚 RESPUESTA CLÍNICA Y EVIDENCIA</summary>
<p><b>[Conclusión Directa]</b> [Certeza: XX%]. [Aplica obligatoriamente la advertencia ⚠️ si no se usó PubMed. Luego da tu respuesta clara.]</p>
</details>

<details class="radiomics-box" open>
<summary>🔬 FARMACOLOGÍA Y ESTUDIOS (REFERENCIAS)</summary>
<p>[Explicación científica profunda. Pon AQUÍ la lista de referencias con sus enlaces HTML clicables obligatorios.]</p>
</details>

<details class="action-box" open>
<summary>⚖️ RECOMENDACIÓN PRÁCTICA (CUIDADOS)</summary>
<p>[Cómo aplicar esto en el paciente a pie de cama: precauciones y advertencias de seguridad]</p>
</details>
"""
                else:
                    html_requerido = f"""
<details class="diagnosis-box" open>
<summary>🚨 HALLAZGOS Y RAZONAMIENTO</summary>
<p><b>[Diagnóstico]</b> [Certeza: XX%]. [Tu análisis]</p>
</details>

<details class="action-box" open>
<summary>⚡ ACCIÓN INMEDIATA</summary>
<p>[Plan médico]</p>
</details>

<details class="{'pubmed-box' if 'PubMed' in modo else 'material-box'}" open>
<summary>🛠️ TRATAMIENTO Y SEGUIMIENTO</summary>
<p>[Desarrollo de tratamiento a largo plazo]</p>
</details>

<details class="pocus-box" open>
<summary>👩‍⚕️ CUIDADOS DE ENFERMERÍA</summary>
<p>[Plan de cuidados específicos]</p>
</details>
"""

                # --- FIX V147: DIRECTRIZ SUPREMA (CADENA DE VERIFICACIÓN) ---
                prompt = f"""
                DIRECTRIZ SUPREMA (PROTOCOLOS DE SEGURIDAD DEL PACIENTE):
                Eres LabMind, una IA de grado médico estricto. Tu prioridad absoluta es NO INVENTAR DATOS (Cero Alucinaciones). Eres un auditor clínico implacable.

                Contexto: {contexto}. Especialidad: {modo}.
                Usuario (Notas): "{notas}"
                Datos Aportados: {txt_docs[:15000]}

                CADENA DE VERIFICACIÓN OBLIGATORIA (CoVe):
                Antes de escribir tu respuesta final, debes procesar mentalmente estos pasos:
                1. Analizar evidencias.
                2. Formular hipótesis.
                3. AUTOCRÍTICA DE RED TEAM: Cuestiona tu propia hipótesis. ¿Faltan datos? ¿La imagen es borrosa? ¿Asumiste un valor no escrito?
                4. CÁLCULO DE CERTEZA: Asigna un porcentaje real de fiabilidad a tu respuesta (0% a 100%). Si la imagen es mala o faltan datos, el porcentaje debe ser inferior al 50%.
                
                REGLAS EXTRA (SEGURIDAD MÁXIMA):
                - {instruccion_bbox}
                - {instruccion_anatomia}
                - ANCLAJE DE DATOS ESTRICTO: Para describir al paciente o emitir juicios, básate ÚNICA Y EXCLUSIVAMENTE en los Datos, Imágenes y Notas aportadas.
                - CLÁUSULA DE IGNORANCIA: Si la imagen es borrosa o los datos son insuficientes para una conclusión segura, dilo explícitamente y baja tu % de Certeza. NO inventes hallazgos.
                {instrucciones_especificas}

                INSTRUCCIÓN DE FORMATO MUY ESTRICTA:
                Debes responder ÚNICA y EXCLUSIVAMENTE copiando el siguiente bloque HTML y rellenando los corchetes. Reemplaza [Certeza: XX%] por tu cálculo numérico. NO uses Markdown como ```html. 

                {html_requerido}
                """
                
                # --- FIX V147: TEMPERATURA 0.0 PARA RESPUESTAS DETERMINISTAS MATEMÁTICAS ---
                res = model.generate_content(
                    [prompt, *con] if con else prompt, 
                    safety_settings=MEDICAL_SAFETY_SETTINGS,
                    generation_config={"temperature": 0.0, "top_p": 0.8, "top_k": 10}
                )
                
                raw_txt = res.text.replace("```html", "").replace("```", "").strip()
                raw_txt = raw_txt[raw_txt.find("<details"):] if "<details" in raw_txt else raw_txt
                
                img_base_para_bbox = imagen_para_visor
                
                if video_presente and st.session_state.get("last_video_path") and "FRAME:" in raw_txt:
                    st.toast("🎞️ Buscando el fotograma exacto en el vídeo...")
                    frame_extraido, raw_txt = extraer_frame_video(st.session_state.last_video_path, raw_txt)
                    if frame_extraido:
                        img_base_para_bbox = anonimizar_imagen(frame_extraido, modo)
                
                if img_base_para_bbox and not sam_utilizado:
                    im_m, clean_t, det = extraer_y_dibujar_bboxes(raw_txt, img_base_para_bbox)
                    if det:
                        st.session_state.img_marcada = im_m
                    elif not st.session_state.img_marcada:
                        st.session_state.img_marcada = img_base_para_bbox
                    raw_txt = clean_t

                st.session_state.resultado_analisis = raw_txt
                st.session_state.pdf_bytes = create_pdf(st.session_state.resultado_analisis)
            except Exception as e: 
                st.error(f"Error procesando el análisis: {e}")

    if st.session_state.resultado_analisis:
        st.markdown(st.session_state.resultado_analisis, unsafe_allow_html=True)
        if st.session_state.pdf_bytes: st.download_button("📥 Descargar PDF", st.session_state.pdf_bytes, "informe.pdf")

with col_r:
    abierto = True if (st.session_state.get("img_marcada") or st.session_state.get("video_bytes")) else False
    with st.expander("👁️ Dashboard Visual", expanded=abierto):
        if st.session_state.pocus_metrics:
            st.markdown("### 🧮 Telemetría POCUS")
            st.metric("FEVI (Proxy Vol)", f"{st.session_state.pocus_metrics.get('FEVI', 'N/A')}%")
            st.divider()
        if st.session_state.sam_metrics:
            st.markdown("### 🦠 Planimetría SAM")
            st.metric("Área de Lesión", f"{st.session_state.sam_metrics.get('Area_PX', 0)} px")
            st.divider()
            
        if st.session_state.get("video_bytes"): 
            st.video(st.session_state.video_bytes)
            
        if st.session_state.get("img_marcada"):
            st.markdown("#### 🎯 Fotograma / Visión IA")
            try:
                fig = px.imshow(st.session_state.img_marcada)
                fig.update_layout(coloraxis_showscale=False, margin=dict(l=0, r=0, t=0, b=0), xaxis_visible=False, yaxis_visible=False)
                st.plotly_chart(fig, use_container_width=True, config={'scrollZoom': True, 'displayModeBar': True})
            except Exception as e:
                st.image(st.session_state.img_marcada, use_container_width=True)

# ==========================================
# --- CHAT FLOTANTE ---
# ==========================================
if st.session_state.resultado_analisis:
    st.divider()
    st.markdown("### 💬 Chat Interactivo IA")
    
    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if query := st.chat_input("Duda clínica sobre este paciente o investigación..."):
        st.session_state.chat_messages.append({"role": "user", "content": query})
        
        try:
            chat_model = genai.GenerativeModel(st.session_state.modelo_seleccionado)
            historial = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.chat_messages[:-1]])
            ctx_chat = f"DIRECTRIZ SUPREMA: Eres una IA médica estricta. Responde sin inventar datos.\n\nInforme clínico base:\n{st.session_state.resultado_analisis}\n\nHistorial de conversación:\n{historial}\n\nResponde a esta nueva duda del médico/enfermero de forma concisa y estricta: {query}"
            
            resp = chat_model.generate_content(
                ctx_chat, 
                safety_settings=MEDICAL_SAFETY_SETTINGS,
                generation_config={"temperature": 0.0, "top_p": 0.8, "top_k": 10} 
            )
            
            st.session_state.chat_messages.append({"role": "assistant", "content": resp.text})
            
        except Exception as e:
            st.session_state.chat_messages.append({"role": "assistant", "content": f"Error del servidor: {e}"})
            
        st.rerun()
