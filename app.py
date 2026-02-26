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
import base64
import io
import requests
from openai import OpenAI

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

# --- CONFIGURACIÓN SEGURIDAD GOOGLE ---
MEDICAL_SAFETY_SETTINGS = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"}
]

# --- GESTOR DE ESTADO ---
cookie_manager = stx.CookieManager()
time.sleep(0.1)

for key in ["autenticado", "api_key", "api_key_or", "resultado_analisis", "pdf_bytes", "chat_messages", "img_marcada", "video_bytes", "modelos_disponibles", "modelos_or", "last_video_path"]:
    if key not in st.session_state: st.session_state[key] = [] if key in ["chat_messages", "modelos_disponibles", "modelos_or"] else None
if "autenticado" not in st.session_state or not st.session_state.autenticado:
    st.session_state.autenticado = False
if "pocus_metrics" not in st.session_state: st.session_state.pocus_metrics = {}
if "sam_metrics" not in st.session_state: st.session_state.sam_metrics = {}

cookie_api_key = cookie_manager.get(cookie="labmind_secret_key")
cookie_api_key_or = cookie_manager.get(cookie="labmind_or_key")

if not st.session_state.autenticado:
    if cookie_api_key:
        st.session_state.api_key = cookie_api_key
        if cookie_api_key_or:
            st.session_state.api_key_or = cookie_api_key_or
        st.session_state.autenticado = True
        st.rerun()
    else:
        st.title("LabMind Acceso")
        k = st.text_input("API Key Google (Gemini) [Obligatorio]:", type="password")
        k_or = st.text_input("API Key OpenRouter (Llama Vision) [Opcional - GRATIS]:", type="password")
        if st.button("Entrar", type="primary"):
            if k:
                expires = datetime.datetime.now() + datetime.timedelta(days=30)
                cookie_manager.set("labmind_secret_key", k, expires_at=expires, key="set_cookie_gemini")
                st.session_state.api_key = k
                if k_or:
                    cookie_manager.set("labmind_or_key", k_or, expires_at=expires, key="set_cookie_or")
                    st.session_state.api_key_or = k_or
                st.session_state.autenticado = True
                time.sleep(0.5)
                st.rerun()
            else:
                st.error("La API Key de Google es obligatoria.")
        st.stop()

# ==========================================
#      FUNCIONES DE UTILIDAD Y BASE64
# ==========================================
def image_to_base64(img_pil):
    img_resized = img_pil.copy()
    img_resized.thumbnail((1024, 1024))
    buffered = io.BytesIO()
    img_resized.convert('RGB').save(buffered, format="JPEG", quality=85)
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

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
        return Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
    except:
        return img_pil

# ==========================================
#      CACHÉ PARA MOTOR MOBILE SAM 1
# ==========================================
@st.cache_resource
def load_sam_model():
    try:
        from ultralytics import SAM
        return SAM('mobile_sam.pt')
    except:
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
    except: pass
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
                    if vol >= max(volumes): best_frame = f.copy()
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
#      FUNCIONES IA Y AUXILIARES
# ==========================================
def buscar_en_pubmed(query, max_results=10):
    try:
        base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
        search_url = f"{base_url}esearch.fcgi?db=pubmed&term={urllib.parse.quote(query)}&retmode=json&retmax={max_results}&sort=relevance"
        req = urllib.request.Request(search_url, headers={'User-Agent': 'Mozilla/5.0'})
        data = json.loads(urllib.request.urlopen(req).read().decode('utf-8'))
        ids = data.get('esearchresult', {}).get('idlist', [])
        if not ids: return "No se encontraron artículos."
        fetch_url = f"{base_url}efetch.fcgi?db=pubmed&id={','.join(ids)}&retmode=xml"
        root = ET.fromstring(urllib.request.urlopen(urllib.request.Request(fetch_url, headers={'User-Agent': 'Mozilla/5.0'})).read())
        resultados = ""
        for article in root.findall('.//PubmedArticle'):
            pmid = article.find('.//PMID').text
            title = article.find('.//ArticleTitle').text if article.find('.//ArticleTitle') is not None else "Sin título"
            resultados += f"PMID: {pmid}\nTÍTULO: {title}\n\n"
        return resultados
    except: return "Error en PubMed."

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

def create_pdf(texto):
    pdf = FPDF(); pdf.add_page(); pdf.set_font("helvetica", size=10)
    pdf.cell(0, 10, f"Fecha: {datetime.datetime.now().strftime('%d/%m/%Y %H:%M')}", new_x="LMARGIN", new_y="NEXT")
    clean = re.sub(r'<[^>]+>', '', texto).replace('€','EUR')
    pdf.multi_cell(0, 5, txt="".join(c for c in clean if ord(c) < 256))
    return bytes(pdf.output())

# --- INTERFAZ PRINCIPAL ---
st.title("🩺 LabMind")
col_l, col_c, col_r = st.columns([1, 2, 1])

with col_l:
    st.subheader("⚙️ Motor IA Principal")
    try:
        genai.configure(api_key=st.session_state.api_key)
        if not st.session_state.modelos_disponibles:
            st.session_state.modelos_disponibles = [m.name.replace('models/','') for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
    except: st.session_state.modelos_disponibles = ["gemini-1.5-flash"]
    
    # --- BUSCAR GEMINI 3 FLASH PREVIEW POR DEFECTO ---
    idx_defecto_gemini = 0
    if st.session_state.modelos_disponibles:
        for i, mod in enumerate(st.session_state.modelos_disponibles):
            if "gemini-3-flash-preview" in mod.lower():
                idx_defecto_gemini = i
                break

    st.session_state.modelo_seleccionado = st.selectbox("Versión de Gemini:", st.session_state.modelos_disponibles, index=idx_defecto_gemini)
    
    st.divider()
    st.subheader("🕵️‍♂️ Motor IA Auditor")
    if st.session_state.get("api_key_or"):
        try:
            if not st.session_state.modelos_or:
                response = requests.get("https://openrouter.ai/api/v1/models", timeout=5)
                if response.status_code == 200:
                    data = response.json().get("data", [])
                    st.session_state.modelos_or = sorted([m["id"] for m in data if "free" in m["id"].lower()])
            idx_or_defecto = 0
            for i, mod in enumerate(st.session_state.modelos_or):
                if "vision" in mod.lower():
                    idx_or_defecto = i
                    break
                elif "llama-3.3-70b-instruct:free" in mod.lower():
                    idx_or_defecto = i
                    break
            st.session_state.modelo_or = st.selectbox("Versión de Llama:", st.session_state.modelos_or, index=idx_or_defecto)
        except: pass
    else: st.info("ℹ️ Gemini se auto-auditará.")
    
    st.divider()
    st.session_state.punto_cuerpo = st.selectbox("Anatomía:", ["✨ Autodetectar", "Cabeza", "Tórax", "Abdomen", "Brazo", "Mano", "Pierna", "Pie"])

with col_c:
    st.subheader("1. Selección de Modo")
    modo = st.selectbox("Especialidad:", ["✨ Autodetectar", "🧠 Medicina Interna", "🩸 Analíticas", "🩹 Heridas", "📈 ECG", "💀 RX/TAC", "🦇 Ecografía", "📚 PubMed"])
    contexto = st.selectbox("🏥 Contexto:", ["Urgencias", "Hospitalización", "UCI", "Residencia", "Domicilio"], index=1)
    metodo_captura = st.radio("Entrada", ["📁 Subir Archivos", "📸 Tomar Foto"], horizontal=True, label_visibility="collapsed")
    
    archivos = []; fs = None; cam_pic = None
    if metodo_captura == "📁 Subir Archivos": fs = st.file_uploader("Archivos:", type=['jpg','png','pdf','mp4','mov'], accept_multiple_files=True)
    else: cam_pic = st.camera_input("Cámara")
    
    if fs:
        for f in fs:
            if f.type.startswith('video'): archivos.append(("video", f))
            elif "pdf" in f.type: archivos.append(("doc", f))
            else: archivos.append(("img", f))
    if cam_pic: archivos.append(("img", cam_pic))
    
    notas = st.text_area("Notas / Preguntas", height=70, placeholder="Contexto del paciente...")
    
    c1, c2 = st.columns([3, 1])
    with c1: btn_analizar = st.button("🚀 ANALIZAR", type="primary", use_container_width=True)
    with c2: 
        if st.button("🔄 NUEVO", use_container_width=True):
            st.session_state.resultado_analisis = None
            st.session_state.img_marcada = None
            st.rerun()

    if btn_analizar:
        with st.spinner("IA #1: Generando borrador..."):
            try:
                model = genai.GenerativeModel(st.session_state.modelo_seleccionado)
                con = []; txt_docs = ""; imagen_para_visor = None; video_path = None
                
                for tipo, f in archivos:
                    if tipo == "video":
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tf:
                            tf.write(f.read()); video_path = tf.name
                        if "Ecografía" in modo:
                            _, eb, met = procesar_pocus_v6_singularidad(video_path)
                            st.session_state.pocus_metrics = met
                            imagen_para_visor = eb
                        v_file = genai.upload_file(video_path)
                        while v_file.state.name == "PROCESSING": time.sleep(1); v_file = genai.get_file(v_file.name)
                        con.append(v_file)
                    elif tipo == "doc":
                        txt_docs += "".join([p.extract_text() for p in pypdf.PdfReader(f).pages])
                    elif tipo == "img":
                        img = anonimizar_imagen(ImageOps.exif_transpose(Image.open(f)).convert("RGB"), modo)
                        if "Heridas" in modo:
                            res, a = segmentar_herida_sam_v2(img)
                            if a > 0: st.session_state.img_marcada = res; txt_docs += f"\nÁrea SAM: {a} px."
                        con.append(img); imagen_para_visor = img

                # --- PROMPT ---
                puente = ""
                if "vision" not in st.session_state.get("modelo_or", "").lower() and imagen_para_visor:
                    puente = "\n<details class='radiomics-box' open><summary>👁️ ANÁLISIS VISUAL</summary><p>[Describe físicamente la imagen]</p></details>"
                
                prompt = f"Eres LabMind. Contexto: {contexto}. Especialidad: {modo}. Notas: {notas}. Datos: {txt_docs}. {puente}\n<details class='diagnosis-box' open><summary>🚨 HALLAZGOS</summary><p>[Análisis]</p></details><details class='action-box' open><summary>⚡ ACCIÓN</summary><p>[Plan]</p></details>"
                res = model.generate_content([prompt, *con] if con else prompt, safety_settings=MEDICAL_SAFETY_SETTINGS)
                raw_txt = res.text.replace("```html", "").replace("```", "").strip()

                # --- AUDITORÍA OPENROUTER ---
                if st.session_state.get("api_key_or"):
                    with st.spinner("IA #2: Verificando seguridad..."):
                        client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=st.session_state.api_key_or)
                        content = [{"type": "text", "text": f"Audita este HTML médico. Si hay errores o alucinaciones, corrígelo. Si es correcto, añade '✅ Auditado' al título: {raw_txt}"}]
                        if "vision" in st.session_state.modelo_or.lower() and imagen_para_visor:
                            content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_to_base64(imagen_para_visor)}"}})
                        audit = client.chat.completions.create(model=st.session_state.modelo_or, messages=[{"role": "user", "content": content}])
                        raw_txt = audit.choices[0].message.content.replace("```html", "").replace("```", "").strip()

                st.session_state.resultado_analisis = raw_txt
                st.session_state.pdf_bytes = create_pdf(raw_txt)
                if not st.session_state.img_marcada: st.session_state.img_marcada = imagen_para_visor
            except Exception as e: st.error(f"Error: {e}")

    if st.session_state.resultado_analisis:
        st.markdown(st.session_state.resultado_analisis, unsafe_allow_html=True)
        st.download_button("📥 Descargar PDF", st.session_state.pdf_bytes, "informe.pdf")

with col_r:
    with st.expander("👁️ Dashboard Visual", expanded=True):
        if st.session_state.get("pocus_metrics"): st.metric("FEVI Estimada", f"{st.session_state.pocus_metrics.get('FEVI')}%")
        if st.session_state.get("img_marcada"): st.image(st.session_state.img_marcada, use_container_width=True)

# --- CHAT ---
if st.session_state.resultado_analisis:
    st.divider(); st.markdown("### 💬 Chat IA")
    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"]): st.markdown(msg["content"])
    if q := st.chat_input("Duda clínica..."):
        st.session_state.chat_messages.append({"role": "user", "content": q})
        hist = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.chat_messages])
        resp = genai.GenerativeModel(st.session_state.modelo_seleccionado).generate_content(f"Informe: {st.session_state.resultado_analisis}\nChat: {hist}")
        st.session_state.chat_messages.append({"role": "assistant", "content": resp.text})
        st.rerun()
