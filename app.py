import streamlit as st
import google.generativeai as genai
from PIL import Image
import pypdf
import tempfile
import time
import os
from fpdf import FPDF
import datetime
import re
import matplotlib.pyplot as plt
import cv2
import numpy as np
import extra_streamlit_components as stx
import pandas as pd
from streamlit_image_coordinates import streamlit_image_coordinates

# --- CONFIGURACIÓN ---
st.set_page_config(page_title="LabMind 28.2 (RX Video)", page_icon="💀", layout="wide")

# --- ESTILOS CSS ---
st.markdown("""
<style>
    .stButton>button { width: 100%; border-radius: 8px; font-weight: bold; background-color: #0066cc; color: white; }
    .sync-alert { border: 2px solid #d32f2f; padding: 15px; border-radius: 10px; background-color: #ffebee; color: #b71c1c; font-weight: bold; margin-bottom: 10px; animation: pulse 2s infinite; }
    .biofilm-alert { border: 2px solid #ffd600; padding: 10px; border-radius: 10px; background-color: #fff9c4; color: #827717; font-weight: bold; }
    .prediction-box { background-color: #e8f5e9; padding: 15px; border-radius: 10px; border-left: 5px solid #4caf50; margin-top: 10px; }
    .tissue-bar-container { display: flex; width: 100%; height: 25px; border-radius: 12px; overflow: hidden; margin: 10px 0; }
    .tissue-gran { background-color: #ef5350; color: white; display: flex; align-items: center; justify-content: center; font-size: 0.7em; font-weight: bold; }
    .tissue-slough { background-color: #fdd835; color: #333; display: flex; align-items: center; justify-content: center; font-size: 0.7em; font-weight: bold; }
    .tissue-nec { background-color: #212121; color: white; display: flex; align-items: center; justify-content: center; font-size: 0.7em; font-weight: bold; }
    @keyframes pulse { 0% { box-shadow: 0 0 0 0 rgba(198, 40, 40, 0.4); } 70% { box-shadow: 0 0 0 10px rgba(198, 40, 40, 0); } 100% { box-shadow: 0 0 0 0 rgba(198, 40, 40, 0); } }
    [data-testid='stFileUploaderDropzone'] div div span { display: none; }
    [data-testid='stFileUploaderDropzone'] div div::after { content: "Arrastra Archivos"; font-size: 1rem; font-weight: bold; color: #444; display: block; }
</style>
""", unsafe_allow_html=True)

# --- GESTOR DE ESTADO ---
cookie_manager = stx.CookieManager()
if "autenticado" not in st.session_state: st.session_state.autenticado = False
if "api_key" not in st.session_state: st.session_state.api_key = ""
if "resultado_analisis" not in st.session_state: st.session_state.resultado_analisis = None
if "pdf_bytes" not in st.session_state: st.session_state.pdf_bytes = None
if "historial_evolucion" not in st.session_state: st.session_state.historial_evolucion = []
if "area_herida" not in st.session_state: st.session_state.area_herida = 0.0
if "log_privacidad" not in st.session_state: st.session_state.log_privacidad = []
if "punto_cuerpo" not in st.session_state: st.session_state.punto_cuerpo = None

# --- LOGIN ---
time.sleep(0.1)
cookie_api_key = cookie_manager.get(cookie="labmind_secret_key")
if not st.session_state.autenticado:
    if cookie_api_key:
        st.session_state.api_key = cookie_api_key; st.session_state.autenticado = True; st.rerun()
    else:
        st.title("LabMind Acceso")
        k = st.text_input("API Key:", type="password")
        if st.button("Entrar"):
            expires = datetime.datetime.now() + datetime.timedelta(days=30)
            cookie_manager.set("labmind_secret_key", k, expires_at=expires)
            st.session_state.api_key = k; st.session_state.autenticado = True; st.rerun()
        st.stop()

# ==========================================
#      FUNCIONES AUXILIARES
# ==========================================

def procesar_termografia(pil_image):
    try:
        img = np.array(pil_image.convert('RGB'))
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
        _, a, _ = cv2.split(lab)
        thermal_map = cv2.normalize(a, None, 0, 255, cv2.NORM_MINMAX)
        thermal_colored = cv2.applyColorMap(thermal_map, cv2.COLORMAP_JET)
        thermal_colored = cv2.GaussianBlur(thermal_colored, (15, 15), 0)
        return Image.fromarray(cv2.cvtColor(thermal_colored, cv2.COLOR_BGR2RGB))
    except: return pil_image

def detectar_biofilm(pil_image):
    try:
        img = np.array(pil_image.convert('RGB'))
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        kernel = np.ones((5,5),np.uint8)
        gradient = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel)
        _, mask = cv2.threshold(gradient, 10, 255, cv2.THRESH_BINARY_INV)
        img_res = img.copy()
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img_res, contours, -1, (255, 214, 0), 2)
        return Image.fromarray(img_res), len(contours) > 0
    except: return pil_image, False

def medir_herida(pil_image):
    try:
        img = np.array(pil_image.convert('RGB'))
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        circles = cv2.HoughCircles(cv2.GaussianBlur(gray, (9,9), 2), cv2.HOUGH_GRADIENT, 1.2, 50, param1=100, param2=30, minRadius=20, maxRadius=200)
        area = 0.0
        if circles is not None:
            r_c = circles[0][0][2]
            pixels_per_cm = (r_c * 2) / 2.325
            hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, np.array([0, 40, 40]), np.array([30, 255, 255]))
            cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            area_px = sum(cv2.contourArea(c) for c in cnts if cv2.contourArea(c) > 500)
            area = area_px * ((1 / pixels_per_cm) ** 2)
        return area
    except: return 0.0

def anonymize_face(pil_image):
    try:
        img_np = np.array(pil_image.convert('RGB'))
        img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        processed = False
        for (x, y, w, h) in faces:
            roi_color = img_cv[y:y+h, x:x+w]
            blurred = cv2.GaussianBlur(roi_color, (99, 99), 30)
            img_cv[y:y+h, x:x+w] = blurred
            processed = True
        return Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)), processed
    except: return pil_image, False

def predecir_cierre():
    hist = st.session_state.historial_evolucion
    if len(hist) < 3: return "Necesito +3 registros para predecir."
    areas = [h['Area'] for h in hist]
    reduccion = (areas[0] - areas[-1]) / len(hist)
    if reduccion <= 0: return "⚠️ Estancamiento. No hay cierre previsto."
    dias = areas[-1] / reduccion
    return f"Cierre estimado en: **{int(dias)} días**."

def create_pdf(texto_analisis):
    class PDF(FPDF):
        def header(self): self.set_font('Arial','B',12); self.cell(0,10,'LabMind - Informe IA',0,1,'C'); self.ln(5)
        def footer(self): self.set_y(-15); self.set_font('Arial','I',8); self.cell(0,10,f'Pag {self.page_no()}',0,0,'C')
    pdf = PDF(); pdf.add_page(); pdf.set_font("Arial", size=10)
    pdf.cell(0,10,f"Fecha: {datetime.datetime.now().strftime('%d/%m/%Y %H:%M')}",0,1); pdf.ln(5)
    clean = texto_analisis.replace('€','EUR').replace('’',"'").replace('“','"').replace('”','"')
    clean = re.sub(r'<[^>]+>', '', clean) 
    pdf.multi_cell(0,5, clean.encode('latin-1','replace').decode('latin-1'))
    return pdf.output(dest='S').encode('latin-1')

# ==========================================
#      INTERFAZ DE USUARIO
# ==========================================

st.title("🩺 LabMind 28.2 (RX Video Support)")
col_left, col_center, col_right = st.columns([1, 2, 1])

# --- COLUMNA 1: MAPA CORPORAL ---
with col_left:
    st.subheader("📍 Localización")
    body_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/a/ad/Human_body_features.svg/267px-Human_body_features.svg.png"
    value = streamlit_image_coordinates(body_url, key="body_click")
    if value:
        st.session_state.punto_cuerpo = value
        st.success(f"📌 Punto: X={value['x']}, Y={value['y']}")
    else: st.info("👆 Toca la zona en la imagen")

    st.divider()
    with st.expander("💊 Subir Fármacos", expanded=True):
        meds_files = st.file_uploader("Fotos Caja/Receta", accept_multiple_files=True, key="meds")

# --- COLUMNA 2: NÚCLEO ---
with col_center:
    st.subheader("1. Evidencia Clínica")
    c_ctx, c_mod = st.columns(2)
    contexto = c_ctx.selectbox("🏥 Contexto:", ["Hospitalización", "Residencia", "Urgencias", "UCI", "Domicilio"])
    modo = c_mod.radio("Modo:", ["🧩 Integral", "🩹 Heridas", "🧴 Dermatología", "📊 Analíticas", "📈 ECG", "💊 Farmacia", "💀 RX / TAC / RMN"])
    
    with st.expander("📊 Analíticas / ECG / RX", expanded=True):
        tests_files = st.file_uploader("PDF/Fotos", accept_multiple_files=True, key="tests")

    st.markdown("---")
    st.write("📸 **Evidencia Visual (Foto/Video):**")
    st.caption("Acepta: JPG, PNG, MP4, MOV (Ideal para scroll de TAC/RMN)")
    
    fuente = st.radio("Fuente:", ["📁 Archivo", "📸 WebCam"], horizontal=True, label_visibility="collapsed")
    archivos = []
    
    if fuente == "📸 WebCam":
        if f := st.camera_input("Foto"): archivos.append(("cam", f))
    else:
        # AQUI ES DONDE SE PERMITE VIDEO Y FOTO EN RX
        if fs := st.file_uploader("Subir Archivos", type=['jpg','png','mp4','mov'], accept_multiple_files=True):
            for f in fs:
                ftype = "video" if "video" in f.type else "img"
                archivos.append((ftype, f))

    audio = st.audio_input("🎙️ Notas de Voz")
    notas = st.text_area("Notas Clínicas:", height=60)

    if st.button("🚀 ANALIZAR CASO", type="primary"):
        st.session_state.log_privacidad = []; st.session_state.area_herida = 0.0
        
        with st.spinner("🧠 Procesando video/imagen y cruzando datos..."):
            try:
                genai.configure(api_key=st.session_state.api_key)
                model = genai.GenerativeModel("models/gemini-3-flash-preview")
                
                con = []; txt_meds = ""; txt_tests = ""

                # Procesar Textos
                if meds_files:
                    for f in meds_files:
                        if "pdf" in f.type: r = pypdf.PdfReader(f); txt_meds += "".join([p.extract_text() for p in r.pages])
                        else: con.append(Image.open(f))
                if tests_files:
                    for f in tests_files:
                        if "pdf" in f.type: r = pypdf.PdfReader(f); txt_tests += "".join([p.extract_text() for p in r.pages])
                        else: con.append(Image.open(f))
                if audio: con.append(genai.upload_file(audio, mime_type="audio/wav"))
                
                # Procesar Visual (Video/Foto)
                img_display = None; img_thermal = None; img_biofilm = None; biofilm_detectado = False
                has_video = False
                
                for t, a in archivos:
                    if t == "video":
                        has_video = True
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tf: tf.write(a.read()); tp = tf.name
                        vf = genai.upload_file(path=tp)
                        while vf.state.name == "PROCESSING": time.sleep(1); vf = genai.get_file(vf.name)
                        con.append(vf); os.remove(tp)
                    else: 
                        img_pil = Image.open(a)
                        if modo == "🩹 Heridas":
                            area = medir_herida(img_pil)
                            if area > 0: st.session_state.area_herida = area
                            img_thermal = procesar_termografia(img_pil)
                            img_biofilm, biofilm_detectado = detectar_biofilm(img_pil)
                            img_display = img_pil
                            con.append(img_pil); con.append(img_thermal)
                        elif modo == "💊 Farmacia": img_display = img_pil; con.append(img_pil)
                        elif modo == "🧴 Dermatología" or "RX" in modo:
                            img_display = img_pil; con.append(img_pil)
                        else:
                            img_final, proc = anonymize_face(img_pil)
                            img_display = img_final; con.append(img_final)

                coord_txt = ""
                if st.session_state.punto_cuerpo:
                    coord_txt = f"El usuario tocó en mapa corporal coords: X={st.session_state.punto_cuerpo['x']}, Y={st.session_state.punto_cuerpo['y']}."

                prompt = f"""
                Rol: Enfermera APN / Radiólogo. Contexto: {contexto}. Modo: {modo}.
                {coord_txt}
                Notas: "{notas}"
                
                DATA:
                1. 💊 TRATAMIENTO: {txt_meds}
                2. 📊 PRUEBAS: {txt_tests}
                3. 📸 EVIDENCIA VISUAL (Foto o Video).
                
                INSTRUCCIONES CLAVE:
                - Si es RX/TAC/RMN y hay VIDEO: Analiza la secuencia de cortes (scroll). Busca hallazgos que aparecen/desaparecen.
                - Si es HERIDA: Analiza termografía y biofilm.
                - CROSS-CHECK: Cruza fármacos con hallazgos visuales.
                
                OUTPUT:
                ### ⚡ RESUMEN
                SYNC_ALERT: [ALERTA SI HAY CONFLICTO]
                * **📍 ZONA:** [Zona deducida]
                * **🚨 DIAGNÓSTICO:**
                ...
                ### 📊 Análisis Visual
                [Si es herida, HTML barra tejidos sin bloques de código]
                ...
                """
                
                resp = model.generate_content([prompt, *con] if con else prompt)
                st.session_state.resultado_analisis = resp.text
                
                if modo == "🩹 Heridas" and st.session_state.area_herida > 0:
                    st.session_state.historial_evolucion.append({
                        "Fecha": datetime.datetime.now().strftime("%d/%m %H:%M"), "Area": st.session_state.area_herida
                    })
                
                st.session_state.pdf_bytes = create_pdf(resp.text.replace("*","").replace("#",""))

                if img_display: st.image(img_display, caption="Evidencia", width=300)
                if has_video: st.info("🎥 Video Procesado por IA")
                if img_thermal: st.image(img_thermal, caption="Termografía", width=300)
                if biofilm_detectado:
                    st.image(img_biofilm, caption="Biofilm", width=300)
                    st.markdown('<div class="biofilm-alert">🔍 SOSPECHA BIOFILM</div>', unsafe_allow_html=True)

            except Exception as e: st.error(f"Error: {e}")

    if st.session_state.resultado_analisis:
        txt = st.session_state.resultado_analisis.replace("```html", "").replace("```", "")
        sync_match = re.search(r'SYNC_ALERT: (.*)', txt)
        if sync_match and len(sync_match.group(1).strip()) > 5:
            st.markdown(f'<div class="sync-alert">⚠️ {sync_match.group(1)}</div>', unsafe_allow_html=True)
        st.markdown(txt, unsafe_allow_html=True)
    
    if st.session_state.pdf_bytes:
        st.download_button("📥 Descargar Informe PDF", st.session_state.pdf_bytes, "informe.pdf", "application/pdf")

# --- COLUMNA 3: ESTADÍSTICAS ---
with col_right:
    st.subheader("📈 Pronóstico")
    if st.session_state.historial_evolucion:
        df = pd.DataFrame(st.session_state.historial_evolucion)
        st.line_chart(df.set_index("Fecha"))
        pred = predecir_cierre()
        st.markdown(f'<div class="prediction-box">🔮 <b>IA Supervivencia:</b><br>{pred}</div>', unsafe_allow_html=True)
    else:
        st.info("Sin datos evolutivos.")

st.divider()
if st.button("🔒 Salir"):
    cookie_manager.delete("labmind_secret_key")
    st.session_state.autenticado = False; st.rerun()


