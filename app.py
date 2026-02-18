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

# --- CONFIGURACIÓN ---
st.set_page_config(page_title="LabMind 27.4 (Total Sync)", page_icon="🧬", layout="wide")

# --- ESTILOS CSS ---
st.markdown("""
<style>
    .stButton>button { width: 100%; border-radius: 8px; font-weight: bold; background-color: #0066cc; color: white; }
    
    /* ALERTAS DE SEGURIDAD */
    .sync-alert { 
        border: 2px solid #d32f2f; 
        padding: 15px; 
        border-radius: 10px; 
        background-color: #ffebee; 
        color: #b71c1c; 
        font-weight: bold; 
        margin-bottom: 10px; 
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        animation: pulse 2s infinite;
    }
    
    .biofilm-alert { border: 2px solid #ffd600; padding: 10px; border-radius: 10px; background-color: #fff9c4; color: #827717; font-weight: bold; }
    .prediction-box { background-color: #e8f5e9; padding: 15px; border-radius: 10px; border-left: 5px solid #4caf50; margin-top: 10px; }
    .ar-guide { border: 2px dashed #2ecc71; padding: 10px; text-align: center; color: #27ae60; font-weight: bold; margin-bottom: 10px; border-radius: 8px; background-color: #e8f8f5; }

    /* BARRA DE TEJIDOS */
    .tissue-bar-container { display: flex; width: 100%; height: 25px; border-radius: 12px; overflow: hidden; margin: 10px 0; }
    .tissue-gran { background-color: #ef5350; color: white; display: flex; align-items: center; justify-content: center; font-size: 0.7em; font-weight: bold; }
    .tissue-slough { background-color: #fdd835; color: #333; display: flex; align-items: center; justify-content: center; font-size: 0.7em; font-weight: bold; }
    .tissue-nec { background-color: #212121; color: white; display: flex; align-items: center; justify-content: center; font-size: 0.7em; font-weight: bold; }

    @keyframes pulse { 0% { box-shadow: 0 0 0 0 rgba(198, 40, 40, 0.4); } 70% { box-shadow: 0 0 0 10px rgba(198, 40, 40, 0); } 100% { box-shadow: 0 0 0 0 rgba(198, 40, 40, 0); } }
    
    /* UPLOADER */
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
    pdf.multi_cell(0,5, clean.encode('latin-1','replace').decode('latin-1'))
    return pdf.output(dest='S').encode('latin-1')

# ==========================================
#      INTERFAZ DE USUARIO
# ==========================================

st.title("🩺 LabMind 27.4 (Total Sync)")
col_left, col_center, col_right = st.columns([1, 2, 1])

# --- COLUMNA 1: CONTEXTO ---
with col_left:
    st.subheader("📍 Datos Paciente")
    st.session_state.punto_mapa = st.selectbox("Zona Cuerpo:", ["Talón", "Sacro", "Pie Diabético", "Tibia", "Brazo", "Tórax", "N/A"])
    st.image("https://cdn-icons-png.flaticon.com/512/3133/3133694.png", width=150)
    
    st.divider()
    st.markdown("### 💊 Tratamiento")
    with st.expander("Subir Fármacos", expanded=True):
        meds_files = st.file_uploader("Fotos Caja/Receta", accept_multiple_files=True, key="meds")

# --- COLUMNA 2: NÚCLEO ---
with col_center:
    st.subheader("1. Evidencia Clínica")
    c_ctx, c_mod = st.columns(2)
    contexto = c_ctx.selectbox("🏥 Contexto:", ["Hospitalización", "Residencia", "Urgencias", "UCI", "Domicilio"])
    modo = c_mod.radio("Modo:", ["🧩 Integral", "🩹 Heridas", "🧴 Dermatología", "📊 Analíticas", "📈 ECG", "💊 Farmacia"])
    
    # --- BLOQUE DE ANALÍTICAS/PRUEBAS ---
    st.markdown("### 📊 Analíticas y Pruebas (ECG/RX)")
    with st.expander("Subir Analíticas / ECG / Pruebas", expanded=True):
        tests_files = st.file_uploader("Sube PDF o Fotos de los resultados", accept_multiple_files=True, key="tests")

    st.markdown("---")
    st.write("📸 **Estado Actual del Paciente (Foto/Video):**")
    fuente = st.radio("Fuente:", ["📁 Archivo", "📸 WebCam"], horizontal=True, label_visibility="collapsed")
    archivos = []
    
    if fuente == "📸 WebCam":
        if f := st.camera_input("Foto"): archivos.append(("cam", f))
    else:
        if fs := st.file_uploader("Evidencia Visual", type=['jpg','png','mp4','mov'], accept_multiple_files=True):
            for f in fs:
                ftype = "video" if "video" in f.type else "img"
                archivos.append((ftype, f))

    audio = st.audio_input("🎙️ Notas de Voz")
    notas = st.text_area("Notas Clínicas:", height=60)

    if st.button("🚀 SINCRONIZAR Y ANALIZAR", type="primary"):
        st.session_state.log_privacidad = []; st.session_state.area_herida = 0.0
        
        with st.spinner("🧠 Cruzando: Fármacos + Analítica + Clínica..."):
            try:
                genai.configure(api_key=st.session_state.api_key)
                model = genai.GenerativeModel("models/gemini-3-flash-preview")
                
                con = []; txt_c = ""
                txt_meds = ""; txt_tests = ""

                # 1. PROCESAR FÁRMACOS
                if meds_files:
                    for f in meds_files:
                        if "pdf" in f.type:
                            r = pypdf.PdfReader(f); txt_meds += "".join([p.extract_text() for p in r.pages])
                        else: con.append(Image.open(f)); txt_c += "\n[IMG FÁRMACO]\n"

                # 2. PROCESAR ANALÍTICAS/PRUEBAS
                if tests_files:
                    for f in tests_files:
                        if "pdf" in f.type:
                            r = pypdf.PdfReader(f); txt_tests += "".join([p.extract_text() for p in r.pages])
                        else: con.append(Image.open(f)); txt_c += "\n[IMG ANALITICA/ECG]\n"

                # 3. PROCESAR EVIDENCIA VISUAL
                if audio: con.append(genai.upload_file(audio, mime_type="audio/wav")); txt_c += "\n[AUDIO]\n"
                
                img_display = None; img_thermal = None; img_biofilm = None; biofilm_detectado = False
                
                for t, a in archivos:
                    if t == "video":
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tf: tf.write(a.read()); tp = tf.name
                        vf = genai.upload_file(path=tp)
                        while vf.state.name == "PROCESSING": time.sleep(1); vf = genai.get_file(vf.name)
                        con.append(vf); txt_c += "\n[VIDEO PACIENTE]\n"; os.remove(tp)
                    else: # Imagen
                        img_pil = Image.open(a)
                        if modo == "🩹 Heridas":
                            area = medir_herida(img_pil)
                            if area > 0: st.session_state.area_herida = area
                            img_thermal = procesar_termografia(img_pil)
                            img_biofilm, biofilm_detectado = detectar_biofilm(img_pil)
                            img_display = img_pil
                            con.append(img_pil); con.append(img_thermal)
                        elif modo == "💊 Farmacia":
                            img_display = img_pil; con.append(img_pil)
                        else:
                            img_final, proc = anonymize_face(img_pil)
                            if proc: st.session_state.log_privacidad.append("🛡️ Rostro Anonimizado")
                            img_display = img_final; con.append(img_final)

                # PROMPT DE SINCRONIZACIÓN TOTAL
                prompt = f"""
                Rol: Enfermera Clínica Experta (APN).
                Contexto: {contexto}. Modo: {modo}.
                Notas Usuario: "{notas}"
                
                DATOS CARGADOS:
                1. 💊 TRATAMIENTO (Texto extraído): {txt_meds}
                2. 📊 ANALÍTICAS/PRUEBAS (Texto extraído): {txt_tests}
                3. 📸 IMÁGENES ADJUNTAS: (Fármacos, Analíticas, ECG, Herida).
                
                TAREA CRÍTICA (CROSS-CHECK):
                Cruza TODAS las fuentes de datos. Busca relaciones Causa-Efecto peligrosas.
                
                EJEMPLOS DE BÚSQUEDA:
                - Fármaco + Analítica: ¿Toma diuréticos y el Sodio/Potasio está alterado?
                - Fármaco + Analítica: ¿Toma estatinas y la CPK/Hígado está alta?
                - Fármaco + ECG: ¿Toma QT-prolonging drugs y el ECG está alterado?
                - Fármaco + Herida: ¿Toma corticoides/Sintrom y la herida no cierra o sangra?
                
                OUTPUT FORMAT:
                ### ⚡ RESUMEN DE SINCRONIZACIÓN
                SYNC_ALERT: [Si hay conflicto Fármaco-Analítica-Clínica, escribe ALERTA MAYÚSCULAS AQUÍ].
                * **🚨 DIAGNÓSTICO INTEGRAL:**
                ...
                ### 🔍 DETALLE DE CRUCE (Fármacos vs Pruebas)
                * **Correlación Detectada:** [Ej: Enalapril + Potasio 5.8]
                * **Acción Sugerida:** [Ej: Revisar dosis / Suspender]
                ...
                ### 📊 Análisis Visual
                [Si es herida, barra HTML tejidos: <div class="tissue-bar-container">...]
                ...
                """
                
                resp = model.generate_content([prompt, *con] if con else prompt)
                st.session_state.resultado_analisis = resp.text
                
                if modo == "🩹 Heridas" and st.session_state.area_herida > 0:
                    st.session_state.historial_evolucion.append({
                        "Fecha": datetime.datetime.now().strftime("%d/%m %H:%M"), "Area": st.session_state.area_herida
                    })
                
                st.session_state.pdf_bytes = create_pdf(resp.text.replace("*","").replace("#",""))

                if img_display: st.image(img_display, caption="Evidencia Clínica", width=300)
                if img_thermal: st.image(img_thermal, caption="Termografía", width=300)
                if biofilm_detectado:
                    st.image(img_biofilm, caption="Biofilm", width=300)
                    st.markdown('<div class="biofilm-alert">🔍 SOSPECHA DE BIOFILM</div>', unsafe_allow_html=True)

            except Exception as e: st.error(f"Error: {e}")

    # RENDERIZADO
    if st.session_state.resultado_analisis:
        txt = st.session_state.resultado_analisis
        
        # Alerta de Sincronización
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

