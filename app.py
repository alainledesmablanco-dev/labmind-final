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
st.set_page_config(page_title="LabMind 30.1", page_icon="🧬", layout="wide")

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
    
    /* Ocultar elementos sobrantes del uploader para limpiar visualmente */
    [data-testid='stFileUploaderDropzone'] div div span { display: none; }
    [data-testid='stFileUploaderDropzone'] div div::after { content: "📂 Toca para adjuntar archivo"; font-size: 1rem; color: #555; display: block; }
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
            area_px = np.pi * (r_c**2) 
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

st.title("🩺 LabMind 30.1 (Video Support)")
col_left, col_center, col_right = st.columns([1, 2, 1])

# --- COLUMNA 1: MAPA CORPORAL ---
with col_left:
    st.subheader("📍 Mapa Interactivo")
    # URL MÁS ROBUSTA
    body_url = "https://cdn-icons-png.flaticon.com/512/3022/3022286.png" 
    
    st.caption("Toca la zona afectada:")
    try:
        value = streamlit_image_coordinates(body_url, key="body", height=300)
        if value:
            st.session_state.punto_cuerpo = value
            st.success("✅ Zona marcada")
    except:
        st.error("⚠️ Error cargando imagen interactiva. Usa el selector abajo.")
        value = None

    # SELECTOR DE RESPALDO
    if not value:
        zona_backup = st.selectbox("Selección Manual de Zona:", 
                                 ["No especificada", "Talón (Izq/Dch)", "Sacro/Glúteos", 
                                  "Espalda", "Abdomen", "Pecho", "Cabeza", "Brazo", "Pierna", "Pie"])
        if zona_backup != "No especificada":
             st.session_state.punto_cuerpo = {"x": 0, "y": 0, "label": zona_backup}

# --- COLUMNA 2: NÚCLEO CENTRAL (SEPARACIÓN ESTRICTA) ---
with col_center:
    st.subheader("1. Selección de Modo")
    
    # LISTA DE MODOS
    modo = st.selectbox("Especialidad:", 
                 ["🩹 Heridas / Úlceras", 
                  "🧴 Dermatología", 
                  "💊 Farmacia (Interacciones)", 
                  "📈 ECG (Cardiología)", 
                  "💀 RX / TAC / RMN (Imagen)", 
                  "📂 Analizar Informes"])
    
    contexto = st.selectbox("🏥 Contexto:", ["Hospitalización", "Residencia", "Urgencias", "UCI", "Domicilio"])
    
    st.markdown("---")
    
    # --- LÓGICA DE VISUALIZACIÓN EXCLUSIVA (LIMPIEZA) ---
    archivos = []
    meds_files = None
    tests_files = None
    
    # 1. HERIDAS (FOTO Y VIDEO AÑADIDO)
    if modo == "🩹 Heridas / Úlceras":
        st.info("🩹 **Modo Heridas**: Sube Foto o Video de la lesión.")
        
        with st.expander("Opcional: Adjuntar Medicación/Analítica", expanded=False):
            c1, c2 = st.columns(2)
            meds_files = c1.file_uploader("Medicación", accept_multiple_files=True, key="wound_meds")
            tests_files = c2.file_uploader("Analítica", accept_multiple_files=True, key="wound_labs")
            
        fuente = st.radio("Fuente:", ["📁 Archivo", "📸 WebCam"], horizontal=True, label_visibility="collapsed")
        if fuente == "📸 WebCam":
            if f := st.camera_input("Foto Herida"): archivos.append(("cam", f))
        else:
            # AHORA ACEPTA VIDEO MP4/MOV TAMBIÉN
            if fs := st.file_uploader("Subir Foto/Video Herida", type=['jpg','png','mp4','mov'], accept_multiple_files=True, key="wound_img"):
                for f in fs:
                    ftype = "video" if "video" in f.type else "img"
                    archivos.append((ftype, f))

    # 2. DERMATOLOGÍA (FOTO Y VIDEO AÑADIDO)
    elif modo == "🧴 Dermatología":
        st.info("🧴 **Modo Dermatología**: Sube Foto o Video macro de la piel.")
        # AHORA ACEPTA VIDEO MP4/MOV TAMBIÉN
        if fs := st.file_uploader("Subir Foto/Video Piel", type=['jpg','png','mp4','mov'], accept_multiple_files=True, key="derma_img"):
            for f in fs:
                ftype = "video" if "video" in f.type else "img"
                archivos.append((ftype, f))

    # 3. FARMACIA (Solo documentos/cajas)
    elif modo == "💊 Farmacia (Interacciones)":
        st.info("💊 **Modo Farmacia**: Sube foto de **Receta, Lista o Caja**.")
        meds_files = st.file_uploader("Documento o Caja", accept_multiple_files=True, key="pharma_docs")

    # 4. ECG (Solo Electro)
    elif modo == "📈 ECG (Cardiología)":
        st.info("📈 **Modo Cardiología**: Sube foto del **Electrocardiograma**.")
        if fs := st.file_uploader("Imagen ECG", type=['jpg','png','pdf'], accept_multiple_files=True, key="ecg_docs"):
            for f in fs: archivos.append(("img", f))

    # 5. RX/TAC (Video ya estaba activo)
    elif modo == "💀 RX / TAC / RMN (Imagen)":
        st.info("💀 **Modo Radiología**: Sube **Video (Scroll)** o **Placa**.")
        if fs := st.file_uploader("Video/Imagen RX", type=['jpg','png','mp4','mov'], accept_multiple_files=True, key="rx_docs"):
            for f in fs: 
                ftype = "video" if "video" in f.type else "img"
                archivos.append((ftype, f))

    # 6. INFORMES (Solo Docs)
    elif modo == "📂 Analizar Informes":
        st.info("📂 **Modo Gestión Documental**: Sube PDFs o Fotos de informes.")
        tests_files = st.file_uploader("Informes PDF/JPG", accept_multiple_files=True, key="report_docs")

    st.markdown("---")
    audio = st.audio_input("🎙️ Notas de Voz")
    notas = st.text_area("Notas Clínicas:", height=60, placeholder="Escribe síntomas, alergias...")

    # --- BOTÓN DE ANÁLISIS ---
    if st.button("🚀 ANALIZAR", type="primary"):
        st.session_state.log_privacidad = []; st.session_state.area_herida = 0.0
        
        with st.spinner(f"🧠 Analizando {modo}..."):
            try:
                genai.configure(api_key=st.session_state.api_key)
                model = genai.GenerativeModel("models/gemini-3-flash-preview")
                
                con = []; txt_meds = ""; txt_tests = ""

                # Procesamiento Condicional según lo que se haya subido
                if meds_files:
                    for f in meds_files:
                        if "pdf" in f.type: r = pypdf.PdfReader(f); txt_meds += "".join([p.extract_text() for p in r.pages])
                        else: con.append(Image.open(f))
                
                if tests_files:
                    for f in tests_files:
                        if "pdf" in f.type: r = pypdf.PdfReader(f); txt_tests += "".join([p.extract_text() for p in r.pages])
                        else: con.append(Image.open(f))
                
                if audio: con.append(genai.upload_file(audio, mime_type="audio/wav"))
                
                img_display = None; img_thermal = None; img_biofilm = None; biofilm_detectado = False
                
                for t, a in archivos:
                    if t == "video":
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tf: tf.write(a.read()); tp = tf.name
                        vf = genai.upload_file(path=tp)
                        while vf.state.name == "PROCESSING": time.sleep(1); vf = genai.get_file(vf.name)
                        con.append(vf); os.remove(tp)
                    else: 
                        img_pil = Image.open(a)
                        # Lógica Visual por Modo
                        if "Heridas" in modo:
                            area = medir_herida(img_pil)
                            if area > 0: st.session_state.area_herida = area
                            img_thermal = procesar_termografia(img_pil)
                            img_biofilm, biofilm_detectado = detectar_biofilm(img_pil)
                            img_display = img_pil
                            con.append(img_pil); con.append(img_thermal)
                        elif "RX" in modo or "ECG" in modo or "Farmacia" in modo or "Derma" in modo:
                            img_display = img_pil; con.append(img_pil)
                        else:
                            img_final, proc = anonymize_face(img_pil)
                            img_display = img_final; con.append(img_final)

                # Prompt
                coord_txt = ""
                if st.session_state.punto_cuerpo:
                    if isinstance(st.session_state.punto_cuerpo, dict) and 'label' in st.session_state.punto_cuerpo:
                         coord_txt = f"Zona cuerpo: {st.session_state.punto_cuerpo['label']}"
                    else:
                         coord_txt = f"Usuario tocó coordenadas mapa: {st.session_state.punto_cuerpo}"

                prompt = f"""
                Rol: APN / Especialista. Contexto: {contexto}. Modo: {modo}.
                {coord_txt}. Notas: "{notas}"
                
                INPUTS:
                - FARMACIA: {txt_meds}
                - DOCUMENTOS: {txt_tests}
                - VISUAL: {len(archivos)} archivos.
                
                INSTRUCCIONES CLAVE PARA {modo}:
                """
                
                if "Farmacia" in modo:
                    prompt += " IDENTIFICA EL FÁRMACO (Caja/Receta). Chequea dosis, posología y cruza con 'Notas' para buscar alergias."
                elif "RX" in modo:
                    prompt += " ACTÚA COMO RADIÓLOGO. Describe hallazgos en la imagen/video. Sé técnico y preciso."
                elif "ECG" in modo:
                    prompt += " ACTÚA COMO CARDIÓLOGO. Lectura sistemática del ECG (Frecuencia, Ritmo, Eje, Hipertrofias, Isquemia)."
                elif "Informes" in modo:
                    prompt += " RESUMEN CLÍNICO. Sintetiza los hallazgos de los documentos subidos."
                elif "Heridas" in modo:
                    prompt += " ANÁLISIS DE HERIDA. Usa termografía (si hay) y detecta tejido (Granulación/Esfacelo/Necrosis). Genera HTML BARRA TEJIDOS."
                elif "Dermatología" in modo:
                    prompt += " ANÁLISIS DERMATOLÓGICO. Describe lesión elemental (pápula, mácula, etc) y patrón (ABCDE si es nevus)."

                prompt += """
                \nOUTPUT:
                ### ⚡ RESUMEN
                SYNC_ALERT: [ALERTA SI HAY CONFLICTO]
                * **🚨 DIAGNÓSTICO SUGERIDO:**
                ...
                ### 📊 Análisis Detallado
                [Si es herida, HTML puro de barra tejidos]
                ...
                """
                
                resp = model.generate_content([prompt, *con] if con else prompt)
                st.session_state.resultado_analisis = resp.text
                
                if "Heridas" in modo and st.session_state.area_herida > 0:
                    st.session_state.historial_evolucion.append({
                        "Fecha": datetime.datetime.now().strftime("%d/%m %H:%M"), "Area": st.session_state.area_herida
                    })
                
                st.session_state.pdf_bytes = create_pdf(resp.text.replace("*","").replace("#",""))

                if img_display: st.image(img_display, caption="Evidencia", width=300)
                if img_thermal: st.image(img_thermal, caption="Termografía", width=300)

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
