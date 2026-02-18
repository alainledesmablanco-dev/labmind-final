import streamlit as st
import google.generativeai as genai
from PIL import Image, ImageDraw
import pypdf
import tempfile
import time
import os
from fpdf import FPDF
import datetime
import re
import cv2
import numpy as np
import extra_streamlit_components as stx
import pandas as pd
import uuid

# --- CONFIGURACIÓN ---
st.set_page_config(page_title="LabMind 50.0 (Derma Coin)", page_icon="🧬", layout="wide")

# --- ESTILOS CSS ---
st.markdown("""
<style>
    .stButton>button { width: 100%; border-radius: 8px; font-weight: bold; background-color: #0066cc; color: white; }
    
    /* BOTÓN BORRAR AUDIO (ESTILO MINIMALISTA) */
    div[data-testid="column"] .stButton button[kind="secondary"] {
        background-color: #ffebee;
        color: #c62828;
        border: 1px solid #ef9a9a;
        height: 52px; 
        margin-top: 0px;
    }

    /* CAJAS DE RESUMEN */
    .diagnosis-box { background-color: #e3f2fd; border-left: 6px solid #2196f3; padding: 15px; border-radius: 8px; margin-bottom: 10px; color: #0d47a1; font-family: sans-serif; }
    .action-box { background-color: #ffebee; border-left: 6px solid #f44336; padding: 15px; border-radius: 8px; margin-bottom: 10px; color: #b71c1c; font-family: sans-serif; }
    .material-box { background-color: #e8f5e9; border-left: 6px solid #4caf50; padding: 15px; border-radius: 8px; margin-bottom: 15px; color: #1b5e20; font-family: sans-serif; }

    /* BARRA DE TEJIDOS */
    .tissue-labels { display: flex; width: 100%; margin-bottom: 2px; }
    .tissue-label-text { font-size: 0.75rem; text-align: center; font-weight: bold; color: #555; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
    .tissue-bar-container { display: flex; width: 100%; height: 20px; border-radius: 10px; overflow: hidden; margin-bottom: 15px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
    .tissue-gran { background-color: #ef5350; height: 100%; }
    .tissue-slough { background-color: #fdd835; height: 100%; }
    .tissue-nec { background-color: #212121; height: 100%; }
    
    .sync-alert { border: 2px solid #d32f2f; padding: 15px; border-radius: 10px; background-color: #fff8f8; color: #b71c1c; font-weight: bold; margin-bottom: 10px; animation: pulse 2s infinite; }
    
    /* HISTORIAL */
    .history-card { border: 1px solid #ddd; padding: 10px; border-radius: 8px; margin-bottom: 10px; background-color: #f9f9f9; }
    
    [data-testid='stFileUploaderDropzone'] div div span { display: none; }
    [data-testid='stFileUploaderDropzone'] div div::after { content: "📂 Adjuntar"; font-size: 0.9rem; color: #555; display: block; }
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
if "punto_cuerpo" not in st.session_state: st.session_state.punto_cuerpo = "No especificado"
if "chat_messages" not in st.session_state: st.session_state.chat_messages = []
if "history_db" not in st.session_state: st.session_state.history_db = []

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

def medir_herida_con_referencia(pil_image, usar_moneda=False):
    area_final = 0.0
    img_annotated = pil_image.copy()
    try:
        img_np = np.array(pil_image.convert('RGB'))
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        pixels_per_cm = 0
        circles = cv2.HoughCircles(cv2.GaussianBlur(gray, (9, 9), 2), cv2.HOUGH_GRADIENT, 1.2, 50, param1=100, param2=30, minRadius=20, maxRadius=300)
        moneda_detectada = False
        if circles is not None and usar_moneda:
            circles = np.round(circles[0, :]).astype("int")
            for (x, y, r) in circles:
                cv2.circle(img_bgr, (x, y), r, (0, 255, 0), 4)
                cv2.putText(img_bgr, "1 EUR", (x - 20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                pixels_per_cm = (r * 2) / 2.325
                moneda_detectada = True
                break
        if pixels_per_cm == 0: pixels_per_cm = 100.0 
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        mask1 = cv2.inRange(hsv, np.array([0, 50, 50]), np.array([10, 255, 255]))
        mask2 = cv2.inRange(hsv, np.array([170, 50, 50]), np.array([180, 255, 255]))
        mask = mask1 + mask2
        kernel = np.ones((5,5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        area_pixels_total = 0
        for c in contours:
            if cv2.contourArea(c) > 500:
                area_pixels_total += cv2.contourArea(c)
                cv2.drawContours(img_bgr, [c], -1, (0, 0, 255), 2)
        if pixels_per_cm > 0: area_final = area_pixels_total * ((1 / pixels_per_cm) ** 2)
        img_annotated = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        return area_final, img_annotated, moneda_detectada
    except: return 0.0, pil_image, False

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
    clean = re.sub(r'<[^>]+>', '', texto_analisis).replace('€','EUR').replace('’',"'").replace('“','"').replace('”','"')
    pdf.multi_cell(0,5, clean.encode('latin-1','replace').decode('latin-1'))
    return pdf.output(dest='S').encode('latin-1')

# ==========================================
#      INTERFAZ DE USUARIO
# ==========================================

st.title("🩺 LabMind 50.0")
col_left, col_center, col_right = st.columns([1, 2, 1])

# --- COLUMNA 1: CONTEXTO GLOBAL ---
with col_left:
    st.subheader("📍 Datos Paciente")
    zonas_cuerpo = [
        "No especificada", "--- CABEZA ---", "Cara", "Cuello", "--- TRONCO ---",
        "Pecho", "Abdomen", "Espalda", "--- PELVIS ---", "Sacro/Glúteo", "Genitales",
        "--- EXTREMIDADES ---", "Brazo", "Mano", "Pierna", "Talón", "Pie"
    ]
    seleccion_zona = st.selectbox("Zona anatómica:", zonas_cuerpo)
    st.session_state.punto_cuerpo = seleccion_zona
    
    st.divider()
    
    with st.expander("📚 Protocolo de Unidad (Opcional)", expanded=False):
        st.caption("Sube el PDF/Foto de tu guía para que la IA la respete.")
        proto_file = st.file_uploader("Subir", type=["pdf", "jpg", "png"], key="global_proto")

# --- COLUMNA 2: NÚCLEO CENTRAL ---
with col_center:
    tab_analisis, tab_historial = st.tabs(["🔍 Analizar Caso", "🗂️ Historial Guardado"])
    
    with tab_analisis:
        st.subheader("1. Selección de Modo")
        modo = st.selectbox("Especialidad:", 
                     ["🧩 Integral (Analizar Todo)",
                      "🩹 Heridas / Úlceras", 
                      "🧴 Dermatología", 
                      "💊 Farmacia (Interacciones)", 
                      "📈 ECG (Cardiología)", 
                      "💀 RX / TAC / RMN (Imagen)", 
                      "📂 Analizar Informes"])
        contexto = st.selectbox("🏥 Contexto:", ["Hospitalización", "Residencia", "Urgencias", "UCI", "Domicilio"])
        
        st.markdown("---")
        
        archivos = []
        meds_files = None; labs_files = None; reports_files = None; ecg_files = None; rad_files = None 
        usar_moneda = False
        mostrar_imagenes = False # Default global
        
        if modo == "🧩 Integral (Analizar Todo)":
            st.info("🧩 **Modo Integral**: Sube cualquier evidencia.")
            with st.expander("📂 Documentación Clínica (Desplegar)", expanded=False):
                c1, c2, c3 = st.columns(3)
                meds_files = c1.file_uploader("💊 Fármacos", accept_multiple_files=True, key="int_meds")
                labs_files = c2.file_uploader("📊 Analíticas", accept_multiple_files=True, key="int_labs")
                reports_files = c3.file_uploader("📄 Informes", accept_multiple_files=True, key="int_reports")
                st.markdown("---")
                c4, c5 = st.columns(2)
                ecg_files = c4.file_uploader("📈 ECG", accept_multiple_files=True, key="int_ecg")
                rad_files = c5.file_uploader("💀 RX/TAC", accept_multiple_files=True, key="int_rad")
            
            st.write("📸 **Estado Visual Paciente (Foto/Video):**")
            mostrar_imagenes = st.checkbox("👁️ Mostrar Análisis Visual (Biofilm/Térmica)", value=False)
            
            fuente = st.radio("Fuente:", ["📁 Archivo", "📸 WebCam"], horizontal=True, label_visibility="collapsed")
            if fuente == "📸 WebCam":
                if f := st.camera_input("Foto Paciente"): archivos.append(("cam", f))
            else:
                if fs := st.file_uploader("Subir Foto Paciente", type=['jpg','png','mp4','mov'], accept_multiple_files=True, key="int_main"):
                    for f in fs: archivos.append(("video" if "video" in f.type else "img", f))

        elif modo == "🩹 Heridas / Úlceras":
            st.info("🩹 **Modo Heridas**")
            # --- ORDEN: MONEDA -> VISUAL ---
            usar_moneda = st.checkbox("🪙 Usar moneda de 1€ para medir")
            mostrar_imagenes = st.checkbox("👁️ Mostrar Análisis Visual (Biofilm/Térmica)", value=False)
            
            with st.expander("⏮️ Ver Evolución (Subir Foto/Video Previo)", expanded=False):
                if prev := st.file_uploader("Estado Previo", type=['jpg','png','mp4','mov'], accept_multiple_files=True, key="w_prev"):
                    for p in prev: archivos.append(("prev_video" if "video" in p.type else "prev_img", p))
            with st.expander("💊 Medicación / Analítica (Opcional)", expanded=False):
                c1, c2 = st.columns(2)
                meds_files = c1.file_uploader("Medicación", accept_multiple_files=True, key="w_meds")
                labs_files = c2.file_uploader("Analítica", accept_multiple_files=True, key="w_labs")
            st.write("📸 **Estado ACTUAL (Foto/Video):**")
            fuente = st.radio("Fuente:", ["📁 Archivo", "📸 WebCam"], horizontal=True, label_visibility="collapsed")
            if fuente == "📸 WebCam":
                if f := st.camera_input("Foto Herida"): archivos.append(("cam", f))
            else:
                if fs := st.file_uploader("Subir Foto/Video Actual", type=['jpg','png','mp4','mov'], accept_multiple_files=True, key="w_img"):
                    for f in fs: archivos.append(("video" if "video" in f.type else "img", f))

        elif modo == "🧴 Dermatología":
            st.info("🧴 **Modo Dermatología**")
            # --- ORDEN IGUAL QUE HERIDAS ---
            usar_moneda = st.checkbox("🪙 Usar moneda de 1€ para medir")
            mostrar_imagenes = st.checkbox("👁️ Mostrar Análisis Visual (Biofilm/Térmica)", value=False)
            
            with st.expander("⏮️ Ver Evolución (Subir Foto/Video Previo)", expanded=False):
                if prev := st.file_uploader("Estado Previo", type=['jpg','png','mp4','mov'], accept_multiple_files=True, key="d_prev"):
                    for p in prev: archivos.append(("prev_video" if "video" in p.type else "prev_img", p))
            st.write("📸 **Estado ACTUAL (Foto/Video):**")
            if fs := st.file_uploader("Subir Foto/Video Actual", type=['jpg','png','mp4','mov'], accept_multiple_files=True, key="d_img"):
                for f in fs: archivos.append(("video" if "video" in f.type else "img", f))

        elif modo == "💊 Farmacia (Interacciones)":
            st.info("💊 **Modo Farmacia**")
            meds_files = st.file_uploader("Receta/Caja", accept_multiple_files=True, key="p_docs")

        elif modo == "📈 ECG (Cardiología)":
            st.info("📈 **Modo Cardiología**")
            if fs := st.file_uploader("Imagen ECG", type=['jpg','png','pdf'], accept_multiple_files=True, key="ecg_docs"):
                for f in fs: archivos.append(("img", f))

        elif modo == "💀 RX / TAC / RMN (Imagen)":
            st.info("💀 **Modo Radiología**")
            if fs := st.file_uploader("Video/Imagen RX", type=['jpg','png','mp4','mov'], accept_multiple_files=True, key="rx_docs"):
                for f in fs: archivos.append(("video" if "video" in f.type else "img", f))

        elif modo == "📂 Analizar Informes":
            st.info("📂 **Modo Informes**")
            reports_files = st.file_uploader("PDFs/Fotos", accept_multiple_files=True, key="rep_docs")

        st.markdown("---")
        
        # --- AUDIO ---
        c_del, c_audio, c_tag = st.columns([0.15, 0.45, 0.40])
        with c_del:
            st.write("") 
            st.write("") 
            if st.button("❌", help="Borrar audio", key="btn_clear_audio", type="secondary"):
                st.rerun()
        with c_audio:
            audio_val = st.audio_input("🎙️ Voz", key="audio_recorder")
        with c_tag:
            st.write("") 
            nota_historial = st.text_input("Etiqueta Historial:", placeholder="Ej: Cama 304", label_visibility="collapsed")

        notas = st.text_area("Notas Clínicas (Texto):", height=60, placeholder="Escribe síntomas, alergias...")

        if st.button("🚀 ANALIZAR", type="primary"):
            st.session_state.log_privacidad = []; st.session_state.area_herida = 0.0
            st.session_state.chat_messages = [] 
            
            with st.spinner(f"🧠 Analizando {modo}..."):
                try:
                    genai.configure(api_key=st.session_state.api_key)
                    model = genai.GenerativeModel("models/gemini-3-flash-preview")
                    
                    con = []; txt_meds = ""; txt_labs = ""; txt_reports = ""; txt_proto = ""; txt_rad_desc = ""

                    if proto_file:
                        if "pdf" in proto_file.type: r = pypdf.PdfReader(proto_file); txt_proto += "".join([p.extract_text() for p in r.pages])
                        else: con.append(Image.open(proto_file))
                    
                    for file_list, var_name in [(meds_files, "txt_meds"), (labs_files, "txt_labs"), (reports_files, "txt_reports")]:
                        if file_list:
                            temp_txt = ""
                            for f in file_list:
                                if "pdf" in f.type: 
                                    try: r = pypdf.PdfReader(f); temp_txt += "".join([p.extract_text() for p in r.pages])
                                    except: pass
                                else: con.append(Image.open(f))
                            if var_name == "txt_meds": txt_meds = temp_txt
                            elif var_name == "txt_labs": txt_labs = temp_txt
                            elif var_name == "txt_reports": txt_reports = temp_txt

                    if ecg_files:
                        for f in ecg_files:
                            if "pdf" in f.type: r = pypdf.PdfReader(f); txt_reports += "\n[ECG PDF]: " + "".join([p.extract_text() for p in r.pages])
                            else: con.append(Image.open(f)); txt_reports += "\n[IMAGEN ECG ADJUNTA]"

                    if rad_files:
                        for f in rad_files:
                            if "pdf" in f.type: r = pypdf.PdfReader(f); txt_reports += "\n[INFORME RX]: " + "".join([p.extract_text() for p in r.pages])
                            elif "video" in f.type:
                                with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tf: tf.write(f.read()); tp = tf.name
                                vf = genai.upload_file(path=tp); 
                                while vf.state.name == "PROCESSING": time.sleep(1); vf = genai.get_file(vf.name)
                                con.append(vf); os.remove(tp)
                                txt_rad_desc += "\n[VIDEO RADIOLÓGICO]"
                            else: con.append(Image.open(f)); txt_rad_desc += "\n[IMAGEN RX]"

                    if audio_val: con.append(genai.upload_file(audio_val, mime_type="audio/wav"))
                    
                    img_display = None; img_thermal = None; img_biofilm = None; biofilm_detectado = False
                    
                    for label, a in archivos:
                        is_video = "video" in label
                        if is_video:
                            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tf: tf.write(a.read()); tp = tf.name
                            vf = genai.upload_file(path=tp)
                            while vf.state.name == "PROCESSING": time.sleep(1); vf = genai.get_file(vf.name)
                            con.append(vf); os.remove(tp)
                        else: 
                            img_pil = Image.open(a)
                            # --- LÓGICA DE PROCESADO VISUAL ---
                            if ("Heridas" in modo or "Dermatología" in modo) and "prev" not in label: 
                                area, img_medida, coin = medir_herida_con_referencia(img_pil, usar_moneda)
                                if area > 0: st.session_state.area_herida = area
                                img_thermal = procesar_termografia(img_pil)
                                img_biofilm, biofilm_detectado = detectar_biofilm(img_pil)
                                img_display = img_medida
                                con.append(img_pil); con.append(img_thermal)
                            elif modo == "🧩 Integral (Analizar Todo)":
                                img_final, proc = anonymize_face(img_pil)
                                img_display = img_final; con.append(img_final)
                            elif "RX" in modo or "ECG" in modo or "Farmacia" in modo:
                                img_display = img_pil; con.append(img_pil)
                            else: 
                                img_final, proc = anonymize_face(img_pil)
                                if "prev" not in label: img_display = img_final
                                con.append(img_final)

                    # Prompt
                    prompt = f"""
                    Rol: APN / Especialista. Contexto: {contexto}. Modo: {modo}.
                    Zona Anatómica: {st.session_state.punto_cuerpo}.
                    Notas: "{notas}"
                    
                    INPUTS:
                    - PROTOCOLO UNIDAD: {txt_proto}
                    - FARMACIA: {txt_meds}
                    - ANALÍTICAS: {txt_labs}
                    - INFORMES: {txt_reports}
                    - RADIOLOGÍA: {txt_rad_desc}
                    
                    INSTRUCCIONES DE FORMATO:
                    1. RESUMEN ESQUEMÁTICO:
                    <div class="diagnosis-box"><b>🚨 DIAGNÓSTICO:</b><br>[Texto]</div>
                    <div class="action-box"><b>⚡ ACCIÓN INMEDIATA:</b><br>[Texto]</div>
                    <div class="material-box"><b>🛠️ CURA:</b><br>[Texto]</div>
                    
                    2. Si es Heridas, BARRA TEJIDOS (Etiquetas fuera):
                    <div class="tissue-labels">
                        <div style="width:G%" class="tissue-label-text">Granulación G%</div>
                        <div style="width:E%" class="tissue-label-text">Esfacelos E%</div>
                        <div style="width:N%" class="tissue-label-text">Necrosis N%</div>
                    </div>
                    <div class="tissue-bar-container">
                       <div class="tissue-gran" style="width:G%"></div>
                       <div class="tissue-slough" style="width:E%"></div>
                       <div class="tissue-nec" style="width:N%"></div>
                    </div>
                    """
                    if "Farmacia" in modo: prompt += " CHECK DOSIS."
                    elif "RX" in modo: prompt += " RADIOLOGÍA TÉCNICA."

                    prompt += """\nLuego detalle completo."""
                    
                    for attempt in range(3):
                        try:
                            resp = model.generate_content([prompt, *con] if con else prompt)
                            st.session_state.resultado_analisis = resp.text
                            
                            new_entry = {
                                "id": str(uuid.uuid4()),
                                "date": datetime.datetime.now().strftime("%d/%m %H:%M"),
                                "mode": modo,
                                "note": nota_historial if nota_historial else "Sin etiqueta",
                                "result": resp.text
                            }
                            st.session_state.history_db.append(new_entry)
                            break 
                        except Exception as e:
                            if "429" in str(e) and attempt < 2: time.sleep(5); continue
                            elif attempt == 2: st.error("⚠️ Saturado. Reintenta.")
                            else: raise e
                    
                    if "Heridas" in modo and st.session_state.area_herida > 0:
                        st.session_state.historial_evolucion.append({
                            "Fecha": datetime.datetime.now().strftime("%d/%m"), "Area": st.session_state.area_herida
                        })
                    
                    if st.session_state.resultado_analisis:
                        st.session_state.pdf_bytes = create_pdf(st.session_state.resultado_analisis)

                    # --- VISUALIZACIÓN CONDICIONAL ---
                    if mostrar_imagenes:
                        if img_display: st.image(img_display, caption="Evidencia", width=300)
                        if img_thermal: st.image(img_thermal, caption="Termografía", width=300)

                except Exception as e: st.error(f"Error: {e}")

        if st.session_state.resultado_analisis:
            txt = st.session_state.resultado_analisis.replace("```html", "").replace("```", "")
            sync_match = re.search(r'SYNC_ALERT: (.*)', txt)
            if sync_match and len(sync_match.group(1).strip()) > 5:
                st.markdown(f'<div class="sync-alert">⚠️ {sync_match.group(1)}</div>', unsafe_allow_html=True)
            st.markdown(txt, unsafe_allow_html=True)
            
            st.markdown("---")
            
            with st.expander("💬 Abrir Asistente Clínico (Chat)", expanded=False):
                for message in st.session_state.chat_messages:
                    with st.chat_message(message["role"]): st.markdown(message["content"])

                if prompt := st.chat_input("Pregunta sobre el caso..."):
                    st.session_state.chat_messages.append({"role": "user", "content": prompt})
                    with st.chat_message("user"): st.markdown(prompt)
                    with st.chat_message("assistant"):
                        try:
                            chat_model = genai.GenerativeModel("models/gemini-3-flash-preview")
                            ctx = f"CONTEXTO: {st.session_state.resultado_analisis}\nPREGUNTA: {prompt}"
                            full_resp = chat_model.generate_content(ctx)
                            st.markdown(full_resp.text)
                            st.session_state.chat_messages.append({"role": "assistant", "content": full_resp.text})
                        except Exception as e: st.error(f"Error chat: {e}")

        if st.session_state.pdf_bytes:
            st.download_button("📥 Descargar Informe PDF", st.session_state.pdf_bytes, "informe.pdf", "application/pdf")

    with tab_historial:
        st.subheader("🗂️ Historial de Análisis")
        if not st.session_state.history_db:
            st.info("No hay análisis guardados.")
        else:
            col_del_all, _ = st.columns([1, 4])
            if col_del_all.button("🗑️ Borrar TODO", type="primary"):
                st.session_state.history_db = []
                st.rerun()
            st.divider()
            items_to_delete = []
            for item in reversed(st.session_state.history_db):
                with st.container():
                    c_check, c_content = st.columns([0.5, 9])
                    if c_check.checkbox("Sel", key=f"chk_{item['id']}", label_visibility="collapsed"):
                        items_to_delete.append(item['id'])
                    with c_content.expander(f"📅 {item['date']} | {item['mode']} | {item['note']}"):
                        st.markdown(item['result'], unsafe_allow_html=True)
            if items_to_delete:
                st.markdown("---")
                if st.button(f"🗑️ Borrar {len(items_to_delete)} seleccionados"):
                    st.session_state.history_db = [i for i in st.session_state.history_db if i['id'] not in items_to_delete]
                    st.rerun()

# --- COLUMNA 3: ESTADÍSTICAS ---
with col_right:
    with st.expander("📈 Pronóstico (Ver Gráfica)", expanded=False):
        if len(st.session_state.historial_evolucion) > 0:
            df = pd.DataFrame(st.session_state.historial_evolucion)
            st.line_chart(df.set_index("Fecha"))
            pred = predecir_cierre()
            st.markdown(f'<div class="prediction-box">🔮 <b>IA Supervivencia:</b><br>{pred}</div>', unsafe_allow_html=True)
        else:
            st.caption("Sin datos suficientes.")

st.divider()
if st.button("🔒 Salir"):
    cookie_manager.delete("labmind_secret_key")
    st.session_state.autenticado = False; st.rerun()
