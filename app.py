import streamlit as st
import google.generativeai as genai
from PIL import Image
import pypdf
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
st.set_page_config(page_title="LabMind 66.0 (Callback Fix)", page_icon="🧬", layout="wide")

# --- ESTILOS CSS ---
st.markdown("""
<style>
    .block-container { padding-top: 0.5rem !important; padding-bottom: 2rem !important; }
    div[data-testid="stVerticalBlock"] > div { gap: 0rem !important; }
    div[data-testid="stSelectbox"] { margin-bottom: -15px !important; }
    .stButton>button { width: 100%; border-radius: 8px; font-weight: bold; background-color: #0066cc; color: white; margin-top: 10px; }
    
    /* CAJAS CLÍNICAS */
    .diagnosis-box { background-color: #e3f2fd; border-left: 6px solid #2196f3; padding: 15px; border-radius: 8px; margin-bottom: 10px; color: #0d47a1; font-family: sans-serif; }
    .action-box { background-color: #ffebee; border-left: 6px solid #f44336; padding: 15px; border-radius: 8px; margin-bottom: 10px; color: #b71c1c; font-family: sans-serif; }
    .material-box { background-color: #e8f5e9; border-left: 6px solid #4caf50; padding: 15px; border-radius: 8px; margin-bottom: 15px; color: #1b5e20; font-family: sans-serif; }
    
    .tissue-labels { display: flex; width: 100%; margin-bottom: 2px; }
    .tissue-label-text { font-size: 0.75rem; text-align: center; font-weight: bold; color: #555; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
    .tissue-bar-container { display: flex; width: 100%; height: 20px; border-radius: 10px; overflow: hidden; margin-bottom: 15px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
    .tissue-gran { background-color: #ef5350; height: 100%; }
    .tissue-slough { background-color: #fdd835; height: 100%; }
    .tissue-nec { background-color: #212121; height: 100%; }
    
    .proto-success { background-color: #e8f5e9; color: #2e7d32; padding: 8px; border-radius: 5px; font-size: 0.85rem; border: 1px solid #c8e6c9; margin-bottom: 5px; }
    .pull-up { margin-top: -25px !important; margin-bottom: 5px !important; height: 1px !important; display: block !important; }
    
    [data-testid='stFileUploaderDropzone'] { padding: 5px !important; min-height: 60px; }
    [data-testid='stFileUploaderDropzone'] div div span { display: none; }
    [data-testid='stFileUploaderDropzone'] div div::after { content: "📂 Adjuntar"; font-size: 0.9rem; color: #555; display: block; }
</style>
""", unsafe_allow_html=True)

# --- GESTOR DE ESTADO & COOKIES ---
cookie_manager = stx.CookieManager()
time.sleep(0.1)

# Inicialización de variables de sesión
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

# --- CARGA INICIAL DE PREFERENCIAS ---
if "prefs_loaded" not in st.session_state:
    try:
        c_mon = cookie_manager.get("pref_moneda")
        c_vis = cookie_manager.get("pref_visual")
        c_src = cookie_manager.get("pref_fuente")
        st.session_state.pref_moneda = True if c_mon == "True" else False
        st.session_state.pref_visual = True if c_vis == "True" else False
        st.session_state.pref_fuente = 1 if c_src == "WebCam" else 0
        st.session_state.prefs_loaded = True
    except:
        st.session_state.pref_moneda = False
        st.session_state.pref_visual = False
        st.session_state.pref_fuente = 0
        st.session_state.prefs_loaded = True

# --- CALLBACKS PARA GUARDAR COOKIES (CLAVE PARA EVITAR ERRORES) ---
def update_cookie_moneda():
    val = st.session_state.get("chk_moneda_global", False)
    st.session_state.pref_moneda = val
    cookie_manager.set("pref_moneda", str(val), expires_at=datetime.datetime.now() + datetime.timedelta(days=30))

def update_cookie_visual():
    val = st.session_state.get("chk_visual_global", False)
    st.session_state.pref_visual = val
    cookie_manager.set("pref_visual", str(val), expires_at=datetime.datetime.now() + datetime.timedelta(days=30))

def update_cookie_fuente():
    # Detectar cuál widget disparó el cambio
    val = "Archivo" # Default
    # Revisamos los keys posibles
    if "rad_src_wounds" in st.session_state:
        val = st.session_state.rad_src_wounds
    elif "rad_src_integral" in st.session_state:
        val = st.session_state.rad_src_integral
    
    idx = 1 if val == "📸 WebCam" else 0
    st.session_state.pref_fuente = idx
    cookie_manager.set("pref_fuente", "WebCam" if idx==1 else "Archivo", expires_at=datetime.datetime.now() + datetime.timedelta(days=30))


# --- LOGIN ---
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
#      FUNCIONES
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

st.title("🩺 LabMind 66.0")
col_left, col_center, col_right = st.columns([1, 2, 1])

# --- COLUMNA 1 ---
with col_left:
    st.subheader("📍 Datos Paciente")
    zonas_cuerpo = ["No especificada", "--- CABEZA ---", "Cara", "Cuello", "--- TRONCO ---", "Pecho", "Abdomen", "Espalda", "--- PELVIS ---", "Sacro/Glúteo", "Genitales", "--- EXTREMIDADES ---", "Brazo", "Mano", "Pierna", "Talón", "Pie"]
    seleccion_zona = st.selectbox("Zona anatómica:", zonas_cuerpo)
    st.session_state.punto_cuerpo = seleccion_zona
    
    with st.expander("📚 Protocolo Unidad", expanded=False):
        fixed_proto_path = None; detected_name = ""
        if os.path.exists("protocolo.pdf"): fixed_proto_path = "protocolo.pdf"; detected_name = "PDF"
        elif os.path.exists("protocolo.jpg"): fixed_proto_path = "protocolo.jpg"; detected_name = "JPG"
        elif os.path.exists("protocolo.png"): fixed_proto_path = "protocolo.png"; detected_name = "PNG"
        
        using_fixed_proto = False
        if fixed_proto_path:
            st.markdown(f'<div class="proto-success">✅ <b>protocolo.{detected_name.lower()}</b> detectado</div>', unsafe_allow_html=True)
            using_fixed_proto = True
        else:
            st.caption("Guarda 'protocolo.jpg' o '.pdf' en la carpeta.")

        proto_uploaded = st.file_uploader("Subir (Sobrescribe Fijo)", type=["pdf", "jpg", "png"], key="global_proto")

# --- COLUMNA 2 ---
with col_center:
    tab_analisis, tab_historial = st.tabs(["🔍 Analizar Caso", "🗂️ Historial Guardado"])
    
    with tab_analisis:
        st.subheader("1. Selección de Modo")
        
        modo = st.selectbox("Especialidad:", 
                     ["🩹 Heridas / Úlceras", "🧴 Dermatología", "🧩 Integral (Analizar Todo)", "💊 Farmacia", "📈 ECG", "💀 RX/TAC", "📂 Informes"])
        contexto = st.selectbox("🏥 Contexto:", ["Hospitalización", "Residencia", "Urgencias", "UCI", "Domicilio"])
        
        st.markdown('<div class="pull-up"></div>', unsafe_allow_html=True)
        
        archivos = []
        meds_files = None; labs_files = None; reports_files = None; ecg_files = None; rad_files = None 
        
        # LÓGICA MODOS
        if modo == "🧩 Integral (Analizar Todo)":
            with st.expander("📂 Documentación", expanded=False):
                c1, c2 = st.columns(2)
                meds_files = c1.file_uploader("💊 Fármacos", accept_multiple_files=True, key="int_meds")
                labs_files = c2.file_uploader("📊 Analíticas", accept_multiple_files=True, key="int_labs")
            st.write("📸 **Visual:**")
            
            # --- WIDGETS GLOBALES (SIN ERROR DUPLICATE) ---
            mostrar_imagenes = st.checkbox("👁️ Mostrar Análisis Visual", value=st.session_state.pref_visual, key="chk_visual_global", on_change=update_cookie_visual)
            
            fuente_label = st.radio("Fuente:", ["📁 Archivo", "📸 WebCam"], horizontal=True, label_visibility="collapsed", index=st.session_state.pref_fuente, key="rad_src_integral", on_change=update_cookie_fuente)
            
            if fuente_label == "📸 WebCam":
                if f := st.camera_input("Foto Paciente"): archivos.append(("cam", f))
            else:
                if fs := st.file_uploader("Subir", type=['jpg','png','mp4','mov'], accept_multiple_files=True, key="int_main"):
                    for f in fs: archivos.append(("video" if "video" in f.type else "img", f))

        elif modo == "🩹 Heridas / Úlceras" or modo == "🧴 Dermatología":
            # --- WIDGETS GLOBALES ---
            usar_moneda = st.checkbox("🪙 Usar moneda de 1€ para medir", value=st.session_state.pref_moneda, key="chk_moneda_global", on_change=update_cookie_moneda)
            mostrar_imagenes = st.checkbox("👁️ Mostrar Análisis Visual (Biofilm/Térmica)", value=st.session_state.pref_visual, key="chk_visual_global", on_change=update_cookie_visual)
            
            with st.expander("⏮️ Ver Evolución", expanded=False):
                prev = st.file_uploader("Foto Previa", type=['jpg','png'], accept_multiple_files=True, key="w_prev")
                c_d, c_a, c_b = st.columns([0.4,0.4,0.2])
                with c_d: d_m = st.date_input("Fecha", value=datetime.date.today()-datetime.timedelta(days=7))
                with c_a: a_m = st.number_input("Área (cm²)", min_value=0.0, step=0.1)
                with c_b: 
                    st.write(""); st.write("")
                    if st.button("➕", key="btn_add"): st.session_state.historial_evolucion.append({"Fecha": d_m.strftime("%d/%m"), "Area": a_m})
                if prev: 
                    for p in prev: archivos.append(("prev_img", p))

            with st.expander("💊 Contexto (Opcional)", expanded=False):
                meds_files = st.file_uploader("Docs", accept_multiple_files=True, key="w_meds")
            
            st.write("📸 **Estado ACTUAL:**")
            
            fuente_label = st.radio("Fuente:", ["📁 Archivo", "📸 WebCam"], horizontal=True, label_visibility="collapsed", index=st.session_state.pref_fuente, key="rad_src_wounds", on_change=update_cookie_fuente)
            
            if fuente_label == "📸 WebCam":
                if f := st.camera_input("Foto"): archivos.append(("cam", f))
            else:
                if fs := st.file_uploader("Subir", type=['jpg','png','mp4','mov'], accept_multiple_files=True, key="w_img"):
                    for f in fs: archivos.append(("video" if "video" in f.type else "img", f))

        # OTROS MODOS
        elif modo == "💊 Farmacia": meds_files = st.file_uploader("Receta", accept_multiple_files=True, key="p_docs")
        elif modo == "📈 ECG": 
            if fs:=st.file_uploader("ECG", type=['jpg','pdf'], accept_multiple_files=True): 
                for f in fs: archivos.append(("img",f))
        elif modo == "💀 RX/TAC": 
            if fs:=st.file_uploader("RX", type=['jpg','png'], accept_multiple_files=True): 
                for f in fs: archivos.append(("img",f))
        elif modo == "📂 Informes": reports_files = st.file_uploader("PDFs", accept_multiple_files=True, key="rep_docs")

        st.markdown('<div class="pull-up"></div>', unsafe_allow_html=True)
        
        # --- ORDEN VERTICAL: AUDIO -> NOTAS -> ETIQUETA ---
        audio_val = st.audio_input("🎙️ Notas de Voz", key="audio_recorder", label_visibility="collapsed")
        notas = st.text_area("Notas Clínicas:", height=60, placeholder="Escribe síntomas...")
        nota_historial = st.text_input("🏷️ Etiqueta Historial (Opcional):", placeholder="Ej: Cama 304", label_visibility="collapsed")

        # BOTÓN ANALIZAR
        if st.button("🚀 ANALIZAR", type="primary"):
            st.session_state.log_privacidad = []; st.session_state.area_herida = 0.0
            st.session_state.chat_messages = [] 
            
            with st.spinner(f"🧠 Analizando {modo}..."):
                try:
                    genai.configure(api_key=st.session_state.api_key)
                    model = genai.GenerativeModel("models/gemini-3-flash-preview")
                    
                    con = []; txt_meds = ""; txt_labs = ""; txt_reports = ""; txt_proto = ""

                    # --- GESTION PROTOCOLO (PDF / IMG) ---
                    final_proto_obj = None; is_local = False
                    if proto_uploaded: final_proto_obj = proto_uploaded
                    elif using_fixed_proto and fixed_proto_path: final_proto_obj = fixed_proto_path; is_local = True

                    if final_proto_obj:
                        is_pdf = False
                        if is_local:
                            if fixed_proto_path.endswith(".pdf"): is_pdf = True
                            file_handle = open(fixed_proto_path, "rb")
                        else:
                            if "pdf" in final_proto_obj.type: is_pdf = True
                            file_handle = final_proto_obj

                        if is_pdf:
                            r = pypdf.PdfReader(file_handle)
                            txt_proto += "".join([p.extract_text() for p in r.pages])
                        else:
                            img_proto = Image.open(file_handle)
                            con.append(img_proto)
                            txt_proto = "[PROTOCOLO EN IMAGEN ADJUNTA - IDENTIFICA MARCAS COMERCIALES]"
                    
                    # Docs
                    for fs in [meds_files, labs_files, reports_files]:
                        if fs: 
                            for f in fs:
                                if "pdf" in f.type: r=pypdf.PdfReader(f); txt_meds += "".join([p.extract_text() for p in r.pages])
                                else: con.append(Image.open(f))

                    if audio_val: con.append(genai.upload_file(audio_val, mime_type="audio/wav"))
                    
                    img_display = None; img_thermal = None; img_prev_display = None
                    
                    for label, a in archivos:
                        if "video" in label: pass 
                        else: 
                            img_pil = Image.open(a)
                            if ("Heridas" in modo or "Dermatología" in modo):
                                if "prev" in label:
                                    img_prev_display = img_pil; con.append(img_pil)
                                else:
                                    area, img_medida, coin = medir_herida_con_referencia(img_pil, usar_moneda)
                                    if area > 0: st.session_state.area_herida = area
                                    img_thermal = procesar_termografia(img_pil)
                                    detectar_biofilm(img_pil)
                                    img_display = img_medida
                                    con.append(img_pil); con.append(img_thermal)
                            else: 
                                img_final, proc = anonymize_face(img_pil)
                                img_display = img_final; con.append(img_final)

                    # --- PROMPT MEJORADO ---
                    prompt = f"""
                    Rol: APN / Especialista. Contexto: {contexto}. Modo: {modo}.
                    Zona Anatómica: {st.session_state.punto_cuerpo}. Notas: "{notas}"
                    
                    INPUTS DISPONIBLES (SOLO PARA ANÁLISIS INTERNO, NO REPETIR EN SALIDA):
                    - PROTOCOLO UNIDAD: {txt_proto}
                    - FARMACIA/HISTORIAL: {txt_meds}
                    - ANALÍTICAS: {txt_labs}
                    - INFORMES: {txt_reports}
                    
                    INSTRUCCIONES DE SALIDA (ESTRICTO):
                    1. **PROHIBIDO** repetir el listado de fármacos o el texto del protocolo fuera de las cajas.
                    2. En la caja "CURA", **EXTRAE Y RECOMIENDA** únicamente los productos con su **MARCA COMERCIAL** que aparezcan en el protocolo/farmacia adjuntos y sean adecuados para este caso (Ej: Si protocolo tiene Aquacel, pon "Aquacel Ag"). Si no hay marcas, usa genéricos.
                    
                    FORMATO HTML REQUERIDO:
                    <div class="diagnosis-box"><b>🚨 DIAGNÓSTICO:</b><br>[Diagnóstico preciso]</div>
                    <div class="action-box"><b>⚡ ACCIÓN:</b><br>[Acciones inmediatas]</div>
                    <div class="material-box"><b>🛠️ CURA (Materiales del Protocolo):</b><br>[Lista de productos con MARCA COMERCIAL detectada]</div>
                    
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

                    resp = model.generate_content([prompt, *con] if con else prompt)
                    st.session_state.resultado_analisis = resp.text
                    
                    # Guardar Historial
                    new_entry = { "id": str(uuid.uuid4()), "date": datetime.datetime.now().strftime("%d/%m %H:%M"), "mode": modo, "note": nota_historial if nota_historial else "Sin etiqueta", "result": resp.text }
                    st.session_state.history_db.append(new_entry)
                    
                    if "Heridas" in modo and st.session_state.area_herida > 0:
                        st.session_state.historial_evolucion.append({"Fecha": datetime.datetime.now().strftime("%d/%m"), "Area": st.session_state.area_herida})
                    
                    if st.session_state.resultado_analisis:
                        st.session_state.pdf_bytes = create_pdf(st.session_state.resultado_analisis)

                    if mostrar_imagenes and img_display:
                        if img_prev_display:
                            c1,c2=st.columns(2); c1.image(img_prev_display,"Previo"); c2.image(img_display,"Actual")
                        else: st.image(img_display,"Actual", width=300)
                        if img_thermal: st.image(img_thermal,"Térmica", width=300)

                except Exception as e: st.error(f"Error: {e}")

        if st.session_state.resultado_analisis:
            st.markdown(st.session_state.resultado_analisis.replace("```html","").replace("```",""), unsafe_allow_html=True)
            
            st.markdown("---")
            with st.expander("💬 Chat Asistente", expanded=False):
                for m in st.session_state.chat_messages:
                    with st.chat_message(m["role"]): st.markdown(m["content"])
                
                col_c1, col_c2, col_c3 = st.columns(3)
                cp = None
                if col_c1.button("📝 Alta", key="c1", type="secondary"): cp="Informe Alta"
                if col_c2.button("🩹 Cuidados", key="c2", type="secondary"): cp="Plan Cuidados"
                if col_c3.button("⚠️ Alarma", key="c3", type="secondary"): cp="Signos Alarma"

                if p := st.chat_input("Duda..."):
                    st.session_state.chat_messages.append({"role": "user", "content": p})
                    with st.chat_message("user"): st.markdown(p)
                    with st.chat_message("assistant"):
                        try:
                            chat = genai.GenerativeModel("models/gemini-3-flash-preview")
                            r = chat.generate_content(f"CTX:{st.session_state.resultado_analisis}\nQ:{p}")
                            st.markdown(r.text)
                            st.session_state.chat_messages.append({"role": "assistant", "content": r.text})
                        except: st.error("Error chat")
                
                if cp:
                    st.session_state.chat_messages.append({"role": "user", "content": cp})
                    with st.chat_message("user"): st.markdown(cp)
                    with st.chat_message("assistant"):
                        try:
                            chat = genai.GenerativeModel("models/gemini-3-flash-preview")
                            r = chat.generate_content(f"CTX:{st.session_state.resultado_analisis}\nQ:{cp}")
                            st.markdown(r.text)
                            st.session_state.chat_messages.append({"role": "assistant", "content": r.text})
                        except: st.error("Error chat")

        if st.session_state.pdf_bytes:
            st.download_button("📥 PDF", st.session_state.pdf_bytes, "informe.pdf", "application/pdf")

    with tab_historial:
        if not st.session_state.history_db: st.info("Vacío.")
        else:
            if st.button("🗑️ Todo"): st.session_state.history_db=[]; st.rerun()
            for item in reversed(st.session_state.history_db):
                with st.expander(f"📅 {item['date']} | {item['note']}"): st.markdown(item['result'], unsafe_allow_html=True)

# --- COLUMNA 3 ---
with col_right:
    with st.expander("📈 Gráfica", expanded=False):
        if len(st.session_state.historial_evolucion)>0:
            df = pd.DataFrame(st.session_state.historial_evolucion)
            st.line_chart(df.set_index("Fecha"))
            st.markdown(f'<div class="prediction-box">{predecir_cierre()}</div>', unsafe_allow_html=True)
        else: st.caption("Sin datos.")

st.divider()
if st.button("🔒"):
    cookie_manager.delete("labmind_secret_key")
    st.session_state.autenticado = False; st.rerun()
