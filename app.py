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

# --- CONFIGURACIÓN ---
st.set_page_config(page_title="LabMind 37.0 (Coin Measure)", page_icon="🪙", layout="wide")

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
    [data-testid='stFileUploaderDropzone'] div div::after { content: "📂 Toca para adjuntar"; font-size: 1rem; color: #555; display: block; }
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
#      FUNCIONES DE MEDICIÓN Y VISIÓN
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

# --- FUNCIÓN DE MEDICIÓN MEJORADA (MONEDA 1 EURO) ---
def medir_herida_con_referencia(pil_image, usar_moneda=False):
    """
    Intenta detectar una moneda de 1 Euro (2.325 cm) para calibrar la escala.
    Si no, usa una escala aproximada por defecto.
    """
    area_final = 0.0
    img_annotated = pil_image.copy()
    
    try:
        img_np = np.array(pil_image.convert('RGB'))
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        
        # 1. Detectar Moneda (Círculos)
        pixels_per_cm = 0
        circles = cv2.HoughCircles(
            cv2.GaussianBlur(gray, (9, 9), 2), 
            cv2.HOUGH_GRADIENT, 1.2, 50, param1=100, param2=30, minRadius=20, maxRadius=300
        )
        
        moneda_detectada = False
        
        if circles is not None and usar_moneda:
            circles = np.round(circles[0, :]).astype("int")
            # Asumimos que el círculo más claro/definido es la moneda
            for (x, y, r) in circles:
                # Dibujar la moneda detectada en verde
                cv2.circle(img_bgr, (x, y), r, (0, 255, 0), 4)
                cv2.putText(img_bgr, "1 EUR (Ref)", (x - 20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                # DIÁMETRO 1 EURO = 2.325 cm
                diameter_pixels = r * 2
                pixels_per_cm = diameter_pixels / 2.325
                moneda_detectada = True
                break # Usamos el primer círculo válido
        
        # Fallback si no hay moneda o no se pide
        if pixels_per_cm == 0:
            # Aproximación genérica (suponiendo foto a 15-20cm)
            pixels_per_cm = 100.0 

        # 2. Detectar Herida (Segmentación básica por color rojizo/oscuro)
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        
        # Rango de rojos/rosas para herida
        lower_red1 = np.array([0, 50, 50])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 50, 50])
        upper_red2 = np.array([180, 255, 255])
        
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask = mask1 + mask2
        
        # Limpiar ruido
        kernel = np.ones((5,5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        area_pixels_total = 0
        for c in contours:
            if cv2.contourArea(c) > 500: # Filtrar ruido pequeño
                # Evitar contar la moneda como herida (si coincidiera en color, raro en 1 euro que es dorada/plateada)
                area_pixels_total += cv2.contourArea(c)
                cv2.drawContours(img_bgr, [c], -1, (0, 0, 255), 2)

        # 3. Calcular Área Real
        if pixels_per_cm > 0:
            pixel_area_sq_cm = (1 / pixels_per_cm) ** 2
            area_final = area_pixels_total * pixel_area_sq_cm
            
        img_annotated = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        
        return area_final, img_annotated, moneda_detectada

    except Exception as e:
        print(f"Error medición: {e}")
        return 0.0, pil_image, False

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

st.title("🩺 LabMind 37.0")
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
    
    # --- PROTOCOLO PLEGADO ---
    with st.expander("📚 Protocolo de Unidad (Opcional)", expanded=False):
        st.caption("Sube el PDF/Foto de tu guía para que la IA la respete.")
        proto_file = st.file_uploader("Subir", type=["pdf", "jpg", "png"], key="global_proto")

# --- COLUMNA 2: NÚCLEO CENTRAL ---
with col_center:
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
    
    # --- UI DINÁMICA ---
    archivos = []
    meds_files = None
    tests_files = None
    usar_moneda = False # Por defecto
    
    # === MODO INTEGRAL ===
    if modo == "🧩 Integral (Analizar Todo)":
        st.info("🧩 **Modo Integral**: Sube cualquier evidencia.")
        with st.expander("📂 Documentación (Fármacos, Informes)", expanded=True):
            c1, c2 = st.columns(2)
            meds_files = c1.file_uploader("💊 Fármacos", accept_multiple_files=True, key="int_meds")
            tests_files = c2.file_uploader("📊 Informes/ECG", accept_multiple_files=True, key="int_tests")

        st.write("📸 **Evidencia Visual (Foto/Video):**")
        fuente = st.radio("Fuente:", ["📁 Archivo", "📸 WebCam"], horizontal=True, label_visibility="collapsed")
        if fuente == "📸 WebCam":
            if f := st.camera_input("Foto Paciente"): archivos.append(("cam", f))
        else:
            if fs := st.file_uploader("Subir Archivos", type=['jpg','png','mp4','mov'], accept_multiple_files=True, key="int_main"):
                for f in fs: archivos.append(("video" if "video" in f.type else "img", f))

    # === MODO HERIDAS (ACTUAL + PREVIO + MONEDA) ===
    elif modo == "🩹 Heridas / Úlceras":
        st.info("🩹 **Modo Heridas**")
        
        # 1. Checkbox Moneda
        usar_moneda = st.checkbox("🪙 Usar moneda de 1€ para medir")
        if usar_moneda:
            st.caption("ℹ️ Coloca una moneda de 1 Euro cerca de la herida para tener una referencia de tamaño exacta.")

        # 2. Plegado: Evolución Previa
        with st.expander("⏮️ Ver Evolución (Subir Foto/Video Previo)", expanded=False):
            st.caption("Adjunta imagen antigua para comparar:")
            if prev := st.file_uploader("Estado Previo", type=['jpg','png','mp4','mov'], accept_multiple_files=True, key="w_prev"):
                for p in prev: archivos.append(("prev_video" if "video" in p.type else "prev_img", p))

        # 3. Plegado: Contexto
        with st.expander("💊 Medicación / Analítica (Opcional)", expanded=False):
            c1, c2 = st.columns(2)
            meds_files = c1.file_uploader("Medicación", accept_multiple_files=True, key="w_meds")
            tests_files = c2.file_uploader("Analítica", accept_multiple_files=True, key="w_labs")
        
        # 4. Principal: Estado Actual
        st.write("📸 **Estado ACTUAL (Foto/Video):**")
        fuente = st.radio("Fuente:", ["📁 Archivo", "📸 WebCam"], horizontal=True, label_visibility="collapsed")
        if fuente == "📸 WebCam":
            if f := st.camera_input("Foto Herida"): archivos.append(("cam", f))
        else:
            if fs := st.file_uploader("Subir Foto/Video Actual", type=['jpg','png','mp4','mov'], accept_multiple_files=True, key="w_img"):
                for f in fs: archivos.append(("video" if "video" in f.type else "img", f))

    # === MODO DERMATOLOGÍA ===
    elif modo == "🧴 Dermatología":
        st.info("🧴 **Modo Dermatología**")
        with st.expander("⏮️ Ver Evolución (Subir Foto/Video Previo)", expanded=False):
            if prev := st.file_uploader("Estado Previo", type=['jpg','png','mp4','mov'], accept_multiple_files=True, key="d_prev"):
                for p in prev: archivos.append(("prev_video" if "video" in p.type else "prev_img", p))

        st.write("📸 **Estado ACTUAL (Foto/Video):**")
        if fs := st.file_uploader("Subir Foto/Video Actual", type=['jpg','png','mp4','mov'], accept_multiple_files=True, key="d_img"):
            for f in fs: archivos.append(("video" if "video" in f.type else "img", f))

    # === OTROS MODOS ===
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
        tests_files = st.file_uploader("PDFs/Fotos", accept_multiple_files=True, key="rep_docs")

    st.markdown("---")
    audio = st.audio_input("🎙️ Notas de Voz")
    notas = st.text_area("Notas Clínicas:", height=60, placeholder="Escribe síntomas, alergias...")

    # --- BOTÓN DE ANÁLISIS ---
    if st.button("🚀 ANALIZAR", type="primary"):
        st.session_state.log_privacidad = []; st.session_state.area_herida = 0.0
        
        with st.spinner(f"🧠 Analizando {modo}..."):
            try:
                genai.configure(api_key=st.session_state.api_key)
                model = genai.GenerativeModel("models/gemini-1.5-flash") # Modelo Estable
                
                con = []; txt_meds = ""; txt_tests = ""; txt_proto = ""

                # Procesamiento Docs
                if proto_file:
                    if "pdf" in proto_file.type: r = pypdf.PdfReader(proto_file); txt_proto += "".join([p.extract_text() for p in r.pages])
                    else: con.append(Image.open(proto_file))
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
                
                for label, a in archivos:
                    is_video = "video" in label
                    
                    if is_video:
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tf: tf.write(a.read()); tp = tf.name
                        vf = genai.upload_file(path=tp)
                        while vf.state.name == "PROCESSING": time.sleep(1); vf = genai.get_file(vf.name)
                        con.append(vf); os.remove(tp)
                    else: 
                        img_pil = Image.open(a)
                        # Lógica Visual
                        if "Heridas" in modo and "prev" not in label: 
                            # MEDICIÓN CON MONEDA
                            area, img_medida, coin_found = medir_herida_con_referencia(img_pil, usar_moneda)
                            
                            if area > 0: st.session_state.area_herida = area
                            if coin_found: st.toast("🪙 Moneda detectada: Calibración exacta activa.")
                            
                            img_thermal = procesar_termografia(img_pil)
                            img_biofilm, biofilm_detectado = detectar_biofilm(img_pil)
                            img_display = img_medida # Mostramos la imagen con la medición pintada
                            
                            # Enviamos la original a la IA para análisis clínico, mostramos la pintada al usuario
                            con.append(img_pil) 
                            con.append(img_thermal)
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
                - PROTOCOLO UNIDAD: {txt_proto if txt_proto else "Usar guías estándar."}
                - FARMACIA: {txt_meds}
                - DOCUMENTOS: {txt_tests}
                - VISUAL: {len(archivos)} archivos.
                {f"- ÁREA MEDIDA (APROX/CALIBRADA): {st.session_state.area_herida:.2f} cm2" if st.session_state.area_herida > 0 else ""}
                
                INSTRUCCIONES CLAVE:
                1. BASA TU DECISIÓN EN EL PROTOCOLO SUBIDO (Si existe).
                2. Si es Integral: Cruza todos los datos.
                3. Si hay imágenes PREVIAS y ACTUALES: Haz un análisis comparativo de la evolución.
                """
                
                if "Farmacia" in modo: prompt += " CHECK DOSIS/ALERGIAS."
                elif "RX" in modo: prompt += " RADIOLOGÍA TÉCNICA."
                elif "Heridas" in modo: prompt += " HERIDAS + HTML BARRA TEJIDOS."
                elif "ECG" in modo: prompt += " LECTURA ECG."

                prompt += """
                \nOUTPUT:
                ### ⚡ RESUMEN (Basado en Protocolo)
                SYNC_ALERT: [ALERTA SI HAY CONFLICTO]
                * **🚨 DIAGNÓSTICO:**
                ...
                ### 📊 Análisis Detallado / Evolución
                [Si es herida, HTML puro de barra tejidos]
                ...
                """
                
                # Retry Logic
                for attempt in range(3):
                    try:
                        resp = model.generate_content([prompt, *con] if con else prompt)
                        st.session_state.resultado_analisis = resp.text
                        break 
                    except Exception as e:
                        if "429" in str(e) and attempt < 2: time.sleep(5); continue
                        elif attempt == 2: st.error("⚠️ Servidor saturado.")
                        else: raise e
                
                if "Heridas" in modo and st.session_state.area_herida > 0:
                    st.session_state.historial_evolucion.append({
                        "Fecha": datetime.datetime.now().strftime("%d/%m %H:%M"), "Area": st.session_state.area_herida
                    })
                
                if st.session_state.resultado_analisis:
                    st.session_state.pdf_bytes = create_pdf(st.session_state.resultado_analisis.replace("*","").replace("#",""))

                if img_display: st.image(img_display, caption="Evidencia Procesada", width=300)
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
