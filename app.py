import streamlit as st
import google.generativeai as genai
from PIL import Image, ImageDraw, ImageFont
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

# --- CONFIGURACIÓN ---
st.set_page_config(page_title="LabMind 18.0 (Inteligencia Clínica)", page_icon="🧬", layout="wide")

# --- ESTILOS CSS ---
st.markdown("""
<style>
    .stButton>button { width: 100%; border-radius: 8px; font-weight: bold; background-color: #0066cc; color: white; }
    
    /* CAJAS DE RESUMEN */
    .resumen-container { font-family: sans-serif; }
    .box-diag { background-color: #ffebee; border-left: 6px solid #ef5350; padding: 12px; margin-bottom: 8px; border-radius: 4px; color: #c62828; }
    .box-action { background-color: #e3f2fd; border-left: 6px solid #2196f3; padding: 12px; margin-bottom: 8px; border-radius: 4px; color: #1565c0; }
    .box-mat { background-color: #e8f5e9; border-left: 6px solid #4caf50; padding: 12px; margin-bottom: 8px; border-radius: 4px; color: #2e7d32; }
    .box-ai { background-color: #f3e5f5; border-left: 6px solid #9c27b0; padding: 12px; margin-bottom: 8px; border-radius: 4px; color: #6a1b9a; }
    .box-patient { font-weight: bold; color: #555; margin-bottom: 10px; display: block; }

    .alerta-dispositivo { background-color: #fff3cd; padding: 10px; border-radius: 5px; border-left: 5px solid #ffc107; color: #856404; font-weight: bold; margin-bottom: 10px;}
    .btn-safari { display: block; width: 100%; padding: 10px; background-color: #2ecc71; color: white !important; text-align: center; border-radius: 8px; text-decoration: none; font-weight: bold; margin-top: 10px; border: 1px solid #27ae60; }
    .btn-safari:hover { background-color: #27ae60; }
    .privacidad-tag { background-color: #e8eaf6; color: #3f51b5; padding: 2px 6px; border-radius: 4px; font-size: 0.8em; font-weight: bold; }

    /* UPLOADER */
    [data-testid='stFileUploaderDropzone'] div div span { display: none; }
    [data-testid='stFileUploaderDropzone'] div div::after { content: "Arrastra y suelta archivos aquí"; font-size: 1rem; font-weight: bold; color: #444; display: block; }
    [data-testid='stFileUploaderDropzone'] div div small { display: none; }
    [data-testid='stFileUploaderDropzone'] div div::before { content: "Límite: 200MB por archivo"; font-size: 0.8rem; color: #888; display: block; margin-bottom: 5px; }
    [data-testid='stFileUploaderDropzone'] button { border-color: #0066cc; }
</style>
""", unsafe_allow_html=True)

# --- GESTIÓN DE SESIÓN ---
if "autenticado" not in st.session_state: st.session_state.autenticado = False
if "api_key" not in st.session_state: st.session_state.api_key = ""
if "resultado_analisis" not in st.session_state: st.session_state.resultado_analisis = None
if "datos_grafica" not in st.session_state: st.session_state.datos_grafica = None
if "pdf_bytes" not in st.session_state: st.session_state.pdf_bytes = None
if "mostrar_enlace_magico" not in st.session_state: st.session_state.mostrar_enlace_magico = False
if "log_privacidad" not in st.session_state: st.session_state.log_privacidad = []
if "area_herida" not in st.session_state: st.session_state.area_herida = None

# --- AUTO-LOGIN ---
try:
    if "k" in st.query_params and not st.session_state.autenticado:
        clave_url = st.query_params["k"]
        if len(clave_url) > 10: 
            st.session_state.api_key = clave_url; st.session_state.autenticado = True; st.rerun()
except: pass

# --- LOGIN ---
def mostrar_login():
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.image("https://cdn-icons-png.flaticon.com/512/3063/3063176.png", width=100)
        st.title("LabMind Acceso")
        with st.form("login_form"):
            st.text_input("Usuario:", placeholder="Sanitario")
            k = st.text_input("API Key:", type="password")
            if st.form_submit_button("🔓 ENTRAR"):
                if k: st.session_state.api_key = k; st.session_state.autenticado = True; st.rerun()

if not st.session_state.autenticado: mostrar_login(); st.stop()

# ==========================================
#      FUNCIONES AUXILIARES (OPEN CV & PDF)
# ==========================================

def create_pdf(texto_analisis):
    class PDF(FPDF):
        def header(self): self.set_font('Arial','B',12); self.cell(0,10,'LabMind - Informe IA',0,1,'C'); self.ln(5)
        def footer(self): self.set_y(-15); self.set_font('Arial','I',8); self.cell(0,10,f'Pag {self.page_no()}',0,0,'C')
    pdf = PDF(); pdf.add_page(); pdf.set_font("Arial", size=10)
    pdf.cell(0,10,f"Fecha: {datetime.datetime.now().strftime('%d/%m/%Y %H:%M')}",0,1); pdf.ln(5)
    clean = texto_analisis.replace('€','EUR').replace('’',"'").replace('“','"').replace('”','"')
    pdf.multi_cell(0,5, clean.encode('latin-1','replace').decode('latin-1'))
    return pdf.output(dest='S').encode('latin-1')

def extraer_datos_grafica(txt):
    match = re.search(r'GRÁFICA_DATA: ({.*?})', txt)
    return eval(match.group(1)) if match else None

# --- FUNCIÓN 1: ANONIMIZAR ROSTROS (PRIVACIDAD) ---
def anonymize_face(pil_image):
    img_np = np.array(pil_image.convert('RGB'))
    img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    processed = False
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]; roi_color = img_cv[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            eye_roi = roi_color[ey:ey+eh, ex:ex+ew]
            blurred_eye = cv2.GaussianBlur(eye_roi, (99, 99), 30)
            roi_color[ey:ey+eh, ex:ex+ew] = blurred_eye
            processed = True
    if processed:
        img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
        return Image.fromarray(img_rgb), True
    else: return pil_image, False

# --- FUNCIÓN 2: MEDICIÓN DE HERIDAS (Pilar 1) ---
def medir_herida(pil_image):
    """
    Intenta detectar una moneda de 1 euro (23.25mm) y calcular el área de la herida (tejido rojo/amarillo).
    """
    try:
        # Convertir a OpenCV
        img = np.array(pil_image.convert('RGB'))
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        
        # 1. Detectar Moneda (Hough Circles)
        # Suavizamos para evitar ruido
        gray_blurred = cv2.GaussianBlur(gray, (9, 9), 2)
        circles = cv2.HoughCircles(gray_blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=50, param1=100, param2=30, minRadius=20, maxRadius=200)

        area_real_cm2 = None
        mensaje = "No se detectó moneda de referencia."
        
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            # Asumimos que el círculo más definido es la moneda
            (x_c, y_c, r_c) = circles[0]
            
            # Dibujar moneda (Azul)
            cv2.circle(img, (x_c, y_c), r_c, (0, 0, 255), 4)
            cv2.putText(img, "Ref 1 Euro", (x_c - 20, y_c), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            # CALCULO DE ESCALA
            # Diámetro 1 Euro = 23.25 mm = 2.325 cm
            # Radio real = 1.1625 cm
            # Area real moneda = pi * r^2 = 3.1416 * (1.1625)^2 = 4.24 cm2
            
            # Píxeles por cm
            pixels_per_cm = (r_c * 2) / 2.325
            scale_factor = (1 / pixels_per_cm) ** 2 # cm2 per pixel

            # 2. Detectar Herida (Color Thresholding)
            # Convertimos a HSV para detectar rojo/rosa (carne) y amarillo (fibrina)
            hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
            
            # Máscara para rojos (Tejido granulación) - Rango 1
            lower_red1 = np.array([0, 50, 50]); upper_red1 = np.array([10, 255, 255])
            mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
            # Rango 2
            lower_red2 = np.array([170, 50, 50]); upper_red2 = np.array([180, 255, 255])
            mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
            # Máscara para amarillos/negros (Fibrina/Necrosis)
            lower_yel = np.array([10, 50, 50]); upper_yel = np.array([30, 255, 255])
            mask3 = cv2.inRange(hsv, lower_yel, upper_yel)
            
            # Máscara combinada
            mask_herida = mask1 + mask2 + mask3
            
            # Limpiamos ruido
            kernel = np.ones((5,5),np.uint8)
            mask_herida = cv2.morphologyEx(mask_herida, cv2.MORPH_OPEN, kernel)
            
            # IMPORTANTE: Borrar la zona de la moneda de la máscara de la herida
            # Para que no cuente la moneda como herida si es dorada
            cv2.circle(mask_herida, (x_c, y_c), r_c + 10, 0, -1)

            # Encontrar contornos de la herida
            cnts, _ = cv2.findContours(mask_herida, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            area_pixels_herida = 0
            for c in cnts:
                if cv2.contourArea(c) > 500: # Filtrar ruido pequeño
                    cv2.drawContours(img, [c], -1, (0, 255, 0), 2) # Verde
                    area_pixels_herida += cv2.contourArea(c)
            
            if area_pixels_herida > 0:
                area_real_cm2 = area_pixels_herida * scale_factor
                mensaje = f"Área Calculada: {area_real_cm2:.2f} cm²"
                cv2.putText(img, f"Area: {area_real_cm2:.1f} cm2", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                mensaje = "Moneda detectada, pero no tejido herida claro."

        return Image.fromarray(img), area_real_cm2, mensaje
    except Exception as e:
        return pil_image, None, f"Error medición: {e}"


# ==========================================
#      APP PRINCIPAL
# ==========================================

# --- SIDEBAR ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3063/3063176.png", width=60)
    if st.button("🔗 Generar Auto-Login"): st.session_state.mostrar_enlace_magico = True
    if st.session_state.mostrar_enlace_magico:
        st.markdown(f'''<a href="/?k={st.session_state.api_key}" target="_blank" class="btn-safari">🌍 ABRIR EN SAFARI</a>''', unsafe_allow_html=True)
    st.divider()
    if st.button("🔒 Salir"): st.session_state.autenticado = False; st.query_params.clear(); st.rerun()
    st.divider()
    protocolo_pdf = st.file_uploader("📚 Protocolo (PDF)", type="pdf")
    if protocolo_pdf: st.success("✅ Protocolo")

# --- MAIN ---
st.title("🩺 LabMind 18.0")
col1, col2 = st.columns([1.2, 2])

with col1:
    c1, c2 = st.columns([1, 1.5])
    with c1: st.subheader("1. Captura")
    with c2: contexto = st.selectbox("🏥 Contexto:", ["Hospitalización", "Residencia (Geriatría)", "Urgencias", "UCI", "Domicilio"])
    
    modo = st.radio("Modo:", [
        "🩹 Heridas (Curas)", 
        "🧴 Dermatología (Piel)", 
        "📊 Analíticas", 
        "📈 ECG", 
        "💊 Farmacia", 
        "💀 RX / TAC / RMN", 
        "🧩 Integral"
    ])
    st.markdown("---")
    
    # OPCIONES EXTRA SEGÚN MODO
    activar_detector = False
    activar_medicion = False
    
    if "RX" in modo or "Integral" in modo:
        activar_detector = st.checkbox("🕵️ Revisar Tubos/Vías", value=True)
    
    if "Heridas" in modo:
        activar_medicion = st.checkbox("📏 Medición Automática (Pon una moneda de 1€)", value=False, help="Coloca una moneda de 1 Euro cerca de la herida para calcular su tamaño real.")

    if "Dermatología" in modo:
        st.warning("⚠️ MODO DERMATOLOGÍA: Nublado de rostros DESACTIVADO.")

    fuente = st.radio("Entrada:", ["📁 Archivo/Grabar", "📸 WebCam"], horizontal=True)
    archivos = []
    
    # Lógica de Inputs
    if fuente == "📸 WebCam":
        if f := st.camera_input("Foto"): archivos.append(("cam", f))
    else:
        if "Heridas" in modo:
            if f1:=st.file_uploader("Herida Actual (Con Moneda 1€ si mides)",type=['jpg','png'],key="h1"): archivos.append(("img",f1))
            if f2:=st.file_uploader("Herida Previa",type=['jpg','png'],key="h2"): archivos.append(("img",f2))
        elif "Dermatología" in modo:
            if f:=st.file_uploader("Lesión Piel",type=['jpg','png','mp4','mov'],key="d1"):
                archivos.append(("video",f) if "video" in f.type else ("img",f))
        elif "ECG" in modo:
            if f:=st.file_uploader("ECG",type=['jpg','png','pdf'],key="e1"): archivos.append(("img",f))
        elif "RX" in modo:
            if f:=st.file_uploader("RX/TAC/Video",type=['jpg','mp4','mov'],key="r1"):
                archivos.append(("video",f) if "video" in f.type else ("img",f))
        else:
            if fs:=st.file_uploader("Docs",accept_multiple_files=True,key="g1"):
                for f in fs: archivos.append(("doc",f))

    st.markdown("---")
    audio = st.audio_input("🎙️ Notas de Voz")
    notas = st.text_area("Texto (Opcional):", placeholder="Ej: Paciente diabético, toma Sintrom...", height=80)

with col2:
    st.subheader("2. Análisis Clínico")
    
    if (archivos or audio) and st.button("🚀 ANALIZAR", type="primary"):
        st.session_state.log_privacidad = [] 
        st.session_state.area_herida = None
        
        with st.spinner("🧠 Procesando imágenes y datos..."):
            try:
                genai.configure(api_key=st.session_state.api_key)
                model = genai.GenerativeModel("models/gemini-3-flash-preview")
                con = []; txt_c = ""
                
                if audio: con.append(genai.upload_file(audio, mime_type="audio/wav")); txt_c += "\n[AUDIO]\n"
                
                for t, a in archivos:
                    if t == "video":
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tf: tf.write(a.read()); tp = tf.name
                        vf = genai.upload_file(path=tp)
                        while vf.state.name == "PROCESSING": time.sleep(1); vf = genai.get_file(vf.name)
                        con.append(vf); txt_c += "\n[VIDEO]\n"; os.remove(tp)
                    elif hasattr(a,'type') and a.type == "application/pdf":
                        r = pypdf.PdfReader(a); txt_c += "\nPDF: " + "".join([p.extract_text() for p in r.pages])
                    else:
                        img_pil = Image.open(a)
                        
                        # A) PROCESO DE MEDICIÓN (SI ACTIVO Y ES HERIDA)
                        if activar_medicion and "Heridas" in modo:
                            img_medida, area, msg_medida = medir_herida(img_pil)
                            if area:
                                st.session_state.area_herida = f"{area:.2f} cm²"
                                st.session_state.log_privacidad.append(f"📏 Medición Exitosa: {msg_medida}")
                                img_pil = img_medida # Usamos la imagen con dibujos para el análisis
                            else:
                                st.session_state.log_privacidad.append(f"⚠️ Fallo Medición: {msg_medida}")
                        
                        # B) PROCESO DE PRIVACIDAD
                        if "Dermatología" not in modo:
                            img_final, fue_procesada = anonymize_face(img_pil)
                            if fue_procesada: st.session_state.log_privacidad.append(f"🛡️ Rostro nublado por seguridad.")
                        else:
                            img_final = img_pil
                        
                        con.append(img_final); txt_c += "\n[IMG]\n"

                # CONTEXTO MEDICIÓN EN EL PROMPT
                dato_medicion = ""
                if st.session_state.area_herida:
                    dato_medicion = f"DATOS OBJETIVOS: La herida ha sido medida por visión artificial y tiene un ÁREA DE {st.session_state.area_herida}. Úsalo para el pronóstico."

                # PROMPTS AVANZADOS
                prompt_esp = ""
                if "Dermato" in modo: 
                    prompt_esp = "MODO DERMA: Regla ABCDE. Morfología. NO NUBLAR."
                elif "Heridas" in modo: 
                    prompt_esp = f"""MODO HERIDAS AVANZADO:
                    1. Analiza lecho (TIME).
                    2. PREDICCIÓN: Estima probabilidad de cicatrización en 4 semanas basado en el tejido (necrótico/granulado) y comorbilidades.
                    3. INTERACCIÓN: Si detectas fármacos (corticoides, anticoagulantes), alerta sobre su efecto en la herida.
                    {dato_medicion}"""
                elif "ECG" in modo: 
                    prompt_esp = "MODO ECG: Ritmo, Frecuencia, Eje, QRS, ST, T."

                prompt = f"""
                Rol: Enfermera Especialista (APN). Contexto: {contexto}. Modo: {modo}. Notas: "{notas}"
                {prompt_esp}
                { "VERIFICA TUBOS/VÍAS: TET, SNG, CVC." if activar_detector else "" }
                MATERIAL: {txt_c}
                {f"PROTOCOLO: {texto_protocolo[:10000]}" if texto_protocolo else ""}
                
                OUTPUT FORMAT (STRICT):
                ---
                ### ⚡ RESUMEN
                * **👤 PACIENTE:** [Anonimizado]
                * **🚨 DIAGNÓSTICO:** [Breve]
                * **🩹 ACCIÓN:** [Inmediata]
                * **🔮 PREDICCIÓN:** [Tiempo estimado cierre / Pronóstico]
                * **🧴 MATERIAL:** [Lista]
                ---
                ### 📝 DETALLE
                [Resto del análisis]
                """
                
                resp = model.generate_content([prompt, *con] if con else prompt)
                st.session_state.resultado_analisis = resp.text
                st.session_state.datos_grafica = extraer_datos_grafica(resp.text)
                
                clean_txt = resp.text.replace("GRÁFICA_DATA:", "").split("{'")[0]
                st.session_state.pdf_bytes = create_pdf(clean_txt.replace("*","").replace("#","").replace("---",""))

            except Exception as e: st.error(f"Error: {e}")

    # RENDERIZADO
    if st.session_state.resultado_analisis:
        # VISOR DE IMAGEN PROCESADA (Con moneda y contornos)
        if activar_medicion and con and isinstance(con[0], Image.Image):
             with st.expander("📸 Ver Imagen Procesada (Medición)", expanded=True):
                 st.image(con[0], caption="Análisis Visión Artificial", use_container_width=True)

        # LOGS
        if st.session_state.log_privacidad:
            with st.expander("ℹ️ Información del Sistema", expanded=False):
                for log in st.session_state.log_privacidad: st.caption(f"✅ {log}")

        txt = st.session_state.resultado_analisis
        if "⚠️ ALERTA" in txt: st.markdown('<div class="alerta-dispositivo">🚨 ALERTA CLÍNICA / SEGURIDAD</div>', unsafe_allow_html=True)
        
        parts = txt.split("---")
        if len(parts) >= 3:
            resumen_raw = parts[1]; detalle = parts[2]
            html_resumen = '<div class="resumen-container">'
            for line in resumen_raw.strip().split('\n'):
                line = line.replace('*', '').strip()
                if not line: continue
                if "👤 PACIENTE" in line: html_resumen += f'<span class="box-patient">👤 {line.replace("👤 PACIENTE:", "").strip()} <span class="privacidad-tag">Anonimizado</span></span>'
                elif "🚨 DIAGNÓSTICO" in line: html_resumen += f'<div class="box-diag"><b>🚨 DIAGNÓSTICO:</b><br>{line.replace("🚨 DIAGNÓSTICO:", "").strip()}</div>'
                elif "🩹 ACCIÓN" in line: html_resumen += f'<div class="box-action"><b>🩹 ACCIÓN:</b><br>{line.replace("🩹 ACCIÓN:", "").strip()}</div>'
                elif "🔮 PREDICCIÓN" in line: html_resumen += f'<div class="box-ai"><b>🔮 PREDICCIÓN IA:</b><br>{line.replace("🔮 PREDICCIÓN:", "").strip()}</div>'
                elif "🧴 MATERIAL" in line: html_resumen += f'<div class="box-mat"><b>🧴 MATERIAL:</b><br>{line.replace("🧴 MATERIAL:", "").strip()}</div>'
            html_resumen += '</div>'
            st.markdown(html_resumen, unsafe_allow_html=True)
            st.markdown(detalle)
        else:
            st.markdown(txt)
            
        st.divider()
        if st.session_state.pdf_bytes:
            n = f"Informe_{datetime.datetime.now().strftime('%H%M')}.pdf"
            st.download_button("📥 DESCARGAR PDF", st.session_state.pdf_bytes, n, "application/pdf")
