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

# --- CONFIGURACIÓN ---
st.set_page_config(page_title="LabMind 20.0 (Gemini 3 Preview)", page_icon="🧬", layout="wide")

# --- ESTILOS CSS ---
st.markdown("""
<style>
    .stButton>button { width: 100%; border-radius: 8px; font-weight: bold; background-color: #0066cc; color: white; }
    
    /* CAJAS DE RESULTADOS */
    .resumen-container { font-family: sans-serif; }
    .box-diag { background-color: #ffebee; border-left: 6px solid #ef5350; padding: 12px; margin-bottom: 8px; border-radius: 4px; color: #c62828; }
    .box-action { background-color: #e3f2fd; border-left: 6px solid #2196f3; padding: 12px; margin-bottom: 8px; border-radius: 4px; color: #1565c0; }
    .box-mat { background-color: #e8f5e9; border-left: 6px solid #4caf50; padding: 12px; margin-bottom: 8px; border-radius: 4px; color: #2e7d32; }
    .box-ai { background-color: #f3e5f5; border-left: 6px solid #9c27b0; padding: 12px; margin-bottom: 8px; border-radius: 4px; color: #6a1b9a; }
    
    /* ACADEMIA FLASH */
    .box-edu { 
        background-color: #fff8e1; 
        border: 1px solid #ffecb3;
        border-radius: 8px; 
        padding: 15px;
        margin-top: 20px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .edu-title { color: #f57f17; font-weight: bold; font-size: 1.1em; display: flex; align-items: center; gap: 10px; }

    .alerta-dispositivo { background-color: #fff3cd; padding: 10px; border-radius: 5px; border-left: 5px solid #ffc107; color: #856404; font-weight: bold; margin-bottom: 10px;}
    .btn-safari { display: block; width: 100%; padding: 10px; background-color: #2ecc71; color: white !important; text-align: center; border-radius: 8px; text-decoration: none; font-weight: bold; margin-top: 10px; border: 1px solid #27ae60; }
    .privacidad-tag { background-color: #e8eaf6; color: #3f51b5; padding: 2px 6px; border-radius: 4px; font-size: 0.8em; font-weight: bold; }

    /* UPLOADER */
    [data-testid='stFileUploaderDropzone'] div div span { display: none; }
    [data-testid='stFileUploaderDropzone'] div div::after { content: "Arrastra y suelta archivos aquí"; font-size: 1rem; font-weight: bold; color: #444; display: block; }
    [data-testid='stFileUploaderDropzone'] div div::before { content: "Límite: 200MB por archivo"; font-size: 0.8rem; color: #888; display: block; margin-bottom: 5px; }
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
#      FUNCIONES AUXILIARES
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

# --- OPENCV FUNCIONES (PRIVACIDAD + MEDICIÓN) ---
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
    return (Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)), True) if processed else (pil_image, False)

def medir_herida(pil_image):
    try:
        img = np.array(pil_image.convert('RGB'))
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        gray_blurred = cv2.GaussianBlur(gray, (9, 9), 2)
        circles = cv2.HoughCircles(gray_blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=50, param1=100, param2=30, minRadius=20, maxRadius=200)
        area_real_cm2 = None; mensaje = "No se detectó moneda."
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            (x_c, y_c, r_c) = circles[0]
            cv2.circle(img, (x_c, y_c), r_c, (0, 0, 255), 4)
            cv2.putText(img, "Ref 1 Euro", (x_c - 20, y_c), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            pixels_per_cm = (r_c * 2) / 2.325; scale_factor = (1 / pixels_per_cm) ** 2
            hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
            # Rangos rojo/amarillo
            m1 = cv2.inRange(hsv, np.array([0, 50, 50]), np.array([10, 255, 255]))
            m2 = cv2.inRange(hsv, np.array([170, 50, 50]), np.array([180, 255, 255]))
            m3 = cv2.inRange(hsv, np.array([10, 50, 50]), np.array([30, 255, 255]))
            mask_herida = m1 + m2 + m3
            mask_herida = cv2.morphologyEx(mask_herida, cv2.MORPH_OPEN, np.ones((5,5),np.uint8))
            cv2.circle(mask_herida, (x_c, y_c), r_c + 10, 0, -1) # Borrar moneda de mascara
            cnts, _ = cv2.findContours(mask_herida, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            area_px = sum(cv2.contourArea(c) for c in cnts if cv2.contourArea(c) > 500)
            if area_px > 0:
                cv2.drawContours(img, cnts, -1, (0, 255, 0), 2)
                area_real_cm2 = area_px * scale_factor
                mensaje = f"Área: {area_real_cm2:.2f} cm²"
                cv2.putText(img, mensaje, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        return Image.fromarray(img), area_real_cm2, mensaje
    except Exception as e: return pil_image, None, f"Error: {e}"

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
    if st.file_uploader("📚 Protocolo (PDF)", type="pdf"): st.success("✅ Protocolo")

# --- MAIN ---
st.title("🩺 LabMind 20.0")
col1, col2 = st.columns([1.2, 2])

with col1:
    c1, c2 = st.columns([1, 1.5])
    with c1: st.subheader("1. Captura")
    with c2: contexto = st.selectbox("🏥 Contexto:", ["Hospitalización", "Residencia (Geriatría)", "Urgencias", "UCI", "Domicilio"])
    
    modo = st.radio("Modo:", ["🩹 Heridas", "🧴 Dermatología", "📊 Analíticas", "📈 ECG", "💊 Farmacia", "💀 RX / TAC / RMN", "🧩 Integral"])
    st.markdown("---")
    
    activar_detector = False; activar_medicion = False
    if "RX" in modo or "Integral" in modo: activar_detector = st.checkbox("🕵️ Revisar Tubos/Vías", value=True)
    if "Heridas" in modo: activar_medicion = st.checkbox("📏 Medición Automática (Moneda 1€)", value=False)
    if "Dermatología" in modo: st.warning("⚠️ MODO DERMA: Nublado desactivado.")

    fuente = st.radio("Entrada:", ["📁 Archivo/Grabar", "📸 WebCam"], horizontal=True)
    archivos = []
    
    if fuente == "📸 WebCam":
        if f := st.camera_input("Foto"): archivos.append(("cam", f))
    else:
        if "Heridas" in modo:
            if f1:=st.file_uploader("Actual",type=['jpg','png'],key="h1"): archivos.append(("img",f1))
            if f2:=st.file_uploader("Previa",type=['jpg','png'],key="h2"): archivos.append(("img",f2))
        elif "Dermatología" in modo:
            if f:=st.file_uploader("Lesión",type=['jpg','png','mp4','mov'],key="d1"): archivos.append(("video",f) if "video" in f.type else ("img",f))
        else: # Generico
            if fs:=st.file_uploader("Docs/Fotos",accept_multiple_files=True,key="g1"):
                for f in fs: archivos.append(("doc",f))

    st.markdown("---")
    audio = st.audio_input("🎙️ Notas de Voz")
    notas = st.text_area("Texto:", height=80)

with col2:
    st.subheader("2. Análisis Clínico")
    
    if (archivos or audio) and st.button("🚀 ANALIZAR", type="primary"):
        st.session_state.log_privacidad = []; st.session_state.area_herida = None
        
        with st.spinner("🧠 Analizando..."):
            try:
                genai.configure(api_key=st.session_state.api_key)
                
                # --- AQUÍ ESTÁ EL CAMBIO SOLICITADO ---
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
                        if activar_medicion and "Heridas" in modo:
                            img_med, area, msg = medir_herida(img_pil)
                            if area: 
                                st.session_state.area_herida = f"{area:.2f} cm²"
                                st.session_state.log_privacidad.append(f"📏 {msg}")
                                img_pil = img_med
                        
                        if "Dermatología" not in modo:
                            img_final, proc = anonymize_face(img_pil)
                            if proc: st.session_state.log_privacidad.append("🛡️ Rostro nublado")
                        else: img_final = img_pil
                        
                        con.append(img_final); txt_c += "\n[IMG]\n"

                dato_med = f"ÁREA HERIDA: {st.session_state.area_herida}" if st.session_state.area_herida else ""
                
                # PROMPT 
                prompt = f"""
                Rol: Enfermera Especialista (APN). Contexto: {contexto}. Modo: {modo}. Notas: "{notas}"
                {dato_med}
                { "VERIFICA TUBOS/VÍAS." if activar_detector else "" }
                MATERIAL: {txt_c}
                
                OUTPUT FORMAT (STRICT):
                ---
                ### ⚡ RESUMEN
                * **👤 PACIENTE:** [Anonimizado]
                * **🚨 DIAGNÓSTICO:** [Breve]
                * **🩹 ACCIÓN:** [Inmediata]
                * **🔮 PREDICCIÓN:** [Pronóstico]
                * **🧴 MATERIAL:** [Lista]
                ---
                ### 🎓 FORMACIÓN FLASH
                * **Patología:** [Nombre Técnico]
                * **Perlas Clínicas:** [3 puntos clave para aprender sobre esto]
                * **Tip Experto:** [Un consejo avanzado]
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
        if activar_medicion and con and isinstance(con[0], Image.Image):
             with st.expander("📸 Imagen Procesada", expanded=True): st.image(con[0], caption="Análisis Visión", use_container_width=True)
        if st.session_state.log_privacidad:
            with st.expander("ℹ️ Logs Sistema", expanded=False):
                 for log in st.session_state.log_privacidad: st.caption(f"✅ {log}")

        txt = st.session_state.resultado_analisis
        if "⚠️ ALERTA" in txt: st.markdown('<div class="alerta-dispositivo">🚨 ALERTA CLÍNICA</div>', unsafe_allow_html=True)
        
        parts = txt.split("---")
        
        resumen_html = ""; educacion_html = ""; detalle_txt = ""

        resumen_part = re.search(r'### ⚡ RESUMEN(.*?)---', txt, re.DOTALL)
        edu_part = re.search(r'### 🎓 FORMACIÓN FLASH(.*?)---', txt, re.DOTALL)
        detalle_part = txt.split("---")[-1]

        if resumen_part:
            resumen_raw = resumen_part.group(1).strip()
            html_resumen = '<div class="resumen-container">'
            for line in resumen_raw.split('\n'):
                line = line.replace('*', '').strip()
                if not line: continue
                if "👤 PACIENTE" in line: html_resumen += f'<span class="box-patient">👤 {line.replace("👤 PACIENTE:", "").strip()} <span class="privacidad-tag">Anonimizado</span></span>'
                elif "🚨 DIAGNÓSTICO" in line: html_resumen += f'<div class="box-diag"><b>🚨 DIAGNÓSTICO:</b><br>{line.replace("🚨 DIAGNÓSTICO:", "").strip()}</div>'
                elif "🩹 ACCIÓN" in line: html_resumen += f'<div class="box-action"><b>🩹 ACCIÓN:</b><br>{line.replace("🩹 ACCIÓN:", "").strip()}</div>'
                elif "🔮 PREDICCIÓN" in line: html_resumen += f'<div class="box-ai"><b>🔮 PREDICCIÓN IA:</b><br>{line.replace("🔮 PREDICCIÓN:", "").strip()}</div>'
                elif "🧴 MATERIAL" in line: html_resumen += f'<div class="box-mat"><b>🧴 MATERIAL:</b><br>{line.replace("🧴 MATERIAL:", "").strip()}</div>'
            html_resumen += '</div>'
            st.markdown(html_resumen, unsafe_allow_html=True)
        
        if edu_part:
            edu_raw = edu_part.group(1).strip()
            st.markdown(f"""
            <div class="box-edu">
                <div class="edu-title">🎓 Academia al Vuelo</div>
                <div style="color: #444; margin-top: 10px; font-style: italic;">{edu_raw.replace('*', '•')}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown(detalle_part)
            
        st.divider()
        if st.session_state.pdf_bytes:
            n = f"Informe_{datetime.datetime.now().strftime('%H%M')}.pdf"
            st.download_button("📥 DESCARGAR PDF", st.session_state.pdf_bytes, n, "application/pdf")
