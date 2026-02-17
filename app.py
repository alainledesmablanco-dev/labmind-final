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

# --- CONFIGURACIÓN ---
st.set_page_config(page_title="LabMind 23.0 (Clinical Intelligence)", page_icon="🧠", layout="wide")

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
    
    /* SEGURIDAD Y TEJIDOS */
    .safety-alert { background-color: #ffcdd2; border: 2px solid #c62828; color: #b71c1c; padding: 15px; border-radius: 8px; font-weight: bold; margin-bottom: 20px; animation: pulse 2s infinite; }
    @keyframes pulse { 0% { box-shadow: 0 0 0 0 rgba(198, 40, 40, 0.4); } 70% { box-shadow: 0 0 0 10px rgba(198, 40, 40, 0); } 100% { box-shadow: 0 0 0 0 rgba(198, 40, 40, 0); } }

    .tissue-bar-container { display: flex; width: 100%; height: 25px; border-radius: 12px; overflow: hidden; margin-top: 5px; margin-bottom: 5px; }
    .tissue-gran { background-color: #ef5350; height: 100%; display: flex; align-items: center; justify-content: center; color: white; font-size: 0.7em; font-weight: bold; }
    .tissue-slough { background-color: #fdd835; height: 100%; display: flex; align-items: center; justify-content: center; color: #333; font-size: 0.7em; font-weight: bold; }
    .tissue-nec { background-color: #212121; height: 100%; display: flex; align-items: center; justify-content: center; color: white; font-size: 0.7em; font-weight: bold; }

    /* ACADEMIA FLASH */
    .box-edu { background-color: #fff8e1; border: 1px solid #ffecb3; border-radius: 8px; padding: 15px; margin-top: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    .edu-title { color: #f57f17; font-weight: bold; font-size: 1.1em; display: flex; align-items: center; gap: 10px; }

    /* FUENTES DISCRETAS */
    .source-footer { font-size: 0.8em; color: #7f8c8d; margin-top: 15px; border-top: 1px solid #eee; padding-top: 5px; font-style: italic; }

    .alerta-dispositivo { background-color: #fff3cd; padding: 10px; border-radius: 5px; border-left: 5px solid #ffc107; color: #856404; font-weight: bold; margin-bottom: 10px;}
    .privacidad-tag { background-color: #e8eaf6; color: #3f51b5; padding: 2px 6px; border-radius: 4px; font-size: 0.8em; font-weight: bold; }

    /* UPLOADER */
    [data-testid='stFileUploaderDropzone'] div div span { display: none; }
    [data-testid='stFileUploaderDropzone'] div div::after { content: "Arrastra y suelta archivos aquí"; font-size: 1rem; font-weight: bold; color: #444; display: block; }
    [data-testid='stFileUploaderDropzone'] div div::before { content: "Límite: 200MB por archivo"; font-size: 0.8rem; color: #888; display: block; margin-bottom: 5px; }
</style>
""", unsafe_allow_html=True)

# --- GESTOR COOKIES ---
cookie_manager = stx.CookieManager()

# --- ESTADO ---
if "autenticado" not in st.session_state: st.session_state.autenticado = False
if "api_key" not in st.session_state: st.session_state.api_key = ""
if "resultado_analisis" not in st.session_state: st.session_state.resultado_analisis = None
if "datos_grafica" not in st.session_state: st.session_state.datos_grafica = None
if "pdf_bytes" not in st.session_state: st.session_state.pdf_bytes = None
if "log_privacidad" not in st.session_state: st.session_state.log_privacidad = []
if "area_herida" not in st.session_state: st.session_state.area_herida = None
# NUEVO: Estado para el Chat
if "chat_history" not in st.session_state: st.session_state.chat_history = []
if "contexto_chat" not in st.session_state: st.session_state.contexto_chat = ""

# --- LOGIN (COOKIES) ---
time.sleep(0.1)
cookie_api_key = cookie_manager.get(cookie="labmind_secret_key")

if not st.session_state.autenticado:
    if cookie_api_key:
        st.session_state.api_key = cookie_api_key
        st.session_state.autenticado = True
        st.rerun()
    else:
        col1, col2, col3 = st.columns([1,2,1])
        with col2:
            st.markdown("<br><br>", unsafe_allow_html=True)
            st.image("https://cdn-icons-png.flaticon.com/512/3063/3063176.png", width=100)
            st.title("LabMind Acceso")
            st.info("🔐 Tu clave se guardará 30 días.")
            with st.form("login_form"):
                st.text_input("Usuario:", placeholder="Sanitario")
                k_input = st.text_input("API Key:", type="password")
                if st.form_submit_button("🔓 ENTRAR"):
                    if k_input:
                        expires = datetime.datetime.now() + datetime.timedelta(days=30)
                        cookie_manager.set("labmind_secret_key", k_input, expires_at=expires)
                        st.session_state.api_key = k_input
                        st.session_state.autenticado = True
                        time.sleep(1)
                        st.rerun()
        st.stop()

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

def extraer_tejidos(txt):
    # Busca el patrón de tejidos en el texto
    match = re.search(r'TEJIDOS_DATA: (\[.*?\])', txt)
    if match:
        try: return eval(match.group(1)) # Retorna [Gran, Esf, Nec] ej: [60, 30, 10]
        except: return None
    return None

# --- OPENCV ---
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
            m1 = cv2.inRange(hsv, np.array([0, 50, 50]), np.array([10, 255, 255]))
            m2 = cv2.inRange(hsv, np.array([170, 50, 50]), np.array([180, 255, 255]))
            m3 = cv2.inRange(hsv, np.array([10, 50, 50]), np.array([30, 255, 255]))
            mask_herida = m1 + m2 + m3
            mask_herida = cv2.morphologyEx(mask_herida, cv2.MORPH_OPEN, np.ones((5,5),np.uint8))
            cv2.circle(mask_herida, (x_c, y_c), r_c + 10, 0, -1)
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

with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3063/3063176.png", width=60)
    
    if st.button("🔒 Cerrar Sesión"):
        cookie_manager.delete("labmind_secret_key")
        st.session_state.autenticado = False; st.rerun()

    st.divider()
    
    proto_file = st.file_uploader("📚 Protocolo (PDF/Foto)", type=["pdf", "jpg", "png"], help="Sube una guía clínica.")
    proto_content = []; proto_text = ""
    if proto_file:
        st.success("✅ Protocolo Cargado")
        if proto_file.type == "application/pdf":
            try:
                pdf_reader = pypdf.PdfReader(proto_file)
                for page in pdf_reader.pages: proto_text += page.extract_text() or ""
            except: pass
        else:
            proto_content.append(Image.open(proto_file))

# --- MAIN ---
st.title("🩺 LabMind 23.0")
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
        else:
            if fs:=st.file_uploader("Docs/Fotos",accept_multiple_files=True,key="g1"):
                for f in fs: archivos.append(("doc",f))

    st.markdown("---")
    audio = st.audio_input("🎙️ Notas de Voz")
    notas = st.text_area("Texto / Notas Clínicas (Alergias, Fármacos):", height=80)

with col2:
    st.subheader("2. Análisis Clínico")
    
    if (archivos or audio) and st.button("🚀 ANALIZAR (Preview)", type="primary"):
        st.session_state.log_privacidad = []; st.session_state.area_herida = None
        st.session_state.chat_history = [] # Reset chat al analizar nuevo caso
        
        with st.spinner("🧠 Analizando, midiendo tejidos y revisando seguridad..."):
            try:
                genai.configure(api_key=st.session_state.api_key)
                model = genai.GenerativeModel("models/gemini-3-flash-preview")
                
                con = []; txt_c = ""
                if proto_content: con.extend(proto_content); txt_c += "\n[PROTOCOLO VISUAL]\n"
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
                
                # --- PROMPT CLÍNICO AVANZADO ---
                prompt = f"""
                Rol: Enfermera Especialista (APN). Contexto: {contexto}. Modo: {modo}. Notas del Usuario: "{notas}"
                {dato_med}
                { "VERIFICA TUBOS/VÍAS." if activar_detector else "" }
                MATERIAL: {txt_c}
                {f"PROTOCOLO: {proto_text[:10000]}" if proto_text else ""}
                
                INSTRUCCIONES INTELIGENTES:
                1. SEGURIDAD: Revisa las 'Notas del Usuario'. Si menciona Alergias o Fármacos, verifica que tu recomendación NO sea incompatible. Si hay riesgo, genera una alerta 'SAFETY_ALERT'.
                2. TEJIDOS (Solo Heridas): Estima visualmente el porcentaje de: Granulación (Rojo), Esfacelo (Amarillo/Blanco), Necrosis (Negro). Devuelve lista python: TEJIDOS_DATA: [60, 30, 10]
                
                OUTPUT FORMAT (STRICT):
                ---
                SAFETY_ALERT: [Si hay riesgo, escribe aquí el mensaje en mayúsculas. Si no, deja vacío]
                TEJIDOS_DATA: [G, E, N]
                ---
                ### ⚡ RESUMEN
                * **👤 PACIENTE:** [Anonimizado]
                * **🚨 DIAGNÓSTICO:** [Breve]
                * **🩹 ACCIÓN:** [Inmediata]
                * **🔮 PREDICCIÓN:** [Pronóstico]
                * **🧴 MATERIAL:** [Lista]
                ---
                ### 🎓 FORMACIÓN FLASH
                * **Patología:** [Nombre]
                * **Perlas Clínicas:** [Puntos clave]
                * **Tip Experto:** [Consejo]
                ---
                ### 📝 DETALLE
                [Análisis completo]

                ### 🔗 FUENTES
                [Lista de referencias breve]
                """
                
                resp = model.generate_content([prompt, *con] if con else prompt)
                st.session_state.resultado_analisis = resp.text
                st.session_state.datos_grafica = extraer_datos_grafica(resp.text)
                st.session_state.contexto_chat = resp.text # Guardar para el chat
                
                clean_txt = resp.text.replace("GRÁFICA_DATA:", "").split("{'")[0]
                st.session_state.pdf_bytes = create_pdf(clean_txt.replace("*","").replace("#","").replace("---",""))

            except Exception as e: st.error(f"Error: {e}")

    # --- RENDERIZADO DEL INFORME ---
    if st.session_state.resultado_analisis:
        if activar_medicion and con and isinstance(con[-1], Image.Image):
             with st.expander("📸 Imagen Procesada", expanded=True): st.image(con[-1], caption="Análisis Visión", use_container_width=True)
        if st.session_state.log_privacidad:
            with st.expander("ℹ️ Logs Sistema", expanded=False):
                 for log in st.session_state.log_privacidad: st.caption(f"✅ {log}")

        txt = st.session_state.resultado_analisis
        
        # 1. PARSEO DE DATOS INTELIGENTES
        tejidos = extraer_tejidos(txt)
        safety_match = re.search(r'SAFETY_ALERT: (.*)', txt)
        safety_msg = safety_match.group(1).strip() if safety_match else ""

        # 2. ALERTA DE SEGURIDAD (SAFETY NET)
        if safety_msg and len(safety_msg) > 5:
            st.markdown(f'<div class="safety-alert">⚠️ ALERTA DE SEGURIDAD: {safety_msg}</div>', unsafe_allow_html=True)
        elif "⚠️ ALERTA" in txt:
             st.markdown('<div class="alerta-dispositivo">🚨 ALERTA CLÍNICA</div>', unsafe_allow_html=True)

        # 3. BARRA VISUAL DE TEJIDOS (SEGMENTACIÓN)
        if tejidos and "Heridas" in modo:
            g, e, n = tejidos
            st.markdown(f"""
            <div style="margin-bottom:5px; font-weight:bold; color:#555;">📊 Composición Tisular Estimada:</div>
            <div class="tissue-bar-container">
                <div class="tissue-gran" style="width: {g}%;">Granulación {g}%</div>
                <div class="tissue-slough" style="width: {e}%;">Esfacelo {e}%</div>
                <div class="tissue-nec" style="width: {n}%;">Necrosis {n}%</div>
            </div>
            """, unsafe_allow_html=True)

        parts = txt.split("---")
        
        # Lógica de extracción (Resumen, Edu, Detalle, Fuentes)
        resumen_part = re.search(r'### ⚡ RESUMEN(.*?)---', txt, re.DOTALL)
        edu_part = re.search(r'### 🎓 FORMACIÓN FLASH(.*?)---', txt, re.DOTALL)
        detalle_match = re.search(r'### 📝 DETALLE(.*?)### 🔗 FUENTES', txt, re.DOTALL)
        if not detalle_match: detalle_match = re.search(r'### 📝 DETALLE(.*)', txt, re.DOTALL)
        fuentes_part = re.search(r'### 🔗 FUENTES(.*)', txt, re.DOTALL)

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
            st.markdown(f"""<div class="box-edu"><div class="edu-title">🎓 Academia al Vuelo</div><div style="color: #444; margin-top: 10px; font-style: italic;">{edu_raw.replace('*', '•')}</div></div>""", unsafe_allow_html=True)

        if detalle_match:
            st.markdown("### 📝 Detalle")
            st.markdown(detalle_match.group(1).strip())

        if fuentes_part:
            st.markdown(f"""<div class="source-footer">📚 <b>Fuentes y Referencias:</b> {fuentes_part.group(1).strip()}</div>""", unsafe_allow_html=True)
            
        st.divider()
        if st.session_state.pdf_bytes:
            n = f"Informe_{datetime.datetime.now().strftime('%H%M')}.pdf"
            st.download_button("📥 DESCARGAR PDF", st.session_state.pdf_bytes, n, "application/pdf")

    # --- CHAT CON EL CASO (INTERCONSULTA) ---
    if st.session_state.resultado_analisis:
        st.markdown("---")
        st.subheader("💬 Chat con el Caso")
        st.caption("Pregunta dudas sobre este diagnóstico específico (Ej: '¿Puedo usar hidrogel?', '¿Cómo hago la cura si duele?')")
        
        # Mostrar historial
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Input usuario
        if prompt := st.chat_input("Escribe tu duda clínica..."):
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            
            with st.chat_message("assistant"):
                with st.spinner("Consultando..."):
                    try:
                        # Contexto para el chat: El informe previo + la duda
                        full_chat_prompt = f"""
                        CONTEXTO CLÍNICO PREVIO:
                        {st.session_state.contexto_chat}
                        
                        PREGUNTA DEL USUARIO:
                        {prompt}
                        
                        Responde como experto clínico, breve y conciso.
                        """
                        genai.configure(api_key=st.session_state.api_key)
                        model_chat = genai.GenerativeModel("models/gemini-3-flash-preview")
                        response = model_chat.generate_content(full_chat_prompt)
                        st.markdown(response.text)
                        st.session_state.chat_history.append({"role": "assistant", "content": response.text})
                    except Exception as e:
                        st.error(f"Error en el chat: {e}")
