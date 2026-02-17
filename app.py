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
st.set_page_config(page_title="LabMind 17.1", page_icon="🛡️", layout="wide")

# --- ESTILOS CSS ---
st.markdown("""
<style>
    .stButton>button { width: 100%; border-radius: 8px; font-weight: bold; background-color: #0066cc; color: white; }
    
    /* CAJAS DE RESUMEN */
    .resumen-container { font-family: sans-serif; }
    .box-diag { background-color: #ffebee; border-left: 6px solid #ef5350; padding: 12px; margin-bottom: 8px; border-radius: 4px; color: #c62828; }
    .box-action { background-color: #e3f2fd; border-left: 6px solid #2196f3; padding: 12px; margin-bottom: 8px; border-radius: 4px; color: #1565c0; }
    .box-mat { background-color: #e8f5e9; border-left: 6px solid #4caf50; padding: 12px; margin-bottom: 8px; border-radius: 4px; color: #2e7d32; }
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

# --- FUNCIÓN ANONIMIZAR ROSTROS (OPENCV) ---
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
st.title("🩺 LabMind 17.1 🛡️")
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
    
    # --- AVISOS DE PRIVACIDAD SEGÚN MODO ---
    if "Dermatología" in modo:
        st.warning("⚠️ MODO DERMATOLOGÍA: El nublado automático de rostros está DESACTIVADO para permitir el análisis de lesiones faciales. Por favor, encuadre la foto con responsabilidad.")
    else:
        st.info("🛡️ MODO SEGURO: Se nublarán automáticamente los ojos/rostros detectados en las imágenes.")

    activar_detector = False
    if "RX" in modo or "Integral" in modo:
        activar_detector = st.checkbox("🕵️ Revisar Tubos/Vías", value=True)

    fuente = st.radio("Entrada:", ["📁 Archivo/Grabar", "📸 WebCam"], horizontal=True)
    archivos = []
    
    if fuente == "📸 WebCam":
        if f := st.camera_input("Foto"): archivos.append(("cam", f))
    else:
        if "Heridas" in modo:
            if f1:=st.file_uploader("Actual",type=['jpg','png'],key="h1"): archivos.append(("img",f1))
            if f2:=st.file_uploader("Previa",type=['jpg','png'],key="h2"): archivos.append(("img",f2))
        elif "Dermatología" in modo:
            st.info("📸 Sube foto/vídeo de la lesión.")
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
    notas = st.text_area("Texto:", height=80)

with col2:
    st.subheader("2. Análisis Clínico")
    
    if (archivos or audio) and st.button("🚀 ANALIZAR", type="primary"):
        st.session_state.log_privacidad = [] # Limpiar log
        
        with st.spinner("🧠 Procesando..."):
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
                        # --- LÓGICA CONDICIONAL DE PRIVACIDAD ---
                        img_pil = Image.open(a)
                        
                        # SOLO aplicamos anonimización SI NO estamos en Dermato
                        if "Dermatología" not in modo:
                            img_final, fue_procesada = anonymize_face(img_pil)
                            if fue_procesada:
                                st.session_state.log_privacidad.append(f"🛡️ Rostro nublado en: {getattr(a, 'name', 'Cámara')}")
                            txt_c += "\n[IMG (Procesada)]\n"
                        else:
                            # En dermato, pasamos la imagen original
                            img_final = img_pil
                            txt_c += "\n[IMG (Original Dermato)]\n"

                        con.append(img_final)

                res_ins = "CONTEXTO RESIDENCIA: Material in situ. NO pruebas complejas." if "Residencia" in contexto else ""
                
                prompt_esp = ""
                if "Dermato" in modo: prompt_esp = "MODO DERMA: ABCDE, morfología, sugiere tópico/derivación. NO NUBLAR LESIONES."
                elif "Heridas" in modo: prompt_esp = "MODO HERIDAS: TIME, bordes, exudado. Sugiere APÓSITOS."
                elif "ECG" in modo: prompt_esp = "MODO ECG: Ritmo, Frecuencia, Eje, QRS, ST, T."

                prompt = f"""
                Rol: Enfermera Especialista (APN). Contexto: {contexto}. Modo: {modo}. Notas: "{notas}"
                {res_ins} {prompt_esp}
                { "VERIFICA TUBOS/VÍAS: TET, SNG, CVC." if activar_detector else "" }
                MATERIAL: {txt_c}
                
                OUTPUT FORMAT (STRICT):
                ---
                ### ⚡ RESUMEN
                * **👤 PACIENTE:** [Datos Anonimizados]
                * **🚨 DIAGNÓSTICO:** [Texto breve]
                * **🩹 ACCIÓN:** [Texto breve]
                * **🧴 MATERIAL:** [Lista breve]
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
        # MOSTRAR LOG SI HUBO NUBLADO
        if st.session_state.log_privacidad and "Dermatología" not in modo:
            with st.expander("🛡️ Reporte de Privacidad", expanded=True):
                for log in st.session_state.log_privacidad: st.caption(f"✅ {log}")

        txt = st.session_state.resultado_analisis
        
        if "⚠️ ALERTA" in txt or "MAL POSICIONADO" in txt:
            st.markdown('<div class="alerta-dispositivo">🚨 ALERTA: VERIFICAR DISPOSITIVO</div>', unsafe_allow_html=True)
        
        if st.session_state.datos_grafica:
            d = st.session_state.datos_grafica
            f, ax = plt.subplots(figsize=(6,2)); ax.plot(list(d.keys()), list(d.values()), 'o-r'); ax.grid(True, alpha=0.3); st.pyplot(f)
        
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
