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

# --- CONFIGURACIÓN ---
st.set_page_config(page_title="LabMind 15.4", page_icon="🧬", layout="wide")

# --- ESTILOS CSS ---
st.markdown("""
<style>
    .stButton>button { width: 100%; border-radius: 8px; font-weight: bold; background-color: #0066cc; color: white; }
    .esquema-rapido { background-color: #e8f4ff; padding: 15px; border-radius: 10px; border-left: 5px solid #0066cc; margin-bottom: 20px; }
    .alerta-dispositivo { background-color: #ffe6e6; padding: 10px; border-radius: 5px; border-left: 5px solid #ff4444; color: #cc0000; font-weight: bold; margin-bottom: 10px;}
    .login-box { max-width: 400px; margin: 0 auto; padding: 40px; border-radius: 10px; background-color: #f8f9fa; border: 1px solid #ddd; text-align: center; }

    /* --- TRADUCCIÓN DEL UPLOADER --- */
    [data-testid='stFileUploaderDropzone'] div div span { display: none; }
    [data-testid='stFileUploaderDropzone'] div div::after { content: "Arrastra y suelta archivos aquí"; font-size: 1rem; font-weight: bold; color: #444; display: block; }
    [data-testid='stFileUploaderDropzone'] div div small { display: none; }
    [data-testid='stFileUploaderDropzone'] div div::before { content: "Límite: 200MB por archivo"; font-size: 0.8rem; color: #888; display: block; margin-bottom: 5px; }
    [data-testid='stFileUploaderDropzone'] button { border-color: #0066cc; }
</style>
""", unsafe_allow_html=True)

# --- GESTIÓN DE SESIÓN ---
if "autenticado" not in st.session_state:
    st.session_state.autenticado = False
if "api_key" not in st.session_state:
    st.session_state.api_key = ""
if "resultado_analisis" not in st.session_state:
    st.session_state.resultado_analisis = None
if "datos_grafica" not in st.session_state:
    st.session_state.datos_grafica = None
if "pdf_bytes" not in st.session_state:
    st.session_state.pdf_bytes = None
if "mostrar_enlace_magico" not in st.session_state:
    st.session_state.mostrar_enlace_magico = False

# --- 1. COMPROBAR ENLACE MÁGICO (URL) ---
try:
    query_params = st.query_params
    if "k" in query_params and not st.session_state.autenticado:
        clave_url = query_params["k"]
        if len(clave_url) > 10: 
            st.session_state.api_key = clave_url
            st.session_state.autenticado = True
            st.success("⚡ Auto-Login correcto.")
            time.sleep(0.5)
            st.rerun()
except: pass

# --- PANTALLA DE LOGIN ---
def mostrar_login():
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.image("https://cdn-icons-png.flaticon.com/512/3063/3063176.png", width=100)
        st.title("LabMind Acceso")
        st.info("🔐 Entra una vez con tu clave.")
        
        with st.form("login_form"):
            usuario = st.text_input("Usuario:", placeholder="Sanitario")
            clave_input = st.text_input("API Key:", type="password")
            submit_button = st.form_submit_button("🔓 ENTRAR")
            
            if submit_button:
                if clave_input:
                    st.session_state.api_key = clave_input
                    st.session_state.autenticado = True
                    st.rerun()
                else:
                    st.warning("⚠️ Introduce la API Key.")

if not st.session_state.autenticado:
    mostrar_login()
    st.stop()

# ==========================================
#      A PARTIR DE AQUÍ: APP PRINCIPAL
# ==========================================

# --- FUNCIONES AUXILIARES ---
def create_pdf(texto_analisis):
    class PDF(FPDF):
        def header(self):
            self.set_font('Arial', 'B', 12)
            self.cell(0, 10, 'LabMind - Informe IA', 0, 1, 'C'); self.ln(5)
        def footer(self):
            self.set_y(-15); self.set_font('Arial', 'I', 8); self.cell(0, 10, f'Pagina {self.page_no()}', 0, 0, 'C')
    pdf = PDF(); pdf.add_page(); pdf.set_font("Arial", size=10)
    fecha = datetime.datetime.now().strftime("%d/%m/%Y %H:%M")
    pdf.cell(0, 10, f"Fecha del Informe: {fecha}", 0, 1); pdf.ln(5)
    texto_limpio = texto_analisis.replace('€', 'EUR').replace('’', "'").replace('“', '"').replace('”', '"')
    texto_encoded = texto_limpio.encode('latin-1', 'replace').decode('latin-1')
    pdf.multi_cell(0, 5, texto_encoded)
    return pdf.output(dest='S').encode('latin-1')

def extraer_datos_grafica(texto):
    match = re.search(r'GRÁFICA_DATA: ({.*?})', texto)
    if match:
        try: return eval(match.group(1))
        except: return None
    return None

# --- BARRA LATERAL (CON SOLUCIÓN AUTO-LOGIN) ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3063/3063176.png", width=60)
    
    # --- BOTÓN DE ENLACE MÁGICO ---
    if st.button("🔗 Generar Auto-Login"):
        st.session_state.mostrar_enlace_magico = True
    
    if st.session_state.mostrar_enlace_magico:
        st.success("👇 Pulsa el enlace azul para recargar con la clave:")
        # Enlace relativo que funciona en cualquier dominio
        st.markdown(f"### [➡️ PULSA AQUÍ PARA ACTIVAR](/?k={st.session_state.api_key})")
        st.caption("Cuando la página se recargue, añádela a Favoritos o Pantalla de Inicio y ya no te pedirá clave.")
    
    st.divider()
    if st.button("🔒 Cerrar Sesión"):
        st.session_state.autenticado = False
        st.query_params.clear()
        st.rerun()

    st.divider()
    protocolo_pdf = st.file_uploader("📚 Protocolo (PDF)", type="pdf")
    texto_protocolo = ""
    if protocolo_pdf:
        try:
            pdf_reader = pypdf.PdfReader(protocolo_pdf)
            for page in pdf_reader.pages: texto_protocolo += page.extract_text() or ""
            st.success("✅ Protocolo Activo")
        except: pass

# --- ZONA PRINCIPAL ---
st.title("🩺 LabMind 15.4")

col1, col2 = st.columns([1.2, 2])

with col1:
    cabecera_col1, cabecera_col2 = st.columns([1, 1.5])
    with cabecera_col1:
        st.subheader("1. Captura")
    with cabecera_col2:
        contexto = st.selectbox("🏥 Contexto:", ["Hospitalización", "Residencia (Geriatría)", "Urgencias", "UCI", "Domicilio"])
    
    modo = st.radio("Modo:", ["🩹 Heridas", "📊 Analíticas", "📈 ECG", "💊 Farmacia", "💀 RX / TAC / RMN (Patología + Disp)", "🧩 Integral"])
    st.markdown("---")
    
    activar_detector = False
    if modo == "💀 RX / TAC / RMN (Patología + Disp)" or modo == "🧩 Integral":
        activar_detector = st.checkbox("🕵️ Revisar Tubos/Vías (Seguridad)", value=True, help="Verifica posición de SNG, TET, etc.")

    fuente = st.radio("Entrada:", ["📁 Archivo/Grabar", "📸 WebCam"], horizontal=True)
    archivos = []
    
    if fuente == "📸 WebCam":
        foto = st.camera_input("Foto")
        if foto: archivos.append(("cam", foto))
    else:
        if modo == "🩹 Heridas":
            f1 = st.file_uploader("Subir Foto Actual", type=['jpg','png'], key="u1")
            f2 = st.file_uploader("Subir Foto Previa", type=['jpg','png'], key="u2")
            if f1: archivos.append(("img", f1)); 
            if f2: archivos.append(("img", f2))
        elif modo == "📈 ECG": 
            f = st.file_uploader("Subir Electro (Foto/PDF)", type=['jpg','png','pdf'], key="u3")
            if f: archivos.append(("img", f))
        elif modo == "💀 RX / TAC / RMN (Patología + Disp)":
            f = st.file_uploader("Subir Imagen o VÍDEO", type=['jpg','png','mp4','mov','avi'], key="u4")
            if f:
                if f.type in ['video/mp4','video/quicktime','video/x-msvideo']: archivos.append(("video", f))
                else: archivos.append(("img", f))
        else:
            fs = st.file_uploader("Subir Documentos/Fotos", accept_multiple_files=True, key="u5")
            if fs: 
                for f in fs: archivos.append(("doc", f))

    st.markdown("---")
    audio = st.audio_input("🎙️ Notas de Voz")
    notas = st.text_area("Texto:", height=80)

with col2:
    st.subheader("2. Análisis Clínico")
    
    if (archivos or audio) and st.button("🚀 ANALIZAR", type="primary"):
        with st.spinner("🧠 Procesando..."):
            try:
                genai.configure(api_key=st.session_state.api_key)
                model = genai.GenerativeModel("models/gemini-3-flash-preview")
                contenido_ia = []; txt_contexto = ""
                
                if audio:
                     contenido_ia.append(genai.upload_file(audio, mime_type="audio/wav")); txt_contexto += "\n[AUDIO ADJUNTO]\n"
                for t, a in archivos:
                    if t == "video":
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tf:
                            tf.write(a.read()); tpath = tf.name
                        vf = genai.upload_file(path=tpath)
                        while vf.state.name == "PROCESSING": time.sleep(1); vf = genai.get_file(vf.name)
                        contenido_ia.append(vf); txt_contexto += "\n[VIDEO TAC/RMN]\n"; os.remove(tpath)
                    elif hasattr(a, 'type') and a.type == "application/pdf":
                        pdf_reader = pypdf.PdfReader(a); pdf_text = ""
                        for p in pdf_reader.pages: pdf_text += p.extract_text()
                        txt_contexto += f"\nPDF: {pdf_text}\n"
                    else:
                        img = Image.open(a); contenido_ia.append(img); txt_contexto += "\n[IMAGEN]\n"

                instruccion_residencia = ""
                if "Residencia" in contexto:
                    instruccion_residencia = "CONTEXTO RESIDENCIA (GERIATRÍA): Céntrate en cuidados in situ. Dispones de material. NO pidas pruebas hospitalarias complejas salvo vital."
                prompt_detector = ""; prompt_especifico = ""
                if modo == "📈 ECG": prompt_especifico = "ANÁLISIS ECG: Ritmo, Frecuencia, Eje, QRS, ST, T."
                elif activar_detector: prompt_detector = "VERIFICA TUBOS/VÍAS: TET, SNG, CVC."

                full_prompt = f"""
                Actúa como Experto Clínico. Contexto: {contexto}. Modo: {modo}. Notas: "{notas}"
                {instruccion_residencia} {prompt_especifico} {prompt_detector}
                MATERIAL: {txt_contexto} {f"PROTOCOLO: {texto_protocolo[:15000]}" if texto_protocolo else ""}
                INSTRUCCIONES: 1. DIAGNÓSTICO. 2. IMAGEN: Hallazgos. 3. ANONIMIZA.
                SALIDA (Usa "---"):
                ---
                ### ⚡ RESUMEN
                * **👤 PACIENTE:** [Datos]
                * **🚨 DIAGNÓSTICO:** [Principal]
                * **🩹 ACCIÓN:** [Inmediata]
                * **🧴 MATERIAL:** [LISTA ESQUEMÁTICA]
                ---
                ### 📝 ANÁLISIS DETALLADO
                [Desarrollo]
                """
                
                if contenido_ia: resp = model.generate_content([full_prompt, *contenido_ia])
                else: resp = model.generate_content(full_prompt)
                
                st.session_state.resultado_analisis = resp.text
                st.session_state.datos_grafica = extraer_datos_grafica(resp.text)
                
                texto_limpio = resp.text.replace("GRÁFICA_DATA:", "").split("{'")[0]
                st.session_state.pdf_bytes = create_pdf(texto_limpio.replace("*","").replace("#","").replace("---",""))

            except Exception as e: st.error(f"Error: {e}")

    if st.session_state.resultado_analisis:
        texto = st.session_state.resultado_analisis
        if "⚠️ ALERTA" in texto or "MAL POSICIONADO" in texto:
            st.markdown('<div class="alerta-dispositivo">🚨 ALERTA: VERIFICAR POSICIÓN DE DISPOSITIVO MÉDICO</div>', unsafe_allow_html=True)
        if st.session_state.datos_grafica:
            data = st.session_
