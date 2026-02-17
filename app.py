import streamlit as st
import google.generativeai as genai
from PIL import Image
import pypdf
import tempfile
import time
import os
from fpdf import FPDF
import datetime

# --- CONFIGURACIÓN ---
st.set_page_config(page_title="LabMind 10.0", page_icon="🏥", layout="wide")

# --- GESTIÓN DE ESTADO (MEMORIA) ---
if "resultado_analisis" not in st.session_state:
    st.session_state.resultado_analisis = None

# --- ESTILOS ---
st.markdown("""
<style>
    .stButton>button { width: 100%; border-radius: 8px; font-weight: bold; background-color: #0066cc; color: white; }
    .esquema-rapido { background-color: #e8f4ff; padding: 15px; border-radius: 10px; border-left: 5px solid #0066cc; margin-bottom: 20px; }
</style>
""", unsafe_allow_html=True)

# --- FUNCIONES AUXILIARES (PDF) ---
def create_pdf(texto_analisis):
    class PDF(FPDF):
        def header(self):
            self.set_font('Arial', 'B', 12)
            self.cell(0, 10, 'LabMind - Informe Clínico IA', 0, 1, 'C')
            self.ln(5)
        def footer(self):
            self.set_y(-15)
            self.set_font('Arial', 'I', 8)
            self.cell(0, 10, f'Pagina {self.page_no()}', 0, 0, 'C')

    pdf = PDF()
    pdf.add_page()
    pdf.set_font("Arial", size=10)
    
    # Encabezado con fecha
    fecha = datetime.datetime.now().strftime("%d/%m/%Y %H:%M")
    pdf.cell(0, 10, f"Fecha del Informe: {fecha}", 0, 1)
    pdf.ln(5)
    
    # Limpieza básica de texto para evitar errores de codificación en PDF
    # FPDF no soporta emojis, así que los quitamos o reemplazamos simplificando
    texto_limpio = texto_analisis.encode('latin-1', 'replace').decode('latin-1')
    
    pdf.multi_cell(0, 5, texto_limpio)
    return pdf.output(dest='S').encode('latin-1')

# --- BARRA LATERAL ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3063/3063176.png", width=80)
    st.title("LabMind 10.0")
    
    st.markdown("### 🔑 Acceso")
    api_key = st.text_input("Pega tu API Key aquí:", type="password")
    
    st.divider()
    st.caption("v10.0 - Voz + PDF + Vídeo Integral")
    
    protocolo_pdf = st.file_uploader("📚 Protocolo (PDF)", type="pdf")
    texto_protocolo = ""
    if protocolo_pdf:
        try:
            pdf_reader = pypdf.PdfReader(protocolo_pdf)
            for page in pdf_reader.pages: texto_protocolo += page.extract_text() or ""
            st.success("✅ Protocolo Activo")
        except: st.error("Error PDF")

    contexto = st.selectbox("Contexto:", ["Hospitalización", "Urgencias", "UCI", "Domicilio", "Consulta"])

# --- ZONA PRINCIPAL ---
st.title("🩺 Estación Clínica Multimodal")

col1, col2 = st.columns([1.2, 2])

with col1:
    st.subheader("1. Captura de Datos")
    
    modo = st.radio("Modo:", [
        "🩹 Heridas", 
        "📊 Analíticas/Informes", 
        "📉 ECG", 
        "💀 TAC/RMN (Solo Imagen/Vídeo)", 
        "🧩 ESTUDIO INTEGRAL (Todo junto)"
    ])
    st.markdown("---")
    
    opciones_fuente = ["📁 Subir o Grabar (Móvil)", "📸 Cámara Web (Solo Fotos)"]
    if modo == "💀 TAC/RMN (Solo Imagen/Vídeo)" or modo == "🧩 ESTUDIO INTEGRAL (Todo junto)":
        st.info("💡 Soporte de VÍDEO activo.")
    
    fuente_imagen = st.radio("Método de entrada:", opciones_fuente, horizontal=True)
    
    archivos_procesar = [] 

    # --- INPUTS DE IMAGEN/VIDEO ---
    if fuente_imagen == "📸 Cámara Web (Solo Fotos)":
        foto_camara = st.camera_input("Hacer foto")
        if foto_camara: archivos_procesar.append(("foto_camara", foto_camara))
    else:
        if modo == "🩹 Heridas":
            st.info("📸 Foto Actual + Previa")
            f_actual = st.file_uploader("FOTO ACTUAL", type=['jpg', 'png', 'jpeg'])
            f_previa = st.file_uploader("FOTO PREVIA", type=['jpg', 'png', 'jpeg'])
            if f_actual: archivos_procesar.append(("img_actual", f_actual))
            if f_previa: archivos_procesar.append(("img_previa", f_previa))

        elif modo == "📊 Analíticas/Informes":
            st.info("📂 Documentos")
            files = st.file_uploader("Archivos:", type=['pdf', 'jpg', 'png', 'jpeg'], accept_multiple_files=True)
            if files:
                for f in files: archivos_procesar.append(("doc", f))
        
        elif modo == "💀 TAC/RMN (Solo Imagen/Vídeo)":
            f = st.file_uploader("Sube Imagen o VÍDEO:", type=['jpg', 'png', 'jpeg', 'mp4', 'mov', 'avi'])
            if f: 
                if f.type in ['video/mp4', 'video/quicktime', 'video/x-msvideo']: archivos_procesar.append(("video", f))
                else: archivos_procesar.append(("unico", f))

        elif modo == "🧩 ESTUDIO INTEGRAL (Todo junto)":
            st.info("🗂️ Sube TODO mezclado: Informes, Fotos y VÍDEOS.")
            files = st.file_uploader("Archivos del caso:", type=['pdf', 'jpg', 'png', 'jpeg', 'mp4', 'mov', 'avi'], accept_multiple_files=True)
            if files:
                for f in files:
                    if f.type in ['video/mp4', 'video/quicktime', 'video/x-msvideo']: archivos_procesar.append(("video", f))
                    else: archivos_procesar.append(("doc_mix", f))

        else: # ECG
            f = st.file_uploader("Imagen ECG:", type=['jpg', 'png', 'jpeg'])
            if f: archivos_procesar.append(("unico", f))

    st.markdown("---")
    
    # --- NUEVO: AUDIO INPUT (NOTA DE VOZ) ---
    st.write("🎙️ **Nota de Voz (Opcional):**")
    audio_nota = st.audio_input("Grabar explicación clínica")
    if audio_nota:
        st.success("✅ Audio grabado. Se enviará a la IA.")
        
    notas_texto = st.text_area("✍️ O escribir notas:", placeholder="Ej: Paciente politraumatizado...", height=100)

with col2:
    st.subheader("2. Resultados del Análisis")
    
    # Botón de análisis
    if (archivos_procesar or audio_nota) and st.button("🚀 ANALIZAR AHORA", type="primary"):
        if not api_key:
            st.warning("⚠️ Falta API Key.")
        else:
            with st.spinner("🧠 Gemini 3 Flash analizando vídeo, audio y documentos..."):
                try:
                    genai.configure(api_key=api_key)
                    model = genai.GenerativeModel("models/gemini-3-flash-preview")
                    safety_settings = [{"category": c, "threshold": "BLOCK_NONE"} for c in ["HARM_CATEGORY_HARASSMENT", "HARM_CATEGORY_HATE_SPEECH", "HARM_CATEGORY_SEXUALLY_EXPLICIT", "HARM_CATEGORY_DANGEROUS_CONTENT"]]
                    
                    contenido_ia = []
                    contexto_archivos = ""
                    
                    # 1. PROCESAR AUDIO (SI HAY)
                    if audio_nota:
                         contenido_ia.append(genai.upload_file(audio_nota, mime_type="audio/wav"))
                         contexto_archivos += "\n[NOTA DE VOZ DEL ENFERMERO ADJUNTA]\n"

                    # 2. PROCESAR ARCHIVOS
                    for tipo, archivo in archivos_procesar:
                        if tipo == "video":
                            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
                                tmp_file.write(archivo.read())
                                tmp_path = tmp_file.name
                            video_file = genai.upload_file(path=tmp_path)
                            while video_file.state.name == "PROCESSING":
                                time.sleep(1)
                                video_file = genai.get_file(video_file.name)
                            contenido_ia.append(video_file)
                            contexto_archivos += f"\n[VÍDEO ADJUNTO: {archivo.name}]\n"
                            os.remove(tmp_path)
                        
                        elif hasattr(archivo, 'type') and archivo.type == "application/pdf":
                            pdf_reader = pypdf.PdfReader(archivo)
                            texto_pdf = ""
                            for page in pdf_reader.pages: texto_pdf += page.extract_text() or ""
                            contexto_archivos += f"\n--- PDF ---\n{texto_pdf}\n"
                        
                        else:
                            img = Image.open(archivo)
                            contenido_ia.append(img)
                            contexto_archivos += f"\n[IMAGEN ADJUNTA]\n"

                    # 3. PROMPT
                    full_prompt = f"""
                    Actúa como Enfermera Clínica Especialista (APN) y Experta en Radiología.
                    CONTEXTO: {contexto}. MODO: {modo}.
                    NOTAS TEXTO: "{notas_texto}"
                    
                    ⚠️ PRIVACIDAD: Anonimiza nombres.

                    MATERIAL ADJUNTO (Puede incluir Audio de voz, Vídeos, PDFs):
                    {contexto_archivos}
                    {f"PROTOCOLO: {texto_protocolo[:20000]}" if texto_protocolo else "USA EVIDENCIA CIENTÍFICA."}

                    INSTRUCCIONES:
                    - Si hay AUDIO: Escúchalo atentamente e integra la información verbal en el análisis.
                    - Si hay VÍDEO: Analiza la secuencia.
                    
                    FORMATO SALIDA (2 PARTES con "---"):
                    ---
                    ### ⚡ RESUMEN CLÍNICO
                    * **👤 PACIENTE:** [Anonimizado].
                    * **🚨 DIAGNÓSTICO:** [Síntesis].
                    * **🩹 ACCIÓN:** [Inmediata].
                    ---
                    ### 📝 ANÁLISIS DETALLADO
                    1. Hallazgos (Visuales/Auditivos/Documentales).
                    2. Plan de Cuidados.
                    """
                    
                    # LLAMADA
                    if contenido_ia:
                        response = model.generate_content([full_prompt, *contenido_ia], safety_settings=safety_settings)
                    else:
                        response = model.generate_content(full_prompt, safety_settings=safety_settings)
                    
                    # GUARDAR EN SESIÓN (MEMORIA)
                    st.session_state.resultado_analisis = response.text
                    
                except Exception as e:
                    st.error("❌ Error:")
                    st.write(e)

    # --- MOSTRAR RESULTADO (PERSISTENTE) ---
    if st.session_state.resultado_analisis:
        texto_final = st.session_state.resultado_analisis
        
        # Renderizar en pantalla
        partes = texto_final.split("---")
        if len(partes) >= 3:
            st.markdown(f'<div class="esquema-rapido">{partes[1]}</div>', unsafe_allow_html=True)
            st.markdown(partes[2])
        else:
            st.markdown(texto_final)
            
        st.divider()
        
        # --- NUEVO: BOTÓN EXPORTAR PDF ---
        # Limpiamos el texto de marcadores markdown para que el PDF salga limpio
        texto_para_pdf = texto_final.replace("*", "").replace("#", "").replace("---", "")
        
        pdf_bytes = create_pdf(texto_para_pdf)
        
        st.download_button(
            label="📥 DESCARGAR INFORME PDF",
            data=pdf_bytes,
            file_name=f"Informe_LabMind_{datetime.datetime.now().strftime('%Y%m%d')}.pdf",
            mime="application/pdf"
        )
