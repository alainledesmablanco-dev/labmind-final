import streamlit as st
import google.generativeai as genai
from PIL import Image
import pypdf
import tempfile
import time
import os

# --- CONFIGURACIÓN ---
st.set_page_config(page_title="LabMind Integral", page_icon="🏥", layout="wide")

# --- ESTILOS ---
st.markdown("""
<style>
    .stButton>button { width: 100%; border-radius: 8px; font-weight: bold; background-color: #0066cc; color: white; }
    .esquema-rapido { background-color: #e8f4ff; padding: 15px; border-radius: 10px; border-left: 5px solid #0066cc; margin-bottom: 20px; }
</style>
""", unsafe_allow_html=True)

# --- BARRA LATERAL ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3063/3063176.png", width=80)
    st.title("LabMind 9.2")
    
    st.markdown("### 🔑 Acceso")
    api_key = st.text_input("Pega tu API Key aquí:", type="password")
    
    st.divider()
    st.caption("v9.2 - Integral con Vídeo")
    
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
    
    # --- SELECTOR DE FUENTE ---
    opciones_fuente = ["📁 Subir o Grabar (Móvil)", "📸 Cámara Web (Solo Fotos)"]
    if modo == "💀 TAC/RMN (Solo Imagen/Vídeo)" or modo == "🧩 ESTUDIO INTEGRAL (Todo junto)":
        st.info("💡 Soporte de VÍDEO activo para TAC/RMN.")
    
    fuente_imagen = st.radio("Método de entrada:", opciones_fuente, horizontal=True)
    
    archivos_procesar = [] 

    # CASO 1: CÁMARA WEB
    if fuente_imagen == "📸 Cámara Web (Solo Fotos)":
        foto_camara = st.camera_input("Hacer foto")
        if foto_camara:
            archivos_procesar.append(("foto_camara", foto_camara))

    # CASO 2: SUBIR O GRABAR (MÓVIL)
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
                if f.type in ['video/mp4', 'video/quicktime', 'video/x-msvideo']:
                    archivos_procesar.append(("video", f))
                else:
                    archivos_procesar.append(("unico", f))

        # --- AQUÍ ESTÁ EL CAMBIO IMPORTANTE: INTEGRAL CON VÍDEO ---
        elif modo == "🧩 ESTUDIO INTEGRAL (Todo junto)":
            st.info("🗂️ Sube TODO mezclado: Informes PDF, Fotos y VÍDEOS de TAC/RMN.")
            # Ahora aceptamos mp4, mov, avi también aquí
            files = st.file_uploader("Archivos del caso:", type=['pdf', 'jpg', 'png', 'jpeg', 'mp4', 'mov', 'avi'], accept_multiple_files=True)
            if files:
                for f in files:
                    # Clasificamos si es vídeo o documento/imagen
                    if f.type in ['video/mp4', 'video/quicktime', 'video/x-msvideo']:
                        archivos_procesar.append(("video", f))
                    else:
                        archivos_procesar.append(("doc_mix", f))

        else: # ECG
            f = st.file_uploader("Imagen ECG:", type=['jpg', 'png', 'jpeg'])
            if f: archivos_procesar.append(("unico", f))

    st.markdown("---")
    notas = st.text_area("✍️ Notas clínicas:", placeholder="Ej: Paciente politraumatizado...", height=100)

with col2:
    st.subheader("2. Resultados del Análisis")
    
    if archivos_procesar and st.button("🚀 ANALIZAR AHORA", type="primary"):
        if not api_key:
            st.warning("⚠️ Falta API Key (Arriba a la izquierda).")
        else:
            with st.spinner("🧠 Procesando caso complejo (Gemini 3 Flash)..."):
                try:
                    genai.configure(api_key=api_key)
                    model = genai.GenerativeModel("models/gemini-3-flash-preview")
                    
                    # SEGURIDAD OFF
                    safety_settings = [{"category": c, "threshold": "BLOCK_NONE"} for c in ["HARM_CATEGORY_HARASSMENT", "HARM_CATEGORY_HATE_SPEECH", "HARM_CATEGORY_SEXUALLY_EXPLICIT", "HARM_CATEGORY_DANGEROUS_CONTENT"]]
                    
                    contenido_ia = []
                    contexto_archivos = ""
                    
                    for tipo, archivo in archivos_procesar:
                        
                        # 1. SI ES VÍDEO (TAC/RMN) - Lógica de subida a nube
                        if tipo == "video":
                            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
                                tmp_file.write(archivo.read())
                                tmp_path = tmp_file.name
                            
                            st.info(f"Subiendo vídeo {archivo.name} a la IA...")
                            video_file = genai.upload_file(path=tmp_path)
                            
                            while video_file.state.name == "PROCESSING":
                                time.sleep(1)
                                video_file = genai.get_file(video_file.name)
                                
                            if video_file.state.name == "FAILED":
                                st.error(f"Error procesando vídeo {archivo.name}")
                            else:
                                contenido_ia.append(video_file)
                                contexto_archivos += f"\n[VÍDEO ADJUNTO: {archivo.name}]\n"
                            
                            os.remove(tmp_path)
                        
                        # 2. SI ES PDF
                        elif hasattr(archivo, 'type') and archivo.type == "application/pdf":
                            pdf_reader = pypdf.PdfReader(archivo)
                            texto_pdf = ""
                            for page in pdf_reader.pages: texto_pdf += page.extract_text() or ""
                            contexto_archivos += f"\n--- DOCUMENTO PDF ({archivo.name}) ---\n{texto_pdf}\n"
                        
                        # 3. SI ES IMAGEN
                        else:
                            img = Image.open(archivo)
                            contenido_ia.append(img)
                            if tipo == "foto_camara": contexto_archivos += "\n[FOTO DE CÁMARA]\n"
                            elif tipo == "img_previa": contexto_archivos += "\n[IMAGEN PREVIA]\n"
                            else: contexto_archivos += f"\n[IMAGEN ADJUNTA: {archivo.name}]\n"

                    # PROMPT
                    full_prompt = f"""
                    Actúa como Enfermera Clínica Especialista (APN) y Experta en Radiología.
                    CONTEXTO: {contexto}. MODO: {modo}.
                    NOTAS: "{notas}"

                    ⚠️ PRIVACIDAD: Anonimiza nombres.

                    MATERIAL ADJUNTO (Puede incluir Vídeos de TAC/RMN, Informes y Fotos):
                    {contexto_archivos}

                    {f"PROTOCOLO: {texto_protocolo[:20000]}" if texto_protocolo else "USA EVIDENCIA CIENTÍFICA."}

                    INSTRUCCIONES INTEGRALES:
                    - Cruza la información: Si ves algo en el VÍDEO (TAC), búscalo en el PDF (Informe/Analítica).
                    - Analiza la secuencia completa de los vídeos.

                    FORMATO DE SALIDA (2 PARTES con "---"):
                    ---
                    ### ⚡ RESUMEN DEL CASO
                    * **👤 PACIENTE:** [Anonimizado].
                    * **🚨 DIAGNÓSTICO INTEGRAL:** [Síntesis de todas las pruebas].
                    * **🩹 ACCIÓN PRIORITARIA:** [Lo urgente].
                    ---
                    ### 📝 ANÁLISIS DETALLADO
                    1. Hallazgos por prueba (Vídeo, Analítica, etc.).
                    2. Correlación clínica.
                    3. Plan de Cuidados.
                    """
                    
                    if contenido_ia:
                        response = model.generate_content([full_prompt, *contenido_ia], safety_settings=safety_settings)
                    else:
                        response = model.generate_content(full_prompt, safety_settings=safety_settings)
                    
                    partes = response.text.split("---")
                    if len(partes) >= 3:
                        st.markdown(f'<div class="esquema-rapido">{partes[1]}</div>', unsafe_allow_html=True)
                        st.markdown(partes[2])
                    else:
                        st.markdown(response.text)
                        
                    st.balloons()
                    
                except Exception as e:
                    st.error("❌ Error:")
                    st.write(e)
    
    elif not archivos_procesar and st.button("🚀 ANALIZAR AHORA"):
        st.warning("⚠️ Sube al menos un archivo.")
