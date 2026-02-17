import streamlit as st
import google.generativeai as genai
from PIL import Image
import pypdf

# --- CONFIGURACIÓN ---
st.set_page_config(page_title="LabMind Camera", page_icon="📸", layout="wide")

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
    st.title("LabMind 8.0")
    st.caption("📸 Cámara Nativa + Gemini 3")
    
    api_key = st.text_input("🔑 API Key:", type="password")
    
    st.divider()
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
st.title("🩺 Estación Clínica Móvil")

col1, col2 = st.columns([1.2, 2])

with col1:
    st.subheader("1. Captura de Datos")
    
    modo = st.radio("Modo:", ["🩹 Heridas", "📊 Analíticas/Informes", "📉 ECG/Imagen", "🧩 Integral"])
    st.markdown("---")
    
    # --- SELECTOR: ¿ARCHIVO O CÁMARA? ---
    fuente_imagen = st.radio("¿Cómo quieres subir la imagen?", ["📁 Subir Archivo", "📸 Cámara Directa"], horizontal=True)
    
    archivos_procesar = [] 

    # CASO 1: CÁMARA DIRECTA (NUEVO)
    if fuente_imagen == "📸 Cámara Directa":
        # Nota: La cámara solo permite 1 foto a la vez por limitación del navegador
        foto_camara = st.camera_input("Haz la foto ahora")
        if foto_camara:
            archivos_procesar.append(("foto_camara", foto_camara))
            st.success("Foto capturada correctamente.")

    # CASO 2: SUBIR ARCHIVO (EL DE SIEMPRE)
    else:
        if modo == "🩹 Heridas":
            st.info("📸 Sube foto actual (y previa opcional).")
            f_actual = st.file_uploader("FOTO ACTUAL", type=['jpg', 'png', 'jpeg'])
            f_previa = st.file_uploader("FOTO PREVIA (Comparar)", type=['jpg', 'png', 'jpeg'])
            if f_actual: archivos_procesar.append(("img_actual", f_actual))
            if f_previa: archivos_procesar.append(("img_previa", f_previa))

        elif modo == "📊 Analíticas/Informes" or modo == "🧩 Integral":
            st.info("📂 Sube todos los documentos.")
            files = st.file_uploader("Archivos:", type=['pdf', 'jpg', 'png', 'jpeg'], accept_multiple_files=True)
            if files:
                for f in files: archivos_procesar.append(("doc", f))
                
        else: # ECG / Imagen simple
            f = st.file_uploader("Imagen:", type=['jpg', 'png', 'jpeg'])
            if f: archivos_procesar.append(("unico", f))

    st.markdown("---")
    notas = st.text_area("✍️ Notas rápidas:", placeholder="Ej: Úlcera sacra, mal olor...", height=100)

with col2:
    st.subheader("2. Análisis (Gemini 3 Flash)")
    
    if archivos_procesar and st.button("🚀 ANALIZAR AHORA", type="primary"):
        if not api_key:
            st.warning("⚠️ Falta API Key.")
        else:
            with st.spinner("🧠 Procesando imagen..."):
                try:
                    genai.configure(api_key=api_key)
                    model = genai.GenerativeModel("models/gemini-3-flash-preview")
                    
                    # SEGURIDAD OFF
                    safety_settings = [{"category": c, "threshold": "BLOCK_NONE"} for c in ["HARM_CATEGORY_HARASSMENT", "HARM_CATEGORY_HATE_SPEECH", "HARM_CATEGORY_SEXUALLY_EXPLICIT", "HARM_CATEGORY_DANGEROUS_CONTENT"]]
                    
                    # --- PROCESAMIENTO ---
                    contenido_ia = []
                    contexto_archivos = ""
                    
                    for tipo, archivo in archivos_procesar:
                        if archivo.type == "application/pdf":
                            pdf_reader = pypdf.PdfReader(archivo)
                            texto_pdf = ""
                            for page in pdf_reader.pages: texto_pdf += page.extract_text() or ""
                            contexto_archivos += f"\n--- PDF ---\n{texto_pdf}\n"
                        else:
                            # Imagen (sea de archivo o de cámara)
                            img = Image.open(archivo)
                            contenido_ia.append(img)
                            if tipo == "foto_camara": contexto_archivos += "\n[IMAGEN TOMADA CON CÁMARA AHORA MISMO]\n"
                            elif tipo == "img_previa": contexto_archivos += "\n[IMAGEN PREVIA (HISTÓRICO)]\n"
                            else: contexto_archivos += "\n[IMAGEN ADJUNTA]\n"

                    # --- PROMPT ---
                    full_prompt = f"""
                    Actúa como Enfermera Clínica Especialista (APN).
                    CONTEXTO: {contexto}. MODO: {modo}.
                    NOTAS: "{notas}"

                    ⚠️ PRIVACIDAD: Anonimiza nombres (Usa "Paciente [Edad] [Sexo]").

                    MATERIAL ADJUNTO:
                    {contexto_archivos}

                    {f"PROTOCOLO: {texto_protocolo[:20000]}" if texto_protocolo else "USA EVIDENCIA CIENTÍFICA."}

                    FORMATO DE SALIDA (2 PARTES con "---"):
                    ---
                    ### ⚡ RESUMEN
                    * **👤 PACIENTE:** [Anonimizado].
                    * **🚨 DIAGNÓSTICO:** [Principal].
                    * **🩹 ACCIÓN:** [Inmediata].
                    ---
                    ### 📝 ANÁLISIS DETALLADO
                    1. Hallazgos técnicos.
                    2. Plan de Cuidados con EVIDENCIA.
                    """
                    
                    # Llamada
                    if contenido_ia:
                        response = model.generate_content([full_prompt, *contenido_ia], safety_settings=safety_settings)
                    else:
                        response = model.generate_content(full_prompt, safety_settings=safety_settings)
                    
                    # Render
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
        st.warning("⚠️ Haz una foto o sube un archivo.")
