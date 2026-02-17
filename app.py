import streamlit as st
import google.generativeai as genai
from PIL import Image
import pypdf

# --- CONFIGURACIÓN ---
st.set_page_config(page_title="LabMind 3.0 Flash", page_icon="⚡", layout="wide")

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
    st.title("LabMind Pro")
    st.caption("🚀 Motor Fijo: Gemini 3 Flash")
    
    api_key = st.text_input("🔑 API Key:", type="password")
    
    st.divider()
    st.write("📚 **Protocolo (Opcional)**")
    protocolo_pdf = st.file_uploader("Sube Guía PDF", type="pdf")
    texto_protocolo = ""
    if protocolo_pdf:
        try:
            pdf_reader = pypdf.PdfReader(protocolo_pdf)
            for page in pdf_reader.pages: texto_protocolo += page.extract_text() or ""
            st.success("✅ Protocolo Activo")
        except: st.error("Error PDF")

    contexto = st.selectbox("Contexto:", ["Hospitalización", "Urgencias", "UCI", "Domicilio"])

# --- ZONA PRINCIPAL ---
st.title("🩺 Unidad Clínica (Versión Gemini 3)")

col1, col2 = st.columns([1.2, 2])

with col1:
    st.subheader("1. Selección")
    modo = st.radio("Modo:", ["🩹 Heridas (Evolución)", "🩸 Analítica", "📈 ECG", "💀 Rx/TAC/RMN"])
    st.markdown("---")
    
    # Lógica de Archivos
    archivo_actual = None
    archivo_previo = None
    archivo_gen = None 

    if modo == "🩹 Heridas (Evolución)":
        st.info("📸 Sube foto actual y previa (opcional).")
        archivo_actual = st.file_uploader("1️⃣ FOTO ACTUAL", type=['jpg', 'png', 'jpeg'])
        archivo_previo = st.file_uploader("2️⃣ FOTO PREVIA", type=['jpg', 'png', 'jpeg'])
    else:
        archivo_gen = st.file_uploader("Subir Archivo:", type=['jpg', 'png', 'jpeg', 'pdf'])

    st.markdown("---")
    notas = st.text_area("✍️ Notas:", placeholder="Ej: Diabético, úlcera en talón...", height=100)

with col2:
    st.subheader("2. Análisis IA")
    
    # Validar si hay archivo
    listo = False
    if modo == "🩹 Heridas (Evolución)" and archivo_actual: listo = True
    elif modo != "🩹 Heridas (Evolución)" and archivo_gen: listo = True

    if listo and st.button("🚀 ANALIZAR CON GEMINI 3", type="primary"):
        if not api_key:
            st.warning("⚠️ Falta API Key.")
        else:
            with st.spinner("🧠 Gemini 3 Flash pensando (Modo Privado)..."):
                try:
                    genai.configure(api_key=api_key)
                    
                    # --- AQUÍ FORZAMOS EL MODELO QUE TE GUSTA ---
                    model = genai.GenerativeModel("models/gemini-3-flash-preview")
                    
                    # --- SEGURIDAD OFF (Para ver heridas) ---
                    safety_settings = [
                        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
                    ]
                    
                    # Preparar contenido
                    contenido = []
                    prompt_imgs_text = ""
                    
                    if modo == "🩹 Heridas (Evolución)":
                        contenido.append(Image.open(archivo_actual))
                        prompt_imgs_text = "IMAGEN 1: ESTADO ACTUAL.\n"
                        if archivo_previo:
                            contenido.append(Image.open(archivo_previo))
                            prompt_imgs_text += "IMAGEN 2: ESTADO PREVIO (Comparar).\n"
                            
                    elif archivo_gen: 
                        if archivo_gen.type == "application/pdf":
                             if not texto_protocolo:
                                pdf_reader = pypdf.PdfReader(archivo_gen)
                                text = ""
                                for page in pdf_reader.pages: text += page.extract_text()
                                prompt_imgs_text = f"CONTENIDO PDF:\n{text}"
                        else:
                            contenido.append(Image.open(archivo_gen))
                            prompt_imgs_text = "Analiza esta imagen médica."
                    
                    # --- PROMPT MAESTRO (CON PRIVACIDAD Y EVIDENCIA) ---
                    full_prompt = f"""
                    Actúa como Enfermera Clínica Especialista (APN).
                    CONTEXTO: {contexto}. MODO: {modo}.
                    NOTAS: "{notas}"

                    {prompt_imgs_text}
                    {f"USA ESTE PROTOCOLO: {texto_protocolo[:20000]}" if texto_protocolo else "USA EVIDENCIA CIENTÍFICA."}

                    ⚠️ PRIVACIDAD: NO ESCRIBAS NOMBRES REALES. Usa "Paciente [Edad] [Sexo]".

                    ***FORMATO DE SALIDA (2 PARTES)***:
                    Usa "---" para separar.

                    ---
                    ### ⚡ RESUMEN
                    * **👤 PACIENTE:** [Edad/Sexo Anonimizado].
                    * **👁️ DIAGNÓSTICO:** [Principal].
                    * **🩹 ACCIÓN CLAVE:** [Lo urgente].
                    * **🔄 EVOLUCIÓN:** [Mejora/Empeora].
                    ---
                    
                    ### 📝 ANÁLISIS DETALLADO
                    1. **Valoración Técnica:** (TIME en heridas, valores en analítica).
                    2. **Comparativa** (si hay datos previos).
                    3. **PLAN DE CUIDADOS:**
                       - Pasos exactos.
                       - **CITA LA EVIDENCIA** en cada recomendación.
                    """
                    
                    # Llamada
                    response = model.generate_content(
                        [full_prompt, *contenido], 
                        safety_settings=safety_settings
                    )
                    
                    # Renderizado
                    texto = response.text
                    partes = texto.split("---")
                    
                    if len(partes) >= 3:
                        st.markdown(f'<div class="esquema-rapido">{partes[1]}</div>', unsafe_allow_html=True)
                        st.markdown(partes[2])
                    else:
                        st.markdown(texto)
                        
                    st.balloons()
                    
                except Exception as e:
                    st.error("❌ Error:")
                    st.write(e)
                    if "429" in str(e): st.warning("Gemini 3 está saturado. Espera 1 minuto.")
    
    elif not listo and st.button("🚀 ANALIZAR CON GEMINI 3"):
        st.warning("⚠️ Sube el archivo primero.")
