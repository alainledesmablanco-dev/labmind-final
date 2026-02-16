import streamlit as st
import google.generativeai as genai
from PIL import Image
import pypdf

# --- CONFIGURACIÓN ---
st.set_page_config(page_title="LabMind Privacy", page_icon="🛡️", layout="wide")

# --- ESTILOS ---
st.markdown("""
<style>
    .stButton>button { width: 100%; border-radius: 8px; font-weight: bold; background-color: #0066cc; color: white; }
    .esquema-rapido { background-color: #e8f4ff; padding: 15px; border-radius: 10px; border-left: 5px solid #0066cc; margin-bottom: 20px; }
    h3 { color: #004a99; }
</style>
""", unsafe_allow_html=True)

# --- BARRA LATERAL ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3063/3063176.png", width=80)
    st.title("LabMind 6.1")
    st.caption("🛡️ Privacidad + Heridas Pro")
    
    api_key = st.text_input("🔑 API Key:", type="password")
    
    st.divider()
    st.write("📚 **Evidencia / Protocolo**")
    protocolo_pdf = st.file_uploader("Sube tu guía (PDF)", type="pdf")
    texto_protocolo = ""
    if protocolo_pdf:
        try:
            pdf_reader = pypdf.PdfReader(protocolo_pdf)
            for page in pdf_reader.pages: texto_protocolo += page.extract_text() or ""
            st.success("✅ Protocolo memorizado.")
        except: st.error("Error PDF")

    contexto = st.selectbox("Contexto Paciente:", ["Hospitalización", "Atención Primaria/Domicilio", "UCI", "Residencia"])

# --- ZONA PRINCIPAL ---
st.title("🩺 Unidad Clínica (Datos Anonimizados)")

col1, col2 = st.columns([1.2, 2])

with col1:
    st.subheader("1. Datos del Caso")
    modo = st.radio("Selecciona Modo:", ["🩹 Heridas (UPP/Evolución)", "🩸 Analítica", "📈 ECG", "💀 Rx/TAC"])
    st.markdown("---")
    
    # --- LÓGICA DE ARCHIVOS ---
    archivo_actual = None
    archivo_previo = None
    archivo_gen = None 

    if modo == "🩹 Heridas (UPP/Evolución)":
        st.info("📸 Modo Evolutivo: Sube foto actual y previa.")
        archivo_actual = st.file_uploader("1️⃣ FOTO ACTUAL (Obligatoria)", type=['jpg', 'png', 'jpeg'])
        archivo_previo = st.file_uploader("2️⃣ FOTO PREVIA (Opcional)", type=['jpg', 'png', 'jpeg'])
    else:
        archivo_gen = st.file_uploader("Subir Documento/Foto:", type=['jpg', 'png', 'jpeg', 'pdf'])

    st.markdown("---")
    notas = st.text_area("✍️ Notas:", placeholder="Ej: Diabético tipo 2...", height=100)

with col2:
    st.subheader("2. Análisis Estructurado IA")
    
    # Comprobar si hay archivos para activar botón
    listo = False
    if modo == "🩹 Heridas (UPP/Evolución)" and archivo_actual: listo = True
    elif modo != "🩹 Heridas (UPP/Evolución)" and archivo_gen: listo = True

    if listo and st.button("🚀 ANALIZAR (ANÓNIMO)", type="primary"):
        if not api_key:
            st.warning("⚠️ Falta API Key.")
        else:
            with st.spinner("🧠 Analizando, anonimizando datos y consultando evidencia..."):
                try:
                    genai.configure(api_key=api_key)
                    # Usamos el modelo potente
                    model = genai.GenerativeModel("models/gemini-3-flash-preview")
                    
                    # Seguridad OFF para ver heridas
                    safety_settings = [{"category": c, "threshold": "BLOCK_NONE"} for c in ["HARM_CATEGORY_HARASSMENT", "HARM_CATEGORY_HATE_SPEECH", "HARM_CATEGORY_SEXUALLY_EXPLICIT", "HARM_CATEGORY_DANGEROUS_CONTENT"]]
                    
                    # Preparar contenido
                    contenido = []
                    prompt_imgs_text = ""
                    
                    if modo == "🩹 Heridas (UPP/Evolución)":
                        contenido.append(Image.open(archivo_actual))
                        prompt_imgs_text = "IMAGEN 1: ESTADO ACTUAL.\n"
                        if archivo_previo:
                            contenido.append(Image.open(archivo_previo))
                            prompt_imgs_text += "IMAGEN 2: ESTADO PREVIO (Comparar evolución).\n"
                            
                    elif archivo_gen: 
                        if archivo_gen.type == "application/pdf":
                             if not texto_protocolo: # Solo leer si no es protocolo
                                pdf_reader = pypdf.PdfReader(archivo_gen)
                                text = ""
                                for page in pdf_reader.pages: text += page.extract_text()
                                prompt_imgs_text = f"CONTENIDO DEL PDF:\n{text}"
                        else:
                            contenido.append(Image.open(archivo_gen))
                            prompt_imgs_text = "Analiza esta imagen clínica."
                    
                    # --- PROMPT CON ESCUDO DE PRIVACIDAD ---
                    full_prompt = f"""
                    Actúa como Enfermera Clínica Especialista (APN).
                    CONTEXTO: {contexto}. MODO: {modo}.
                    NOTAS: "{notas}"

                    {prompt_imgs_text}
                    {f"USA ESTE PROTOCOLO: {texto_protocolo[:20000]}" if texto_protocolo else "USA GUÍAS GNEAUPP/EPUAP."}

                    ⚠️ REGLA DE ORO DE PRIVACIDAD (GDPR):
                    1. ESTÁ PROHIBIDO ESCRIBIR EL NOMBRE REAL DEL PACIENTE.
                    2. Si detectas un nombre en el documento (Ej: "Alain...", "María..."), IGNÓRALO.
                    3. Refiérete al paciente ÚNICAMENTE como: "Paciente [Varón/Mujer] de [Edad] años".

                    ***FORMATO DE SALIDA (2 PARTES)***:
                    Usa una línea separadora "---" entre las dos partes.

                    ---
                    ### ⚡ RESUMEN RÁPIDO
                    (Formato lista breve con iconos)
                    * **👤 PACIENTE:** [Solo Edad y Sexo detectados].
                    * **👁️ DIAGNÓSTICO:** [Lo que ves principal].
                    * **🩹 ACCIÓN INMEDIATA:** [Producto/Acción clave].
                    * **🔄 EVOLUCIÓN:** [Mejora/Empeora/Estable/No valorable].
                    ---
                    
                    ### 📝 ANÁLISIS DETALLADO Y EVIDENCIA
                    1. **Valoración Completa:**
                       - Si es Herida: TIME (Tejido, Infección, Bordes, Exudado).
                       - Si es Analítica: Valores fuera de rango y su significado clínico.
                    2. **Comparativa Evolutiva** (si hay datos previos).
                    3. **PLAN DE CUIDADOS (Justificado):**
                       - Pasos exactos.
                       - **CITA LA EVIDENCIA** en cada recomendación. Ej: "Usar Plata [Fuente: Guía GNEAUPP]".
                    """
                    
                    # Llamada
                    response = model.generate_content([full_prompt, *contenido], safety_settings=safety_settings)
                    
                    # Renderizado bonito
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
    elif not listo and st.button("🚀 ANALIZAR (ANÓNIMO)"):
        st.warning("⚠️ Sube el archivo primero.")
