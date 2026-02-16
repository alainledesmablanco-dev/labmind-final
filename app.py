import streamlit as st
import google.generativeai as genai
from PIL import Image
import pypdf

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="LabMind Wound Care", page_icon="🩹", layout="wide")

# --- ESTILOS CSS ---
st.markdown("""
<style>
    .stButton>button { width: 100%; border-radius: 10px; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# --- GESTIÓN DE MEMORIA ---
if "mensajes" not in st.session_state:
    st.session_state.mensajes = []

# --- FUNCIONES ---
def leer_pdf(archivo):
    pdf_reader = pypdf.PdfReader(archivo)
    texto = ""
    for page in pdf_reader.pages:
        texto += page.extract_text()
    return texto

# --- BARRA LATERAL (CONFIGURACIÓN) ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3063/3063176.png", width=70)
    st.title("LabMind 5.2")
    st.caption("Especialista en Heridas (Protocolo Seguro)")
    
    api_key = st.text_input("🔑 Google API Key:", type="password")
    
    st.divider()
    
    # SECCIÓN DE PROTOCOLOS (EVIDENCIA)
    st.write("📚 **Validación con Evidencia**")
    protocolo_pdf = st.file_uploader("Sube tu Protocolo de Heridas/Unidad (PDF)", type="pdf")
    texto_protocolo = ""
    if protocolo_pdf:
        texto_protocolo = leer_pdf(protocolo_pdf)
        st.success("✅ Protocolo aprendido.")
    else:
        st.info("ℹ️ Sin PDF, usaré Guías GNEAUPP/EPUAP.")

    st.divider()
    contexto = st.selectbox("Contexto Paciente:", ["Hospitalización", "Urgencias", "Atención Primaria / Domicilio", "UCI", "Residencia"])

# --- CUERPO PRINCIPAL ---
st.title("🩺 Unidad de Análisis Clínico")

# PESTAÑAS
tab_analisis, tab_chat = st.tabs(["👁️ Análisis & Curas", "💬 Chat / Segunda Opinión"])

# --- PESTAÑA 1: EL ESCÁNER VISUAL ---
with tab_analisis:
    col1, col2 = st.columns([1.5, 2])
    
    with col1:
        st.subheader("1. Configuración del Caso")
        
        # SELECTOR DE MODO
        modo = st.radio("¿Qué analizamos?", 
                        ["🩹 Heridas & Úlceras (UPP)", "🩸 Analítica", "📈 ECG", "💀 Rx/TAC", "📝 Informe Médico"])
        
        st.markdown("---")
        
        # SUBIDA DE IMÁGENES
        archivo_actual = st.file_uploader("📸 FOTO ACTUAL (Obligatoria)", type=['jpg', 'png', 'jpeg', 'pdf'])
        archivo_previo = st.file_uploader("FOTO PREVIA (Opcional - Evolución)", type=['jpg', 'png', 'jpeg'])
        
        # INPUT DE CONTEXTO / TRATAMIENTO ACTUAL
        st.markdown("---")
        info_extra = st.text_area("✍️ Localización y Notas:", 
                                  placeholder="Ej: Talón derecho. Placa negra seca. ¿Le pongo hidrogel?",
                                  height=100)

    with col2:
        st.subheader("2. Resultados y Validación")
        
        if archivo_actual and st.button("🚀 ANALIZAR Y VALIDAR", type="primary"):
            if not api_key:
                st.error("❌ Falta la API Key")
            else:
                with st.spinner("🔍 Analizando tejidos, localización y aplicando protocolos de seguridad..."):
                    try:
                        genai.configure(api_key=api_key)
                        model = genai.GenerativeModel("gemini-1.5-pro") 
                        
                        # PREPARACIÓN DE IMÁGENES
                        contenido = []
                        prompt_archivos = ""
                        
                        if archivo_actual.type == "application/pdf":
                            prompt_archivos += f"\nDOCUMENTO ACTUAL:\n{leer_pdf(archivo_actual)}"
                        else:
                            contenido.append(Image.open(archivo_actual))
                            prompt_archivos += "\n[IMAGEN 1: ESTADO ACTUAL]"

                        if archivo_previo:
                            contenido.append(Image.open(archivo_previo))
                            prompt_archivos += "\n[IMAGEN 2: ESTADO PREVIO - COMPARAR EVOLUCIÓN]"

                        # PREPARACIÓN DEL CONOCIMIENTO (PDF)
                        prompt_protocolo = ""
                        if texto_protocolo:
                            prompt_protocolo = f"⚠️ IMPORTANTE: JUSTIFICA TUS RESPUESTAS USANDO ESTE PROTOCOLO:\n{texto_protocolo[:30000]}\nCita la página si es posible."
                        else:
                            prompt_protocolo = "⚠️ IMPORTANTE: JUSTIFICA TUS RESPUESTAS USANDO GUÍAS INTERNACIONALES (GNEAUPP, EPUAP)."

                        # --- EL CEREBRO DE LA HERIDA (CON REGLA DEL TALÓN) ---
                        full_prompt = f"""
                        Actúa como Enfermera Clínica Especialista en Heridas (Estomaterapeuta).
                        CONTEXTO: {contexto}. MODO: {modo}.
                        NOTAS USUARIO (Localización/Dudas): "{info_extra}"
                        
                        {prompt_archivos}
                        {prompt_protocolo}
                        
                        TAREA ESPECÍFICA SEGÚN MODO:
                        
                        SI ES 🩹 HERIDAS & ÚLCERAS:
                        1. DIAGNÓSTICO:
                           - Tipo y Estadio.
                           - **LOCALIZACIÓN:** Intenta inferirla por la imagen o las notas (¿Es Sacro? ¿Es Talón?).
                        
                        2. ANÁLISIS TISULAR (TIME):
                           - % Granulación / % Esfacelos / % Necrosis.
                           - Signos de Infección (Eritema, calor, exudado purulento).
                        
                        3. REGLA DE SEGURIDAD (TALÓN vs RESTO):
                           - **SI ES TALÓN + NECROSIS SECA (Sin infección):** ¡ALERTA ROJA! NO RECOMENDAR DESBRIDAMIENTO NI HUMEDAD (Hidrogeles).
                             La indicación correcta es: MANTENER SECA, PINTAR CON POVIDONA/BETADINE Y PROTEGER DE PRESIÓN (Flotación).
                           - **SI ES OTRA ZONA o HAY INFECCIÓN:**
                             Entonces sí, sugiere desbridamiento (Enzimático/Autolítico).
                        
                        4. VALIDACIÓN TRATAMIENTO: 
                           - Compara lo que hace el usuario con la regla de seguridad anterior.
                        
                        5. PLAN DE CURAS:
                           - Producto exacto.
                           - Frecuencia de cura.
                        
                        FORMATO DE SALIDA (Markdown):
                        - 🩺 DIAGNÓSTICO Y TEJIDOS
                        - 🚨 REGLA DE SEGURIDAD APLICADA (Explica por qué)
                        - ✅/❌ VALIDACIÓN TRATAMIENTO
                        - 📝 PLAN DE CUIDADOS (Con Citas)
                        """
                        
                        # GENERAR
                        response = model.generate_content([full_prompt, *contenido])
                        st.markdown(response.text)
                        
                        # GUARDAR EN CHAT
                        st.session_state.mensajes.append({"role": "assistant", "content": f"**Análisis {modo}:**\n{response.text}"})
                        
                    except Exception as e:
                        st.error(f"Error: {e}")

# --- PESTAÑA 2: CHAT CLÍNICO ---
with tab_chat:
    st.info("💬 Habla con la IA sobre el caso.")
    for msg in st.session_state.mensajes:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
    if prompt := st.chat_input("Duda sobre el caso..."):
        if not api_key: st.warning("Falta API Key")
        else:
            st.session_state.mensajes.append({"role": "user", "content": prompt})
            with st.chat_message("user"): st.markdown(prompt)
            with st.chat_message("assistant"):
                with st.spinner("Pensando..."):
                    genai.configure(api_key=api_key)
                    model = genai.GenerativeModel("gemini-1.5-flash")
                    historial = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.mensajes[-6:]])
                    response = model.generate_content(f"Actúa como Enfermera Experta. Historial: {historial}\nPregunta Usuario: {prompt}\nUsa el protocolo PDF si existe.")
                    st.markdown(response.text)
                    st.session_state.mensajes.append({"role": "assistant", "content": response.text})
