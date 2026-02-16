import streamlit as st
import google.generativeai as genai
from PIL import Image
import pypdf

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="LabMind Clásico", page_icon="🛡️", layout="wide")

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

# --- BARRA LATERAL ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3063/3063176.png", width=70)
    st.title("LabMind v5.4")
    st.caption("Modo Compatibilidad (Sin errores)")
    
    api_key = st.text_input("🔑 Google API Key:", type="password")
    
    st.divider()
    
    st.write("📚 **Evidencia (Opcional)**")
    protocolo_pdf = st.file_uploader("Sube Protocolo (PDF)", type="pdf")
    texto_protocolo = ""
    if protocolo_pdf:
        texto_protocolo = leer_pdf(protocolo_pdf)
        st.success("✅ Protocolo cargado.")

    st.divider()
    contexto = st.selectbox("Contexto:", ["Hospitalización", "Urgencias", "UCI", "Residencia", "Domicilio"])

# --- CUERPO PRINCIPAL ---
st.title("🩺 Unidad de Análisis Clínico")

tab_analisis, tab_chat = st.tabs(["👁️ Escáner Visual", "💬 Chat Clínico"])

# --- PESTAÑA 1: ANÁLISIS ---
with tab_analisis:
    col1, col2 = st.columns([1.5, 2])
    
    with col1:
        st.subheader("1. Caso")
        modo = st.radio("Tipo:", ["🩹 Heridas (UPP)", "🩸 Analítica", "📈 ECG", "💀 Rx/TAC", "📝 Informe"])
        st.markdown("---")
        archivo_actual = st.file_uploader("📸 FOTO (Obligatoria)", type=['jpg', 'png', 'jpeg', 'pdf'])
        st.markdown("---")
        info_extra = st.text_area("✍️ Notas / Duda:", placeholder="Ej: Talón negro. ¿Desbrido?", height=100)

    with col2:
        st.subheader("2. Resultados")
        
        if archivo_actual and st.button("🚀 ANALIZAR", type="primary"):
            if not api_key:
                st.error("❌ Pega tu API Key primero.")
            else:
                with st.spinner("Procesando con motor clásico..."):
                    try:
                        genai.configure(api_key=api_key)
                        
                        # LÓGICA DE SELECCIÓN DE MODELO (EL TRUCO ANTI-ERROR)
                        # Si es PDF (solo texto) -> Usamos gemini-pro
                        # Si es Imagen -> Usamos gemini-pro-vision
                        
                        contenido = []
                        nombre_modelo = "gemini-pro" # Por defecto texto
                        
                        prompt_archivos = ""
                        
                        if archivo_actual.type == "application/pdf":
                            prompt_archivos += f"\nDOCUMENTO PDF:\n{leer_pdf(archivo_actual)}"
                            nombre_modelo = "gemini-pro" 
                        else:
                            contenido.append(Image.open(archivo_actual))
                            prompt_archivos += "\n[IMAGEN ADJUNTA]"
                            nombre_modelo = "gemini-pro-vision" # Modelo visual clásico
                        
                        # Cargamos el modelo seguro
                        model = genai.GenerativeModel(nombre_modelo)

                        # Preparar Protocolo
                        prompt_protocolo = ""
                        if texto_protocolo:
                            prompt_protocolo = f"USA ESTE PROTOCOLO:\n{texto_protocolo[:10000]}\n"
                        else:
                            prompt_protocolo = "USA GUÍAS CLÍNICAS ESTÁNDAR (GNEAUPP, AHA)."

                        # El Prompt Maestro
                        full_prompt = f"""
                        Actúa como Enfermera Experta. Contexto: {contexto}. Modo: {modo}.
                        Notas Usuario: "{info_extra}"
                        
                        {prompt_archivos}
                        {prompt_protocolo}
                        
                        TAREAS:
                        1. Si es HERIDA: Diagnostica (TIME), busca NECROSIS.
                           * REGLA SEGURIDAD: SI ES TALÓN + NECROSIS SECA -> NO DESBRIDAR.
                           * Valida el tratamiento del usuario.
                        2. Si es ECG/Analítica: Busca valores críticos.
                        
                        Responde en Markdown claro con: Diagnóstico, Alertas y Plan.
                        """
                        
                        # Generar
                        if nombre_modelo == "gemini-pro-vision":
                            response = model.generate_content([full_prompt, *contenido])
                        else:
                            response = model.generate_content(full_prompt)
                            
                        st.markdown(response.text)
                        st.session_state.mensajes.append({"role": "assistant", "content": f"**Análisis:**\n{response.text}"})
                        
                    except Exception as e:
                        st.error(f"Error técnico: {e}")
                        st.info("Prueba a refrescar la página.")

# --- PESTAÑA 2: CHAT ---
with tab_chat:
    st.info("💬 Chat (Solo Texto)")
    for msg in st.session_state.mensajes:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            
    if prompt := st.chat_input("Duda..."):
        if not api_key: st.warning("Falta API Key")
        else:
            st.session_state.mensajes.append({"role": "user", "content": prompt})
            with st.chat_message("user"): st.markdown(prompt)
            with st.chat_message("assistant"):
                with st.spinner("Pensando..."):
                    try:
                        genai.configure(api_key=api_key)
                        # Para chat usamos siempre el modelo de texto clásico
                        model = genai.GenerativeModel("gemini-pro")
                        
                        historial = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.mensajes[-6:]])
                        response = model.generate_content(f"Actúa como Enfermera. Historial:\n{historial}\nUsuario: {prompt}")
                        
                        st.markdown(response.text)
                        st.session_state.mensajes.append({"role": "assistant", "content": response.text})
                    except Exception as e:
                        st.error(f"Error: {e}")
