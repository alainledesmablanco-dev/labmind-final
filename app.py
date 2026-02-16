import streamlit as st
import google.generativeai as genai
from PIL import Image
import pypdf

# --- CONFIGURACIÓN ---
st.set_page_config(page_title="LabMind Uncensored", page_icon="🧬", layout="wide")

# --- ESTILOS ---
st.markdown("""
<style>
    .stButton>button { width: 100%; border-radius: 8px; font-weight: bold; }
    .reportview-container { background: #f0f2f6; }
</style>
""", unsafe_allow_html=True)

# --- BARRA LATERAL ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3063/3063176.png", width=80)
    st.title("LabMind Libre")
    st.caption("🔓 Sin filtros de seguridad")
    
    # 1. API KEY
    api_key = st.text_input("🔑 Tu API Key:", type="password")
    
    # 2. SELECTOR DE MODELOS
    modelo_elegido = "models/gemini-1.5-flash" # Valor por defecto seguro
    
    if api_key:
        try:
            genai.configure(api_key=api_key)
            # Obtenemos la lista real de modelos que tu llave permite
            lista_modelos = []
            for m in genai.list_models():
                if 'generateContent' in m.supported_generation_methods:
                    lista_modelos.append(m.name)
            
            st.success(f"✅ Llave válida. {len(lista_modelos)} modelos disponibles.")
            
            # EL MENÚ DESPLEGABLE
            modelo_elegido = st.selectbox("🧠 ELIGE CEREBRO:", lista_modelos, index=0)
            st.caption("Recomendado: gemini-1.5-flash o gemini-2.0-flash (No dan error de cuota)")
            
        except Exception as e:
            st.error("❌ Clave inválida o error de conexión.")

    st.divider()
    
    # 3. PROTOCOLO
    protocolo_pdf = st.file_uploader("Sube Protocolo (PDF)", type="pdf")
    texto_protocolo = ""
    if protocolo_pdf:
        try:
            pdf_reader = pypdf.PdfReader(protocolo_pdf)
            for page in pdf_reader.pages: texto_protocolo += page.extract_text() or ""
            st.success("✅ Protocolo activo")
        except: st.error("Error PDF")

    contexto = st.selectbox("Contexto:", ["Hospitalización", "Urgencias", "UCI", "Domicilio"])

# --- ZONA PRINCIPAL ---
st.title("🩺 Unidad Clínica (Modo Selector)")

col1, col2 = st.columns([1, 2])

with col1:
    modo = st.radio("Modo:", ["🩹 Heridas (UPP)", "🩸 Analítica", "📈 ECG", "💀 Rx/TAC"])
    archivo = st.file_uploader("Subir Caso:", type=['jpg', 'png', 'jpeg', 'pdf'])
    notas = st.text_area("Notas / Dudas:", height=100)

with col2:
    if archivo and st.button("🚀 ANALIZAR (SIN FILTROS)", type="primary"):
        if not api_key:
            st.warning("⚠️ Falta API Key en la izquierda.")
        else:
            with st.spinner(f"Analizando con {modelo_elegido} y filtros desactivados..."):
                try:
                    genai.configure(api_key=api_key)
                    model = genai.GenerativeModel(modelo_elegido)
                    
                    # --- AQUÍ ESTÁ LA MAGIA ANTI-CENSURA ---
                    # Configuramos todos los filtros en BLOCK_NONE (Permitir todo)
                    configuracion_seguridad = [
                        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
                    ]
                    
                    # Preparar contenido
                    contenido = []
                    prompt_extra = ""
                    
                    if archivo.type == "application/pdf":
                         if not texto_protocolo: # Si es el archivo principal
                            pdf_reader = pypdf.PdfReader(archivo)
                            text = ""
                            for page in pdf_reader.pages: text += page.extract_text()
                            prompt_extra = f"PDF ADJUNTO:\n{text}"
                    else:
                        contenido.append(Image.open(archivo))
                        prompt_extra = "Analiza esta imagen clínica detalladamente."
                    
                    # Prompt
                    full_prompt = f"""
                    Actúa como Enfermera Experta. Contexto: {contexto}. Modo: {modo}.
                    Notas Usuario: {notas}.
                    
                    {prompt_extra}
                    {f"USA ESTE PROTOCOLO: {texto_protocolo[:10000]}" if texto_protocolo else ""}
                    
                    IMPORTANTE: Es una consulta médica profesional. Describe hallazgos objetivos (tejidos, sangre, heridas) con precisión técnica.
                    
                    TAREA: Diagnóstico, Alertas y Plan de Cuidados.
                    """
                    
                    # Llamada con Safety Settings
                    response = model.generate_content(
                        [full_prompt, *contenido],
                        safety_settings=configuracion_seguridad
                    )
                    
                    st.markdown(response.text)
                    
                except Exception as e:
                    st.error("❌ Error:")
                    st.write(e)
                    # Explicación de errores comunes
                    err_msg = str(e)
                    if "429" in err_msg:
                        st.warning("💡 Has elegido un modelo muy potente (Pro/Deep) y Google te ha frenado. Elige un modelo 'Flash' en la lista.")
                    elif "block" in err_msg.lower():
                        st.warning("🛡️ Incluso sin filtros, Google ha bloqueado la imagen. Intenta recortarla un poco.")
