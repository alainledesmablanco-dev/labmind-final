import streamlit as st
import google.generativeai as genai
from PIL import Image
import pypdf

# --- CONFIGURACIÓN ---
st.set_page_config(page_title="LabMind Uncensored", page_icon="🏥", layout="wide")

# --- BARRA LATERAL ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3063/3063176.png", width=80)
    st.title("LabMind 6.0")
    st.caption("✅ Filtros Médicos Activados")
    
    api_key = st.text_input("🔑 API Key:", type="password")
    
    st.divider()
    protocolo_pdf = st.file_uploader("Sube Protocolo (PDF)", type="pdf")
    texto_protocolo = ""
    if protocolo_pdf:
        try:
            pdf_reader = pypdf.PdfReader(protocolo_pdf)
            for page in pdf_reader.pages: texto_protocolo += page.extract_text() or ""
            st.success("✅ Protocolo cargado")
        except: st.error("Error PDF")

    contexto = st.selectbox("Contexto:", ["Hospitalización", "Urgencias", "UCI", "Domicilio"])

# --- CUERPO PRINCIPAL ---
st.title("🩺 Unidad Clínica (Sin Bloqueos)")

col1, col2 = st.columns([1, 2])
with col1:
    modo = st.radio("Modo:", ["🩹 Heridas (UPP)", "🩸 Analítica", "📈 ECG", "💀 Rx/TAC"])
    archivo = st.file_uploader("Foto:", type=['jpg', 'png', 'jpeg', 'pdf'])
    notas = st.text_area("Notas:", height=100)

with col2:
    if archivo and st.button("🚀 ANALIZAR", type="primary"):
        if not api_key: st.warning("⚠️ Falta API Key")
        else:
            with st.spinner("Analizando sin filtros de seguridad..."):
                try:
                    genai.configure(api_key=api_key)
                    
                    # 1. USAMOS FLASH (Es el que no da error de Cuota)
                    model = genai.GenerativeModel("models/gemini-1.5-flash")
                    
                    # 2. CONFIGURACIÓN DE SEGURIDAD (LA CLAVE DEL ÉXITO)
                    # Esto le dice a Google que permita ver "Gore" (Heridas) y contenido médico
                    safety_settings = [
                        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
                    ]
                    
                    # Preparar contenido
                    contenido = []
                    prompt_extra = ""
                    
                    if archivo.type == "application/pdf":
                         if not texto_protocolo:
                            pdf_reader = pypdf.PdfReader(archivo)
                            text = ""
                            for page in pdf_reader.pages: text += page.extract_text()
                            prompt_extra = f"PDF ADJUNTO:\n{text}"
                    else:
                        contenido.append(Image.open(archivo))
                        prompt_extra = "Analiza esta imagen médica."
                    
                    prompt = f"""
                    Actúa como Enfermero Experto. Contexto: {contexto}. Modo: {modo}.
                    Notas: {notas}.
                    
                    {prompt_extra}
                    {f"Protocolo: {texto_protocolo[:10000]}" if texto_protocolo else ""}
                    
                    Dame Diagnóstico, Alertas y Plan de Cuidados.
                    """
                    
                    # 3. LLAMADA CON LOS SETTINGS DE SEGURIDAD
                    response = model.generate_content(
                        [prompt, *contenido],
                        safety_settings=safety_settings
                    )
                    
                    st.markdown(response.text)
                    
                except Exception as e:
                    # Si falla, mostramos el error exacto
                    st.error("Error:")
                    st.write(e)
                    if "block_reason" in str(e):
                        st.warning("La IA sigue intentando bloquear la imagen por seguridad. Intenta recortar la foto para que se vea menos 'sangrienta' o prueba otra vez.")
