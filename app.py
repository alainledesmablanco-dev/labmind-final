import streamlit as st
import google.generativeai as genai
from PIL import Image
import pypdf

# Configuración
st.set_page_config(page_title="LabMind Final", page_icon="🏥", layout="wide")

# --- CHIVATO DE VERSIÓN (Para confirmar que se arregló) ---
try:
    ver = genai.__version__
except:
    ver = "Vieja"

# --- BARRA LATERAL ---
with st.sidebar:
    st.title("LabMind v5.5")
    # Si esto sale verde y pone 0.7.2 o superior, ¡HA TRIUNFADO!
    if ver >= "0.7.0":
        st.success(f"✅ Sistema Actualizado: v{ver}")
    else:
        st.error(f"❌ Sistema Obsoleto: v{ver}")
    
    api_key = st.text_input("🔑 Tu API Key:", type="password")
    
    st.divider()
    contexto = st.selectbox("Contexto:", ["Hospitalización", "Urgencias", "UCI"])

# --- CUERPO ---
st.title("🩺 Unidad Clínica IA")
st.info("Sube una foto para analizarla.")

col1, col2 = st.columns([1, 2])
with col1:
    modo = st.radio("Modo:", ["🩹 Heridas", "🩸 Analítica", "📈 ECG", "💀 Rx/TAC"])
    archivo = st.file_uploader("Subir Foto:", type=['jpg', 'png', 'jpeg', 'pdf'])
    notas = st.text_area("Dudas:", height=100)

with col2:
    if archivo and st.button("🚀 ANALIZAR"):
        if not api_key: st.error("Falta API Key")
        else:
            with st.spinner("Conectando con Gemini 1.5 Flash..."):
                try:
                    genai.configure(api_key=api_key)
                    # Usamos el modelo moderno que requiere la librería nueva
                    model = genai.GenerativeModel("gemini-1.5-flash")
                    
                    contenido = []
                    if archivo.type == "application/pdf":
                        pdf_reader = pypdf.PdfReader(archivo)
                        texto = ""
                        for page in pdf_reader.pages: texto += page.extract_text()
                        prompt_contenido = f"PDF:\n{texto}"
                    else:
                        contenido.append(Image.open(archivo))
                        prompt_contenido = "Analiza esta imagen."
                    
                    response = model.generate_content([f"Actúa como enfermero. Modo: {modo}. Dudas: {notas}. {prompt_contenido}", *contenido])
                    st.markdown(response.text)
                except Exception as e:
                    st.error(f"Error: {e}")
