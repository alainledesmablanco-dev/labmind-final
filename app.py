import streamlit as st
import google.generativeai as genai
from PIL import Image
import pypdf

st.set_page_config(page_title="LabMind Selector", page_icon="🧬", layout="wide")

# --- BARRA LATERAL ---
with st.sidebar:
    st.title("LabMind Diagnóstico")
    
    # 1. METER LA CLAVE
    api_key = st.text_input("1. Pega tu API Key:", type="password")
    
    st.divider()
    
    # 2. BUSCADOR DE MODELOS DISPONIBLES
    modelo_elegido = "models/gemini-1.5-flash" # Por defecto
    
    if api_key:
        try:
            genai.configure(api_key=api_key)
            # Preguntamos a Google qué modelos nos deja usar con tu llave
            lista_modelos = []
            for m in genai.list_models():
                if 'generateContent' in m.supported_generation_methods:
                    lista_modelos.append(m.name)
            
            st.success(f"✅ ¡Conectado! Tu llave permite usar {len(lista_modelos)} modelos.")
            # Creamos el menú desplegable
            modelo_elegido = st.selectbox("2. ELIGE EL CEREBRO:", lista_modelos, index=0)
            st.caption("Si uno falla, prueba el siguiente de la lista.")
            
        except Exception as e:
            st.error("❌ La clave parece incorrecta o no tiene permisos.")
            st.write(e)

# --- CUERPO PRINCIPAL ---
st.title("🩺 Unidad Clínica (Modo Selector)")

col1, col2 = st.columns([1, 2])

with col1:
    modo = st.radio("Modo:", ["🩹 Heridas", "🩸 Analítica", "📈 ECG", "💀 Rx/TAC"])
    archivo = st.file_uploader("Subir Foto:", type=['jpg', 'png', 'jpeg', 'pdf'])
    notas = st.text_area("Dudas / Notas:", height=100)

with col2:
    if archivo and st.button("🚀 ANALIZAR AHORA"):
        if not api_key:
            st.warning("⚠️ Primero pon la API Key en la izquierda.")
        else:
            with st.spinner(f"Analizando con {modelo_elegido}..."):
                try:
                    # Configuración Directa
                    genai.configure(api_key=api_key)
                    model = genai.GenerativeModel(modelo_elegido)
                    
                    # Preparar imagen/pdf
                    contenido = []
                    if archivo.type == "application/pdf":
                        pdf_reader = pypdf.PdfReader(archivo)
                        texto = ""
                        for page in pdf_reader.pages: texto += page.extract_text()
                        prompt_contenido = f"CONTENIDO PDF:\n{texto}"
                    else:
                        contenido.append(Image.open(archivo))
                        prompt_contenido = "Analiza esta imagen."
                    
                    # Prompt
                    full_prompt = f"""
                    Actúa como Enfermero Experto. Modo: {modo}.
                    Notas Usuario: {notas}.
                    
                    {prompt_contenido}
                    
                    TAREA: Dame Diagnóstico, Alertas y Plan de Cuidados.
                    """
                    
                    # Ejecutar
                    response = model.generate_content([full_prompt, *contenido])
                    st.markdown(response.text)
                    st.balloons() # ¡Celebración si funciona!
                    
                except Exception as e:
                    st.error(f"❌ Error con el modelo {modelo_elegido}:")
                    st.code(e)
                    st.info("💡 CONSEJO: Cambia el modelo en la barra lateral (paso 2) y vuelve a darle al botón Analizar.")
