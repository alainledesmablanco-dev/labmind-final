import streamlit as st
import google.generativeai as genai
from PIL import Image
import pypdf

# --- CONFIGURACIÓN ---
st.set_page_config(page_title="LabMind Integral", page_icon="🧩", layout="wide")

# --- ESTILOS ---
st.markdown("""
<style>
    .stButton>button { width: 100%; border-radius: 8px; font-weight: bold; background-color: #0066cc; color: white; }
    .esquema-rapido { background-color: #e8f4ff; padding: 15px; border-radius: 10px; border-left: 5px solid #0066cc; margin-bottom: 20px; }
    .alerta-seguridad { background-color: #fff0f0; padding: 10px; border-radius: 5px; border-left: 5px solid #ff4444; color: #cc0000; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# --- BARRA LATERAL ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3063/3063176.png", width=80)
    st.title("LabMind 7.0")
    st.caption("🧩 Multimodal + Evolutivo")
    
    api_key = st.text_input("🔑 API Key:", type="password")
    
    st.divider()
    st.write("📚 **Protocolo Unidad**")
    protocolo_pdf = st.file_uploader("Sube Guía/Protocolo (PDF)", type="pdf")
    texto_protocolo = ""
    if protocolo_pdf:
        try:
            pdf_reader = pypdf.PdfReader(protocolo_pdf)
            for page in pdf_reader.pages: texto_protocolo += page.extract_text() or ""
            st.success("✅ Protocolo Activo")
        except: st.error("Error PDF")

    contexto = st.selectbox("Contexto Paciente:", ["Hospitalización", "Urgencias", "UCI", "Primaria", "Consulta Externa"])

# --- ZONA PRINCIPAL ---
st.title("🩺 Estación de Análisis Clínico Integral")

col1, col2 = st.columns([1.2, 2])

with col1:
    st.subheader("1. Selección de Pruebas")
    
    # NUEVOS MODOS
    modo = st.radio("Tipo de Estudio:", [
        "🩹 Heridas (Evolución Foto a Foto)", 
        "🩸 Analíticas (Evolución/Serie)", 
        "💀 Imagen (Rx / TAC / RMN)", 
        "📈 ECG",
        "🧩 ESTUDIO INTEGRAL (Analítica + Imagen + Informes)"
    ])
    
    st.markdown("---")
    
    # --- GESTOR DE ARCHIVOS INTELIGENTE ---
    archivos_subidos = [] # Lista para guardar todo lo que subas
    
    if modo == "🩹 Heridas (Evolución Foto a Foto)":
        st.info("📸 Para comparar, sube Foto Actual y Previa.")
        f_actual = st.file_uploader("1️⃣ FOTO ACTUAL (Obligatoria)", type=['jpg', 'png', 'jpeg'])
        f_previa = st.file_uploader("2️⃣ FOTO PREVIA (Opcional)", type=['jpg', 'png', 'jpeg'])
        if f_actual: archivos_subidos.append(("actual", f_actual))
        if f_previa: archivos_subidos.append(("previa", f_previa))
        
    elif modo == "🩸 Analíticas (Evolución/Serie)":
        st.info("📊 Sube VARIAS analíticas para ver la gráfica de evolución.")
        files = st.file_uploader("Sube todos los PDFs/Fotos de analíticas:", type=['pdf', 'jpg', 'png'], accept_multiple_files=True)
        if files: 
            for f in files: archivos_subidos.append(("doc", f))

    elif modo == "🧩 ESTUDIO INTEGRAL (Analítica + Imagen + Informes)":
        st.info("🗂️ Sube TODO lo que tengas del paciente (PDFs, Placas, ECGs). La IA cruzará los datos.")
        files = st.file_uploader("Sube todo el caso mezclado:", type=['pdf', 'jpg', 'png'], accept_multiple_files=True)
        if files: 
            for f in files: archivos_subidos.append(("mix", f))
            
    else: # Modos simples (Imagen, ECG)
        st.info("Sube la imagen o informe.")
        files = st.file_uploader("Sube archivo:", type=['pdf', 'jpg', 'png'], accept_multiple_files=True)
        if files: 
            for f in files: archivos_subidos.append(("doc", f))

    st.markdown("---")
    notas = st.text_area("✍️ Notas Clínicas / Cronología:", placeholder="Ej: Paciente ingresó ayer por disnea. Adjunto analítica de urgencias y la de planta de hoy + Placa tórax.", height=120)

with col2:
    st.subheader("2. Resultados del Análisis")
    
    if archivos_subidos and st.button("🚀 ANALIZAR CASO COMPLETO", type="primary"):
        if not api_key:
            st.warning("⚠️ Falta API Key.")
        else:
            with st.spinner("🧠 Procesando múltiples archivos, cruzando datos y protegiendo identidad..."):
                try:
                    genai.configure(api_key=api_key)
                    # Usamos Gemini 2.0 Flash (o 1.5 Pro) porque tienen una ventana de contexto GIGANTE para leer muchos PDFs
                    model = genai.GenerativeModel("models/gemini-2.0-flash")
                    
                    # Seguridad OFF para imágenes médicas
                    safety_settings = [{"category": c, "threshold": "BLOCK_NONE"} for c in ["HARM_CATEGORY_HARASSMENT", "HARM_CATEGORY_HATE_SPEECH", "HARM_CATEGORY_SEXUALLY_EXPLICIT", "HARM_CATEGORY_DANGEROUS_CONTENT"]]
                    
                    # --- PROCESADOR MULTIMODAL ---
                    contenido_ia = []
                    descripcion_archivos = ""
                    
                    for tipo, archivo in archivos_subidos:
                        if archivo.type == "application/pdf":
                            # Leer PDF
                            pdf_reader = pypdf.PdfReader(archivo)
                            texto_pdf = ""
                            for page in pdf_reader.pages: texto_pdf += page.extract_text() or ""
                            # Añadir al prompt como texto
                            descripcion_archivos += f"\n--- CONTENIDO DE DOCUMENTO ({archivo.name}) ---\n{texto_pdf}\n"
                        else:
                            # Es imagen
                            img = Image.open(archivo)
                            contenido_ia.append(img)
                            # Si es herida, etiquetamos si es actual o previa
                            if tipo == "actual": descripcion_archivos += "\n[SE ADJUNTA IMAGEN: ESTADO ACTUAL DE LA LESIÓN]\n"
                            elif tipo == "previa": descripcion_archivos += "\n[SE ADJUNTA IMAGEN: ESTADO PREVIO PARA COMPARAR]\n"
                            else: descripcion_archivos += f"\n[SE ADJUNTA IMAGEN DIAGNÓSTICA: {archivo.name}]\n"

                    # --- PROMPT INTEGRAL ---
                    full_prompt = f"""
                    Actúa como Experto Clínico Multidisciplinar (Medicina Interna / Enfermería Avanzada).
                    CONTEXTO: {contexto}. MODO SELECCIONADO: {modo}.
                    NOTAS DEL USUARIO: "{notas}"

                    ⚠️ REGLA DE PRIVACIDAD: NO reveles nombres reales. Usa "Paciente [Edad] [Sexo]".

                    ARCHIVOS ADJUNTOS PARA ANÁLISIS:
                    {descripcion_archivos}

                    {f"USA ESTE PROTOCOLO: {texto_protocolo[:15000]}" if texto_protocolo else "USA EVIDENCIA CIENTÍFICA ACTUALIZADA."}

                    INSTRUCCIONES ESPECÍFICAS:
                    
                    1. **SI HAY MÚLTIPLES ANALÍTICAS (Modo Evolutivo):**
                       - Crea una pequeña tabla o resumen textual de la TENDENCIA de los valores críticos (Ej: "La Creatinina ha empeorado de 1.2 -> 2.4").
                       - Identifica patrones (Ej: "Caída de Hemoglobina compatible con sangrado activo").
                    
                    2. **SI ES ESTUDIO INTEGRAL (Mix de pruebas):**
                       - CORRELACIONA los hallazgos. Ej: "La leucocitosis en la analítica (18.000) coincide con la consolidación en la Rx de tórax".
                    
                    3. **SI ES IMAGEN (Rx/TAC/RMN):**
                       - Describe hallazgos radiológicos clave.
                    
                    ---
                    FORMATO DE SALIDA (Estructurado):
                    
                    ### ⚡ RESUMEN DEL CASO
                    * **👤 Paciente:** [Edad/Sexo Anonimizado]
                    * **🚨 Hallazgo Crítico Principal:** [Lo más urgente].
                    * **📉 Tendencia/Evolución:** [¿Mejora o Empeora?].

                    ### 🔍 ANÁLISIS INTEGRADO
                    [Aquí cruza los datos de las diferentes pruebas. Si hay analíticas seriadas, comenta la evolución de los parámetros alterados].

                    ### 📝 PLAN DE ACTUACIÓN & TRATAMIENTO
                    [Lista de acciones recomendadas, citando evidencia o protocolo].
                    """
                    
                    # Si solo hay texto (PDFs)
                    if not contenido_ia:
                        response = model.generate_content(full_prompt, safety_settings=safety_settings)
                    else:
                        # Si hay imágenes + texto
                        response = model.generate_content([full_prompt, *contenido_ia], safety_settings=safety_settings)
                    
                    # Renderizado
                    texto = response.text
                    partes = texto.split("### ⚡ RESUMEN DEL CASO")
                    
                    if len(partes) > 1:
                        st.markdown(f"### ⚡ RESUMEN DEL CASO {partes[1]}") # Reconstruimos el título
                    else:
                        st.markdown(texto)
                        
                    st.balloons()

                except Exception as e:
                    st.error("❌ Error en el análisis:")
                    st.write(e)
                    if "429" in str(e): st.warning("Mucha carga. Espera unos segundos.")
    
    elif not archivos_subidos and st.button("🚀 ANALIZAR CASO COMPLETO"):
        st.warning("⚠️ Debes subir al menos un archivo.")
