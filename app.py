import streamlit as st
import google.generativeai as genai
from PIL import Image
import pypdf

# --- CONFIGURACIÓN ---
st.set_page_config(page_title="LabMind Integral", page_icon="🧬", layout="wide")

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
    st.title("LabMind 3.1")
    st.caption("🚀 Gemini 3 Flash + Evolutivo")
    
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

    contexto = st.selectbox("Contexto:", ["Hospitalización", "Urgencias", "UCI", "Domicilio", "Consulta"])

# --- ZONA PRINCIPAL ---
st.title("🩺 Estación de Análisis Clínico Integral")

col1, col2 = st.columns([1.2, 2])

with col1:
    st.subheader("1. Configuración")
    
    # SELECTOR DE MODO AMPLIADO
    modo = st.radio("Tipo de Análisis:", [
        "🩹 Heridas (Evolución Visual)", 
        "📊 Analíticas (Serie Evolutiva)", 
        "🧩 ESTUDIO INTEGRAL (Pruebas + Informes)",
        "📉 ECG / Imagen Única"
    ])
    
    st.markdown("---")
    
    # --- GESTOR DE ARCHIVOS MULTIMODAL ---
    archivos_procesar = [] # Lista maestra de archivos
    
    if modo == "🩹 Heridas (Evolución Visual)":
        st.info("📸 Sube fotos para comparar el antes y después.")
        f_actual = st.file_uploader("1️⃣ FOTO ACTUAL", type=['jpg', 'png', 'jpeg'])
        f_previa = st.file_uploader("2️⃣ FOTO PREVIA (Opcional)", type=['jpg', 'png', 'jpeg'])
        if f_actual: archivos_procesar.append(("img_actual", f_actual))
        if f_previa: archivos_procesar.append(("img_previa", f_previa))

    elif modo == "📊 Analíticas (Serie Evolutiva)":
        st.info("📈 Sube VARIAS analíticas (PDF o Foto) para ver la tendencia.")
        files = st.file_uploader("Sube todos los informes:", type=['pdf', 'jpg', 'png', 'jpeg'], accept_multiple_files=True)
        if files:
            for f in files: archivos_procesar.append(("doc_serie", f))

    elif modo == "🧩 ESTUDIO INTEGRAL (Pruebas + Informes)":
        st.info("🗂️ Sube TODO el caso: Informes, Placas, Analíticas...")
        files = st.file_uploader("Archivos del paciente:", type=['pdf', 'jpg', 'png', 'jpeg'], accept_multiple_files=True)
        if files:
            for f in files: archivos_procesar.append(("mix", f))
            
    else: # Modo simple
        f = st.file_uploader("Sube archivo:", type=['jpg', 'png', 'jpeg', 'pdf'])
        if f: archivos_procesar.append(("unico", f))

    st.markdown("---")
    notas = st.text_area("✍️ Notas / Cronología:", placeholder="Ej: Paciente ingresado hace 3 días. Fiebre persistente...", height=120)

with col2:
    st.subheader("2. Análisis IA (Gemini 3)")
    
    if archivos_procesar and st.button("🚀 ANALIZAR CASO COMPLETO", type="primary"):
        if not api_key:
            st.warning("⚠️ Falta API Key.")
        else:
            with st.spinner("🧠 Procesando múltiples documentos y cruzando datos..."):
                try:
                    genai.configure(api_key=api_key)
                    # MOTOR FIJO GEMINI 3 FLASH
                    model = genai.GenerativeModel("models/gemini-3-flash-preview")
                    
                    # SEGURIDAD OFF
                    safety_settings = [{"category": c, "threshold": "BLOCK_NONE"} for c in ["HARM_CATEGORY_HARASSMENT", "HARM_CATEGORY_HATE_SPEECH", "HARM_CATEGORY_SEXUALLY_EXPLICIT", "HARM_CATEGORY_DANGEROUS_CONTENT"]]
                    
                    # --- PROCESAMIENTO DE ARCHIVOS ---
                    contenido_ia = []
                    contexto_archivos = ""
                    
                    for tipo, archivo in archivos_procesar:
                        nombre = archivo.name
                        
                        if archivo.type == "application/pdf":
                            # Extraer texto de PDF
                            pdf_reader = pypdf.PdfReader(archivo)
                            texto_pdf = ""
                            for page in pdf_reader.pages: texto_pdf += page.extract_text() or ""
                            contexto_archivos += f"\n--- DOCUMENTO ({nombre}) ---\n{texto_pdf}\n"
                        else:
                            # Es imagen
                            img = Image.open(archivo)
                            contenido_ia.append(img)
                            if tipo == "img_actual": contexto_archivos += "\n\n"
                            elif tipo == "img_previa": contexto_archivos += "\n\n"
                            else: contexto_archivos += f"\n[IMAGEN DIAGNÓSTICA: {nombre}]\n"

                    # --- PROMPT MAESTRO INTEGRAL ---
                    full_prompt = f"""
                    Actúa como Enfermera Clínica Especialista (APN) y Gestora de Casos.
                    CONTEXTO: {contexto}. MODO: {modo}.
                    NOTAS USUARIO: "{notas}"

                    ⚠️ PRIVACIDAD: Si detectas nombres reales ("{nombre}"), SUSTITÚYELOS por "Paciente [Edad] [Sexo]".

                    ARCHIVOS ADJUNTOS:
                    {contexto_archivos}

                    {f"USA ESTE PROTOCOLO: {texto_protocolo[:20000]}" if texto_protocolo else "USA EVIDENCIA CIENTÍFICA."}

                    INSTRUCCIONES ESPECÍFICAS SEGÚN MODO:
                    1. **SI ES SERIE ANALÍTICA:** Detecta fechas y comenta la EVOLUCIÓN de los parámetros (¿Mejora o empeora?).
                    2. **SI ES INTEGRAL:** Relaciona los hallazgos de las pruebas (Ej: "La Rx coincide con la analítica").
                    3. **SI ES HERIDA:** Análisis TIME y comparativa visual.

                    ***FORMATO DE SALIDA (2 PARTES)***:
                    Usa "---" para separar.

                    ---
                    ### ⚡ RESUMEN DEL CASO
                    * **👤 PACIENTE:** [Edad/Sexo Anonimizado].
                    * **🚨 PROBLEMA PRINCIPAL:** [Diagnóstico síntesis].
                    * **🔄 TENDENCIA/EVOLUCIÓN:** [Resumen de la progresión].
                    ---
                    
                    ### 📝 ANÁLISIS CLÍNICO PROFUNDO
                    1. **Hallazgos Detallados:** (Valores alterados, descripción visual, etc.).
                    2. **Correlación de Pruebas:** (Cómo encajan las piezas del puzzle).
                    3. **PLAN DE CUIDADOS INTEGRAL:**
                       - Intervenciones prioritarias.
                       - **CITA EVIDENCIA** en cada recomendación.
                    """
                    
                    # Llamada
                    if contenido_ia:
                        response = model.generate_content([full_prompt, *contenido_ia], safety_settings=safety_settings)
                    else:
                        response = model.generate_content(full_prompt, safety_settings=safety_settings)
                    
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
                    if "429" in str(e): st.warning("Gemini 3 saturado. Espera 1 min.")
    
    elif not archivos_procesar and st.button("🚀 ANALIZAR CASO COMPLETO"):
        st.warning("⚠️ Sube al menos un archivo.")
