import streamlit as st
import google.generativeai as genai
from PIL import Image
import pypdf
import tempfile
import time
import os
from fpdf import FPDF
import datetime
import re
import matplotlib.pyplot as plt

# --- CONFIGURACIÓN ---
st.set_page_config(page_title="LabMind Ultra", page_icon="🧬", layout="wide")

# --- GESTIÓN DE ESTADO ---
if "resultado_analisis" not in st.session_state:
    st.session_state.resultado_analisis = None
if "datos_grafica" not in st.session_state:
    st.session_state.datos_grafica = None

# --- ESTILOS ---
st.markdown("""
<style>
    .stButton>button { width: 100%; border-radius: 8px; font-weight: bold; background-color: #0066cc; color: white; }
    .esquema-rapido { background-color: #e8f4ff; padding: 15px; border-radius: 10px; border-left: 5px solid #0066cc; margin-bottom: 20px; }
    .segunda-opinion { background-color: #f0fff4; padding: 10px; border-radius: 5px; border-left: 5px solid #2ecc71; font-size: 0.9em; }
    .calculadora-box { background-color: #fff3cd; padding: 10px; border-radius: 5px; margin-bottom: 10px; }
</style>
""", unsafe_allow_html=True)

# --- FUNCIONES AUXILIARES (PDF & GRÁFICAS) ---
def create_pdf(texto_analisis):
    class PDF(FPDF):
        def header(self):
            self.set_font('Arial', 'B', 12)
            self.cell(0, 10, 'LabMind - Informe Clínico Avanzado', 0, 1, 'C')
            self.ln(5)
        def footer(self):
            self.set_y(-15)
            self.set_font('Arial', 'I', 8)
            self.cell(0, 10, f'Pagina {self.page_no()}', 0, 0, 'C')

    pdf = PDF()
    pdf.add_page()
    pdf.set_font("Arial", size=10)
    fecha = datetime.datetime.now().strftime("%d/%m/%Y %H:%M")
    pdf.cell(0, 10, f"Fecha: {fecha}", 0, 1)
    pdf.ln(5)
    texto_limpio = texto_analisis.encode('latin-1', 'replace').decode('latin-1')
    pdf.multi_cell(0, 5, texto_limpio)
    return pdf.output(dest='S').encode('latin-1')

def extraer_datos_grafica(texto):
    """Intenta buscar un bloque JSON/Data oculto en la respuesta de la IA para graficar"""
    # Buscamos patrón tipo: GRÁFICA: {Fecha: Valor, Fecha: Valor}
    match = re.search(r'GRÁFICA_DATA: ({.*?})', texto)
    if match:
        try:
            dict_str = match.group(1)
            # Convertir string a dict de forma segura
            data = eval(dict_str) 
            return data
        except:
            return None
    return None

# --- BARRA LATERAL (CONFIGURACIÓN + CALCULADORA REAL) ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3063/3063176.png", width=80)
    st.title("LabMind 11.0")
    
    st.markdown("### 🔑 Acceso")
    api_key = st.text_input("Pega tu API Key:", type="password")
    
    st.divider()
    
    # --- 🧮 CALCULADORA INTEGRADA (PYTHON REAL) ---
    with st.expander("🧮 Calculadora Renal (Python)", expanded=True):
        st.caption("Cálculo exacto CKD-EPI")
        calc_creatinina = st.number_input("Creatinina (mg/dL)", 0.0, 15.0, 1.0, step=0.1)
        calc_edad = st.number_input("Edad", 18, 120, 60)
        calc_sexo = st.selectbox("Sexo", ["Hombre", "Mujer"])
        
        # Fórmula Matemática Real (CKD-EPI 2021)
        k = 0.7 if calc_sexo == "Mujer" else 0.9
        a = -0.329 if calc_sexo == "Mujer" else -0.411
        factor_sexo = 1.018 if calc_sexo == "Mujer" else 1
        
        try:
            fg_exacto = 141 * (min(calc_creatinina/k, 1)**a) * (max(calc_creatinina/k, 1)**-1.209) * (0.993**calc_edad) * factor_sexo
            st.markdown(f"**FG Estimado:** `{fg_exacto:.1f} mL/min`")
            
            # Semáforo renal
            if fg_exacto > 60: estado_renal = "✅ Función Normal"
            elif fg_exacto > 30: estado_renal = "⚠️ Insuficiencia Moderada"
            else: estado_renal = "🚨 Insuficiencia Grave"
            st.caption(estado_renal)
            
            # Guardamos el dato para pasárselo a la IA
            contexto_calculado = f"DATO CALCULADO POR PYTHON: El Filtrado Glomerular exacto (CKD-EPI) es {fg_exacto:.1f} mL/min/1.73m2 ({estado_renal})."
        except:
            contexto_calculado = ""

    st.divider()
    
    # Activar Segunda Opinión
    modo_agente = st.checkbox("🤖 Activar Segunda Opinión (Doble chequeo)", value=True, help="Tarda el doble, pero un segundo 'Cerebro IA' revisa el diagnóstico.")

    protocolo_pdf = st.file_uploader("📚 Protocolo (PDF)", type="pdf")
    texto_protocolo = ""
    if protocolo_pdf:
        try:
            pdf_reader = pypdf.PdfReader(protocolo_pdf)
            for page in pdf_reader.pages: texto_protocolo += page.extract_text() or ""
            st.success("✅ Protocolo Activo")
        except: st.error("Error PDF")

    contexto = st.selectbox("Contexto:", ["Hospitalización", "Urgencias", "UCI", "Domicilio", "Consulta"])

# --- ZONA PRINCIPAL ---
st.title("🩺 Estación Clínica Inteligente")

col1, col2 = st.columns([1.2, 2])

with col1:
    st.subheader("1. Captura de Datos")
    
    # --- NUEVO MODO FARMACIA ---
    modo = st.radio("Modo:", [
        "🩹 Heridas", 
        "📊 Analíticas (Serie)", 
        "💊 Farmacia/Interacciones",
        "💀 TAC/RMN/ECG", 
        "🧩 Integral"
    ])
    st.markdown("---")
    
    opciones_fuente = ["📁 Subir o Grabar (Móvil)", "📸 Cámara Web (Solo Fotos)"]
    if modo in ["💀 TAC/RMN/ECG", "🧩 Integral"]:
        st.info("💡 Soporte de VÍDEO activo.")
    
    fuente_imagen = st.radio("Método de entrada:", opciones_fuente, horizontal=True)
    
    archivos_procesar = [] 

    # --- INPUTS ---
    if fuente_imagen == "📸 Cámara Web (Solo Fotos)":
        foto_camara = st.camera_input("Hacer foto")
        if foto_camara: archivos_procesar.append(("foto_camara", foto_camara))
    else:
        if modo == "🩹 Heridas":
            st.info("📸 Foto Actual + Previa")
            f_actual = st.file_uploader("FOTO ACTUAL", type=['jpg', 'png', 'jpeg'])
            f_previa = st.file_uploader("FOTO PREVIA", type=['jpg', 'png', 'jpeg'])
            if f_actual: archivos_procesar.append(("img_actual", f_actual))
            if f_previa: archivos_procesar.append(("img_previa", f_previa))

        elif modo == "📊 Analíticas (Serie)":
            st.info("📈 Sube VARIAS analíticas para ver la GRÁFICA evolutiva.")
            files = st.file_uploader("Informes:", type=['pdf', 'jpg', 'png'], accept_multiple_files=True)
            if files:
                for f in files: archivos_procesar.append(("doc", f))
        
        elif modo == "💊 Farmacia/Interacciones":
            st.info("💊 Sube foto de la caja, blister u hoja de tratamiento.")
            f = st.file_uploader("Foto Medicación:", type=['jpg', 'png', 'jpeg', 'pdf'])
            if f: archivos_procesar.append(("farmacia", f))
        
        elif modo == "💀 TAC/RMN/ECG":
            f = st.file_uploader("Imagen o VÍDEO:", type=['jpg', 'png', 'jpeg', 'mp4', 'mov'])
            if f: 
                if f.type in ['video/mp4', 'video/quicktime']: archivos_procesar.append(("video", f))
                else: archivos_procesar.append(("unico", f))

        elif modo == "🧩 Integral":
            st.info("🗂️ Todo mezclado: Informes, Fotos y VÍDEOS.")
            files = st.file_uploader("Archivos del caso:", type=['pdf', 'jpg', 'png', 'jpeg', 'mp4', 'mov'], accept_multiple_files=True)
            if files:
                for f in files:
                    if f.type in ['video/mp4', 'video/quicktime']: archivos_procesar.append(("video", f))
                    else: archivos_procesar.append(("doc_mix", f))

    st.markdown("---")
    
    st.write("🎙️ **Dictado de Voz:**")
    audio_nota = st.audio_input("Grabar notas")
    notas_texto = st.text_area("✍️ Texto:", height=80)

with col2:
    st.subheader("2. Análisis & Validación")
    
    if (archivos_procesar or audio_nota) and st.button("🚀 ANALIZAR (AGENTE CLÍNICO)", type="primary"):
        if not api_key:
            st.warning("⚠️ Falta API Key.")
        else:
            with st.spinner("🧠 Fase 1: Análisis Inicial..."):
                try:
                    genai.configure(api_key=api_key)
                    model = genai.GenerativeModel("models/gemini-3-flash-preview")
                    safety_settings = [{"category": c, "threshold": "BLOCK_NONE"} for c in ["HARM_CATEGORY_HARASSMENT", "HARM_CATEGORY_HATE_SPEECH", "HARM_CATEGORY_SEXUALLY_EXPLICIT", "HARM_CATEGORY_DANGEROUS_CONTENT"]]
                    
                    contenido_ia = []
                    contexto_archivos = ""
                    
                    if audio_nota:
                         contenido_ia.append(genai.upload_file(audio_nota, mime_type="audio/wav"))
                         contexto_archivos += "\n[AUDIO: NOTA DE VOZ ADJUNTA]\n"

                    for tipo, archivo in archivos_procesar:
                        if tipo == "video":
                            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
                                tmp_file.write(archivo.read()); tmp_path = tmp_file.name
                            video_file = genai.upload_file(path=tmp_path)
                            while video_file.state.name == "PROCESSING": time.sleep(1); video_file = genai.get_file(video_file.name)
                            contenido_ia.append(video_file); contexto_archivos += f"\n[VÍDEO: {archivo.name}]\n"; os.remove(tmp_path)
                        elif hasattr(archivo, 'type') and archivo.type == "application/pdf":
                            pdf_reader = pypdf.PdfReader(archivo); texto_pdf = ""
                            for page in pdf_reader.pages: texto_pdf += page.extract_text() or ""
                            contexto_archivos += f"\n--- PDF ---\n{texto_pdf}\n"
                        else:
                            img = Image.open(archivo); contenido_ia.append(img); contexto_archivos += f"\n[IMAGEN ADJUNTA]\n"

                    # --- PROMPT FASE 1: ANÁLISIS ---
                    prompt_fase_1 = f"""
                    Actúa como Experto Clínico Multidisciplinar.
                    CONTEXTO: {contexto}. MODO: {modo}.
                    NOTAS: "{notas_texto}"
                    {contexto_calculado} (Este valor es EXACTO, úsalo).

                    MATERIAL ADJUNTO:
                    {contexto_archivos}
                    {f"PROTOCOLO: {texto_protocolo[:15000]}" if texto_protocolo else "USA EVIDENCIA."}

                    INSTRUCCIONES CLAVE:
                    1. Si es MODO FARMACIA: Cruza la medicación detectada en la foto con las patologías del paciente. Busca interacciones (ej: AINEs + Fallo Renal).
                    2. Si es MODO SERIE: Extrae fechas y valores. Si detectas una evolución clara, GENERA AL FINAL DEL TEXTO EL SIGUIENTE FORMATO EXACTO: GRÁFICA_DATA: {{'Ene': 10, 'Feb': 12, 'Mar': 14}} (Solo con los valores numéricos principales).
                    3. Anonimiza nombres.

                    FORMATO SALIDA:
                    ---
                    ### ⚡ DIAGNÓSTICO
                    * **👤 PACIENTE:** [Datos anonimizados].
                    * **🚨 PROBLEMA:** [Principal].
                    * **💊 ALERTA FARMACIA:** [Solo si hay riesgo].
                    ---
                    ### 📝 ANÁLISIS DETALLADO
                    [Desarrollo completo].
                    """
                    
                    # EJECUCIÓN FASE 1
                    if contenido_ia: response_1 = model.generate_content([prompt_fase_1, *contenido_ia], safety_settings=safety_settings)
                    else: response_1 = model.generate_content(prompt_fase_1, safety_settings=safety_settings)
                    
                    texto_final = response_1.text

                    # --- FASE 2: SEGUNDA OPINIÓN (AGENTE CRÍTICO) ---
                    if modo_agente:
                        with st.spinner("🕵️ Fase 2: Agente Crítico revisando errores..."):
                            prompt_critico = f"""
                            Actúa como Supervisor Clínico Senior. Revisa este informe generado por una IA junior:
                            
                            "{texto_final}"
                            
                            TU TAREA:
                            1. Busca errores clínicos, alucinaciones o incongruencias (ej: ¿Dice que hay edema pero la foto es normal?).
                            2. Si todo está bien, déjalo igual.
                            3. Si hay errores, CORRÍGELOS y reescribe el informe final mejorado.
                            4. Mantén el formato con "---".
                            """
                            response_2 = model.generate_content(prompt_critico, safety_settings=safety_settings)
                            texto_final = response_2.text

                    st.session_state.resultado_analisis = texto_final
                    
                    # Extraer datos gráfica si existen
                    datos_grafica = extraer_datos_grafica(texto_final)
                    st.session_state.datos_grafica = datos_grafica

                except Exception as e:
                    st.error("❌ Error:"); st.write(e)

    # --- MOSTRAR RESULTADOS ---
    if st.session_state.resultado_analisis:
        # 1. MOSTRAR GRÁFICA (Si se detectaron datos)
        if st.session_state.datos_grafica:
            st.caption("📈 Tendencia Detectada Automáticamente")
            data = st.session_state.datos_grafica
            fechas = list(data.keys())
            valores = list(data.values())
            
            fig, ax = plt.subplots(figsize=(6, 2))
            ax.plot(fechas, valores, marker='o', color='red', linestyle='-')
            ax.set_title("Evolución Clínica")
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)

        # 2. TEXTO DEL INFORME
        texto_limpio = st.session_state.resultado_analisis.replace("GRÁFICA_DATA:", "").split("{'")[0] # Limpiar código técnico
        
        partes = texto_limpio.split("---")
        if len(partes) >= 3:
            st.markdown(f'<div class="esquema-rapido">{partes[1]}</div>', unsafe_allow_html=True)
            if modo_agente:
                st.markdown('<div class="segunda-opinion">✅ Validado por Agente Supervisor</div>', unsafe_allow_html=True)
            st.markdown(partes[2])
        else:
            st.markdown(texto_limpio)
            
        st.divider()
        
        # 3. EXPORTAR PDF
        texto_pdf = texto_limpio.replace("*", "").replace("#", "").replace("---", "")
        pdf_bytes = create_pdf(texto_pdf)
        st.download_button("📥 DESCARGAR INFORME PDF", pdf_bytes, "Informe_LabMind.pdf", "application/pdf")
