import streamlit as st
import google.generativeai as genai
from PIL import Image, ImageOps
import pypdf
import time
import os
import tempfile
from fpdf import FPDF
import datetime
import re
import cv2
import numpy as np
import extra_streamlit_components as stx
import pandas as pd
import urllib.request
import json
import xml.etree.ElementTree as ET

# --- CONFIGURACIÓN ---
st.set_page_config(page_title="LabMind", page_icon="🧬", layout="wide")

# --- ESTILOS CSS REVISADOS ---
st.markdown("""
<style>
    .block-container { padding-top: 0.5rem !important; padding-bottom: 120px !important; }
    div[data-testid="stSelectbox"] { margin-bottom: -15px !important; }
    .stButton>button { width: 100%; border-radius: 8px; font-weight: bold; margin-top: 10px; }
    button[data-testid="baseButton-primary"] { background-color: #0066cc !important; color: white !important; border: none !important; }
    
    details { padding: 15px; border-radius: 8px; margin-bottom: 10px; }
    summary { cursor: pointer; font-size: 1.05rem; outline: none; font-weight: bold; list-style: none; }
    summary::-webkit-details-marker { display: none; }
    details > summary::before { content: '🔽 '; font-size: 0.9em; }
    details[open] > summary::before { content: '🔼 '; }
    details[open] summary { border-bottom: 1px dashed currentcolor; padding-bottom: 8px; margin-bottom: 8px; }
    details p { margin-top: 5px; margin-bottom: 0; line-height: 1.5; }
    
    details.diagnosis-box { background-color: #e3f2fd; border-left: 6px solid #2196f3; color: #0d47a1; }
    details.action-box { background-color: #ffebee; border-left: 6px solid #f44336; color: #b71c1c; }
    details.material-box { background-color: #e8f5e9; border-left: 6px solid #4caf50; color: #1b5e20; }
    details.radiomics-box { background-color: #f3e5f5; border-left: 6px solid #9c27b0; color: #4a148c; }
    details.pocus-box { background-color: #e0f2f1; border-left: 6px solid #00897b; color: #004d40; }
    details.pubmed-box { background-color: #e8eaf6; border-left: 6px solid #3f51b5; color: #1a237e; }
    
    div[data-testid="stChatInput"] {
        position: fixed !important; bottom: 0px !important; left: 0px !important;
        width: 100% !important; background-color: white !important;
        padding: 10px 20px 25px 20px !important; z-index: 9999 !important;
        box-shadow: 0px -4px 10px rgba(0,0,0,0.1) !important;
    }
</style>
""", unsafe_allow_html=True)

# --- CONFIGURACIÓN SEGURIDAD GOOGLE (ANTIBLOQUEO) ---
MEDICAL_SAFETY_SETTINGS = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"}
]

# --- GESTOR DE ESTADO ---
cookie_manager = stx.CookieManager()
time.sleep(0.1)

for key in ["autenticado", "api_key", "resultado_analisis", "pdf_bytes", "chat_messages", "img_marcada", "video_bytes", "modelos_disponibles", "last_video_path"]:
    if key not in st.session_state: st.session_state[key] = [] if key in ["chat_messages", "modelos_disponibles"] else None
if "autenticado" not in st.session_state or not st.session_state.autenticado:
    st.session_state.autenticado = False
if "pocus_metrics" not in st.session_state: st.session_state.pocus_metrics = {}
if "sam_metrics" not in st.session_state: st.session_state.sam_metrics = {}

cookie_api_key = cookie_manager.get(cookie="labmind_secret_key")
if not st.session_state.autenticado:
    if cookie_api_key:
        st.session_state.api_key = cookie_api_key; st.session_state.autenticado = True; st.rerun()
    else:
        st.title("LabMind Acceso")
        k = st.text_input("API Key:", type="password")
        if st.button("Entrar", type="primary"):
            expires = datetime.datetime.now() + datetime.timedelta(days=30)
            cookie_manager.set("labmind_secret_key", k, expires_at=expires)
            st.session_state.api_key = k; st.session_state.autenticado = True; st.rerun()
        st.stop()

# ==========================================
#      MOTOR MOBILE SAM 1 OPTIMIZADO
# ==========================================
def segmentar_herida_sam_v2(img_pil):
    try:
        from ultralytics import SAM
        if not os.path.exists('mobile_sam.pt'):
            return None, 0
        model_sam = SAM('mobile_sam.pt')
        img_cv = cv2.cvtColor(np.array(img_pil.convert('RGB')), cv2.COLOR_RGB2BGR)
        h, w = img_cv.shape[:2]
        puntos = [[w//2, h//2]]
        results = model_sam.predict(img_cv, points=puntos, labels=[1], verbose=False)
        
        if results and len(results)>0 and results[0].masks is not None:
            mask = results[0].masks.data[0].cpu().numpy()
            mask = cv2.resize(mask, (w, h))
            overlay = img_cv.copy()
            overlay[mask > 0.5] = overlay[mask > 0.5] * 0.5 + np.array([255, 150, 0]) * 0.5 
            contours, _ = cv2.findContours((mask > 0.5).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay, contours, -1, (0, 255, 0), 3)
            return Image.fromarray(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)), int(np.sum(mask > 0.5))
    except Exception as e: print(f"Error SAM: {e}")
    return None, 0

# ==========================================
#      MOTOR POCUS GOD MODE V6
# ==========================================
def procesar_pocus_v6_singularidad(video_path):
    try:
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        ret, frame1 = cap.read()
        if not ret: return None, None, None, {}

        h_orig, w_orig = frame1.shape[:2]
        scale = 320.0 / w_orig 
        new_w, new_h = 320, int(h_orig * scale)
        frame1 = cv2.resize(frame1, (new_w, new_h))
        gray_init = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        
        _, beam_t = cv2.threshold(gray_init, 15, 255, cv2.THRESH_BINARY)
        cnts, _ = cv2.findContours(beam_t, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        roi_mask = np.zeros_like(gray_init)
        if cnts: cv2.drawContours(roi_mask, [max(cnts, key=cv2.contourArea)], -1, 255, -1)
        else: cv2.ellipse(roi_mask, (new_w//2, new_h//2), (int(new_w*0.4), int(new_h*0.4)), 0, 0, 360, 255, -1)

        volumes = []; m_mode = []; frame_count = 0
        best_frame = frame1.copy()
        
        while True:
            ret, f = cap.read()
            if not ret: break
            frame_count += 1
            if frame_count % max(1, int(fps // 15)) != 0: continue
            f = cv2.resize(f, (new_w, new_h))
            g = cv2.medianBlur(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), 5)
            masked = cv2.bitwise_and(g, g, mask=roi_mask)
            _, th = cv2.threshold(masked, 35, 255, cv2.THRESH_BINARY_INV)
            th = cv2.bitwise_and(th, th, mask=roi_mask)
            c_pocus, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if c_pocus:
                best_c = max(c_pocus, key=cv2.contourArea)
                if cv2.contourArea(best_c) > (new_w*new_h*0.03):
                    x, y, wb, hb = cv2.boundingRect(best_c)
                    vol = (cv2.contourArea(best_c)**2)/hb if hb>0 else 0
                    volumes.append(vol)
                    m_mode.append(g[:, x+(wb//2)])
                    if vol >= max(volumes):
                        best_frame = f.copy()
                        cv2.drawContours(best_frame, [best_c], -1, (0, 255, 0), 2)
        
        cap.release()
        metrics = {}
        if len(volumes) > 5:
            v_arr = np.array(volumes)
            edv, esv = np.percentile(v_arr, 95), np.percentile(v_arr, 5)
            metrics['FEVI'] = round(max(10.0, min(((edv-esv)/edv)*100, 85.0)), 1) if edv>0 else "N/A"
        
        m_mode_pil = None
        if m_mode:
            mm_img = cv2.resize(np.transpose(np.array(m_mode)), (new_w, new_h))
            m_mode_pil = Image.fromarray(cv2.applyColorMap(mm_img, cv2.COLORMAP_OCEAN))
            
        return m_mode_pil, Image.fromarray(cv2.cvtColor(best_frame, cv2.COLOR_BGR2RGB)), metrics
    except: return None, None, {}

# ==========================================
#      NUEVO: EXTRACTOR DE FOTOGRAMAS (VÍDEO)
# ==========================================
def extraer_frame_video(video_path, texto):
    """Busca FRAME: [X.X] en el texto, extrae ese fotograma del vídeo y lo devuelve."""
    patron_frame = r'FRAME:\s*\[([\d\.]+)\]'
    match = re.search(patron_frame, texto)
    if not match or not video_path: return None, texto
    
    try:
        segundo = float(match.group(1))
        cap = cv2.VideoCapture(video_path)
        # Rebobina el vídeo al milisegundo exacto
        cap.set(cv2.CAP_PROP_POS_MSEC, segundo * 1000)
        ret, frame = cap.read()
        cap.release()
        
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(frame_rgb)
            texto_limpio = re.sub(patron_frame, '', texto).strip()
            return img_pil, texto_limpio
    except Exception as e: print(f"Error Extrayendo Frame: {e}")
    return None, texto

# ==========================================
#      FUNCIONES IA Y AUXILIARES
# ==========================================
def buscar_en_pubmed(query, max_results=4):
    try:
        base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
        search_url = f"{base_url}esearch.fcgi?db=pubmed&term={urllib.parse.quote(query)}&retmode=json&retmax={max_results}&sort=relevance"
        req = urllib.request.Request(search_url, headers={'User-Agent': 'Mozilla/5.0'})
        data = json.loads(urllib.request.urlopen(req).read().decode('utf-8'))
        ids = data.get('esearchresult', {}).get('idlist', [])
        if not ids: return "No se encontraron artículos en PubMed."
        
        fetch_url = f"{base_url}efetch.fcgi?db=pubmed&id={','.join(ids)}&retmode=xml"
        root = ET.fromstring(urllib.request.urlopen(urllib.request.Request(fetch_url, headers={'User-Agent': 'Mozilla/5.0'})).read())
        
        resultados = ""
        for article in root.findall('.//PubmedArticle'):
            pmid = article.find('.//PMID').text
            title = article.find('.//ArticleTitle').text if article.find('.//ArticleTitle') is not None else "Sin título"
            abs_node = article.find('.//AbstractText')
            abstract = "".join(abs_node.itertext()) if abs_node is not None else "Sin abstract."
            resultados += f"PMID: {pmid}\nTÍTULO: {title}\nABSTRACT: {abstract}\n\n"
        return resultados
    except Exception as e: return f"Error PubMed: {e}"

def extraer_y_dibujar_bboxes(texto, img_pil):
    patron = r'BBOX:\s*\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]\s*LABEL:\s*([^\n<]+)'
    matches = re.findall(patron, texto)
    if not matches: return None, texto, False
    
    matches_unicos = list({m[:4]: m for m in matches}.values())
    img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    h, w = img_cv.shape[:2]
    
    for m in matches_unicos:
        try:
            y1, x1, y2, x2 = [int(int(c)*dim/1000) for c, dim in zip(m[:4], [h, w, h, w])]
            cv2.rectangle(img_cv, (x1, y1), (x2, y2), (0, 0, 255), 3)
            cv2.putText(img_cv, m[4].upper(), (x1, max(0, y1-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
        except: pass
        
    texto_limpio = re.sub(patron, '', texto).strip()
    return Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)), texto_limpio, True

def aislar_trazado_ecg(pil_image):
    try:
        img_cv = cv2.cvtColor(np.array(pil_image.convert('RGB')), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 15)
        smooth = cv2.GaussianBlur(thresh, (3, 3), 0)
        return Image.fromarray(cv2.cvtColor(smooth, cv2.COLOR_GRAY2RGB))
    except: return pil_image

def create_pdf(texto):
    class PDF(FPDF):
        def header(self): 
            self.set_font('helvetica', 'B', 12)
            self.cell(0, 10, 'LabMind - Informe IA', align='C', new_x="LMARGIN", new_y="NEXT")
            self.ln(5)
        def footer(self): 
            self.set_y(-15); self.set_font('helvetica', 'I', 8)
            self.cell(0, 10, f'Pag {self.page_no()}', align='C')
            
    pdf = PDF(); pdf.add_page(); pdf.set_font("helvetica", size=10)
    pdf.cell(0, 10, f"Fecha: {datetime.datetime.now().strftime('%d/%m/%Y %H:%M')}", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(5)
    clean = re.sub(r'<[^>]+>', '', texto).replace('€','EUR').replace('’',"'").replace('“','"').replace('”','"')
    pdf.multi_cell(0, 5, txt="".join(c for c in clean if ord(c) < 256))
    return bytes(pdf.output())

# --- INTERFAZ PRINCIPAL ---
st.title("🩺 LabMind")
col_l, col_c, col_r = st.columns([1, 2, 1])

with col_l:
    st.subheader("⚙️ Motor IA")
    try:
        genai.configure(api_key=st.session_state.api_key)
        if not st.session_state.modelos_disponibles:
            st.session_state.modelos_disponibles = [m.name.replace('models/','') for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
    except: st.session_state.modelos_disponibles = ["gemini-1.5-flash"]
    
    idx_defecto = 0
    if st.session_state.modelos_disponibles:
        for i, modelo in enumerate(st.session_state.modelos_disponibles):
            if "gemini-3" in modelo.lower() and "flash" in modelo.lower():
                idx_defecto = i
                break
                
    st.session_state.modelo_seleccionado = st.selectbox("Versión de Gemini:", st.session_state.modelos_disponibles, index=idx_defecto)
    st.session_state.punto_cuerpo = st.selectbox("Anatomía:", ["✨ Autodetectar", "Cara", "Pecho", "Abdomen", "Sacro", "Pierna", "Pie"])

with col_c:
    st.subheader("1. Selección de Modo")
    modo = st.selectbox("Especialidad:", ["✨ Autodetectar", "🩹 Heridas / Úlceras", "🦇 Ecografía / POCUS", "📚 Agente Investigador (PubMed)", "📈 ECG", "💀 RX/TAC", "🧴 Dermatología"])
    contexto = st.selectbox("🏥 Contexto:", ["Urgencias", "Hospitalización", "UCI", "Residencia", "Domicilio"], index=1)
    
    archivos = []; audio_val = None
    notas = ""
    
    if modo == "📚 Agente Investigador (PubMed)":
        st.info("🤖 **Agente Activo:** Conectado a PubMed.")
        query_pubmed = st.text_input("🔍 Duda clínica a investigar:")
        with st.expander("📝 Notas Clínicas / Contexto", expanded=False):
            notas = st.text_area("Contexto:", height=70, label_visibility="collapsed")
    else:
        fs = st.file_uploader("Archivos Clínicos:", type=['jpg','png','pdf','mp4','mov'], accept_multiple_files=True)
        if fs:
            for f in fs:
                if f.type.startswith('video') or f.name.lower().endswith(('mp4', 'mov')): archivos.append(("video", f))
                elif "pdf" in f.type: archivos.append(("doc", f))
                else: archivos.append(("img", f))
                
        with st.expander("📝 Notas Clínicas / Preguntas Específicas", expanded=False):
            notas = st.text_area("Notas", height=70, placeholder="Escribe el contexto del paciente...", label_visibility="collapsed")
            
        with st.expander("🎙️ Adjuntar Nota de Voz", expanded=False):
            if hasattr(st, "audio_input"):
                audio_val = st.audio_input("Dictar notas", key="mic", label_visibility="collapsed")
            else:
                st.warning("⚠️ Tu servidor necesita actualizar Streamlit para usar la nueva grabadora de voz.")
    
    c1, c2 = st.columns([3, 1])
    with c1: 
        btn_analizar = st.button("🚀 ANALIZAR", type="primary", use_container_width=True)
    with c2:
        if st.button("🔄 NUEVO", type="secondary", use_container_width=True):
            for k in ["resultado_analisis", "img_marcada", "video_bytes", "chat_messages", "last_video_path"]: 
                st.session_state[k] = None if ("img" in k or "video" in k or "resultado" in k) else []
            st.session_state.pocus_metrics = {}
            st.session_state.sam_metrics = {}
            st.rerun()
    
    if btn_analizar:
        st.session_state.resultado_analisis = None
        st.session_state.img_marcada = None
        st.session_state.video_bytes = None
        st.session_state.last_video_path = None
        st.session_state.pocus_metrics = {}
        st.session_state.sam_metrics = {}
        st.session_state.chat_messages = []
        
        with st.spinner("Procesando datos (Razonamiento en Cadena activado)..."):
            try:
                model = genai.GenerativeModel(st.session_state.modelo_seleccionado)
                con = []
                txt_docs = ""
                sam_utilizado = False
                imagen_para_visor = None
                video_presente = False
                
                if modo == "📚 Agente Investigador (PubMed)" and query_pubmed:
                    txt_docs += buscar_en_pubmed(query_pubmed)
                    
                if audio_val:
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tf_audio:
                        tf_audio.write(audio_val.read())
                    con.append(genai.upload_file(path=tf_audio.name))
                
                for tipo, f in archivos:
                    if tipo == "video":
                        video_presente = True
                        st.session_state.video_bytes = f.read()
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tf:
                            tf.write(st.session_state.video_bytes)
                            vp = tf.name
                            st.session_state.last_video_path = vp
                            
                        if "POCUS" in modo or modo == "✨ Autodetectar":
                            st.toast("Activando God Mode POCUS...")
                            mm, eb, met = procesar_pocus_v6_singularidad(vp)
                            st.session_state.pocus_metrics = met
                            # Guardamos la imagen de POCUS pero no pisamos si luego Gemini saca otra
                            if not st.session_state.img_marcada:
                                st.session_state.img_marcada = mm if mm else eb
                            txt_docs += f"\n[POCUS] FEVI Estimada: {met.get('FEVI','N/A')}%\n"
                            
                        v_file = genai.upload_file(vp)
                        while v_file.state.name == "PROCESSING": time.sleep(1)
                        con.append(v_file)
                    elif tipo == "doc":
                        txt_docs += "".join([p.extract_text() for p in pypdf.PdfReader(f).pages])
                    elif tipo == "img":
                        img = ImageOps.exif_transpose(Image.open(f)).convert("RGB")
                        
                        if modo in ["🩹 Heridas / Úlceras", "🧴 Dermatología"]:
                            sam_res, a_px = segmentar_herida_sam_v2(img)
                            if a_px > 100: 
                                st.session_state.img_marcada = sam_res
                                st.session_state.sam_metrics = {'Area_PX': a_px}
                                sam_utilizado = True
                                txt_docs += f"\n[SAM] Área de lesión aislada: {a_px} píxeles.\n"
                        
                        if "ECG" in modo:
                            img_aislada = aislar_trazado_ecg(img)
                            con.extend([img, img_aislada])
                        else:
                            con.append(img)
                        if not imagen_para_visor: imagen_para_visor = img

                # --- PROMPT DINÁMICO (VÍDEO VS IMAGEN) ---
                if sam_utilizado:
                    instruccion_bbox = "La imagen ya ha sido segmentada milimétricamente. NO devuelvas BBOX."
                else:
                    if video_presente:
                        instruccion_bbox = "INSTRUCCIÓN DE RADIOLOGÍA DINÁMICA: Estás analizando un VÍDEO. Si detectas patología, busca el fotograma donde se vea más clara. Al final de tu texto, imprime UNA SOLA VEZ el segundo exacto y sus coordenadas así: FRAME: [segundos] BBOX: [ymin, xmin, ymax, xmax] LABEL: TuTexto. (Ejemplo: FRAME: [14.5] BBOX: [150, 200, 450, 600] LABEL: Tumor)."
                    else:
                        instruccion_bbox = "INSTRUCCIÓN DE ANCLAJE ESPACIAL: Si detectas patología, usa tu anclaje visual nativo para marcarla. Imprime esto UNA SOLA VEZ al final: BBOX: [ymin, xmin, ymax, xmax] LABEL: TuTexto."

                if st.session_state.punto_cuerpo == "✨ Autodetectar":
                    instruccion_anatomia = "Deduce visualmente la zona anatómica."
                else:
                    instruccion_anatomia = f"El usuario especifica que la zona es: {st.session_state.punto_cuerpo}. Basa tu análisis en ello."

                prompt = f"""
                Rol: Especialista Senior en Diagnóstico por Imagen, Medicina de Precisión y Cuidados de Enfermería.
                Contexto: {contexto}. Especialidad: {modo}.
                Usuario (Notas): "{notas}"
                Datos: {txt_docs[:5000]}

                RAZONAMIENTO EN CADENA:
                1. EXAMEN VISUAL: Describe brevemente qué ves en la prueba aportada.
                2. IDENTIFICACIÓN DE HALLAZGOS: Busca signos patológicos.
                3. JUICIO CLÍNICO: Emite el diagnóstico basado estrictamente en la evidencia.
                
                REGLAS EXTRA:
                - {instruccion_bbox}
                - {instruccion_anatomia}

                FORMATO HTML REQUERIDO:
                <details class="diagnosis-box" open><summary>🚨 HALLAZGOS Y RAZONAMIENTO</summary><p><b>[Diagnóstico]</b>. [Tu análisis]</p></details>
                <details class="action-box" open><summary>⚡ ACCIÓN INMEDIATA</summary><p>[Plan médico]</p></details>
                <details class="pocus-box" open><summary>👩‍⚕️ CUIDADOS DE ENFERMERÍA</summary><p>[Plan de cuidados específicos]</p></details>
                <details class="{'pubmed-box' if 'PubMed' in modo else 'material-box'}" open><summary>🛠️ TRATAMIENTO Y SEGUIMIENTO</summary><p>[Desarrollo de tratamiento a largo plazo]</p></details>
                """
                
                res = model.generate_content([prompt, *con] if con else prompt, safety_settings=MEDICAL_SAFETY_SETTINGS)
                raw_txt = res.text[res.text.find("<details"):] if "<details" in res.text else res.text
                
                # --- LÓGICA DE DIBUJO Y EXTRACCIÓN (VIAJE EN EL TIEMPO) ---
                img_base_para_bbox = imagen_para_visor
                
                # Si hay video y la IA nos dio el FRAME
                if video_presente and st.session_state.get("last_video_path") and "FRAME:" in raw_txt:
                    st.toast("🎞️ Buscando el fotograma exacto en el vídeo...")
                    frame_extraido, raw_txt = extraer_frame_video(st.session_state.last_video_path, raw_txt)
                    if frame_extraido:
                        img_base_para_bbox = frame_extraido
                
                # Aplicamos las cajas rojas sobre la imagen estática o sobre el fotograma extraído
                if img_base_para_bbox and not sam_utilizado:
                    im_m, clean_t, det = extraer_y_dibujar_bboxes(raw_txt, img_base_para_bbox)
                    if det:
                        st.session_state.img_marcada = im_m
                    elif not st.session_state.img_marcada: # Por si POCUS ya puso una
                        st.session_state.img_marcada = img_base_para_bbox
                    raw_txt = clean_t

                st.session_state.resultado_analisis = raw_txt
                st.session_state.pdf_bytes = create_pdf(st.session_state.resultado_analisis)
            except Exception as e: 
                st.error(f"Error procesando el análisis: {e}")

    if st.session_state.resultado_analisis:
        st.markdown(st.session_state.resultado_analisis, unsafe_allow_html=True)
        if st.session_state.pdf_bytes: st.download_button("📥 Descargar PDF", st.session_state.pdf_bytes, "informe.pdf")

with col_r:
    abierto = True if (st.session_state.get("img_marcada") or st.session_state.get("video_bytes")) else False
    with st.expander("👁️ Dashboard Visual", expanded=abierto):
        if st.session_state.pocus_metrics:
            st.markdown("### 🧮 Telemetría POCUS")
            st.metric("FEVI (Proxy Vol)", f"{st.session_state.pocus_metrics.get('FEVI', 'N/A')}%")
            st.divider()
        if st.session_state.sam_metrics:
            st.markdown("### 🦠 Planimetría SAM")
            st.metric("Área de Lesión", f"{st.session_state.sam_metrics.get('Area_PX', 0)} px")
            st.divider()
            
        if st.session_state.get("video_bytes"): 
            st.video(st.session_state.video_bytes)
        if st.session_state.get("img_marcada"):
            st.markdown("#### 🎯 Fotograma / Visión IA")
            st.image(st.session_state.img_marcada, use_container_width=True)

# ==========================================
# --- CHAT FLOTANTE ---
# ==========================================
if st.session_state.resultado_analisis:
    if query := st.chat_input("Duda clínica sobre este paciente..."):
        st.session_state.chat_messages.append({"role":"user","content":query})
        try:
            chat_model = genai.GenerativeModel(st.session_state.modelo_seleccionado)
            ctx_chat = f"Informe clínico previo:\n{st.session_state.resultado_analisis}\nResponde a esta duda del médico/enfermero: {query}"
            resp = chat_model.generate_content(ctx_chat, safety_settings=MEDICAL_SAFETY_SETTINGS)
            st.session_state.chat_messages.append({"role":"assistant","content":resp.text})
        except Exception as e:
            st.session_state.chat_messages.append({"role":"assistant","content":f"Error: {e}"})
        st.rerun()
