import streamlit as st
import google.generativeai as genai
from PIL import Image
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
import uuid
from streamlit_drawable_canvas import st_canvas
import urllib.request
import json
import xml.etree.ElementTree as ET

# --- CONFIGURACIÓN ---
st.set_page_config(page_title="LabMind 94.1", page_icon="🧬", layout="wide")

# --- ESTILOS CSS ---
st.markdown("""
<style>
    .block-container { padding-top: 0.5rem !important; padding-bottom: 2rem !important; }
    div[data-testid="stVerticalBlock"] > div { gap: 0rem !important; }
    div[data-testid="stSelectbox"] { margin-bottom: -15px !important; }
    
    .stButton>button { width: 100%; border-radius: 8px; font-weight: bold; margin-top: 10px; }
    button[data-testid="baseButton-primary"] { background-color: #0066cc !important; color: white !important; border: none !important; }
    div.element-container:has(.btn-nuevo-hook) + div.element-container button {
        background-color: #2e7d32 !important; color: white !important; border: none !important;
    }
    
    .cot-box { background-color: #f8f9fa; border: 1px dashed #6c757d; border-left: 5px solid #343a40; padding: 12px; border-radius: 5px; margin-bottom: 15px; font-family: monospace; font-size: 0.85rem; color: #495057; }
    .diagnosis-box { background-color: #e3f2fd; border-left: 6px solid #2196f3; padding: 15px; border-radius: 8px; margin-bottom: 10px; color: #0d47a1; font-family: sans-serif; }
    .action-box { background-color: #ffebee; border-left: 6px solid #f44336; padding: 15px; border-radius: 8px; margin-bottom: 10px; color: #b71c1c; font-family: sans-serif; }
    .material-box { background-color: #e8f5e9; border-left: 6px solid #4caf50; padding: 15px; border-radius: 8px; margin-bottom: 15px; color: #1b5e20; font-family: sans-serif; }
    .radiomics-box { background-color: #f3e5f5; border-left: 6px solid #9c27b0; padding: 15px; border-radius: 8px; margin-bottom: 10px; color: #4a148c; font-family: sans-serif; }
    .pocus-box { background-color: #e0f2f1; border-left: 6px solid #00897b; padding: 15px; border-radius: 8px; margin-bottom: 10px; color: #004d40; font-family: sans-serif; }
    .longevity-box { background-color: #fff8e1; border-left: 6px solid #ffc107; padding: 15px; border-radius: 8px; margin-bottom: 10px; color: #ff6f00; font-family: sans-serif; }
    .pubmed-box { background-color: #e8eaf6; border-left: 6px solid #3f51b5; padding: 15px; border-radius: 8px; margin-bottom: 10px; color: #1a237e; font-family: sans-serif; }
    
    .tissue-labels { display: flex; width: 100%; margin-bottom: 2px; }
    .tissue-label-text { font-size: 0.75rem; text-align: center; font-weight: bold; color: #555; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
    .tissue-bar-container { display: flex; width: 100%; height: 20px; border-radius: 10px; overflow: hidden; margin-bottom: 15px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
    .tissue-gran { background-color: #ef5350; height: 100%; }
    .tissue-slough { background-color: #fdd835; height: 100%; }
    .tissue-nec { background-color: #212121; height: 100%; }
    
    .pull-up { margin-top: -25px !important; margin-bottom: 5px !important; height: 1px !important; display: block !important; }
</style>
""", unsafe_allow_html=True)

# --- GESTOR DE ESTADO ---
cookie_manager = stx.CookieManager()
time.sleep(0.1)

if "autenticado" not in st.session_state: st.session_state.autenticado = False
if "api_key" not in st.session_state: st.session_state.api_key = ""
if "resultado_analisis" not in st.session_state: st.session_state.resultado_analisis = None
if "pdf_bytes" not in st.session_state: st.session_state.pdf_bytes = None
if "historial_evolucion" not in st.session_state: st.session_state.historial_evolucion = []
if "area_herida" not in st.session_state: st.session_state.area_herida = 0.0
if "chat_messages" not in st.session_state: st.session_state.chat_messages = []
if "history_db" not in st.session_state: st.session_state.history_db = []
if "img_previo" not in st.session_state: st.session_state.img_previo = None 
if "img_actual" not in st.session_state: st.session_state.img_actual = None 
if "img_ghost" not in st.session_state: st.session_state.img_ghost = None   
if "img_marcada" not in st.session_state: st.session_state.img_marcada = None 
if "last_cv_data" not in st.session_state: st.session_state.last_cv_data = None 
if "last_biofilm_detected" not in st.session_state: st.session_state.last_biofilm_detected = False
if "video_bytes" not in st.session_state: st.session_state.video_bytes = None 
if "modelos_disponibles" not in st.session_state: st.session_state.modelos_disponibles = []
if "pocus_metrics" not in st.session_state: st.session_state.pocus_metrics = {}

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
#      FUNCIONES CLÍNICAS & PUBMED
# ==========================================

def buscar_en_pubmed(query, max_results=4):
    try:
        base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
        search_url = f"{base_url}esearch.fcgi?db=pubmed&term={urllib.parse.quote(query)}&retmode=json&retmax={max_results}&sort=relevance"
        req = urllib.request.Request(search_url, headers={'User-Agent': 'Mozilla/5.0'})
        response = urllib.request.urlopen(req)
        data = json.loads(response.read().decode('utf-8'))
        ids = data.get('esearchresult', {}).get('idlist', [])
        
        if not ids: return "No se encontraron artículos recientes en PubMed para esta consulta."
        
        fetch_url = f"{base_url}efetch.fcgi?db=pubmed&id={','.join(ids)}&retmode=xml"
        fetch_req = urllib.request.Request(fetch_url, headers={'User-Agent': 'Mozilla/5.0'})
        fetch_resp = urllib.request.urlopen(fetch_req).read()
        root = ET.fromstring(fetch_resp)
        
        resultados = ""
        for article in root.findall('.//PubmedArticle'):
            pmid = article.find('.//PMID').text
            title_node = article.find('.//ArticleTitle')
            title = title_node.text if title_node is not None else "Sin título"
            abstract_node = article.find('.//AbstractText')
            abstract = "".join(abstract_node.itertext()) if abstract_node is not None else "Sin abstract disponible."
            
            resultados += f"PMID: {pmid}\nTÍTULO: {title}\nABSTRACT: {abstract}\n\n"
        return resultados
    except Exception as e:
        return f"Error accediendo a la base de datos de PubMed: {e}"

def procesar_pocus_nivel10(video_path):
    return None, None, None, None, {}

def aislar_trazado_ecg(pil_image):
    try:
        img_cv = cv2.cvtColor(np.array(pil_image.convert('RGB')), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 15)
        smooth = cv2.GaussianBlur(thresh, (3, 3), 0)
        return Image.fromarray(cv2.cvtColor(smooth, cv2.COLOR_GRAY2RGB))
    except: return pil_image

def extraer_y_dibujar_bboxes(texto, img_pil=None):
    patron = r'BBOX:\s*\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]\s*LABEL:\s*([^\n<]+)'
    matches = re.findall(patron, texto)
    if not matches or img_pil is None: return None, texto, False
    
    img_cv = cv2.cvtColor(np.array(img_pil.convert('RGB')), cv2.COLOR_RGB2BGR)
    h, w = img_cv.shape[:2]
    
    for match in matches:
        ymin, xmin, ymax, xmax, label = match
        try:
            x1 = max(0, min(w, int(int(xmin) * w / 1000)))
            y1 = max(0, min(h, int(int(ymin) * h / 1000)))
            x2 = max(0, min(w, int(int(xmax) * w / 1000)))
            y2 = max(0, min(h, int(int(ymax) * h / 1000)))
            
            cv2.rectangle(img_cv, (x1, y1), (x2, y2), (0, 0, 255), 4) 
            texto_label = label.strip().upper()
            escala_fuente = max(0.6, w/1000)
            grosor_fuente = max(2, int(w/500))
            (w_txt, h_txt), _ = cv2.getTextSize(texto_label, cv2.FONT_HERSHEY_SIMPLEX, escala_fuente, grosor_fuente)
            cv2.rectangle(img_cv, (x1, max(0, y1-h_txt-10)), (x1 + w_txt, max(0, y1)), (0,0,0), -1)
            cv2.putText(img_cv, texto_label, (x1, max(0, y1-5)), cv2.FONT_HERSHEY_SIMPLEX, escala_fuente, (255, 255, 255), grosor_fuente, cv2.LINE_AA)
        except: pass

    texto_limpio = re.sub(patron, '', texto)
    return Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)), texto_limpio.strip(), True

def aislar_herida_nucleo(img_bgr):
    try:
        from ultralytics import SAM
        h, w = img_bgr.shape[:2]
        max_dim = 640
        scale = min(max_dim/w, max_dim/h)
        if scale < 1:
            new_w, new_h = int(w * scale), int(h * scale)
            img_resized = cv2.resize(img_bgr, (new_w, new_h))
        else:
            img_resized = img_bgr
            new_w, new_h = w, h

        model = SAM("mobile_sam.pt")
        resultados = model(img_resized, points=[[new_w // 2, new_h // 2]], labels=[1], device="cpu", verbose=False)
        
        if resultados and len(resultados) > 0 and resultados[0].masks is not None:
            mask_small = resultados[0].masks.data[0].cpu().numpy()
            mask_small = (mask_small * 255).astype(np.uint8)
            mask_full = cv2.resize(mask_small, (w, h), interpolation=cv2.INTER_NEAREST)
            return mask_full, "DL (MobileSAM)"
        else: return None, "CV2 (Mascara vacia)"
    except Exception as e:
        return None, "CV2 (Fallback RAM)"

def create_pdf(texto_analisis):
    class PDF(FPDF):
        def header(self): 
            self.set_font('helvetica', 'B', 12)
            self.cell(0, 10, 'LabMind - Informe IA', align='C', new_x="LMARGIN", new_y="NEXT")
            self.ln(5)
        def footer(self): 
            self.set_y(-15)
            self.set_font('helvetica', 'I', 8)
            self.cell(0, 10, f'Pag {self.page_no()}', align='C')
            
    pdf = PDF()
    pdf.add_page()
    pdf.set_font("helvetica", size=10)
    pdf.cell(0, 10, f"Fecha: {datetime.datetime.now().strftime('%d/%m/%Y %H:%M')}", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(5)
    
    clean = re.sub(r'<[^>]+>', '', texto_analisis).replace('€','EUR').replace('’',"'").replace('“','"').replace('”','"')
    clean = "".join(c for c in clean if ord(c) < 256)
    pdf.multi_cell(0, 5, txt=clean)
    return bytes(pdf.output())

# ==========================================
#      INTERFAZ DE USUARIO
# ==========================================

st.title("🩺 LabMind 94.1")
col_left, col_center, col_right = st.columns([1, 2, 1])

with col_left:
    st.subheader("⚙️ Motor de IA")
    if st.session_state.autenticado and not st.session_state.modelos_disponibles:
        try:
            genai.configure(api_key=st.session_state.api_key)
            modelos_encontrados = [m.name.replace('models/', '') for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
            st.session_state.modelos_disponibles = sorted(modelos_encontrados, reverse=True) if modelos_encontrados else ["gemini-1.5-pro"]
        except: st.session_state.modelos_disponibles = ["gemini-1.5-pro"]

    lista_para_mostrar = st.session_state.modelos_disponibles if st.session_state.modelos_disponibles else ["Inicia sesión..."]
    st.session_state.modelo_seleccionado = st.selectbox("Versión de Gemini:", lista_para_mostrar)
    st.session_state.punto_cuerpo = st.selectbox("Zona anatómica:", ["No especificada", "Cara", "Pecho", "Abdomen", "Sacro/Glúteo", "Pierna", "Talón", "Pie"])

with col_center:
    st.subheader("1. Selección de Modo")
    modo = st.selectbox("Especialidad:", 
                 ["🩹 Heridas / Úlceras", "📚 Agente Investigador (PubMed)", "📈 ECG", "💀 RX/TAC/Resonancia", "🧴 Dermatología", "🩸 Analítica Funcional", "🧩 Integral"])
    contexto = st.selectbox("🏥 Contexto:", ["Hospitalización", "Residencia", "Urgencias", "UCI", "Domicilio"])
    
    archivos = []; con_final_dibujo = []
    
    if modo == "📚 Agente Investigador (PubMed)":
        st.info("🤖 **Agente Activo:** Conectado a la API oficial de la Biblioteca Nacional de Medicina de EE. UU. Buscará literatura científica real y generará un informe citado.")
        query_pubmed = st.text_input("🔍 ¿Qué duda clínica quieres investigar en PubMed?", placeholder="Ej: Effectiveness of honey dressings for diabetic foot ulcers")
        notas = st.text_area("Contexto de tu paciente (opcional):", placeholder="Paciente de 70 años con DM2...")
    
    else:
        fs = st.file_uploader("Subir Archivos Clínicos", type=['jpg','png','pdf'], accept_multiple_files=True)
        if fs:
            for f in fs:
                if "pdf" in f.type: archivos.append(("doc", f))
                else: archivos.append(("img", f))
        
        # --- PIZARRA INTERACTIVA ---
        imagen_dibujada = None
        if archivos and any(t == "img" for t, _ in archivos):
            img_file = next(f for t, f in archivos if t == "img")
            img_file.seek(0) # REBOBINADO DE SEGURIDAD
            img_pil_original = Image.open(img_file).convert("RGB")
            
            habilitar_dibujo = st.checkbox("✏️ Dibujar / Señalar en la foto para la IA")
            if habilitar_dibujo:
                st.caption("Usa el ratón o el dedo para marcar la anomalía.")
                w_canvas = 500
                h_canvas = int(500 * img_pil_original.height / img_pil_original.width)
                
                canvas_result = st_canvas(
                    fill_color="rgba(255, 0, 0, 0.0)", 
                    stroke_width=4, stroke_color="#00FF00", 
                    background_image=img_pil_original, 
                    update_streamlit=True, height=h_canvas, width=w_canvas, 
                    drawing_mode="freedraw", key="canvas"
                )
                
                if canvas_result.image_data is not None and np.sum(canvas_result.image_data) > 0:
                    mask = Image.fromarray(canvas_result.image_data.astype('uint8'), 'RGBA')
                    mask = mask.resize(img_pil_original.size)
                    imagen_dibujada = Image.alpha_composite(img_pil_original.convert('RGBA'), mask).convert('RGB')
                    st.success("✅ Dibujo fusionado con la imagen. La IA lo verá.")
            
        notas = st.text_area("Notas / Preguntas específicas:", height=60, placeholder="Ej: Analiza la zona que he rodeado en verde...")

    col_btn1, col_btn2 = st.columns([3, 1])
    with col_btn1: btn_analizar = st.button("🚀 ANALIZAR / INVESTIGAR", type="primary", use_container_width=True)
    with col_btn2:
        if st.button("🔄 NUEVO", type="secondary", use_container_width=True):
            st.session_state.resultado_analisis = None; st.session_state.img_marcada = None; st.rerun()

    if btn_analizar:
        st.session_state.resultado_analisis = None
        with st.spinner("🧠 Computando y consultando bases de datos..."):
            try:
                genai.configure(api_key=st.session_state.api_key)
                model = genai.GenerativeModel(st.session_state.modelo_seleccionado) 
                con = []; txt_docs = ""; datos_pubmed = ""
                
                if modo == "📚 Agente Investigador (PubMed)" and query_pubmed:
                    st.toast("📡 Conectando con servidores del NCBI...")
                    datos_pubmed = buscar_en_pubmed(query_pubmed)
                    if "Error" in datos_pubmed: st.error(datos_pubmed)
                    else: st.toast("✅ Artículos científicos obtenidos.")
                
                imagen_para_visor = None
                if modo != "📚 Agente Investigador (PubMed)":
                    for tipo, file in archivos:
                        if tipo == "doc":
                            file.seek(0) # REBOBINADO DE SEGURIDAD
                            r = pypdf.PdfReader(file)
                            txt_docs += "".join([p.extract_text() for p in r.pages])
                        else:
                            if imagen_dibujada is not None:
                                con.append(imagen_dibujada)
                                if not imagen_para_visor: imagen_para_visor = imagen_dibujada
                                imagen_dibujada = None 
                            else:
                                file.seek(0) # REBOBINADO DE SEGURIDAD
                                i_pil = Image.open(file)
                                if "ECG" in modo:
                                    img_aislada = aislar_trazado_ecg(i_pil)
                                    con.extend([i_pil, img_aislada])
                                    if not imagen_para_visor: imagen_para_visor = i_pil
                                else:
                                    con.append(i_pil)
                                    if not imagen_para_visor: imagen_para_visor = i_pil

                instruccion_bbox = ""
                if "ECG" in modo:
                    titulo_caja = "💡 LECTURA ECG Y MANEJO"
                    instruccion_modo = 'ERES UN CARDIÓLOGO CLÍNICO. Analiza ritmo, eje, ondas y segmentos.'
                    instruccion_bbox = "INSTRUCCIÓN OBLIGATORIA: Si detectas isquemia (ej. elevación/descenso ST), arritmia grave o bloqueo, ES VITAL que devuelvas exactamente esta etiqueta indicando dónde está: BBOX: [ymin, xmin, ymax, xmax] LABEL: Nombre Patología. Escala de 0 a 1000."
                elif modo == "📚 Agente Investigador (PubMed)":
                    titulo_caja = "📚 CONCLUSIÓN BASADA EN EVIDENCIA"
                    instruccion_modo = f'Actúas como un investigador académico. He buscado en PubMed los abstracts de los artículos más recientes sobre la consulta del usuario. Tu trabajo es leerlos y redactar un informe clínico riguroso.\n\nEVIDENCIA EXTRAÍDA DE PUBMED:\n{datos_pubmed}'
                else:
                    titulo_caja = "🛠️ PLAN DE ACCIÓN"
                    instruccion_modo = 'Analiza el caso clínico proporcionado.'

                prompt = f"""
                Rol: Médico Especialista IA V94. Contexto: {contexto}. Modo: {modo}.
                Pregunta del usuario: "{notas}"
                Docs: {txt_docs[:10000]}
                
                INSTRUCCIONES CRÍTICAS:
                1. MÁXIMA ESTRICTEZ: PROHIBIDO escribir texto gris o saludos fuera de las cajas HTML. Tu respuesta debe empezar directamente con la primera caja <div class="diagnosis-box">.
                2. Si el usuario ha hecho un dibujo en la foto (en verde brillante), céntrate en responder qué hay dentro de ese dibujo o zona marcada.
                3. {instruccion_modo}
                4. {instruccion_bbox}
                5. REGLA DE OMISIÓN: Si faltan datos de laboratorio o analíticas, NO menciones que faltan. Ignóralo.

                FORMATO HTML REQUERIDO:
                <div class="diagnosis-box"><b>🚨 HALLAZGOS PRINCIPALES:</b><br>[Descripción clínica]</div>
                <div class="action-box"><b>⚡ ACCIÓN INMEDIATA:</b><br>[Explicación]</div>
                <div class="{'pubmed-box' if 'PubMed' in modo else 'material-box'}"><b>{titulo_caja}:</b><br>[Desarrollo / Bibliografía referenciada con PMIDs]</div>
                """

                resp = model.generate_content([prompt, *con] if con else prompt)
                texto_generado = resp.text

                if "<div" in texto_generado:
                    texto_generado = texto_generado[texto_generado.find("<div"):]

                if imagen_para_visor:
                    img_marcada, texto_generado, detectado = extraer_y_dibujar_bboxes(texto_generado, imagen_para_visor)
                    if detectado: st.session_state.img_marcada = img_marcada
                    else: st.session_state.img_marcada = imagen_para_visor

                st.session_state.resultado_analisis = texto_generado.strip()
                st.session_state.pdf_bytes = create_pdf(st.session_state.resultado_analisis)

            except Exception as e: st.error(f"Error: {e}")

    if st.session_state.resultado_analisis:
        st.markdown(st.session_state.resultado_analisis, unsafe_allow_html=True)
        if st.session_state.pdf_bytes:
            st.download_button("📥 Descargar Informe PDF", st.session_state.pdf_bytes, "informe.pdf")

with col_right:
    with st.expander("👁️ Visor Visual / IA", expanded=True):
        if st.session_state.get("img_marcada"):
            st.markdown("#### 🎯 Detección IA / Dibujo Usuario")
            st.image(st.session_state.img_marcada, use_container_width=True)
        else:
            st.caption("Aquí aparecerá la imagen procesada o el análisis del agente.")
