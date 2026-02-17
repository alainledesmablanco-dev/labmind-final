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
import cv2
import numpy as np
import extra_streamlit_components as stx
from streamlit_bokeh_events import streamlit_bokeh_events
import pandas as pd

# --- CONFIGURACIÓN ---
st.set_page_config(page_title="LabMind 24.0 (Map & Timeline)", page_icon="🧬", layout="wide")

# --- ESTILOS CSS ---
st.markdown("""
<style>
    .stButton>button { width: 100%; border-radius: 8px; font-weight: bold; background-color: #0066cc; color: white; }
    .box-diag { background-color: #ffebee; border-left: 6px solid #ef5350; padding: 12px; margin-bottom: 8px; border-radius: 4px; color: #c62828; }
    .box-action { background-color: #e3f2fd; border-left: 6px solid #2196f3; padding: 12px; margin-bottom: 8px; border-radius: 4px; color: #1565c0; }
    .box-mat { background-color: #e8f5e9; border-left: 6px solid #4caf50; padding: 12px; margin-bottom: 8px; border-radius: 4px; color: #2e7d32; }
    .tissue-bar-container { display: flex; width: 100%; height: 25px; border-radius: 12px; overflow: hidden; margin: 10px 0; }
    .tissue-gran { background-color: #ef5350; height: 100%; display: flex; align-items: center; justify-content: center; color: white; font-size: 0.7em; }
    .tissue-slough { background-color: #fdd835; height: 100%; display: flex; align-items: center; justify-content: center; color: #333; font-size: 0.7em; }
    .tissue-nec { background-color: #212121; height: 100%; display: flex; align-items: center; justify-content: center; color: white; font-size: 0.7em; }
    .timeline-card { background-color: #f8f9fa; border: 1px solid #dee2e6; padding: 10px; border-radius: 8px; margin-bottom: 5px; }
</style>
""", unsafe_allow_html=True)

# --- GESTOR COOKIES ---
cookie_manager = stx.CookieManager()

# --- ESTADO ---
if "autenticado" not in st.session_state: st.session_state.autenticado = False
if "api_key" not in st.session_state: st.session_state.api_key = ""
if "resultado_analisis" not in st.session_state: st.session_state.resultado_analisis = None
if "pdf_bytes" not in st.session_state: st.session_state.pdf_bytes = None
if "area_herida" not in st.session_state: st.session_state.area_herida = 0.0
if "historial_evolucion" not in st.session_state: st.session_state.historial_evolucion = []
if "punto_mapa" not in st.session_state: st.session_state.punto_mapa = "No especificado"

# --- LOGIN (COOKIES) ---
cookie_api_key = cookie_manager.get(cookie="labmind_secret_key")
if not st.session_state.autenticado:
    if cookie_api_key:
        st.session_state.api_key = cookie_api_key
        st.session_state.autenticado = True
        st.rerun()
    else:
        st.title("LabMind Acceso")
        with st.form("login"):
            k = st.text_input("API Key:", type="password")
            if st.form_submit_button("Entrar"):
                expires = datetime.datetime.now() + datetime.timedelta(days=30)
                cookie_manager.set("labmind_secret_key", k, expires_at=expires)
                st.session_state.api_key = k; st.session_state.autenticado = True; st.rerun()
        st.stop()

# ==========================================
#      FUNCIONES DE ANÁLISIS
# ==========================================

def medir_herida(pil_image):
    img = np.array(pil_image.convert('RGB'))
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray_blurred = cv2.GaussianBlur(gray, (9, 9), 2)
    circles = cv2.HoughCircles(gray_blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=50, param1=100, param2=30, minRadius=20, maxRadius=200)
    area = 0.0
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        r_c = circles[0][2]
        pixels_per_cm = (r_c * 2) / 2.325
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array([0, 50, 50]), np.array([30, 255, 255])) + cv2.inRange(hsv, np.array([170, 50, 50]), np.array([180, 255, 255]))
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        area_px = sum(cv2.contourArea(c) for c in cnts if cv2.contourArea(c) > 500)
        area = area_px * ((1 / pixels_per_cm) ** 2)
    return area

# ==========================================
#      MAIN APP
# ==========================================

st.title("🩺 LabMind 24.0")
col_mapa, col_main, col_evol = st.columns([1, 2, 1])

# --- COLUMNA 1: MAPA CORPORAL ---
with col_mapa:
    st.subheader("📍 Localización")
    # Imagen de homúnculo médico
    img_mapa = "https://cdn-icons-png.flaticon.com/512/3133/3133694.png"
    st.image(img_mapa, width=250)
    
    opciones_cuerpo = ["Cabeza", "Tronco", "Brazo Dch", "Brazo Izq", "Mano Dch", "Mano Izq", "Sacro/Glúteos", "Muslo", "Rodilla", "Talón Dch", "Talón Izq", "Pie"]
    st.session_state.punto_mapa = st.selectbox("Indique zona afectada:", opciones_cuerpo)
    st.info(f"Seleccionado: **{st.session_state.punto_mapa}**")

# --- COLUMNA 2: CAPTURA Y ANÁLISIS ---
with col_main:
    st.subheader("1. Captura")
    modo = st.radio("Modo:", ["🩹 Heridas", "Dermatología", "ECG", "RX/Analíticas"], horizontal=True)
    fuente = st.file_uploader("Subir Imagen", type=["jpg", "png"])
    notas = st.text_area("Notas Clínicas:", placeholder="Alergias, fármacos, síntomas...")

    if fuente and st.button("🚀 ANALIZAR CASO", type="primary"):
        with st.spinner("🧠 Procesando..."):
            try:
                img_pil = Image.open(fuente)
                area_calc = medir_herida(img_pil) if modo == "🩹 Heridas" else 0.0
                st.session_state.area_herida = area_calc
                
                genai.configure(api_key=st.session_state.api_key)
                model = genai.GenerativeModel("models/gemini-3-flash-preview")
                
                prompt = f"""
                Actúa como Experto Clínico. Localización: {st.session_state.punto_mapa}. Modo: {modo}.
                Área medida: {area_calc:.2f} cm2. Notas: {notas}
                
                OUTPUT:
                TEJIDOS_DATA: [Gran%, Esf%, Nec%]
                ---
                ### ⚡ RESUMEN
                * **👤 LOCALIZACIÓN:** {st.session_state.punto_mapa}
                * **🚨 DIAGNÓSTICO:** [Breve]
                * **🩹 ACCIÓN:** [Inmediata]
                * **🔮 PREDICCIÓN:** [Pronóstico]
                * **🧴 MATERIAL:** [Lista]
                ---
                ### 📝 DETALLE
                [Análisis]
                """
                resp = model.generate_content([prompt, img_pil])
                st.session_state.resultado_analisis = resp.text
                
                # Guardar en Historial para la gráfica
                if modo == "🩹 Heridas":
                    st.session_state.historial_evolucion.append({
                        "Fecha": datetime.datetime.now().strftime("%H:%M"),
                        "Area": area_calc
                    })
            except Exception as e: st.error(f"Error: {e}")

    # Mostrar Resultados
    if st.session_state.resultado_analisis:
        txt = st.session_state.resultado_analisis
        parts = txt.split("---")
        
        # Tejidos
        tej_match = re.search(r'TEJIDOS_DATA: (\[.*?\])', txt)
        if tej_match and modo == "🩹 Heridas":
            g, e, n = eval(tej_match.group(1))
            st.markdown(f"""
            <div class="tissue-bar-container">
                <div class="tissue-gran" style="width: {g}%;">Rojo {g}%</div>
                <div class="tissue-slough" style="width: {e}%;">Amarillo {e}%</div>
                <div class="tissue-nec" style="width: {n}%;">Negro {n}%</div>
            </div>""", unsafe_allow_html=True)
            
        st.markdown(parts[1] if len(parts)>1 else txt)

# --- COLUMNA 3: HISTORIAL EVOLUTIVO ---
with col_evol:
    st.subheader("⏳ Evolución")
    if len(st.session_state.historial_evolucion) > 0:
        df = pd.DataFrame(st.session_state.historial_evolucion)
        st.line_chart(df.set_index("Fecha"))
        
        st.write("Registros hoy:")
        for item in reversed(st.session_state.historial_evolucion):
            st.markdown(f"""
            <div class="timeline-card">
                <b>{item['Fecha']}</b><br>
                Área: {item['Area']:.2f} cm²
            </div>""", unsafe_allow_html=True)
    else:
        st.info("Sin datos evolutivos aún.")

st.divider()
if st.button("🔒 Cerrar Sesión"):
    cookie_manager.delete("labmind_secret_key")
    st.session_state.autenticado = False; st.rerun()
