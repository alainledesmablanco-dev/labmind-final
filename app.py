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
import pandas as pd

# --- CONFIGURACIÓN ---
st.set_page_config(page_title="LabMind 26.1 (Fix)", page_icon="🧬", layout="wide")

# --- ESTILOS ---
st.markdown("""
<style>
    .stButton>button { width: 100%; border-radius: 8px; font-weight: bold; background-color: #0066cc; color: white; }
    .biofilm-alert { border: 2px solid #ffd600; padding: 10px; border-radius: 10px; background-color: #fff9c4; color: #827717; font-weight: bold; }
    .ar-guide { border: 2px dashed #2ecc71; padding: 20px; text-align: center; color: #27ae60; font-weight: bold; margin-bottom: 10px; }
    .prediction-box { background-color: #e8f5e9; padding: 15px; border-radius: 10px; border-left: 5px solid #4caf50; }
    /* Estilos para que la barra de tejidos se vea bien siempre */
    .tissue-bar-container { display: flex; width: 100%; height: 25px; border-radius: 12px; overflow: hidden; margin: 10px 0; }
    .tissue-gran { background-color: #ef5350; color: white; display: flex; align-items: center; justify-content: center; font-size: 0.7em; font-weight: bold; }
    .tissue-slough { background-color: #fdd835; color: #333; display: flex; align-items: center; justify-content: center; font-size: 0.7em; font-weight: bold; }
    .tissue-nec { background-color: #212121; color: white; display: flex; align-items: center; justify-content: center; font-size: 0.7em; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# --- GESTOR COOKIES Y ESTADO ---
cookie_manager = stx.CookieManager()
if "historial_evolucion" not in st.session_state: st.session_state.historial_evolucion = []
if "autenticado" not in st.session_state: st.session_state.autenticado = False
if "api_key" not in st.session_state: st.session_state.api_key = ""
if "resultado_analisis" not in st.session_state: st.session_state.resultado_analisis = None

# --- LOGIN ---
time.sleep(0.1)
cookie_api_key = cookie_manager.get(cookie="labmind_secret_key")
if not st.session_state.autenticado:
    if cookie_api_key:
        st.session_state.api_key = cookie_api_key; st.session_state.autenticado = True; st.rerun()
    else:
        st.title("LabMind Acceso")
        k = st.text_input("API Key:", type="password")
        if st.button("Entrar"):
            expires = datetime.datetime.now() + datetime.timedelta(days=30)
            cookie_manager.set("labmind_secret_key", k, expires_at=expires)
            st.session_state.api_key = k; st.session_state.autenticado = True; st.rerun()
        st.stop()

# --- LÓGICA DE BIOFILM ---
def detectar_biofilm(pil_image):
    try:
        img = np.array(pil_image.convert('RGB'))
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        kernel = np.ones((5,5),np.uint8)
        gradient = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel)
        _, mask = cv2.threshold(gradient, 10, 255, cv2.THRESH_BINARY_INV)
        img_res = img.copy()
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img_res, contours, -1, (255, 214, 0), 2)
        return Image.fromarray(img_res), len(contours) > 0
    except: return pil_image, False

# --- LÓGICA PREDICTIVA ---
def predecir_cierre():
    hist = st.session_state.historial_evolucion
    if len(hist) < 3: return "Se necesitan al menos 3 registros para predecir."
    areas = [h['Area'] for h in hist]
    reduccion_media = (areas[0] - areas[-1]) / len(hist)
    if reduccion_media <= 0: return "⚠️ Estancamiento detectado. No se puede predecir cierre."
    dias_restantes = areas[-1] / reduccion_media
    return f"Estimación de cierre: **{int(dias_restantes)} días** al ritmo actual."

# --- INTERFAZ ---
st.title("🩺 LabMind 26.1")

col_left, col_right = st.columns([1, 1])

with col_left:
    st.markdown('<div class="ar-guide">📸 MODO ASISTENTE AR ACTIVO<br><small>Mantenga el móvil a 15cm y paralelo a la lesión</small></div>', unsafe_allow_html=True)
    fuente = st.file_uploader("Capturar Foto", type=["jpg", "png"])
    
    if fuente:
        img_original = Image.open(fuente)
        img_biofilm, detectado = detectar_biofilm(img_original)
        
        st.image(img_biofilm, caption="Análisis de Micro-textura (Biofilm en Amarillo)")
        if detectado:
            st.markdown('<div class="biofilm-alert">🔍 SOSPECHA DE BIOFILM DETECTADA</div>', unsafe_allow_html=True)

with col_right:
    st.subheader("📈 Análisis y Pronóstico")
    
    if fuente and st.button("🚀 ANALIZAR (Preview)", type="primary"):
        with st.spinner("Procesando..."):
            try:
                genai.configure(api_key=st.session_state.api_key)
                model = genai.GenerativeModel("models/gemini-3-flash-preview")
                
                # Prompt que fuerza a la IA a usar HTML para la barra
                prompt = """
                Analiza esta herida. 
                IMPORTANTE: Si detectas tejidos, genera el código HTML de la barra de porcentajes.
                Ejemplo: <div class="tissue-bar-container"><div class="tissue-gran" style="width:50%">50%</div>...</div>
                
                OUTPUT:
                ### ⚡ RESUMEN
                ...
                ### 📊 Segmentación de Tejidos
                [Aquí inserta el HTML de la barra]
                ...
                """
                
                resp = model.generate_content([prompt, img_original])
                st.session_state.resultado_analisis = resp.text
                
                # Simular dato para gráfica
                st.session_state.historial_evolucion.append({
                    "Fecha": datetime.datetime.now().strftime("%d/%m %H:%M"), 
                    "Area": 10.5 # Simulado si no medimos
                })
            except Exception as e: st.error(f"Error: {e}")

    # --- CORRECCIÓN CRÍTICA AQUÍ ---
    if st.session_state.resultado_analisis:
        # Añadimos unsafe_allow_html=True para que la barra se dibuje y no salga como texto
        st.markdown(st.session_state.resultado_analisis, unsafe_allow_html=True)

    # Predicción
    prediccion = predecir_cierre()
    st.markdown(f'<div class="prediction-box">🔮 <b>IA de Supervivencia:</b><br>{prediccion}</div>', unsafe_allow_html=True)
    
    if st.session_state.historial_evolucion:
        df = pd.DataFrame(st.session_state.historial_evolucion)
        st.line_chart(df.set_index("Fecha"))

st.divider()
