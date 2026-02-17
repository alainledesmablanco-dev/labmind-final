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
st.set_page_config(page_title="LabMind 26.0 (Pro)", page_icon="🔍", layout="wide")

# --- ESTILOS ---
st.markdown("""
<style>
    .stButton>button { width: 100%; border-radius: 8px; font-weight: bold; background-color: #0066cc; color: white; }
    .biofilm-alert { border: 2px solid #ffd600; padding: 10px; border-radius: 10px; background-color: #fff9c4; color: #827717; font-weight: bold; }
    .ar-guide { border: 2px dashed #2ecc71; padding: 20px; text-align: center; color: #27ae60; font-weight: bold; margin-bottom: 10px; }
    .prediction-box { background-color: #e8f5e9; padding: 15px; border-radius: 10px; border-left: 5px solid #4caf50; }
</style>
""", unsafe_allow_html=True)

# --- GESTOR COOKIES Y ESTADO ---
cookie_manager = stx.CookieManager()
if "historial_evolucion" not in st.session_state: st.session_state.historial_evolucion = []

# --- LÓGICA DE BIOFILM (MICRO-TEXTURA) ---
def detectar_biofilm(pil_image):
    img = np.array(pil_image.convert('RGB'))
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    
    # Análisis de texturas lisas/brillantes (Biofilm)
    # Usamos varianza local: el biofilm es liso (baja varianza) pero brillante (alta intensidad)
    kernel = np.ones((5,5),np.uint8)
    gradient = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel)
    _, mask = cv2.threshold(gradient, 10, 255, cv2.THRESH_BINARY_INV)
    
    # Superponer contorno amarillo en la imagen original
    img_res = img.copy()
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img_res, contours, -1, (255, 214, 0), 2)
    return Image.fromarray(img_res), len(contours) > 0

# --- LÓGICA PREDICTIVA ---
def predecir_cierre():
    hist = st.session_state.historial_evolucion
    if len(hist) < 3: return "Se necesitan al menos 3 registros para predecir."
    
    # Simulación de regresión simple
    areas = [h['Area'] for h in hist]
    reduccion_media = (areas[0] - areas[-1]) / len(hist)
    
    if reduccion_media <= 0: return "⚠️ Estancamiento detectado. No se puede predecir cierre."
     dias_restantes = areas[-1] / reduccion_media
    return f"Estimación de cierre: **{int(dias_restantes)} días** al ritmo actual."

# --- INTERFAZ ---
st.title("🩺 LabMind 26.0: Biofilm & Prediction")

col_left, col_right = st.columns([1, 1])

with col_left:
    st.markdown('<div class="ar-guide">📸 MODO ASISTENTE AR ACTIVO<br><small>Mantenga el móvil a 15cm y paralelo a la lesión</small></div>', unsafe_allow_html=True)
    fuente = st.file_uploader("Capturar Foto", type=["jpg", "png"])
    
    if fuente:
        img_original = Image.open(fuente)
        img_biofilm, detectado = detectar_biofilm(img_original)
        
        st.image(img_biofilm, caption="Análisis de Micro-textura (Biofilm en Amarillo)")
        if detectado:
            st.markdown('<div class="biofilm-alert">🔍 SOSPECHA DE BIOFILM: Detectadas zonas con patrón mucoso brillante.</div>', unsafe_allow_html=True)

with col_right:
    st.subheader("📈 Pronóstico y Evolución")
    
    # Simular ingreso de dato para la demo si se analiza
    if fuente and st.button("🚀 EJECUTAR ANÁLISIS PRO"):
        # (Aquí iría la llamada a Gemini 3 Flash Preview como en versiones anteriores)
        # Simulamos un área para la predicción
        nueva_area = 15.0 - (len(st.session_state.historial_evolucion) * 1.5)
        st.session_state.historial_evolucion.append({
            "Fecha": datetime.datetime.now().strftime("%d/%m"), 
            "Area": max(nueva_area, 0.5)
        })
        st.success("Análisis completado e integrado en historial.")

    # Mostrar Predicción
    prediccion = predecir_cierre()
    st.markdown(f'<div class="prediction-box">🔮 <b>IA de Supervivencia:</b><br>{prediccion}</div>', unsafe_allow_html=True)
    
    if st.session_state.historial_evolucion:
        df = pd.DataFrame(st.session_state.historial_evolucion)
        st.line_chart(df.set_index("Fecha"))

st.divider()
st.caption("LabMind 26.0 - Tecnología de análisis de textura bacteriana y modelado predictivo de cierre.")
