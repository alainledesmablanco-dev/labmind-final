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

# --- CONFIGURACIÓN ---
st.set_page_config(page_title="LabMind 88.0 (Radiomics & Vector ECG)", page_icon="🧬", layout="wide")

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
    
    .diagnosis-box { background-color: #e3f2fd; border-left: 6px solid #2196f3; padding: 15px; border-radius: 8px; margin-bottom: 10px; color: #0d47a1; font-family: sans-serif; }
    .action-box { background-color: #ffebee; border-left: 6px solid #f44336; padding: 15px; border-radius: 8px; margin-bottom: 10px; color: #b71c1c; font-family: sans-serif; }
    .material-box { background-color: #e8f5e9; border-left: 6px solid #4caf50; padding: 15px; border-radius: 8px; margin-bottom: 15px; color: #1b5e20; font-family: sans-serif; }
    .radiomics-box { background-color: #f3e5f5; border-left: 6px solid #9c27b0; padding: 15px; border-radius: 8px; margin-bottom: 10px; color: #4a148c; font-family: sans-serif; }
    
    .tissue-labels { display: flex; width: 100%; margin-bottom: 2px; }
    .tissue-label-text { font-size: 0.75rem; text-align: center; font-weight: bold; color: #555; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
    .tissue-bar-container { display: flex; width: 100%; height: 20px; border-radius: 10px; overflow: hidden; margin-bottom: 15px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
    .tissue-gran { background-color: #ef5350; height: 100%; }
    .tissue-slough { background-color: #fdd835; height: 100%; }
    .tissue-nec { background-color: #212121; height: 100%; }
    
    .ghost-alert { background-color: #e0f7fa; color: #006064; padding: 10px; border-radius: 8px; border: 1px dashed #00bcd4; margin-bottom: 10px; text-align: center; font-weight: bold; }
    .prediction-box { background-color: #f8f9fa; border-left: 5px solid #6c757d; padding: 15px; border-radius: 8px; margin-bottom: 15px; }
    .push-badge { display: inline-block; background-color: #3f51b5; color: white; padding: 3px 8px; border-radius: 12px; font-size: 0.8rem; font-weight: bold; margin-bottom: 10px;}
    .biofilm-alert { color: #d32f2f; font-weight: bold; font-size: 0.85rem; margin-top: 5px; }
    
    .pull-up { margin-top: -25px !important; margin-bottom: 5px !important; height: 1px !important; display: block !important; }
    [data-testid='stFileUploaderDropzone'] { padding: 5px !important; min-height: 60px; }
    [data-testid='stFileUploaderDropzone'] div div span { display: none; }
    [data-testid='stFileUploaderDropzone'] div div::after { content: "📂 Adjuntar"; font-size: 0.9rem; color: #555; display: block; }
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
if "log_privacidad" not in st.session_state: st.session_state.log_privacidad = []
if "punto_cuerpo" not in st.session_state: st.session_state.punto_cuerpo = "No especificada"
if "chat_messages" not in st.session_state: st.session_state.chat_messages = []
if "history_db" not in st.session_state: st.session_state.history_db = []

if "img_previo" not in st.session_state: st.session_state.img_previo = None 
if "img_actual" not in st.session_state: st.session_state.img_actual = None 
if "img_ghost" not in st.session_state: st.session_state.img_ghost = None   
if "img_marcada" not in st.session_state: st.session_state.img_marcada = None 
if "last_cv_data" not in st.session_state: st.session_state.last_cv_data = None 
if "last_biofilm_detected" not in st.session_state: st.session_state.last_biofilm_detected = False
if "ecg_vector_data" not in st.session_state: st.session_state.ecg_vector_data = None

if "patient_risk_factor" not in st.session_state: st.session_state.patient_risk_factor = 1.0
if "patient_risk_reason" not in st.session_state: st.session_state.patient_risk_reason = "Estándar"
if "lab_albumin" not in st.session_state: st.session_state.lab_albumin = None
if "lab_hba1c" not in st.session_state: st.session_state.lab_hba1c = None
if "lab_itb" not in st.session_state: st.session_state.lab_itb = None

if "prefs_loaded" not in st.session_state:
    try:
        c_mon = cookie_manager.get("pref_moneda")
        c_vis = cookie_manager.get("pref_visual")
        c_src = cookie_manager.get("pref_fuente")
        st.session_state.pref_moneda = True if c_mon == "True" else False
        st.session_state.pref_visual = True if c_vis == "True" else False
        st.session_state.pref_fuente = 1 if c_src == "WebCam" else 0
        st.session_state.prefs_loaded = True
    except:
        st.session_state.pref_moneda = False; st.session_state.pref_visual = False; st.session_state.pref_fuente = 0; st.session_state.prefs_loaded = True

def update_cookie_moneda():
    val = st.session_state.get("chk_moneda_global", False)
    st.session_state.pref_moneda = val
    cookie_manager.set("pref_moneda", str(val), expires_at=datetime.datetime.now() + datetime.timedelta(days=30))

def update_cookie_visual():
    val = st.session_state.get("chk_visual_global", False)
    st.session_state.pref_visual = val
    cookie_manager.set("pref_visual", str(val), expires_at=datetime.datetime.now() + datetime.timedelta(days=30))

def update_cookie_fuente():
    val = "Archivo"
    if "rad_src_wounds" in st.session_state: val = st.session_state.rad_src_wounds
    elif "rad_src_integral" in st.session_state: val = st.session_state.rad_src_integral
    idx = 1 if val == "📸 WebCam" else 0
    st.session_state.pref_fuente = idx
    cookie_manager.set("pref_fuente", "WebCam" if idx==1 else "Archivo", expires_at=datetime.datetime.now() + datetime.timedelta(days=30))

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
#      FUNCIONES VISIÓN & CLÍNICAS (V88)
# ==========================================

def generar_vistas_radiologicas(pil_image):
    try:
        img_cv = cv2.cvtColor(np.array(pil_image.convert('RGB')), cv2.COLOR_RGB2BGR)
        lab = cv2.cvtColor(img_cv, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl,a,b))
        img_clahe = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        img_neg = cv2.bitwise_not(img_cv)
        return (Image.fromarray(cv2.cvtColor(img_clahe, cv2.COLOR_BGR2RGB)),
                Image.fromarray(cv2.cvtColor(img_neg, cv2.COLOR_BGR2RGB)))
    except: return None, None

def aislar_trazado_ecg(pil_image):
    try:
        img_cv = cv2.cvtColor(np.array(pil_image.convert('RGB')), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 15)
        smooth = cv2.GaussianBlur(thresh, (3, 3), 0)
        return Image.fromarray(cv2.cvtColor(smooth, cv2.COLOR_GRAY2RGB))
    except: return pil_image

# V88.0: VECTORIZACIÓN 1D MATEMÁTICA
def vectorizar_ecg_1d(img_aislada_pil):
    """Convierte la imagen del ECG aislado en un DataFrame de Pandas (Serie Temporal)"""
    try:
        img_cv = cv2.cvtColor(np.array(img_aislada_pil), cv2.COLOR_RGB2GRAY)
        _, thresh = cv2.threshold(img_cv, 127, 255, cv2.THRESH_BINARY_INV) # Tinta es blanco
        
        vector = []
        h, w = thresh.shape
        for x in range(w):
            col = thresh[:, x]
            y_indices = np.where(col > 0)[0]
            if len(y_indices) > 0:
                vector.append(h - np.mean(y_indices)) # Invertir para que los picos vayan arriba
            else:
                vector.append(np.nan)
                
        df = pd.DataFrame({'Amplitud (mV rel)': vector})
        df = df.interpolate().fillna(method='bfill').fillna(method='ffill')
        return df
    except: return None

# V88.0: SLICING DE 12 DERIVACIONES
def hacer_slicing_ecg(pil_image):
    """Corta el ECG en una cuadrícula 3 filas x 4 columnas (Estándar 12 leads)"""
    try:
        img_cv = cv2.cvtColor(np.array(pil_image.convert('RGB')), cv2.COLOR_RGB2BGR)
        h, w = img_cv.shape[:2]
        # Asumimos que el 80% superior contiene la rejilla 3x4 y el 20% inferior el DII largo
        grid_h = int((h * 0.8) / 3)
        grid_w = int(w / 4)
        
        slices = []
        nombres = [['I', 'aVR', 'V1', 'V4'], 
                   ['II', 'aVL', 'V2', 'V5'], 
                   ['III', 'aVF', 'V3', 'V6']]
                   
        for row in range(3):
            for col in range(4):
                y1 = row * grid_h
                y2 = (row + 1) * grid_h
                x1 = col * grid_w
                x2 = (col + 1) * grid_w
                trozo = img_cv[y1:y2, x1:x2]
                
                # Añadir etiqueta
                cv2.rectangle(trozo, (0,0), (60, 30), (255,255,255), -1)
                cv2.putText(trozo, nombres[row][col], (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
                slices.append(Image.fromarray(cv2.cvtColor(trozo, cv2.COLOR_BGR2RGB)))
        
        # Combinar en una sola imagen de galería resumen
        h_concat = []
        for r in range(3):
            fila = np.concatenate([np.array(slices[r*4 + c]) for c in range(4)], axis=1)
            h_concat.append(fila)
        final_grid = np.concatenate(h_concat, axis=0)
        
        # Redimensionar si es muy grande
        scale = 1000 / final_grid.shape[1]
        final_grid = cv2.resize(final_grid, (0,0), fx=scale, fy=scale)
        return Image.fromarray(final_grid)
    except: return None

# V88.0: MAPAS DE CALOR (GRAD-CAM PSEUDO-SALIENCY)
def extraer_y_dibujar_bboxes(texto, img_pil=None, video_path=None):
    patron = r'(?:TIMESTAMP:\s*([\d\.]+)\s*)?BBOX:\s*\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]\s*LABEL:\s*([^\n<]+)'
    matches = re.findall(patron, texto)
    if not matches: return None, texto, False
        
    base_img = img_pil
    primer_ts = matches[0][0]
    
    if primer_ts and video_path:
        try:
            ts_segundos = float(primer_ts)
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps > 0:
                frame_objetivo = int(ts_segundos * fps)
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_objetivo)
                ret, frame = cap.read()
                if ret: base_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            cap.release()
        except: pass
            
    if base_img is None: return None, re.sub(patron, '', texto).strip(), False

    img_cv = cv2.cvtColor(np.array(base_img.convert('RGB')), cv2.COLOR_RGB2BGR)
    h, w = img_cv.shape[:2]
    
    # Crear máscara para el Mapa de Calor
    heatmap_mask = np.zeros((h, w), dtype=np.float32)
    
    for match in matches:
        ts_str, ymin, xmin, ymax, xmax, label = match
        try:
            x1 = max(0, min(w, int(int(xmin) * w / 1000)))
            y1 = max(0, min(h, int(int(ymin) * h / 1000)))
            x2 = max(0, min(w, int(int(xmax) * w / 1000)))
            y2 = max(0, min(h, int(int(ymax) * h / 1000)))
            
            # Dibujar núcleo térmico en la máscara
            cx, cy = (x1+x2)//2, (y1+y2)//2
            radio = max((x2-x1)//2, (y2-y1)//2)
            cv2.circle(heatmap_mask, (cx, cy), radio, 255, -1)
            
            # Textos
            texto_label = label.strip().upper()
            escala_fuente = max(0.5, w/1200)
            grosor_fuente = max(1, int(w/600))
            (tw, th), _ = cv2.getTextSize(texto_label, cv2.FONT_HERSHEY_SIMPLEX, escala_fuente, grosor_fuente)
            cv2.putText(img_cv, texto_label, (x1, max(0, y1-10)), cv2.FONT_HERSHEY_SIMPLEX, escala_fuente, (255, 255, 255), grosor_fuente, cv2.LINE_AA)
        except: pass

    # Aplicar desenfoque masivo para efecto nube de calor
    heatmap_mask = cv2.GaussianBlur(heatmap_mask, (151, 151), 0)
    heatmap_mask = np.uint8(255 * (heatmap_mask / np.max(heatmap_mask))) if np.max(heatmap_mask) > 0 else np.uint8(heatmap_mask)
    
    # Colorear y fusionar (Rojo/Jet sobre original)
    colored_heatmap = cv2.applyColorMap(heatmap_mask, cv2.COLORMAP_JET)
    
    # Crear un alpha dinámico (solo oscurecer donde no hay calor)
    alpha_mask = heatmap_mask.astype(float) / 255.0
    alpha_mask = np.expand_dims(alpha_mask, axis=2)
    
    img_res = (img_cv * (1 - alpha_mask*0.5) + colored_heatmap * (alpha_mask*0.5)).astype(np.uint8)

    texto_limpio = re.sub(patron, '', texto)
    return Image.fromarray(cv2.cvtColor(img_res, cv2.COLOR_BGR2RGB)), texto_limpio.strip(), True

def procesar_termografia(pil_image):
    try:
        img = np.array(pil_image.convert('RGB'))
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
        _, a, _ = cv2.split(lab)
        thermal_map = cv2.normalize(a, None, 0, 255, cv2.NORM_MINMAX)
        thermal_colored = cv2.applyColorMap(thermal_map, cv2.COLORMAP_JET)
        thermal_colored = cv2.GaussianBlur(thermal_colored, (15, 15), 0)
        return Image.fromarray(cv2.cvtColor(thermal_colored, cv2.COLOR_BGR2RGB))
    except: return pil_image

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

def analisis_avanzado_heridas(pil_image, usar_moneda=False):
    img_np = np.array(pil_image.convert('RGB'))
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    
    img_calibrada = img_bgr.copy()
    pixels_per_cm = 0
    
    circles = cv2.HoughCircles(cv2.GaussianBlur(gray, (9, 9), 2), cv2.HOUGH_GRADIENT, 1.2, 50, param1=100, param2=30, minRadius=20, maxRadius=300)
    
    if circles is not None and usar_moneda:
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
            cv2.circle(img_calibrada, (x, y), r, (0, 255, 0), 4)
            cv2.putText(img_calibrada, "1 EUR", (x - 40, y - r - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            pixels_per_cm = (r * 2) / 2.325
            mask_moneda = np.zeros_like(gray)
            cv2.circle(mask_moneda, (x, y), int(r*0.6), 255, -1)
            mean_moneda = cv2.mean(img_bgr, mask=mask_moneda)[:3]
            if all(m > 0 for m in mean_moneda):
                f_b, f_g, f_r = 150.0/mean_moneda[0], 150.0/mean_moneda[1], 150.0/mean_moneda[2]
                img_calibrada = cv2.convertScaleAbs(img_bgr, alpha=(f_b+f_g+f_r)/3.0)
            break
            
    if pixels_per_cm == 0: pixels_per_cm = 100.0 

    hsv = cv2.cvtColor(img_calibrada, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(hsv, np.array([0, 50, 50]), np.array([10, 255, 255]))
    mask2 = cv2.inRange(hsv, np.array([170, 50, 50]), np.array([180, 255, 255]))
    mask_herida = cv2.morphologyEx(mask1 + mask2, cv2.MORPH_OPEN, np.ones((5,5), np.uint8))
    
    contours, _ = cv2.findContours(mask_herida, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    area_pixels_total = 0
    best_contour = None
    
    for c in contours:
        if cv2.contourArea(c) > 500:
            area_pixels_total += cv2.contourArea(c)
            if best_contour is None or cv2.contourArea(c) > cv2.contourArea(best_contour):
                best_contour = c
                
    area_final = area_pixels_total * ((1 / pixels_per_cm) ** 2) if pixels_per_cm > 0 else 0

    fitzpatrick = "No evaluable"; estado_bordes = "No evaluable"; riesgo_isquemia = "No evaluable"
    img_segmentada = img_calibrada.copy()
    img_depth = None
    p_nec, p_esf, p_gra = 0, 0, 0

    if best_contour is not None:
        cv2.drawContours(img_calibrada, [best_contour], -1, (0, 0, 255), 2)
        mask_piel = cv2.bitwise_not(mask_herida)
        mean_piel = cv2.mean(img_calibrada, mask=mask_piel)[:3]
        luma = 0.299*mean_piel[2] + 0.587*mean_piel[1] + 0.114*mean_piel[0]
        
        if luma > 170: fitzpatrick = "Tipo I-II (Clara)"
        elif luma > 100: fitzpatrick = "Tipo III-IV (Intermedia)"
        else: fitzpatrick = "Tipo V-VI (Oscura)"
        
        if "Oscura" not in fitzpatrick and mean_piel[2] < (mean_piel[0] + 15): riesgo_isquemia = "ALTO"
        else: riesgo_isquemia = "Bajo"

        perimeter = cv2.arcLength(best_contour, True)
        circularity = 4 * np.pi * (cv2.contourArea(best_contour) / (perimeter * perimeter)) if perimeter > 0 else 0
        if circularity < 0.35: estado_bordes = "Irregulares/Socavados"
        else: estado_bordes = "Regulares"

        mask_roi = np.zeros(img_calibrada.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask_roi, [best_contour], -1, 255, -1)
        roi_hsv = cv2.bitwise_and(hsv, hsv, mask=mask_roi)
        
        m_nec = cv2.inRange(roi_hsv, (0,0,0), (180, 255, 60))
        m_esf = cv2.inRange(roi_hsv, (15, 50, 50), (35, 255, 255))
        m_gra = cv2.inRange(roi_hsv, (0, 50, 61), (10, 255, 255)) + cv2.inRange(roi_hsv, (170, 50, 61), (180, 255, 255))
        
        total_wound_pixels = cv2.countNonZero(mask_roi)
        if total_wound_pixels > 0:
            p_nec = (cv2.countNonZero(cv2.bitwise_and(m_nec, mask_roi)) / total_wound_pixels) * 100
            p_esf = (cv2.countNonZero(cv2.bitwise_and(m_esf, mask_roi)) / total_wound_pixels) * 100
            p_gra = (cv2.countNonZero(cv2.bitwise_and(m_gra, mask_roi)) / total_wound_pixels) * 100

        overlay = img_segmentada.copy()
        overlay[m_nec > 0] = [0, 0, 0]; overlay[m_esf > 0] = [0, 255, 255]; overlay[m_gra > 0] = [0, 0, 255]     
        img_segmentada = cv2.addWeighted(img_calibrada, 0.6, overlay, 0.4, 0)

        x,y,w,h = cv2.boundingRect(best_contour)
        roi_gray = cv2.cvtColor(img_calibrada[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)
        depth_map = cv2.applyColorMap(255 - roi_gray, cv2.COLORMAP_JET) 
        img_depth = Image.fromarray(cv2.cvtColor(depth_map, cv2.COLOR_BGR2RGB))

    return {
        "area": area_final, "fitzpatrick": fitzpatrick, "bordes": estado_bordes, "isquemia": riesgo_isquemia,
        "p_nec": p_nec, "p_esf": p_esf, "p_gra": p_gra,
        "img_calibrada": Image.fromarray(cv2.cvtColor(img_calibrada, cv2.COLOR_BGR2RGB)),
        "img_segmentada": Image.fromarray(cv2.cvtColor(img_segmentada, cv2.COLOR_BGR2RGB)),
        "img_depth": img_depth
    }

def anonymize_face(pil_image):
    try:
        img_np = np.array(pil_image.convert('RGB'))
        img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            roi_color = img_cv[y:y+h, x:x+w]
            blurred = cv2.GaussianBlur(roi_color, (99, 99), 30)
            img_cv[y:y+h, x:x+w] = blurred
        return Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)), True
    except: return pil_image, False

def alinear_imagenes(img_ref_pil, img_mov_pil):
    try:
        ref = cv2.cvtColor(np.array(img_ref_pil), cv2.COLOR_RGB2GRAY)
        mov = cv2.cvtColor(np.array(img_mov_pil), cv2.COLOR_RGB2GRAY)
        if max(ref.shape) > 1000:
            s = 1000 / max(ref.shape); ref = cv2.resize(ref, (0,0), fx=s, fy=s); mov = cv2.resize(mov, (0,0), fx=s, fy=s)
        orb = cv2.ORB_create(5000)
        kp1, des1 = orb.detectAndCompute(ref, None)
        kp2, des2 = orb.detectAndCompute(mov, None)
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = sorted(bf.match(des1, des2), key=lambda x: x.distance)
        good = matches[:int(len(matches) * 0.15)]
        if len(good) < 4: return img_mov_pil, None
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        M, _ = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
        h, w = ref.shape
        img_mov_color = np.array(img_mov_pil.convert('RGB'))
        img_mov_color = cv2.resize(img_mov_color, (w, h)) if img_mov_color.shape[:2] != (h,w) else img_mov_color
        aligned_cv = cv2.warpPerspective(img_mov_color, M, (w, h))
        ref_color = cv2.resize(np.array(img_ref_pil.convert('RGB')), (w, h))
        blend_cv = cv2.addWeighted(ref_color, 0.6, aligned_cv, 0.4, 0)
        return Image.fromarray(aligned_cv), Image.fromarray(blend_cv)
    except: return img_mov_pil, None

def predecir_cierre_inteligente():
    hist = st.session_state.historial_evolucion
    if len(hist) < 2: return "Necesito al menos 2 registros (Previo y Actual) para estimar."
    
    try:
        hist_sorted = sorted(hist, key=lambda x: datetime.datetime.strptime(x['Fecha'], "%d/%m").replace(year=datetime.datetime.now().year))
    except:
        hist_sorted = hist
        
    area_actual = hist_sorted[-1]['Area']
    prof_actual = hist_sorted[-1].get('Profundidad', 0.0)
    area_antigua = hist_sorted[0]['Area']
    prof_antigua = hist_sorted[0].get('Profundidad', 0.0)
    
    cv = st.session_state.get('last_cv_data', {})
    
    p_area = 0
    if area_actual > 0:
        if area_actual < 0.3: p_area = 1
        elif area_actual <= 0.6: p_area = 2
        elif area_actual <= 1.0: p_area = 3
        elif area_actual <= 2.0: p_area = 4
        elif area_actual <= 3.0: p_area = 5
        elif area_actual <= 4.0: p_area = 6
        elif area_actual <= 8.0: p_area = 7
        elif area_actual <= 12.0: p_area = 8
        elif area_actual <= 24.0: p_area = 9
        else: p_area = 10
        
    p_nec = cv.get('p_nec', 0); p_esf = cv.get('p_esf', 0); p_gra = cv.get('p_gra', 0)
    p_tejido = 1 
    if p_nec > 10: p_tejido = 4
    elif p_esf > 10: p_tejido = 3
    elif p_gra > 10: p_tejido = 2
    
    p_exudado = 2 
    if p_esf > 40: p_exudado = 3 
    push_score = p_area + p_tejido + p_exudado

    alb = st.session_state.get('lab_albumin')
    hba1c = st.session_state.get('lab_hba1c')
    itb = st.session_state.get('lab_itb')
    factor_paciente = st.session_state.get('patient_risk_factor', 1.0)
    motivo_paciente = st.session_state.get('patient_risk_reason', 'Estándar')
    alerta_muro = ""; bloqueo_absoluto = False

    if alb is not None and alb < 2.5:
        bloqueo_absoluto = True
        alerta_muro += f"<div style='color:#d32f2f; font-weight:bold; margin-top:5px;'>🛑 Muro Biológico: Albúmina ({alb} g/dL). Cierre fisiológicamente imposible sin soporte nutricional.</div>"
    if itb is not None:
        if itb < 0.5:
            bloqueo_absoluto = True
            alerta_muro += f"<div style='color:#d32f2f; font-weight:bold; margin-top:5px;'>🛑 Isquemia Severa (ITB: {itb}). Contraindicada compresión. Cierre imposible sin revascularización urgente.</div>"
        elif 0.8 <= itb <= 1.2:
            factor_paciente *= 0.8 
            alerta_muro += f"<div style='color:#388e3c; font-weight:bold; margin-top:5px;'>🟢 ITB Óptimo ({itb}). Apto para compresión.</div>"
    if hba1c is not None and hba1c > 8.0:
        factor_paciente *= 1.5 
        alerta_muro += f"<div style='color:#e65100; font-weight:bold; margin-top:5px;'>⚠️ Alerta Metabólica: HbA1c ({hba1c}%). Microcirculación comprometida.</div>"

    if bloqueo_absoluto:
        return f"""
        <div class="prediction-box" style="border-left: 5px solid #d32f2f;">
            <div class="push-badge">Score PUSH: {push_score}/17</div>
            <h4 style="margin-top:5px; color:#d32f2f;">🛑 Gemelo Digital: Pronóstico Fallido</h4>
            {alerta_muro}
            <hr style="margin: 8px 0px; border-top: 1px dashed #ccc;">
            <span style='font-size: 0.85rem; color: #555;'>No se puede calcular fecha de alta hasta corregir los parámetros críticos.</span>
        </div>
        """

    reduccion_area = area_antigua - area_actual
    reduccion_prof = prof_antigua - prof_actual
    
    try:
        d1 = datetime.datetime.strptime(hist_sorted[0]['Fecha'], "%d/%m").replace(year=datetime.datetime.now().year)
        d2 = datetime.datetime.strptime(hist_sorted[-1]['Fecha'], "%d/%m").replace(year=datetime.datetime.now().year)
        dias_pasados = max(1, (d2 - d1).days)
    except:
        dias_pasados = max(1, 7 * (len(hist_sorted)-1))

    aceleracion_texto = "Geometría Constante"
    
    if reduccion_area <= 0: 
        if prof_antigua > 0 and reduccion_prof > 0:
            aceleracion_texto = "🧊 Falso Estancamiento: Área estable, reducción de cavidad."
            tasa_diaria = reduccion_prof / dias_pasados
            dias_base = prof_actual / tasa_diaria
            dias_estimados = dias_base * 1.5 
        else:
            return f"<div class='push-badge'>Score PUSH: {push_score}/17</div><br>⚠️ Sin mejoría detectada. Revisar plan."
    else:
        tasa_diaria = reduccion_area / dias_pasados
        dias_base = area_actual / tasa_diaria
        dias_estimados = dias_base * 1.3 
        
        if len(hist_sorted) >= 3:
            try:
                d_mid = datetime.datetime.strptime(hist_sorted[-2]['Fecha'], "%d/%m").replace(year=datetime.datetime.now().year)
                v_reciente = (hist_sorted[-2]['Area'] - area_actual) / max(1, (d2 - d_mid).days)
                v_antigua = (hist_sorted[0]['Area'] - hist_sorted[-2]['Area']) / max(1, (d_mid - d1).days)
                
                if v_reciente < (v_antigua * 0.8):
                    dias_estimados *= 1.4; aceleracion_texto = "Frenando (Cronificación)"
                elif v_reciente > (v_antigua * 1.2):
                    dias_estimados *= 0.8; aceleracion_texto = "Acelerando (Proliferativa)"
            except: pass

    penalizacion = 0
    alerta_biofilm = ""
    if st.session_state.get('last_biofilm_detected', False):
        penalizacion += 14
        alerta_biofilm = "<div class='biofilm-alert'>🦠 +14 días por carga bacteriana/biofilm.</div>"
        
    if cv:
        if "ALTO" in cv.get('isquemia', '') and itb is None:
            return f"<div class='push-badge'>Score PUSH: {push_score}/17</div><br>🛑 <b>Isquemia Visual ALTA:</b> Imposible predecir sin revascularización."
        penalizacion += (p_nec / 10.0) * 2 
        penalizacion += (p_esf / 10.0) * 1  
            
    zona = st.session_state.punto_cuerpo
    factor_anatomico = 1.0
    if zona in ["Cara", "Cuello"]: factor_anatomico = 0.7
    elif zona in ["Sacro/Glúteo"]: factor_anatomico = 1.3
    elif zona in ["Talón"]: factor_anatomico = 1.8
    elif zona in ["Pie"]: factor_anatomico = 1.4
    
    dias_base_ajustados = (dias_estimados + penalizacion)
    dias_finales = int(max(1, dias_base_ajustados * factor_anatomico * factor_paciente))
    
    html = f"""
    <div class="prediction-box">
        <div class="push-badge">Score PUSH: {push_score}/17</div>
        <h4 style="margin-top:5px; color:#1976d2;">📉 Gemelo Digital Predictivo</h4>
        Alta estimada en: <b>{dias_finales} a {dias_finales + 7} días</b>.
        {alerta_muro}
        {alerta_biofilm}
        <hr style="margin: 8px 0px; border-top: 1px dashed #ccc;">
        <ul style='font-size: 0.85rem; color: #555; margin-bottom: 0px; padding-left: 15px;'>
          <li><b>Evolución 3D:</b> {aceleracion_texto}.</li>
          <li><b>Perfusión ({zona}):</b> x{factor_anatomico}</li>
          <li><b>Metabolismo:</b> {motivo_paciente} (x{factor_paciente:.2f})</li>
        </ul>
    </div>
    """
    return html

def create_pdf(texto_analisis):
    class PDF(FPDF):
        def header(self): self.set_font('Arial','B',12); self.cell(0,10,'LabMind - Informe IA',0,1,'C'); self.ln(5)
        def footer(self): self.set_y(-15); self.set_font('Arial','I',8); self.cell(0,10,f'Pag {self.page_no()}',0,0,'C')
    pdf = PDF(); pdf.add_page(); pdf.set_font("Arial", size=10)
    pdf.cell(0,10,f"Fecha: {datetime.datetime.now().strftime('%d/%m/%Y %H:%M')}",0,1); pdf.ln(5)
    clean = re.sub(r'<[^>]+>', '', texto_analisis).replace('€','EUR').replace('’',"'").replace('“','"').replace('”','"')
    pdf.multi_cell(0,5, clean.encode('latin-1','replace').decode('latin-1'))
    return pdf.output(dest='S').encode('latin-1')

# ==========================================
#      INTERFAZ DE USUARIO
# ==========================================

st.title("🩺 LabMind 88.0")
col_left, col_center, col_right = st.columns([1, 2, 1])

# --- COLUMNA 1 ---
with col_left:
    st.subheader("📍 Datos Paciente")
    zonas_cuerpo = ["No especificada", "--- CABEZA ---", "Cara", "Cuello", "--- TRONCO ---", "Pecho", "Abdomen", "Espalda", "--- PELVIS ---", "Sacro/Glúteo", "Genitales", "--- EXTREMIDADES ---", "Brazo", "Mano", "Pierna", "Talón", "Pie"]
    seleccion_zona = st.selectbox("Zona anatómica:", zonas_cuerpo)
    st.session_state.punto_cuerpo = seleccion_zona
    
    with st.expander("📚 Protocolo Unidad", expanded=False):
        fixed_proto_path = None; detected_name = ""
        if os.path.exists("protocolo.pdf"): fixed_proto_path = "protocolo.pdf"; detected_name = "PDF"
        elif os.path.exists("protocolo.jpg"): fixed_proto_path = "protocolo.jpg"; detected_name = "JPG"
        elif os.path.exists("protocolo.png"): fixed_proto_path = "protocolo.png"; detected_name = "PNG"
        
        using_fixed_proto = False
        if fixed_proto_path:
            st.markdown(f'<div class="proto-success">✅ <b>protocolo.{detected_name.lower()}</b> detectado</div>', unsafe_allow_html=True)
            using_fixed_proto = True
        else:
            st.caption("Guarda 'protocolo.jpg' o '.pdf' en la carpeta.")

        proto_uploaded = st.file_uploader("Subir (Sobrescribe Fijo)", type=["pdf", "jpg", "png"], key="global_proto")

# --- COLUMNA 2 ---
with col_center:
    tab_analisis, tab_historial = st.tabs(["🔍 Analizar Caso", "🗂️ Historial Guardado"])
    
    with tab_analisis:
        st.subheader("1. Selección de Modo")
        
        modo = st.selectbox("Especialidad:", 
                     ["🩹 Heridas / Úlceras", "🧴 Dermatología", "🧩 Integral (Analizar Todo)", "💊 Farmacia", "📈 ECG", "💀 RX/TAC/Resonancia", "📂 Informes"])
        contexto = st.selectbox("🏥 Contexto:", ["Hospitalización", "Residencia", "Urgencias", "UCI", "Domicilio"])
        
        st.markdown('<div class="pull-up"></div>', unsafe_allow_html=True)
        
        archivos = []
        meds_files = None; labs_files = None; reports_files = None; ecg_files = None; rad_files = None 
        
        mostrar_imagenes = False
        usar_moneda = False
        
        if modo == "🧩 Integral (Analizar Todo)":
            with st.expander("📂 Documentación (PDFs, Analíticas)", expanded=False):
                c1, c2 = st.columns(2)
                meds_files = c1.file_uploader("💊 Fármacos", accept_multiple_files=True, key="int_meds")
                labs_files = c2.file_uploader("📊 Analíticas", accept_multiple_files=True, key="int_labs")
            st.write("📸 **Visual / Videos:**")
            
            mostrar_imagenes = st.checkbox("👁️ Mostrar Mapas y Heatmaps", value=st.session_state.pref_visual, key="chk_visual_global", on_change=update_cookie_visual)
            fuente_label = st.radio("Fuente:", ["📁 Archivo", "📸 WebCam"], horizontal=True, label_visibility="collapsed", index=st.session_state.pref_fuente, key="rad_src_integral", on_change=update_cookie_fuente)
            
            if fuente_label == "📸 WebCam":
                if f := st.camera_input("Foto Paciente"): archivos.append(("cam", f))
            else:
                if fs := st.file_uploader("Subir Imágenes o Videos", type=['jpg','png','mp4','mov'], accept_multiple_files=True, key="int_main"):
                    for f in fs: archivos.append(("video" if "video" in f.type or "mp4" in f.name.lower() or "mov" in f.name.lower() else "img", f))

        elif modo == "🩹 Heridas / Úlceras" or modo == "🧴 Dermatología":
            usar_moneda = st.checkbox("🪙 Usar moneda de 1€ para calibrar y medir", value=st.session_state.pref_moneda, key="chk_moneda_global", on_change=update_cookie_moneda)
            mostrar_imagenes = st.checkbox("👁️ Mostrar Segmentación y Heatmaps", value=st.session_state.pref_visual, key="chk_visual_global", on_change=update_cookie_visual)
            
            with st.expander("⏮️ Ver Evolución", expanded=False):
                prev = st.file_uploader("Foto Previa (Activa Modo Fantasma)", type=['jpg','png'], accept_multiple_files=True, key="w_prev")
                if prev:
                    try:
                        st.session_state.img_previo = Image.open(prev[0])
                        st.caption("✅ Foto previa cargada para alineación automática.")
                        for p in prev: archivos.append(("prev_img", p))
                    except: pass

                c_d, c_a, c_p, c_b = st.columns([0.3, 0.3, 0.3, 0.1])
                with c_d: d_m = st.date_input("Fecha", value=datetime.date.today()-datetime.timedelta(days=7))
                with c_a: a_m = st.number_input("Área (cm²)", min_value=0.0, step=0.1)
                with c_p: p_m = st.number_input("Prof. (cm)", min_value=0.0, step=0.1)
                with c_b: 
                    st.write(""); st.write("")
                    if st.button("➕", key="btn_add"): st.session_state.historial_evolucion.append({"Fecha": d_m.strftime("%d/%m"), "Area": a_m, "Profundidad": p_m})

            with st.expander("🩸 Docs y Analíticas (Afecta a Predicción)", expanded=False):
                st.caption("Sube PDFs de analíticas aquí para que la IA extraiga Albúmina, HbA1c o ITB.")
                meds_files = st.file_uploader("Docs / Analíticas", accept_multiple_files=True, key="w_meds")
            
            st.write("📸 **Estado ACTUAL:**")
            
            if st.session_state.img_previo and "Heridas" in modo:
                st.markdown('<div class="ghost-alert">👻 <b>GHOST MODE ACTIVO:</b> Al tomar la foto, la IA intentará alinearla automáticamente.</div>', unsafe_allow_html=True)
            
            fuente_label = st.radio("Fuente:", ["📁 Archivo", "📸 WebCam"], horizontal=True, label_visibility="collapsed", index=st.session_state.pref_fuente, key="rad_src_wounds", on_change=update_cookie_fuente)
            
            if fuente_label == "📸 WebCam":
                if st.session_state.img_previo and "Heridas" in modo:
                    st.image(st.session_state.img_previo, caption="REFERENCIA (Intenta imitar este ángulo)", width=150)
                if f := st.camera_input("Foto"): archivos.append(("cam", f))
            else:
                if fs := st.file_uploader("Subir", type=['jpg','png','mp4','mov'], accept_multiple_files=True, key="w_img"):
                    for f in fs: archivos.append(("video" if "video" in f.type else "img", f))

        elif modo == "📈 ECG": 
            mostrar_imagenes = st.checkbox("👁️ Mostrar Vectores 1D y Slicing 12-Leads", value=st.session_state.pref_visual, key="chk_visual_global", on_change=update_cookie_visual)
            if fs:=st.file_uploader("ECG", type=['jpg','pdf', 'png'], accept_multiple_files=True): 
                for f in fs: archivos.append(("img",f))
        elif modo == "💀 RX/TAC/Resonancia": 
            st.info("💀 **Radiómica Cuantitativa**: Si es RX y tiene barra de escala/DICOM, la IA calculará proporciones.")
            mostrar_imagenes = st.checkbox("👁️ Activar Multiespectral y Mapas de Calor Clínicos", value=st.session_state.pref_visual, key="chk_visual_global", on_change=update_cookie_visual)
            if fs:=st.file_uploader("Imágenes/Videos (RX, TAC, RMN)", type=['jpg','png','mp4','mov'], accept_multiple_files=True): 
                for f in fs: archivos.append(("video" if "video" in f.type or "mp4" in f.name.lower() or "mov" in f.name.lower() else "img", f))
                
        elif modo == "💊 Farmacia": meds_files = st.file_uploader("Receta", accept_multiple_files=True, key="p_docs")
        elif modo == "📂 Informes": reports_files = st.file_uploader("PDFs", accept_multiple_files=True, key="rep_docs")

        st.markdown('<div class="pull-up"></div>', unsafe_allow_html=True)
        
        audio_val = st.audio_input("🎙️ Notas de Voz", key="audio_recorder", label_visibility="collapsed")
        notas = st.text_area("Notas Clínicas:", height=60, placeholder="Escribe síntomas, patologías previas...")
        nota_historial = st.text_input("🏷️ Etiqueta Historial (Opcional):", placeholder="Ej: Cama 304", label_visibility="collapsed")

        galeria_avanzada = []

        col_btn1, col_btn2 = st.columns([3, 1])
        with col_btn1:
            btn_analizar = st.button("🚀 ANALIZAR", type="primary", use_container_width=True)
        with col_btn2:
            st.markdown('<div class="btn-nuevo-hook" style="display:none;"></div>', unsafe_allow_html=True)
            btn_nuevo = st.button("🔄 NUEVO", type="secondary", use_container_width=True)

        if btn_nuevo:
            st.session_state.resultado_analisis = None
            st.session_state.pdf_bytes = None
            st.session_state.historial_evolucion = []
            st.session_state.area_herida = 0.0
            st.session_state.chat_messages = []
            st.session_state.img_previo = None 
            st.session_state.img_actual = None 
            st.session_state.img_ghost = None   
            st.session_state.img_marcada = None 
            st.session_state.last_cv_data = None 
            st.session_state.last_biofilm_detected = False
            st.session_state.patient_risk_factor = 1.0
            st.session_state.patient_risk_reason = "Estándar"
            st.session_state.lab_albumin = None
            st.session_state.lab_hba1c = None
            st.session_state.lab_itb = None
            st.session_state.ecg_vector_data = None
            for key in list(st.session_state.keys()):
                if key in ["int_meds", "int_labs", "int_main", "w_prev", "w_meds", "w_img", "p_docs", "rep_docs", "audio_recorder"]:
                    del st.session_state[key]
            st.rerun()

        if btn_analizar:
            st.session_state.log_privacidad = []; st.session_state.area_herida = 0.0
            st.session_state.chat_messages = [] 
            st.session_state.img_actual = None; st.session_state.img_ghost = None ; st.session_state.img_marcada = None 
            st.session_state.patient_risk_factor = 1.0; st.session_state.patient_risk_reason = "Estándar"
            st.session_state.last_biofilm_detected = False
            st.session_state.ecg_vector_data = None
            
            st.session_state.lab_albumin = None
            st.session_state.lab_hba1c = None
            st.session_state.lab_itb = None
            
            with st.spinner(f"🧠 Procesando Visión Radiómica / Vectores ({modo})..."):
                video_paths_local = [] 
                primer_video_local = None 
                
                try:
                    genai.configure(api_key=st.session_state.api_key)
                    model = genai.GenerativeModel("models/gemini-3-flash-preview")
                    
                    con = []; txt_meds = ""; txt_labs = ""; txt_reports = ""; txt_proto = ""
                    datos_cv_texto = ""

                    final_proto_obj = None; is_local = False
                    if proto_uploaded: final_proto_obj = proto_uploaded
                    elif using_fixed_proto and fixed_proto_path: final_proto_obj = fixed_proto_path; is_local = True

                    if final_proto_obj:
                        is_pdf = False
                        if is_local:
                            if fixed_proto_path.endswith(".pdf"): is_pdf = True
                            file_handle = open(fixed_proto_path, "rb")
                        else:
                            if "pdf" in final_proto_obj.type: is_pdf = True
                            file_handle = final_proto_obj

                        if is_pdf:
                            r = pypdf.PdfReader(file_handle)
                            txt_proto += "".join([p.extract_text() for p in r.pages])
                        else:
                            img_proto = Image.open(file_handle)
                            con.append(img_proto)
                            txt_proto = "[PROTOCOLO EN IMAGEN ADJUNTA]"
                    
                    for fs in [meds_files, labs_files, reports_files]:
                        if fs: 
                            for f in fs:
                                if "pdf" in f.type: r=pypdf.PdfReader(f); txt_meds += "".join([p.extract_text() for p in r.pages])
                                else: con.append(Image.open(f))

                    if audio_val: con.append(genai.upload_file(audio_val, mime_type="audio/wav"))
                    
                    imagen_principal_para_marcar = None
                    
                    for label, a in archivos:
                        if "video" in label:
                            st.toast(f"⏳ Subiendo video {a.name} a Gemini...")
                            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tf:
                                tf.write(a.read())
                                tp = tf.name
                                video_paths_local.append(tp)
                                if not primer_video_local: primer_video_local = tp
                                
                            vf = genai.upload_file(path=tp)
                            while vf.state.name == "PROCESSING":
                                time.sleep(2)
                                vf = genai.get_file(vf.name)
                            con.append(vf)
                            st.toast(f"✅ Video subido y procesado")
                        else: 
                            img_pil = Image.open(a)
                            
                            # --- PRE-PROCESAMIENTO V88.0 (Slicing + Vectores) ---
                            if modo == "💀 RX/TAC/Resonancia":
                                img_clahe, img_neg = generar_vistas_radiologicas(img_pil)
                                if img_clahe and img_neg:
                                    con.extend([img_pil, img_clahe, img_neg])
                                    if mostrar_imagenes: 
                                        galeria_avanzada.append(("🩻 Alto Contraste (Filtro CLAHE)", img_clahe))
                                        galeria_avanzada.append(("🌑 Inversión Ósea (Negativo)", img_neg))
                                else: con.append(img_pil)
                            
                            elif modo == "📈 ECG":
                                img_aislada = aislar_trazado_ecg(img_pil)
                                
                                # Slicing Paralelo (Nivel 7)
                                img_slicing = hacer_slicing_ecg(img_pil)
                                if img_slicing: con.append(img_slicing)
                                
                                # Vectorización Matemática (Nivel 7)
                                df_vector = vectorizar_ecg_1d(img_aislada)
                                st.session_state.ecg_vector_data = df_vector
                                
                                con.extend([img_pil, img_aislada])
                                
                                if mostrar_imagenes: 
                                    if img_slicing: galeria_avanzada.append(("🧩 Slicing 12-Leads", img_slicing))
                                    galeria_avanzada.append(("📈 Señal Eléctrica Aislada", img_aislada))
                                
                            else:
                                if not imagen_principal_para_marcar:
                                    imagen_principal_para_marcar = img_pil
                                    
                                if ("Heridas" in modo or "Dermatología" in modo):
                                    if "prev" in label:
                                        con.append(img_pil)
                                    else:
                                        img_final_proc = img_pil
                                        if st.session_state.img_previo and "Heridas" in modo:
                                            aligned, ghost_view = alinear_imagenes(st.session_state.img_previo, img_pil)
                                            if ghost_view:
                                                st.session_state.img_ghost = ghost_view 
                                                img_final_proc = aligned 
                                        
                                        cv_data = analisis_avanzado_heridas(img_final_proc, usar_moneda)
                                        st.session_state.last_cv_data = cv_data 
                                        
                                        img_bio, has_bio = detectar_biofilm(img_final_proc)
                                        st.session_state.last_biofilm_detected = has_bio
                                        
                                        if cv_data["area"] > 0: st.session_state.area_herida = cv_data["area"]
                                        st.session_state.img_actual = cv_data["img_calibrada"]
                                        
                                        datos_cv_texto = f"""
                                        [DATOS VISIÓN COMPUTACIONAL]:
                                        - Piel: {cv_data['fitzpatrick']}
                                        - Bordes: {cv_data['bordes']}
                                        - Isquemia Perilesional: {cv_data['isquemia']}
                                        - Biofilm Óptico: {'Detectado' if has_bio else 'No detectado'}
                                        """
                                        
                                        galeria_avanzada.append(("🗺️ Segmentación", cv_data["img_segmentada"]))
                                        if cv_data["img_depth"]: galeria_avanzada.append(("🗻 Profundidad 3D", cv_data["img_depth"]))
                                        galeria_avanzada.append(("🌡️ Termografía", procesar_termografia(img_final_proc)))

                                        con.append(cv_data["img_calibrada"]); con.append(cv_data["img_segmentada"])
                                else: 
                                    img_final, proc = anonymize_face(img_pil)
                                    st.session_state.img_actual = img_final
                                    con.append(img_final)

                            if modo in ["💀 RX/TAC/Resonancia", "📈 ECG"] and not imagen_principal_para_marcar:
                                imagen_principal_para_marcar = img_pil

                    # --- LÓGICA DINÁMICA DEL PROMPT Y RADIÓMICA (V88) ---
                    if "Heridas" in modo or "Dermatología" in modo:
                        titulo_caja = "🛠️ CURA / TRATAMIENTO LOCAL"
                        instruccion_modo = 'Enfoque: Cuidado de heridas y piel. En la caja "CURA", **EXTRAE Y RECOMIENDA** productos basándote EXCLUSIVAMENTE en el protocolo adjunto.'
                        html_extra = """
                        2. BARRA TEJIDOS (Etiquetas fuera):
                        <div class="tissue-labels">
                            <div style="width:G%" class="tissue-label-text">Granulación G%</div>
                            <div style="width:E%" class="tissue-label-text">Esfacelos E%</div>
                            <div style="width:N%" class="tissue-label-text">Necrosis N%</div>
                        </div>
                        <div class="tissue-bar-container">
                           <div class="tissue-gran" style="width:G%"></div>
                           <div class="tissue-slough" style="width:E%"></div>
                           <div class="tissue-nec" style="width:N%"></div>
                        </div>
                        """
                    elif "RX" in modo:
                        titulo_caja = "💡 MANEJO Y RECOMENDACIONES"
                        instruccion_modo = 'Enfoque: Radiómica Cuantitativa. 1. Busca escala DICOM impresa o usa proporciones. 2. Calcula automáticamente el Índice Cardiotorácico (ancho máximo corazón / ancho interno tórax). 3. LECTURA SISTEMÁTICA: Estructura tu lectura siguiendo el ABCDE.'
                        html_extra = ""
                    elif "ECG" in modo:
                        titulo_caja = "💡 MANEJO Y RECOMENDACIONES"
                        instruccion_modo = 'Enfoque: Cardiología de Precisión. Te hemos enviado el trazado original, la señal aislada y un Slicing 3x4. LECTURA SISTEMÁTICA OBLIGATORIA: 1. Frecuencia. 2. Ritmo. 3. Eje eléctrico. 4. Intervalos matemáticos (PR, QRS, QT). 5. Segmento ST.'
                        html_extra = ""
                    elif "Farmacia" in modo:
                        titulo_caja = "💊 PAUTAS FARMACOLÓGICAS"
                        instruccion_modo = 'Enfoque: Farmacología. Analiza medicamentos, interacciones y dosis.'
                        html_extra = ""
                    else:
                        titulo_caja = "💡 PLAN INTEGRAL DE ACCIÓN"
                        instruccion_modo = 'Enfoque: Evaluación médica general.'
                        html_extra = ""

                    instruccion_bbox = ""
                    if mostrar_imagenes and (imagen_principal_para_marcar or primer_video_local):
                        instruccion_bbox = """
                        3. MAPA DE CALOR: Si identificas patología focal (nódulo, consolidación, infarto), añade al final: BBOX: [ymin, xmin, ymax, xmax] LABEL: Nombre. (El sistema renderizará un Heatmap Grad-CAM en esa zona).
                        Si es VIDEO añade también: TIMESTAMP: [segundos].
                        """
                    
                    instruccion_enfermeria = ""
                    if contexto in ["Hospitalización", "Residencia", "Urgencias", "UCI"]:
                        instruccion_enfermeria = "4. ROL DE ENFERMERÍA: Al estar el paciente en un entorno vigilado, INCLUYE OBLIGATORIAMENTE un apartado de '👩‍⚕️ Cuidados de Enfermería' detallando vigilancia y postura."

                    instruccion_nlp_riesgo = """
                        5. INSTRUCCIÓN LABORATORIO: Busca niveles de Albúmina (g/dL), Hemoglobina Glicosilada (%) y el Índice Tobillo-Brazo (ITB). Añade al final:
                        LAB_ALBUMIN: [valor] | LAB_HBA1C: [valor] | LAB_ITB: [valor] | RISK_MULTIPLIER: [decimal] LABEL: [motivo]
                        Para Radiómica, si es RX Tórax, añade también al final: CTR_RATIO: [valor calculado, ej: 0.52]
                    """

                    prompt = f"""
                    Rol: Especialista Clínico Experto (Motor Radiómico V88). Contexto: {contexto}. Modo: {modo}.
                    Zona Anatómica: {st.session_state.punto_cuerpo}. Notas del paciente: "{notas}"
                    
                    INPUTS ADICIONALES:
                    - PROTOCOLO: {txt_proto}
                    - DOCUMENTACIÓN: {txt_meds}
                    {datos_cv_texto}
                    
                    INSTRUCCIONES DE COMPORTAMIENTO:
                    1. Ve al grano usando las cajas.
                    2. {instruccion_modo}
                    {instruccion_bbox}
                    {instruccion_enfermeria}
                    {instruccion_nlp_riesgo}
                    
                    FORMATO HTML REQUERIDO:
                    <div class="diagnosis-box"><b>🚨 DIAGNÓSTICO / HALLAZGOS:</b><br>[Descripción clínica detallada]</div>
                    <div class="action-box"><b>⚡ ACCIÓN INMEDIATA:</b><br>[Pasos urgentes a tomar]</div>
                    <div class="material-box"><b>{titulo_caja}:</b><br>[Recomendaciones/Cuidados enfermería]</div>
                    {html_extra}
                    """

                    resp = model.generate_content([prompt, *con] if con else prompt)
                    texto_generado = resp.text
                    
                    # Extracción variables
                    patron_riesgo = r'RISK_MULTIPLIER:\s*([\d\.]+)\s*LABEL:\s*([^\n<]+)'
                    if match_riesgo := re.search(patron_riesgo, texto_generado):
                        try: st.session_state.patient_risk_factor = float(match_riesgo.group(1)); st.session_state.patient_risk_reason = match_riesgo.group(2).strip(); texto_generado = re.sub(patron_riesgo, '', texto_generado)
                        except: pass
                        
                    if match_alb := re.search(r'LAB_ALBUMIN:\s*([\d\.,]+)', texto_generado):
                        try: st.session_state.lab_albumin = float(match_alb.group(1).replace(',','.')); texto_generado = re.sub(r'LAB_ALBUMIN:\s*([\d\.,]+)', '', texto_generado)
                        except: pass

                    if match_hba1c := re.search(r'LAB_HBA1C:\s*([\d\.,]+)', texto_generado):
                        try: st.session_state.lab_hba1c = float(match_hba1c.group(1).replace(',','.')); texto_generado = re.sub(r'LAB_HBA1C:\s*([\d\.,]+)', '', texto_generado)
                        except: pass
                        
                    if match_itb := re.search(r'LAB_ITB:\s*([\d\.,]+)', texto_generado):
                        try: st.session_state.lab_itb = float(match_itb.group(1).replace(',','.')); texto_generado = re.sub(r'LAB_ITB:\s*([\d\.,]+)', '', texto_generado)
                        except: pass
                        
                    # Extraer CTR para Radiómica
                    if match_ctr := re.search(r'CTR_RATIO:\s*([\d\.,]+)', texto_generado):
                        val_ctr = match_ctr.group(1)
                        caja_radiomica = f'<div class="radiomics-box"><b>📐 Radiómica Cuantitativa:</b><br>Índice Cardiotorácico calculado: {val_ctr} (Límite normal < 0.50).</div>'
                        texto_generado = texto_generado.replace('<div class="diagnosis-box">', caja_radiomica + '\n<div class="diagnosis-box">')
                        texto_generado = re.sub(r'CTR_RATIO:\s*([\d\.,]+)', '', texto_generado)
                    
                    if mostrar_imagenes and (imagen_principal_para_marcar or primer_video_local):
                        img_marcada, texto_generado, detectado = extraer_y_dibujar_bboxes(texto_generado, imagen_principal_para_marcar, primer_video_local)
                        if detectado:
                            st.session_state.img_marcada = img_marcada 

                    st.session_state.resultado_analisis = texto_generado.strip()
                    
                    new_entry = { "id": str(uuid.uuid4()), "date": datetime.datetime.now().strftime("%d/%m %H:%M"), "mode": modo, "note": nota_historial if nota_historial else "Sin etiqueta", "result": texto_generado }
                    st.session_state.history_db.append(new_entry)
                    
                    if "Heridas" in modo and st.session_state.area_herida > 0:
                        p_manual = 0.0
                        if 'p_m' in locals(): p_manual = p_m
                        st.session_state.historial_evolucion.append({"Fecha": datetime.datetime.now().strftime("%d/%m"), "Area": st.session_state.area_herida, "Profundidad": p_manual})
                    
                    if st.session_state.resultado_analisis:
                        st.session_state.pdf_bytes = create_pdf(st.session_state.resultado_analisis)

                    if mostrar_imagenes and galeria_avanzada:
                        st.markdown("##### 🔬 Análisis de Capas Múltiples")
                        cols = st.columns(len(galeria_avanzada))
                        for i, (titulo, img) in enumerate(galeria_avanzada):
                            with cols[i]:
                                st.image(img, caption=titulo, use_container_width=True)

                except Exception as e: st.error(f"Error: {e}")
                
                finally:
                    for vp in video_paths_local:
                        try: os.remove(vp)
                        except: pass

        if st.session_state.resultado_analisis:
            st.markdown(st.session_state.resultado_analisis.replace("```html","").replace("```",""), unsafe_allow_html=True)
            
            st.markdown("---")
            with st.expander("💬 Chat Asistente", expanded=False):
                for m in st.session_state.chat_messages:
                    with st.chat_message(m["role"]): st.markdown(m["content"])
                
                if p := st.chat_input("Escribe tu duda sobre este caso..."):
                    st.session_state.chat_messages.append({"role": "user", "content": p})
                    with st.chat_message("user"): st.markdown(p)
                    with st.chat_message("assistant"):
                        try:
                            chat = genai.GenerativeModel("models/gemini-3-flash-preview")
                            r = chat.generate_content(f"CTX:{st.session_state.resultado_analisis}\nQ:{p}")
                            st.markdown(r.text)
                            st.session_state.chat_messages.append({"role": "assistant", "content": r.text})
                        except: st.error("Error chat")

        if st.session_state.pdf_bytes:
            st.download_button("📥 Descargar Informe Clínico PDF", st.session_state.pdf_bytes, "informe.pdf", "application/pdf")

    with tab_historial:
        if not st.session_state.history_db: st.info("Vacío.")
        else:
            if st.button("🗑️ Vaciar Historial"): st.session_state.history_db=[]; st.rerun()
            for item in reversed(st.session_state.history_db):
                with st.expander(f"📅 {item['date']} | {item['note']}"): st.markdown(item['result'], unsafe_allow_html=True)

# --- COLUMNA 3 (VISOR CLÍNICO UNIVERSAL) ---
with col_right:
    if "Heridas" in modo or "Dermatología" in modo:
        with st.expander("🔮 Visor y Predicción 3D", expanded=True):
            if len(st.session_state.historial_evolucion) > 0:
                pred_text = predecir_cierre_inteligente()
                st.markdown(pred_text, unsafe_allow_html=True)
            
            if st.session_state.img_marcada:
                st.markdown("#### 🎯 Mapa de Calor (Saliency)")
                st.image(st.session_state.img_marcada, caption="Concentración térmica IA", use_container_width=True)
                
            if st.session_state.img_ghost:
                st.markdown("#### 👻 Ghost Mode (Alineación)")
                st.image(st.session_state.img_ghost, caption="Mezcla: Previo (60%) + Actual (40%)", use_container_width=True)
            
            if st.session_state.img_previo and st.session_state.img_actual:
                st.markdown("---")
                c_prev, c_curr = st.columns(2)
                with c_prev: st.image(st.session_state.img_previo, caption="Inicio", use_container_width=True)
                with c_curr: st.image(st.session_state.img_actual, caption="Actual (Calibrada)", use_container_width=True)
            elif st.session_state.img_actual:
                st.image(st.session_state.img_actual, caption="Estado Actual (Color Calibrado)", use_container_width=True)
            else:
                if len(st.session_state.historial_evolucion) == 0:
                    st.caption("Analiza una herida para ver la evolución visual.")
    else:
        with st.expander("🔬 Visor Clínico Cuantitativo", expanded=True):
            # Renderizado de vector ECG si existe
            if st.session_state.get("ecg_vector_data") is not None:
                st.markdown("#### 📊 Vectorización 1D (Onda de Voltaje)")
                st.line_chart(st.session_state.ecg_vector_data, use_container_width=True)
                st.caption("Gráfico interactivo extraído píxel a píxel del trazado original.")
                
            if st.session_state.get("img_marcada"):
                st.markdown("#### 🎯 Mapa de Calor (Grad-CAM)")
                st.image(st.session_state.img_marcada, caption="Foco Patológico Detectado", use_container_width=True)
            elif st.session_state.get("img_actual"):
                st.markdown("#### 📸 Imagen Original")
                st.image(st.session_state.img_actual, use_container_width=True)
            else:
                st.info("Sube un estudio (Imagen o Vídeo) y marca las opciones visuales para activar la Radiómica y los Mapas de Calor Clínicos.")

st.divider()
if st.button("🔒 Cerrar Sesión"):
    cookie_manager.delete("labmind_secret_key")
    st.session_state.autenticado = False; st.rerun()
