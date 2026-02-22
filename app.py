def extraer_y_dibujar_bboxes(texto, img_pil):
    patron = r'BBOX:\s*\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]\s*LABEL:\s*([^\n<]+)'
    matches = re.findall(patron, texto)
    if not matches: return None, texto, False
    
    # --- NOVEDAD: Filtro inteligente para borrar BBOX duplicados ---
    # Si la IA escupe las mismas coordenadas 2 veces, solo nos quedamos con 1
    matches_unicos = list({m[:4]: m for m in matches}.values())
    
    img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    h, w = img_cv.shape[:2]
    for m in matches_unicos:
        try:
            y1, x1, y2, x2 = [int(int(c)*dim/1000) for c, dim in zip(m[:4], [h, w, h, w])]
            cv2.rectangle(img_cv, (x1, y1), (x2, y2), (0, 0, 255), 3)
            cv2.putText(img_cv, m[4].upper(), (x1, max(0, y1-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
        except: pass
    
    # Borramos los textos BBOX del informe para que el médico no vea números feos
    texto_limpio = re.sub(patron, '', texto).strip()
    return Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)), texto_limpio, True
