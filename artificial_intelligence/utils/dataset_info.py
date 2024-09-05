import os
import cv2
import json
import matplotlib.pyplot as plt
import numpy as np

with open('/home/alejo/repos/LSARecognitionPI/server/labels.json') as json_file:
    sign_map = json.load(json_file)

# Ruta de la carpeta donde se encuentran los videos
folder_path = "/home/alejo/Downloads/lsa_db"

# Diccionarios para almacenar la información
fps_dict = {}
frames_dict = {}

# Obtener la lista de archivos
video_files = [f for f in os.listdir(folder_path) if f.endswith(".mp4")]
total_videos = len(video_files)

# Recorremos los archivos en la carpeta
for idx, filename in enumerate(video_files, start=1):
    # Mostrar progreso
    print(f"Procesando video {idx}/{total_videos}: {filename}")

    # Extraer el ID de la seña del nombre del archivo
    sign_id = filename.split("_")[0]
    
    # Obtener la ruta completa del video
    video_path = os.path.join(folder_path, filename)
    
    # Leer el video
    video = cv2.VideoCapture(video_path)
    
    # Obtener el Frame Rate y la cantidad de frames
    fps = video.get(cv2.CAP_PROP_FPS)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Añadir los valores al diccionario
    if sign_id not in fps_dict:
        fps_dict[sign_id] = []
        frames_dict[sign_id] = []
    
    fps_dict[sign_id].append(fps)
    frames_dict[sign_id].append(total_frames)

# Calcular los promedios
avg_fps = {k: np.average(v) for k, v in fps_dict.items()}
avg_frames = {k: np.average(v) for k, v in frames_dict.items()}

# Ordenar por ID de seña
avg_fps = dict(sorted(avg_fps.items()))
avg_frames = dict(sorted(avg_frames.items()))

# Graficar los datos
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 8))

avg_fps = {sign_map[str(int(k))]: v for k, v in avg_fps.items()}
avg_frames = {sign_map[str(int(k))]: v for k, v in avg_frames.items()}

# Gráfico de promedio de FPS
ax1.bar(avg_fps.keys(), avg_fps.values(), color='skyblue')
ax1.axhline(np.average(list(avg_fps.values())), color='red', linestyle='--', label=f'Total avg. FPS: {np.mean(list(avg_fps.values())):.2f}')
ax1.set_title('Signs - Average FPS')
ax1.set_ylabel('Average FPS')
ax1.legend()
ax1.set_xticklabels(avg_fps.keys(), rotation=90)

# Gráfico de promedio de cantidad de frames
ax2.bar(avg_frames.keys(), avg_frames.values(), color='lightgreen')
ax2.axhline(np.average(list(avg_frames.values())), color='red', linestyle='--', label=f'Total avg. frames: {np.mean(list(avg_frames.values())):.0f}')
ax2.set_title('Signs - Average Frames')
ax2.set_ylabel('Average frames')
ax2.legend()
ax2.set_xticklabels(avg_fps.keys(), rotation=90)

plt.tight_layout()

# Guardar el gráfico en un archivo en lugar de mostrarlo
output_path = os.path.join(folder_path, "average_signs.png")
plt.savefig(output_path)
print(f"Gráfico guardado en: {output_path}")