import json
import matplotlib.pyplot as plt

def plot_cam_performance_results(var):
    with open("cam_performance_results.json", "r") as f:
        data = json.load(f)

    configs = []
    elapsed_time = []
    real_fps = []

    total_frames = None

    for key, value in data.items():
        configs.append(key)
        elapsed_time.append(value["elapsed_time"])
        real_fps.append(value["real_fps"])
        total_frames = value["frames_read"]

    if var=='elapsed_time':
        bars = plt.bar(configs, elapsed_time, label="Time [secs]", align='edge', width=0.8)
    elif var=='real_fps':
        bars = plt.bar(configs, real_fps, label="Real FPS", align='edge', width=0.8)

    plt.xlabel("Configuration")
    plt.ylabel("Value")
    plt.title(f"Cam Configurations Performances on {total_frames} frames readed")
    plt.legend(loc='best', bbox_to_anchor=(1, 0.5),  fontsize='medium')
    plt.xticks(rotation=80)
    plt.xticks(fontsize=9)

    # Agrega esta parte para mostrar los valores encima de las barras:
    for bar in bars:
        height = bar.get_height()  # Obtener la altura de la barra
        plt.annotate(
            f"{int(height)}",
            xy=(bar.get_x() + bar.get_width() / 2, height),  # Centrar la anotación
            xytext=(0, 3),  # Desplazar la anotación un poco hacia arriba
            textcoords="offset points",
            ha='center',  # Alinear el texto al centro
            va='bottom',  # Alinear el texto a la parte inferior de la anotación
        )

    plt.show()

plot_cam_performance_results(var='real_fps')