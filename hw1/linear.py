
import cv2
import matplotlib.pyplot as plt
from pathlib import Path

from enhancement import linear_contrast_stretching


# Cartella principale del progetto
BASE_DIR = Path(__file__).resolve().parent

# Cartelle delle immagini
ORIGINAL_DIR = BASE_DIR / "images" / "original"
DEGRADED_DIR = BASE_DIR / "images" / "degraded"

# Cartelle dei risultati
PROCESSED_DIR = BASE_DIR / "results" / "processed"
COMPARISONS_DIR = BASE_DIR / "results" / "comparisons"

# Creazione delle cartelle, se non esistono
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
COMPARISONS_DIR.mkdir(parents=True, exist_ok=True)

# Elenco delle immagini da elaborare
images = ["mare", "alba", "acquario"]

for name in images:

    # 1. Caricamento dell'immagine originale
    original_path = ORIGINAL_DIR / f"{name}.jpeg"
    original = cv2.imread(str(original_path))

    # 2. Caricamento dell'immagine degradata
    degraded_path = DEGRADED_DIR / f"{name}.png"
    degraded = cv2.imread(str(degraded_path))

    # Controllo caricamento
    if original is None or degraded is None:
        raise FileNotFoundError(
            f"Impossibile caricare le immagini di {name}"
        )

    # 3. Applicazione del Linear Contrast Stretching
    enhanced = linear_contrast_stretching(degraded)

    # 4. Salvataggio dell'immagine elaborata
    output_path = PROCESSED_DIR / f"{name}_linear.png"

    if not cv2.imwrite(str(output_path), enhanced):
        raise IOError(f"Errore salvataggio: {output_path}")

    # 5. Preparazione del confronto visivo
    comparison_images = [original, degraded, enhanced]

    titles = [ "Original", "Degraded", "Linear Contrast Stretching"]

    # Creazione della figura
    fig, axes = plt.subplots( 1, 3, figsize=(15, 5))

    for ax, image, title in zip(axes, comparison_images, titles):

        # OpenCV usa BGR, Matplotlib usa RGB
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        ax.imshow(rgb)
        ax.set_title(title)
        ax.axis("off")

    fig.suptitle(f"Image Enhancement - {name.capitalize()}", fontsize=14)

    plt.tight_layout()

    # 6. Salvataggio del confronto
    comparison_path = (COMPARISONS_DIR / f"{name}_linear_comparison.png")

    plt.savefig(comparison_path, dpi=150, bbox_inches="tight")

    # 7. Visualizzazione
    plt.show()
    plt.close(fig)

    print(f"Risultato salvato: {output_path.name}")
    print(f"Confronto salvato: {comparison_path.name}")

print("Elaborazione completata!")
