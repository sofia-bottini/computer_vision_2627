import cv2
import numpy as np
from pathlib import Path

# Parametri della degradazione
GAIN = 0.6
BIAS = 30
NOISE_SIGMA = 5
RANDOM_SEED = 42 # Per ottenere sempre lo stesso risultato, utile per la riproducibilità

# Cartelle del progetto
BASE_DIR = Path(__file__).resolve().parent
INPUT_DIR = BASE_DIR / "images" / "original"
OUTPUT_DIR = BASE_DIR / "images" / "degraded"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Fotografie originali
images = ["mare.jpeg", "alba.jpeg", "acquario.jpeg"]

# Generatore casuale con seed fisso
rng = np.random.default_rng(RANDOM_SEED)

for filename in images:

    # Caricamento dell'immagine
    image_path = INPUT_DIR / filename
    image = cv2.imread(str(image_path))

    if image is None:
        raise FileNotFoundError(
            f"Impossibile leggere: {image_path}"
        )

    # Conversione a float
    image_float = image.astype(np.float32)

    # Riduzione del contrasto (Bias and Gain)
    low_contrast = GAIN * image_float + BIAS

    # Rumore gaussiano
    noise = rng.normal(
        loc=0, # voglio aggiungere disturbo sneza modificare la luce
        scale=NOISE_SIGMA,
        size=image.shape
    )

    # Aggiunta del rumore
    degraded = low_contrast + noise

    # Clipping e conversione a uint8
    degraded = np.clip(degraded, 0, 255)
    degraded = degraded.astype(np.uint8)

    # Salvataggio PNG senza perdita
    output_path = OUTPUT_DIR / (
        Path(filename).stem + ".png"
    )

    if not cv2.imwrite(str(output_path), degraded):
        raise IOError(f"Salvataggio fallito: {output_path}")

    print(f"Creata: {output_path.name}")

print("Degradazione completata!")
