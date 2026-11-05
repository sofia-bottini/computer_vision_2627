import cv2
import csv
from pathlib import Path

from enhancement import calculate_metrics

# Cartella principale del progetto
BASE_DIR = Path(__file__).resolve().parent

# Cartelle delle immagini
ORIGINAL_DIR = BASE_DIR / "images" / "original"
DEGRADED_DIR = BASE_DIR / "images" / "degraded"
PROCESSED_DIR = BASE_DIR / "results" / "processed"

# Cartella dei risultati
RESULTS_DIR = BASE_DIR / "results"

# Immagini da elaborare
images = ["mare", "alba", "acquario"]

# Metodi da confrontare
methods = {"Baseline": None, "Linear Contrast Stretching": "_linear", "Histogram Equalization": "_histogram", "Gamma Correction": "_gamma" }

# Creare file CSV dei risultati
csv_path = RESULTS_DIR / "metrics.csv"

with open(csv_path, mode="w", newline="", encoding="utf-8") as file:

    writer = csv.writer(file)

    # Intestazione
    writer.writerow(["Image", "Method", "PSNR (dB)", "SSIM"])

    for name in images:

        # Immagine originale di riferimento
        original = cv2.imread(str(ORIGINAL_DIR / f"{name}.jpeg"))

        if original is None:
            raise FileNotFoundError(f"Originale non trovato: {name}")

        for method, suffix in methods.items():

            # Baseline: immagine degradata
            if suffix is None:
                image_path = DEGRADED_DIR / f"{name}.png"

            # Altri metodi: immagini elaborate
            else:
                image_path = (PROCESSED_DIR / f"{name}{suffix}.png")

            processed = cv2.imread(str(image_path))

            if processed is None:
                raise FileNotFoundError(f"Immagine non trovata: {image_path}")

            # Calcolo delle metriche
            psnr, ssim = calculate_metrics(original, processed)

            # Salvataggio nel CSV
            writer.writerow([
                name,
                method,
                round(float(psnr), 2),
                round(float(ssim), 4)
            ])

            # Visualizzazione nel terminale
            print(
                f"{name:10} | "
                f"{method:28} | "
                f"PSNR: {psnr:.2f} dB | "
                f"SSIM: {ssim:.4f}"
            )

print(f"\nRisultati salvati in: {csv_path}")
