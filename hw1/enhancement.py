import cv2
import numpy as np
from skimage.metrics import (peak_signal_noise_ratio, structural_similarity)


def linear_contrast_stretching(image):
    """
    Aumenta il contrasto espandendo i valori
    della componente V dello spazio colore HSV.
    """

    # Conversione da BGR a HSV perché voglio lavorare solo su value senza toccare hue e saturation
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Estrazione della componente V
    h, s, v = cv2.split(hsv)

    # Conversione a float per i calcoli
    v_float = v.astype(np.float32)

    # Intensità minima e massima
    v_min = np.min(v_float)
    v_max = np.max(v_float)

    # Evitiamo divisioni per zero
    if v_max == v_min:
        return image.copy()

    # Linear Contrast Stretching
    v_stretched = ((v_float - v_min) / (v_max - v_min)) * 255

    # Conversione a uint8
    v_stretched = np.clip(v_stretched, 0, 255).astype(np.uint8)

    # Ricomposizione HSV
    hsv_result = cv2.merge([h, s, v_stretched])

    # Ritorno allo spazio BGR
    result = cv2.cvtColor(hsv_result, cv2.COLOR_HSV2BGR)

    return result


def histogram_equalization(image):
    """
    Applica Histogram Equalization alla componente V
    dello spazio colore HSV.
    """

    # 1. Conversione da BGR a HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # 2. Separazione dei canali
    h, s, v = cv2.split(hsv)

    # 3. Equalizzazione della componente V
    v_equalized = cv2.equalizeHist(v)

    # 4. Ricomposizione dei canali HSV
    hsv_result = cv2.merge([h, s, v_equalized])

    # 5. Conversione da HSV a BGR
    result = cv2.cvtColor(hsv_result, cv2.COLOR_HSV2BGR)

    return result



def gamma_correction(image, gamma=2.2):
    """
    Gamma Correction sulla componente V in HSV.

    Formula:
    I_out = 255 * (I_in / 255) ** (1 / gamma)
    """

    if gamma <= 0:
        raise ValueError("Gamma deve essere maggiore di zero")

    # Conversione BGR -> HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Separazione dei canali
    h, s, v = cv2.split(hsv)

    # Conversione a float e normalizzazione [0, 1]
    v_float = v.astype(np.float32) / 255.0

    # Applicazione della formula
    v_gamma = np.power(v_float, 1.0 / gamma)

    # Ritorno nell'intervallo [0, 255]
    v_gamma = np.clip(v_gamma * 255.0, 0, 255).astype(np.uint8)

    # Ricomposizione dei canali
    hsv_result = cv2.merge([h, s, v_gamma])

    # Ritorno a BGR
    result = cv2.cvtColor(hsv_result,cv2.COLOR_HSV2BGR)

    return result



def calculate_metrics(original, processed):
    """
    Calcola PSNR e SSIM rispetto all'immagine originale.

    Le immagini devono avere le stesse dimensioni
    ed essere in formato uint8.
    """

    if original.shape != processed.shape:
        raise ValueError("Le immagini devono avere le stesse dimensioni")

    psnr = peak_signal_noise_ratio(original, processed, data_range=255)

    ssim = structural_similarity(original, processed, data_range=255, channel_axis=2)

    return psnr, ssim
