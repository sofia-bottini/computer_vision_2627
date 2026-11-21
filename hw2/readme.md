
# Homework 2: Image Classification with Transfer Learning

Pipeline riproducibile di classificazione immagini in PyTorch e scikit-learn sul dataset **Oxford-IIIT Pet** (37 razze di cani e gatti) utilizzando **MobileNetV3-Small** (`MobileNet_V3_Small_Weights.DEFAULT`, pre-addestrata su ImageNet-1K).

Il progetto confronta tre strategie di transfer learning:
1. **Full fine-tuning** (intera rete addestrabile)
2. **Classifier-head training** (`model.features` congelato, solo layer lineari del classificatore addestrabili)
3. **Feature extraction + classificatori tradizionali** (feature da 576 dimensioni estratte dopo `avgpool` + `StandardScaler` + Logistic Regression, Linear SVM e Random Forest)

## Ambiente e versioni testate

* **Python:** `3.14.4`
* **PyTorch:** `2.15.0`
* **TorchVision:** `0.29.2`
* **scikit-learn:** `1.9.1`
* **NumPy:** `2.5.5`
* **Hardware utilizzato:** Apple Silicon (`mps`)

## Installazione delle dipendenze

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install tabulate
