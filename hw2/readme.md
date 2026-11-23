# Homework 2 — Image Classification with Transfer Learning

Progetto di **Computer Vision (Fall 2026)** per classificare le **37 razze** del dataset [Oxford-IIIT Pet](https://www.robots.ox.ac.uk/~vgg/data/pets/) (25 razze canine e 12 feline), utilizzando **PyTorch**, **MobileNetV3-Small** e **scikit-learn**. Il target è la razza (`target_types="category"`), **non** la classificazione binaria gatto/cane.

Il progetto confronta tre strategie basate sulla stessa rete `torchvision.models.mobilenet_v3_small`, inizializzata con `MobileNet_V3_Small_Weights.DEFAULT` (ImageNet-1K):

1. **Full fine-tuning:** tutti i pesi della rete vengono aggiornati.
2. **Classifier-head training:** `model.features` è congelato; vengono addestrati soltanto i layer lineari `classifier[0]` e `classifier[3]`.
3. **Frozen feature extraction + scikit-learn:** si estraggono vettori da **576 feature**, dopo l'average pooling e prima del classificatore, e si confrontano **Logistic Regression**, **Linear SVM** e **Random Forest**.

## 1. Ambiente sperimentale

Le metriche presentate nel report sono state ottenute nell'**esecuzione originale su macOS / Apple Silicon**, con PyTorch su **MPS** e i classificatori scikit-learn sulla **CPU**.

| Componente | Versione registrata nell'esecuzione originale |
| --- | --- |
| Python | `3.14.4` |
| PyTorch | `2.15.0` |
| TorchVision | `0.29.2` |
| scikit-learn | `1.9.1` |
| NumPy | `2.5.5` |
| Backend PyTorch | `mps` |
| Seed | `42` |

Altre dipendenze utilizzate: `pandas`, `matplotlib` e le dipendenze transitive delle librerie sopra. Le versioni di queste librerie non sono state riportate nell'output originale: **non sono dichiarate qui come versioni testate**. Utilizzare un `requirements.txt` verificato nell'ambiente effettivo (ad esempio generato con `python -m pip freeze > requirements.txt`).

> **Nota sulle piattaforme:** il codice seleziona automaticamente `cuda` (GPU NVIDIA), `mps` (Apple Silicon) oppure `cpu`. Cambiando sistema operativo, backend o versioni possono variare tempi e, in misura minore, le metriche. Per riprodurre il più fedelmente possibile i risultati del report è preferibile l'ambiente macOS/MPS originale.

## 2. Installazione

Clonare il repository e aprire un terminale nella cartella del progetto.

**macOS (Terminale o VS Code)**

```bash
python3 -m venv venv
source venv/bin/activate
python -m pip install -r requirements.txt
```

**Windows (PowerShell, ad esempio da VS Code)**

```powershell
py -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Se PowerShell blocca l'attivazione degli script, per la **sola sessione corrente** eseguire `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`, quindi ripetere il comando di attivazione.

Verificare l'ambiente:

```bash
python -c "import torch, torchvision, sklearn, numpy, pandas, matplotlib; print('PyTorch:', torch.__version__); print('TorchVision:', torchvision.__version__); print('scikit-learn:', sklearn.__version__); print('CUDA:', torch.cuda.is_available()); print('MPS:', torch.backends.mps.is_available())"
```

Non è necessario installare `tabulate`: lo script verificato non lo importa.

## 3. Dataset, split e preprocessing

Il dataset viene scaricato automaticamente da TorchVision durante la prima esecuzione (`download=True`) nella cartella `data/`. È richiesta una connessione Internet per il primo download e per scaricare i pesi ImageNet se non sono già in cache.

- **Split ufficiali:** `trainval = 3.680` immagini; `test = 3.669` immagini.
- **Suddivisione trainval stratificata per razza, seed 42:** `train = 2.944`; `validation = 736`.
- **Test ufficiale:** mantenuto separato dalla selezione dei checkpoint e degli iperparametri.
- **Conteggi per razza:** `results/class_counts.csv` (37 classi, colonne train/val/test).
- **Bilanciamento:** nessun oversampling, undersampling o class weighting; le razze sono quasi bilanciate.

**Validation/test:** preprocessing definito da `MobileNet_V3_Small_Weights.DEFAULT.transforms()` (resize a 256, center crop a 224 × 224, normalizzazione ImageNet con media `(0.485, 0.456, 0.406)` e deviazione standard `(0.229, 0.224, 0.225)`).

**Train, soltanto per le reti neurali:** `RandomResizedCrop(224, scale=(0.6, 1.0))`, `RandomHorizontalFlip`, `ColorJitter` (`brightness=contrast=saturation=0.2`), conversione in tensore e normalizzazione ImageNet. Le feature per i classificatori scikit-learn vengono invece estratte dalle immagini di training **senza augmentation**.

### Fonte, licenza e redistribuzione

- Dataset e istruzioni di download: [Oxford Visual Geometry Group — Oxford-IIIT Pet](https://www.robots.ox.ac.uk/~vgg/data/pets/).
- Pubblicazione: Parkhi, Vedaldi, Zisserman e Jawahar, *Cats and Dogs*, CVPR 2012.
- Il sito ufficiale indica la licenza [Creative Commons Attribution-ShareAlike 4.0 (CC BY-SA 4.0)](https://creativecommons.org/licenses/by-sa/4.0/) e segnala che il copyright delle singole immagini rimane ai rispettivi proprietari.
- **Non ridistribuire le immagini nel repository**: il codice le scarica tramite TorchVision.

## 4. Configurazione degli esperimenti

| Impostazione | Full fine-tuning | Head training | Feature extraction + scikit-learn |
| --- | --- | --- | --- |
| Backbone | MobileNetV3-Small | MobileNetV3-Small congelato | MobileNetV3-Small congelato |
| Parte adattata | Tutta la rete | `classifier[0]`, `classifier[3]` | Classificatore scikit-learn |
| Output | 37 classi | 37 classi | 37 classi |
| Loss | Cross-entropy | Cross-entropy | Obiettivo proprio di ciascun classificatore |
| Ottimizzatore | AdamW | AdamW | Implementazione scikit-learn |
| Learning rate | `3e-4` | `1e-3` | Non applicabile |
| Weight decay | `1e-4` | `1e-4` | Non applicabile |
| Batch size | 64 | 64 | 64 per estrazione feature |
| Epoche | 15 | 15 | Non applicabile |
| Scheduler | CosineAnnealingLR, `T_max=15` | CosineAnnealingLR, `T_max=15` | Non applicabile |

Nel **classifier-head training** il backbone congelato è mantenuto in modalità `eval()` durante l'addestramento per non aggiornare le statistiche BatchNorm.

**Regola di selezione per i modelli neurali:** massima accuracy sul validation set; in caso di parità, minore validation loss. Nell'esecuzione originale: **epoca 11** per full fine-tuning e **epoca 12** per head training. I checkpoint selezionati sono salvati in `results/full_best.pt` e `results/head_best.pt`.

**Griglie scikit-learn (selezione sulla validation accuracy):**

| Classificatore | Parametro | Valori provati | Selezionato nell'esecuzione originale |
| --- | --- | --- | --- |
| Logistic Regression | `C` | `0.01`, `0.1`, `1`, `10` | `0.01` |
| LinearSVC | `C` | `0.001`, `0.01`, `0.1`, `1` | `0.01` |
| Random Forest | `n_estimators` | `100`, `300`, `500` | `300` |

Per questi classificatori, `StandardScaler` è inserito in una `Pipeline` e viene adattato **solo sulle feature di training**. In caso di parità dell'accuracy viene mantenuto il primo candidato della griglia. Gli iperparametri PyTorch (learning rate, weight decay, epoche, batch size) sono fissati a priori, senza ricerca sistematica.

## 5. Esecuzione e riproduzione

Lo script aggiornato disponibile è `main_corrected.py`. **Se nel repository lo rinomini in `main.py`, utilizza il secondo comando.**

```bash
python main_corrected.py
# oppure, se il file è stato rinominato:
python main.py
```

Un solo comando esegue tutte le strategie: scarica dati e pesi se necessario, crea lo split stratificato, addestra le due varianti neurali, seleziona i checkpoint, estrae le feature congelate, prova le configurazioni scikit-learn, valuta i cinque modelli sul test e genera i file di risultati e le figure.

**Non utilizzare il test set per scegliere iperparametri o checkpoint.** Nello script aggiornato il test è caricato nella sezione finale, dopo la selezione dei modelli.

L'esecuzione completa può richiedere tempo; la durata dipende dall'hardware. Le directory `results/` e `splits/` sono create automaticamente.

### Riproducibilità

- Seed `42` per `random`, NumPy e PyTorch; generatore con seed fisso per il `DataLoader` di training.
- `NUM_WORKERS = 0`.
- `torch.use_deterministic_algorithms(True, warn_only=True)` e impostazioni cuDNN per favorire il determinismo; **`warn_only=True` non garantisce determinismo assoluto**.
- Il backend MPS e le diverse versioni delle dipendenze possono causare variazioni numeriche.
- Lo script registra versione delle librerie, dispositivo e seed in `results/metrics.json`.
- Il file `splits/split_seed42.json` contiene gli indici dello split per la riproduzione e deve essere conservato nel repository.

## 6. File generati

```text
.
├── main_corrected.py             # oppure main.py, se rinominato
├── README.md
├── requirements.txt
├── splits/
│   └── split_seed42.json
├── results/
│   ├── metrics.json
│   ├── results_table.csv
│   ├── class_counts.csv
│   ├── top_confusions.csv
│   ├── full_history.csv
│   ├── head_history.csv
│   ├── sklearn_search.csv
│   ├── full_best.pt
│   ├── head_best.pt
│   └── figures/
│       ├── learning_curves.png
│       ├── cm_full.png
│       ├── cm_head.png
│       ├── cm_sk_logreg.png
│       ├── cm_sk_linsvm.png
│       ├── cm_sk_rf.png
│       ├── examples_full.png
│       ├── examples_head.png
│       ├── examples_sk_logreg.png
│       ├── examples_sk_linsvm.png
│       └── examples_sk_rf.png
└── data/                         # scaricato automaticamente; NON committare
```

I checkpoint `.pt` possono essere inclusi nel repository **se dimensione e condizioni di licenza lo consentono**. Le cartelle `data/` e `venv/` non devono essere caricate.

### Metriche e tempi

Per ogni modello si riportano **accuracy train/validation/test**, **macro-F1 sul test**, **matrice di confusione con nomi delle 37 classi**, esempi di predizioni corrette/errate e indicatori di costo computazionale.

Il codice aggiornato distingue in `results/results_table.csv` e `results/metrics.json`:

- Tempo di training delle reti neurali, comprensivo di validation e checkpointing durante le 15 epoche.
- Tempo di estrazione delle feature **train + validation**.
- Tempo di fit del **solo** classificatore scikit-learn selezionato.
- Somma dei tempi di fit di **tutti** i candidati provati nella ricerca degli iperparametri.
- Tempo di estrazione delle feature del test, misurato separatamente.

Per Logistic Regression e Linear SVM la complessità viene riportata come numero di coefficienti e intercette. Per Random Forest è riportato separatamente il numero di nodi degli alberi, **non equivalente ai parametri trainabili di una rete neurale**.

## 7. Risultati del report originale (macOS / MPS)

Questi sono i risultati registrati con `main.py`** sul Mac.

| Approccio | Test accuracy | Test macro-F1 | Tempo riportato (s) |
| --- | ---: | ---: | ---: |
| Full fine-tuning | 0.8283 | 0.8245 | 281.0 |
| Head training | **0.8700** | **0.8687** | 165.6 |
| Features + Logistic Regression | 0.8569 | 0.8551 | 11.5 |
| Features + Linear SVM | 0.8335 | 0.8309 | 17.2 |
| Features + Random Forest | 0.8307 | 0.8255 | 12.9 |

**Nota metodologica sui tempi originali:** per le reti neurali i tempi comprendono le 15 epoche con le valutazioni di validation, ma non la valutazione finale sul test. Per i classificatori tradizionali comprendono l'estrazione delle feature di train, validation **e test**, più il fit del solo candidato selezionato; **non** l'intera ricerca degli iperparametri. La parte PyTorch usa MPS, mentre scikit-learn usa CPU: non si tratta di un benchmark hardware perfettamente omogeneo.

Una nuova esecuzione di `main_corrected.py` genera metriche proprie e tempi con definizioni differenti. **Non mescolare figure, checkpoint o metriche di esecuzioni diverse** senza documentare la provenienza. Se nel repository si mantengono gli output originali del Mac, indicarli come tali e conservarli coerentemente con il report.

## 8. Riferimenti

1. Parkhi, O. M., Vedaldi, A., Zisserman, A., & Jawahar, C. V. (2012). *Cats and Dogs*. IEEE Conference on Computer Vision and Pattern Recognition (CVPR).
2. Oxford Visual Geometry Group. [Oxford-IIIT Pet Dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/).
3. Howard, A., Sandler, M., Chen, B., et al. (2019). *Searching for MobileNetV3*. IEEE/CVF International Conference on Computer Vision (ICCV).
4. TorchVision. [MobileNet V3 Small](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.mobilenet_v3_small.html).
5. Creative Commons. [Attribution-ShareAlike 4.0 International](https://creativecommons.org/licenses/by-sa/4.0/).
