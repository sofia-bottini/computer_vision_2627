import json
import os
import platform
import random
import time
from pathlib import Path

# Must be set before CUDA is initialized (has no effect on MPS).
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms


# 0. CONFIGURATION AND REPRODUCIBILITY

SEED = 42
BATCH_SIZE = 64
EPOCHS = 15
WEIGHT_DECAY = 1e-4
LR_FULL = 3e-4
LR_HEAD = 1e-3
NUM_WORKERS = 0
NUM_CLASSES = 37

DATA_DIR = Path("data")
RESULTS_DIR = Path("results")
FIGURES_DIR = RESULTS_DIR / "figures"
SPLITS_DIR = Path("splits")
for directory in (RESULTS_DIR, FIGURES_DIR, SPLITS_DIR):
    directory.mkdir(parents=True, exist_ok=True)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
# warn_only=True does NOT guarantee full determinism, especially on MPS.
torch.use_deterministic_algorithms(True, warn_only=True)

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

env = {
    "python": platform.python_version(),
    "torch": torch.__version__,
    "torchvision": torchvision.__version__,
    "sklearn": sklearn.__version__,
    "numpy": np.__version__,
    "device": str(device),
    "seed": SEED,
    "deterministic_algorithms_requested": True,
    "deterministic_warn_only": True,
    "cudnn_deterministic": True,
    "num_workers": NUM_WORKERS,
}
print(f"Using device: {device}")
print(f"Environment: {env}\n")



# 1. DATA: OFFICIAL TRAINVAL / TEST, STRATIFIED TRAIN / VALIDATION

weights = models.MobileNet_V3_Small_Weights.DEFAULT  # ImageNet-1K
# This evaluation transform belongs to the selected pretrained weights.
eval_transform = weights.transforms()
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.6, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=eval_transform.mean, std=eval_transform.std),
])

trainval_aug = datasets.OxfordIIITPet(
    root=str(DATA_DIR), split="trainval", target_types="category",
    download=True, transform=train_transform,
)
trainval_eval = datasets.OxfordIIITPet(
    root=str(DATA_DIR), split="trainval", target_types="category",
    download=True, transform=eval_transform,
)
# Do not construct/read the test dataset until final evaluation.
class_names = trainval_eval.classes
trainval_labels = np.asarray(trainval_eval._labels, dtype=int)
assert len(class_names) == NUM_CLASSES

indices = np.arange(len(trainval_labels))
train_idx, val_idx = train_test_split(
    indices, test_size=0.2, stratify=trainval_labels, random_state=SEED,
)
with (SPLITS_DIR / "split_seed42.json").open("w", encoding="utf-8") as f:
    json.dump({
        "seed": SEED,
        "train": sorted(train_idx.tolist()),
        "val": sorted(val_idx.tolist()),
    }, f, indent=2)

loader_generator = torch.Generator().manual_seed(SEED)
train_loader = DataLoader(
    Subset(trainval_aug, train_idx), batch_size=BATCH_SIZE, shuffle=True,
    generator=loader_generator, num_workers=NUM_WORKERS,
)
train_eval_loader = DataLoader(
    Subset(trainval_eval, train_idx), batch_size=BATCH_SIZE, shuffle=False,
    num_workers=NUM_WORKERS,
)
val_loader = DataLoader(
    Subset(trainval_eval, val_idx), batch_size=BATCH_SIZE, shuffle=False,
    num_workers=NUM_WORKERS,
)

print(f"Images: train={len(train_idx)}, val={len(val_idx)} (test held out)")



# 2. SHARED HELPERS

def build_model(freeze_backbone: bool) -> nn.Module:
    """ImageNet-pretrained MobileNetV3-Small with 37 output classes.

    Its classifier is Linear(576,1024), Hardswish, Dropout, Linear(1024,37).
    Head-only training updates classifier[0] and classifier[3].
    """
    torch.manual_seed(SEED)  # same initialization of the new layer
    model = models.mobilenet_v3_small(weights=weights)
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, NUM_CLASSES)
    if freeze_backbone:
        for param in model.features.parameters():
            param.requires_grad = False
    return model.to(device)


def count_trainable(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def evaluate(model: nn.Module, loader: DataLoader):
    """Evaluate deterministically (no augmentation): y_true, y_pred, avg CE loss."""
    model.eval()
    criterion = nn.CrossEntropyLoss()
    all_true, all_pred = [], []
    total_loss, count = 0.0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            batch_size = labels.size(0)
            total_loss += criterion(logits, labels).item() * batch_size
            count += batch_size
            all_true.append(labels.cpu())
            all_pred.append(logits.argmax(dim=1).cpu())
    return (
        torch.cat(all_true).numpy(),
        torch.cat(all_pred).numpy(),
        total_loss / count,
    )


def train_model(model: nn.Module, name: str, lr: float, freeze_backbone: bool):
    """Train 15 epochs; select highest validation accuracy (ties: lower loss).

    Time includes all training epochs, validation passes and checkpoint writes,
    but excludes final train/test evaluation.
    """
    torch.manual_seed(SEED)
    loader_generator.manual_seed(SEED)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=lr, weight_decay=WEIGHT_DECAY,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    checkpoint = RESULTS_DIR / f"{name}_best.pt"
    best_val_acc = -1.0
    best_val_loss = float("inf")
    best_epoch = None
    history = []
    start = time.perf_counter()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        if freeze_backbone:
            # Prevent the frozen feature extractor's BatchNorm running stats
            # from being modified by training batches.
            model.features.eval()
        running_loss, correct, n = 0.0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * labels.size(0)
            correct += (logits.argmax(1) == labels).sum().item()
            n += labels.size(0)
        scheduler.step()

        y_val, pred_val, val_loss = evaluate(model, val_loader)
        val_acc = accuracy_score(y_val, pred_val)
        train_loss, train_acc = running_loss / n, correct / n
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
        })
        print(
            f"[{name}] epoch {epoch:02d}/{EPOCHS} | "
            f"train loss {train_loss:.3f} acc {train_acc:.3f} | "
            f"val loss {val_loss:.3f} acc {val_acc:.3f}"
        )
        if (val_acc, -val_loss) > (best_val_acc, -best_val_loss):
            best_val_acc, best_val_loss, best_epoch = val_acc, val_loss, epoch
            torch.save(model.state_dict(), checkpoint)

    elapsed = time.perf_counter() - start
    # Only load checkpoints created by this same trusted script.
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    pd.DataFrame(history).to_csv(RESULTS_DIR / f"{name}_history.csv", index=False)
    print(
        f"[{name}] selected epoch {best_epoch}; val acc {best_val_acc:.4f}; "
        f"training + validation time {elapsed:.1f}s\n"
    )
    return history, elapsed, best_epoch


def extract_features(feature_extractor: nn.Module, loader: DataLoader):
    """Extract 576-dimensional fixed features after MobileNet's avgpool."""
    feature_extractor.eval()
    features, labels = [], []
    with torch.no_grad():
        for images, targets in loader:
            outputs = feature_extractor(images.to(device))
            features.append(torch.flatten(outputs, start_dim=1).cpu().numpy())
            labels.append(targets.numpy())
    return np.concatenate(features), np.concatenate(labels)


results = {}
histories = {}
nn_models = {}
sk_models = {}



# 3. APPROACH 1: FULL FINE-TUNING

model_full = build_model(freeze_backbone=False)
print(f"Full fine-tuning | trainable parameters: {count_trainable(model_full):,}")
hist, elapsed, best_epoch = train_model(model_full, "full", LR_FULL, False)
histories["full"] = hist
nn_models["full"] = model_full
results["full"] = {
    "approach": "Full fine-tuning",
    "trainable_params": count_trainable(model_full),
    "training_time_s": elapsed,
    "feature_extraction_time_s": None,
    "hyperparameter_search_time_s": None,
    "best_epoch": best_epoch,
    "hyperparams": (
        f"AdamW lr={LR_FULL} wd={WEIGHT_DECAY} bs={BATCH_SIZE} "
        f"epochs={EPOCHS} cosine schedule, CE loss"
    ),
}



# 4. APPROACH 2: CLASSIFIER-HEAD TRAINING

model_head = build_model(freeze_backbone=True)
print(f"Head training | trainable parameters: {count_trainable(model_head):,}")
hist, elapsed, best_epoch = train_model(model_head, "head", LR_HEAD, True)
histories["head"] = hist
nn_models["head"] = model_head
results["head"] = {
    "approach": "Head training",
    "trainable_params": count_trainable(model_head),
    "training_time_s": elapsed,
    "feature_extraction_time_s": None,
    "hyperparameter_search_time_s": None,
    "best_epoch": best_epoch,
    "hyperparams": (
        f"AdamW lr={LR_HEAD} wd={WEIGHT_DECAY} bs={BATCH_SIZE} "
        f"epochs={EPOCHS} cosine schedule, CE loss"
    ),
}

# Selected checkpoints evaluated on train/val with evaluation preprocessing.
# Train accuracy during epochs used random training augmentation; this does not.
for key, model in nn_models.items():
    y_train, pred_train, _ = evaluate(model, train_eval_loader)
    y_val, pred_val, _ = evaluate(model, val_loader)
    results[key]["train_acc"] = accuracy_score(y_train, pred_train)
    results[key]["val_acc"] = accuracy_score(y_val, pred_val)



# 5. APPROACH 3: FROZEN FEATURES + SCIKIT-LEARN CLASSIFIERS

backbone = models.mobilenet_v3_small(weights=weights).to(device)
feature_extractor = nn.Sequential(*list(backbone.children())[:-1]).to(device)
feature_extractor.eval()
for param in feature_extractor.parameters():
    param.requires_grad = False

feature_start = time.perf_counter()
X_train, y_train = extract_features(feature_extractor, train_eval_loader)
X_val, y_val = extract_features(feature_extractor, val_loader)
trainval_feature_time = time.perf_counter() - feature_start
print(
    f"Frozen features: train {X_train.shape}, val {X_val.shape}; "
    f"train+val extraction {trainval_feature_time:.1f}s\n"
)

sk_configs = {
    "logreg": (
        "Logistic regression", "C", [0.01, 0.1, 1, 10],
        lambda value: LogisticRegression(C=value, max_iter=3000),
    ),
    "linsvm": (
        "Linear SVM", "C", [0.001, 0.01, 0.1, 1],
        lambda value: LinearSVC(C=value, dual=False, max_iter=5000),
    ),
    "rf": (
        "Random forest", "n_estimators", [100, 300, 500],
        lambda value: RandomForestClassifier(
            n_estimators=value, n_jobs=-1, random_state=SEED,
        ),
    ),
}

search_records = []
for key, (classifier_name, param_name, candidates, make_classifier) in sk_configs.items():
    best = None
    total_search_time = 0.0
    for value in candidates:
        # StandardScaler is fitted only on X_train. RF also uses a scaler for
        # pipeline consistency, although tree models do not need scaling.
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", make_classifier(value)),
        ])
        start = time.perf_counter()
        pipe.fit(X_train, y_train)
        fit_time = time.perf_counter() - start
        total_search_time += fit_time
        val_acc = accuracy_score(y_val, pipe.predict(X_val))
        search_records.append({
            "classifier": classifier_name,
            "hyperparameter": param_name,
            "value": value,
            "val_acc": val_acc,
            "fit_time_s": fit_time,
        })
        print(f"[{key}] {param_name}={value}: val acc {val_acc:.4f} (fit {fit_time:.1f}s)")
        # Strict '>' keeps the first listed value on exact accuracy ties.
        if best is None or val_acc > best[0]:
            best = (val_acc, value, pipe, fit_time)

    val_acc, selected_value, selected_pipe, selected_fit_time = best
    clf = selected_pipe.named_steps["clf"]
    if hasattr(clf, "coef_"):
        n_coeff = int(clf.coef_.size + clf.intercept_.size)
        n_tree_nodes = None
    else:
        # Number of RF tree nodes is NOT comparable to NN trainable parameters.
        n_coeff = None
        n_tree_nodes = int(sum(tree.tree_.node_count for tree in clf.estimators_))

    model_key = "sk_" + key
    sk_models[model_key] = selected_pipe
    results[model_key] = {
        "approach": f"Features + {classifier_name}",
        "trainable_params": n_coeff,  # N/A for Random Forest
        "tree_nodes": n_tree_nodes,
        "training_time_s": selected_fit_time,
        "feature_extraction_time_s": trainval_feature_time,
        "hyperparameter_search_time_s": total_search_time,
        "selected_model_pipeline_time_s": trainval_feature_time + selected_fit_time,
        "full_search_pipeline_time_s": trainval_feature_time + total_search_time,
        "best_epoch": None,
        "train_acc": accuracy_score(y_train, selected_pipe.predict(X_train)),
        "val_acc": val_acc,
        "hyperparams": f"{param_name}={selected_value}; StandardScaler fitted on train only",
    }
    print(
        f"[{key}] selected {param_name}={selected_value}; "
        f"selected fit {selected_fit_time:.1f}s; all candidate fits {total_search_time:.1f}s\n"
    )

pd.DataFrame(search_records).to_csv(RESULTS_DIR / "sklearn_search.csv", index=False)



# 6. HELD-OUT TEST: USED ONLY AFTER ALL MODEL SELECTION IS COMPLETE

test_dataset = datasets.OxfordIIITPet(
    root=str(DATA_DIR), split="test", target_types="category",
    download=True, transform=eval_transform,
)
test_labels = np.asarray(test_dataset._labels, dtype=int)
assert class_names == test_dataset.classes, "Train and test class orders differ!"
test_loader = DataLoader(
    test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS,
)

counts = pd.DataFrame({
    "breed": class_names,
    "train": np.bincount(trainval_labels[train_idx], minlength=NUM_CLASSES),
    "val": np.bincount(trainval_labels[val_idx], minlength=NUM_CLASSES),
    "test": np.bincount(test_labels, minlength=NUM_CLASSES),
})
counts.to_csv(RESULTS_DIR / "class_counts.csv", index=False)
print(f"Final split sizes: train={len(train_idx)}, val={len(val_idx)}, test={len(test_labels)}")
print(counts[["train", "val", "test"]].describe().loc[["min", "max"]], "\n")

predictions = {}
for key, model in nn_models.items():
    y_true, y_pred, _ = evaluate(model, test_loader)
    predictions[key] = (y_true, y_pred)

# Frozen test features are extracted once, only for final evaluation.
test_extraction_start = time.perf_counter()
X_test, y_test = extract_features(feature_extractor, test_loader)
test_extraction_time = time.perf_counter() - test_extraction_start
print(f"Final test feature extraction: {X_test.shape}, {test_extraction_time:.1f}s")
for key, pipe in sk_models.items():
    predictions[key] = (y_test, pipe.predict(X_test))
    results[key]["test_feature_extraction_time_s"] = test_extraction_time

for key, (y_true, y_pred) in predictions.items():
    results[key]["test_acc"] = accuracy_score(y_true, y_pred)
    results[key]["test_macro_f1"] = f1_score(
        y_true, y_pred, labels=np.arange(NUM_CLASSES), average="macro", zero_division=0,
    )

rows = []
for result in results.values():
    rows.append({
        "Approach": result["approach"],
        "Trainable params / coefficients": result["trainable_params"],
        "RF tree nodes (not trainable params)": result.get("tree_nodes"),
        "Train/val feature extraction (s)": result["feature_extraction_time_s"],
        "Training / selected fit (s)": result["training_time_s"],
        "Hyperparameter search fits (s)": result["hyperparameter_search_time_s"],
        "Selected-model pipeline time (s)": result.get(
            "selected_model_pipeline_time_s", result["training_time_s"]
        ),
        "Full-search pipeline time (s)": result.get(
            "full_search_pipeline_time_s", result["training_time_s"]
        ),
        "Train acc": result["train_acc"],
        "Val acc": result["val_acc"],
        "Test acc": result["test_acc"],
        "Test macro-F1": result["test_macro_f1"],
        "Best epoch": result["best_epoch"],
        "Hyperparameters": result["hyperparams"],
    })
results_table = pd.DataFrame(rows)
results_table.to_csv(RESULTS_DIR / "results_table.csv", index=False)

with (RESULTS_DIR / "metrics.json").open("w", encoding="utf-8") as f:
    json.dump({
        "env": env,
        "dataset": {
            "name": "Oxford-IIIT Pet",
            "train_samples": len(train_idx),
            "val_samples": len(val_idx),
            "test_samples": len(test_labels),
            "num_classes": NUM_CLASSES,
            "split_seed": SEED,
        },
        "timing_notes": {
            "neural_training": "15 epochs including validation/checkpointing; excludes final evaluation",
            "sklearn_selected": "train+val frozen feature extraction plus one selected classifier fit",
            "sklearn_full_search": "train+val extraction plus all candidate fit times; excludes prediction times",
            "test_extraction": "measured separately and excluded from training time",
            "hardware": "PyTorch backbone on selected torch device; scikit-learn fitting on CPU",
        },
        "test_feature_extraction_time_s": test_extraction_time,
        "results": results,
    }, f, indent=2)

print("\nFINAL RESULTS (training times have different definitions; see metrics.json)")
print(results_table[[
    "Approach", "Trainable params / coefficients", "Training / selected fit (s)",
    "Selected-model pipeline time (s)", "Train acc", "Val acc",
    "Test acc", "Test macro-F1",
]].to_string(index=False, float_format=lambda x: f"{x:.4f}"))



# 7. FIGURES AND ERROR ANALYSIS

raw_test = datasets.OxfordIIITPet(
    root=str(DATA_DIR), split="test", target_types="category", download=False,
)
rng = np.random.RandomState(0)
confusions = []

for key, (y_true, y_pred) in predictions.items():
    title = results[key]["approach"]
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(NUM_CLASSES))
    row_totals = cm.sum(axis=1, keepdims=True)
    norm_cm = np.divide(
        cm, row_totals,
        out=np.zeros(cm.shape, dtype=float),
        where=row_totals != 0,
    )
    fig, ax = plt.subplots(figsize=(14, 12))
    image = ax.imshow(norm_cm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(NUM_CLASSES))
    ax.set_yticks(range(NUM_CLASSES))
    ax.set_xticklabels(class_names, rotation=90, fontsize=7)
    ax.set_yticklabels(class_names, fontsize=7)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"{title} — test confusion matrix (row normalized)")
    fig.colorbar(image, fraction=0.046, pad=0.02)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f"cm_{key}.png", dpi=150)
    plt.close(fig)

    k = 6
    correct_idx = rng.permutation(np.flatnonzero(y_true == y_pred))[:k]
    wrong_idx = rng.permutation(np.flatnonzero(y_true != y_pred))[:k]
    fig, axes = plt.subplots(2, k, figsize=(2.6 * k, 6))
    for row, (indices_for_row, tag, color) in enumerate([
        (correct_idx, "OK", "green"),
        (wrong_idx, "WRONG", "red"),
    ]):
        for col in range(k):
            ax = axes[row, col]
            ax.axis("off")
            if col < len(indices_for_row):
                idx = int(indices_for_row[col])
                ax.imshow(raw_test[idx][0])
                ax.set_title(
                    f"{tag}\ntrue: {class_names[y_true[idx]]}\n"
                    f"pred: {class_names[y_pred[idx]]}",
                    fontsize=7, color=color,
                )
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f"examples_{key}.png", dpi=150)
    plt.close(fig)

    off_diagonal = cm.copy()
    np.fill_diagonal(off_diagonal, 0)
    for flat in np.argsort(off_diagonal, axis=None)[::-1][:10]:
        true_idx, pred_idx = np.unravel_index(flat, off_diagonal.shape)
        if off_diagonal[true_idx, pred_idx] > 0:
            confusions.append({
                "approach": title,
                "true": class_names[true_idx],
                "predicted": class_names[pred_idx],
                "count": int(off_diagonal[true_idx, pred_idx]),
            })

pd.DataFrame(confusions).to_csv(RESULTS_DIR / "top_confusions.csv", index=False)

fig, axes = plt.subplots(2, 2, figsize=(10, 7))
for row, key in enumerate(("full", "head")):
    history = pd.DataFrame(histories[key])
    axes[row, 0].plot(history.epoch, history.train_loss, label="train (augmented)")
    axes[row, 0].plot(history.epoch, history.val_loss, label="val")
    axes[row, 0].set_title(f"{results[key]['approach']} — loss")
    axes[row, 0].legend()
    axes[row, 1].plot(history.epoch, history.train_acc, label="train (augmented)")
    axes[row, 1].plot(history.epoch, history.val_acc, label="val")
    axes[row, 1].set_title(f"{results[key]['approach']} — accuracy")
    axes[row, 1].legend()
for ax in axes.flat:
    ax.set_xlabel("Epoch")
fig.tight_layout()
fig.savefig(FIGURES_DIR / "learning_curves.png", dpi=150)
plt.close(fig)

print("\nDone. CSV/JSON in results/, plots in results/figures/.")
print("Note: Dataset source/license and the analysis discussion belong in the report/README.")
