# capstone_group4.py
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import List, Dict, Union, Optional

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import matplotlib.pyplot as plt

RANDOM_STATE = 42

# =========================
# 1) 薪资 → 等级（分位数）
# =========================
def make_salary_tiers(
    df: pd.DataFrame,
    salary_col: str = "Total Cash",
    n_tiers: int = 4,
    labels: Optional[List[str]] = None
) -> pd.DataFrame:
    if labels is None:
        labels = [f"T{i+1}" for i in range(n_tiers)]
    s = df[salary_col].astype(float)
    df = df.copy()
    df["salary_tier"] = pd.qcut(s, q=n_tiers, labels=labels, duplicates="drop")
    return df

# ===========================================
# 2) 自动数值特征（排除 ID/目标/薪资等非特征列）
# ===========================================
def infer_feature_columns(
    df: pd.DataFrame,
    target_col: str = "salary_tier",
    drop_like: Optional[List[str]] = None
) -> List[str]:
    drop_like = (drop_like or []) + ["Player", "Year", "Total Cash", target_col]
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    features = [c for c in numeric_cols if c not in drop_like]
    if not features:
        raise ValueError("未找到可用于建模的数值特征列。")
    return features

# ====================================
# 3A) 原版：GridSearch 调参的决策树
#  —— 已收紧网格，避免过深、过拟合
# ====================================
def train_decision_tree_classifier(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str = "salary_tier",
    test_size: float = 0.2,
    cv_splits: int = 5,
    scoring: str = "f1_macro",
    param_grid: Optional[Dict[str, List[Union[int, float, str]]]] = None,
    random_state: int = RANDOM_STATE
):
    X = df[feature_cols].copy()
    y = df[target_col].astype(str).copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), feature_cols)
        ],
        remainder="drop"
    )

    clf = DecisionTreeClassifier(
        random_state=random_state,
        class_weight="balanced"
    )

    pipe = Pipeline(steps=[
        ("prep", preprocessor),
        ("clf", clf)
    ])

    if param_grid is None:
        param_grid = {
            "clf__criterion": ["gini", "entropy", "log_loss"],
            "clf__max_depth": [None, 12, 16],
            "clf__min_samples_leaf": [1, 2, 5],
            "clf__min_samples_split": [2, 4, 6],
            "clf__ccp_alpha": [0.0, 1e-7, 1e-6, 1e-5]
        }

         

    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
    gs = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        scoring=scoring,
        cv=cv,
        n_jobs=-1,
        verbose=0
    )
    gs.fit(X_train, y_train)
    best_model: Pipeline = gs.best_estimator_

    # 测试集评估
    y_pred = best_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1m = f1_score(y_test, y_pred, average="macro")
    report = classification_report(y_test, y_pred, digits=3)

    summary = {
        "best_params": gs.best_params_,
        "cv_best_score": float(gs.best_score_),
        "holdout_accuracy": float(acc),
        "holdout_f1_macro": float(f1m),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "classes_": sorted(y.unique().tolist()),
        "classification_report": report
    }
    return best_model, (X_test, y_test, y_pred), summary

# ==========================================================
# 3B) 简洁稳健版：限深 + 自动挑选 ccp_alpha 的剪枝决策树
#  —— 可读性优先，生成更“干净”的树（推荐报告使用）
# ==========================================================
def train_decision_tree_classifier_simple(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str = "salary_tier",
    test_size: float = 0.2,
    random_state: int = 42,
    base_max_depth: int = 6,
    base_min_samples_leaf: int = 10,
    search_alphas: int = 20,        # 采样多少个 alpha 做交叉验证挑选
    scoring: str = "f1_macro",
    cv_splits: int = 5
):
    X = df[feature_cols].copy()
    y = df[target_col].astype(str).copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), feature_cols)
        ],
        remainder="drop"
    )

    # 基础约束（限深/叶子），先求剪枝路径
    base_tree = DecisionTreeClassifier(
        random_state=random_state,
        class_weight="balanced",
        max_depth=base_max_depth,
        min_samples_leaf=base_min_samples_leaf
    )
    X_train_tr = preprocessor.fit_transform(X_train, y_train)
    path = base_tree.cost_complexity_pruning_path(X_train_tr, y_train)
    ccp_alphas = path.ccp_alphas

    # 采样若干 alpha 做CV
    if len(ccp_alphas) > search_alphas:
        idx = np.linspace(0, len(ccp_alphas)-1, search_alphas, dtype=int)
        ccp_alphas = ccp_alphas[idx]

    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
    best_alpha, best_score = None, -np.inf

    for a in ccp_alphas:
        clf = DecisionTreeClassifier(
            random_state=random_state,
            class_weight="balanced",
            max_depth=base_max_depth,
            min_samples_leaf=base_min_samples_leaf,
            ccp_alpha=a
        )
        pipe = Pipeline([("prep", preprocessor), ("clf", clf)])
        scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)
        if scores.mean() > best_score:
            best_score = scores.mean()
            best_alpha = a

    # 用最佳 alpha 重新在全训练集上拟合
    best_clf = DecisionTreeClassifier(
        random_state=random_state,
        class_weight="balanced",
        max_depth=base_max_depth,
        min_samples_leaf=base_min_samples_leaf,
        ccp_alpha=best_alpha
    )
    best_model = Pipeline([("prep", preprocessor), ("clf", best_clf)])
    best_model.fit(X_train, y_train)

    # 测试集评估
    y_pred = best_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1m = f1_score(y_test, y_pred, average="macro")
    report = classification_report(y_test, y_pred, digits=3)

    summary = {
        "chosen_ccp_alpha": float(best_alpha) if best_alpha is not None else 0.0,
        "cv_best_score": float(best_score),
        "holdout_accuracy": float(acc),
        "holdout_f1_macro": float(f1m),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "classes_": sorted(y.unique().tolist()),
        "classification_report": report,
        "constraints": {
            "max_depth": base_max_depth,
            "min_samples_leaf": base_min_samples_leaf
        }
    }
    return best_model, (X_test, y_test, y_pred), summary

# ================================
# 4) 评估：混淆矩阵（支持标准化）
# ================================
def plot_confusion(y_true, y_pred, normalize: Optional[str] = None):
    cm = confusion_matrix(y_true, y_pred, normalize=normalize)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, aspect="auto")
    ax.set_title("Confusion Matrix")
    plt.colorbar(im, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    plt.tight_layout()
    plt.show()

# =============================================
# 5) 树结构可视化（新增：只画前 N 层 + 可调画质）
# =============================================
def plot_tree_structure(
    model: Pipeline,
    feature_names: List[str],
    max_depth_to_plot: int = 5,
    figsize=(14, 8),
    dpi: int = 150,
    fontsize: int = 8
):
    tree_model: DecisionTreeClassifier = model.named_steps["clf"]
    plt.figure(figsize=figsize, dpi=dpi)
    plot_tree(
        tree_model,
        feature_names=feature_names,
        class_names=[str(c) for c in tree_model.classes_],
        filled=True,
        impurity=True,
        rounded=True,
        max_depth=max_depth_to_plot,
        fontsize=fontsize
    )
    plt.show()

# ==================================
# 6) 文本规则（更适合在报告里展示）
# ==================================
def export_tree_rules(model: Pipeline, feature_names: List[str], max_depth: int = 5) -> str:
    tree_model: DecisionTreeClassifier = model.named_steps["clf"]
    return export_text(tree_model, feature_names=list(feature_names), max_depth=max_depth)

# ==================================================
# 7) （可选）按重要性筛前 K 特征，再训一棵更小的树
# ==================================================
def top_k_features_by_importance(model: Pipeline, feature_names: List[str], k: int = 15) -> List[str]:
    clf = model.named_steps["clf"]
    imp = clf.feature_importances_
    imp_df = pd.DataFrame({"f": feature_names, "w": imp}).sort_values("w", ascending=False)
    keep = imp_df.query("w > 0").head(k)["f"].tolist()
    if not keep:
        keep = list(imp_df.head(k)["f"])
    return keep
