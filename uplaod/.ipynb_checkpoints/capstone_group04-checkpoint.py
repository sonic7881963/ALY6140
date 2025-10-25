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
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import time; time.sleep(0.5)
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# ===== Begin capstone_group4.py =====
# capstone_group4.py
def load_and_inspect_data(file_path):
    try:
        mlb_df = pd.read_csv(file_path)
        print("--- 1. Data Loading and Inspection ---")
        print("Data loaded successfully.")
        print(f"Shape of the dataset: {mlb_df.shape}")
        print("\nFirst 5 rows:")
        print(mlb_df.head())
        print("\nBasic Info:")
        mlb_df.info()
        print("-" * 50)
        return mlb_df
    except FileNotFoundError:
        print(f"Error: The file was not found at {file_path}")
        return None

def perform_linear_regression_analysis(mlb_df):
    print("\n--- 2. Building Linear Regression Model (Predicting ERA) ---")
    
    # --- 1. Data Preparation ---
    mlb_df_cleaned = mlb_df.copy()
    
    # --- Feature Engineering ---
    mlb_df_cleaned['SO9'] = mlb_df_cleaned.apply(lambda row: (row['SO'] / row['IP'] * 9) if row['IP'] > 0 else 0, axis=1)
    mlb_df_cleaned['HR9'] = mlb_df_cleaned.apply(lambda row: (row['HR'] / row['IP'] * 9) if row['IP'] > 0 else 0, axis=1)

    # --- features list ---
    reg_features = ['WAR', 'SO9', 'HR9', 'WHIP'] 
    reg_target = 'ERA'
    
    model_df = mlb_df_cleaned[reg_features + [reg_target]].copy()

    # Missing value -> median
    imputer = SimpleImputer(strategy='median')
    model_df[reg_features] = imputer.fit_transform(model_df[reg_features])
    
    # Drop missing targe value
    model_df.dropna(subset=[reg_target], inplace=True)

    X = model_df[reg_features]
    y = model_df[reg_target]

    # Standardization
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train set and test set
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # --- 2. Model Training ---
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    # --- 3. Evaluation ---
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    print(f"R-squared: {r2:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    
    coefficients = pd.DataFrame(model.coef_, X.columns, columns=['Coefficient'])
    
    print("\nImpact of Each Stat on ERA (Coefficients):")
    print(coefficients)
    
    # --- 4. Visualization ---
    plt.figure(figsize=(5, 4))
    sns.scatterplot(x=y_test, y=y_pred, alpha=0.7)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r', linewidth=2)
    plt.title('Actual ERA vs. Predicted ERA (Linear Model)')
    plt.xlabel('Actual ERA')
    plt.ylabel('Predicted ERA')
    plt.show()


RANDOM_STATE = 42

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
            "clf__max_depth": [3, 4, 5, 7, 9],          
            "clf__min_samples_leaf": [5, 10, 20],       
            "clf__min_samples_split": [10, 20, 40],     
            "clf__ccp_alpha": [0.0, 0.001, 0.005, 0.01] 
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


def train_decision_tree_classifier_simple(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str = "salary_tier",
    test_size: float = 0.2,
    random_state: int = 42,
    base_max_depth: int = 6,
    base_min_samples_leaf: int = 10,
    search_alphas: int = 20,
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

  
    base_tree = DecisionTreeClassifier(
        random_state=random_state,
        class_weight="balanced",
        max_depth=base_max_depth,
        min_samples_leaf=base_min_samples_leaf
    )
    X_train_tr = preprocessor.fit_transform(X_train, y_train)
    path = base_tree.cost_complexity_pruning_path(X_train_tr, y_train)
    ccp_alphas = path.ccp_alphas


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


    best_clf = DecisionTreeClassifier(
        random_state=random_state,
        class_weight="balanced",
        max_depth=base_max_depth,
        min_samples_leaf=base_min_samples_leaf,
        ccp_alpha=best_alpha
    )
    best_model = Pipeline([("prep", preprocessor), ("clf", best_clf)])
    best_model.fit(X_train, y_train)


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



def export_tree_rules(model: Pipeline, feature_names: List[str], max_depth: int = 5) -> str:
    tree_model: DecisionTreeClassifier = model.named_steps["clf"]
    return export_text(tree_model, feature_names=list(feature_names), max_depth=max_depth)

def top_k_features_by_importance(model: Pipeline, feature_names: List[str], k: int = 15) -> List[str]:
    clf = model.named_steps["clf"]
    imp = clf.feature_importances_
    imp_df = pd.DataFrame({"f": feature_names, "w": imp}).sort_values("w", ascending=False)
    keep = imp_df.query("w > 0").head(k)["f"].tolist()
    if not keep:
        keep = list(imp_df.head(k)["f"])
    return keep
# ===== End capstone_group4.py =====

# ===== Begin RF.py =====
def random_forest(mlb_df):
    for col in ['WAR','W','L','G','GS','IP','H','R','ER','HR','BB','SO','ERA','WHIP']:
        mlb_df[col].fillna(mlb_df[col].median(), inplace=True)
    mlb_df['Player_Stats_Match'].fillna('', inplace=True)
    agg_dict = {
        'Total Cash': 'mean',
        'WAR': 'sum',
        'IP': 'sum',
        'SO': 'sum',
        'BB': 'sum',
        'ERA': 'mean',
        'WHIP': 'mean'
    }

    mlbg = mlb_df.groupby(['Player', 'Year'], as_index=False).agg(agg_dict)
    mlbg['SO9'] = (mlbg['SO'] / (mlbg['IP']*9)).fillna(0)
    mlbg['Recent'] = (mlbg['Year'] >= 2019).astype(int)

    # 2) consist varies by year 
    mlbg['Year_num'] = mlbg['Year'] - mlbg['Year'].min()   

    # 3) how long the player have already played
    mlbg = mlbg.sort_values(['Player','Year'])
    mlbg['Seasons_Played'] = mlbg.groupby('Player').cumcount() + 1

    # 4) new player
    mlbg['Rookie'] = (mlbg['Seasons_Played'] == 1).astype(int)

    # 5) previous performance is one thing to get high salary

    mlbg['WAR_prev'] = mlbg.groupby('Player')['WAR'].shift(1).fillna(0)
    mlbg['IP_prev']  = mlbg.groupby('Player')['IP'].shift(1).fillna(0)
    mlbg['ERA_prev'] = mlbg.groupby('Player')['ERA'].shift(1).fillna(mlbg['ERA'].median())
    # print(mlbg.shape)
    # mlb.head()


    q3salary = mlbg['Total Cash'].quantile(0.75)
    mlbg['HighPay'] = (mlbg['Total Cash'] > q3salary).astype(int)

    # features
    features = ['WAR', 'ERA', 'IP', 'SO9', 'WHIP','Year_num','Recent','Seasons_Played',
        'WAR_prev','IP_prev','ERA_prev']
    X = mlbg[features].astype('float64')
    y = mlbg['HighPay']

    # split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    #  build and train model
    rf = RandomForestClassifier(
        n_estimators=500,        # tree numbers
        max_depth=20,             # limit the depth avoiding overfitting
        min_samples_split=20,    
        min_samples_leaf=5,      # each leaf at least need 
        class_weight='balanced',
        random_state=42
    )
    rf.fit(X_train, y_train)

    y_prob = rf.predict_proba(X_test)[:,1]
    threshold = 0.6
    y_pred_adj = (y_prob > threshold).astype(int)



    print("Results")
    print("Training Accuracy:", rf.score(X_train, y_train))
    print("Testing Accuracy:", score_with_threshold(rf, X_test, y_test, threshold=0.6))
    print("\nClassification Report:\n", classification_report(y_test, y_pred_adj))

    # Confusion Matrix 
    plt.figure(figsize=(5,4))
    sns.heatmap(confusion_matrix(y_test, y_pred_adj), annot=True, fmt='d', cmap='Blues', 
                xticklabels=['LowPay','HighPay'], yticklabels=['LowPay','HighPay'])
    plt.title('Confusion Matrix (Random Forest)')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show(block = False)


    # Feature Importance
    importances = pd.DataFrame({
        'Feature': features,
        'Importance': rf.feature_importances_
    }).sort_values(by='Importance', ascending=False)

    plt.figure(figsize=(7,4))
    sns.barplot(x='Importance', y='Feature', data=importances, palette='viridis')
    plt.title('Feature Importance')
    plt.show()

    print("\nTop Feature Importance:\n", importances)

def score_with_threshold(model, X, y, threshold=0.7):
    y_prob = model.predict_proba(X)[:,1]
    y_pred = (y_prob > threshold).astype(int)
    return (y_pred == y).mean()
# ===== End RF.py =====
