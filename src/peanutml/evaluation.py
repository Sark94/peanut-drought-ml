"""Repeated nested CV with training-fold-only scaling and selection."""
import json
import re
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold
from sklearn.metrics import balanced_accuracy_score, f1_score, matthews_corrcoef, roc_auc_score
from .data import validate_xy

def make_cv(outer_splits=5, outer_repeats=10, inner_splits=4, inner_repeats=3, random_state=42):
    """Return outer and inner repeated stratified splitters (not study-grouped)."""
    return (RepeatedStratifiedKFold(n_splits=outer_splits,n_repeats=outer_repeats,random_state=random_state),
            RepeatedStratifiedKFold(n_splits=inner_splits,n_repeats=inner_repeats,random_state=random_state))

def run_nested_cv(
    model_name,
    pipeline,
    param_grid,
    X,
    y_encoded,
    sample_ids,
    outer_cv,
    inner_cv,
    output_dir=None,
    n_jobs=1
):
    validate_xy(X, y_encoded, sample_ids)
    total_folds = outer_cv.get_n_splits(X, y_encoded)
    fold_results = []
    prediction_records = []
    selected_gene_records = []
    hyperparameter_records = []

    print(f"\nStarting nested CV for {model_name}...")

    for fold, (train_idx, test_idx) in enumerate(
        outer_cv.split(X, y_encoded), start=1
    ):
        X_train = X.iloc[train_idx]
        X_test  = X.iloc[test_idx]

        y_train = y_encoded[train_idx]
        y_test  = y_encoded[test_idx]

        # Inner-CV tuning
        grid = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            scoring="balanced_accuracy",
            cv=inner_cv,
            n_jobs=n_jobs,
            error_score="raise",
            refit=True,
            verbose=0
        )

        grid.fit(X_train, y_train)

        best_model = grid.best_estimator_

        # Outer-test predictions
        y_pred = best_model.predict(X_test)

        if hasattr(best_model, "predict_proba"):
            y_score = best_model.predict_proba(X_test)[:, 1]
        elif hasattr(best_model, "decision_function"):
            y_score = best_model.decision_function(X_test)
        else:
            y_score = None

        # Metrics
        bal_acc = balanced_accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        mcc = matthews_corrcoef(y_test, y_pred)

        if y_score is not None:
            auc = roc_auc_score(y_test, y_score)
        else:
            auc = np.nan

        fold_results.append({
            "Model": model_name,
            "Outer_Fold": fold,
            "Balanced_Accuracy": bal_acc,
            "F1_Score": f1,
            "MCC": mcc,
            "ROC_AUC": auc,
            "Best_Inner_Balanced_Accuracy": grid.best_score_,
            "Selected_k": grid.best_params_.get("select__k", np.nan)
        })

        # Best hyperparameters
        hyperparameter_records.append({
            "Model": model_name,
            "Outer_Fold": fold,
            "Best_Params": json.dumps(grid.best_params_)
        })

        # Selected genes
        selector = best_model.named_steps["select"]
        selected_mask = selector.get_support()
        selected_genes = X.columns[selected_mask].tolist()

        for gene in selected_genes:
            selected_gene_records.append({
                "Model": model_name,
                "Outer_Fold": fold,
                "Gene": gene
            })

        # Outer-test predictions
        for j, idx in enumerate(test_idx):
            prediction_records.append({
                "Model": model_name,
                "Outer_Fold": fold,
                "Sample_ID": sample_ids.iloc[idx],
                "True_Label": int(y_encoded[idx]),
                "Predicted_Label": int(y_pred[j]),
                "Score": float(y_score[j]) if y_score is not None else np.nan
            })

        print(
            f"{model_name} | fold {fold:02d}/{total_folds} | "
            f"BA={bal_acc:.4f} | F1={f1:.4f} | MCC={mcc:.4f}"
        )

    # Convert to DataFrames
    fold_df = pd.DataFrame(fold_results)
    pred_df = pd.DataFrame(prediction_records)
    genes_df = pd.DataFrame(selected_gene_records)
    params_df = pd.DataFrame(hyperparameter_records)

    # Summary
    summary_df = pd.DataFrame({
        "Model": [model_name],
        "Balanced_Accuracy_Mean": [fold_df["Balanced_Accuracy"].mean()],
        "Balanced_Accuracy_SD": [fold_df["Balanced_Accuracy"].std(ddof=1)],
        "F1_Mean": [fold_df["F1_Score"].mean()],
        "F1_SD": [fold_df["F1_Score"].std(ddof=1)],
        "MCC_Mean": [fold_df["MCC"].mean()],
        "MCC_SD": [fold_df["MCC"].std(ddof=1)],
        "ROC_AUC_Mean": [fold_df["ROC_AUC"].mean()],
        "ROC_AUC_SD": [fold_df["ROC_AUC"].std(ddof=1)]
    })

    # Gene-selection frequency
    gene_freq_df = (
        genes_df.groupby(["Model", "Gene"])
        .size()
        .reset_index(name="Selection_Count")
    )

    gene_freq_df["Selection_Percentage"] = (
        gene_freq_df["Selection_Count"] / total_folds * 100
    )

    gene_freq_df = gene_freq_df.sort_values(
        ["Selection_Count", "Gene"],
        ascending=[False, True]
    )

    if output_dir is not None:
        folder = Path(output_dir)
        folder.mkdir(parents=True, exist_ok=True)
        safe_name = re.sub(r'[^A-Za-z0-9_]+', '_', model_name)
        for suffix, table in [('outer_fold_results',fold_df),('outer_predictions',pred_df),
                              ('selected_genes_by_fold',genes_df),('best_hyperparameters',params_df),
                              ('summary_mean_SD',summary_df),('gene_selection_frequency',gene_freq_df)]:
            table.to_csv(folder / f'{safe_name}_{suffix}.csv', index=False)

    print(f"\n✓ {model_name} complete.")
    print(summary_df)

    return {
        "fold_results": fold_df,
        "predictions": pred_df,
        "selected_genes": genes_df,
        "best_params": params_df,
        "summary": summary_df,
        "gene_frequency": gene_freq_df
    }
