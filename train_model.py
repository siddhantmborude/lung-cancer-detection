import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


# ==========================================
# 1. LOAD DATASET
# ==========================================

df = pd.read_csv("dataset/survey lung cancer.csv")

print("\n========== DATASET ==========")
print("Shape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())

print("\nFirst 5 rows:")
print(df.head())


# ==========================================
# 2. CLEAN COLUMN NAMES
# ==========================================

# Remove unwanted spaces from column names
df.columns = df.columns.str.strip()

print("\n========== CLEANED COLUMNS ==========")
print(df.columns.tolist())


# ==========================================
# 3. CHECK MISSING VALUES
# ==========================================

print("\n========== MISSING VALUES ==========")
print(df.isnull().sum())


# ==========================================
# 4. CHECK TARGET DISTRIBUTION
# ==========================================

print("\n========== TARGET DISTRIBUTION ==========")
print(df["LUNG_CANCER"].value_counts())


# ==========================================
# 5. DATA VISUALIZATION
# ==========================================

plt.figure(figsize=(6, 4))

sns.countplot(
    data=df,
    x="LUNG_CANCER"
)

plt.title("Lung Cancer Distribution")
plt.xlabel("Lung Cancer")
plt.ylabel("Number of Patients")

plt.tight_layout()
plt.show()


# ==========================================
# 6. ENCODE CATEGORICAL DATA
# ==========================================

# Gender
df["GENDER"] = df["GENDER"].map({
    "M": 1,
    "F": 0
})

# Target
df["LUNG_CANCER"] = df["LUNG_CANCER"].map({
    "YES": 1,
    "NO": 0
})


# ==========================================
# 7. SEPARATE FEATURES AND TARGET
# ==========================================

X = df.drop("LUNG_CANCER", axis=1)

y = df["LUNG_CANCER"]


print("\n========== FEATURES ==========")
print(X.columns.tolist())

print("\nNumber of features:", X.shape[1])


# ==========================================
# 8. TRAIN TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\n========== DATA SPLIT ==========")

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))


# ==========================================
# 9. CREATE MODELS
# ==========================================

models = {

    "Logistic Regression":
        LogisticRegression(max_iter=1000),

    "Decision Tree":
        DecisionTreeClassifier(
            random_state=42,
            max_depth=5
        ),

    "Random Forest":
        RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            class_weight="balanced"
        )
}


# ==========================================
# 10. TRAIN AND EVALUATE MODELS
# ==========================================

results = []

trained_models = {}

for name, model in models.items():

    print("\n===================================")
    print(name)
    print("===================================")

    # Train
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )

    print("Accuracy :", round(accuracy * 100, 2), "%")
    print("Precision:", round(precision * 100, 2), "%")
    print("Recall   :", round(recall * 100, 2), "%")
    print("F1 Score :", round(f1 * 100, 2), "%")

    print("\nClassification Report:")
    print(classification_report(
        y_test,
        y_pred,
        zero_division=0
    ))

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1
    })

    trained_models[name] = model


# ==========================================
# 11. MODEL COMPARISON
# ==========================================

results_df = pd.DataFrame(results)

print("\n\n========== MODEL COMPARISON ==========")
print(results_df)


# ==========================================
# 12. VISUALIZE MODEL PERFORMANCE
# ==========================================

results_plot = results_df.set_index("Model")

results_plot.plot(
    kind="bar",
    figsize=(10, 6)
)

plt.title("ML Model Performance Comparison")
plt.ylabel("Score")
plt.ylim(0, 1.1)

plt.xticks(rotation=0)

plt.tight_layout()
plt.show()


# ==========================================
# 13. SELECT BEST MODEL
# ==========================================

best_model_name = results_df.loc[
    results_df["F1 Score"].idxmax(),
    "Model"
]

best_model = trained_models[best_model_name]

print("\n===================================")
print("BEST MODEL")
print("===================================")

print("Best Model:", best_model_name)


# ==========================================
# 14. CONFUSION MATRIX
# ==========================================

best_predictions = best_model.predict(X_test)

cm = confusion_matrix(
    y_test,
    best_predictions
)

plt.figure(figsize=(6, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues"
)

plt.title(
    f"Confusion Matrix - {best_model_name}"
)

plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.tight_layout()
plt.show()


# ==========================================
# 15. FEATURE IMPORTANCE
# ==========================================

if best_model_name == "Random Forest":

    importance = pd.Series(
        best_model.feature_importances_,
        index=X.columns
    ).sort_values(ascending=False)

    plt.figure(figsize=(10, 6))

    importance.plot(kind="bar")

    plt.title("Feature Importance - Random Forest")

    plt.xlabel("Features")
    plt.ylabel("Importance")

    plt.tight_layout()
    plt.show()


# ==========================================
# 16. SAVE BEST MODEL
# ==========================================

joblib.dump(
    best_model,
    "models/lung_cancer_model.pkl"
)

print("\n===================================")
print("MODEL SAVED")
print("===================================")

print(
    "Saved as: models/lung_cancer_model.pkl"
)

print("\nProject training completed successfully!")