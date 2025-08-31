import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib

# 1. Load Dataset
data = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Drop customerID column if present
if 'customerID' in data.columns:
    data = data.drop("customerID", axis=1)

# Convert target column "Churn" to binary
data["Churn"] = data["Churn"].map({"Yes": 1, "No": 0})

# Handle missing values
data = data.replace(" ", np.nan)
data = data.dropna()

# 2. Features & Target
X = data.drop("Churn", axis=1)
y = data["Churn"]

# Identify categorical & numerical features
categorical_features = X.select_dtypes(include=["object"]).columns.tolist()
numerical_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

# 3. Preprocessing
numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown="ignore")

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numerical_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)

# 4. Pipelines for models
log_reg = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000))
])

rf_clf = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier())
])

# 5. Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 6. Hyperparameter tuning with GridSearchCV
param_grid = [
    {
        "classifier": [LogisticRegression(max_iter=1000)],
        "classifier__C": [0.01, 0.1, 1, 10],
        "classifier__solver": ["liblinear", "lbfgs"]
    },
    {
        "classifier": [RandomForestClassifier()],
        "classifier__n_estimators": [100, 200],
        "classifier__max_depth": [5, 10, None],
        "classifier__min_samples_split": [2, 5]
    }
]

pipe = Pipeline(steps=[("preprocessor", preprocessor),
                       ("classifier", LogisticRegression())])  # placeholder

grid_search = GridSearchCV(pipe, param_grid, cv=5, scoring="accuracy", n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

# 7. Best model evaluation
print("Best Parameters:", grid_search.best_params_)
best_model = grid_search.best_estimator_

y_pred = best_model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# 8. Export final pipeline
joblib.dump(best_model, "churn_model.pkl")
print("✅ Pipeline saved as churn_model.pkl")
