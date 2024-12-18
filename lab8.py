import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, LeaveOneOut
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix

# Cargar los datasets
iris = pd.read_csv("C:\\Users\\crist\\Downloads\\iris\\bezdekIris.csv", header=None)
cancer = pd.read_csv("C:\\Users\\crist\\Downloads\\breast+cancer+wisconsin+diagnostic\\wdbc.csv", header=None)
wine = pd.read_csv("C:\\Users\\crist\\Downloads\\wine+quality (1)\\winequality-red.csv", sep=";")
 #"C:\Users\crist\Downloads\iris\bezdekIris.csv"
# Preprocesar los datasets
datasets = {
    "Iris": (iris.iloc[:, :-1], iris.iloc[:, -1]),
    "Cancer": (cancer.iloc[:, 2:], cancer.iloc[:, 1]),
    "Wine": (wine.iloc[:, :-1], wine.iloc[:, -1]),
}

# Hold-Out 70/30 Estratificado
def hold_out_validation(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
    model = GaussianNB()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    return acc, cm

# 10-Fold Cross-Validation Estratificado
def k_fold_validation(X, y, k=10):
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    model = GaussianNB()
    accuracies = cross_val_score(model, X, y, cv=skf, scoring='accuracy')
    return accuracies.mean(), accuracies

# Leave-One-Out Validation
def leave_one_out_validation(X, y):
    loo = LeaveOneOut()
    model = GaussianNB()
    y_true, y_pred = [], []
    for train_index, test_index in loo.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        model.fit(X_train, y_train)
        y_pred.append(model.predict(X_test)[0])
        y_true.append(y_test.values[0])
    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    return acc, cm

# Validar los datasets
for name, (X, y) in datasets.items():
    print(f"\n=== Dataset: {name} ===")
    
    # Hold-Out
    acc, cm = hold_out_validation(X, y)
    print(f"Hold-Out - Accuracy: {acc}")
    print(f"Hold-Out - Confusion Matrix:\n{cm}")
    
    # 10-Fold Cross-Validation
    acc_mean, acc_all = k_fold_validation(X, y, k=10)
    print(f"10-Fold CV - Mean Accuracy: {acc_mean}")
    print(f"10-Fold CV - Accuracies per Fold: {acc_all}")
    
    # Leave-One-Out
    acc, cm = leave_one_out_validation(X, y)
    print(f"Leave-One-Out - Accuracy: {acc}")
    print(f"Leave-One-Out - Confusion Matrix:\n{cm}")
