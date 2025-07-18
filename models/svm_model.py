from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

def get_svm():
    param_grid = {
        'C': [1],              # Simplified for debugging
        'gamma': [0.1],        # Simplified for debugging
        'kernel': ['rbf']      # ✅ Fixed: added missing closing bracket
    }

    grid_search = GridSearchCV(
        estimator=SVC(probability=True, random_state=42),
        param_grid=param_grid,
        scoring='accuracy',
        cv=3,                   # Simplified cross-validation
        n_jobs=-1,
        verbose=1              # Enable verbose logs
    )

    return grid_search
