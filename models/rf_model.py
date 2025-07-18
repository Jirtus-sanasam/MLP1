from sklearn.ensemble import RandomForestClassifier

def get_rf():
    return RandomForestClassifier(
        n_estimators=200,            # more trees = better averaging
        max_depth=5,                 # restrict depth to prevent overfitting
        min_samples_split=10,        # require more samples to split
        min_samples_leaf=4,          # each leaf must have at least 4 samples
        max_features='sqrt',         # best practice for classification
        random_state=42,
        n_jobs=-1                    # use all CPU cores
    )
