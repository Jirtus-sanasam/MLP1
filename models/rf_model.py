from sklearn.ensemble import RandomForestClassifier

def get_rf():
    return RandomForestClassifier(n_estimators=100, random_state=42)
