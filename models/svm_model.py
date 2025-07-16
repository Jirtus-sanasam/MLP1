from sklearn.svm import SVC

def get_svm():
    return SVC(probability=True, kernel='rbf', random_state=42)
