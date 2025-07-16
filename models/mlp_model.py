from sklearn.neural_network import MLPClassifier

def get_mlp():
    return MLPClassifier(hidden_layer_sizes=(64, 64), max_iter=500, random_state=42)
