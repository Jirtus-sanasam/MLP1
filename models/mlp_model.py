from sklearn.neural_network import MLPClassifier

def get_mlp():
    return MLPClassifier(
        hidden_layer_sizes=(32, 16),   # simpler architecture
        activation='relu',
        alpha=0.001,                   # L2 regularization
        solver='adam',
        learning_rate='adaptive',     # adaptive learning rate
        max_iter=500,
        early_stopping=True,          # stop if no improvement
        n_iter_no_change=10,
        validation_fraction=0.2,
        random_state=42
    )
