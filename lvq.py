# Motor da Rede LVQ-1 - Diagnóstico de Estresse
import numpy as np

class LVQ1Net:
    def __init__(self, input_dim=20, num_classes=3, alpha=0.05):
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.alpha = alpha
        self.W = None
        self.classes_w = np.array([0, 1, 2]) # Classes reais do dataset: 0, 1, 2

    def initialize_weights(self, X_train, y_train):
        """Inicializa os pesos pegando o primeiro aluno encontrado de cada classe."""
        self.W = np.zeros((self.num_classes, self.input_dim))
        classes_encontradas = set()
        
        for x, y in zip(X_train, y_train):
            if y not in classes_encontradas:
                idx = np.where(self.classes_w == y)[0][0]
                self.W[idx] = x.copy()
                classes_encontradas.add(y)
            if len(classes_encontradas) == self.num_classes:
                break

    def find_bmu(self, x):
        """Encontra o neurônio vencedor por Menor Distância Euclidiana."""
        distancias = np.linalg.norm(self.W - x, axis=1)
        return np.argmin(distancias)

    def train(self, X_train, y_train, epochs=500):
        X_train = np.array(X_train, dtype=float)
        y_train = np.array(y_train, dtype=int)
        
        if self.W is None:
            self.initialize_weights(X_train, y_train)
            
        for epoch in range(epochs):
            # Embaralha os dados
            indices = np.random.permutation(len(X_train))
            X_train, y_train = X_train[indices], y_train[indices]
                
            for x, y_real in zip(X_train, y_train):
                bmu_idx = self.find_bmu(x)
                y_predito = self.classes_w[bmu_idx]
                
                # LVQ-1: Recompensa x Punição
                if y_predito == y_real:
                    self.W[bmu_idx] += self.alpha * (x - self.W[bmu_idx])
                else:
                    self.W[bmu_idx] -= self.alpha * (x - self.W[bmu_idx])
                    
        return self.W

    def predict(self, X_test):
        X_test = np.array(X_test, dtype=float)
        previsoes = [self.classes_w[self.find_bmu(x)] for x in X_test]
        return previsoes