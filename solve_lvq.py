# Script de Execução, Validação e Geração de Gráficos
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, accuracy_score
from lvq import LVQ1Net

def executar_e_gerar_graficos():
    print("=== ETAPA 1: Carregamento e Pré-processamento ===")
    # 1. Carrega o dataset
    df = pd.read_csv('StressLevelDataset.csv')
    
    # Separa os recursos (20 primeiras colunas) e o alvo (última coluna)
    X = df.iloc[:, :-1].values
    y = df['stress_level'].values
    
    # 2. Normalização (CRUCIAL para distância Euclidiana)
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 3. Divisão: 80% Treino (880 alunos) e 20% Teste (220 alunos inéditos)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    print(f"Alunos para Treino: {len(X_train)} | Alunos para Teste: {len(X_test)}")

    print("\n=== ETAPA 2: Treinamento da LVQ-1 ===")
    lvq = LVQ1Net(input_dim=20, num_classes=3, alpha=0.05)
    lvq.train(X_train, y_train, epochs=500)
    print("-> Rede treinada com sucesso!")

    print("\n=== ETAPA 3: Inferência e Resultados ===")
    y_pred = lvq.predict(X_test)
    acuracia = accuracy_score(y_test, y_pred)
    print(f"-> Acurácia Global nos dados inéditos: {acuracia * 100:.2f}%")

    print("\n=== ETAPA 4: Gerando Gráficos para Apresentação ===")
    
    # GRÁFICO 1: Matriz de Confusão
    # Mostra exatamente onde a rede acertou e onde confundiu os níveis de estresse
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Baixo (0)', 'Médio (1)', 'Alto (2)'], 
                yticklabels=['Baixo (0)', 'Médio (1)', 'Alto (2)'])
    plt.title('Matriz de Confusão - Diagnóstico de Estresse (LVQ-1)')
    plt.ylabel('Classe Real')
    plt.xlabel('Previsão da Rede')
    plt.tight_layout()
    plt.savefig('matriz_confusao.png', dpi=300)
    print("-> 'matriz_confusao.png' salvo com sucesso!")

    # GRÁFICO 2: Dispersão 2D (Ansiedade vs Qualidade do Sono)
    # Mostra os clusters de alunos no espaço bidimensional com os pesos protótipos da rede
    plt.figure(figsize=(8, 6))
    
    # Plota os alunos do teste (desnormalizando apenas para o gráfico ficar com os valores reais)
    X_test_real = scaler.inverse_transform(X_test)
    idx_ansiedade = 0 # anxiety_level é a coluna 0
    idx_sono = 6      # sleep_quality é a coluna 6
    
    sns.scatterplot(x=X_test_real[:, idx_ansiedade], y=X_test_real[:, idx_sono], 
                    hue=y_pred, palette=['green', 'orange', 'red'], alpha=0.6, s=50)
    
    # Plota onde os "neurônios vencedores" pararam (desnormalizados)
    Pesos_Reais = scaler.inverse_transform(lvq.W)
    plt.scatter(Pesos_Reais[:, idx_ansiedade], Pesos_Reais[:, idx_sono], 
                c='black', marker='X', s=200, label='Protótipos (Pesos LVQ)')
    
    plt.title('Separação de Perfis: Ansiedade vs Qualidade do Sono')
    plt.xlabel('Nível de Ansiedade (0-21)')
    plt.ylabel('Qualidade do Sono (1-5)')
    plt.legend(title='Nível Previsto', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('dispersao_perfis.png', dpi=300)
    print("-> 'dispersao_perfis.png' salvo com sucesso!")

if __name__ == '__main__':
    executar_e_gerar_graficos()