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
    df = pd.read_csv('StressLevelDataset.csv')
    
    X = df.iloc[:, :-1].values
    y = df['stress_level'].values
    
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    print(f"Alunos para Treino: {len(X_train)} | Alunos para Teste: {len(X_test)}")

    print("\n=== ETAPA 2: Treinamento da LVQ-1 ===")
    lvq = LVQ1Net(input_dim=20, num_classes=3, alpha=0.05)
    lvq.train(X_train, y_train, epochs=500)
    print("-> Rede treinada com sucesso!")

    print("\n=== ETAPA 3: Inferência e Resultados ===")
    y_pred = lvq.predict(X_test)
    acuracia = accuracy_score(y_test, y_pred)
    print(f"-> Acurácia Global nos dados inéditos: {acuracia * 100:.2f}%\n")

    # --- NOVA PARTE: MOSTRAR DIAGNÓSTICO DE 50 ALUNOS ---
    print("=== EXEMPLO DE DIAGNÓSTICO PREVENTIVO (50 ALUNOS) ===")
    print(f"{'ID':<5} | {'Ansiedade (0-21)':<16} | {'Sono (1-5)':<10} | {'Status Real':<12} | {'Diagnóstico IA':<15}")
    print("-" * 70)
    
    X_test_real = scaler.inverse_transform(X_test)
    mapeamento = {0: "Baixo", 1: "Médio", 2: "Alto"}
    
    for i in range(50):
        ansiedade = int(X_test_real[i][0]) # Coluna 0
        sono = int(X_test_real[i][6])      # Coluna 6
        real_str = mapeamento[y_test[i]]
        pred_str = mapeamento[y_pred[i]]
        
        # Sinaliza se a IA acertou ou errou o diagnóstico
        checagem = "✅" if y_test[i] == y_pred[i] else "❌"
        
        print(f"#{i+1:<4} | {ansiedade:<16} | {sono:<10} | {real_str:<12} | {pred_str:<15} {checagem}")
    print("-" * 70)
    # ----------------------------------------------------

    print("\n=== ETAPA 4: Gerando Gráficos para Apresentação ===")
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Baixo', 'Médio', 'Alto'], 
                yticklabels=['Baixo', 'Médio', 'Alto'])
    plt.title('Matriz de Confusão - Diagnóstico de Estresse (LVQ-1)')
    plt.ylabel('Classe Real')
    plt.xlabel('Previsão da Rede')
    plt.tight_layout()
    plt.savefig('matriz_confusao.png', dpi=300)
    print("-> 'matriz_confusao.png' salvo com sucesso!")

    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=X_test_real[:, 0], y=X_test_real[:, 6], 
                    hue=y_pred, palette=['green', 'orange', 'red'], alpha=0.6, s=50)
    
    Pesos_Reais = scaler.inverse_transform(lvq.W)
    plt.scatter(Pesos_Reais[:, 0], Pesos_Reais[:, 6], 
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