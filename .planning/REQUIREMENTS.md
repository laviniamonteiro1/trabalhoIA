# Requisitos do Sistema - Diagnóstico Preditivo de Estresse Acadêmico (LVQ-1)

## 1. Descrição do Problema
O sistema visa identificar e classificar automaticamente o nível de risco psicológico e acadêmico de estudantes. Utilizando um conjunto de 1100 registros reais, o algoritmo deve analisar o cruzamento de variáveis comportamentais, fisiológicas e de rotina para emitir um diagnóstico preventivo.

## 2. Especificações da Arquitetura
* **Base de Dados:** `StressLevelDataset.csv` (1100 amostras).
* **Vetor de Entrada (x):** 20 dimensões, englobando fatores como `anxiety_level`, `sleep_quality`, `study_load`, `peer_pressure`, entre outros.
* **Camada Competitiva (Saída):** 3 neurônios, correspondendo diretamente aos perfis de estresse (Target: `stress_level`):
  * **Classe 0:** Nível Baixo (Saudável).
  * **Classe 1:** Nível Médio (Alerta).
  * **Classe 2:** Nível Alto (Risco Crítico).
* **Algoritmo:** LVQ-1 (Learning Vector Quantization - Aprendizado Supervisionado Competitivo).

## 3. Dinâmica de Atualização dos Pesos (LVQ-1)
O sistema calculará a distância Euclidiana para encontrar o neurônio vencedor (BMU) e aplicará o reforço supervisionado com base no rótulo real do aluno:
* **Se a classificação for correta ($C_w = C_x$):** O protótipo se aproxima da amostra.
  $$W_{nova} = W_{atual} + \alpha (x - W_{atual})$$
* **Se a classificação for incorreta ($C_w \neq C_x$):** O protótipo é repelido pela amostra.
  $$W_{nova} = W_{atual} - \alpha (x - W_{atual})$$

## 4. Estratégia de Validação (Entregáveis)
Para comprovar a eficácia da rede neural, os dados serão particionados:
1. **Treinamento (80%):** 880 alunos serão usados para ajustar a matriz de pesos e criar as bacias de atração.
2. **Teste (20%):** 220 alunos inéditos serão submetidos à rede estabilizada para o cálculo da matriz de confusão e acurácia final.