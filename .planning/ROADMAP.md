# Roadmap de Desenvolvimento - Sistema de Diagnóstico (LVQ-1)

- [ ] **Fase 1: Preparação de Dados e Estrutura**
  - [ ] Criar a hierarquia de pastas `RNA/stress_lvq/`.
  - [ ] Mover o arquivo oficial `StressLevelDataset.csv` para a raiz do projeto.

- [ ] **Fase 2: Motor Algorítmico (`lvq_stress.py`)**
  - [ ] Implementar a inicialização inteligente da matriz de pesos (tamanho $3 \times 20$), "roubando" a primeira amostra de cada uma das 3 classes para acelerar a convergência.
  - [ ] Implementar o cálculo de distância Euclidiana para 20 dimensões.
  - [ ] Implementar a regra condicional de ajuste de pesos (LVQ-1) com base no gabarito `stress_level`.

- [ ] **Fase 3: Execução e Automação (`solve_stress.py`)**
  - [ ] Criar o pipeline de leitura segura do arquivo CSV utilizando `pandas` ou `csv` nativo do Python.
  - [ ] Implementar a divisão do dataset (80% treino, 20% teste).
  - [ ] Treinar a rede neural com o conjunto de treinamento.
  - [ ] Realizar a inferência no conjunto de testes.
  - [ ] Gerar um relatório no terminal exibindo a taxa de acerto (acurácia) e o mapeamento dos resultados.

- [ ] **Fase 4: Documentação Final (`respostas_stress.md`)**
  - [ ] Compilar o cabeçalho padrão do Lab de Inteligência Artificial.
  - [ ] Documentar a performance da rede e as conclusões lógicas da classificação das amostras de teste.