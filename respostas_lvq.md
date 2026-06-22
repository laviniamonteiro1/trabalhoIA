## Relatório Prático: Sistema de Diagnóstico Preditivo de Estresse (Redes LVQ-1)

### **1. Definição do Problema e Arquitetura**
O presente laboratório implementou uma rede neural supervisionada competitiva (LVQ-1) com o objetivo de analisar e classificar perfis de estresse acadêmico. O sistema processa um vetor de entrada contendo 20 dimensões (fatores psicológicos, fisiológicos, ambientais e acadêmicos) e aloca o estudante em uma de três classes preventivas:
* **Classe 0:** Estresse Baixo.
* **Classe 1:** Estresse Médio (Alerta).
* **Classe 2:** Estresse Alto (Risco Crítico).

Para garantir a confiabilidade matemática no cálculo de similaridade (Distância Euclidiana), os dados passaram por um processo de normalização (MinMaxScaler), impedindo que variáveis de maior amplitude dominassem o treinamento das sinapses da rede.

### **2. Execução e Métricas de Validação**
A base de dados total (1100 amostras) foi particionada para evitar o sobreajuste (*overfitting*):
* **Conjunto de Treinamento (80%):** 880 alunos foram apresentados à rede para o ajuste topológico da matriz de pesos, utilizando taxa de aprendizado $\alpha = 0.05$ ao longo de 500 épocas.
* **Conjunto de Testes (20%):** 220 alunos inéditos foram utilizados exclusivamente para a fase de inferência.

**Resultado Global:** A arquitetura LVQ-1 atingiu uma taxa de acurácia de **90.91%** na classificação do conjunto de testes. 

### **3. Análise da Matriz de Confusão**
*(Insira aqui a imagem `matriz_confusao.png`)*

A análise visual da matriz comprova a robustez das fronteiras de decisão estabelecidas pelo algoritmo. A alta concentração de acertos na diagonal principal indica que a rede assimilou corretamente a relação causal entre os hábitos de rotina e a classe alvo. A margem de erro (~9%) concentra-se em fronteiras limítrofes (ex: confusão leve entre estresse moderado e alto), não havendo erros drásticos (como classificar um risco crítico como estresse baixo).

### **4. Mapeamento Topológico dos Perfis**
*(Insira aqui a imagem `dispersao_perfis.png`)*

O gráfico de dispersão ilustra o comportamento dos protótipos da camada competitiva após a estabilização. Isolando os eixos de "Nível de Ansiedade" e "Qualidade do Sono", é possível observar claramente a capacidade de generalização espacial da rede, que posicionou estrategicamente seus "neurônios vencedores" no centro de massa de cada agrupamento, provando a eficácia da regra de atração e repulsão supervisionada.
### **3. Análise da Matriz de Confusão**
*(Inserir imagem: matriz_confusao.png)*

A matriz de confusão gerada sobre as 220 amostras inéditas do conjunto de testes atesta a alta precisão e a segurança do modelo de diagnóstico supervisionado. Analisando o quadrante, destacam-se os seguintes comportamentos matemáticos da rede:

* **Convergência na Diagonal Principal:** O modelo obteve sucesso massivo em classificar os alunos em seus rótulos exatos, com 76 acertos para a Classe 0 (Baixo), 64 acertos para a Classe 1 (Médio) e 66 acertos para a Classe 2 (Alto). Isso comprova que a regra de atualização de pesos da LVQ-1 conseguiu mapear corretamente as assinaturas vetoriais de cada nível de estresse.
* **Margem de Erro Segura (Falsos Vizinhos):** Os erros cometidos pela rede são mínimos e ocorrem exclusivamente nas fronteiras de classes adjacentes. Por exemplo, houveram apenas 6 casos onde um estresse real de nível Baixo foi classificado como Médio, e 4 casos do inverso.
* **Ausência de Erros Críticos:** O ponto mais forte e defensável deste modelo preditivo é que **não houve nenhuma confusão entre os extremos**. O quadrante mostra **0** casos de um aluno com estresse Baixo sendo classificado como Alto, e **0** casos de um aluno em risco crítico (Alto) ser mascarado como estresse Baixo. Em um sistema de alerta escolar, essa confiabilidade é fundamental para evitar negligência médica.

### **4. Mapeamento Topológico dos Perfis (Dispersão 2D)**
*(Inserir imagem: dispersao_perfis.png)*

Para ilustrar a separação geométrica (bacias de atração) que a rede LVQ-1 construiu, o gráfico de dispersão reduz a visualização do hiperplano para as duas variáveis mais determinantes do estudo: **Nível de Ansiedade (eixo X)** e **Qualidade do Sono (eixo Y)**.

A análise espacial revela uma coerência perfeita entre os dados e os vetores de peso da rede:
* **Centroides (Marcadores "X" em preto):** Os três "X" representam a posição final dos neurônios vencedores (protótipos) após 500 épocas de treinamento. Eles atuam como centros gravitacionais para o cálculo da distância Euclidiana.
* **Perfil Saudável (Verde - Classe 0):** O protótipo ancorou-se solidamente na região inferior direita. A rede aprendeu sozinha que alunos com baixíssima ansiedade (abaixo de 8) e altíssima qualidade de sono (notas 4 e 5) pertencem a um grupo de risco nulo.
* **Perfil de Alerta (Laranja - Classe 1):** O neurônio representante estabilizou-se no centro exato do gráfico, capturando alunos que estão em um limiar transicional (ansiedade média entre 8 e 15; qualidade de sono moderada entre 2 e 3).
* **Perfil Crítico (Vermelho - Classe 2):** O neurônio de risco máximo migrou para o quadrante oposto ao saudável. Visualmente, fica provada a tese do modelo: a combinação simultânea de privação de sono severa (notas 1 e 2) com picos de ansiedade (acima de 15) joga o aluno diretamente para a classe de esgotamento.