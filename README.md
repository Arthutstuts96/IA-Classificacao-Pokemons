# IA-Classificacao-Pokemons

O objetivo desta atividade foi criar uma rede neural MLP capaz de resolver um problema de classificação definido, particularmente o de adivinhar o tipo de um dado pokemon a partir de alguns parâmetros coletados de uma base de dados. Caso o pokemon possua dois tipos, e pelo menos um tipo foi previsto com sucesso, considera-se como um todo sucesso a previsão

<h2>Ambiente</h2>

Foram usadas as bibliotecas do python <code>scikit-learn</code>, para criar a rede e tratar o conjunto de dados, <code>pandas</code>, para leitura do CSV e normatização, <code>matplotlib</code>, para criar o gráfico da matriz de confusão, e <code>joblib</code>, para salvar em arquivo o conjunto de dados normatizado e a rede MLP

A base de dados utilizada no algoritmo está disponível <a href="https://www.kaggle.com/datasets/alopez247/pokemon/code">neste link</a>, e foi baixada como CSV

O código foi escrito no VSCode, e tudo foi executado no sistema operacional Linux. A versão do python utilizada é a <code>3.12.3</code>

<h2>Primeiro teste</h2>

Inicialmente, os parâmetros foram:

```python
modelo = MLPClassifier(
    hidden_layer_sizes=(128, 64),  
    activation='relu',             
    solver='adam',                
    max_iter=500,                
    random_state=42
)
```

Nesse molde, os resultados não foram satisfatórios, estão dispostos na imagem abaixo. Foram utilizadas as métricas de avaliação <i>precision, recall</i> e <i>f1-score</i>, contudo, para contabilizar sucessos no acerto de pelo menos um dos tipos (ambos serviriam como rótulos), foi criado um método diferente para cálculo da acurácia

<img src="/imagens/primera_iteracao.png" alt="Imagem do primeiro teste" />

A acurácia do primeiro tipo foi de 65.44%, enquanto que considerando ambos tipos foi 68.66%. O objetivo é obter uma acurácia de 90% para cima, de modo a considerar a rede satisfatória

...

Eventualmente, foi percebido o maior problema da rede: as classes estavam desbalanceadas, de modo que, a exemplo, o tipo voador "Flying", possuía somente dois registros, sendo ambos da mesma família Pokemon. Isso se deu pela característica dos jogos de Pokemon priorizarem o tipo voador como tipo secundário, o que fez com que o número de Pokemon dessa classe e outras caíssem tanto de modo que o algoritmo SMOTE também não foi capaz de suprir, pois criar dados sintéticos a partir de duas instâncias não gerou resultados desejáveis. Deste modo, foi necessária uma intervenção manual no conjunto de dados  