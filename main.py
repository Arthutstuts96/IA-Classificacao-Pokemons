# TODO: Remover alguns parâmetros desnecessários para avaliar o tipo
# TODO: Mexer na função de classificação para melhorar acurácia

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

dados = pd.read_csv('pokemon_alopez247.csv')

X = dados.drop(['Type_1', 'Type_2', 'Name'], axis=1, errors='ignore')  # Remove colunas irrelevantes
y = dados['Type_1']

# Removendo nulos
X = X.fillna(0)
X = pd.get_dummies(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

modelo = MLPClassifier(
    hidden_layer_sizes=(128, 64),  
    activation='relu',             
    solver='adam',                
    max_iter=500,                
    random_state=42
)
modelo.fit(X_train, y_train)
y_pred = modelo.predict(X_test)

# O cálculo da acurácia considera acertar tanto o tipo primário quanto o tipo secundário, para pokemons que o possuem
acertos_tipo1 = 0
acertos_tipo1_ou_tipo2 = 0

for real_tipo1, real_tipo2, pred in zip(dados.loc[y_test.index, 'Type_1'], dados.loc[y_test.index, 'Type_2'], y_pred):
    if pred == real_tipo1:
        acertos_tipo1 += 1
        acertos_tipo1_ou_tipo2 += 1
    elif pd.notna(real_tipo2) and pred == real_tipo2:
        acertos_tipo1_ou_tipo2 += 1

acc_tipo1 = acertos_tipo1 / len(y_test)
acc_tipo1_ou_tipo2 = acertos_tipo1_ou_tipo2 / len(y_test)

print(f"Acurácia tipo primário exato: {acc_tipo1:.2%}")
print(f"Acurácia tipo primário OU tipo secundário: {acc_tipo1_ou_tipo2:.2%}")