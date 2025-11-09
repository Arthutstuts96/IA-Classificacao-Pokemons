import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import accuracy_score, classification_report
import joblib

dados = pd.read_csv('pokemon_alopez247.csv')

X = dados.drop('Type_1', axis=1)
y = dados['Type_1']

X = pd.get_dummies(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

modelo = RandomForestClassifier(n_estimators=100, random_state=42)
modelo.fit(X_train, y_train)


y_pred = modelo.predict(X_test)
print("Acurácia:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

acertos_tipo1 = 0
acertos_tipo1_ou_tipo2 = 0

# Queremos verificar a acurácia tanto para acertar o tipo principal quanto para o tipo secundário
for real_tipo1, real_tipo2, pred in zip(dados.loc[y_test.index, 'Type_1'], dados.loc[y_test.index, 'Type_2'], y_pred):
    if pred == real_tipo1:
        acertos_tipo1 += 1
        acertos_tipo1_ou_tipo2 += 1
    elif pd.notna(real_tipo2) and pred == real_tipo2:
        acertos_tipo1_ou_tipo2 += 1

acc_tipo1 = acertos_tipo1 / len(y_test)
acc_tipo1_ou_tipo2 = acertos_tipo1_ou_tipo2 / len(y_test)

print(f"Acurácia tipo primário exato: {acc_tipo1:.2%}")
print(f"Acurácia tipo primário OU secundário: {acc_tipo1_ou_tipo2:.2%}")


