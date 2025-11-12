import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
import joblib

dados = pd.read_csv('pokemon_alopez247.csv')

X = dados.drop(['Type_1', 'Type_2', 'Number', 'Pr_Male', 'Generation', 'hasGender', 'Name', 'hasMegaEvolution', 'isLegendary', 'Catch_Rate'], axis=1, errors='ignore')  # Removendo dados irrelevantes
y = dados['Type_1']

# Removendo valores nulos
X = X.fillna(0)
X = pd.get_dummies(X)

scaler = RobustScaler()
num_cols = ['HP', 'Attack', 'Defense', 'Sp_Atk', 'Sp_Def', 'Speed', 'Total', 'Height_m', 'Weight_kg']
X_scaled = X.copy()
X_scaled[num_cols] = scaler.fit_transform(X_scaled[num_cols])

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.15, random_state=42, stratify=y
)

# Balanceando classes via SMOTE
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
print(f"Antes do SMOTE: {y_train.value_counts()}")
print(f"Depois do SMOTE: {y_train_balanced.value_counts()}")

X_train = scaler.fit_transform(X_train_balanced)

modelo = MLPClassifier(
    hidden_layer_sizes=(2048, 1024, 512, 256, 128),
    activation='relu',             
    solver='adam',
    batch_size=32,  
    max_iter=2000,                
    random_state=42
)

modelo.fit(X_train_balanced, y_train_balanced)
y_pred = modelo.predict(X_test)

print(classification_report(y_test, y_pred))

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

# MATRIZ DE CONFUSÃO
cm = confusion_matrix(y_test, y_pred, labels=modelo.classes_)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=modelo.classes_)
disp.plot(xticks_rotation=90, cmap='viridis')
plt.title("Matriz de Confusão - Classificação de Pokémons")
plt.savefig('resultados/matriz_confusao.png', dpi=300, bbox_inches='tight')

'''
Caso vá utilizar Oversampling ao invés do SMOTE
y_counts = y_train.value_counts()
X_train_res = X_train.copy()
y_train_res = y_train.copy()

for classe, count in y_counts.items():
    if count < 79:
        # Seleciona todos os exemplos dessa classe
        X_class = X_train[y_train == classe]
        y_class = y_train[y_train == classe]
        
        # Quantos registros faltam para atingir max_count
        n_to_add = 79 - count
        
        # Amostra aleatória com reposição
        X_oversample = X_class.sample(n=n_to_add, replace=True, random_state=42)
        y_oversample = y_class.loc[X_oversample.index]
        
        # Adiciona ao conjunto de treino
        X_train_res = pd.concat([X_train_res, X_oversample], axis=0)
        y_train_res = pd.concat([y_train_res, y_oversample], axis=0)


'''