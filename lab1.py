from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
import os
import csv
from numpy import asarray
from numpy import savetxt


# Generujemy zestaw danych
X, y = load_wine(return_X_y=True)

# Wyświwtlenie całej struktury zwracanej przez metodę load_wine zad. 1
print(load_wine)

# Przetwarzanie danych za pomocą train_test_split - zadanie 2

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)

print(y_train)
print(X_train)
print("Długość tablicy X_test: ", len(X_test))

# Sprawdź czy istnieje folder out, jeśli nie to utwórz go 
dir = os.path.join(os.getcwd(), "out")

print(os.getcwd())

if not os.path.exists(dir):
    os.mkdir(dir)


def print_to_csv(data_array, file_name):
    savetxt(fname=file_name, X=data_array, delimiter=',')

