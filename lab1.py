from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
import os
import csv
from numpy import asarray
from numpy import savetxt
from numpy import unique, amax, amin, bincount
from numpy import ndarray
from itertools import chain


# Zapisywanie wartości w set-ach - wyciąganie unikalnych wartości
def get_unique_values(data_array):
    unique_values = unique(data_array)
    for i in unique_values:
        print(i, end='\t')


# Funkcja zapisująca dane w plikacj csv
def print_to_csv(data_array, file_name):
    # Sprawdź czy istnieje folder out, jeśli nie to utwórz go 
    dir = os.path.join(os.getcwd(), "out")

    if not os.path.exists(dir):
        os.mkdir(dir)

    file_to_save = os.path.join(os.getcwd(), "out", file_name)
    savetxt(fname=file_to_save, X=data_array, delimiter=',')


# Wartość średnia, bez null-i, ilość wartości null
def get_mean_value(data_array):
    total_sum = 0
    total_elements = 0
    total_null_elements = 0
    for i in data_array:
        if i is not None:
            total_sum += i
            total_elements += 1
        else:
            total_null_elements += 1
    mean = total_sum / total_elements
    print("Wartość średnia elementów: " + str(mean))
    print("Ilość elementów null: " + str(total_null_elements))


# Wartość maksymalna i minimalna w zbiorze
def get_max_and_min_vals(data_array):
    print("Wartość maksymalna: " + str(amax(data_array)))
    print("Wartość minimalna: " + str(amin(data_array)))


# Wyznaczanie dominanty
def get_most_frequent_val(data_array):
    # Data array jest dwu-wymiarowa - trzeba ją "spłaszczyć" do tablicy 1-wymiarowej, aby móc wyznaczyć dominantę
    flatten_array = ndarray.flatten(data_array)
    converted_array = []

    # Dane w "spłaszczonej" tablicy mogą mieć różne typy - trzeba je zrzutować na jeden typ
    for i in flatten_array:
        converted_array.append(int(i))
    print("Wartość najczęściej występująca w zbiorze (dominanta): " + str(bincount(converted_array).argmax()))


# Przygotowywanie outputa
def prepare_output(data_array):
    print("Rozmiar danych: " + str(len(data_array)))
    print("Wartości unikatowe:")
    get_unique_values(data_array)
    get_mean_value(data_array)
    get_max_and_min_vals(data_array)
    get_most_frequent_val(data_array)


# Generujemy zestaw danych
X, y = load_wine(return_X_y=True)

# Wyświwtlenie całej struktury zwracanej przez metodę load_wine zad. 1
print(load_wine)

# Przetwarzanie danych za pomocą train_test_split - zadanie 2

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)

print(y_train)
print(X_train)
print("Długość tablicy X_test: ", len(X_test))

# Zapis tablic do poszczególnych plików w module "out"

print_to_csv(X_train, "X_train.csv")
print_to_csv(y_train, "y_train.csv")
print_to_csv(X_test, "X_test.csv")
print_to_csv(y_test, "y_test.csv")

# Zadanie 3 - analiza ilościowa zbiorów

print("Zbiór X_train:\n")
prepare_output(X_train)
print("\n\n")
print("Zbiór y_train:\n")
prepare_output(y_train)
print("\n\n")
print("Zbiór X_test:\n")
prepare_output(X_test)
print("\n\n")
print("Zbiór y_test:\n")
prepare_output(y_test)
print("\n\n")
