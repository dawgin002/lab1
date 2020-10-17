from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
import os

data = load_wine()

# Wyświwtlenie całej struktury data 
print(data.data)

# Sprawdź czy istnieje folder out, jeśli nie to utwórz go 
dir = os.path.join(os.getcwd(), "out")

print(os.getcwd())

if not os.path.exists(dir):
    os.mkdir(dir)
