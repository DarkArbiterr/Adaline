# Adaline - Rozpoznawanie odręcznie pisanych liter

## Program
Aplikacja została napisana w języku **Python**, głównie przy użyciu bilbiotek *pygame*, *tensorflow*, *numpy* i *matplotlib*.
Program wykorzystuje **modele neuronowe Adaline** (własna implementacja) do nauki rozpoznawania odręcznie pisanych cyfr z bazy
danych *mnist* z biblioteki *keras.datasets*.

> [!NOTE]
> **Model Adaline** - starszy kolega perceptronu. W Adaline, wyjście jest obliczane jako suma ważona wejść, ale zamiast funkcji skokowej, używa się funkcji liniowej.
> Kluczową cechą Adaline jest zdolność do dostosowywania wag na podstawie błędów wyjścia, co sprawia, że jest to model uczenia nadzorowanego.

Najpierw program przeprowadza uczenie 10 modelów Adaline na zbiorze treningowym, następnie wyświetla **wykresy zmiany błędu** na przestrzeni 15 iteracji uczenia.
Ostatecznie następuje przetestowanie modelu na zbiorze treningowym i testowym, a na terminal wypisana zostaje liczba poprawnie rozpoznanych cyfr.

![Zrzut ekranu 2024-02-03 152050](https://github.com/DarkArbiterr/Adaline/assets/75552617/a655f3e5-f083-4697-97e7-fb3809263df3)
![Zrzut ekranu 2024-02-03 152150](https://github.com/DarkArbiterr/Adaline/assets/75552617/2024265d-be40-4a9e-ae2c-6140d36231e0)

> [!WARNING]
> Uczenie odbywa się na bardzo dużym zbiorze i na parunastu iteracjach, z tego powodu ten proces może trwać około 5-10 minut (na początku również wczytywane są dane co też trwa trochę).
> Po wczytaniu danych na terminalu wyświetla się która jest iteracja i który model jest uczony, w celu monitorowania postępów.

## Dane
Ze zbioru *mnist* zostało pobrane 60000 przykładów uczących i 10000 testowych. Przykłady są w postaci czarnobiałych obrazów **28x28**. Zostały one wpierw **spłaszczone** do 
jednowymiarowej macierzy po czym **sprogowane** w celu poprawnej klasyfikacji 0/1:
> **Piksele > 128** w macierzy zyskują wartość 1 (kolor ciemniejszy)\
> **Piksele =< 128** w macierzy zyskują wartość 0 (jaśniejszy kolor)

## Model Adaline
Model Adaline wykorzystuje **algorytm gradientowy** do uczenia.

**Inicjalizacja**: klasa jest inicjalizowana z liczbą wejść (*nrOfInputs*), opcjonalnym *biasem*, liczbą iteracji (*iterations*), i współczynnikiem nauki (*learningEta*).

**Losowe wagi**: wagi są inicjowane losowymi wartościami, z uwzględnieniem opcjonalnego *biasu*.

**Funkcja ucząca**: iteruje przez zadaną liczbę iteracji, mieszając dane na każdym kroku. Aktualizuje wagi na podstawie błędu i danych wejściowych.

**Funkcja aktywacji**: wykorzystuje **sigmoidalną** funkcję aktywacji do obliczenia wyniku.
```math
 \sigma(x) = \frac{1}{1 + e^{-x}}
```

**Trasformata Fouriera**: funkcja używa biblioteki *NumPy* do wykonania transformaty Fouriera na danych wejściowych. 
W rezultacie otrzymujemy **znormalizowaną** reprezentację danych wejściowych.

**Wyjście modelu**: oblicza wynik na podstawie danych wejściowych i przekształconych danych Fouriera.
