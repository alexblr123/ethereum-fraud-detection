# Ethereum Fraud Detection

Wykrywanie fraudowych kont w sieci Ethereum na podstawie zagregowanych cech transakcyjnych. Pełny pipeline ML w R / tidymodels: czyszczenie danych, selekcja cech, obsługa niezbalansowania klas, porównanie modeli, kalibracja, analiza błędów i wyjaśnialność.

Projekt zaliczeniowy z przedmiotu *Eksploracja danych* (Politechnika Lubelska).

## Problem

Binarna klasyfikacja: na podstawie statystyk aktywności konta Ethereum (liczba i częstotliwość transakcji, wartości przelewów w ETH, saldo, statystyki tokenów ERC20) przewidujemy, czy konto jest fraudowe (`FLAG = 1`), czy legalne (`FLAG = 0`). Fraud to ~22% obserwacji, więc zbiór jest niezbalansowany (~1:3,5).

## Dane

**Ethereum Fraud Detection Dataset** (Kaggle) — 9841 kont × 51 kolumn. Źródło: Farrugia, Ellul & Azzopardi (2020), *Detection of illicit accounts over the Ethereum blockchain*, Expert Systems with Applications, 150, 113318.

## Co jest w środku

Pipeline (`main.qmd`) przechodzi przez:

1. Czyszczenie i selekcję cech — usunięcie identyfikatorów, analiza braków (braki ERC20 są strukturalne → 0), usunięcie cech o zerowej wariancji, redukcja współliniowości (`|r| > 0.9`).
2. EDA — balans klas, macierz korelacji, rozkłady cech wg klasy, scatter aktywności.
3. Przygotowanie danych — podział 80/20 ze stratyfikacją *przed* transformacjami; imputacja, normalizacja i SMOTE w recipe (bez wycieku danych).
4. Modelowanie — cztery rodziny: regresja logistyczna (L1), las losowy, XGBoost (strojone) + MLP jako baseline. Strojenie na 5-krotnej walidacji krzyżowej.
5. Ewaluacja i kalibracja — metryki train vs test, krzywe ROC i Precision-Recall, macierze pomyłek, wykres kalibracji.
6. Analiza błędów — false negatives vs false positives, rozkład P(fraud), dobór progu decyzyjnego.
7. Wyjaśnialność — ważność zmiennych (VIP) dla obu modeli drzewiastych.
8. Wdrożenie — zapis finalnego modelu (`fraud_model.rds`) + test predykcyjny.

## Najważniejsze wyniki

- Modele drzewiaste (XGBoost, las losowy) dają najlepsze wyniki — ROC AUC na poziomie literatury, wyraźnie ponad regresją i MLP.
- Najsilniejsze predyktory: cechy ERC20 oraz statystyki czasowe i wolumenowe — zgodnie z Farrugia i in. (2020).
- Domyślny próg 0,5 nie jest optymalny: przeoczony fraud kosztuje więcej niż fałszywy alarm, więc warto obniżyć próg w okolice maksimum F1.

Dokładne liczby są w wyrenderowanym raporcie.

## Stack

R · tidymodels (recipes, parsnip, tune, yardstick, workflows) · themis (SMOTE) · glmnet · ranger · xgboost · nnet · vip · probably · Quarto

## Pliki

- `main.qmd` — pełna analiza (Quarto → raport HTML)
- `prezentacja.qmd` / `prezentacja.html` — slajdy (Reveal.js)
- `transaction_dataset.csv` — dane
- `fraud_model.rds` — zapisany finalny model
- `predict.R` — demo: wczytanie modelu i scoring kont

## Uruchomienie

Wymaga R (≥ 4.2) i [Quarto](https://quarto.org/).

```bash
# Pakiety (raz)
Rscript -e 'install.packages(c("rio","tidyverse","tidymodels","themis","vip","corrplot","probably","janitor","glmnet","ranger","xgboost","nnet"))'

# Render raportu i slajdów
quarto render main.qmd
quarto render prezentacja.qmd

# Demo predykcji
Rscript predict.R
```

Przełącznik `FAST` na górze `main.qmd`: `TRUE` = szybki render (mniejsze siatki), `FALSE` = pełne siatki hiperparametrów.

## Dalsze kierunki

Metody grafowe (GNN) wykorzystujące graf transakcji zamiast cech zagregowanych; lepsze przetwarzanie wyrzuconych kolumn typu tokenu ERC20 (grupowanie rzadkich kategorii, target encoding); przeliczenie wartości ERC20 na wspólną walutę po kursach historycznych; walidacja na nowszych danych.
