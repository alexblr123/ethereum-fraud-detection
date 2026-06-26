#!/usr/bin/env Rscript
# ---------------------------------------------------------------------------
# Demo wdrożenia: wczytuje zapisany model i scoruje konta.
# Uruchom z katalogu repo:
#
#     Rscript predict.R
#
# fraud_model.rds to w pełni dopasowany workflow tidymodels (recipe + model),
# więc przyjmuje surowe cechy konta i sam robi preprocessing.
# ---------------------------------------------------------------------------

suppressPackageStartupMessages({
  library(tidymodels)
  library(rio)
  library(janitor)
})
tidymodels_prefer()

if (!file.exists("fraud_model.rds")) {
  stop("Brak fraud_model.rds — najpierw wyrenderuj main.qmd.")
}
fraud_model <- readRDS("fraud_model.rds")
cat("Wczytano model z fraud_model.rds\n\n")

# Kilka kont jako "nowe" dane wejściowe (tu po prostu próbka ze zbioru).
df <- import("transaction_dataset.csv") %>%
  select(-1, -Index, -Address) %>%
  janitor::clean_names() %>%
  select(-any_of(c("erc20_most_sent_token_type", "erc20_most_rec_token_type")))

set.seed(2026)
sample_accounts <- df %>% slice_sample(n = 10)
truth <- factor(sample_accounts$flag, levels = c(1, 0),
                labels = c("fraud", "legit"))

preds <- predict(fraud_model, new_data = sample_accounts, type = "prob") %>%
  bind_cols(predict(fraud_model, new_data = sample_accounts)) %>%
  mutate(actual = truth) %>%
  select(actual, predicted = .pred_class, p_fraud = .pred_fraud)

cat("Test predykcyjny na 10 kontach:\n\n")
print(preds, n = 10)

cat("\nUwaga: domyślny próg 0,5 nie jest tu optymalny — obniżenie go w stronę\n")
cat("maksimum F1 podnosi recall kosztem precyzji.\n")
