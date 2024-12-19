
La classe `Result` calcule différentes statistiques de performance: %%différentes méthodes pour le plotting%%
- Performance totale
\[
\text{Performance totale} = \prod_{t=1}^T (1 + r_t) - 1
\]

- Performance annualisée
\[
\text{Performance annualisée} = \left( \prod_{t=1}^T (1 + r_t) \right)^{\frac{N}{T}} - 1
\]

- Facteur de profitabilité
\[
\text{Facteur de profitabilité} = \frac{\text{Gain total}}{\text{Perte totale}}
\]

- Volatilité annualisée
\[
\text{Volatilité} = \sigma(r) \cdot \sqrt{N}
\]

- Ratio de Sharpe
\[
\text{Ratio de Sharpe} = \frac{\bar{r} - r_f}{\sigma(r)}
\]

- Drawdown maximum
\[
\text{Maximum Drawdown} = \min \left( \frac{C_t - \max(C_{1:t})}{\max(C_{1:t})} \right)
\]

- Ratio de Sortino
\[
\text{Ratio de Sortino} = \frac{\bar{r} - r_f}{\sigma_{\text{négatif}}(r)}
\]

- VaR (Value at Risk) à 95%
\[
\text{VaR}_{95\%} = \text{Quantile}_{5\%}(-r)
\]

- CVaR (Conditional Value at Risk) à 95%
\[
\text{CVaR}_{95\%} = \mathbb{E}[r \mid r \leq \text{VaR}_{95\%}]
\]

- Gain/Pertes moyen (Profit/Loss Ratio)
\[
\text{Profit/Loss Ratio} = \frac{\text{Gain moyen}}{\text{Perte moyenne}}
\]

- Nombre de trades
\[
\text{Nombre de trades} = \text{Nombre total d’observations de rendements}
\]

- Pourcentage de trades gagnants
\[
\text{Pourcentage de trades gagnants} = \frac{\text{Nombre de trades gagnants}}{\text{Nombre total de trades}} \cdot 100
\]




## Description du projet

Conception d'un framework de backtesting permettant d'évaluer et de comparer différentes stratégies d'investissement sur des données historiques. Ce framework vous aidera à implémenter, tester et analyser diverses stratégies de trading, en fournissant des métriques de performance et des visualisations pour faciliter la prise de décision. L'objectif est de créer un outil flexible et extensible qui permet aux utilisateurs de développer et d'évaluer leurs propres stratégies d'investissement.

## Consignes

1. **Données** :
   - Le framework doit accepter des données d'entrée au format CSV ou Parquet.
   - Les étudiants sont responsables de fournir leurs propres données. Ils peuvent utiliser des données réelles (par exemple, crypto-monnaies, actions, ou autres actifs financiers) ou générer des données synthétiques.
   - Le framework doit pouvoir accepter les données soit sous forme de DataFrame pandas, soit comme un chemin vers un fichier CSV ou Parquet.
   - Le framework n'a pas besoin de récupérer les données lui-même ; elles doivent être fournies par l'utilisateur.

2. **Implémentation du backtester** :
   - Créez une classe abstraite `Strategy` avec les contraintes suivantes :
     - Méthode obligatoire : `get_position(historical_data, current_position)`
     - Méthode optionnelle : `fit(data)` (par défaut, cette méthode ne fait rien si elle n'est pas implémentée)
   - Permettez la création de stratégies soit par héritage de la classe abstraite, soit par un décorateur pour les stratégies simples ne nécessitant que `get_position`.
   - Implémentez une classe `Backtester` qui :
     - Est instanciée avec une série de données d'entrée
     - Possède une méthode pour exécuter le backtest en prenant une instance de `Strategy`
     - Renvoie une instance de la classe `Result` après l'exécution du backtest
   - Implémentez une classe `Result` pour afficher et visualiser les résultats du backtest, avec différentes méthodes pour le plotting et l'affichage des statistiques de performance, incluant (mais non limité à) :
     - Performance totale et annualisée
     - Volatilité
     - Ratio de Sharpe
     - Drawdown maximum
     - Ratio de Sortino
     - Nombre de trades
     - Pourcentage de trades gagnants
   - Ajoutez une fonction `compare_results(result_1, result_2, ...)` pour comparer les résultats de différentes stratégies.
   - Implémentez la possibilité de choisir le backend pour les visualisations (matplotlib par défaut, avec options pour seaborn et plotly).
   - Permettez de spécifier une fréquence de rééquilibrage pour chaque stratégie.

3. **Fonctionnalités du framework** :
   - Permettez à l'utilisateur de définir et de tester facilement de nouvelles stratégies.
   - Offrez la possibilité de comparer plusieurs stratégies entre elles.
   - Implémentez des fonctionnalités pour gérer les frais de transaction et le slippage.
   - Le framework doit permettre de créer et de tester des stratégies sur un ou plusieurs actifs simultanément.

4. **Structuration du code** :
   - Adoptez une approche orientée objet pour la structure de votre projet.
   - Divisez votre code en modules et classes distincts pour chaque fonctionnalité.
   - Commentez et documentez votre code pour faciliter sa compréhension.
   - Créez un fichier `pyproject.toml` pour permettre l'installation via pip.
   - Incluez des tests unitaires et d'intégration pour votre code.

5. **Exemple d'utilisation** :
   - Fournissez un notebook Jupyter (ou équivalent) qui démontre l'utilisation du package.
   - Ce notebook doit inclure un exemple complet d'utilisation du framework, de la création d'une stratégie à l'analyse des résultats.
   - Assurez-vous que l'exemple montre les principales fonctionnalités du framework et inclut des visualisations des résultats.

## Critères d'évaluation

1. **Structure du code** :
   - Organisation et modularité du code.
   - Utilisation appropriée des concepts de programmation orientée objet vus en cours.
   - Clarté et propreté du code (pas de redondance, noms de variables explicites, etc.).

2. **Documentation** :
   - Présence de commentaires explicatifs.
   - Qualité et clarté de la documentation fournie (par exemple, docstrings pour les classes et méthodes).

3. **Fonctionnalités** :
   - Implémentation correcte du backtester et des différentes composantes du framework.
   - Flexibilité et extensibilité du framework pour accommoder diverses stratégies.
   - Capacité à gérer des stratégies sur un ou plusieurs actifs.
   - Qualité des visualisations et des résultats produits.

4. **Tests** :
   - Couverture et pertinence des tests unitaires et d'intégration.

5. **Installation et déploiement** :
   - Configuration correcte du fichier `pyproject.toml`.
   - Facilité d'installation et d'utilisation du package.

6. **Exemple d'utilisation** :
   - Qualité et exhaustivité du notebook d'exemple fourni.
   - Clarté des explications et pertinence des exemples choisis.

## Conseils

- Commencez par implémenter des stratégies simples (par exemple, une moyenne mobile croisée) pour tester votre framework avant de passer à des stratégies plus complexes.
- Pensez à l'efficacité de votre code, surtout pour les backtests sur de grandes quantités de données historiques.
