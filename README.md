## Description du projet

Conception d'un framework de backtesting permettant d'évaluer et de comparer différentes stratégies d'investissement sur des données historiques. Ce framework permet de tester et analyser diverses stratégies de trading, en fournissant des métriques de performance et des visualisations pour faciliter la prise de décision. 
L'outil permet aux utilisateurs de développer et d'évaluer leurs propres stratégies d'investissement.

**Fonctionnalités du framework** : L'utilisateur peut 
- définir et tester facilement de nouvelles stratégies.
- comparer plusieurs stratégies entre elles.

%%%% A voir
- Implémentez des fonctionnalités pour gérer les frais de transaction et le slippage.
- Le framework doit permettre de créer et de tester des stratégies sur un ou plusieurs actifs simultanément.

## Données:
   - Le framework accepte des données d'entrée au format CSV ou Parquet.
   - Les données doivent être fournies par l'utilisateur.

%%%% 

## Implémentation du backtester:

### Classe `Strategy`

- Méthode obligatoire : `get_position(historical_data, current_position)`
- Méthode optionnelle : `fit(data)` (par défaut, cette méthode ne fait rien si elle n'est pas implémentée)

- Permettez la création de stratégies soit par héritage de la classe abstraite, soit par un décorateur pour les stratégies simples ne nécessitant que `get_position`.

### Classe `Backtester`

La classe `Backtester` est instanciée avec une série de données d'entrée
- Possède une méthode pour exécuter le backtest en prenant une instance de `Strategy`
- Renvoie une instance de la classe `Result` après l'exécution du backtest

### Classe `Result` 

La classe `Result` calcule différentes statistiques de performance: %%différentes méthodes pour le plotting%%

- $\text{Performance totale} = \prod_{t=1}^T (1 + r_t) - 1$

- $\text{Performance annualisée} = \left( \prod_{t=1}^T (1 + r_t) \right)^{\frac{N}{T}} - 1$

- $\text{Facteur de profitabilité} = \frac{\text{Gain total}}{\text{Perte totale}}$

- $\text{Volatilité annualisée} = \sigma(r) \cdot \sqrt{N}$

- $\text{Ratio de Sharpe} = \frac{\bar{r} - r_f}{\sigma(r)}$

- $\text{Maximum Drawdown} = \min \left( \frac{C_t - \max(C_{1:t})}{\max(C_{1:t})} \right)$

- $\text{Ratio de Sortino} = \frac{\bar{r} - r_f}{\sigma_{\text{négatif}}(r)}$

- $\text{VaR (Value at Risk) à 95\%} = \text{Quantile}_{0.05}(-r)$

- $\text{CVaR (Conditional Value at Risk) à 95\%} = \mathbb{E}[r \mid r \leq \text{VaR}_{0.95}]$

- $\text{Profit/Loss Ratio} = \frac{\text{Gain moyen}}{\text{Perte moyenne}}$

- $\text{Nombre de trades} = \text{Nombre total d’observations de rendements}$

- $\text{Pourcentage de trades gagnants} = \frac{\text{Nombre de trades gagnants}}{\text{Nombre total de trades}} \cdot 100$

### Fonction `compare_results(result_1, result_2, ...)` 
pour comparer les résultats de différentes stratégies.
- Implémentez la possibilité de choisir le backend pour les visualisations (matplotlib par défaut, avec options pour seaborn et plotly).
- Permettez de spécifier une fréquence de rééquilibrage pour chaque stratégie.



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
- Pensez à l'efficacité de votre code, surtout pour les backtests sur de grandes quantités de données historiques.
