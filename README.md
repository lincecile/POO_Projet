## Description du projet

Conception d'un framework de backtesting permettant d'évaluer et de comparer différentes stratégies d'investissement sur des données historiques. Ce framework permet de tester et analyser diverses stratégies de trading, en fournissant des métriques de performance et des visualisations pour faciliter la prise de décision. 
L'outil permet aux utilisateurs de développer et d'évaluer leurs propres stratégies d'investissement.

**Fonctionnalités du framework** : L'utilisateur peut 
- Définir et tester facilement de nouvelles stratégies sur un ou plusieurs actifs simultanément.
- Comparer plusieurs stratégies entre elles.
- Définir des frais de transaction et de slippage.
- Créer des stratégies par héritage de la classe abstraite ou par un décorateur.
- Spécifier une fréquence de rebalancement pour chaque stratégie.
- Créer et de tester des stratégies 

## Données:
   - Le framework accepte des données d'entrée au format CSV ou Parquet.
   - Les données doivent être fournies par l'utilisateur.

## Implémentation du backtester:

### Classe abstraite `Strategy`

La classe `Strategy` possède :
- Méthode obligatoire : `get_position(historical_data, current_position)`
- Méthode optionnelle : `fit(data)`

### Classe `Backtester`

La classe `Backtester` est instanciée avec une série de données d'entrée :
- Possède une méthode pour exécuter le backtest en prenant une instance de `Strategy`
- Renvoie une instance de la classe `Result` après l'exécution du backtest

### Classe `Result` 

La classe `Result` calcule différentes statistiques de performance: 

- $\text{Performance totale} = \prod_{t=1}^T (1 + r_t) - 1$

- $\text{Performance annualisée} = \left( \prod_{t=1}^T (1 + r_t) \right)^{\frac{252}{T}} - 1$

- $\text{Facteur de profitabilité} = \frac{\text{Gain total}}{\text{Perte totale}}$

- $\text{Volatilité annualisée} = \sigma(r) \cdot \sqrt{252}$

- $\text{Ratio de Sharpe} = \frac{\text{Annual returns}}{\text{Volatilité annualisée}}$

- $\text{Maximum Drawdown} = \min \left( \frac{C_t - \max(C_{1:t})}{\max(C_{1:t})} \right)$
Où $C_t$ est la valeur du capital cumulé au temps 
$C_{1:t}$ est la valeur maximale du capital cumulé jusqu'au temps t.

- $\text{Ratio de Sortino} = \frac{\text{Annual returns}}{\text{Volatilité des returns négatifs}}$

- $\text{VaR (Value at Risk) à 95\%} = \text{Quantile}_{0.05}(r)$

- $\text{CVaR (Conditional Value at Risk) à 95\%} = \frac{1}{N_{0.05}}*\sum_{r_i \leq \text{Quantile}_{0.05}}(r_i)$

- $\text{Profit/Loss Ratio} = \frac{\text{Gain moyen}}{\text{Perte moyenne}}$

- $\text{Nombre de trades} = \text{Nombre total d’observations de rendements}$

- $\text{Pourcentage de trades gagnants} = \frac{\text{Nombre de trades gagnants}}{\text{Nombre total de trades}} \cdot 100$

### Fonction `compare_results(result_1, result_2, ...)` 

Cette fonction permet de comparer les résultats de différentes stratégies de manière graphique.
- L'utilisateur a la possibilité de choisir le backend pour les visualisations (matplotlib, seaborn ou plotly).

## Flexibilité et extensibilité du framework:

### Classe `DataFileReader`

La classe `DataFileReader` permet à l'utilisateur:
- Lire un fichier csv ou parquet avec une même méthode `read_file(filepath, date_column)`.

### Classe `Strategy_Manager` 

La classe `Strategy_Manager` facilite l'utilisation à grande échelle du backtester, sur un grand nombre de stratégie avec différentes méthodes:
- `run_backtests()`: (Voir le notebook avec plusieurs stratégies)  
   - Cette méthode lance le backtest pour chaque stratégie.
      - Chaque stratégie peut avoir un coût de transaction et de slippage unique: l'utilisateur entre alors un float.
      - Chaque stratégie peut avoir des coûts de transaction et de slippage différents: l'utilisateur entre alors un dictionnaire de coût avec en clé les differents actifs et en valeurs les différents coûts.
      - Si l'utilisateur n'indique rien alors un coût pas défaut est appliqué.

- `plot_all_strategies(backend, include_costs)`: cette méthode permet d'afficher toutes les stratégies testées.
   - L'utilisateur peut choisir
      - différent backend pour le graphique
      - d'inclure ou non les coûts d'execution sur les graphiques

- `plot_strategy(strategy_name, backend, include_costs)`: cette méthode permet d'afficher une graphique précis si plusieurs stratégie ont été testé.
   - L'utilisateur peut choisir
      - différent backend pour le graphique
      - d'inclure ou non les coûts d'execution sur les graphiques

- `compare_strategies(backend, show_plot)`: cette méthode permet d'afficher toutes les métriques sous forme d'histogramme.
   - Afin de facilité la visualisation des backends `matplotlib` et `seaborn`, deux axes d'échelles a été mis en place sur le graphe.

- `print_statistics(strategy_name, detail)`: cette méthode permet d'afficher toutes les métriques sous forme de tableau.
   - L'utilisateur peut choisir :
      - d'afficher un tableau sur les statistiques d'une stratégie précise.
      - d'afficher un tableau détaillé sur les statistiques actifs par actifs de chaque stratégie.
      - d'afficher un tableau détaillé sur les statistiques actifs par actifs d'une stratégie précise.

- `remove_strategy(name)`: cette méthode permet de supprimer une stratégie précise. 

            



  

