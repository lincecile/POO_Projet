import pandas as pd
from typing import Dict, Tuple, Union
import matplotlib.pyplot as plt
from .Strategy_Class import Strategy
from .Backtester_Class import Backtester
from .Result_Class import Result
import plotly.graph_objects as go


class Strategy_Manager:
    """
    Classe pour gérer et comparer plusieurs stratégies de trading.
    Permet de gérer les backtests, les visualisations et les comparaisons de plusieurs stratégies.
    """
    
    def __init__(self, data: pd.DataFrame, strategies_dict: Dict[str, Tuple[Strategy, float, float]] = None):
        """
        Initialise le Strategy_Manager.
        
        Args:
            data: DataFrame contenant les données de marché.
            strategies_dict: Dictionnaire des stratégies au format :
                             {nom: (instance_stratégie, coûts_de_transaction, slippage)}
        """
        self.data = data
        self.strategies: Dict[str, Tuple[Strategy, float, float]] = {}
        self.results: Dict[str, Result] = {}
        
        if strategies_dict:
            self.add_strategies_from_dict(strategies_dict)
    
    def add_strategies_from_dict(self, strategies_dict: Dict[str, Tuple[Strategy, float, float]]) -> None:
        """
        Ajoute des stratégies depuis un dictionnaire créé par l'utilisateur.
        
        Args:
            strategies_dict: Dictionnaire au format :
                             {nom: (instance_stratégie, coûts_de_transaction, slippage)}
        """
        for name, (strategy, costs, slip) in strategies_dict.items():
            self.add_strategy(name, strategy, costs, slip)
        
    def add_strategy(self, name: str, strategy: Strategy, transaction_costs: float = 0.001, slippage: float = 0.0005) -> None:
        """
        Ajout d'une stratégie au gestionnaire.
        
        Args:
            name: Identifiant unique pour la stratégie.
            strategy: Instance de la stratégie.
            transaction_costs: Coûts de transaction pour la stratégie.
            slippage: Coûts liés au slippage pour la stratégie.
        """
        if name in self.strategies:
            raise ValueError(f"Une stratégie existe déjà au nom de '{name}'")
        
        self.strategies[name] = (strategy, transaction_costs, slippage)
        
    def remove_strategy(self, name: str) -> None:
        """
        Supprimer une stratégie du gestionnaire.
        
        Args:
            name: Nom de la stratégie à supprimer.
        """
        if name in self.strategies:
            del self.strategies[name]
            if name in self.results:
                del self.results[name]
                
    def run_backtests(self) -> None:
        """Exécute les backtests pour toutes les stratégies enregistrées."""
        self.results.clear()

        for name, (strategy, costs, slip) in self.strategies.items():
            backtester = Backtester(self.data, transaction_costs=costs, slippage=slip)
            self.results[name] = backtester.exec_backtest(strategy)
            
    def get_statistics(self, strategy_name: Union[str, None] = None) -> Union[Dict, pd.DataFrame]:
        """
        Obtenir les statistiques pour une stratégie ou pour toutes les stratégies.
        
        Args:
            strategy_name: Nom d'une stratégie spécifique, ou None pour toutes les stratégies.
            
        Returns:
            DataFrame des statistiques pour toutes les stratégies.
        """
        if not self.results:
            raise ValueError("Il faut lancer le backtest d'abord.")
            
        if strategy_name is not None:
            if strategy_name not in self.results:
                raise ValueError(f"Pas de stratégie à ce nom : '{strategy_name}'")
            return self.results[strategy_name].statistics
        
        return pd.DataFrame([result.statistics for result in self.results.values()], index=self.results.keys())
    
    def plot_strategy(self, strategy_name: str, backend: str = 'matplotlib', include_costs: bool = True) -> None:
        """
        Graphique des résultats pour une stratégie spécifique.
        
        Args:
            strategy_name: Nom de la stratégie à tracer.
            backend: Backend de visualisation ('matplotlib' ou 'plotly').
            include_costs: Inclure ou non les coûts de transaction dans la visualisation.
        """
        if strategy_name not in self.results:
            raise ValueError(f"Pas de backtest à ce nom : '{strategy_name}'")
            
        self.results[strategy_name].plot(
            name_strat=strategy_name,
            backend=backend,
            include_costs=include_costs
        )
    
    def compare_strategies(self, backend: str = 'matplotlib') -> None:
        """
        Comparer toutes les stratégies en utilisant la fonction compare_results.
        
        Args:
            backend: Backend de visualisation ('matplotlib' ou 'plotly').
        """
        if not self.results:
            raise ValueError("Pas de backtest sur les stratégies.")
        
        # Extraction des noms de stratégies
        strategy_names = list(self.results.keys())
        stats_comparison = pd.DataFrame([r.statistics for r in self.results.values()], index=strategy_names)
        
        num_trades = stats_comparison['num_trades']
        other_metrics = stats_comparison.drop(columns=['num_trades'])

        if backend == 'matplotlib':
            fig, ax1 = plt.subplots(figsize=(12, 6))
            x = range(len(strategy_names))
            
            # Ajout explicite des stratégies sur l'axe X
            ax1.bar(x, num_trades, color='skyblue', alpha=0.7, label='Num Trades')
            ax1.set_xticks(x)
            ax1.set_xticklabels(strategy_names, rotation=45, ha='right')
            ax1.set_ylabel('Nombre de Trades')
            ax1.legend(loc='upper left')

            ax2 = ax1.twinx()
            other_metrics_normalized = other_metrics / other_metrics.abs().max()
            for column in other_metrics_normalized.columns:
                ax2.bar(x, other_metrics_normalized[column], alpha=0.5, label=column)

            ax2.set_ylabel('Métriques Normalisées (Autres)')
            ax2.legend(loc='upper right', fontsize=8)

            plt.title('Comparaison des Stratégies avec toutes les Stratégies affichées')
            plt.tight_layout()
            plt.show()

        elif backend == 'plotly':
            fig = go.Figure()

            # Ajout des barres pour 'num_trades'
            fig.add_trace(go.Bar(x=strategy_names, y=num_trades, name='Num Trades', marker_color='skyblue'))

            # Ajout des barres pour les autres métriques
            for column in other_metrics.columns:
                fig.add_trace(go.Bar(x=strategy_names, y=other_metrics[column] / other_metrics.abs().max(), name=column))

            fig.update_layout(
                title='Comparaison des Stratégies avec toutes les Stratégies affichées',
                xaxis_title='Stratégies',
                yaxis_title='Valeurs Normalisées',
                barmode='group',
                legend=dict(title='Métriques'),
                height=600
            )
            fig.show()
    
    def print_statistics(self, strategy_name: Union[str, None] = None) -> None:
        """
        Afficher les statistiques pour toutes les stratégies.
        
        Args:
            strategy_name: Nom d'une stratégie spécifique, ou None pour toutes les stratégies.
        """
        stats = self.get_statistics(strategy_name)
        
        if isinstance(stats, dict):
            print(f"\nStatistiques de la strategie '{strategy_name}':")
            for key, value in stats.items():
                print(f"{key}: {value:.4f}")
        else:
            print("\nStatistiques des stratégies:")
            print(stats.round(4))
