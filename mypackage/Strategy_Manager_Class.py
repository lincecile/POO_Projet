import pandas as pd
from typing import Dict, Tuple, Union
import matplotlib.pyplot as plt
from .Strategy_Class import Strategy
from .Backtester_Class import Backtester
from .Result_Class import Result, compare_results
import plotly.graph_objects as go
import seaborn as sns

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
        
        # Pour une stratégie particuliere
        if strategy_name is not None:
            if strategy_name not in self.results:
                raise ValueError(f"Pas de stratégie à ce nom : '{strategy_name}'")
            return self.results[strategy_name].statistics
        
        return pd.DataFrame([result.statistics for result in self.results.values()], index=self.results.keys())
    
    def get_statistics_detail(self, strategy_name: Union[str, None] = None) -> Union[Dict, pd.DataFrame]:
        """
        Obtenir les statistiques pour une stratégie ou pour toutes les stratégies.
        
        Args:
            strategy_name: Nom d'une stratégie spécifique, ou None pour toutes les stratégies.
            
        Returns:
            DataFrame des statistiques pour toutes les stratégies.
        """
        if not self.results:
            raise ValueError("Il faut lancer le backtest d'abord.")
        
        df = pd.DataFrame([result.statistics_each_asset for result in self.results.values()], index=self.results.keys())
        data = df.applymap(lambda x: x.to_dict())  
        data = data.stack() 
        data = data.apply(pd.Series).stack()  
        data = data.unstack(level=1)

        # Pour une stratégie particuliere, avoir le détail du portefeuille
        if strategy_name is not None:
            if strategy_name not in self.results:
                raise ValueError(f"Pas de stratégie à ce nom : '{strategy_name}'")
            return data.loc[strategy_name]

        return data
    
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
    
    def plot_all_strategies(self, backend: str = 'matplotlib', include_costs: bool = True) -> None:
        """
        Comparer toutes les stratégies en utilisant la fonction compare_results.
        
        Args:
            backend: Backend de visualisation ('matplotlib' ou 'plotly').
        """
        if not self.results:
            raise ValueError("Pas de backtest sur les stratégies.")
        
        
        if backend == 'matplotlib':
            # Graphique des rendements
            plt.figure(figsize=(12, 6))
            for name, result in self.results.items():
                
                # Les returns prennent déjà en compte les coûts
                if include_costs:
                    cumulative_returns = (1 + result.returns).cumprod()

                # Si on ne souhaite pas intégré les coûts, on recalcule les returns sans les coûts
                else:
                    price_returns = self.data.pct_change()
                    cumulative_returns = pd.DataFrame(index=price_returns.index)  # Même index que 'price_returns'
                    print(result.positions.columns)
                    for column in result.positions.columns:  # Parcourir chaque actif
                        # Calcule les rendements cumulés pour l'actif en utilisant 'price_returns' et les positions
                        cumulative_returns[column] = (1 + price_returns[column] * result.positions[column].shift(1)).cumprod()
                    cumulative_returns['portfolio'] = cumulative_returns.mean(axis=1)  # Moyenne simple, peut être modifiée pour pondération personnalisée
                    
                # print(cumulative_returns)
                print(cumulative_returns.columns)
                for column in cumulative_returns.columns:
                    plt.plot(cumulative_returns.index, cumulative_returns[column], label=f"{name} - {column}")

            plt.title('Rendements cumulatifs')
            plt.grid(True)
            plt.legend()
            
            # Graphique des positions
            plt.figure(figsize=(12, 6))
            for name, result in self.results.items():
                for column in result.positions.columns:  # Parcourir chaque colonne (actif)
                    plt.plot(result.positions.index, result.positions[column].values, label=f"{name} - {column}")
            
            plt.title('Positions')
            plt.grid(True)
            plt.legend()
            
        elif backend == 'seaborn':
            # Graphique des rendements
            plt.figure(figsize=(12, 6))
            for name, result in self.results.items():
                if include_costs:
                    cumulative_returns = (1 + result.returns).cumprod()
                else:
                    price_returns = self.data.pct_change()
                    cumulative_returns = pd.DataFrame(index=price_returns.index)  # Même index que 'price_returns'
                    for column in result.positions.columns:  # Parcourir chaque actif
                        # Calcule les rendements cumulés pour l'actif en utilisant 'price_returns' et les positions
                        cumulative_returns[column] = (1 + price_returns[column] * result.positions[column].shift(1)).cumprod()

                for column in cumulative_returns.columns:
                    sns.lineplot(data=cumulative_returns[column], label=f"{name} - {column}")

            plt.title('Rendements cumulatifs')
            
            # Graphique des positions
            plt.figure(figsize=(12, 6))
            for name, result in self.results.items():
                for column in result.positions.columns:  # Parcourir chaque colonne (actif)
                    plt.plot(result.positions.index, result.positions[column].values, label=f"{name} - {column}")
            
            plt.title('Positions')
            
        elif backend == 'plotly':
            # Graphique des rendements
            fig_returns = go.Figure()
            
            for name, result in self.results.items():
                if include_costs:
                    cumulative_returns = (1 + result.returns).cumprod()
                else:
                    price_returns = self.data.pct_change()
                    cumulative_returns = pd.DataFrame(index=price_returns.index)  # Même index que 'price_returns'
                    for column in result.positions.columns:  # Parcourir chaque actif
                        # Calcule les rendements cumulés pour l'actif en utilisant 'price_returns' et les positions
                        cumulative_returns[column] = (1 + price_returns[column] * result.positions[column].shift(1)).cumprod()

                for column in cumulative_returns.columns:
                    fig_returns.add_trace(
                        go.Scatter(
                            x=cumulative_returns.index,
                            y=cumulative_returns[column].values,
                            name=name,
                            mode='lines'
                        )
                    )
            
                    fig_returns.update_layout(
                        title='Rendements cumulatifs',
                        showlegend=True,
                        height=600
                    )
            
            # Graphique des positions
            fig_positions = go.Figure()
            
            for name, result in self.results.items():
                for column in result.positions.columns:
                    fig_positions.add_trace(
                        go.Scatter(
                            x=result.positions.index,
                            y=result.positions[column].values,
                            name=f"{name} - {column}",
                            mode='lines'
                        )
                    )
            
                fig_positions.update_layout(
                    title='Positions',
                    showlegend=True,
                    height=600
                )
            
            fig_returns.show()
            fig_positions.show()

    def compare_strategies(self, backend: str = 'matplotlib') -> None:
        """
        Comparer toutes les stratégies en utilisant la fonction compare_results.
        
        Args:
            backend: Backend de visualisation ('matplotlib', 'seaborn', ou 'plotly').
        """
        if not self.results:
            raise ValueError("Pas de backtest sur les stratégies.")
                
        fig = compare_results(self.results,backend=backend)
        
        if backend == 'plotly':
            fig.show()

    def print_statistics(self, strategy_name: Union[str, None] = None, detail=False) -> None:
        """
        Afficher les statistiques pour toutes les stratégies.
        
        Args:
            strategy_name: Nom d'une stratégie spécifique, ou None pour toutes les stratégies.
        """

        stats = self.get_statistics_detail(strategy_name) if detail else self.get_statistics(strategy_name)

        strat_name = strategy_name if strategy_name is not None else ''

        if isinstance(stats, dict):
            print(f"\nStatistiques de la strategie '{strat_name}':")
            df_choisi = pd.DataFrame(stats,index=[strat_name])
            print(df_choisi.round(4))
        else:
            print(f"\nStatistiques : {strat_name}")
            print(stats.round(4))
