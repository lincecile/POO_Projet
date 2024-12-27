import pandas as pd
from .Result_Class import Result
from .Strategy_Class import Strategy
from typing import Union

class Backtester:
    """Classe pour exécuter les backtests."""
    
    def __init__(self, data: pd.DataFrame, transaction_costs: Union[float, dict] = None, slippage: Union[float, dict] = None):
        """
        Initialise le backtester avec des coûts sous format float ou dict.
        
        Args:
            data: DataFrame avec les données de tous les actifs
            transaction_costs: Float (coût générale pour tous les actifs) ou dict (coûts spécifique à chaque actif)
            slippage: Float (coût générale pour tous les actifs) ou dict (coûts spécifique à chaque actif)
        """
        self.data = data
        self.transaction_costs = self._process_costs(transaction_costs, default=0.001)
        self.slippage = self._process_costs(slippage, default=0.0005)
    
    def _process_costs(self, costs: Union[float, dict, None], default: float) -> dict:
        """Convert costs input to dictionary format."""

        # Si l'utilisateur a entré un dictionnaire complet de coût
        if isinstance(costs, dict) and len(costs) == len(self.data.columns):
            return costs
        
        # Si l'utilisateur entre un coût général
        elif isinstance(costs, float):
            return {col: costs for col in self.data.columns}
        
        # Si l'utilisateur entre uniquement un dictionnaire des actifs concernés par les coûts
        elif isinstance(costs, dict):
            processed_costs = {col: default for col in self.data.columns}
            processed_costs.update(costs)        
            return processed_costs
        
        # Si aucun coût n'est indiqué, des coûts par défaut sont appliqués
        return {col: default for col in self.data.columns}
    

    def exec_backtest(self, strategy: Strategy) -> Result:
        """
        Exécute le backtest pour une stratégie donnée.
        
        Args:
            strategy: Instance de Strategy à tester
            
        Returns:
            Result: Résultats du backtest
        """
        positions = []
        current_position = {asset: 0 for asset in strategy.assets}
        trades = []
        
        # Rééchantillonnage des données selon la fréquence de rééquilibrage
        resampled_data = self.data.resample(strategy.rebalancing_frequency).last()
        
        # Appel à la méthode fit mais ne fait rien si non implémentée
        strategy.fit(self.data)
        
        for timestamp in resampled_data.index:
            historical_data = self.data.loc[:timestamp]

            # Calcul de la nouvelle position en fonction de la stratégie
            new_position = strategy.get_position(historical_data, current_position)
            
            # Si la position change, on enregistre le trade et son coût
            for asset in strategy.assets:
                if new_position[asset] != current_position[asset]:
                    trade_cost = (abs(new_position[asset] - current_position[asset]) * 
                                (self.transaction_costs[asset] + self.slippage[asset]))
                    trades.append({
                        'timestamp': timestamp,
                        'asset': asset,
                        'from_pos': current_position[asset],
                        'to_pos': new_position[asset],
                        'cost': trade_cost
                    })

            # Ajout de la position au timestamp t, que la position ait changé ou non
            positions.append({
                'timestamp': timestamp,
                **{f"{asset}": new_position[asset] for asset in strategy.assets}
            })

            # Mise à jour la position actuelle pour la prochaine itération
            current_position = new_position.copy()
        
        # Tableau de position
        positions_df = pd.DataFrame(positions).set_index('timestamp')
        print(positions_df)
        # Tableau de trade, possiblement vide
        trades_df = pd.DataFrame(trades).set_index('timestamp') if trades else pd.DataFrame()

        return Result(self.data, positions_df, trades_df)
