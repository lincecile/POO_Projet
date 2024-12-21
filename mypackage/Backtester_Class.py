import pandas as pd
from .Result_Class import Result
from .Strategy_Class import Strategy

class Backtester:
    """Classe pour exécuter les backtests."""
    
    def __init__(self, data: pd.DataFrame, transaction_costs: float = 0.001, slippage: float = 0.0005):
        self.data = data
        self.transaction_costs = transaction_costs
        self.slippage = slippage
    
    def exec_backtest(self, strategy: Strategy) -> Result:
        """
        Exécute le backtest pour une stratégie donnée.
        
        Args:
            strategy: Instance de Strategy à tester
            
        Returns:
            Result: Résultats du backtest
        """
        positions = []
        current_position = 0
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
            if new_position != current_position:
                trade_cost = abs(new_position - current_position) * (self.transaction_costs + self.slippage)
                trades.append({
                    'timestamp': timestamp,
                    'from_pos': current_position,
                    'to_pos': new_position,
                    'cost': trade_cost
                })
            
            # Ajout de la position au timestamp t, que la position ait changé ou non
            positions.append({
                'timestamp': timestamp,
                'position': new_position
            })

            # Mise à jour la position actuelle pour la prochaine itération
            current_position = new_position
        
        # Tableau de position
        positions_df = pd.DataFrame(positions).set_index('timestamp')

        # Tableau de trade, possiblement vide
        trades_df = pd.DataFrame(trades).set_index('timestamp') if trades else pd.DataFrame()

        return Result(self.data, positions_df, trades_df)
