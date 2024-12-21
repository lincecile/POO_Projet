import pandas as pd
from .Result_Class import Result
from .Strategy_Class import Strategy

# Implémentez une classe Backtester qui 
class Backtester:
    """Classe principale pour exécuter les backtests."""
    
    # Est instanciée avec une série de données d’entrée
    def __init__(self, data: pd.DataFrame, transaction_costs: float = 0.001, slippage: float = 0.0005):
        self.data = data
        self.transaction_costs = transaction_costs
        self.slippage = slippage
    
    # Possède une méthode pour exécuter le backtest en prenant une instance de Strategy
    # Renvoie une instance de la classe Result après l’exécution du backtest
    def run(self, strategy: Strategy) -> Result:
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
        
        strategy.fit(self.data)
        
        for timestamp, row in resampled_data.iterrows():
            historical_data = self.data.loc[:timestamp]
            new_position = strategy.get_position(historical_data, current_position)
            
            if new_position != current_position:
                trade_cost = abs(new_position - current_position) * (
                    self.transaction_costs + self.slippage
                )
                trades.append({
                    'timestamp': timestamp,
                    'from_pos': current_position,
                    'to_pos': new_position,
                    'cost': trade_cost
                })
            
            positions.append({
                'timestamp': timestamp,
                'position': new_position
            })
            current_position = new_position
        
        positions_df = pd.DataFrame(positions).set_index('timestamp')
        trades_df = pd.DataFrame(trades).set_index('timestamp') if trades else pd.DataFrame()

        return Result(self.data, positions_df, trades_df)
