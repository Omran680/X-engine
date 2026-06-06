import numpy as np
from typing import Tuple, Dict, List

class PortfolioRiskManager:
    """Gestion de risque pour le portefeuille de trading"""
    
    def __init__(self, initial_capital: float = 10000, max_risk_per_trade: float = 0.02):
        """
        Args:
            initial_capital: Capital initial
            max_risk_per_trade: Risque max par trade (défaut 2%)
        """
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.max_risk_per_trade = max_risk_per_trade
        
        # Tracking
        self.daily_pnl = []
        self.daily_trades = []
        self.max_drawdown = 0
        self.peak_capital = initial_capital
        
        # Risk limits
        self.max_daily_loss_pct = 0.05  # 5% max daily loss
        self.max_consecutive_losses = 3
        self.consecutive_losses = 0

    def calculate_position_size(self, 
                               entry_price: float, 
                               stop_loss_price: float,
                               risk_amount: float = None) -> float:
        """
        Calcul de la taille de position basée sur le risque
        
        Args:
            entry_price: Prix d'entrée
            stop_loss_price: Prix du stop loss
            risk_amount: Montant à risquer (défaut: capital * max_risk_per_trade)
        
        Returns:
            Position size (nombre de lots)
        """
        if risk_amount is None:
            risk_amount = self.current_capital * self.max_risk_per_trade
        
        risk_per_unit = abs(entry_price - stop_loss_price)
        
        if risk_per_unit == 0:
            return 0.1  # Minimum lot
        
        position_size = risk_amount / risk_per_unit
        
        # Limiter la position size
        max_position = self.current_capital / (entry_price * 2)  # Max 50% du capital
        position_size = min(position_size, max_position)
        
        return max(position_size, 0.01)  # Minimum 0.01 lot

    def calculate_kelly_criterion(self, 
                                 win_rate: float, 
                                 avg_win: float,
                                 avg_loss: float) -> float:
        """
        Critère de Kelly pour taille de position optimale
        
        Kelly % = (bp - q) / b
        où:
        - b = ratio win/loss
        - p = win probability
        - q = loss probability
        """
        if avg_loss == 0:
            return 0
        
        b = avg_win / avg_loss
        p = win_rate
        q = 1 - win_rate
        
        kelly = (b * p - q) / b
        
        # Utiliser fraction de Kelly (défaut: 25%) pour plus de sécurité
        kelly = kelly * 0.25
        
        return np.clip(kelly, 0, 0.25)

    def check_trade_allowed(self, expected_loss: float) -> Tuple[bool, str]:
        """
        Vérifier si un trade est autorisé selon les limites de risque
        
        Returns:
            (allowed: bool, reason: str)
        """
        # Check daily loss limit
        daily_loss = sum(self.daily_pnl[-1:]) if self.daily_pnl else 0
        if daily_loss + expected_loss < -self.current_capital * self.max_daily_loss_pct:
            return False, "Daily loss limit reached"
        
        # Check consecutive losses
        if self.consecutive_losses >= self.max_consecutive_losses:
            return False, "Max consecutive losses reached"
        
        # Check capital preservation
        if self.current_capital < self.initial_capital * 0.5:
            return False, "Capital below 50% threshold"
        
        return True, "Trade allowed"

    def update_pnl(self, pnl: float):
        """Mettre à jour PnL"""
        self.current_capital += pnl
        self.daily_pnl.append(pnl)
        self.daily_trades.append(1)
        
        # Update consecutive losses
        if pnl < 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0
        
        # Update max drawdown
        if self.current_capital > self.peak_capital:
            self.peak_capital = self.current_capital
        
        drawdown = (self.peak_capital - self.current_capital) / self.peak_capital
        if drawdown > self.max_drawdown:
            self.max_drawdown = drawdown

    def calculate_trailing_stop(self, 
                               entry_price: float, 
                               current_price: float,
                               trailing_stop_pct: float = 1.0) -> float:
        """
        Calculer un trailing stop dynamique
        
        Args:
            entry_price: Prix d'entrée
            current_price: Prix actuel
            trailing_stop_pct: Pourcentage trailing (défaut 1%)
        
        Returns:
            Prix du stop loss ajusté
        """
        if current_price <= entry_price:
            # Pas de profit - stop loss classique
            return entry_price * (1 - trailing_stop_pct / 100)
        
        # Profit existe - trailing stop
        profit_pct = (current_price - entry_price) / entry_price
        trailing_distance = current_price * (trailing_stop_pct / 100)
        
        # Ajuster stop loss avec profit
        adjusted_stop = current_price - trailing_distance
        
        return adjusted_stop

    def calculate_volatility_adjustment(self, 
                                      prices: np.ndarray,
                                      base_position_size: float) -> float:
        """
        Ajuster la position size en fonction de la volatilité
        
        Args:
            prices: Prix historiques (derniers 20 bars)
            base_position_size: Taille de base
        
        Returns:
            Position size ajustée
        """
        if len(prices) < 2:
            return base_position_size
        
        returns = np.diff(prices) / prices[:-1]
        volatility = np.std(returns)
        
        # Volatilité de référence (2%)
        base_volatility = 0.02
        
        # Ajuster inversement à la volatilité (moins volatil = plus grande position)
        volatility_ratio = base_volatility / (volatility + 1e-8)
        volatility_ratio = np.clip(volatility_ratio, 0.5, 2.0)  # Limiter ajustement
        
        return base_position_size * volatility_ratio

    def get_risk_metrics(self) -> Dict:
        """Retourner les métriques de risque"""
        total_pnl = sum(self.daily_pnl)
        num_trades = len(self.daily_trades)
        
        # Calcul Sharpe Ratio (simplifié)
        if len(self.daily_pnl) > 1:
            daily_returns = np.array(self.daily_pnl) / self.initial_capital
            sharpe = np.mean(daily_returns) / (np.std(daily_returns) + 1e-8) * np.sqrt(252)
        else:
            sharpe = 0
        
        # Win rate
        winning_trades = sum(1 for p in self.daily_pnl if p > 0)
        win_rate = winning_trades / (num_trades + 1e-8)
        
        # Profit factor
        total_gains = sum(p for p in self.daily_pnl if p > 0)
        total_losses = abs(sum(p for p in self.daily_pnl if p < 0))
        profit_factor = total_gains / (total_losses + 1e-8)
        
        return {
            'total_pnl': total_pnl,
            'return_pct': (self.current_capital / self.initial_capital - 1) * 100,
            'num_trades': num_trades,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'sharpe_ratio': sharpe,
            'max_drawdown': self.max_drawdown * 100,
            'current_capital': self.current_capital,
            'consecutive_losses': self.consecutive_losses
        }


class DynamicRiskAdjustment:
    """Ajustement dynamique du risque basé sur la performance"""
    
    def __init__(self, base_risk: float = 0.02):
        self.base_risk = base_risk
        self.current_risk = base_risk
        self.performance_history = []
        
    def adjust_risk(self, 
                   win_rate: float, 
                   sharpe_ratio: float,
                   drawdown: float) -> float:
        """
        Ajuster le risque en fonction de la performance
        
        Args:
            win_rate: Taux de victoire
            sharpe_ratio: Ratio de Sharpe
            drawdown: Maximum drawdown
        
        Returns:
            Nouveau risque
        """
        # Score de performance composite
        performance_score = (
            win_rate * 0.4 +  # 40% win rate
            (sharpe_ratio / 2) * 0.4 +  # 40% Sharpe (normalize à 2.0)
            (1 - drawdown) * 0.2  # 20% inverse drawdown
        )
        
        # Ajuster risque
        if performance_score > 0.7:
            # Bonne performance - augmenter légèrement
            adjustment = 1.1
        elif performance_score > 0.5:
            # Performance normale
            adjustment = 1.0
        elif performance_score > 0.3:
            # Performance faible
            adjustment = 0.8
        else:
            # Très mauvaise performance
            adjustment = 0.5
        
        self.current_risk = self.base_risk * adjustment
        self.current_risk = np.clip(self.current_risk, 0.005, 0.05)  # Entre 0.5% et 5%
        
        self.performance_history.append({
            'score': performance_score,
            'risk': self.current_risk,
            'adjustment': adjustment
        })
        
        return self.current_risk


class TradingSession:
    """Gestion d'une session de trading avec risque"""
    
    def __init__(self, capital: float = 10000):
        self.risk_manager = PortfolioRiskManager(initial_capital=capital)
        self.dynamic_risk = DynamicRiskAdjustment()
        self.open_positions = []
        
    def open_position(self,
                     entry_price: float,
                     stop_loss_price: float,
                     take_profit_price: float,
                     direction: str) -> Dict:
        """
        Ouvrir une position avec gestion du risque
        
        Args:
            entry_price: Prix d'entrée
            stop_loss_price: Prix du stop loss
            take_profit_price: Prix du take profit
            direction: 'LONG' ou 'SHORT'
        
        Returns:
            Détails de la position
        """
        position_size = self.risk_manager.calculate_position_size(
            entry_price, stop_loss_price
        )
        
        allowed, reason = self.risk_manager.check_trade_allowed(
            position_size * abs(entry_price - stop_loss_price)
        )
        
        if not allowed:
            return {'status': 'rejected', 'reason': reason}
        
        position = {
            'entry_price': entry_price,
            'stop_loss': stop_loss_price,
            'take_profit': take_profit_price,
            'position_size': position_size,
            'direction': direction,
            'entry_time': 0
        }
        
        self.open_positions.append(position)
        
        return {'status': 'opened', 'position': position}
    
    def close_position(self, position_index: int, exit_price: float) -> Dict:
        """Fermer une position"""
        if position_index >= len(self.open_positions):
            return {'status': 'error', 'message': 'Position not found'}
        
        position = self.open_positions[position_index]
        
        # Calculer PnL
        if position['direction'] == 'LONG':
            pnl = (exit_price - position['entry_price']) * position['position_size']
        else:
            pnl = (position['entry_price'] - exit_price) * position['position_size']
        
        # Mettre à jour risk manager
        self.risk_manager.update_pnl(pnl)
        
        # Retirer position
        self.open_positions.pop(position_index)
        
        return {
            'status': 'closed',
            'pnl': pnl,
            'capital': self.risk_manager.current_capital,
            'return_pct': (pnl / position['entry_price'] / position['position_size']) * 100
        }
    
    def get_session_summary(self) -> Dict:
        """Résumé de la session"""
        return self.risk_manager.get_risk_metrics()


# ============= Exemples d'Utilisation =============
if __name__ == "__main__":
    # Exemple 1: Position sizing
    rm = PortfolioRiskManager(initial_capital=10000)
    
    entry = 2050
    stop_loss = 2045
    size = rm.calculate_position_size(entry, stop_loss)
    print(f"Position size: {size:.3f} lots")
    
    # Exemple 2: Session de trading
    session = TradingSession(capital=10000)
    
    # Ouvrir position
    result = session.open_position(
        entry_price=2050,
        stop_loss_price=2045,
        take_profit_price=2060,
        direction='LONG'
    )
    print(f"Open position: {result}")
    
    # Fermer position
    result = session.close_position(0, exit_price=2055)
    print(f"Close position: {result}")
    
    # Résumé
    summary = session.get_session_summary()
    print(f"Session summary: {summary}")
