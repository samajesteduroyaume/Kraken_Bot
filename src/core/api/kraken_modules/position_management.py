from typing import Dict, Optional, Any
from ..kraken_api.validators import KrakenValidator
from ..kraken_api.endpoints import KrakenEndpoints
from ..kraken_api.metrics import KrakenMetrics

class KrakenPositionManagement:
    def __init__(self, validator: KrakenValidator, endpoints: KrakenEndpoints, metrics: KrakenMetrics):
        self._validator = validator
        self._endpoints = endpoints
        self._metrics = metrics

    async def calculate_position_size(self, pair: str, risk_percentage: float, stop_loss: float) -> Dict:
        """Calcule la taille de position appropriée."""
        balance = await self._endpoints.get_balance()
        risk_amount = balance * (risk_percentage / 100)
        position_size = risk_amount / stop_loss
        
        min_size = await self._endpoints.get_min_order_size(pair)
        if position_size < min_size:
            position_size = min_size
        
        return {
            'pair': pair,
            'risk_percentage': risk_percentage,
            'stop_loss': stop_loss,
            'position_size': position_size,
            'min_size': min_size
        }

    async def calculate_stop_loss(self, pair: str, entry_price: float, risk_percentage: float) -> Dict:
        """Calcule le stop loss approprié."""
        volatility = await self._endpoints.get_market_volatility(pair)
        stop_loss = entry_price * (1 - risk_percentage/100)
        
        return {
            'pair': pair,
            'entry_price': entry_price,
            'risk_percentage': risk_percentage,
            'stop_loss': stop_loss,
            'volatility': volatility['volatility']
        }

    async def calculate_take_profit(self, pair: str, entry_price: float, risk_reward_ratio: float) -> Dict:
        """Calcule le take profit approprié."""
        stop_loss = await self.calculate_stop_loss(pair, entry_price, 1.0)
        take_profit = entry_price + (entry_price - stop_loss['stop_loss']) * risk_reward_ratio
        
        return {
            'pair': pair,
            'entry_price': entry_price,
            'risk_reward_ratio': risk_reward_ratio,
            'take_profit': take_profit,
            'stop_loss': stop_loss['stop_loss']
        }

    async def get_position_risk(self, pair: str, position_size: float, entry_price: float, stop_loss: float) -> Dict:
        """Calcule le risque de la position."""
        risk_amount = position_size * (entry_price - stop_loss)
        balance = await self._endpoints.get_balance()
        risk_percentage = (risk_amount / balance) * 100
        
        return {
            'pair': pair,
            'position_size': position_size,
            'entry_price': entry_price,
            'stop_loss': stop_loss,
            'risk_amount': risk_amount,
            'risk_percentage': risk_percentage
        }

    async def get_position_metrics(self, pair: str, position_size: float, entry_price: float, current_price: float) -> Dict:
        """Calcule les métriques de la position."""
        pnl = position_size * (current_price - entry_price)
        pnl_percentage = (pnl / (position_size * entry_price)) * 100
        
        return {
            'pair': pair,
            'position_size': position_size,
            'entry_price': entry_price,
            'current_price': current_price,
            'pnl': pnl,
            'pnl_percentage': pnl_percentage
        }
