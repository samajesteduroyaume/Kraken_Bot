from typing import Dict, List, Optional, Any
from ..kraken_api.validators import KrakenValidator
from ..kraken_api.endpoints import KrakenEndpoints
from ..kraken_api.metrics import KrakenMetrics

class KrakenMarketAnalysis:
    def __init__(self, validator: KrakenValidator, endpoints: KrakenEndpoints, metrics: KrakenMetrics):
        self._validator = validator
        self._endpoints = endpoints
        self._metrics = metrics

    async def calculate_sma(self, pair: str, interval: int = 1, period: int = 20, since: Optional[int] = None) -> Dict:
        """Calcule la moyenne mobile simple (SMA)."""
        ohlc = await self._endpoints.get_ohlc(pair, interval, since)
        closes = [float(c[4]) for c in ohlc[pair]]
        sma = sum(closes[-period:]) / period
        return {
            'pair': pair,
            'interval': interval,
            'period': period,
            'sma': sma
        }

    async def calculate_ema(self, pair: str, interval: int = 1, period: int = 20, since: Optional[int] = None) -> Dict:
        """Calcule la moyenne mobile exponentielle (EMA)."""
        ohlc = await self._endpoints.get_ohlc(pair, interval, since)
        closes = [float(c[4]) for c in ohlc[pair]]
        multiplier = 2 / (period + 1)
        ema = closes[-period]
        for close in closes[-period+1:]:
            ema = (close - ema) * multiplier + ema
        return {
            'pair': pair,
            'interval': interval,
            'period': period,
            'ema': ema
        }

    async def calculate_rsi(self, pair: str, interval: int = 1, period: int = 14, since: Optional[int] = None) -> Dict:
        """Calcule le RSI (Relative Strength Index)."""
        ohlc = await self._endpoints.get_ohlc(pair, interval, since)
        closes = [float(c[4]) for c in ohlc[pair]]
        changes = [closes[i] - closes[i-1] for i in range(1, len(closes))]
        gains = sum([c for c in changes[-period:] if c > 0]) / period
        losses = abs(sum([c for c in changes[-period:] if c < 0]) / period)
        rs = gains / losses if losses != 0 else float('inf')
        rsi = 100 - (100 / (1 + rs))
        return {
            'pair': pair,
            'interval': interval,
            'period': period,
            'rsi': rsi
        }

    async def calculate_macd(self, pair: str, interval: int = 1, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9, since: Optional[int] = None) -> Dict:
        """Calcule le MACD (Moving Average Convergence Divergence)."""
        ohlc = await self._endpoints.get_ohlc(pair, interval, since)
        closes = [float(c[4]) for c in ohlc[pair]]
        
        fast_ema = await self.calculate_ema(pair, interval, fast_period, since)
        slow_ema = await self.calculate_ema(pair, interval, slow_period, since)
        
        macd_line = fast_ema['ema'] - slow_ema['ema']
        signal_line = await self.calculate_ema(pair, interval, signal_period, since)
        histogram = macd_line - signal_line['ema']
        
        return {
            'pair': pair,
            'interval': interval,
            'fast_period': fast_period,
            'slow_period': slow_period,
            'signal_period': signal_period,
            'macd_line': macd_line,
            'signal_line': signal_line['ema'],
            'histogram': histogram
        }

    async def calculate_bollinger_bands(self, pair: str, interval: int = 1, period: int = 20, multiplier: float = 2.0, since: Optional[int] = None) -> Dict:
        """Calcule les bandes de Bollinger."""
        ohlc = await self._endpoints.get_ohlc(pair, interval, since)
        closes = [float(c[4]) for c in ohlc[pair]]
        
        sma = await self.calculate_sma(pair, interval, period, since)
        variances = [(c - sma['sma']) ** 2 for c in closes[-period:]]
        std_dev = (sum(variances) / period) ** 0.5
        
        return {
            'pair': pair,
            'interval': interval,
            'period': period,
            'multiplier': multiplier,
            'middle_band': sma['sma'],
            'upper_band': sma['sma'] + (multiplier * std_dev),
            'lower_band': sma['sma'] - (multiplier * std_dev)
        }
