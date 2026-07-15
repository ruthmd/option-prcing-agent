from typing import Dict, List, Optional, Tuple, Any, Union
from pydantic import BaseModel, Field
from datetime import datetime, timedelta, date
import yfinance as yf
import pandas as pd
import numpy as np
from loguru import logger
import time
import warnings

from ...config.settings import settings
from ...utils.constants import TRADING_DAYS_PER_YEAR, ERROR_MESSAGES

# Suppress yfinance warnings
warnings.filterwarnings('ignore', category=FutureWarning, module='yfinance')

class MarketDataResult(BaseModel):
    """Result of market data fetch operation"""
    success: bool
    data: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)
    data_quality: float = Field(ge=0.0, le=1.0, default=1.0)
    warnings: List[str] = []

class OptionChainData(BaseModel):
    """Structured option chain data"""
    symbol: str
    expiration_dates: List[str]
    calls: Optional[pd.DataFrame] = None
    puts: Optional[pd.DataFrame] = None
    last_updated: datetime = Field(default_factory=datetime.now)
    class Config:
        arbitrary_types_allowed = True

class MarketDataAgent:
    def __init__(self):
        self.cache = {}
        self.cache_timestamps = {}
        self.rate_limit_delay = 0.1  # Minimum delay between requests
        self.last_request_time = 0
        
        # Initialize treasury rate cache
        self._treasury_cache = {}
        self._treasury_cache_time = None
    
    def _rate_limit(self):
        """Implement rate limiting for API calls"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - time_since_last)
        self.last_request_time = time.time()
    
    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cached data is still valid"""
        if cache_key not in self.cache_timestamps:
            return False
        
        cache_time = self.cache_timestamps[cache_key]
        cache_age = (datetime.now() - cache_time).total_seconds() / 60
        return cache_age < settings.CACHE_DURATION_MINUTES


    def get_stock_data(
        self, 
        symbol: str, 
        period: str = "1d",
        include_info: bool = True
    ) -> MarketDataResult:
        """Get current stock data and basic information"""
        
        cache_key = f"stock_{symbol}_{period}"
        
        # Check cache first
        if self._is_cache_valid(cache_key):
            logger.debug(f"Using cached data for {symbol}")
            return MarketDataResult(
                success=True,
                data=self.cache[cache_key],
                data_quality=0.9  # Slightly lower for cached data
            )
        
        try:
            self._rate_limit()
            ticker = yf.Ticker(symbol)
            
            # Get current price data
            hist = ticker.history(period=period)
            if hist.empty:
                return MarketDataResult(
                    success=False,
                    error_message=f"No price data available for {symbol}"
                )
            
            # Get basic info with error handling
            info = {}
            if include_info:
                try:
                    raw_info = ticker.info
                    if raw_info:
                        # Extract safe values with defaults
                        info = {
                            'marketCap': raw_info.get('marketCap'),
                            'dividendYield': raw_info.get('dividendYield', raw_info.get('trailingAnnualDividendYield', 0.0)),
                            'beta': raw_info.get('beta'),
                            'sector': raw_info.get('sector'),
                            'industry': raw_info.get('industry'),
                            'symbol': raw_info.get('symbol', symbol),
                            'quoteType': raw_info.get('quoteType', 'EQUITY')
                        }
                        
                        # Validate dividend yield
                        dividend_yield = info.get('dividendYield', 0.0)
                        if dividend_yield and dividend_yield > 0.25:  # More than 25% is suspicious
                            logger.warning(f"Capping suspicious dividend yield for {symbol}: {dividend_yield:.2%} -> 5%")
                            info['dividendYield'] = 0.05
                        elif dividend_yield and dividend_yield < 0:
                            logger.warning(f"Setting negative dividend yield to 0 for {symbol}")
                            info['dividendYield'] = 0.0
                            
                except Exception as e:
                    logger.warning(f"Failed to get detailed info for {symbol}: {e}")
                    info = {'dividendYield': 0.0, 'quoteType': 'EQUITY'}  # Provide safe default
            
            # Structure the data
            latest_data = hist.iloc[-1]
            data = {
                "symbol": symbol,
                "current_price": float(latest_data['Close']),
                "previous_close": float(hist.iloc[-2]['Close']) if len(hist) > 1 else float(latest_data['Close']),
                "volume": int(latest_data['Volume']),
                "high": float(latest_data['High']),
                "low": float(latest_data['Low']),
                "open": float(latest_data['Open']),
                "market_cap": info.get('marketCap'),
                "dividend_yield": info.get('dividendYield', 0.0),
                "beta": info.get('beta'),
                "sector": info.get('sector'),
                "industry": info.get('industry'),
                "quote_type": info.get('quoteType', 'EQUITY'),
                "last_updated": datetime.now().isoformat()
            }
            
            # Calculate additional metrics
            if len(hist) > 1:
                data["price_change"] = data["current_price"] - data["previous_close"]
                data["price_change_percent"] = (data["price_change"] / data["previous_close"]) * 100
            
            # Cache the result
            self.cache[cache_key] = data
            self.cache_timestamps[cache_key] = datetime.now()
            
            # Assess data quality
            quality_score = self._assess_data_quality(data, hist)
            
            return MarketDataResult(
                success=True,
                data=data,
                data_quality=quality_score
            )

        except Exception as e:
            logger.error(f"Failed to fetch stock data for {symbol}: {e}")
            return MarketDataResult(
                success=False,
                error_message=f"Failed to fetch data for {symbol}: {str(e)}"
            )


    def get_historical_data(
        self, 
        symbol: str, 
        period: str = "1y",
        interval: str = "1d"
    ) -> MarketDataResult:
        """Get historical price data for volatility calculations"""
        
        cache_key = f"historical_{symbol}_{period}_{interval}"
        
        if self._is_cache_valid(cache_key):
            return MarketDataResult(
                success=True,
                data=self.cache[cache_key],
                data_quality=0.95
            )
        
        try:
            self._rate_limit()
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period=period, interval=interval)
            
            if hist.empty:
                return MarketDataResult(
                    success=False,
                    error_message=f"No historical data available for {symbol}"
                )
            
            # Calculate returns and volatility
            hist['Returns'] = hist['Close'].pct_change()
            hist = hist.dropna()
            
            # Calculate various volatility measures
            daily_vol = hist['Returns'].std()
            annualized_vol = daily_vol * np.sqrt(TRADING_DAYS_PER_YEAR)
            
            # Rolling volatilities
            hist['Volatility_30d'] = hist['Returns'].rolling(window=30).std() * np.sqrt(TRADING_DAYS_PER_YEAR)
            hist['Volatility_60d'] = hist['Returns'].rolling(window=60).std() * np.sqrt(TRADING_DAYS_PER_YEAR)
            
            data = {
                "symbol": symbol,
                "data": hist,
                "daily_volatility": daily_vol,
                "annualized_volatility": annualized_vol,
                "period": period,
                "data_points": len(hist),
                "start_date": hist.index[0].strftime('%Y-%m-%d'),
                "end_date": hist.index[-1].strftime('%Y-%m-%d')
            }
            
            # Cache the result
            self.cache[cache_key] = data
            self.cache_timestamps[cache_key] = datetime.now()
            
            return MarketDataResult(
                success=True,
                data=data,
                data_quality=1.0
            )
            
        except Exception as e:
            logger.error(f"Failed to fetch historical data for {symbol}: {e}")
            return MarketDataResult(
                success=False,
                error_message=f"Failed to fetch historical data: {str(e)}"
            )
    
    def get_options_chain(self, symbol: str, expiration: Optional[str] = None) -> MarketDataResult:
        """Get options chain data"""
        
        cache_key = f"options_{symbol}_{expiration or 'all'}"
        
        if self._is_cache_valid(cache_key):
            return MarketDataResult(
                success=True,
                data=self.cache[cache_key],
                data_quality=0.8  # Options data changes frequently
            )
        
        try:
            self._rate_limit()
            ticker = yf.Ticker(symbol)
            
            # Get available expiration dates
            try:
                expirations = ticker.options
                if not expirations:
                    return MarketDataResult(
                        success=False,
                        error_message=f"No options available for {symbol}"
                    )
            except Exception as e:
                return MarketDataResult(
                    success=False,
                    error_message=f"Failed to get options expirations for {symbol}: {str(e)}"
                )
            
            # Get options chain for specific expiration or first available
            target_expiration = expiration or expirations[0]
            
            if target_expiration not in expirations:
                return MarketDataResult(
                    success=False,
                    error_message=f"Expiration {target_expiration} not available for {symbol}"
                )
            
            try:
                option_chain = ticker.option_chain(target_expiration)
                calls = option_chain.calls
                puts = option_chain.puts
                
                # Clean and enhance the data
                for df in [calls, puts]:
                    if not df.empty:
                        # Add derived columns
                        df['mid_price'] = (df['bid'] + df['ask']) / 2
                        df['bid_ask_spread'] = df['ask'] - df['bid']
                        df['moneyness'] = df['strike'] / ticker.history(period="1d")['Close'].iloc[-1]
                
                data = {
                    "symbol": symbol,
                    "expiration": target_expiration,
                    "expiration_dates": list(expirations),
                    "calls": calls,
                    "puts": puts,
                    "calls_count": len(calls),
                    "puts_count": len(puts)
                }
                
                # Cache the result
                self.cache[cache_key] = data
                self.cache_timestamps[cache_key] = datetime.now()
                
                return MarketDataResult(
                    success=True,
                    data=data,
                    data_quality=0.9
                )
                
            except Exception as e:
                return MarketDataResult(
                    success=False,
                    error_message=f"Failed to get option chain for {target_expiration}: {str(e)}"
                )
            
        except Exception as e:
            logger.error(f"Failed to fetch options chain for {symbol}: {e}")
            return MarketDataResult(
                success=False,
                error_message=f"Failed to fetch options chain: {str(e)}"
            )
    
    def get_risk_free_rate(self) -> float:
        """Get current risk-free rate (10-year Treasury)"""
        
        # Check cache (with longer cache time for rates)
        if (self._treasury_cache_time and 
            (datetime.now() - self._treasury_cache_time).total_seconds() < 3600):  # 1 hour cache
            return self._treasury_cache.get('rate', 0.02)  # Default 2%
        
        try:
            self._rate_limit()
            # Try to get 10-year treasury rate
            treasury = yf.Ticker("^TNX")
            hist = treasury.history(period="5d")
            
            if not hist.empty:
                rate = hist['Close'].iloc[-1] / 100  # Convert percentage to decimal
                self._treasury_cache['rate'] = rate
                self._treasury_cache_time = datetime.now()
                logger.info(f"Updated risk-free rate: {rate:.4f}")
                return rate
            else:
                logger.warning("Failed to get treasury rate, using default")
                return 0.02  # Default 2%
                
        except Exception as e:
            logger.warning(f"Failed to fetch risk-free rate: {e}, using default")
            return 0.02  # Default 2%
    
    def get_dividend_yield(self, symbol: str) -> float:
        """Get dividend yield for a symbol"""
        try:
            stock_data = self.get_stock_data(symbol, include_info=True)
            if stock_data.success and stock_data.data:
                dividend_yield = stock_data.data.get('dividend_yield', 0.0) or 0.0
                
                # Validate and cap dividend yield to reasonable bounds
                if dividend_yield > 0.2:  # More than 20% is suspicious
                    logger.warning(f"Unusually high dividend yield for {symbol}: {dividend_yield:.2%}, capping at 5%")
                    dividend_yield = 0.05  # Cap at 5%
                elif dividend_yield < 0:
                    logger.warning(f"Negative dividend yield for {symbol}: {dividend_yield:.2%}, setting to 0%")
                    dividend_yield = 0.0
                
                logger.debug(f"Dividend yield for {symbol}: {dividend_yield:.2%}")
                return dividend_yield
            
            return 0.0
        except Exception as e:
            logger.warning(f"Failed to get dividend yield for {symbol}: {e}")
            return 0.0
    
    def calculate_historical_volatility(
        self, 
        symbol: str, 
        days: int = 30,
        annualized: bool = True
    ) -> MarketDataResult:
        """Calculate historical volatility"""
        
        try:
            # Determine period based on days requested
            if days <= 30:
                period = "1mo"
            elif days <= 90:
                period = "3mo"
            elif days <= 180:
                period = "6mo"
            else:
                period = "1y"
            
            hist_data = self.get_historical_data(symbol, period=period)
            
            if not hist_data.success:
                return hist_data
            
            hist = hist_data.data['data']
            
            # Calculate returns
            returns = hist['Close'].pct_change().dropna()
            
            # Take only the requested number of days
            if len(returns) > days:
                returns = returns.tail(days)
            
            # Calculate volatility
            vol = returns.std()
            
            if annualized:
                vol *= np.sqrt(TRADING_DAYS_PER_YEAR)
            
            data = {
                "symbol": symbol,
                "volatility": vol,
                "days_used": len(returns),
                "annualized": annualized,
                "period_start": returns.index[0].strftime('%Y-%m-%d'),
                "period_end": returns.index[-1].strftime('%Y-%m-%d')
            }
            
            return MarketDataResult(
                success=True,
                data=data,
                data_quality=1.0 if len(returns) >= days * 0.8 else 0.8
            )
            
        except Exception as e:
            logger.error(f"Failed to calculate volatility for {symbol}: {e}")
            return MarketDataResult(
                success=False,
                error_message=f"Volatility calculation failed: {str(e)}"
            )
    
    def _assess_data_quality(self, data: Dict[str, Any], hist: pd.DataFrame) -> float:
        """Assess the quality of fetched data"""
        quality_score = 1.0
        
        # Check for missing essential data
        if not data.get('current_price'):
            quality_score -= 0.3
        
        if data.get('volume', 0) == 0:
            quality_score -= 0.1
        
        # Check data freshness
        if len(hist) < 2:
            quality_score -= 0.2
        
        # Check for suspicious values
        current_price = data.get('current_price', 0)
        if current_price <= 0:
            quality_score -= 0.5
        
        # Check price consistency
        if len(hist) > 1:
            price_changes = hist['Close'].pct_change().abs()
            if price_changes.max() > 0.5:  # More than 50% daily change
                quality_score -= 0.2
        
        return max(quality_score, 0.0)
    
    def validate_symbol(self, symbol: str) -> Tuple[bool, str]:
        """Validate if a symbol exists and is tradeable"""
        try:
            result = self.get_stock_data(symbol, period="1d", include_info=False)
            if result.success:
                return True, "Valid symbol"
            else:
                return False, result.error_message or "Invalid symbol"
        except Exception as e:
            return False, f"Validation failed: {str(e)}"

# Usage example and test
if __name__ == "__main__":
    data_agent = MarketDataAgent()
    
    # Test symbols
    test_symbols = ["AAPL", "GOOGL", "INVALID"]
    
    for symbol in test_symbols:
        print(f"\n{'='*60}")
        print(f"Testing symbol: {symbol}")
        
        # Test stock data
        result = data_agent.get_stock_data(symbol)
        print(f"Stock Data - Success: {result.success}, Quality: {result.data_quality:.2f}")
        if result.success:
            data = result.data
            print(f"  Price: ${data['current_price']:.2f}")
            print(f"  Change: {data.get('price_change_percent', 0):.2f}%")
            print(f"  Volume: {data['volume']:,}")
        else:
            print(f"  Error: {result.error_message}")
        
        if result.success:
            # Test historical volatility
            vol_result = data_agent.calculate_historical_volatility(symbol, days=30)
            print(f"Volatility - Success: {vol_result.success}")
            if vol_result.success:
                vol_data = vol_result.data
                print(f"  30-day volatility: {vol_data['volatility']:.1%}")
                print(f"  Data points: {vol_data['days_used']}")
            
            # Test options chain
            options_result = data_agent.get_options_chain(symbol)
            print(f"Options - Success: {options_result.success}")
            if options_result.success:
                opt_data = options_result.data
                print(f"  Calls: {opt_data['calls_count']}, Puts: {opt_data['puts_count']}")
                print(f"  Expirations: {len(opt_data['expiration_dates'])}")
    
    # Test risk-free rate
    print(f"\n{'='*60}")
    rate = data_agent.get_risk_free_rate()
    print(f"Risk-free rate: {rate:.4f} ({rate:.2%})")