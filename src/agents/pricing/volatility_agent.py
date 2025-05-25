from typing import Dict, List, Any, Optional, Tuple
from pydantic import BaseModel, Field, ConfigDict
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.stats import norm
from datetime import datetime, timedelta
from loguru import logger

from ...config.settings import settings
from ...utils.constants import TRADING_DAYS_PER_YEAR, PRICING_BOUNDS
from .black_scholes import BlackScholesPricingAgent, BlackScholesInputs

class VolatilityInputs(BaseModel):
    """Input parameters for volatility analysis"""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    symbol: str = Field(description="Stock symbol")
    price_data: Optional[pd.DataFrame] = Field(default=None, description="Historical price data")
    options_data: Optional[pd.DataFrame] = Field(default=None, description="Options market data")
    spot_price: float = Field(gt=0, description="Current stock price")
    risk_free_rate: float = Field(ge=-0.1, le=0.5, description="Risk-free interest rate")
    dividend_yield: float = Field(ge=0, le=0.3, default=0.0, description="Dividend yield")
    analysis_period: int = Field(ge=5, le=1000, default=30, description="Analysis period in days")

class VolatilityResult(BaseModel):
   """Result of volatility analysis"""
   historical_volatility: Dict[str, float]
   implied_volatility: Dict[str, Any]
   volatility_surface: Optional[Dict[str, Any]] = None
   volatility_statistics: Dict[str, float]
   garch_forecast: Optional[Dict[str, float]] = None
   regime_analysis: Dict[str, Any]
   inputs: VolatilityInputs

class VolatilityAgent:
   """Volatility analysis agent for historical and implied volatility"""
   
   def __init__(self):
       self.name = "VolatilityAnalysis"
       self.bs_agent = BlackScholesPricingAgent()
       logger.info("Volatility analysis agent initialized")
   
   def analyze_volatility(self, inputs: VolatilityInputs) -> VolatilityResult:
       """Comprehensive volatility analysis"""
       try:
           # Calculate historical volatility
           historical_vol = self._calculate_historical_volatility(inputs)
           
           # Calculate implied volatility
           implied_vol = self._calculate_implied_volatility(inputs)
           
           # Build volatility surface if options data available
           vol_surface = self._build_volatility_surface(inputs) if inputs.options_data is not None else None
           
           # Calculate volatility statistics
           vol_stats = self._calculate_volatility_statistics(inputs)
           
           # Regime analysis
           regime_analysis = self._analyze_volatility_regimes(inputs)
           
           # GARCH forecast (simplified)
           garch_forecast = self._simple_garch_forecast(inputs)
           
           result = VolatilityResult(
               historical_volatility=historical_vol,
               implied_volatility=implied_vol,
               volatility_surface=vol_surface,
               volatility_statistics=vol_stats,
               garch_forecast=garch_forecast,
               regime_analysis=regime_analysis,
               inputs=inputs
           )
           
           logger.debug(f"Volatility analysis completed for {inputs.symbol}")
           return result
           
       except Exception as e:
           logger.error(f"Volatility analysis failed: {e}")
           raise ValueError(f"Volatility analysis error: {str(e)}")
   
   def _calculate_historical_volatility(self, inputs: VolatilityInputs) -> Dict[str, float]:
       """Calculate historical volatility for different periods"""
       if inputs.price_data is None or inputs.price_data.empty:
           logger.warning("No price data available for historical volatility")
           return {"error": "No price data available"}
       
       try:
           prices = inputs.price_data['Close'] if 'Close' in inputs.price_data.columns else inputs.price_data.iloc[:, 0]
           
           # Calculate returns
           returns = prices.pct_change().dropna()
           
           if len(returns) < 10:
               logger.warning("Insufficient data for volatility calculation")
               return {"error": "Insufficient data"}
           
           # Calculate volatility for different periods
           periods = {
               "daily": 1,
               "weekly": 7,
               "monthly": 30,
               "quarterly": 90,
               "annual": 252
           }
           
           historical_vol = {}
           
           for period_name, days in periods.items():
               if len(returns) >= days:
                   period_returns = returns.tail(min(days, len(returns)))
                   vol = period_returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR)
                   historical_vol[f"{period_name}_volatility"] = round(vol, 6)
                   historical_vol[f"{period_name}_data_points"] = len(period_returns)
           
           # Rolling volatility
           if len(returns) >= inputs.analysis_period:
               rolling_vol = returns.rolling(window=inputs.analysis_period).std() * np.sqrt(TRADING_DAYS_PER_YEAR)
               historical_vol["current_rolling_vol"] = round(rolling_vol.iloc[-1], 6)
               historical_vol["avg_rolling_vol"] = round(rolling_vol.mean(), 6)
               historical_vol["max_rolling_vol"] = round(rolling_vol.max(), 6)
               historical_vol["min_rolling_vol"] = round(rolling_vol.min(), 6)
           
           # Realized volatility using high-low estimator (Garman-Klass)
           if all(col in inputs.price_data.columns for col in ['High', 'Low', 'Open', 'Close']):
               gk_vol = self._garman_klass_volatility(inputs.price_data)
               historical_vol["garman_klass_volatility"] = round(gk_vol, 6)
           
           return historical_vol
           
       except Exception as e:
           logger.error(f"Historical volatility calculation failed: {e}")
           return {"error": f"Calculation failed: {str(e)}"}
   
   def _garman_klass_volatility(self, price_data: pd.DataFrame) -> float:
       """Calculate Garman-Klass volatility estimator"""
       try:
           high = price_data['High']
           low = price_data['Low']
           open_price = price_data['Open']
           close = price_data['Close']
           
           # Garman-Klass estimator
           term1 = 0.5 * (np.log(high / low))**2
           term2 = (2 * np.log(2) - 1) * (np.log(close / open_price))**2
           
           gk_variance = (term1 - term2).mean()
           gk_volatility = np.sqrt(gk_variance * TRADING_DAYS_PER_YEAR)
           
           return gk_volatility
           
       except Exception as e:
           logger.warning(f"Garman-Klass calculation failed: {e}")
           return 0.0
   
   def _calculate_implied_volatility(self, inputs: VolatilityInputs) -> Dict[str, Any]:
       """Calculate implied volatility from options data"""
       if inputs.options_data is None or inputs.options_data.empty:
           return {"message": "No options data available for implied volatility"}
       
       try:
           options_data = inputs.options_data.copy()
           implied_vols = []
           option_details = []
           
           for _, option in options_data.iterrows():
               try:
                   # Extract option details
                   strike = float(option.get('strike', 0))
                   market_price = float(option.get('lastPrice', option.get('mid_price', 0)))
                   option_type = option.get('type', 'call').lower()
                   
                   if strike <= 0 or market_price <= 0:
                       continue
                   
                   # Calculate time to expiry (simplified - would need actual expiry date)
                   # For now, assume 30 days as placeholder
                   time_to_expiry = 30 / 365
                   
                   # Calculate implied volatility
                   implied_vol = self.bs_agent.calculate_implied_volatility(
                       market_price=market_price,
                       spot_price=inputs.spot_price,
                       strike_price=strike,
                       time_to_expiry=time_to_expiry,
                       risk_free_rate=inputs.risk_free_rate,
                       dividend_yield=inputs.dividend_yield,
                       option_type=option_type
                   )
                   
                   if implied_vol is not None and 0.01 <= implied_vol <= 3.0:
                       implied_vols.append(implied_vol)
                       option_details.append({
                           'strike': strike,
                           'market_price': market_price,
                           'option_type': option_type,
                           'implied_vol': implied_vol,
                           'moneyness': strike / inputs.spot_price
                       })
               
               except Exception as e:
                   logger.debug(f"Failed to calculate IV for option: {e}")
                   continue
           
           if not implied_vols:
               return {"error": "No valid implied volatilities calculated"}
           
           # Calculate statistics
           implied_vol_stats = {
               "mean_implied_vol": round(np.mean(implied_vols), 6),
               "median_implied_vol": round(np.median(implied_vols), 6),
               "std_implied_vol": round(np.std(implied_vols), 6),
               "min_implied_vol": round(np.min(implied_vols), 6),
               "max_implied_vol": round(np.max(implied_vols), 6),
               "options_analyzed": len(implied_vols)
           }
           
           # ATM implied volatility (closest to spot)
           if option_details:
               atm_options = sorted(option_details, key=lambda x: abs(x['moneyness'] - 1.0))
               if atm_options:
                   implied_vol_stats["atm_implied_vol"] = round(atm_options[0]['implied_vol'], 6)
           
           return {
               "statistics": implied_vol_stats,
               "option_details": option_details[:10]  # Limit to first 10 for brevity
           }
           
       except Exception as e:
           logger.error(f"Implied volatility calculation failed: {e}")
           return {"error": f"Calculation failed: {str(e)}"}
   
   def _build_volatility_surface(self, inputs: VolatilityInputs) -> Dict[str, Any]:
       """Build volatility surface from options data"""
       try:
           if inputs.options_data is None:
               return {"error": "No options data for surface construction"}
           
           # This is a simplified implementation
           # In practice, would need proper expiry dates and more sophisticated interpolation
           
           implied_vol_data = self._calculate_implied_volatility(inputs)
           if "option_details" not in implied_vol_data:
               return {"error": "No implied volatility data available"}
           
           option_details = implied_vol_data["option_details"]
           
           # Group by moneyness ranges
           moneyness_buckets = {
               "deep_otm": [],    # < 0.9
               "otm": [],         # 0.9 - 0.98
               "atm": [],         # 0.98 - 1.02
               "itm": [],         # 1.02 - 1.1
               "deep_itm": []     # > 1.1
           }
           
           for option in option_details:
               moneyness = option['moneyness']
               if moneyness < 0.9:
                   moneyness_buckets["deep_otm"].append(option['implied_vol'])
               elif moneyness < 0.98:
                   moneyness_buckets["otm"].append(option['implied_vol'])
               elif moneyness <= 1.02:
                   moneyness_buckets["atm"].append(option['implied_vol'])
               elif moneyness <= 1.1:
                   moneyness_buckets["itm"].append(option['implied_vol'])
               else:
                   moneyness_buckets["deep_itm"].append(option['implied_vol'])
           
           # Calculate average IV for each bucket
           surface_data = {}
           for bucket, vols in moneyness_buckets.items():
               if vols:
                   surface_data[bucket] = {
                       "avg_implied_vol": round(np.mean(vols), 6),
                       "count": len(vols)
                   }
           
           # Calculate volatility skew
           skew_analysis = self._analyze_volatility_skew(option_details)
           
           return {
               "surface_by_moneyness": surface_data,
               "skew_analysis": skew_analysis,
               "total_options": len(option_details)
           }
           
       except Exception as e:
           logger.error(f"Volatility surface construction failed: {e}")
           return {"error": f"Surface construction failed: {str(e)}"}
   
   def _analyze_volatility_skew(self, option_details: List[Dict]) -> Dict[str, Any]:
       """Analyze volatility skew patterns"""
       try:
           if len(option_details) < 3:
               return {"error": "Insufficient data for skew analysis"}
           
           # Separate calls and puts
           calls = [opt for opt in option_details if opt['option_type'] == 'call']
           puts = [opt for opt in option_details if opt['option_type'] == 'put']
           
           skew_analysis = {}
           
           # Call skew analysis
           if len(calls) >= 3:
               calls_sorted = sorted(calls, key=lambda x: x['moneyness'])
               otm_calls = [c for c in calls_sorted if c['moneyness'] > 1.02]
               atm_calls = [c for c in calls_sorted if 0.98 <= c['moneyness'] <= 1.02]
               
               if otm_calls and atm_calls:
                   otm_vol = np.mean([c['implied_vol'] for c in otm_calls])
                   atm_vol = np.mean([c['implied_vol'] for c in atm_calls])
                   call_skew = otm_vol - atm_vol
                   skew_analysis["call_skew"] = round(call_skew, 6)
           
           # Put skew analysis
           if len(puts) >= 3:
               puts_sorted = sorted(puts, key=lambda x: x['moneyness'])
               otm_puts = [p for p in puts_sorted if p['moneyness'] < 0.98]
               atm_puts = [p for p in puts_sorted if 0.98 <= p['moneyness'] <= 1.02]
               
               if otm_puts and atm_puts:
                   otm_vol = np.mean([p['implied_vol'] for p in otm_puts])
                   atm_vol = np.mean([p['implied_vol'] for p in atm_puts])
                   put_skew = otm_vol - atm_vol
                   skew_analysis["put_skew"] = round(put_skew, 6)
           
           # Overall skew interpretation
           if "put_skew" in skew_analysis:
               if skew_analysis["put_skew"] > 0.05:
                   skew_analysis["interpretation"] = "Strong put skew - fear premium present"
               elif skew_analysis["put_skew"] > 0.02:
                   skew_analysis["interpretation"] = "Moderate put skew - normal market conditions"
               else:
                   skew_analysis["interpretation"] = "Low skew - balanced market sentiment"
           
           return skew_analysis
           
       except Exception as e:
           logger.warning(f"Skew analysis failed: {e}")
           return {"error": f"Skew analysis failed: {str(e)}"}
   
   def _calculate_volatility_statistics(self, inputs: VolatilityInputs) -> Dict[str, float]:
       """Calculate comprehensive volatility statistics"""
       if inputs.price_data is None or inputs.price_data.empty:
           return {"error": "No price data available"}
       
       try:
           prices = inputs.price_data['Close'] if 'Close' in inputs.price_data.columns else inputs.price_data.iloc[:, 0]
           returns = prices.pct_change().dropna()
           
           if len(returns) < 10:
               return {"error": "Insufficient data"}
           
           # Calculate various statistics
           vol_stats = {}
           
           # Basic statistics
           daily_vol = returns.std()
           annual_vol = daily_vol * np.sqrt(TRADING_DAYS_PER_YEAR)
           
           vol_stats["daily_volatility"] = round(daily_vol, 6)
           vol_stats["annualized_volatility"] = round(annual_vol, 6)
           vol_stats["mean_return"] = round(returns.mean(), 6)
           vol_stats["skewness"] = round(returns.skew(), 4)
           vol_stats["kurtosis"] = round(returns.kurtosis(), 4)
           
           # Volatility of volatility
           if len(returns) >= 30:
               rolling_vol = returns.rolling(window=20).std()
               vol_of_vol = rolling_vol.std()
               vol_stats["volatility_of_volatility"] = round(vol_of_vol, 6)
           
           # Percentile analysis
           vol_percentiles = [5, 10, 25, 50, 75, 90, 95]
           if len(returns) >= 50:
               rolling_vol = returns.rolling(window=20).std() * np.sqrt(TRADING_DAYS_PER_YEAR)
               rolling_vol = rolling_vol.dropna()
               
               for percentile in vol_percentiles:
                   vol_stats[f"vol_{percentile}th_percentile"] = round(np.percentile(rolling_vol, percentile), 6)
           
           # Current volatility ranking
           if len(returns) >= inputs.analysis_period:
               current_vol = returns.tail(inputs.analysis_period).std() * np.sqrt(TRADING_DAYS_PER_YEAR)
               historical_vols = []
               
               for i in range(inputs.analysis_period, len(returns), 5):  # Every 5 days
                   period_returns = returns.iloc[i-inputs.analysis_period:i]
                   period_vol = period_returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR)
                   historical_vols.append(period_vol)
               
               if historical_vols:
                   vol_rank = (np.sum(np.array(historical_vols) <= current_vol) / len(historical_vols)) * 100
                   vol_stats["current_volatility_percentile"] = round(vol_rank, 1)
           
           return vol_stats
           
       except Exception as e:
           logger.error(f"Volatility statistics calculation failed: {e}")
           return {"error": f"Statistics calculation failed: {str(e)}"}
   
   def _analyze_volatility_regimes(self, inputs: VolatilityInputs) -> Dict[str, Any]:
       """Analyze volatility regimes using simple clustering"""
       if inputs.price_data is None or inputs.price_data.empty:
           return {"error": "No price data available"}
       
       try:
           prices = inputs.price_data['Close'] if 'Close' in inputs.price_data.columns else inputs.price_data.iloc[:, 0]
           returns = prices.pct_change().dropna()
           
           if len(returns) < 60:  # Need at least 60 days
               return {"error": "Insufficient data for regime analysis"}
           
           # Calculate rolling volatility
           window = 20
           rolling_vol = returns.rolling(window=window).std() * np.sqrt(TRADING_DAYS_PER_YEAR)
           rolling_vol = rolling_vol.dropna()
           
           # Simple regime classification based on percentiles
           vol_25th = rolling_vol.quantile(0.25)
           vol_75th = rolling_vol.quantile(0.75)
           
           regimes = []
           for vol in rolling_vol:
               if vol <= vol_25th:
                   regimes.append("Low")
               elif vol >= vol_75th:
                   regimes.append("High")
               else:
                   regimes.append("Medium")
           
           # Calculate regime statistics
           regime_counts = pd.Series(regimes).value_counts()
           current_regime = regimes[-1] if regimes else "Unknown"
           
           # Calculate persistence (average regime duration)
           regime_changes = []
           current_regime_length = 1
           
           for i in range(1, len(regimes)):
               if regimes[i] != regimes[i-1]:
                   regime_changes.append(current_regime_length)
                   current_regime_length = 1
               else:
                   current_regime_length += 1
           
           avg_regime_duration = np.mean(regime_changes) if regime_changes else len(regimes)
           
           regime_analysis = {
               "current_regime": current_regime,
               "regime_distribution": {
                   "low_vol_days": int(regime_counts.get("Low", 0)),
                   "medium_vol_days": int(regime_counts.get("Medium", 0)),
                   "high_vol_days": int(regime_counts.get("High", 0))
               },
               "regime_thresholds": {
                   "low_threshold": round(vol_25th, 4),
                   "high_threshold": round(vol_75th, 4)
               },
               "average_regime_duration": round(avg_regime_duration, 1),
               "total_regime_changes": len(regime_changes)
           }
           
           return regime_analysis
           
       except Exception as e:
           logger.error(f"Regime analysis failed: {e}")
           return {"error": f"Regime analysis failed: {str(e)}"}
   
   def _simple_garch_forecast(self, inputs: VolatilityInputs) -> Dict[str, float]:
       """Simple GARCH-like volatility forecast"""
       if inputs.price_data is None or inputs.price_data.empty:
           return {"error": "No price data available"}
       
       try:
           prices = inputs.price_data['Close'] if 'Close' in inputs.price_data.columns else inputs.price_data.iloc[:, 0]
           returns = prices.pct_change().dropna()
           
           if len(returns) < 30:
               return {"error": "Insufficient data for forecasting"}
           
           # Simple EWMA model (exponentially weighted moving average)
           # This is a simplified version of GARCH(1,1)
           lambda_param = 0.94  # Decay factor (common value)
           
           # Calculate squared returns
           squared_returns = returns ** 2
           
           # Initialize variance forecast
           long_run_variance = squared_returns.mean()
           current_variance = squared_returns.iloc[-1]
           
           # Simple EWMA forecast
           forecasted_variance = lambda_param * current_variance + (1 - lambda_param) * long_run_variance
           forecasted_volatility = np.sqrt(forecasted_variance * TRADING_DAYS_PER_YEAR)
           
           # Calculate some forecast statistics
           recent_vol = returns.tail(20).std() * np.sqrt(TRADING_DAYS_PER_YEAR)
           long_term_vol = returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR)
           
           forecast_result = {
               "forecasted_volatility": round(forecasted_volatility, 6),
               "current_volatility": round(recent_vol, 6),
               "long_term_volatility": round(long_term_vol, 6),
               "forecast_vs_current": round((forecasted_volatility / recent_vol - 1) * 100, 2),
               "forecast_method": "Simple EWMA"
           }
           
           return forecast_result
           
       except Exception as e:
           logger.warning(f"GARCH forecast failed: {e}")
           return {"error": f"Forecast failed: {str(e)}"}

# Usage example and test
if __name__ == "__main__":
   # Create sample price data for testing
   np.random.seed(42)
   dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
   
   # Generate realistic price data with changing volatility
   returns = []
   vol_regime = 0.15  # Starting volatility
   
   for i in range(len(dates)):
       # Change volatility regime occasionally
       if i % 60 == 0:  # Every 60 days
           vol_regime = np.random.uniform(0.1, 0.4)
       
       daily_return = np.random.normal(0.0005, vol_regime / np.sqrt(252))
       returns.append(daily_return)
   
   # Generate price series
   prices = [100.0]  # Starting price
   for ret in returns:
       prices.append(prices[-1] * (1 + ret))
   
   # Create DataFrame
   price_data = pd.DataFrame({
       'Date': dates[:len(prices)],
       'Close': prices,
       'High': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
       'Low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
       'Open': [p * (1 + np.random.normal(0, 0.005)) for p in prices]
   })
   price_data.set_index('Date', inplace=True)
   
   # Create sample options data
   current_price = prices[-1]
   strikes = np.arange(current_price * 0.8, current_price * 1.2, 2.5)
   
   options_data = []
   for strike in strikes:
       # Generate realistic option prices with IV
       moneyness = strike / current_price
       base_iv = 0.25 + 0.1 * abs(moneyness - 1)  # Volatility smile
       
       # Call option
       bs_inputs = BlackScholesInputs(
           spot_price=current_price,
           strike_price=strike,
           time_to_expiry=30/365,
           risk_free_rate=0.05,
           volatility=base_iv,
           option_type="call"
       )
       
       bs_agent = BlackScholesPricingAgent()
       call_price = bs_agent.calculate_option_price(bs_inputs).option_price
       
       options_data.append({
           'strike': strike,
           'lastPrice': call_price,
           'type': 'call'
       })
       
       # Put option
       bs_inputs.option_type = "put"
       put_price = bs_agent.calculate_option_price(bs_inputs).option_price
       
       options_data.append({
           'strike': strike,
           'lastPrice': put_price,
           'type': 'put'
       })
   
   options_df = pd.DataFrame(options_data)
   
   # Test volatility agent
   vol_agent = VolatilityAgent()
   
   vol_inputs = VolatilityInputs(
       symbol="TEST",
       price_data=price_data,
       options_data=options_df,
       spot_price=current_price,
       risk_free_rate=0.05,
       dividend_yield=0.02,
       analysis_period=30
   )
   
   print("Testing Volatility Analysis Agent")
   print("="*60)
   
   result = vol_agent.analyze_volatility(vol_inputs)
   
   # Display results
   print("Historical Volatility:")
   for key, value in result.historical_volatility.items():
       if isinstance(value, (int, float)):
           if 'volatility' in key:
               print(f"  {key.replace('_', ' ').title()}: {value:.2%}")
           else:
               print(f"  {key.replace('_', ' ').title()}: {value}")
       else:
           print(f"  {key.replace('_', ' ').title()}: {value}")
   
   print("\nImplied Volatility:")
   if "statistics" in result.implied_volatility:
       iv_stats = result.implied_volatility["statistics"]
       for key, value in iv_stats.items():
           if 'vol' in key and isinstance(value, (int, float)):
               print(f"  {key.replace('_', ' ').title()}: {value:.2%}")
           else:
               print(f"  {key.replace('_', ' ').title()}: {value}")
   
   print("\nVolatility Statistics:")
   for key, value in result.volatility_statistics.items():
       if isinstance(value, (int, float)):
           if 'volatility' in key:
               print(f"  {key.replace('_', ' ').title()}: {value:.2%}")
           elif 'percentile' in key:
               print(f"  {key.replace('_', ' ').title()}: {value}%")
           else:
               print(f"  {key.replace('_', ' ').title()}: {value:.4f}")
   
   print("\nRegime Analysis:")
   for key, value in result.regime_analysis.items():
       if isinstance(value, dict):
           print(f"  {key.replace('_', ' ').title()}:")
           for subkey, subvalue in value.items():
               print(f"    {subkey.replace('_', ' ').title()}: {subvalue}")
       else:
           print(f"  {key.replace('_', ' ').title()}: {value}")
   
   if result.garch_forecast:
       print("\nGARCH Forecast:")
       for key, value in result.garch_forecast.items():
           if isinstance(value, (int, float)) and 'volatility' in key:
               print(f"  {key.replace('_', ' ').title()}: {value:.2%}")
           else:
               print(f"  {key.replace('_', ' ').title()}: {value}")
   
   if result.volatility_surface:
       print("\nVolatility Surface:")
       if "surface_by_moneyness" in result.volatility_surface:
           for moneyness, data in result.volatility_surface["surface_by_moneyness"].items():
               print(f"  {moneyness.replace('_', ' ').title()}: {data['avg_implied_vol']:.2%} ({data['count']} options)")
       
       if "skew_analysis" in result.volatility_surface:
           skew = result.volatility_surface["skew_analysis"]
           for key, value in skew.items():
               if isinstance(value, (int, float)):
                   print(f"  {key.replace('_', ' ').title()}: {value:.2%}")
               else:
                   print(f"  {key.replace('_', ' ').title()}: {value}")