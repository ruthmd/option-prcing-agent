from typing import Dict, List, Any, Optional, Tuple
from pydantic import BaseModel, Field
import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq
import math
from datetime import datetime, date
from loguru import logger

from ...config.settings import settings
from ...utils.constants import TRADING_DAYS_PER_YEAR, PRICING_BOUNDS

class BlackScholesInputs(BaseModel):
    """Input parameters for Black-Scholes calculation"""
    spot_price: float = Field(gt=0, description="Current stock price")
    strike_price: float = Field(gt=0, description="Option strike price")
    time_to_expiry: float = Field(gt=0, description="Time to expiry in years")
    risk_free_rate: float = Field(ge=-0.1, le=0.3, description="Risk-free interest rate")  # Relaxed from 0.5
    volatility: float = Field(gt=0, le=3.0, description="Annualized volatility")  # Relaxed from 5.0
    dividend_yield: float = Field(ge=0, le=0.25, default=0.0, description="Dividend yield")  # Reduced from 0.3
    option_type: str = Field(pattern="^(call|put)$", description="Option type: call or put")

class BlackScholesResult(BaseModel):
    """Result of Black-Scholes calculation"""
    option_price: float
    intrinsic_value: float
    time_value: float
    moneyness: float
    greeks: Dict[str, float]
    inputs: BlackScholesInputs
    calculation_method: str = "Black-Scholes-Merton"

class BlackScholesPricingAgent:
    """Black-Scholes option pricing agent with Greeks calculation"""
    
    def __init__(self):
        self.name = "BlackScholes"
        logger.info("Black-Scholes pricing agent initialized")
    
    def calculate_option_price(self, inputs: BlackScholesInputs) -> BlackScholesResult:
        """Calculate option price using Black-Scholes-Merton formula"""
        try:
            # Validate inputs
            self._validate_inputs(inputs)
            
            # Calculate d1 and d2
            d1, d2 = self._calculate_d1_d2(inputs)
            
            # Calculate option price
            if inputs.option_type.lower() == "call":
                option_price = self._calculate_call_price(inputs, d1, d2)
                intrinsic_value = max(0, inputs.spot_price - inputs.strike_price)
            else:
                option_price = self._calculate_put_price(inputs, d1, d2)
                intrinsic_value = max(0, inputs.strike_price - inputs.spot_price)
            
            # Calculate Greeks
            greeks = self._calculate_greeks(inputs, d1, d2)
            
            # Additional metrics
            time_value = option_price - intrinsic_value
            moneyness = inputs.spot_price / inputs.strike_price
            
            result = BlackScholesResult(
                option_price=round(option_price, 4),
                intrinsic_value=round(intrinsic_value, 4),
                time_value=round(time_value, 4),
                moneyness=round(moneyness, 4),
                greeks=greeks,
                inputs=inputs
            )
            
            logger.debug(f"BS calculation completed: {inputs.option_type} price = {option_price:.4f}")
            return result
            
        except Exception as e:
            logger.error(f"Black-Scholes calculation failed: {e}")
            raise ValueError(f"Black-Scholes calculation error: {str(e)}")
    
    def _validate_inputs(self, inputs: BlackScholesInputs):
        """Validate input parameters"""
        if inputs.spot_price <= 0:
            raise ValueError("Spot price must be positive")
        
        if inputs.strike_price <= 0:
            raise ValueError("Strike price must be positive")
        
        if inputs.time_to_expiry <= 0:
            raise ValueError("Time to expiry must be positive")
        
        if inputs.volatility <= 0:
            raise ValueError("Volatility must be positive")
        
        if inputs.volatility > PRICING_BOUNDS["max_volatility"]:
            logger.warning(f"High volatility detected: {inputs.volatility:.1%}")
        
        if inputs.time_to_expiry > PRICING_BOUNDS["max_time_to_expiry"]:
            logger.warning(f"Long time to expiry: {inputs.time_to_expiry:.2f} years")
    
    def _calculate_d1_d2(self, inputs: BlackScholesInputs) -> Tuple[float, float]:
        """Calculate d1 and d2 parameters"""
        S = inputs.spot_price
        K = inputs.strike_price
        T = inputs.time_to_expiry
        r = inputs.risk_free_rate
        q = inputs.dividend_yield
        sigma = inputs.volatility
        
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        return d1, d2
    
    def _calculate_call_price(self, inputs: BlackScholesInputs, d1: float, d2: float) -> float:
        """Calculate call option price"""
        S = inputs.spot_price
        K = inputs.strike_price
        T = inputs.time_to_expiry
        r = inputs.risk_free_rate
        q = inputs.dividend_yield
        
        call_price = (S * np.exp(-q * T) * norm.cdf(d1) - 
                     K * np.exp(-r * T) * norm.cdf(d2))
        
        return max(call_price, 0)  # Ensure non-negative
    
    def _calculate_put_price(self, inputs: BlackScholesInputs, d1: float, d2: float) -> float:
        """Calculate put option price"""
        S = inputs.spot_price
        K = inputs.strike_price
        T = inputs.time_to_expiry
        r = inputs.risk_free_rate
        q = inputs.dividend_yield
        
        put_price = (K * np.exp(-r * T) * norm.cdf(-d2) - 
                    S * np.exp(-q * T) * norm.cdf(-d1))
        
        return max(put_price, 0)  # Ensure non-negative
    
    def _calculate_greeks(self, inputs: BlackScholesInputs, d1: float, d2: float) -> Dict[str, float]:
        """Calculate option Greeks"""
        S = inputs.spot_price
        K = inputs.strike_price
        T = inputs.time_to_expiry
        r = inputs.risk_free_rate
        q = inputs.dividend_yield
        sigma = inputs.volatility
        
        # Common terms
        pdf_d1 = norm.pdf(d1)
        cdf_d1 = norm.cdf(d1)
        cdf_minus_d1 = norm.cdf(-d1)
        cdf_d2 = norm.cdf(d2)
        cdf_minus_d2 = norm.cdf(-d2)
        
        # Delta
        if inputs.option_type.lower() == "call":
            delta = np.exp(-q * T) * cdf_d1
        else:
            delta = -np.exp(-q * T) * cdf_minus_d1
        
        # Gamma (same for calls and puts)
        gamma = (np.exp(-q * T) * pdf_d1) / (S * sigma * np.sqrt(T))
        
        # Theta
        first_term = -(S * pdf_d1 * sigma * np.exp(-q * T)) / (2 * np.sqrt(T))
        
        if inputs.option_type.lower() == "call":
            second_term = r * K * np.exp(-r * T) * cdf_d2
            third_term = -q * S * np.exp(-q * T) * cdf_d1
            theta = first_term - second_term + third_term
        else:
            second_term = -r * K * np.exp(-r * T) * cdf_minus_d2
            third_term = q * S * np.exp(-q * T) * cdf_minus_d1
            theta = first_term + second_term + third_term
        
        # Convert theta to per-day
        theta_per_day = theta / 365
        
        # Vega (same for calls and puts)
        vega = S * np.exp(-q * T) * pdf_d1 * np.sqrt(T)
        
        # Convert vega to per 1% volatility change
        vega_percent = vega / 100
        
        # Rho
        if inputs.option_type.lower() == "call":
            rho = K * T * np.exp(-r * T) * cdf_d2
        else:
            rho = -K * T * np.exp(-r * T) * cdf_minus_d2
        
        # Convert rho to per 1% interest rate change
        rho_percent = rho / 100
        
        return {
            "delta": round(delta, 6),
            "gamma": round(gamma, 6),
            "theta": round(theta_per_day, 6),
            "vega": round(vega_percent, 4),
            "rho": round(rho_percent, 4)
        }
    
    def calculate_implied_volatility(
        self, 
        market_price: float, 
        spot_price: float,
        strike_price: float,
        time_to_expiry: float,
        risk_free_rate: float,
        dividend_yield: float = 0.0,
        option_type: str = "call"
    ) -> Optional[float]:
        """Calculate implied volatility using Brent's method"""
        try:
            def objective_function(vol):
                inputs = BlackScholesInputs(
                    spot_price=spot_price,
                    strike_price=strike_price,
                    time_to_expiry=time_to_expiry,
                    risk_free_rate=risk_free_rate,
                    volatility=vol,
                    dividend_yield=dividend_yield,
                    option_type=option_type
                )
                theoretical_price = self.calculate_option_price(inputs).option_price
                return theoretical_price - market_price
            
            # Use Brent's method to find implied volatility
            implied_vol = brentq(
                objective_function,
                a=0.001,  # 0.1% minimum volatility
                b=5.0,    # 500% maximum volatility
                xtol=1e-6,
                maxiter=100
            )
            
            return round(implied_vol, 6)
            
        except Exception as e:
            logger.warning(f"Implied volatility calculation failed: {e}")
            return None
    
    def sensitivity_analysis(
        self, 
        base_inputs: BlackScholesInputs,
        parameter: str,
        range_percent: float = 0.1
    ) -> Dict[str, List[float]]:
        """Perform sensitivity analysis on a parameter"""
        try:
            base_price = self.calculate_option_price(base_inputs).option_price
            
            # Create parameter variations
            variations = np.linspace(-range_percent, range_percent, 21)
            prices = []
            parameter_values = []
            
            for variation in variations:
                modified_inputs = base_inputs.copy()
                
                if parameter == "spot_price":
                    new_value = base_inputs.spot_price * (1 + variation)
                    modified_inputs.spot_price = new_value
                    parameter_values.append(new_value)
                elif parameter == "volatility":
                    new_value = base_inputs.volatility * (1 + variation)
                    modified_inputs.volatility = max(0.001, new_value)  # Minimum volatility
                    parameter_values.append(new_value)
                elif parameter == "time_to_expiry":
                    new_value = base_inputs.time_to_expiry * (1 + variation)
                    modified_inputs.time_to_expiry = max(1/365, new_value)  # Minimum 1 day
                    parameter_values.append(new_value)
                elif parameter == "risk_free_rate":
                    new_value = base_inputs.risk_free_rate + variation  # Absolute change
                    modified_inputs.risk_free_rate = new_value
                    parameter_values.append(new_value)
                else:
                    raise ValueError(f"Unsupported parameter: {parameter}")
                
                try:
                    price = self.calculate_option_price(modified_inputs).option_price
                    prices.append(price)
                except:
                    prices.append(np.nan)
            
            return {
                "parameter_values": parameter_values,
                "option_prices": prices,
                "base_price": base_price,
                "parameter": parameter
            }
            
        except Exception as e:
            logger.error(f"Sensitivity analysis failed: {e}")
            raise ValueError(f"Sensitivity analysis error: {str(e)}")

# Usage example and test
if __name__ == "__main__":
    bs_agent = BlackScholesPricingAgent()
    
    # Test case: AAPL call option
    test_inputs = BlackScholesInputs(
        spot_price=150.0,
        strike_price=155.0,
        time_to_expiry=30/365,  # 30 days
        risk_free_rate=0.05,    # 5%
        volatility=0.25,        # 25%
        dividend_yield=0.02,    # 2%
        option_type="call"
    )
    
    print("Testing Black-Scholes Pricing Agent")
    print("="*50)
    
    # Calculate option price
    result = bs_agent.calculate_option_price(test_inputs)
    
    print(f"Option Type: {result.inputs.option_type.upper()}")
    print(f"Spot Price: ${result.inputs.spot_price}")
    print(f"Strike Price: ${result.inputs.strike_price}")
    print(f"Time to Expiry: {result.inputs.time_to_expiry:.4f} years")
    print(f"Volatility: {result.inputs.volatility:.1%}")
    print(f"Risk-free Rate: {result.inputs.risk_free_rate:.1%}")
    print(f"Dividend Yield: {result.inputs.dividend_yield:.1%}")
    print()
    print("RESULTS:")
    print(f"Option Price: ${result.option_price:.4f}")
    print(f"Intrinsic Value: ${result.intrinsic_value:.4f}")
    print(f"Time Value: ${result.time_value:.4f}")
    print(f"Moneyness: {result.moneyness:.4f}")
    print()
    print("GREEKS:")
    for greek, value in result.greeks.items():
        print(f"{greek.capitalize()}: {value}")
    
    # Test implied volatility
    print("\n" + "="*50)
    print("Testing Implied Volatility")
    
    market_price = result.option_price  # Use calculated price as "market" price
    implied_vol = bs_agent.calculate_implied_volatility(
        market_price=market_price,
        spot_price=test_inputs.spot_price,
        strike_price=test_inputs.strike_price,
        time_to_expiry=test_inputs.time_to_expiry,
        risk_free_rate=test_inputs.risk_free_rate,
        dividend_yield=test_inputs.dividend_yield,
        option_type=test_inputs.option_type
    )
    
    print(f"Market Price: ${market_price:.4f}")
    print(f"Input Volatility: {test_inputs.volatility:.4f}")
    print(f"Implied Volatility: {implied_vol:.4f}")
    print(f"Difference: {abs(test_inputs.volatility - implied_vol):.6f}")
    
    # Test sensitivity analysis
    print("\n" + "="*50)
    print("Testing Sensitivity Analysis")
    
    sensitivity = bs_agent.sensitivity_analysis(
        test_inputs, 
        parameter="spot_price", 
        range_percent=0.1
    )
    
    print(f"Base Price: ${sensitivity['base_price']:.4f}")
    print(f"Price Range: ${min(sensitivity['option_prices']):.4f} - ${max(sensitivity['option_prices']):.4f}")
    print(f"Spot Price Range: ${min(sensitivity['parameter_values']):.2f} - ${max(sensitivity['parameter_values']):.2f}")