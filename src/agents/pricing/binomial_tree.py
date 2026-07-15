from typing import Dict, List, Any, Optional, Tuple
from pydantic import BaseModel, Field
import numpy as np
from datetime import datetime
from loguru import logger

from ...config.settings import settings
from ...utils.constants import TRADING_DAYS_PER_YEAR, PRICING_BOUNDS

class BinomialTreeInputs(BaseModel):
    """Input parameters for binomial tree calculation"""
    spot_price: float = Field(gt=0, description="Current stock price")
    strike_price: float = Field(gt=0, description="Option strike price") 
    time_to_expiry: float = Field(gt=0, description="Time to expiry in years")
    risk_free_rate: float = Field(ge=-0.1, le=0.5, description="Risk-free interest rate")
    volatility: float = Field(gt=0, le=5.0, description="Annualized volatility")
    dividend_yield: float = Field(ge=0, le=0.3, default=0.0, description="Dividend yield")
    option_type: str = Field(pattern="^(call|put)$", description="Option type: call or put")
    option_style: str = Field(pattern="^(american|european)$", default="american", description="Option exercise style")
    steps: int = Field(ge=10, le=1000, default=100, description="Number of time steps")

class BinomialTreeResult(BaseModel):
    """Result of binomial tree calculation"""
    option_price: float
    intrinsic_value: float
    time_value: float
    early_exercise_premium: float
    optimal_exercise_boundary: List[Tuple[int, float]]  # (step, stock_price)
    greeks: Dict[str, float]
    inputs: BinomialTreeInputs
    calculation_method: str = "Binomial Tree"

class BinomialTreePricingAgent:
    """Binomial tree option pricing agent for American and European options"""
    
    def __init__(self):
        self.name = "BinomialTree"
        logger.info("Binomial tree pricing agent initialized")
    
    def calculate_option_price(self, inputs: BinomialTreeInputs) -> BinomialTreeResult:
        """Calculate option price using binomial tree method"""
        try:
            option_price, intrinsic_value, exercise_boundary, european_price = self._price_core(inputs)

            # Calculate early exercise premium (American vs European)
            early_exercise_premium = option_price - european_price

            # Calculate Greeks using finite differences (does NOT recurse back
            # into this method — it reprices via _price_only to avoid
            # unbounded recursion through repeated Greeks calculation)
            greeks = self._calculate_greeks(inputs)

            result = BinomialTreeResult(
                option_price=round(option_price, 4),
                intrinsic_value=round(intrinsic_value, 4),
                time_value=round(option_price - intrinsic_value, 4),
                early_exercise_premium=round(early_exercise_premium, 4),
                optimal_exercise_boundary=exercise_boundary,
                greeks=greeks,
                inputs=inputs
            )

            logger.debug(f"Binomial tree calculation completed: {inputs.option_type} price = {option_price:.4f}")
            return result

        except Exception as e:
            logger.error(f"Binomial tree calculation failed: {e}")
            raise ValueError(f"Binomial tree calculation error: {str(e)}")

    def _price_core(self, inputs: BinomialTreeInputs) -> Tuple[float, float, List[Tuple[int, float]], float]:
        """Core tree pricing, with no Greeks computation — safe to call repeatedly for bumped inputs"""
        self._validate_inputs(inputs)

        dt = inputs.time_to_expiry / inputs.steps
        u, d, p = self._calculate_tree_parameters(inputs, dt)

        stock_tree = self._build_stock_tree(inputs, u, d)
        option_tree, exercise_boundary = self._build_option_tree(inputs, stock_tree, p, dt)
        option_price = option_tree[0, 0]

        if inputs.option_type.lower() == "call":
            intrinsic_value = max(0, inputs.spot_price - inputs.strike_price)
        else:
            intrinsic_value = max(0, inputs.strike_price - inputs.spot_price)

        european_price = self._calculate_european_price(inputs, stock_tree, p, dt)

        return option_price, intrinsic_value, exercise_boundary, european_price

    def _price_only(self, inputs: BinomialTreeInputs) -> float:
        """Price-only helper for Greeks finite-differencing — never touches Greeks"""
        option_price, _, _, _ = self._price_core(inputs)
        return option_price

    def _validate_inputs(self, inputs: BinomialTreeInputs):
        """Validate input parameters"""
        if inputs.spot_price <= 0:
            raise ValueError("Spot price must be positive")
        
        if inputs.strike_price <= 0:
            raise ValueError("Strike price must be positive")
        
        if inputs.time_to_expiry <= 0:
            raise ValueError("Time to expiry must be positive")
        
        if inputs.volatility <= 0:
            raise ValueError("Volatility must be positive")
        
        if inputs.steps < 10:
            raise ValueError("Minimum 10 steps required for stability")
        
        if inputs.steps > 500:
            logger.warning(f"High number of steps ({inputs.steps}) may be slow")
    
    def _calculate_tree_parameters(self, inputs: BinomialTreeInputs, dt: float) -> Tuple[float, float, float]:
        """Calculate binomial tree parameters (u, d, p)"""
        # Cox-Ross-Rubinstein parameterization
        sigma = inputs.volatility
        r = inputs.risk_free_rate
        q = inputs.dividend_yield
        
        # Up and down factors
        u = np.exp(sigma * np.sqrt(dt))
        d = 1 / u
        
        # Risk-neutral probability
        a = np.exp((r - q) * dt)
        p = (a - d) / (u - d)
        
        # Validate probability
        if not (0 < p < 1):
            raise ValueError(f"Invalid risk-neutral probability: {p}")
        
        return u, d, p
    
    def _build_stock_tree(self, inputs: BinomialTreeInputs, u: float, d: float) -> np.ndarray:
        """Build stock price tree"""
        steps = inputs.steps
        S0 = inputs.spot_price
        
        # Initialize tree
        stock_tree = np.zeros((steps + 1, steps + 1))
        
        # Fill the tree
        for i in range(steps + 1):
            for j in range(i + 1):
                stock_tree[j, i] = S0 * (u ** (i - j)) * (d ** j)
        
        return stock_tree
    
    def _build_option_tree(
        self, 
        inputs: BinomialTreeInputs, 
        stock_tree: np.ndarray, 
        p: float, 
        dt: float
    ) -> Tuple[np.ndarray, List[Tuple[int, float]]]:
        """Build option price tree with early exercise check"""
        steps = inputs.steps
        K = inputs.strike_price
        r = inputs.risk_free_rate
        option_type = inputs.option_type.lower()
        is_american = inputs.option_style.lower() == "american"
        
        # Initialize option tree
        option_tree = np.zeros((steps + 1, steps + 1))
        exercise_boundary = []
        
        # Terminal condition (payoff at expiration)
        for j in range(steps + 1):
            if option_type == "call":
                option_tree[j, steps] = max(0, stock_tree[j, steps] - K)
            else:
                option_tree[j, steps] = max(0, K - stock_tree[j, steps])
        
        # Backward induction
        discount_factor = np.exp(-r * dt)
        
        for i in range(steps - 1, -1, -1):
            for j in range(i + 1):
                # Calculate continuation value
                continuation_value = discount_factor * (
                    p * option_tree[j, i + 1] + (1 - p) * option_tree[j + 1, i + 1]
                )
                
                # Calculate intrinsic value
                if option_type == "call":
                    intrinsic_value = max(0, stock_tree[j, i] - K)
                else:
                    intrinsic_value = max(0, K - stock_tree[j, i])
                
                # For American options, take maximum of continuation and intrinsic
                if is_american:
                    option_tree[j, i] = max(continuation_value, intrinsic_value)
                    
                    # Record exercise boundary
                    if intrinsic_value > continuation_value and intrinsic_value > 0:
                        exercise_boundary.append((i, stock_tree[j, i]))
                else:
                    option_tree[j, i] = continuation_value
        
        return option_tree, exercise_boundary
    
    def _calculate_european_price(
        self, 
        inputs: BinomialTreeInputs, 
        stock_tree: np.ndarray, 
        p: float, 
        dt: float
    ) -> float:
        """Calculate European option price for comparison"""
        # Create European version of inputs
        european_inputs = inputs.copy()
        european_inputs.option_style = "european"
        
        # Build option tree without early exercise
        option_tree, _ = self._build_option_tree(european_inputs, stock_tree, p, dt)
        
        return option_tree[0, 0]
    
    def _calculate_greeks(self, inputs: BinomialTreeInputs) -> Dict[str, float]:
        """Calculate Greeks using finite differences"""
        try:
            base_price = self._price_only(inputs)

            # Delta: sensitivity to underlying price
            spot_shift = inputs.spot_price * 0.01  # 1% shift
            inputs_up = inputs.copy()
            inputs_up.spot_price = inputs.spot_price + spot_shift
            inputs_down = inputs.copy()
            inputs_down.spot_price = inputs.spot_price - spot_shift

            price_up = self._price_only(inputs_up)
            price_down = self._price_only(inputs_down)

            delta = (price_up - price_down) / (2 * spot_shift)

            # Gamma: second derivative with respect to underlying
            gamma = (price_up - 2 * base_price + price_down) / (spot_shift ** 2)

            # Theta: sensitivity to time decay
            time_shift = 1 / 365  # 1 day
            if inputs.time_to_expiry > time_shift:
                inputs_theta = inputs.copy()
                inputs_theta.time_to_expiry = inputs.time_to_expiry - time_shift
                price_theta = self._price_only(inputs_theta)
                theta = (price_theta - base_price) / time_shift
            else:
                theta = 0.0

            # Vega: sensitivity to volatility
            vol_shift = 0.01  # 1% volatility shift
            inputs_vega_up = inputs.copy()
            inputs_vega_up.volatility = inputs.volatility + vol_shift
            inputs_vega_down = inputs.copy()
            inputs_vega_down.volatility = max(0.001, inputs.volatility - vol_shift)

            price_vega_up = self._price_only(inputs_vega_up)
            price_vega_down = self._price_only(inputs_vega_down)

            vega = (price_vega_up - price_vega_down) / (2 * vol_shift)

            # Rho: sensitivity to interest rate
            rate_shift = 0.01  # 1% rate shift
            inputs_rho_up = inputs.copy()
            inputs_rho_up.risk_free_rate = inputs.risk_free_rate + rate_shift
            inputs_rho_down = inputs.copy()
            inputs_rho_down.risk_free_rate = inputs.risk_free_rate - rate_shift

            price_rho_up = self._price_only(inputs_rho_up)
            price_rho_down = self._price_only(inputs_rho_down)

            rho = (price_rho_up - price_rho_down) / (2 * rate_shift)
            
            return {
                "delta": round(delta, 6),
                "gamma": round(gamma, 6),
                "theta": round(theta / 365, 6),  # Per day
                "vega": round(vega / 100, 4),   # Per 1% vol change
                "rho": round(rho / 100, 4)     # Per 1% rate change
            }
            
        except Exception as e:
            logger.warning(f"Greeks calculation failed: {e}")
            return {
                "delta": 0.0,
                "gamma": 0.0,
                "theta": 0.0,
                "vega": 0.0,
                "rho": 0.0
            }
    
    def analyze_early_exercise(self, inputs: BinomialTreeInputs) -> Dict[str, Any]:
        """Analyze early exercise characteristics"""
        try:
            if inputs.option_style.lower() == "european":
                return {"message": "European options cannot be exercised early"}
            
            result = self.calculate_option_price(inputs)
            
            analysis = {
                "early_exercise_premium": result.early_exercise_premium,
                "premium_percentage": (result.early_exercise_premium / result.option_price) * 100 if result.option_price > 0 else 0,
                "exercise_boundary_points": len(result.optimal_exercise_boundary),
                "recommendation": self._get_exercise_recommendation(result)
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Early exercise analysis failed: {e}")
            return {"error": str(e)}
    
    def _get_exercise_recommendation(self, result: BinomialTreeResult) -> str:
        """Get recommendation about early exercise"""
        if result.early_exercise_premium < 0.01:
            return "Early exercise provides minimal benefit - consider holding"
        elif result.early_exercise_premium > result.option_price * 0.1:
            return "Significant early exercise value - monitor for optimal timing"
        else:
            return "Moderate early exercise value - depends on market conditions"

# Usage example and test
if __name__ == "__main__":
    bt_agent = BinomialTreePricingAgent()
    # Test case: American put option (typically has early exercise value)
    test_inputs = BinomialTreeInputs(
        spot_price=95.0,      # Slightly out-of-the-money put
        strike_price=100.0,
        time_to_expiry=60/365,  # 60 days
        risk_free_rate=0.05,    # 5%
        volatility=0.30,        # 30%
        dividend_yield=0.0,     # No dividends
        option_type="put",
        option_style="american",
        steps=50
    )
    
    print("Testing Binomial Tree Pricing Agent")
    print("="*60)
    
    # Calculate American option price
    result = bt_agent.calculate_option_price(test_inputs)
    
    print(f"Option Details:")
    print(f"  Type: {result.inputs.option_type.upper()} ({result.inputs.option_style})")
    print(f"  Spot Price: ${result.inputs.spot_price}")
    print(f"  Strike Price: ${result.inputs.strike_price}")
    print(f"  Time to Expiry: {result.inputs.time_to_expiry:.4f} years ({result.inputs.time_to_expiry*365:.0f} days)")
    print(f"  Volatility: {result.inputs.volatility:.1%}")
    print(f"  Risk-free Rate: {result.inputs.risk_free_rate:.1%}")
    print(f"  Steps: {result.inputs.steps}")
    print()
    
    print("Pricing Results:")
    print(f"  Option Price: ${result.option_price:.4f}")
    print(f"  Intrinsic Value: ${result.intrinsic_value:.4f}")
    print(f"  Time Value: ${result.time_value:.4f}")
    print(f"  Early Exercise Premium: ${result.early_exercise_premium:.4f}")
    print()
    
    print("Greeks:")
    for greek, value in result.greeks.items():
        print(f"  {greek.capitalize()}: {value}")
    
    # Compare with European option
    print("\n" + "="*60)
    print("American vs European Comparison")
    
    european_inputs = test_inputs.copy()
    european_inputs.option_style = "european"
    european_result = bt_agent.calculate_option_price(european_inputs)
    
    print(f"American Option Price: ${result.option_price:.4f}")
    print(f"European Option Price: ${european_result.option_price:.4f}")
    print(f"Early Exercise Premium: ${result.early_exercise_premium:.4f}")
    print(f"Premium as % of Price: {(result.early_exercise_premium/result.option_price)*100:.2f}%")
    
    # Early exercise analysis
    exercise_analysis = bt_agent.analyze_early_exercise(test_inputs)
    print(f"\nEarly Exercise Analysis:")
    for key, value in exercise_analysis.items():
        print(f"  {key.replace('_', ' ').title()}: {value}")
    
    # Test call option (typically less early exercise value)
    print("\n" + "="*60)
    print("Testing American Call Option")
    
    call_inputs = test_inputs.copy()
    call_inputs.option_type = "call"
    call_inputs.spot_price = 105.0  # In-the-money call
    call_inputs.dividend_yield = 0.03  # Add dividend to increase early exercise value
    
    call_result = bt_agent.calculate_option_price(call_inputs)
    
    print(f"Call Option Price: ${call_result.option_price:.4f}")
    print(f"Early Exercise Premium: ${call_result.early_exercise_premium:.4f}")
    print(f"Exercise Boundary Points: {len(call_result.optimal_exercise_boundary)}")