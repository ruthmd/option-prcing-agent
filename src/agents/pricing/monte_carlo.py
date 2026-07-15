from typing import Dict, List, Any, Optional, Tuple, Callable
from pydantic import BaseModel, Field
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from loguru import logger

from ...config.settings import settings
from ...utils.constants import TRADING_DAYS_PER_YEAR, PRICING_BOUNDS

class MonteCarloInputs(BaseModel):
    """Input parameters for Monte Carlo simulation"""
    spot_price: float = Field(gt=0, description="Current stock price")
    strike_price: float = Field(gt=0, description="Option strike price")
    time_to_expiry: float = Field(gt=0, description="Time to expiry in years")
    risk_free_rate: float = Field(ge=-0.1, le=0.5, description="Risk-free interest rate")
    volatility: float = Field(gt=0, le=5.0, description="Annualized volatility")
    dividend_yield: float = Field(ge=0, le=0.3, default=0.0, description="Dividend yield")
    option_type: str = Field(pattern="^(call|put)$", description="Option type: call or put")
    barrier_type: Optional[str] = Field(default=None, pattern="^(up-and-out|up-and-in|down-and-out|down-and-in)$")
    barrier_level: Optional[float] = Field(default=None, gt=0, description="Barrier level for barrier options")
    asian_type: Optional[str] = Field(default=None, pattern="^(arithmetic|geometric)$", description="Asian option type")
    num_simulations: int = Field(ge=1000, le=1000000, default=100000, description="Number of Monte Carlo simulations")
    num_steps: int = Field(ge=1, le=1000, default=252, description="Number of time steps per simulation")
    random_seed: Optional[int] = Field(default=None, description="Random seed for reproducibility")

class MonteCarloResult(BaseModel):
    """Result of Monte Carlo simulation"""
    option_price: float
    intrinsic_value: float
    time_value: float
    standard_error: float
    confidence_interval: Tuple[float, float]
    convergence_data: Dict[str, List[float]]
    path_statistics: Dict[str, float]
    greeks: Dict[str, float]
    inputs: MonteCarloInputs
    calculation_method: str = "Monte Carlo Simulation"

class MonteCarloPricingAgent:
    """Monte Carlo option pricing agent for exotic and standard options"""
    
    def __init__(self):
        self.name = "MonteCarlo"
        logger.info("Monte Carlo pricing agent initialized")
    
    def calculate_option_price(self, inputs: MonteCarloInputs) -> MonteCarloResult:
        """Calculate option price using Monte Carlo simulation"""
        try:
            option_price, std_error, confidence_interval, convergence_data, path_stats = self._price_core(inputs)

            # Calculate Greeks using finite differences (does NOT recurse back
            # into this method — it reprices via _price_only to avoid
            # unbounded recursion through repeated Greeks calculation)
            greeks = self._calculate_greeks(inputs)

            if inputs.option_type.lower() == "call":
                intrinsic_value = max(0, inputs.spot_price - inputs.strike_price)
            else:
                intrinsic_value = max(0, inputs.strike_price - inputs.spot_price)

            result = MonteCarloResult(
                option_price=round(option_price, 4),
                intrinsic_value=round(intrinsic_value, 4),
                time_value=round(option_price - intrinsic_value, 4),
                standard_error=round(std_error, 6),
                confidence_interval=(round(confidence_interval[0], 4), round(confidence_interval[1], 4)),
                convergence_data=convergence_data,
                path_statistics=path_stats,
                greeks=greeks,
                inputs=inputs
            )

            logger.debug(f"Monte Carlo calculation completed: {inputs.option_type} price = {option_price:.4f} ± {std_error:.4f}")
            return result

        except Exception as e:
            logger.error(f"Monte Carlo calculation failed: {e}")
            raise ValueError(f"Monte Carlo calculation error: {str(e)}")

    def _price_core(self, inputs: MonteCarloInputs):
        """Core simulation pricing, with no Greeks computation — safe to call repeatedly for bumped inputs"""
        self._validate_inputs(inputs)

        if inputs.random_seed is not None:
            np.random.seed(inputs.random_seed)

        price_paths = self._generate_price_paths(inputs)
        payoffs = self._calculate_payoffs(inputs, price_paths)
        option_price, std_error, confidence_interval = self._calculate_price_and_stats(payoffs, inputs)
        convergence_data = self._analyze_convergence(payoffs)
        path_stats = self._calculate_path_statistics(price_paths)

        return option_price, std_error, confidence_interval, convergence_data, path_stats

    def _price_only(self, inputs: MonteCarloInputs) -> float:
        """Price-only helper for Greeks finite-differencing — never touches Greeks"""
        option_price, _, _, _, _ = self._price_core(inputs)
        return option_price

    def _validate_inputs(self, inputs: MonteCarloInputs):
        """Validate input parameters"""
        if inputs.spot_price <= 0:
            raise ValueError("Spot price must be positive")
        
        if inputs.strike_price <= 0:
            raise ValueError("Strike price must be positive")
        
        if inputs.time_to_expiry <= 0:
            raise ValueError("Time to expiry must be positive")
        
        if inputs.volatility <= 0:
            raise ValueError("Volatility must be positive")
        
        if inputs.num_simulations < 1000:
            logger.warning("Low number of simulations may give inaccurate results")
        
        if inputs.barrier_type and inputs.barrier_level is None:
            raise ValueError("Barrier level required for barrier options")
        
        if inputs.barrier_level and not inputs.barrier_type:
            raise ValueError("Barrier type required when barrier level is specified")
    
    def _generate_price_paths(self, inputs: MonteCarloInputs) -> np.ndarray:
        """Generate stock price paths using geometric Brownian motion"""
        S0 = inputs.spot_price
        r = inputs.risk_free_rate
        q = inputs.dividend_yield
        sigma = inputs.volatility
        T = inputs.time_to_expiry
        n_sims = inputs.num_simulations
        n_steps = inputs.num_steps
        
        dt = T / n_steps
        
        # Pre-calculate constants
        drift = (r - q - 0.5 * sigma**2) * dt
        diffusion = sigma * np.sqrt(dt)
        
        # Generate random numbers
        random_matrix = np.random.standard_normal((n_sims, n_steps))
        
        # Calculate log price changes
        log_returns = drift + diffusion * random_matrix
        
        # Calculate price paths
        log_prices = np.zeros((n_sims, n_steps + 1))
        log_prices[:, 0] = np.log(S0)
        
        for i in range(n_steps):
            log_prices[:, i + 1] = log_prices[:, i] + log_returns[:, i]
        
        price_paths = np.exp(log_prices)
        
        return price_paths
    
    def _calculate_payoffs(self, inputs: MonteCarloInputs, price_paths: np.ndarray) -> np.ndarray:
        """Calculate option payoffs based on option type"""
        K = inputs.strike_price
        option_type = inputs.option_type.lower()
        
        if inputs.barrier_type:
            return self._calculate_barrier_payoffs(inputs, price_paths)
        elif inputs.asian_type:
            return self._calculate_asian_payoffs(inputs, price_paths)
        else:
            # Standard European option
            final_prices = price_paths[:, -1]
            
            if option_type == "call":
                payoffs = np.maximum(final_prices - K, 0)
            else:
                payoffs = np.maximum(K - final_prices, 0)
            
            return payoffs
    
    def _calculate_barrier_payoffs(self, inputs: MonteCarloInputs, price_paths: np.ndarray) -> np.ndarray:
        """Calculate payoffs for barrier options"""
        K = inputs.strike_price
        B = inputs.barrier_level
        option_type = inputs.option_type.lower()
        barrier_type = inputs.barrier_type.lower()
        
        # Calculate standard payoffs first
        final_prices = price_paths[:, -1]
        if option_type == "call":
            standard_payoffs = np.maximum(final_prices - K, 0)
        else:
            standard_payoffs = np.maximum(K - final_prices, 0)
        
        # Check barrier conditions
        if "up" in barrier_type:
            barrier_hit = np.any(price_paths >= B, axis=1)
        else:  # down barrier
            barrier_hit = np.any(price_paths <= B, axis=1)
        
        # Apply barrier logic
        if "out" in barrier_type:
            # Knock-out: option becomes worthless if barrier is hit
            payoffs = np.where(barrier_hit, 0, standard_payoffs)
        else:  # knock-in
            # Knock-in: option only has value if barrier is hit
            payoffs = np.where(barrier_hit, standard_payoffs, 0)
        
        return payoffs
    
    def _calculate_asian_payoffs(self, inputs: MonteCarloInputs, price_paths: np.ndarray) -> np.ndarray:
        """Calculate payoffs for Asian options"""
        K = inputs.strike_price
        option_type = inputs.option_type.lower()
        asian_type = inputs.asian_type.lower()
        
        if asian_type == "arithmetic":
            # Arithmetic average
            average_prices = np.mean(price_paths, axis=1)
        else:
            # Geometric average
            log_prices = np.log(price_paths)
            average_log_prices = np.mean(log_prices, axis=1)
            average_prices = np.exp(average_log_prices)
        
        # Calculate payoffs based on average price
        if option_type == "call":
            payoffs = np.maximum(average_prices - K, 0)
        else:
            payoffs = np.maximum(K - average_prices, 0)
        
        return payoffs
    
    def _calculate_price_and_stats(self, payoffs: np.ndarray, inputs: MonteCarloInputs) -> Tuple[float, float, Tuple[float, float]]:
        """Calculate option price and statistical measures"""
        # Discount payoffs to present value
        r = inputs.risk_free_rate
        T = inputs.time_to_expiry
        discount_factor = np.exp(-r * T)
        
        discounted_payoffs = payoffs * discount_factor
        
        # Calculate option price (mean of discounted payoffs)
        option_price = np.mean(discounted_payoffs)
        
        # Calculate standard error
        std_error = np.std(discounted_payoffs) / np.sqrt(len(discounted_payoffs))
        
        # Calculate 95% confidence interval
        z_score = 1.96  # For 95% confidence
        margin_of_error = z_score * std_error
        confidence_interval = (option_price - margin_of_error, option_price + margin_of_error)
        
        return option_price, std_error, confidence_interval
    
    def _analyze_convergence(self, payoffs: np.ndarray) -> Dict[str, List[float]]:
        """Analyze convergence of Monte Carlo simulation"""
        n_sims = len(payoffs)
        check_points = np.logspace(2, np.log10(n_sims), 20, dtype=int)  # 20 points from 100 to n_sims
        check_points = np.unique(check_points)  # Remove duplicates
        
        convergence_prices = []
        convergence_errors = []
        
        for n in check_points:
            sample_payoffs = payoffs[:n]
            price = np.mean(sample_payoffs)
            error = np.std(sample_payoffs) / np.sqrt(n)
            
            convergence_prices.append(price)
            convergence_errors.append(error)
        
        return {
            "simulation_counts": check_points.tolist(),
            "option_prices": convergence_prices,
            "standard_errors": convergence_errors
        }
    
    def _calculate_path_statistics(self, price_paths: np.ndarray) -> Dict[str, float]:
        """Calculate statistics about the generated price paths"""
        final_prices = price_paths[:, -1]
        
        # Calculate returns from initial to final price
        initial_price = price_paths[0, 0]  # Same for all paths
        returns = (final_prices / initial_price) - 1
        
        stats = {
            "mean_final_price": float(np.mean(final_prices)),
            "std_final_price": float(np.std(final_prices)),
            "min_final_price": float(np.min(final_prices)),
            "max_final_price": float(np.max(final_prices)),
            "mean_return": float(np.mean(returns)),
            "std_return": float(np.std(returns)),
            "probability_in_money": self._calculate_prob_in_money(price_paths),
            "max_drawdown": self._calculate_max_drawdown(price_paths)
        }
        
        return stats
    
    def _calculate_prob_in_money(self, price_paths: np.ndarray) -> float:
        """Calculate probability of finishing in-the-money"""
        # This is a placeholder - would need option details
        return 0.5  # 50% as placeholder
    
    def _calculate_max_drawdown(self, price_paths: np.ndarray) -> float:
        """Calculate average maximum drawdown across all paths"""
        drawdowns = []
        
        for path in price_paths:
            running_max = np.maximum.accumulate(path)
            drawdown = (running_max - path) / running_max
            max_drawdown = np.max(drawdown)
            drawdowns.append(max_drawdown)
        
        return float(np.mean(drawdowns))
    
    def _calculate_greeks(self, inputs: MonteCarloInputs) -> Dict[str, float]:
        """Calculate Greeks using finite differences"""
        try:
            base_price = self._price_only(inputs)

            # Delta
            spot_shift = inputs.spot_price * 0.01
            inputs_up = inputs.copy()
            inputs_up.spot_price = inputs.spot_price + spot_shift
            inputs_down = inputs.copy()
            inputs_down.spot_price = inputs.spot_price - spot_shift

            price_up = self._price_only(inputs_up)
            price_down = self._price_only(inputs_down)
            delta = (price_up - price_down) / (2 * spot_shift)

            # Gamma
            gamma = (price_up - 2 * base_price + price_down) / (spot_shift ** 2)

            # Vega
            vol_shift = 0.01
            inputs_vega = inputs.copy()
            inputs_vega.volatility = inputs.volatility + vol_shift
            price_vega = self._price_only(inputs_vega)
            vega = (price_vega - base_price) / vol_shift

            # Theta
            time_shift = 1/365
            if inputs.time_to_expiry > time_shift:
                inputs_theta = inputs.copy()
                inputs_theta.time_to_expiry = inputs.time_to_expiry - time_shift
                price_theta = self._price_only(inputs_theta)
                theta = (price_theta - base_price) / time_shift
            else:
                theta = 0.0

            # Rho
            rate_shift = 0.01
            inputs_rho = inputs.copy()
            inputs_rho.risk_free_rate = inputs.risk_free_rate + rate_shift
            price_rho = self._price_only(inputs_rho)
            rho = (price_rho - base_price) / rate_shift
            
            return {
                "delta": round(delta, 6),
                "gamma": round(gamma, 6),
                "theta": round(theta / 365, 6),  # Per day
                "vega": round(vega / 100, 4),   # Per 1% vol change
                "rho": round(rho / 100, 4)     # Per 1% rate change
            }
            
        except Exception as e:
            logger.warning(f"Monte Carlo Greeks calculation failed: {e}")
            return {"delta": 0.0, "gamma": 0.0, "theta": 0.0, "vega": 0.0, "rho": 0.0}
    
    def variance_reduction_analysis(self, inputs: MonteCarloInputs) -> Dict[str, Any]:
        """Compare different variance reduction techniques"""
        try:
            # Standard Monte Carlo
            standard_result = self.calculate_option_price(inputs)
            
            # Antithetic variates
            antithetic_result = self._calculate_with_antithetic_variates(inputs)
            
            # Control variates (using Black-Scholes as control)
            control_result = self._calculate_with_control_variates(inputs)
            
            comparison = {
                "standard": {
                    "price": standard_result.option_price,
                    "std_error": standard_result.standard_error,
                    "confidence_width": standard_result.confidence_interval[1] - standard_result.confidence_interval[0]
                },
                "antithetic_variates": {
                    "price": antithetic_result["price"],
                    "std_error": antithetic_result["std_error"],
                    "variance_reduction": antithetic_result.get("variance_reduction", 0)
                },
                "control_variates": {
                    "price": control_result["price"],
                    "std_error": control_result["std_error"],
                    "variance_reduction": control_result.get("variance_reduction", 0)
                }
            }
            
            return comparison
            
        except Exception as e:
            logger.error(f"Variance reduction analysis failed: {e}")
            return {"error": str(e)}
    
    def _calculate_with_antithetic_variates(self, inputs: MonteCarloInputs) -> Dict[str, float]:
        """Calculate using antithetic variates variance reduction"""
        # Use half the simulations for each path and its antithetic
        half_sims = inputs.num_simulations // 2
        
        # Generate regular paths
        regular_paths = self._generate_price_paths(inputs)[:half_sims]
        
        # Generate antithetic paths (negate the random numbers)
        np.random.seed(inputs.random_seed if inputs.random_seed else 42)
        antithetic_paths = self._generate_antithetic_paths(inputs, half_sims)
        
        # Calculate payoffs for both
        regular_payoffs = self._calculate_payoffs(inputs, regular_paths)
        antithetic_payoffs = self._calculate_payoffs(inputs, antithetic_paths)
        
        # Combine payoffs
        combined_payoffs = (regular_payoffs + antithetic_payoffs) / 2
        
        # Calculate price and standard error
        r = inputs.risk_free_rate
        T = inputs.time_to_expiry
        discount_factor = np.exp(-r * T)
        
        price = np.mean(combined_payoffs) * discount_factor
        std_error = np.std(combined_payoffs) / np.sqrt(len(combined_payoffs)) * discount_factor
        
        return {
            "price": price,
            "std_error": std_error,
            "variance_reduction": "Applied antithetic variates"
        }
    
    def _generate_antithetic_paths(self, inputs: MonteCarloInputs, n_sims: int) -> np.ndarray:
        """Generate antithetic price paths"""
        S0 = inputs.spot_price
        r = inputs.risk_free_rate
        q = inputs.dividend_yield
        sigma = inputs.volatility
        T = inputs.time_to_expiry
        n_steps = inputs.num_steps
        
        dt = T / n_steps
        drift = (r - q - 0.5 * sigma**2) * dt
        diffusion = sigma * np.sqrt(dt)
        
        # Generate random numbers and their antithetic counterparts
        random_matrix = np.random.standard_normal((n_sims, n_steps))
        antithetic_matrix = -random_matrix  # Antithetic variates
        
        # Calculate log returns
        log_returns = drift + diffusion * antithetic_matrix
        
        # Calculate price paths
        log_prices = np.zeros((n_sims, n_steps + 1))
        log_prices[:, 0] = np.log(S0)
        
        for i in range(n_steps):
            log_prices[:, i + 1] = log_prices[:, i] + log_returns[:, i]
        
        return np.exp(log_prices)
    
    def _calculate_with_control_variates(self, inputs: MonteCarloInputs) -> Dict[str, float]:
        """Calculate using control variates (placeholder implementation)"""
        # This would implement control variates using a known analytical solution
        # For now, return standard calculation
        result = self.calculate_option_price(inputs)
        return {
            "price": result.option_price,
            "std_error": result.standard_error,
            "variance_reduction": "Control variates (not fully implemented)"
        }

# Usage example and test
if __name__ == "__main__":
    mc_agent = MonteCarloPricingAgent()
    
    # Test case: European call option
    test_inputs = MonteCarloInputs(
        spot_price=100.0,
        strike_price=105.0,
        time_to_expiry=90/365,  # 90 days
        risk_free_rate=0.05,    # 5%
        volatility=0.25,        # 25%
        dividend_yield=0.0,
        option_type="call",
        num_simulations=50000,
        num_steps=90,
        random_seed=42  # For reproducibility
    )
    
    print("Testing Monte Carlo Pricing Agent")
    print("="*60)
    
    # Standard European option
    result = mc_agent.calculate_option_price(test_inputs)
    
    print(f"Option Details:")
    print(f"  Type: {result.inputs.option_type.upper()}")
    print(f"  Spot Price: ${result.inputs.spot_price}")
    print(f"  Strike Price: ${result.inputs.strike_price}")
    print(f"  Time to Expiry: {result.inputs.time_to_expiry:.4f} years")
    print(f"  Volatility: {result.inputs.volatility:.1%}")
    print(f"  Simulations: {result.inputs.num_simulations:,}")
    print(f"  Steps per Path: {result.inputs.num_steps}")
    print()
    
    print("Monte Carlo Results:")
    print(f"  Option Price: ${result.option_price:.4f}")
    print(f"  Standard Error: ±${result.standard_error:.4f}")
    print(f"  95% CI: [${result.confidence_interval[0]:.4f}, ${result.confidence_interval[1]:.4f}]")
    print()
    
    print("Path Statistics:")
    for key, value in result.path_statistics.items():
        if isinstance(value, float):
            if 'price' in key.lower():
                print(f"  {key.replace('_', ' ').title()}: ${value:.2f}")
            elif 'return' in key.lower() or 'drawdown' in key.lower():
                print(f"  {key.replace('_', ' ').title()}: {value:.2%}")
            else:
                print(f"  {key.replace('_', ' ').title()}: {value:.4f}")
    
    print("\nGreeks:")
    for greek, value in result.greeks.items():
        print(f"  {greek.capitalize()}: {value}")
    
    # Test barrier option
    print("\n" + "="*60)
    print("Testing Barrier Option (Up-and-Out Call)")
    
    barrier_inputs = test_inputs.copy()
    barrier_inputs.barrier_type = "up-and-out"
    barrier_inputs.barrier_level = 110.0
    barrier_inputs.num_simulations = 25000  # Fewer sims for faster test
    
    barrier_result = mc_agent.calculate_option_price(barrier_inputs)
    
    print(f"Barrier Option Price: ${barrier_result.option_price:.4f}")
    print(f"Standard Option Price: ${result.option_price:.4f}")
    print(f"Barrier Discount: {((result.option_price - barrier_result.option_price) / result.option_price) * 100:.1f}%")
    
    # Test Asian option
    print("\n" + "="*60)
    print("Testing Asian Option (Arithmetic Average)")
    
    asian_inputs = test_inputs.copy()
    asian_inputs.asian_type = "arithmetic"
    asian_inputs.num_simulations = 25000
    
    asian_result = mc_agent.calculate_option_price(asian_inputs)
    
    print(f"Asian Option Price: ${asian_result.option_price:.4f}")
    print(f"Standard Option Price: ${result.option_price:.4f}")
    print(f"Asian Discount: {((result.option_price - asian_result.option_price) / result.option_price) * 100:.1f}%")
    
    # Convergence analysis
    print("\n" + "="*60)
    print("Convergence Analysis")
    
    convergence = result.convergence_data
    final_price = convergence["option_prices"][-1]
    final_error = convergence["standard_errors"][-1]
    
    print(f"Final Price: ${final_price:.4f}")
    print(f"Final Standard Error: ±${final_error:.4f}")
    print(f"Convergence Points: {len(convergence['simulation_counts'])}")
    
    # Show a few convergence points
    for i in [0, len(convergence["simulation_counts"])//2, -1]:
        n_sims = convergence["simulation_counts"][i]
        price = convergence["option_prices"][i]
        error = convergence["standard_errors"][i]
        print(f"  At {n_sims:,} sims: ${price:.4f} ± ${error:.4f}")