from typing import Dict, List, Any, Optional, Tuple, Union
from pydantic import BaseModel, Field
from enum import Enum
import numpy as np
import pandas as pd
from dataclasses import dataclass
from datetime import datetime, timedelta
from loguru import logger

from ..pricing.black_scholes import BlackScholesPricingAgent, BlackScholesInputs, BlackScholesResult
from ..pricing.binomial_tree import BinomialTreePricingAgent, BinomialTreeInputs
from ...config.settings import settings
from ...utils.constants import TRADING_DAYS_PER_YEAR, PRICING_BOUNDS

class StrategyType(Enum):
    """Supported option strategy types"""
    LONG_CALL = "long_call"
    LONG_PUT = "long_put"
    SHORT_CALL = "short_call"
    SHORT_PUT = "short_put"
    COVERED_CALL = "covered_call"
    PROTECTIVE_PUT = "protective_put"
    BULL_CALL_SPREAD = "bull_call_spread"
    BEAR_PUT_SPREAD = "bear_put_spread"
    LONG_STRADDLE = "long_straddle"
    SHORT_STRADDLE = "short_straddle"
    LONG_STRANGLE = "long_strangle"
    SHORT_STRANGLE = "short_strangle"
    IRON_CONDOR = "iron_condor"
    IRON_BUTTERFLY = "iron_butterfly"
    LONG_BUTTERFLY = "long_butterfly"
    SHORT_BUTTERFLY = "short_butterfly"
    CALENDAR_SPREAD = "calendar_spread"
    COLLAR = "collar"

@dataclass
class OptionLeg:
    """Individual option leg in a strategy"""
    option_type: str  # 'call' or 'put'
    position: str     # 'long' or 'short'
    strike: float
    quantity: int
    expiry: float     # Time to expiry in years
    premium: float = 0.0
    delta: float = 0.0
    gamma: float = 0.0
    theta: float = 0.0
    vega: float = 0.0

@dataclass
class StockLeg:
    """Stock position leg"""
    position: str  # 'long' or 'short'
    quantity: int
    price: float

class StrategyInputs(BaseModel):
    """Input parameters for strategy analysis"""
    strategy_type: StrategyType
    spot_price: float = Field(gt=0, description="Current stock price")
    strikes: List[float] = Field(description="Strike prices for strategy legs")
    expiries: List[float] = Field(description="Time to expiry for each leg (years)")
    risk_free_rate: float = Field(ge=-0.1, le=0.5, default=0.05)
    volatility: float = Field(gt=0, le=5.0, default=0.25)
    dividend_yield: float = Field(ge=0, le=0.3, default=0.0)
    quantities: Optional[List[int]] = Field(default=None, description="Quantity for each leg")
    market_outlook: Optional[str] = Field(default="neutral", description="bullish/bearish/neutral/volatile")

class StrategyResult(BaseModel):
    """Result of strategy analysis"""
    strategy_name: str
    strategy_type: StrategyType
    legs: List[OptionLeg]
    stock_legs: List[StockLeg] = []
    
    # Financial metrics
    net_premium: float
    max_profit: Union[float, str]
    max_loss: Union[float, str]
    breakeven_points: List[float]
    profit_probability: Optional[float] = None
    
    # Risk metrics
    portfolio_greeks: Dict[str, float]
    risk_reward_ratio: Optional[float] = None
    maximum_risk: float
    
    # Payoff analysis
    payoff_data: Dict[str, List[float]]
    price_range: List[float]
    
    # Strategy characteristics
    market_bias: str
    volatility_bias: str
    time_decay_impact: str
    complexity_rating: int  # 1-5 scale
    
    # Recommendations
    optimal_conditions: Dict[str, Any]
    risk_warnings: List[str]
    management_guidelines: List[str]

class StrategyAnalysisAgent:
    """Comprehensive option strategy analysis agent"""
    
    def __init__(self):
        self.name = "StrategyAnalysis"
        self.bs_agent = BlackScholesPricingAgent()
        self.bt_agent = BinomialTreePricingAgent()
        
        # Strategy templates
        self.strategy_templates = self._initialize_strategy_templates()
        
        logger.info("Strategy Analysis Agent initialized")
    
    def _initialize_strategy_templates(self) -> Dict[StrategyType, Dict[str, Any]]:
        """Initialize strategy templates with characteristics"""
        return {
            StrategyType.LONG_CALL: {
                "legs": [{"option_type": "call", "position": "long", "quantity": 1}],
                "market_bias": "bullish",
                "volatility_bias": "positive",
                "complexity": 1,
                "max_profit": "unlimited",
                "time_decay": "negative"
            },
            StrategyType.LONG_PUT: {
                "legs": [{"option_type": "put", "position": "long", "quantity": 1}],
                "market_bias": "bearish", 
                "volatility_bias": "positive",
                "complexity": 1,
                "max_profit": "strike - premium",
                "time_decay": "negative"
            },
            StrategyType.COVERED_CALL: {
                "legs": [{"option_type": "call", "position": "short", "quantity": 1}],
                "stock_legs": [{"position": "long", "quantity": 100}],
                "market_bias": "neutral_to_bullish",
                "volatility_bias": "negative",
                "complexity": 2,
                "time_decay": "positive"
            },
            StrategyType.PROTECTIVE_PUT: {
                "legs": [{"option_type": "put", "position": "long", "quantity": 1}],
                "stock_legs": [{"position": "long", "quantity": 100}],
                "market_bias": "bullish_with_protection",
                "volatility_bias": "positive",
                "complexity": 2,
                "time_decay": "negative"
            },
            StrategyType.BULL_CALL_SPREAD: {
                "legs": [
                    {"option_type": "call", "position": "long", "quantity": 1, "strike_order": "lower"},
                    {"option_type": "call", "position": "short", "quantity": 1, "strike_order": "higher"}
                ],
                "market_bias": "bullish",
                "volatility_bias": "neutral",
                "complexity": 2,
                "time_decay": "mixed"
            },
            StrategyType.BEAR_PUT_SPREAD: {
                "legs": [
                    {"option_type": "put", "position": "long", "quantity": 1, "strike_order": "higher"},
                    {"option_type": "put", "position": "short", "quantity": 1, "strike_order": "lower"}
                ],
                "market_bias": "bearish",
                "volatility_bias": "neutral",
                "complexity": 2,
                "time_decay": "mixed"
            },
            StrategyType.LONG_STRADDLE: {
                "legs": [
                    {"option_type": "call", "position": "long", "quantity": 1},
                    {"option_type": "put", "position": "long", "quantity": 1}
                ],
                "market_bias": "neutral",
                "volatility_bias": "positive",
                "complexity": 3,
                "time_decay": "negative",
                "same_strike": True
            },
            StrategyType.SHORT_STRADDLE: {
                "legs": [
                    {"option_type": "call", "position": "short", "quantity": 1},
                    {"option_type": "put", "position": "short", "quantity": 1}
                ],
                "market_bias": "neutral",
                "volatility_bias": "negative",
                "complexity": 3,
                "time_decay": "positive",
                "same_strike": True
            },
            StrategyType.LONG_STRANGLE: {
                "legs": [
                    {"option_type": "call", "position": "long", "quantity": 1, "strike_order": "higher"},
                    {"option_type": "put", "position": "long", "quantity": 1, "strike_order": "lower"}
                ],
                "market_bias": "neutral",
                "volatility_bias": "positive",
                "complexity": 3,
                "time_decay": "negative"
            },
            StrategyType.SHORT_STRANGLE: {
                "legs": [
                    {"option_type": "call", "position": "short", "quantity": 1, "strike_order": "higher"},
                    {"option_type": "put", "position": "short", "quantity": 1, "strike_order": "lower"}
                ],
                "market_bias": "neutral",
                "volatility_bias": "negative",
                "complexity": 3,
                "time_decay": "positive"
            },
            StrategyType.IRON_CONDOR: {
                "legs": [
                    {"option_type": "put", "position": "short", "quantity": 1, "strike_order": "lower_mid"},
                    {"option_type": "put", "position": "long", "quantity": 1, "strike_order": "lowest"},
                    {"option_type": "call", "position": "short", "quantity": 1, "strike_order": "upper_mid"},
                    {"option_type": "call", "position": "long", "quantity": 1, "strike_order": "highest"}
                ],
                "market_bias": "neutral",
                "volatility_bias": "negative",
                "complexity": 4,
                "time_decay": "positive"
            },
            StrategyType.IRON_BUTTERFLY: {
                "legs": [
                    {"option_type": "put", "position": "long", "quantity": 1, "strike_order": "lower"},
                    {"option_type": "put", "position": "short", "quantity": 1, "strike_order": "middle"},
                    {"option_type": "call", "position": "short", "quantity": 1, "strike_order": "middle"},
                    {"option_type": "call", "position": "long", "quantity": 1, "strike_order": "higher"}
                ],
                "market_bias": "neutral",
                "volatility_bias": "negative",
                "complexity": 4,
                "time_decay": "positive"
            },
            StrategyType.LONG_BUTTERFLY: {
                "legs": [
                    {"option_type": "call", "position": "long", "quantity": 1, "strike_order": "lower"},
                    {"option_type": "call", "position": "short", "quantity": 2, "strike_order": "middle"},
                    {"option_type": "call", "position": "long", "quantity": 1, "strike_order": "higher"}
                ],
                "market_bias": "neutral",
                "volatility_bias": "negative",
                "complexity": 3,
                "time_decay": "mixed"
            },
            StrategyType.SHORT_BUTTERFLY: {
                "legs": [
                    {"option_type": "call", "position": "short", "quantity": 1, "strike_order": "lower"},
                    {"option_type": "call", "position": "long", "quantity": 2, "strike_order": "middle"},
                    {"option_type": "call", "position": "short", "quantity": 1, "strike_order": "higher"}
                ],
                "market_bias": "neutral",
                "volatility_bias": "positive",
                "complexity": 3,
                "time_decay": "mixed"
            },
            StrategyType.SHORT_CALL: {
                "legs": [{"option_type": "call", "position": "short", "quantity": 1}],
                "market_bias": "bearish_to_neutral",
                "volatility_bias": "negative",
                "complexity": 1,
                "time_decay": "positive"
            },
            StrategyType.SHORT_PUT: {
                "legs": [{"option_type": "put", "position": "short", "quantity": 1}],
                "market_bias": "bullish_to_neutral",
                "volatility_bias": "negative",
                "complexity": 1,
                "time_decay": "positive"
            },
            StrategyType.CALENDAR_SPREAD: {
                "legs": [
                    {"option_type": "call", "position": "short", "quantity": 1},
                    {"option_type": "call", "position": "long", "quantity": 1}
                ],
                "same_strike": True,
                "market_bias": "neutral",
                "volatility_bias": "positive",
                "complexity": 3,
                "time_decay": "positive"
            },
            StrategyType.COLLAR: {
                "legs": [
                    {"option_type": "put", "position": "long", "quantity": 1, "strike_order": "lower"},
                    {"option_type": "call", "position": "short", "quantity": 1, "strike_order": "higher"}
                ],
                "stock_legs": [{"position": "long", "quantity": 100}],
                "market_bias": "neutral_with_protection",
                "volatility_bias": "negative",
                "complexity": 3,
                "time_decay": "mixed"
            }
        }
    
    def analyze_strategy(self, inputs: StrategyInputs) -> StrategyResult:
        """Comprehensive strategy analysis"""
        try:
            logger.info(f"Analyzing {inputs.strategy_type.value} strategy")
            
            # Build strategy legs
            option_legs, stock_legs = self._build_strategy_legs(inputs)
            
            # Calculate individual option prices and Greeks
            for leg in option_legs:
                leg_inputs = BlackScholesInputs(
                    spot_price=inputs.spot_price,
                    strike_price=leg.strike,
                    time_to_expiry=leg.expiry,
                    risk_free_rate=inputs.risk_free_rate,
                    volatility=inputs.volatility,
                    dividend_yield=inputs.dividend_yield,
                    option_type=leg.option_type
                )
                
                bs_result = self.bs_agent.calculate_option_price(leg_inputs)
                
                # Apply position sign (long = positive, short = negative)
                position_sign = 1 if leg.position == "long" else -1
                
                leg.premium = bs_result.option_price * position_sign * leg.quantity
                leg.delta = bs_result.greeks["delta"] * position_sign * leg.quantity
                leg.gamma = bs_result.greeks["gamma"] * position_sign * leg.quantity
                leg.theta = bs_result.greeks["theta"] * position_sign * leg.quantity
                leg.vega = bs_result.greeks["vega"] * position_sign * leg.quantity
            
            # Calculate strategy metrics
            net_premium = sum(leg.premium for leg in option_legs)
            portfolio_greeks = self._calculate_portfolio_greeks(option_legs, stock_legs, inputs.spot_price)
            
            # Generate payoff analysis
            price_range, payoff_data = self._calculate_payoff_analysis(
                option_legs, stock_legs, inputs.spot_price
            )
            
            # Calculate key metrics
            max_profit, max_loss = self._calculate_profit_loss_limits(payoff_data["total_payoff"])
            breakeven_points = self._find_breakeven_points(price_range, payoff_data["total_payoff"])
            
            # Calculate probability of profit
            profit_probability = self._calculate_profit_probability(
                price_range, payoff_data["total_payoff"], inputs.spot_price, inputs.volatility, 
                option_legs[0].expiry if option_legs else 30/365
            )
            
            # Risk assessment
            risk_warnings = self._generate_risk_warnings(inputs, option_legs, stock_legs)
            management_guidelines = self._generate_management_guidelines(inputs.strategy_type)
            
            # Strategy characteristics
            template = self.strategy_templates[inputs.strategy_type]
            
            result = StrategyResult(
                strategy_name=inputs.strategy_type.value.replace("_", " ").title(),
                strategy_type=inputs.strategy_type,
                legs=option_legs,
                stock_legs=stock_legs,
                net_premium=round(net_premium, 2),
                max_profit=max_profit,
                max_loss=max_loss,
                breakeven_points=[round(bp, 2) for bp in breakeven_points],
                profit_probability=profit_probability,
                portfolio_greeks=portfolio_greeks,
                risk_reward_ratio=self._calculate_risk_reward_ratio(max_profit, max_loss),
                maximum_risk=abs(max_loss) if isinstance(max_loss, (int, float)) else float('inf'),
                payoff_data=payoff_data,
                price_range=price_range,
                market_bias=template["market_bias"],
                volatility_bias=template["volatility_bias"],
                time_decay_impact=template["time_decay"],
                complexity_rating=template["complexity"],
                optimal_conditions=self._determine_optimal_conditions(inputs, template),
                risk_warnings=risk_warnings,
                management_guidelines=management_guidelines
            )
            
            logger.info(f"Strategy analysis completed: {result.strategy_name}")
            return result
            
        except Exception as e:
            logger.error(f"Strategy analysis failed: {e}")
            raise ValueError(f"Strategy analysis error: {str(e)}")
    
    def _build_strategy_legs(self, inputs: StrategyInputs) -> Tuple[List[OptionLeg], List[StockLeg]]:
        """Build option and stock legs for the strategy"""
        template = self.strategy_templates[inputs.strategy_type]
        option_legs = []
        stock_legs = []
        
        # Process option legs
        if "legs" in template:
            leg_templates = template["legs"]
            
            # Handle strike assignment based on template requirements
            strikes = self._assign_strikes_to_legs(inputs.strikes, leg_templates, inputs.spot_price)
            
            for i, leg_template in enumerate(leg_templates):
                strike = strikes[i] if i < len(strikes) else inputs.spot_price
                expiry = inputs.expiries[i] if i < len(inputs.expiries) else inputs.expiries[0]
                quantity = inputs.quantities[i] if inputs.quantities and i < len(inputs.quantities) else leg_template.get("quantity", 1)
                
                option_leg = OptionLeg(
                    option_type=leg_template["option_type"],
                    position=leg_template["position"],
                    strike=strike,
                    quantity=quantity,
                    expiry=expiry
                )
                option_legs.append(option_leg)
        
        # Process stock legs
        if "stock_legs" in template:
            for stock_template in template["stock_legs"]:
                stock_leg = StockLeg(
                    position=stock_template["position"],
                    quantity=stock_template["quantity"],
                    price=inputs.spot_price
                )
                stock_legs.append(stock_leg)
        
        return option_legs, stock_legs
    
    def _assign_strikes_to_legs(self, strikes: List[float], leg_templates: List[Dict], spot_price: float) -> List[float]:
        """Assign strikes to legs based on template requirements"""
        if not strikes:
            # Generate default strikes based on strategy
            return [spot_price] * len(leg_templates)
        
        # Check for special strike requirements
        template_requires_sorting = any("strike_order" in leg for leg in leg_templates)
        
        if template_requires_sorting:
            sorted_strikes = sorted(strikes)
            assigned_strikes = []
            
            for leg_template in leg_templates:
                strike_order = leg_template.get("strike_order", "middle")
                
                if strike_order == "lowest":
                    assigned_strikes.append(sorted_strikes[0])
                elif strike_order == "lower" or strike_order == "lower_mid":
                    idx = 0 if len(sorted_strikes) <= 2 else len(sorted_strikes) // 4
                    assigned_strikes.append(sorted_strikes[idx])
                elif strike_order == "middle":
                    idx = len(sorted_strikes) // 2
                    assigned_strikes.append(sorted_strikes[idx])
                elif strike_order == "upper_mid" or strike_order == "higher":
                    idx = -1 if len(sorted_strikes) <= 2 else -(len(sorted_strikes) // 4 + 1)
                    assigned_strikes.append(sorted_strikes[idx])
                elif strike_order == "highest":
                    assigned_strikes.append(sorted_strikes[-1])
                else:
                    assigned_strikes.append(spot_price)
            
            return assigned_strikes
        
        # Handle same strike requirement (straddles)
        template_same_strike = any(leg.get("same_strike", False) for leg in leg_templates)
        if template_same_strike:
            return [strikes[0]] * len(leg_templates)
        
        # Default assignment
        while len(strikes) < len(leg_templates):
            strikes.append(spot_price)
        
        return strikes[:len(leg_templates)]
    
    def _calculate_portfolio_greeks(self, option_legs: List[OptionLeg], stock_legs: List[StockLeg], spot_price: float) -> Dict[str, float]:
        """Calculate portfolio-level Greeks"""
        portfolio_greeks = {
            "delta": 0.0,
            "gamma": 0.0,
            "theta": 0.0,
            "vega": 0.0,
            "rho": 0.0
        }
        
        # Sum option Greeks
        for leg in option_legs:
            portfolio_greeks["delta"] += leg.delta
            portfolio_greeks["gamma"] += leg.gamma
            portfolio_greeks["theta"] += leg.theta
            portfolio_greeks["vega"] += leg.vega
        
        # Add stock delta (1.0 per share)
        for stock_leg in stock_legs:
            stock_delta = 1.0 if stock_leg.position == "long" else -1.0
            portfolio_greeks["delta"] += stock_delta * stock_leg.quantity
        
        # Round for display
        for greek in portfolio_greeks:
            portfolio_greeks[greek] = round(portfolio_greeks[greek], 4)
        
        return portfolio_greeks
    
    def _calculate_payoff_analysis(self, option_legs: List[OptionLeg], stock_legs: List[StockLeg], current_price: float) -> Tuple[List[float], Dict[str, List[float]]]:
        """Calculate payoff analysis across price range"""
        # Generate price range (70% to 130% of current price)
        price_range = np.linspace(current_price * 0.7, current_price * 1.3, 100).tolist()
        
        payoff_data = {
            "option_payoffs": [],
            "stock_payoffs": [],
            "total_payoff": []
        }
        
        option_payoffs = []
        stock_payoffs = []
        total_payoffs = []
        
        for price in price_range:
            # Calculate option payoffs at expiration
            total_option_payoff = 0
            for leg in option_legs:
                if leg.option_type == "call":
                    intrinsic = max(0, price - leg.strike)
                else:
                    intrinsic = max(0, leg.strike - price)
                
                # Account for position and quantity
                position_sign = 1 if leg.position == "long" else -1
                leg_payoff = (intrinsic * position_sign - leg.premium / leg.quantity) * leg.quantity
                total_option_payoff += leg_payoff
            
            option_payoffs.append(total_option_payoff)
            
            # Calculate stock payoffs
            total_stock_payoff = 0
            for stock_leg in stock_legs:
                position_sign = 1 if stock_leg.position == "long" else -1
                stock_payoff = (price - stock_leg.price) * position_sign * stock_leg.quantity
                total_stock_payoff += stock_payoff
            
            stock_payoffs.append(total_stock_payoff)
            
            # Total payoff
            total_payoffs.append(total_option_payoff + total_stock_payoff)
        
        payoff_data["option_payoffs"] = option_payoffs
        payoff_data["stock_payoffs"] = stock_payoffs
        payoff_data["total_payoff"] = total_payoffs
        
        return price_range, payoff_data
    
    def _calculate_profit_loss_limits(self, payoffs: List[float]) -> Tuple[Union[float, str], Union[float, str]]:
        """Calculate maximum profit and loss"""
        max_profit = max(payoffs)
        max_loss = min(payoffs)
        
        # Check for unlimited profit/loss
        if max_profit > 10000:  # Threshold for "unlimited"
            max_profit = "Unlimited"
        else:
            max_profit = round(max_profit, 2)
        
        if max_loss < -10000:  # Threshold for "unlimited"
            max_loss = "Unlimited"
        else:
            max_loss = round(max_loss, 2)
        
        return max_profit, max_loss
    
    def _find_breakeven_points(self, price_range: List[float], payoffs: List[float]) -> List[float]:
        """Find breakeven points where payoff crosses zero"""
        breakeven_points = []
        
        for i in range(len(payoffs) - 1):
            if (payoffs[i] <= 0 <= payoffs[i + 1]) or (payoffs[i] >= 0 >= payoffs[i + 1]):
                # Linear interpolation to find exact breakeven point
                if payoffs[i + 1] != payoffs[i]:
                    ratio = -payoffs[i] / (payoffs[i + 1] - payoffs[i])
                    breakeven = price_range[i] + ratio * (price_range[i + 1] - price_range[i])
                    breakeven_points.append(breakeven)
        
        return breakeven_points
    
    def _calculate_profit_probability(self, price_range: List[float], payoffs: List[float], 
                                   spot_price: float, volatility: float, time_to_expiry: float) -> Optional[float]:
        """Calculate probability of profit using normal distribution assumption"""
        try:
            # Count profitable outcomes
            profitable_outcomes = sum(1 for payoff in payoffs if payoff > 0)
            
            if profitable_outcomes == 0:
                return 0.0
            elif profitable_outcomes == len(payoffs):
                return 1.0
            
            # Use normal distribution to estimate probability
            # This is a simplified model - real calculation would be more complex
            expected_return = 0.0  # Assume no drift for simplicity
            expected_volatility = volatility * np.sqrt(time_to_expiry)
            
            # Find profitable price ranges
            profitable_ranges = []
            in_profitable_range = False
            range_start = None
            
            for i, payoff in enumerate(payoffs):
                if payoff > 0 and not in_profitable_range:
                    in_profitable_range = True
                    range_start = price_range[i]
                elif payoff <= 0 and in_profitable_range:
                    in_profitable_range = False
                    profitable_ranges.append((range_start, price_range[i - 1]))
            
            if in_profitable_range:
                profitable_ranges.append((range_start, price_range[-1]))
            
            # Calculate probability for each range
            total_probability = 0.0
            for start, end in profitable_ranges:
                # Convert to log returns
                log_start = np.log(start / spot_price)
                log_end = np.log(end / spot_price)
                
                # Standard normal probabilities
                z_start = log_start / expected_volatility
                z_end = log_end / expected_volatility
                
                from scipy.stats import norm
                prob_range = norm.cdf(z_end) - norm.cdf(z_start)
                total_probability += prob_range
            
            return min(max(total_probability, 0.0), 1.0)
            
        except Exception as e:
            logger.warning(f"Profit probability calculation failed: {e}")
            return None
    
    def _calculate_risk_reward_ratio(self, max_profit: Union[float, str], max_loss: Union[float, str]) -> Optional[float]:
        """Calculate risk-reward ratio"""
        try:
            if isinstance(max_profit, str) or isinstance(max_loss, str):
                return None
            
            if max_loss == 0:
                return float('inf') if max_profit > 0 else 0.0
            
            return round(abs(max_profit / max_loss), 2)
            
        except:
            return None
    
    def _generate_risk_warnings(self, inputs: StrategyInputs, option_legs: List[OptionLeg], stock_legs: List[StockLeg]) -> List[str]:
        """Generate strategy-specific risk warnings"""
        warnings = []
        
        # Check for unlimited risk
        has_short_options = any(leg.position == "short" for leg in option_legs)
        has_naked_shorts = any(leg.position == "short" for leg in option_legs) and not stock_legs
        
        if has_naked_shorts:
            warnings.append("Strategy contains naked short positions with potentially unlimited risk")
        
        # Check for high complexity
        if len(option_legs) > 2:
            warnings.append("Complex multi-leg strategy requires active management")
        
        # Check for time decay risk
        total_theta = sum(leg.theta for leg in option_legs)
        if total_theta < -10:
            warnings.append("High time decay risk - strategy loses value rapidly over time")
        
        # Check for volatility risk
        total_vega = sum(leg.vega for leg in option_legs)
        if abs(total_vega) > 50:
            warnings.append("High volatility sensitivity - strategy significantly affected by volatility changes")
        
        # Check for liquidity concerns
        if any(abs(leg.strike - inputs.spot_price) / inputs.spot_price > 0.2 for leg in option_legs):
            warnings.append("Strategy includes far out-of-the-money options which may have limited liquidity")
        
        # Check for assignment risk
        if any(leg.position == "short" and leg.option_type == "put" for leg in option_legs):
            warnings.append("Short put positions carry assignment risk if options finish in-the-money")
        
        return warnings
    
    def _generate_management_guidelines(self, strategy_type: StrategyType) -> List[str]:
        """Generate strategy-specific management guidelines"""
        guidelines = {
            StrategyType.LONG_CALL: [
                "Monitor time decay as expiration approaches",
                "Consider rolling up and out if profitable",
                "Have exit plan if stock moves against position"
            ],
            StrategyType.COVERED_CALL: [
                "Be prepared for assignment if call finishes in-the-money",
                "Roll call option if underlying rises near strike",
                "Consider closing early if 50% of premium captured"
            ],
            StrategyType.LONG_STRADDLE: [
                "Target 20-25% profit on total premium paid",
                "Close losing side and let winner run in trending market",
                "Manage before final week due to high gamma risk"
            ],
            StrategyType.IRON_CONDOR: [
                "Close early if 25-50% of maximum profit achieved",
                "Adjust untested side if breached",
                "Avoid holding through earnings or major events"
            ],
            StrategyType.BULL_CALL_SPREAD: [
                "Close early if maximum profit nearly achieved",
                "Consider rolling out if time decay accelerating",
                "Monitor pin risk near expiration if near strikes"
            ]
        }
        
        return guidelines.get(strategy_type, [
            "Monitor position regularly for changes in market conditions",
            "Have predefined profit targets and stop losses",
            "Consider closing early if management becomes difficult"
        ])
    
    def _determine_optimal_conditions(self, inputs: StrategyInputs, template: Dict[str, Any]) -> Dict[str, Any]:
        """Determine optimal market conditions for strategy"""
        return {
            "market_direction": template["market_bias"],
            "volatility_environment": template["volatility_bias"],
            "time_to_expiration": "30-60 days" if template.get("time_decay") == "positive" else "60+ days",
            "ideal_iv_rank": "high" if template["volatility_bias"] == "negative" else "low",
            "recommended_dte": 45 if template["complexity"] <= 2 else 30
        }
    
    def compare_strategies(self, strategies: List[StrategyInputs]) -> Dict[str, Any]:
        """Compare multiple strategies side by side"""
        try:
            results = []
            for strategy_input in strategies:
                result = self.analyze_strategy(strategy_input)
                results.append(result)
            
            comparison = {
                "strategies": results,
                "comparison_metrics": self._generate_comparison_metrics(results),
                "ranking": self._rank_strategies(results),
                "recommendations": self._generate_strategy_recommendations(results)
            }
            return comparison
        except Exception as e:
            logger.error(f"Strategy comparison failed: {e}")
            raise ValueError(f"Strategy comparison error: {str(e)}")
   
    def _generate_comparison_metrics(self, results: List[StrategyResult]) -> Dict[str, Any]:
        """Generate comparison metrics across strategies"""
        comparison = {
            "profit_potential": [],
            "risk_levels": [],
            "complexity_scores": [],
            "time_decay_impact": [],
            "volatility_sensitivity": [],
            "probability_of_profit": []
        }
        
        for result in results:
            # Profit potential (normalized)
            if isinstance(result.max_profit, str):
                profit_score = 5  # Unlimited = highest score
            else:
                profit_score = min(5, max(1, result.max_profit / 100))  # Scale to 1-5
            comparison["profit_potential"].append(profit_score)
            
            # Risk levels
            if isinstance(result.max_loss, str):
                risk_score = 5  # Unlimited = highest risk
            else:
                risk_score = min(5, max(1, abs(result.max_loss) / 100))
            comparison["risk_levels"].append(risk_score)
            
            # Other metrics
            comparison["complexity_scores"].append(result.complexity_rating)
            comparison["time_decay_impact"].append(abs(result.portfolio_greeks.get("theta", 0)))
            comparison["volatility_sensitivity"].append(abs(result.portfolio_greeks.get("vega", 0)))
            comparison["probability_of_profit"].append(result.profit_probability or 0.5)
        
        return comparison
    
    def _rank_strategies(self, results: List[StrategyResult]) -> List[Dict[str, Any]]:
        """Rank strategies based on multiple criteria"""
        rankings = []
        
        for i, result in enumerate(results):
            # Calculate composite score
            profit_score = 5 if isinstance(result.max_profit, str) else min(5, result.max_profit / 100)
            risk_score = 1 if isinstance(result.max_loss, str) else max(1, 5 - abs(result.max_loss) / 100)
            prob_score = (result.profit_probability or 0.5) * 5
            complexity_penalty = (6 - result.complexity_rating) / 5  # Lower complexity = higher score
            
            composite_score = (profit_score + risk_score + prob_score + complexity_penalty) / 4
            
            rankings.append({
                "strategy_name": result.strategy_name,
                "composite_score": round(composite_score, 2),
                "profit_potential": profit_score,
                "risk_adjusted_score": risk_score,
                "probability_score": prob_score,
                "complexity_score": complexity_penalty,
                "recommendation": self._get_strategy_recommendation(result)
            })
        
        # Sort by composite score
        rankings.sort(key=lambda x: x["composite_score"], reverse=True)
        
        return rankings
    
    def _get_strategy_recommendation(self, result: StrategyResult) -> str:
        """Generate recommendation for individual strategy"""
        if result.profit_probability and result.profit_probability > 0.7:
            return "Highly Recommended"
        elif result.profit_probability and result.profit_probability > 0.5:
            return "Recommended"
        elif result.complexity_rating <= 2 and not isinstance(result.max_loss, str):
            return "Suitable for Beginners"
        elif result.complexity_rating >= 4:
            return "Advanced Strategy"
        else:
            return "Moderate Risk"
    
    def _generate_strategy_recommendations(self, results: List[StrategyResult]) -> List[str]:
        """Generate overall recommendations for strategy selection"""
        recommendations = []
        
        # Find best risk-adjusted return
        best_ratio = 0
        best_strategy = None
        for result in results:
            if result.risk_reward_ratio and result.risk_reward_ratio > best_ratio:
                best_ratio = result.risk_reward_ratio
                best_strategy = result.strategy_name
        
        if best_strategy:
            recommendations.append(f"Best risk-adjusted return: {best_strategy}")
        
        # Find highest probability of profit
        best_prob = 0
        best_prob_strategy = None
        for result in results:
            if result.profit_probability and result.profit_probability > best_prob:
                best_prob = result.profit_probability
                best_prob_strategy = result.strategy_name
        
        if best_prob_strategy:
            recommendations.append(f"Highest probability of profit: {best_prob_strategy} ({best_prob:.1%})")
        
        # Find simplest strategy
        simplest = min(results, key=lambda x: x.complexity_rating)
        recommendations.append(f"Simplest to manage: {simplest.strategy_name}")
        
        # Market condition recommendations
        bullish_strategies = [r for r in results if "bullish" in r.market_bias.lower()]
        if bullish_strategies:
            best_bullish = max(bullish_strategies, key=lambda x: x.profit_probability or 0)
            recommendations.append(f"Best for bullish outlook: {best_bullish.strategy_name}")
        
        return recommendations
    
    def generate_strategy_report(self, result: StrategyResult) -> str:
        """Generate comprehensive strategy report"""
        report_sections = []
        
        # Header
        report_sections.append(f"📊 {result.strategy_name.upper()} STRATEGY ANALYSIS")
        report_sections.append("=" * 60)
        
        # Strategy Overview
        report_sections.append("\n🎯 STRATEGY OVERVIEW:")
        report_sections.append(f"• Strategy Type: {result.strategy_name}")
        report_sections.append(f"• Market Bias: {result.market_bias.replace('_', ' ').title()}")
        report_sections.append(f"• Volatility Bias: {result.volatility_bias.title()}")
        report_sections.append(f"• Complexity Rating: {result.complexity_rating}/5")
        report_sections.append(f"• Time Decay Impact: {result.time_decay_impact.title()}")
        
        # Position Details
        report_sections.append("\n📋 POSITION DETAILS:")
        for i, leg in enumerate(result.legs, 1):
            position_desc = f"{leg.position.title()} {leg.quantity} {leg.option_type.upper()}"
            report_sections.append(f"• Leg {i}: {position_desc} @ ${leg.strike:.2f} (Premium: ${leg.premium:.2f})")
        
        if result.stock_legs:
            for i, stock_leg in enumerate(result.stock_legs, 1):
                stock_desc = f"{stock_leg.position.title()} {stock_leg.quantity} shares"
                report_sections.append(f"• Stock Leg {i}: {stock_desc} @ ${stock_leg.price:.2f}")
        
        # Financial Metrics
        report_sections.append("\n💰 FINANCIAL METRICS:")
        report_sections.append(f"• Net Premium: ${result.net_premium:.2f}")
        report_sections.append(f"• Maximum Profit: {result.max_profit}")
        report_sections.append(f"• Maximum Loss: {result.max_loss}")
        report_sections.append(f"• Risk-Reward Ratio: {result.risk_reward_ratio or 'N/A'}")
        
        if result.breakeven_points:
            breakevens = ", ".join([f"${bp:.2f}" for bp in result.breakeven_points])
            report_sections.append(f"• Breakeven Points: {breakevens}")
        
        if result.profit_probability:
            report_sections.append(f"• Probability of Profit: {result.profit_probability:.1%}")
        
        # Greeks Analysis
        report_sections.append("\n📈 RISK PROFILE (GREEKS):")
        greeks = result.portfolio_greeks
        report_sections.append(f"• Delta: {greeks['delta']:.4f} (Directional Risk)")
        report_sections.append(f"• Gamma: {greeks['gamma']:.4f} (Delta Sensitivity)")
        report_sections.append(f"• Theta: {greeks['theta']:.4f} (Time Decay per Day)")
        report_sections.append(f"• Vega: {greeks['vega']:.4f} (Volatility Sensitivity)")
        
        # Optimal Conditions
        report_sections.append("\n🎯 OPTIMAL CONDITIONS:")
        conditions = result.optimal_conditions
        for key, value in conditions.items():
            report_sections.append(f"• {key.replace('_', ' ').title()}: {value}")
        
        # Risk Warnings
        if result.risk_warnings:
            report_sections.append("\n⚠️ RISK WARNINGS:")
            for warning in result.risk_warnings:
                report_sections.append(f"• {warning}")
        
        # Management Guidelines
        if result.management_guidelines:
            report_sections.append("\n📋 MANAGEMENT GUIDELINES:")
            for guideline in result.management_guidelines:
                report_sections.append(f"• {guideline}")
        
        # Footer
        report_sections.append("\n" + "=" * 60)
        report_sections.append("⚠️ This analysis is for educational purposes only.")
        report_sections.append("Past performance does not guarantee future results.")
        
        return "\n".join(report_sections)

    # Usage example and test    
if __name__ == "__main__":
    strategy_agent = StrategyAnalysisAgent()
    
    # Test Long Straddle
    print("Testing Strategy Analysis Agent")
    print("="*50)
    
    straddle_inputs = StrategyInputs(
        strategy_type=StrategyType.LONG_STRADDLE,
        spot_price=100.0,
        strikes=[100.0],  # Same strike for straddle
        expiries=[45/365],  # 45 days
        risk_free_rate=0.05,
        volatility=0.30,
        dividend_yield=0.02,
        market_outlook="volatile"
    )
    
    result = strategy_agent.analyze_strategy(straddle_inputs)
    
    # Display results
    print("LONG STRADDLE ANALYSIS:")
    print(f"Net Premium: ${result.net_premium:.2f}")
    print(f"Max Profit: {result.max_profit}")
    print(f"Max Loss: ${result.max_loss}")
    print(f"Breakevens: {[f'${bp:.2f}' for bp in result.breakeven_points]}")
    print(f"Profit Probability: {result.profit_probability:.1%}" if result.profit_probability else "N/A")
    print(f"Portfolio Delta: {result.portfolio_greeks['delta']:.4f}")
    print(f"Portfolio Theta: {result.portfolio_greeks['theta']:.4f}")
    
    # Generate full report
    print("\n" + "="*50)
    print("COMPREHENSIVE REPORT:")
    print("="*50)
    report = strategy_agent.generate_strategy_report(result)
    print(report)
    
    # Test strategy comparison
    print("\n" + "="*50)
    print("STRATEGY COMPARISON TEST:")
    print("="*50)
    
    comparison_strategies = [
        StrategyInputs(
            strategy_type=StrategyType.LONG_STRADDLE,
            spot_price=100.0,
            strikes=[100.0],
            expiries=[45/365],
            volatility=0.30
        ),
        StrategyInputs(
            strategy_type=StrategyType.LONG_STRANGLE,
            spot_price=100.0,
            strikes=[95.0, 105.0],
            expiries=[45/365, 45/365],
            volatility=0.30
        ),
        StrategyInputs(
            strategy_type=StrategyType.IRON_CONDOR,
            spot_price=100.0,
            strikes=[90.0, 95.0, 105.0, 110.0],
            expiries=[45/365] * 4,
            volatility=0.30
        )
    ]
    
    comparison = strategy_agent.compare_strategies(comparison_strategies)
    
    print("STRATEGY RANKINGS:")
    for i, ranking in enumerate(comparison["ranking"], 1):
        print(f"{i}. {ranking['strategy_name']}: Score {ranking['composite_score']:.2f} - {ranking['recommendation']}")
    
    print("\nRECOMMENDATIONS:")
    for rec in comparison["recommendations"]:
        print(f"• {rec}")