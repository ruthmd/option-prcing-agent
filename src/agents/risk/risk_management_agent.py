from typing import Dict, List, Any, Optional, Tuple, Union
from pydantic import BaseModel, Field
from enum import Enum
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from dataclasses import dataclass
from loguru import logger

from ..pricing.black_scholes import BlackScholesPricingAgent, BlackScholesInputs
from ..strategy.strategy_agent import StrategyAnalysisAgent, StrategyResult, OptionLeg, StockLeg
from ...config.settings import settings
from ...utils.constants import TRADING_DAYS_PER_YEAR

class RiskMetricType(Enum):
    """Types of risk metrics"""
    VAR = "value_at_risk"
    CVAR = "conditional_var"
    MAX_DRAWDOWN = "maximum_drawdown"
    PORTFOLIO_DELTA = "portfolio_delta"
    PORTFOLIO_GAMMA = "portfolio_gamma"
    PORTFOLIO_THETA = "portfolio_theta"
    PORTFOLIO_VEGA = "portfolio_vega"
    CORRELATION_RISK = "correlation_risk"
    LIQUIDITY_RISK = "liquidity_risk"

class HedgeType(Enum):
    """Types of hedging strategies"""
    DELTA_HEDGE = "delta_hedge"
    GAMMA_HEDGE = "gamma_hedge"
    VEGA_HEDGE = "vega_hedge"
    THETA_HEDGE = "theta_hedge"
    PORTFOLIO_HEDGE = "portfolio_hedge"
    PAIRS_HEDGE = "pairs_hedge"

@dataclass
class Position:
    """Individual position in portfolio"""
    symbol: str
    position_type: str  # 'stock', 'option', 'future'
    quantity: int
    current_price: float
    strike: Optional[float] = None
    expiry: Optional[float] = None
    option_type: Optional[str] = None
    implied_vol: Optional[float] = None
    delta: float = 0.0
    gamma: float = 0.0
    theta: float = 0.0
    vega: float = 0.0
    rho: float = 0.0

class PortfolioInputs(BaseModel):
    """Input parameters for portfolio risk analysis"""
    positions: List[Position]
    portfolio_value: float = Field(gt=0, description="Total portfolio value")
    risk_free_rate: float = Field(ge=-0.1, le=0.5, default=0.05)
    correlation_matrix: Optional[Dict[str, Dict[str, float]]] = None
    confidence_level: float = Field(ge=0.5, le=0.99, default=0.95)
    time_horizon_days: int = Field(ge=1, le=365, default=1)

class RiskMetrics(BaseModel):
    """Portfolio risk metrics"""
    var_1day: float = Field(description="1-day Value at Risk")
    var_10day: float = Field(description="10-day Value at Risk")
    cvar_1day: float = Field(description="1-day Conditional VaR")
    expected_shortfall: float = Field(description="Expected Shortfall")
    maximum_drawdown: float = Field(description="Maximum Drawdown estimate")
    portfolio_beta: Optional[float] = Field(description="Portfolio Beta")
    sharpe_ratio: Optional[float] = Field(description="Sharpe Ratio estimate")
    
    # Greeks-based risk
    total_delta: float = Field(description="Portfolio Delta")
    total_gamma: float = Field(description="Portfolio Gamma")
    total_theta: float = Field(description="Portfolio Theta")
    total_vega: float = Field(description="Portfolio Vega")
    
    # Concentration risk
    concentration_risk: Dict[str, float] = Field(description="Position concentration")
    correlation_risk: float = Field(description="Correlation risk measure")

class HedgeRecommendation(BaseModel):
    """Hedge recommendation"""
    hedge_type: HedgeType
    recommended_action: str
    hedge_instrument: str
    hedge_quantity: float
    expected_cost: float
    risk_reduction: float
    confidence: float = Field(ge=0.0, le=1.0)
    explanation: str

class PnLAnalysis(BaseModel):
    """P&L analysis result"""
    current_pnl: float
    unrealized_pnl: float
    realized_pnl: float
    daily_pnl: List[float]
    cumulative_pnl: List[float]
    best_day: float
    worst_day: float
    winning_days: int
    losing_days: int
    win_rate: float
    profit_factor: float
    max_consecutive_wins: int
    max_consecutive_losses: int

class RiskManagementResult(BaseModel):
    """Comprehensive risk management result"""
    portfolio_summary: Dict[str, Any]
    risk_metrics: RiskMetrics
    hedge_recommendations: List[HedgeRecommendation]
    pnl_analysis: Optional[PnLAnalysis]
    stress_test_results: Dict[str, float]
    risk_warnings: List[str]
    action_items: List[str]

class RiskManagementAgent:
    """Comprehensive risk management and hedging agent"""
    
    def __init__(self):
        self.name = "RiskManagement"
        self.bs_agent = BlackScholesPricingAgent()
        self.strategy_agent = StrategyAnalysisAgent()
        
        logger.info("Risk Management Agent initialized")
    
    def analyze_portfolio_risk(self, inputs: PortfolioInputs) -> RiskManagementResult:
        """Comprehensive portfolio risk analysis"""
        try:
            logger.info(f"Analyzing portfolio risk for {len(inputs.positions)} positions")
            
            # Calculate portfolio metrics
            portfolio_summary = self._calculate_portfolio_summary(inputs)
            
            # Calculate risk metrics
            risk_metrics = self._calculate_risk_metrics(inputs)
            
            # Generate hedge recommendations
            hedge_recommendations = self._generate_hedge_recommendations(inputs, risk_metrics)
            
            # Perform stress tests
            stress_results = self._perform_stress_tests(inputs)
            
            # Generate risk warnings and action items
            risk_warnings = self._generate_risk_warnings(inputs, risk_metrics)
            action_items = self._generate_action_items(inputs, risk_metrics)
            
            result = RiskManagementResult(
                portfolio_summary=portfolio_summary,
                risk_metrics=risk_metrics,
                hedge_recommendations=hedge_recommendations,
                pnl_analysis=None,  # Would be calculated with historical data
                stress_test_results=stress_results,
                risk_warnings=risk_warnings,
                action_items=action_items
            )
            
            logger.info("Portfolio risk analysis completed")
            return result
            
        except Exception as e:
            logger.error(f"Portfolio risk analysis failed: {e}")
            raise ValueError(f"Risk analysis error: {str(e)}")
    
    def _calculate_portfolio_summary(self, inputs: PortfolioInputs) -> Dict[str, Any]:
        """Calculate high-level portfolio summary"""
        positions = inputs.positions
        
        total_value = inputs.portfolio_value
        stock_value = sum(pos.quantity * pos.current_price for pos in positions if pos.position_type == 'stock')
        option_value = sum(pos.quantity * pos.current_price for pos in positions if pos.position_type == 'option')
        
        # Position counts
        long_positions = sum(1 for pos in positions if pos.quantity > 0)
        short_positions = sum(1 for pos in positions if pos.quantity < 0)
        
        # Unique symbols
        unique_symbols = len(set(pos.symbol for pos in positions))
        
        # Expiration analysis for options
        option_positions = [pos for pos in positions if pos.position_type == 'option' and pos.expiry]
        
        if option_positions:
            avg_dte = np.mean([pos.expiry * 365 for pos in option_positions])
            min_dte = min(pos.expiry * 365 for pos in option_positions)
        else:
            avg_dte = 0
            min_dte = 0
        
        return {
            "total_portfolio_value": total_value,
            "stock_allocation": stock_value / total_value if total_value > 0 else 0,
            "option_allocation": option_value / total_value if total_value > 0 else 0,
            "number_of_positions": len(positions),
            "long_positions": long_positions,
            "short_positions": short_positions,
            "unique_symbols": unique_symbols,
            "average_dte": avg_dte,
            "minimum_dte": min_dte,
            "concentration_top3": self._calculate_top_concentration(positions, 3)
        }
    
    def _calculate_top_concentration(self, positions: List[Position], top_n: int) -> float:
        """Calculate concentration in top N positions"""
        position_values = {}
        
        for pos in positions:
            symbol = pos.symbol
            value = abs(pos.quantity * pos.current_price)
            position_values[symbol] = position_values.get(symbol, 0) + value
        
        sorted_values = sorted(position_values.values(), reverse=True)
        top_values = sorted_values[:top_n]
        total_value = sum(position_values.values())
        
        return sum(top_values) / total_value if total_value > 0 else 0
    
    def _calculate_risk_metrics(self, inputs: PortfolioInputs) -> RiskMetrics:
        """Calculate comprehensive risk metrics"""
        positions = inputs.positions
        
        # Aggregate Greeks
        total_delta = sum(pos.delta * pos.quantity for pos in positions)
        total_gamma = sum(pos.gamma * pos.quantity for pos in positions)
        total_theta = sum(pos.theta * pos.quantity for pos in positions)
        total_vega = sum(pos.vega * pos.quantity for pos in positions)
        
        # Calculate VaR using parametric method (simplified)
        portfolio_volatility = self._estimate_portfolio_volatility(positions)
        portfolio_value = inputs.portfolio_value
        confidence_level = inputs.confidence_level
        
        # Z-score for confidence level
        from scipy.stats import norm
        z_score = norm.ppf(confidence_level)
        
        # 1-day VaR
        var_1day = portfolio_value * portfolio_volatility * z_score / np.sqrt(TRADING_DAYS_PER_YEAR)
        
        # 10-day VaR (square root of time scaling)
        var_10day = var_1day * np.sqrt(10)
        
        # Conditional VaR (Expected Shortfall)
        cvar_1day = var_1day * norm.pdf(z_score) / (1 - confidence_level)
        
        # Expected Shortfall
        expected_shortfall = cvar_1day
        
        # Maximum Drawdown estimate (based on volatility)
        max_drawdown_estimate = portfolio_volatility * np.sqrt(2 * np.log(TRADING_DAYS_PER_YEAR))
        
        # Concentration risk
        concentration_risk = {}
        symbol_exposure = {}
        
        for pos in positions:
            symbol = pos.symbol
            exposure = abs(pos.quantity * pos.current_price)
            symbol_exposure[symbol] = symbol_exposure.get(symbol, 0) + exposure
        
        total_exposure = sum(symbol_exposure.values())
        for symbol, exposure in symbol_exposure.items():
            concentration_risk[symbol] = exposure / total_exposure if total_exposure > 0 else 0
        
        # Correlation risk (simplified)
        correlation_risk = self._calculate_correlation_risk(positions, inputs.correlation_matrix)
        
        return RiskMetrics(
            var_1day=var_1day,
            var_10day=var_10day,
            cvar_1day=cvar_1day,
            expected_shortfall=expected_shortfall,
            maximum_drawdown=max_drawdown_estimate,
            portfolio_beta=None,  # Would need market data
            sharpe_ratio=None,    # Would need return history
            total_delta=total_delta,
            total_gamma=total_gamma,
            total_theta=total_theta,
            total_vega=total_vega,
            concentration_risk=concentration_risk,
            correlation_risk=correlation_risk
        )
    
    def _estimate_portfolio_volatility(self, positions: List[Position]) -> float:
        """Estimate portfolio volatility"""
        # Simplified calculation - in practice would use covariance matrix
        weighted_vols = []
        total_value = sum(abs(pos.quantity * pos.current_price) for pos in positions)
        
        if total_value == 0:
            return 0.25  # Default 25% volatility
        
        for pos in positions:
            weight = abs(pos.quantity * pos.current_price) / total_value
            
            if pos.position_type == 'stock':
                # Assume 25% volatility for stocks
                vol = 0.25
            elif pos.position_type == 'option' and pos.implied_vol:
                # Use implied volatility
                vol = pos.implied_vol
            else:
                vol = 0.30  # Default option volatility
            
            weighted_vols.append(weight * vol)
        
        return sum(weighted_vols)
    
    def _calculate_correlation_risk(self, positions: List[Position], correlation_matrix: Optional[Dict[str, Dict[str, float]]]) -> float:
        """Calculate correlation risk measure"""
        if not correlation_matrix:
            # Without correlation data, assume moderate correlation
            unique_symbols = set(pos.symbol for pos in positions)
            if len(unique_symbols) <= 1:
                return 1.0  # Maximum correlation risk
            else:
                return 0.5  # Moderate correlation assumption
        
        # Calculate diversification ratio using correlation matrix
        # This is a simplified implementation
        symbols = list(set(pos.symbol for pos in positions))
        
        if len(symbols) <= 1:
            return 1.0
        
        total_correlations = 0
        correlation_count = 0
        
        for i, symbol1 in enumerate(symbols):
            for j, symbol2 in enumerate(symbols):
                if i != j and symbol1 in correlation_matrix and symbol2 in correlation_matrix[symbol1]:
                    total_correlations += abs(correlation_matrix[symbol1][symbol2])
                    correlation_count += 1
        
        if correlation_count == 0:
            return 0.5
        
        avg_correlation = total_correlations / correlation_count
        return avg_correlation
    
    def _generate_hedge_recommendations(self, inputs: PortfolioInputs, risk_metrics: RiskMetrics) -> List[HedgeRecommendation]:
        """Generate hedge recommendations based on risk analysis"""
        recommendations = []
        
        # Delta hedge recommendation
        if abs(risk_metrics.total_delta) > 100:  # Significant directional exposure
            delta_hedge = self._recommend_delta_hedge(risk_metrics.total_delta, inputs.portfolio_value)
            recommendations.append(delta_hedge)
        
        # Gamma hedge recommendation
        if abs(risk_metrics.total_gamma) > 10:  # Significant gamma exposure
            gamma_hedge = self._recommend_gamma_hedge(risk_metrics.total_gamma, inputs)
            recommendations.append(gamma_hedge)
        
        # Vega hedge recommendation
        if abs(risk_metrics.total_vega) > 500:  # Significant volatility exposure
            vega_hedge = self._recommend_vega_hedge(risk_metrics.total_vega, inputs)
            recommendations.append(vega_hedge)
        
        # Theta hedge recommendation
        if risk_metrics.total_theta < -100:  # Significant time decay
            theta_hedge = self._recommend_theta_hedge(risk_metrics.total_theta, inputs)
            recommendations.append(theta_hedge)
        
        # Concentration hedge
        max_concentration = max(risk_metrics.concentration_risk.values()) if risk_metrics.concentration_risk else 0
        if max_concentration > 0.3:  # More than 30% in single name
            concentration_hedge = self._recommend_concentration_hedge(risk_metrics.concentration_risk, inputs)
            recommendations.append(concentration_hedge)
        
        return recommendations
    
    def _recommend_delta_hedge(self, portfolio_delta: float, portfolio_value: float) -> HedgeRecommendation:
        """Recommend delta hedge"""
        hedge_quantity = -portfolio_delta  # Opposite delta to neutralize
        
        if abs(portfolio_delta) > 1000:
            hedge_instrument = "Index Futures (SPY)"
            cost_estimate = abs(hedge_quantity) * 0.1  # Rough futures cost
        else:
            hedge_instrument = "Index ETF (SPY)"
            cost_estimate = abs(hedge_quantity) * 400 * 0.001  # Rough ETF cost
        
        return HedgeRecommendation(
            hedge_type=HedgeType.DELTA_HEDGE,
            recommended_action=f"{'Sell' if hedge_quantity < 0 else 'Buy'} {abs(hedge_quantity):.0f} delta equivalent",
            hedge_instrument=hedge_instrument,
            hedge_quantity=hedge_quantity,
            expected_cost=cost_estimate,
            risk_reduction=0.8,  # Estimated 80% delta risk reduction
            confidence=0.9,
            explanation=f"Portfolio has {portfolio_delta:.0f} delta exposure. Hedge with {hedge_instrument} to achieve delta neutrality."
        )
    
    def _recommend_gamma_hedge(self, portfolio_gamma: float, inputs: PortfolioInputs) -> HedgeRecommendation:
        """Recommend gamma hedge"""
        hedge_quantity = -portfolio_gamma * 0.5  # Partial gamma hedge
        
        return HedgeRecommendation(
            hedge_type=HedgeType.GAMMA_HEDGE,
            recommended_action=f"{'Sell' if portfolio_gamma > 0 else 'Buy'} options to reduce gamma",
            hedge_instrument="ATM Options (30-45 DTE)",
            hedge_quantity=hedge_quantity,
            expected_cost=abs(hedge_quantity) * 2.0,  # Rough option cost
            risk_reduction=0.6,
            confidence=0.7,
            explanation=f"Portfolio gamma of {portfolio_gamma:.2f} creates convexity risk. Consider gamma hedge with ATM options."
        )
    
    def _recommend_vega_hedge(self, portfolio_vega: float, inputs: PortfolioInputs) -> HedgeRecommendation:
        """Recommend vega hedge"""
        hedge_quantity = -portfolio_vega * 0.7  # Partial vega hedge
        return HedgeRecommendation(
           hedge_type=HedgeType.VEGA_HEDGE,
           recommended_action=f"{'Sell' if portfolio_vega > 0 else 'Buy'} volatility exposure",
           hedge_instrument="VIX Options or Long-dated Options",
           hedge_quantity=hedge_quantity,
           expected_cost=abs(hedge_quantity) * 0.05,  # Rough volatility hedge cost
           risk_reduction=0.7,
           confidence=0.8,
           explanation=f"Portfolio vega of {portfolio_vega:.0f} creates volatility risk. Consider VIX hedge or opposite vega position."
       )
   
    def _recommend_theta_hedge(self, portfolio_theta: float, inputs: PortfolioInputs) -> HedgeRecommendation:
        """Recommend theta hedge"""
        hedge_quantity = -portfolio_theta * 0.5  # Partial theta hedge
        
        return HedgeRecommendation(
            hedge_type=HedgeType.THETA_HEDGE,
            recommended_action="Sell options to generate positive theta",
            hedge_instrument="Short-dated OTM Options",
            hedge_quantity=hedge_quantity,
            expected_cost=-abs(hedge_quantity) * 1.0,  # Negative cost = premium received
            risk_reduction=0.5,
            confidence=0.6,
            explanation=f"Portfolio theta of {portfolio_theta:.2f} creates time decay risk. Consider selling options to generate positive theta."
        )
    
    def _recommend_concentration_hedge(self, concentration_risk: Dict[str, float], inputs: PortfolioInputs) -> HedgeRecommendation:
        """Recommend concentration hedge"""
        max_symbol = max(concentration_risk.keys(), key=lambda k: concentration_risk[k])
        max_concentration = concentration_risk[max_symbol]
        
        return HedgeRecommendation(
            hedge_type=HedgeType.PORTFOLIO_HEDGE,
            recommended_action=f"Reduce {max_symbol} exposure or add hedging",
            hedge_instrument=f"Protective Puts on {max_symbol} or Index Hedge",
            hedge_quantity=max_concentration * inputs.portfolio_value * 0.3,  # Hedge 30% of concentration
            expected_cost=max_concentration * inputs.portfolio_value * 0.02,  # 2% hedge cost
            risk_reduction=0.4,
            confidence=0.8,
            explanation=f"{max_symbol} represents {max_concentration:.1%} of portfolio. Consider protective puts or diversification."
        )
    
    def _perform_stress_tests(self, inputs: PortfolioInputs) -> Dict[str, float]:
        """Perform various stress tests on the portfolio"""
        stress_results = {}
        
        positions = inputs.positions
        base_value = inputs.portfolio_value
        
        # Market crash scenario (-20% market move)
        crash_pnl = 0
        for pos in positions:
            if pos.position_type == 'stock':
                crash_pnl += pos.quantity * pos.current_price * -0.20
            elif pos.position_type == 'option':
                # Simplified: use delta approximation
                crash_pnl += pos.quantity * pos.delta * pos.current_price * -0.20
        
        stress_results["market_crash_20pct"] = crash_pnl
        
        # Volatility spike scenario (+50% implied vol)
        vol_spike_pnl = sum(pos.quantity * pos.vega * 0.50 for pos in positions if pos.position_type == 'option')
        stress_results["volatility_spike_50pct"] = vol_spike_pnl
        
        # Time decay scenario (10 days pass)
        time_decay_pnl = sum(pos.quantity * pos.theta * 10 for pos in positions if pos.position_type == 'option')
        stress_results["time_decay_10days"] = time_decay_pnl
        
        # Interest rate shock (+200 basis points)
        rate_shock_pnl = sum(pos.quantity * pos.rho * 0.02 for pos in positions if pos.position_type == 'option')
        stress_results["interest_rate_shock_200bp"] = rate_shock_pnl
        
        # Gap risk scenario (-10% overnight gap)
        gap_pnl = 0
        for pos in positions:
            if pos.position_type == 'stock':
                gap_pnl += pos.quantity * pos.current_price * -0.10
            elif pos.position_type == 'option':
                # Options may not move as much in gaps due to volatility crush
                gap_pnl += pos.quantity * pos.delta * pos.current_price * -0.10 * 0.7  # 70% of delta move
        
        stress_results["overnight_gap_10pct"] = gap_pnl
        
        return stress_results
    
    def _generate_risk_warnings(self, inputs: PortfolioInputs, risk_metrics: RiskMetrics) -> List[str]:
        """Generate risk warnings based on analysis"""
        warnings = []
        
        # High VaR warning
        if risk_metrics.var_1day > inputs.portfolio_value * 0.05:  # More than 5% daily VaR
            warnings.append(f"High daily VaR of ${risk_metrics.var_1day:,.0f} ({risk_metrics.var_1day/inputs.portfolio_value:.1%} of portfolio)")
        
        # Concentration warning
        max_concentration = max(risk_metrics.concentration_risk.values()) if risk_metrics.concentration_risk else 0
        if max_concentration > 0.25:
            symbol = max(risk_metrics.concentration_risk.keys(), key=lambda k: risk_metrics.concentration_risk[k])
            warnings.append(f"High concentration risk: {max_concentration:.1%} in {symbol}")
        
        # Greek warnings
        if abs(risk_metrics.total_delta) > 500:
            warnings.append(f"High directional risk: Portfolio delta of {risk_metrics.total_delta:.0f}")
        
        if abs(risk_metrics.total_gamma) > 20:
            warnings.append(f"High convexity risk: Portfolio gamma of {risk_metrics.total_gamma:.1f}")
        
        if risk_metrics.total_theta < -50:
            warnings.append(f"High time decay: Portfolio loses ${abs(risk_metrics.total_theta):.0f} per day to theta")
        
        if abs(risk_metrics.total_vega) > 1000:
            warnings.append(f"High volatility sensitivity: Portfolio vega of {risk_metrics.total_vega:.0f}")
        
        # Short-term expiration warning
        option_positions = [pos for pos in inputs.positions if pos.position_type == 'option' and pos.expiry]
        if option_positions:
            min_dte = min(pos.expiry * 365 for pos in option_positions)
            if min_dte < 7:
                warnings.append(f"Options expiring in {min_dte:.0f} days - high gamma and theta risk")
        
        # Correlation warning
        if risk_metrics.correlation_risk > 0.8:
            warnings.append("High correlation risk - portfolio not well diversified")
        
        return warnings
    
    def _generate_action_items(self, inputs: PortfolioInputs, risk_metrics: RiskMetrics) -> List[str]:
        """Generate actionable recommendations"""
        actions = []
        
        # Risk monitoring
        actions.append("Monitor portfolio Greeks daily, especially as options approach expiration")
        
        # Position sizing
        max_concentration = max(risk_metrics.concentration_risk.values()) if risk_metrics.concentration_risk else 0
        if max_concentration > 0.2:
            actions.append("Consider reducing position size in concentrated holdings")
        
        # Hedging actions
        if abs(risk_metrics.total_delta) > 200:
            actions.append("Consider delta hedging to reduce directional risk")
        
        if risk_metrics.total_theta < -30:
            actions.append("Review time decay exposure - consider closing or rolling positions")
        
        # Diversification
        if len(set(pos.symbol for pos in inputs.positions)) < 5:
            actions.append("Consider diversifying across more underlying symbols")
        
        # Expiration management
        option_positions = [pos for pos in inputs.positions if pos.position_type == 'option' and pos.expiry]
        if option_positions:
            min_dte = min(pos.expiry * 365 for pos in option_positions)
            if min_dte < 21:
                actions.append("Review options expiring within 21 days for early closure or rolling")
        
        # Volatility management
        if abs(risk_metrics.total_vega) > 500:
            actions.append("Monitor implied volatility levels and consider vega hedging")
        
        return actions
    
    def calculate_optimal_position_size(
        self, 
        new_position: Position, 
        current_portfolio: PortfolioInputs,
        risk_budget: float = 0.02  # 2% of portfolio risk budget
    ) -> Dict[str, Any]:
        """Calculate optimal position size based on risk budget"""
        try:
            # Calculate marginal risk contribution of new position
            position_value = abs(new_position.quantity * new_position.current_price)
            portfolio_value = current_portfolio.portfolio_value
            
            # Simple position sizing based on volatility
            if new_position.position_type == 'stock':
                estimated_vol = 0.25  # Assume 25% volatility
            elif new_position.implied_vol:
                estimated_vol = new_position.implied_vol
            else:
                estimated_vol = 0.30
            
            # Kelly criterion approximation
            win_rate = 0.55  # Assume slight edge
            avg_win_loss_ratio = 1.2  # Assume 1.2:1 win/loss ratio
            
            kelly_fraction = (win_rate * avg_win_loss_ratio - (1 - win_rate)) / avg_win_loss_ratio
            kelly_fraction = max(0, min(kelly_fraction, 0.25))  # Cap at 25%
            
            # Risk-based position sizing
            daily_var_target = portfolio_value * risk_budget
            position_daily_var = position_value * estimated_vol / np.sqrt(TRADING_DAYS_PER_YEAR)
            
            risk_based_fraction = daily_var_target / position_daily_var if position_daily_var > 0 else 0
            
            # Take the more conservative approach
            recommended_fraction = min(kelly_fraction, risk_based_fraction, 0.1)  # Max 10% per position
            recommended_value = portfolio_value * recommended_fraction
            recommended_quantity = int(recommended_value / new_position.current_price)
            
            return {
                "recommended_quantity": recommended_quantity,
                "recommended_value": recommended_value,
                "portfolio_fraction": recommended_fraction,
                "kelly_fraction": kelly_fraction,
                "risk_based_fraction": risk_based_fraction,
                "estimated_daily_var": recommended_value * estimated_vol / np.sqrt(TRADING_DAYS_PER_YEAR),
                "rationale": f"Position sized for {risk_budget:.1%} portfolio risk budget"
            }
            
        except Exception as e:
            logger.error(f"Position sizing calculation failed: {e}")
            return {
                "recommended_quantity": 0,
                "error": str(e)
            }
    
    def analyze_pnl_attribution(
        self, 
        portfolio: PortfolioInputs,
        price_changes: Dict[str, float],
        vol_changes: Dict[str, float] = None,
        time_decay_days: float = 1.0
    ) -> Dict[str, Any]:
        """Analyze P&L attribution to different risk factors"""
        try:
            attribution = {
                "total_pnl": 0.0,
                "delta_pnl": 0.0,
                "gamma_pnl": 0.0,
                "theta_pnl": 0.0,
                "vega_pnl": 0.0,
                "rho_pnl": 0.0,
                "unexplained_pnl": 0.0,
                "position_breakdown": {}
            }
            
            for pos in portfolio.positions:
                symbol = pos.symbol
                price_change = price_changes.get(symbol, 0.0)
                vol_change = vol_changes.get(symbol, 0.0) if vol_changes else 0.0
                
                # Delta P&L
                delta_pnl = pos.delta * pos.quantity * price_change
                
                # Gamma P&L (second-order effect)
                gamma_pnl = 0.5 * pos.gamma * pos.quantity * (price_change ** 2)
                
                # Theta P&L
                theta_pnl = pos.theta * pos.quantity * time_decay_days
                
                # Vega P&L
                vega_pnl = pos.vega * pos.quantity * vol_change
                
                # Rho P&L (assuming no rate change for now)
                rho_pnl = 0.0
                
                position_pnl = delta_pnl + gamma_pnl + theta_pnl + vega_pnl + rho_pnl
                
                attribution["delta_pnl"] += delta_pnl
                attribution["gamma_pnl"] += gamma_pnl
                attribution["theta_pnl"] += theta_pnl
                attribution["vega_pnl"] += vega_pnl
                attribution["rho_pnl"] += rho_pnl
                attribution["total_pnl"] += position_pnl
                
                attribution["position_breakdown"][f"{symbol}_{pos.position_type}"] = {
                    "delta_pnl": delta_pnl,
                    "gamma_pnl": gamma_pnl,
                    "theta_pnl": theta_pnl,
                    "vega_pnl": vega_pnl,
                    "total_pnl": position_pnl
                }
            
            # Round values
            for key in attribution:
                if isinstance(attribution[key], float):
                    attribution[key] = round(attribution[key], 2)
            
            return attribution
            
        except Exception as e:
            logger.error(f"P&L attribution analysis failed: {e}")
            return {"error": str(e)}
    
    def generate_risk_report(self, result: RiskManagementResult) -> str:
        """Generate comprehensive risk management report"""
        report_sections = []
        
        # Header
        report_sections.append("📊 PORTFOLIO RISK MANAGEMENT REPORT")
        report_sections.append("=" * 60)
        
        # Portfolio Summary
        summary = result.portfolio_summary
        report_sections.append("\n📋 PORTFOLIO SUMMARY:")
        report_sections.append(f"• Total Portfolio Value: ${summary['total_portfolio_value']:,.0f}")
        report_sections.append(f"• Number of Positions: {summary['number_of_positions']}")
        report_sections.append(f"• Stock Allocation: {summary['stock_allocation']:.1%}")
        report_sections.append(f"• Option Allocation: {summary['option_allocation']:.1%}")
        report_sections.append(f"• Unique Symbols: {summary['unique_symbols']}")
        report_sections.append(f"• Top 3 Concentration: {summary['concentration_top3']:.1%}")
        
        # Risk Metrics
        risk = result.risk_metrics
        report_sections.append("\n⚠️ RISK METRICS:")
        report_sections.append(f"• 1-Day Value at Risk: ${risk.var_1day:,.0f}")
        report_sections.append(f"• 10-Day Value at Risk: ${risk.var_10day:,.0f}")
        report_sections.append(f"• Expected Shortfall: ${risk.expected_shortfall:,.0f}")
        report_sections.append(f"• Portfolio Delta: {risk.total_delta:,.0f}")
        report_sections.append(f"• Portfolio Gamma: {risk.total_gamma:,.2f}")
        report_sections.append(f"• Portfolio Theta: ${risk.total_theta:,.0f}/day")
        report_sections.append(f"• Portfolio Vega: {risk.total_vega:,.0f}")
        
        # Concentration Risk
        if risk.concentration_risk:
            report_sections.append("\n🎯 CONCENTRATION ANALYSIS:")
            sorted_concentration = sorted(risk.concentration_risk.items(), key=lambda x: x[1], reverse=True)
            for symbol, concentration in sorted_concentration[:5]:
                report_sections.append(f"• {symbol}: {concentration:.1%}")
        
        # Stress Test Results
        report_sections.append("\n💥 STRESS TEST RESULTS:")
        for scenario, pnl in result.stress_test_results.items():
            scenario_name = scenario.replace("_", " ").title()
            report_sections.append(f"• {scenario_name}: ${pnl:,.0f}")
        
        # Hedge Recommendations
        if result.hedge_recommendations:
            report_sections.append("\n🛡️ HEDGE RECOMMENDATIONS:")
            for i, hedge in enumerate(result.hedge_recommendations, 1):
                report_sections.append(f"• {i}. {hedge.recommended_action}")
                report_sections.append(f"   Instrument: {hedge.hedge_instrument}")
                report_sections.append(f"   Expected Cost: ${hedge.expected_cost:,.0f}")
                report_sections.append(f"   Risk Reduction: {hedge.risk_reduction:.1%}")
        
        # Risk Warnings
        if result.risk_warnings:
            report_sections.append("\n⚠️ RISK WARNINGS:")
            for warning in result.risk_warnings:
                report_sections.append(f"• {warning}")
        
        # Action Items
        if result.action_items:
            report_sections.append("\n✅ ACTION ITEMS:")
            for action in result.action_items:
                report_sections.append(f"• {action}")
        
        # Footer
        report_sections.append("\n" + "=" * 60)
        report_sections.append("⚠️ This analysis is for risk management purposes only.")
        report_sections.append("Consult with risk management professionals for complex portfolios.")
        
        return "\n".join(report_sections)

    # Usage example and test
if __name__ == "__main__":
    risk_agent = RiskManagementAgent()
    
    # Create sample portfolio
    sample_positions = [
        Position(
            symbol="AAPL",
            position_type="stock",
            quantity=100,
            current_price=150.0,
            delta=100.0  # 100 shares = 100 delta
        ),
        Position(
            symbol="AAPL",
            position_type="option",
            quantity=10,  # 10 contracts
            current_price=5.0,
            strike=155.0,
            expiry=45/365,
            option_type="call",
            implied_vol=0.25,
            delta=6.0,   # 0.6 delta per contract
            gamma=0.8,   # 0.08 gamma per contract
            theta=-2.0,  # -0.2 theta per contract
            vega=15.0    # 1.5 vega per contract
        ),
        Position(
            symbol="GOOGL",
            position_type="option",
            quantity=-5,  # Short 5 contracts
            current_price=8.0,
            strike=2800.0,
            expiry=30/365,
            option_type="put",
            implied_vol=0.30,
            delta=15.0,   # -3.0 delta per contract (short position makes it positive)
            gamma=-0.4,   # Short gamma
            theta=10.0,   # Positive theta from short position
            vega=-20.0    # Short vega
        )
    ]
    
    portfolio_inputs = PortfolioInputs(
        positions=sample_positions,
        portfolio_value=50000.0,
        risk_free_rate=0.05,
        confidence_level=0.95
    )
    
    print("Testing Risk Management Agent")
    print("="*50)
    
    # Analyze portfolio risk
    result = risk_agent.analyze_portfolio_risk(portfolio_inputs)
    
    # Display key metrics
    print("PORTFOLIO RISK ANALYSIS:")
    print(f"Portfolio Value: ${result.portfolio_summary['total_portfolio_value']:,.0f}")
    print(f"1-Day VaR: ${result.risk_metrics.var_1day:,.0f}")
    print(f"Portfolio Delta: {result.risk_metrics.total_delta:,.0f}")
    print(f"Portfolio Theta: ${result.risk_metrics.total_theta:,.0f}/day")
    
    print(f"\nHEDGE RECOMMENDATIONS: {len(result.hedge_recommendations)}")
    for hedge in result.hedge_recommendations:
        print(f"• {hedge.recommended_action}")
        print(f"  Cost: ${hedge.expected_cost:,.0f}, Risk Reduction: {hedge.risk_reduction:.1%}")
    
    print(f"\nRISK WARNINGS: {len(result.risk_warnings)}")
    for warning in result.risk_warnings:
        print(f"• {warning}")
    
    # Generate full report
    print("\n" + "="*60)
    print("COMPREHENSIVE RISK REPORT:")
    print("="*60)
    report = risk_agent.generate_risk_report(result)
    print(report)
    
    # Test position sizing
    print("\n" + "="*50)
    print("POSITION SIZING TEST:")
    print("="*50)
    
    new_position = Position(
        symbol="MSFT",
        position_type="stock",
        quantity=0,  # To be determined
        current_price=300.0
    )
    
    sizing = risk_agent.calculate_optimal_position_size(new_position, portfolio_inputs)
    print(f"Recommended Quantity: {sizing['recommended_quantity']}")
    print(f"Recommended Value: ${sizing['recommended_value']:,.0f}")
    print(f"Portfolio Fraction: {sizing['portfolio_fraction']:.2%}")
    print(f"Rationale: {sizing['rationale']}")
    
    # Test P&L attribution
    print("\n" + "="*50)
    print("P&L ATTRIBUTION TEST:")
    print("="*50)
    
    price_changes = {"AAPL": 5.0, "GOOGL": -50.0}  # AAPL up $5, GOOGL down $50
    vol_changes = {"AAPL": 0.05, "GOOGL": -0.02}   # AAPL vol up 5%, GOOGL vol down 2%
    
    pnl_attribution = risk_agent.analyze_pnl_attribution(
        portfolio_inputs, 
        price_changes, 
        vol_changes, 
        time_decay_days=1.0
    )
    
    print(f"Total P&L: ${pnl_attribution['total_pnl']:,.2f}")
    print(f"Delta P&L: ${pnl_attribution['delta_pnl']:,.2f}")
    print(f"Gamma P&L: ${pnl_attribution['gamma_pnl']:,.2f}")
    print(f"Theta P&L: ${pnl_attribution['theta_pnl']:,.2f}")
    print(f"Vega P&L: ${pnl_attribution['vega_pnl']:,.2f}")
    
    print("\nPosition Breakdown:")
    for position, breakdown in pnl_attribution['position_breakdown'].items():
        print(f"• {position}: ${breakdown['total_pnl']:,.2f}")