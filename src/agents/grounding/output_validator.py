from typing import Dict, List, Any, Optional, Union
from pydantic import BaseModel, Field
import numpy as np
from datetime import datetime
from loguru import logger

from ...knowledge.options_theory import get_all_risk_warnings
from ...utils.constants import PRICING_BOUNDS, ERROR_MESSAGES 
from ...config.settings import settings

class OutputValidationResult(BaseModel):
    """Result of output validation"""
    is_valid: bool
    confidence: float = Field(ge=0.0, le=1.0)
    warnings: List[str] = []
    errors: List[str] = []
    recommendations: List[str] = []
    risk_level: str = Field(default="medium")  # low, medium, high
    uncertainty_bounds: Optional[Dict[str, float]] = None

class OutputValidator:
    def __init__(self):
        self.validation_rules = self._initialize_validation_rules()
    
    def _initialize_validation_rules(self) -> Dict[str, Any]:
        """Initialize validation rules for different output types"""
        return {
            "option_price": {
                "min_value": 0.0,
                "max_reasonable_multiple": 2.0,  # 2x underlying price
                "intrinsic_value_check": True,
                "time_value_check": True
            },
            "greeks": {
                "delta": {"call_range": (0, 1), "put_range": (-1, 0)},
                "gamma": {"min": 0, "max": np.inf, "warning_threshold": 0.1},
                "theta": {"max": 0, "warning_threshold": -1.0},
                "vega": {"min": 0, "warning_threshold": 100},
                "rho": {"reasonable_range": (-100, 100)}
            },
            "volatility": {
                "min": PRICING_BOUNDS["min_volatility"],
                "max": PRICING_BOUNDS["max_volatility"], 
                "typical_range": (0.1, 1.0),
                "warning_threshold": 2.0
            },
            "strategy_payoff": {
                "finite_loss": True,
                "reasonable_breakeven": True,
                "risk_reward_ratio": True
            }
        }
    
    def validate_output(
        self, 
        output_type: str, 
        result: Any, 
        context: Dict[str, Any] = None
    ) -> OutputValidationResult:
        """Main output validation entry point"""
        
        logger.info(f"Validating {output_type} output...")
        
        validation_result = OutputValidationResult(is_valid=True, confidence=1.0)
        
        try:
            # Route to specific validation based on output type
            if output_type == "option_price":
                self._validate_option_price(result, context or {}, validation_result)
            elif output_type == "greeks":
                self._validate_greeks(result, context or {}, validation_result)
            elif output_type == "volatility":
                self._validate_volatility(result, context or {}, validation_result)
            elif output_type == "strategy_analysis":
                self._validate_strategy_analysis(result, context or {}, validation_result)
            elif output_type == "arbitrage_opportunity":
                self._validate_arbitrage_opportunity(result, context or {}, validation_result)
            else:
                # Generic validation
                self._validate_generic_output(result, validation_result)
            
            # Assess overall risk level
            validation_result.risk_level = self._assess_risk_level(validation_result)
            
            # Add uncertainty bounds if applicable
            if isinstance(result, (int, float)):
                validation_result.uncertainty_bounds = self._calculate_uncertainty_bounds(
                    result, validation_result.confidence
                )
            
            # Add standard risk warnings
            validation_result.warnings.extend(self._get_contextual_risk_warnings(output_type))
            
        except Exception as e:
            logger.error(f"Output validation failed: {e}")
            validation_result.is_valid = False
            validation_result.confidence = 0.0
            validation_result.errors.append(f"Validation error: {str(e)}")
        
        return validation_result
    
    def _validate_option_price(
        self, 
        price: float, 
        context: Dict[str, Any], 
        result: OutputValidationResult
    ):
        """Validate option price output"""
        
        # Basic bounds check
        if price < 0:
            result.is_valid = False
            result.errors.append("Option price cannot be negative")
            return
        
        if price == 0:
            result.warnings.append("Option price is zero - option may be deeply out-of-the-money")
        
        # Context-dependent validation
        if "spot_price" in context and "strike" in context:
            spot = context["spot_price"]
            strike = context["strike"]
            option_type = context.get("option_type", "call").lower()
            
            # Intrinsic value check
            if option_type == "call":
                intrinsic = max(0, spot - strike)
            else:
                intrinsic = max(0, strike - spot)
            
            if price < intrinsic * 0.98:  # Small tolerance
                result.is_valid = False
                result.errors.append(
                    f"Option price ({price:.4f}) below intrinsic value ({intrinsic:.4f})"
                )
            
            # Reasonable upper bound
            max_reasonable = spot * self.validation_rules["option_price"]["max_reasonable_multiple"]
            if price > max_reasonable:
                result.warnings.append(
                    f"Option price ({price:.2f}) seems unusually high compared to spot price ({spot:.2f})"
                )
                result.confidence *= 0.8
        
        # Time value considerations
        if "time_to_expiry" in context:
            tte = context["time_to_expiry"]
            if tte < 1/365 and price > context.get("spot_price", 0) * 0.1:
                result.warnings.append("High time value for option expiring within 1 day")
        
        # Volatility-based validation
        if "volatility" in context:
            vol = context["volatility"]
            if vol > 2.0 and price < context.get("spot_price", 0) * 0.01:
                result.warnings.append("Low option price despite high volatility")
    
    def _validate_greeks(
        self, 
        greeks: Dict[str, float], 
        context: Dict[str, Any], 
        result: OutputValidationResult
    ):
        """Validate Greeks output"""
        
        option_type = context.get("option_type", "call").lower()
        rules = self.validation_rules["greeks"]
        
        for greek_name, value in greeks.items():
            if greek_name == "delta":
                if option_type == "call":
                    min_val, max_val = rules["delta"]["call_range"]
                    if not (min_val <= value <= max_val):
                        result.warnings.append(
                            f"Call delta ({value:.4f}) outside normal range [{min_val}, {max_val}]"
                        )
                        result.confidence *= 0.9
                elif option_type == "put":
                    min_val, max_val = rules["delta"]["put_range"]
                    if not (min_val <= value <= max_val):
                        result.warnings.append(
                            f"Put delta ({value:.4f}) outside normal range [{min_val}, {max_val}]"
                        )
                        result.confidence *= 0.9
            
            elif greek_name == "gamma":
                if value < rules["gamma"]["min"]:
                    result.errors.append("Gamma cannot be negative")
                    result.is_valid = False
                elif value > rules["gamma"]["warning_threshold"]:
                    result.warnings.append(f"High gamma ({value:.4f}) indicates high sensitivity to price changes")
            
            elif greek_name == "theta":
                if value > rules["theta"]["max"]:
                    result.warnings.append("Positive theta is unusual for long options")
                elif value < rules["theta"]["warning_threshold"]:
                    result.warnings.append(f"High time decay (theta: {value:.4f})")
            
            elif greek_name == "vega":
                if value < rules["vega"]["min"]:
                    result.warnings.append("Negative vega is unusual")
                elif value > rules["vega"]["warning_threshold"]:
                    result.warnings.append(f"High vega ({value:.2f}) indicates high volatility sensitivity")
            
            elif greek_name == "rho":
                min_rho, max_rho = rules["rho"]["reasonable_range"]
                if not (min_rho <= value <= max_rho):
                    result.warnings.append(f"Rho ({value:.2f}) outside typical range")
            
            # Check for NaN or infinite values
            if np.isnan(value) or np.isinf(value):
                result.is_valid = False
                result.errors.append(f"Invalid {greek_name} value: {value}")
    
    def _validate_volatility(
        self, 
        volatility: float, 
        context: Dict[str, Any], 
        result: OutputValidationResult
    ):
        """Validate volatility output"""
        
        rules = self.validation_rules["volatility"]
        
        if volatility < rules["min"]:
            result.warnings.append(f"Very low volatility ({volatility:.1%})")
        elif volatility > rules["max"]:
            result.warnings.append(f"Extremely high volatility ({volatility:.1%})")
            result.confidence *= 0.7
        
        # Check if within typical range
        min_typical, max_typical = rules["typical_range"]
        if not (min_typical <= volatility <= max_typical):
            if volatility > max_typical:
                result.warnings.append(f"High volatility ({volatility:.1%}) - ensure this reflects market conditions")
            else:
                result.warnings.append(f"Low volatility ({volatility:.1%}) - verify calculation method")
        
        # Warning for extreme volatility
        if volatility > rules["warning_threshold"]:
            result.warnings.append("Extreme volatility detected - results may be unreliable")
            result.confidence *= 0.6
    
    def _validate_strategy_analysis(
        self, 
        analysis: Dict[str, Any], 
        context: Dict[str, Any], 
        result: OutputValidationResult
    ):
        """Validate option strategy analysis"""
        
        # Check required fields
        required_fields = ["max_profit", "max_loss", "breakeven_points"]
        missing_fields = [field for field in required_fields if field not in analysis]
        if missing_fields:
            result.warnings.append(f"Missing strategy analysis fields: {missing_fields}")
            result.confidence *= 0.8
        
        # Validate profit/loss values
        if "max_profit" in analysis:
            max_profit = analysis["max_profit"]
            if isinstance(max_profit, str) and max_profit.lower() == "unlimited":
                result.recommendations.append("Strategy has unlimited profit potential - monitor position size")
            elif isinstance(max_profit, (int, float)) and max_profit <= 0:
                result.warnings.append("Strategy shows no profit potential")
        
        if "max_loss" in analysis:
            max_loss = analysis["max_loss"]
            if isinstance(max_loss, str) and max_loss.lower() == "unlimited":
                result.warnings.append("Strategy has unlimited loss potential - high risk")
                result.risk_level = "high"
            elif isinstance(max_loss, (int, float)):
                # Check risk-reward ratio
                max_profit = analysis.get("max_profit", 0)
                if isinstance(max_profit, (int, float)) and max_profit > 0:
                    risk_reward = max_profit / abs(max_loss)
                    if risk_reward < 0.5:
                        result.warnings.append(f"Poor risk-reward ratio: {risk_reward:.2f}")
        
        # Validate breakeven points
        if "breakeven_points" in analysis:
            breakevens = analysis["breakeven_points"]
            if isinstance(breakevens, list) and len(breakevens) == 0:
                result.warnings.append("No breakeven points identified")
            elif isinstance(breakevens, list):
                # Check if breakeven points are reasonable
                for be in breakevens:
                    if isinstance(be, (int, float)) and be <= 0:
                        result.warnings.append(f"Invalid breakeven point: {be}")
        
        # Validate probability of profit if provided
        if "probability_of_profit" in analysis:
            prob = analysis["probability_of_profit"]
            if not (0 <= prob <= 1):
                result.warnings.append(f"Invalid probability of profit: {prob}")
            elif prob > 0.8:
                result.warnings.append("High probability of profit - verify assumptions")
            elif prob < 0.3:
                result.warnings.append("Low probability of profit - consider alternatives")
    
    def _validate_arbitrage_opportunity(
        self, 
        opportunity: Dict[str, Any], 
        context: Dict[str, Any], 
        result: OutputValidationResult
    ):
        """Validate arbitrage opportunity detection"""
        
        # Check required fields
        required_fields = ["type", "profit_potential", "risk_level"]
        missing_fields = [field for field in required_fields if field not in opportunity]
        if missing_fields:
            result.warnings.append(f"Incomplete arbitrage analysis: missing {missing_fields}")
        
        # Validate profit potential
        if "profit_potential" in opportunity:
            profit = opportunity["profit_potential"]
            if profit <= 0:
                result.warnings.append("No arbitrage profit identified")
            elif profit < 0.01:  # Less than 1 cent
                result.warnings.append("Arbitrage profit may be below transaction costs")
            elif profit > 10:  # Suspiciously high
                result.warnings.append("High arbitrage profit - verify calculations and market data")
                result.confidence *= 0.7
        
        # Add transaction cost warning
        result.recommendations.append(
            "Consider transaction costs, bid-ask spreads, and execution risks in arbitrage strategies"
        )
        
        # Add timing warning
        result.warnings.append("Arbitrage opportunities may disappear quickly in liquid markets")
    
    def _validate_generic_output(self, result_data: Any, result: OutputValidationResult):
        """Generic validation for any output"""
        
        # Check for None or empty results
        if result_data is None:
            result.is_valid = False
            result.errors.append("No result generated")
            return
        
        # Check for NaN or infinite values in numerical results
        if isinstance(result_data, (int, float)):
            if np.isnan(result_data) or np.isinf(result_data):
                result.is_valid = False
                result.errors.append("Invalid numerical result (NaN or infinite)")
        
        elif isinstance(result_data, dict):
            for key, value in result_data.items():
                if isinstance(value, (int, float)) and (np.isnan(value) or np.isinf(value)):
                    result.warnings.append(f"Invalid value for {key}: {value}")
                    result.confidence *= 0.9
        
        elif isinstance(result_data, list):
            if len(result_data) == 0:
                result.warnings.append("Empty result list")
    
    def _assess_risk_level(self, validation_result: OutputValidationResult) -> str:
        """Assess overall risk level based on validation results"""
        
        # Start with medium risk
        risk_score = 0.5
        
        # Increase risk for errors
        if validation_result.errors:
            risk_score += 0.3 * len(validation_result.errors)
        
        # Increase risk for warnings
        if validation_result.warnings:
            risk_score += 0.1 * len(validation_result.warnings)
        
        # Decrease risk for high confidence
        if validation_result.confidence > 0.9:
            risk_score -= 0.1
        elif validation_result.confidence < 0.7:
            risk_score += 0.2
        
        # Classify risk level
        if risk_score > 0.7:
            return "high"
        elif risk_score < 0.3:
            return "low"
        else:
            return "medium"
    
    def _calculate_uncertainty_bounds(self, value: float, confidence: float) -> Dict[str, float]:
        """Calculate uncertainty bounds based on confidence level"""
        
        # Simple uncertainty model - more sophisticated methods could be used
        uncertainty_factor = (1.0 - confidence) * 0.2  # Max 20% uncertainty
        
        lower_bound = value * (1 - uncertainty_factor)
        upper_bound = value * (1 + uncertainty_factor)
        
        return {
            "lower_bound": lower_bound,
            "upper_bound": upper_bound,
            "uncertainty_range": upper_bound - lower_bound
        }
    
    def _get_contextual_risk_warnings(self, output_type: str) -> List[str]:
        """Get context-specific risk warnings"""
        
        base_warnings = [
            "This analysis is for educational purposes only and not investment advice",
            "Past performance does not guarantee future results",
            "Consider your risk tolerance before making investment decisions"
        ]
        
        context_warnings = {
            "option_price": [
                "Option prices are theoretical and may differ from market prices",
                "Model assumptions may not reflect current market conditions"
            ],
            "greeks": [
                "Greeks change dynamically with market conditions",
                "Use Greeks for risk management, not prediction"
            ],
            "volatility": [
                "Volatility estimates are based on historical data",
                "Implied volatility may differ significantly from realized volatility"
            ],
            "strategy_analysis": [
                "Strategy performance depends heavily on market conditions",
                "Transaction costs can significantly impact profitability",
                "Monitor positions regularly and adjust as needed"
            ],
            "arbitrage_opportunity": [
                "Arbitrage opportunities are rare and typically short-lived",
                "Consider execution risk and transaction costs",
                "Market conditions can change rapidly"
            ]
        }
        
        warnings = base_warnings.copy()
        if output_type in context_warnings:
            warnings.extend(context_warnings[output_type])
        
        return warnings
    
    def format_validation_summary(self, validation_result: OutputValidationResult) -> str:
        """Format validation results into a readable summary"""
        
        summary_parts = []
        
        # Status
        status = "✅ VALID" if validation_result.is_valid else "❌ INVALID"
        summary_parts.append(f"Validation Status: {status}")
        
        # Confidence
        confidence_emoji = "🟢" if validation_result.confidence > 0.8 else "🟡" if validation_result.confidence > 0.6 else "🔴"
        summary_parts.append(f"Confidence: {confidence_emoji} {validation_result.confidence:.1%}")
        
        # Risk Level
        risk_emoji = {"low": "🟢", "medium": "🟡", "high": "🔴"}
        summary_parts.append(f"Risk Level: {risk_emoji.get(validation_result.risk_level, '🟡')} {validation_result.risk_level.upper()}")
        
        # Errors
        if validation_result.errors:
            summary_parts.append(f"❌ Errors ({len(validation_result.errors)}):")
            for error in validation_result.errors:
                summary_parts.append(f"  • {error}")
        
        # Warnings
        if validation_result.warnings:
            summary_parts.append(f"⚠️  Warnings ({len(validation_result.warnings)}):")
            for warning in validation_result.warnings[:3]:  # Limit to first 3
                summary_parts.append(f"  • {warning}")
            if len(validation_result.warnings) > 3:
                summary_parts.append(f"  • ... and {len(validation_result.warnings) - 3} more")
        
        # Recommendations
        if validation_result.recommendations:
            summary_parts.append(f"💡 Recommendations:")
            for recommendation in validation_result.recommendations[:2]:  # Limit to first 2
                summary_parts.append(f"  • {recommendation}")
        
        # Uncertainty bounds
        if validation_result.uncertainty_bounds:
            bounds = validation_result.uncertainty_bounds
            summary_parts.append(
                f"📊 Uncertainty Range: [{bounds['lower_bound']:.4f}, {bounds['upper_bound']:.4f}]"
            )
        
        return "\n".join(summary_parts)

    # Usage example and test
if __name__ == "__main__":
    validator = OutputValidator()
    
    # Test option price validation
    print("Testing Option Price Validation:")
    print("="*50)
    
    # Valid option price
    result = validator.validate_output(
        "option_price", 
        5.50, 
        {"spot_price": 100, "strike": 95, "option_type": "call"}
    )
    print("Valid Call Price:")
    print(validator.format_validation_summary(result))
    
    print("\n" + "-"*30 + "\n")
    
    # Invalid option price (below intrinsic)
    result = validator.validate_output(
        "option_price", 
        2.00, 
        {"spot_price": 100, "strike": 95, "option_type": "call"}
    )
    print("Invalid Call Price (Below Intrinsic):")
    print(validator.format_validation_summary(result))
    
    print("\n" + "="*50 + "\n")
    
    # Test Greeks validation
    print("Testing Greeks Validation:")
    
    greeks = {"delta": 0.65, "gamma": 0.05, "theta": -0.08, "vega": 25.5}
    result = validator.validate_output(
        "greeks", 
        greeks, 
        {"option_type": "call"}
    )
    print("Valid Greeks:")
    print(validator.format_validation_summary(result))
    
    print("\n" + "-"*30 + "\n")
    
    # Invalid Greeks
    invalid_greeks = {"delta": 1.5, "gamma": -0.02, "theta": 0.1, "vega": -10}
    result = validator.validate_output(
        "greeks", 
        invalid_greeks, 
        {"option_type": "call"}
    )
    print("Invalid Greeks:")
    print(validator.format_validation_summary(result))