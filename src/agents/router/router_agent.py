from typing import Dict, List, Any, Optional, TypedDict, Annotated, Sequence
from pydantic import BaseModel, Field
from datetime import datetime
import json
import re
from loguru import logger
from enum import Enum

from langgraph.graph import StateGraph, END
# from langgraph.prebuilt import ToolExecutor
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
import numpy as np

from ...providers.factory import get_llm
from ...providers.openai_provider import OpenAIProvider
from ..grounding.input_validator import InputValidator, ValidationResult
from ..grounding.knowledge_grounding import KnowledgeGroundingAgent, GroundingResult
from ..grounding.output_validator import OutputValidator, OutputValidationResult
from ..strategy.strategy_agent import StrategyAnalysisAgent, StrategyInputs, StrategyType
from ..risk.risk_management_agent import RiskManagementAgent, PortfolioInputs, Position
from ..data.market_data import MarketDataAgent, MarketDataResult
from ..grounding.input_validator import InputValidator, ValidationResult
from ..grounding.knowledge_grounding import KnowledgeGroundingAgent, GroundingResult
from ..grounding.output_validator import OutputValidator, OutputValidationResult
from ..data.market_data import MarketDataAgent, MarketDataResult
from ..pricing.black_scholes import BlackScholesPricingAgent, BlackScholesInputs
from ..pricing.binomial_tree import BinomialTreePricingAgent, BinomialTreeInputs
from ..pricing.monte_carlo import MonteCarloPricingAgent, MonteCarloInputs
from ..pricing.volatility_agent import VolatilityAgent, VolatilityInputs
from ...config.settings import settings
from ...utils.constants import ERROR_MESSAGES

class QueryType(Enum):
    """Types of queries the system can handle"""
    OPTION_PRICING = "option_pricing"
    GREEKS_ANALYSIS = "greeks_analysis"
    VOLATILITY_ANALYSIS = "volatility_analysis"
    STRATEGY_ANALYSIS = "strategy_analysis"
    ARBITRAGE_DETECTION = "arbitrage_detection"
    RISK_MANAGEMENT = "risk_management"
    EDUCATIONAL = "educational"
    INVALID = "invalid"

class AgentState(TypedDict):
    """State object that gets passed between agents"""
    # Input
    user_query: str
    query_type: QueryType
    parsed_parameters: Dict[str, Any]
    
    # Validation
    input_validation: Optional[ValidationResult]
    knowledge_grounding: Optional[GroundingResult]
    output_validation: Optional[OutputValidationResult]
    
    # Data
    market_data: Optional[Dict[str, Any]]
    
    # Processing
    current_agent: str
    processing_steps: List[str]
    intermediate_results: Dict[str, Any]
    errors: List[str]
    warnings: List[str]
    
    # Output
    final_result: Optional[Dict[str, Any]]
    response_message: str
    confidence_score: float
    
    # Messages for LLM interaction
    messages: Annotated[Sequence[BaseMessage], "The messages in the conversation"]

class RouterAgent:
    def __init__(self):
        # Initialize validation and grounding agents
        self.input_validator = InputValidator()
        self.knowledge_grounding = KnowledgeGroundingAgent()
        self.output_validator = OutputValidator()
        self.market_data_agent = MarketDataAgent()
        
        # Initialize specialized pricing agents
        self.black_scholes_agent = BlackScholesPricingAgent()
        self.binomial_tree_agent = BinomialTreePricingAgent()
        self.monte_carlo_agent = MonteCarloPricingAgent()
        self.volatility_agent = VolatilityAgent()
        
        # Initialize LLM
        self.llm = self._initialize_llm()
        
        # Build the workflow graph
        self.workflow = self._build_workflow()
        
        logger.info("Router Agent initialized successfully")
    
    def _initialize_llm(self):
        """Initialize the LLM via the configured provider (LLM_PROVIDER=local|openai|claude)"""
        try:
            llm = get_llm("analysis")
            logger.success(f"LLM provider '{settings.LLM_PROVIDER}' initialized successfully")
            return llm

        except Exception as e:
            logger.error(f"Failed to initialize LLM provider '{settings.LLM_PROVIDER}': {e}")
            logger.info("Falling back to template-based responses (no LLM)")
            return None
    
    def _build_workflow(self) -> StateGraph:
        """Build the LangGraph workflow"""
        
        workflow = StateGraph(AgentState)
        
        # Add nodes for each processing step
        workflow.add_node("validate_input", self._validate_input_node)
        workflow.add_node("classify_query", self._classify_query_node)
        workflow.add_node("ground_knowledge", self._ground_knowledge_node)
        workflow.add_node("fetch_market_data", self._fetch_market_data_node)
        workflow.add_node("route_to_specialist", self._route_to_specialist_node)
        workflow.add_node("validate_output", self._validate_output_node)
        workflow.add_node("generate_response", self._generate_response_node)
        workflow.add_node("judge_response", self._judge_response_node)
        
        # Define the workflow edges
        workflow.set_entry_point("validate_input")
        
        workflow.add_edge("validate_input", "classify_query")
        workflow.add_edge("classify_query", "ground_knowledge") 
        workflow.add_edge("ground_knowledge", "fetch_market_data")
        workflow.add_edge("fetch_market_data", "route_to_specialist")
        workflow.add_edge("route_to_specialist", "validate_output")
        workflow.add_edge("validate_output", "generate_response")
        workflow.add_edge("generate_response", "judge_response")
        workflow.add_edge("judge_response", END)
        
        return workflow.compile()
    
    def _validate_input_node(self, state: AgentState) -> AgentState:
        """Validate user input and extract parameters"""
        logger.info("Validating input...")
        
        try:
            validation_result = self.input_validator.validate_query(state["user_query"])
            state["input_validation"] = validation_result
            state["parsed_parameters"] = validation_result.parsed_params
            state["current_agent"] = "input_validator"
            state["processing_steps"].append("Input validation completed")
            
            if not validation_result.is_valid:
                state["errors"].extend(validation_result.errors)
                state["confidence_score"] = validation_result.confidence
            else:
                # Set initial confidence score from domain relevance
                state["confidence_score"] = validation_result.domain_relevance
            
            logger.info(f"Input validation: {'✅' if validation_result.is_valid else '❌'}")
            
        except Exception as e:
            logger.error(f"Input validation failed: {e}")
            state["errors"].append(f"Input validation error: {str(e)}")
            state["confidence_score"] = 0.0
        
        return state

    def _classify_query_node(self, state: AgentState) -> AgentState:
        """Classify the type of query"""
        logger.info("Classifying query...")
        
        try:
            query = state["user_query"].lower()
            parsed_params = state["parsed_parameters"]
            
            # Check if input validation failed
            if state["input_validation"] and not state["input_validation"].is_valid:
                # Check domain relevance to determine if it's invalid or just low confidence
                if state["input_validation"].domain_relevance < 0.3:
                    query_type = QueryType.INVALID
                else:
                    # Low confidence but domain-relevant, proceed with classification
                    query_type = self._perform_classification(query, parsed_params)
            else:
                query_type = self._perform_classification(query, parsed_params)
            
            state["query_type"] = query_type
            state["current_agent"] = "query_classifier"
            state["processing_steps"].append(f"Query classified as: {query_type.value}")
            
            logger.info(f"Query classified as: {query_type.value}")
            
        except Exception as e:
            logger.error(f"Query classification failed: {e}")
            state["query_type"] = QueryType.INVALID
            state["errors"].append(f"Classification error: {str(e)}")
        
        return state


    def _perform_classification(self, query: str, parsed_params: Dict[str, Any]) -> QueryType:
        """Perform the actual query classification"""
        # Enhanced classification logic
        if any(word in query for word in ["price", "value", "cost", "premium", "worth"]):
            if any(word in query for word in ["american", "early exercise"]):
                return QueryType.OPTION_PRICING  # Will use binomial tree
            elif any(word in query for word in ["barrier", "asian", "exotic"]):
                return QueryType.OPTION_PRICING  # Will use Monte Carlo
            else:
                return QueryType.OPTION_PRICING  # Will use Black-Scholes
        
        elif any(word in query for word in ["delta", "gamma", "theta", "vega", "rho", "greeks"]):
            return QueryType.GREEKS_ANALYSIS
        
        elif any(word in query for word in ["volatility", "vol", "implied", "historical", "garch"]):
            return QueryType.VOLATILITY_ANALYSIS
        
        elif any(word in query for word in ["strategy", "straddle", "strangle", "spread", "butterfly", "condor", "collar"]):
            return QueryType.STRATEGY_ANALYSIS
        
        elif any(word in query for word in ["arbitrage", "put call parity", "mispric"]):
            return QueryType.ARBITRAGE_DETECTION
        
        elif any(word in query for word in ["risk", "hedge", "var", "portfolio", "concentration", "drawdown"]):
            return QueryType.RISK_MANAGEMENT
        
        elif any(word in query for word in ["explain", "what is", "how does", "define", "meaning"]):
            return QueryType.EDUCATIONAL
        
        else:
            # Try to infer from parsed parameters
            if "option_type" in parsed_params or "strike" in parsed_params:
                return QueryType.OPTION_PRICING
            else:
                return QueryType.EDUCATIONAL
    

    def _ground_knowledge_node(self, state: AgentState) -> AgentState:
        """Ground query in knowledge base"""
        logger.info("Grounding knowledge...")
        
        try:
            query = state["user_query"]
            query_type = state["query_type"]
            
            # Determine category filter based on query type
            category_filters = {
                QueryType.OPTION_PRICING: "pricing_models",
                QueryType.GREEKS_ANALYSIS: "greeks", 
                QueryType.VOLATILITY_ANALYSIS: "volatility",
                QueryType.STRATEGY_ANALYSIS: "strategies",
                QueryType.ARBITRAGE_DETECTION: "arbitrage",
                QueryType.EDUCATIONAL: None  # Search all categories
            }
            
            category = category_filters.get(query_type)
            grounding_result = self.knowledge_grounding.ground_query(
                query, 
                top_k=5,
                category_filter=category
            )
            
            state["knowledge_grounding"] = grounding_result
            state["current_agent"] = "knowledge_grounding"
            state["processing_steps"].append("Knowledge grounding completed")
            
            if grounding_result.warnings:
                state["warnings"].extend(grounding_result.warnings)
            
            logger.info(f"Knowledge grounding: confidence {grounding_result.confidence:.2f}")
            
        except Exception as e:
            logger.error(f"Knowledge grounding failed: {e}")
            state["errors"].append(f"Knowledge grounding error: {str(e)}")
        
        return state

    def _fetch_market_data_node(self, state: AgentState) -> AgentState:
        """Fetch required market data"""
        logger.info("Fetching market data...")
        
        try:
            parsed_params = state["parsed_parameters"]
            symbol = parsed_params.get("symbol")
            
            market_data = {}
            
            if symbol:
                logger.info(f"Fetching market data for symbol: {symbol}")
                
                # Get current stock data
                stock_result = self.market_data_agent.get_stock_data(symbol)
                if stock_result.success:
                    market_data["stock_data"] = stock_result.data
                    logger.info(f"Successfully fetched stock data for {symbol}: ${stock_result.data['current_price']:.2f}")
                    
                    # Get historical data for volatility calculations
                    query_type = state["query_type"]
                    if query_type in [QueryType.OPTION_PRICING, QueryType.VOLATILITY_ANALYSIS, QueryType.GREEKS_ANALYSIS]:
                        hist_result = self.market_data_agent.get_historical_data(symbol, period="1y")
                        if hist_result.success:
                            market_data["historical_data"] = hist_result.data
                        
                        vol_result = self.market_data_agent.calculate_historical_volatility(symbol, days=30)
                        if vol_result.success:
                            market_data["volatility_data"] = vol_result.data
                            logger.info(f"Historical volatility for {symbol}: {vol_result.data['volatility']:.1%}")
                    
                    # Get options chain if needed
                    if query_type in [QueryType.OPTION_PRICING, QueryType.STRATEGY_ANALYSIS, QueryType.ARBITRAGE_DETECTION]:
                        options_result = self.market_data_agent.get_options_chain(symbol)
                        if options_result.success:
                            market_data["options_data"] = options_result.data
                            logger.info(f"Options chain fetched for {symbol}")
                else:
                    state["warnings"].append(f"Could not fetch market data for {symbol}: {stock_result.error_message}")
                    logger.warning(f"Failed to fetch stock data for {symbol}: {stock_result.error_message}")
            else:
                logger.warning("No symbol found in parsed parameters")
                state["warnings"].append("No stock symbol identified in query")
            
            # Always get risk-free rate
            market_data["risk_free_rate"] = self.market_data_agent.get_risk_free_rate()
            
            # Get dividend yield if symbol available
            if symbol:
                market_data["dividend_yield"] = self.market_data_agent.get_dividend_yield(symbol)
            
            state["market_data"] = market_data
            state["current_agent"] = "market_data"
            state["processing_steps"].append("Market data fetched")
            
            logger.info(f"Market data summary: symbol={symbol}, has_stock_data={bool(market_data.get('stock_data'))}, has_vol_data={bool(market_data.get('volatility_data'))}")
            
        except Exception as e:
            logger.error(f"Market data fetch failed: {e}")
            state["errors"].append(f"Market data error: {str(e)}")
        
        return state

    def _create_sample_portfolio(self, market_data: Dict[str, Any], parsed_params: Dict[str, Any]) -> List[Position]:
       """Create sample portfolio for risk analysis demo"""
       positions = []
       
       if not market_data or not market_data.get("stock_data"):
           return positions
       
       stock_data = market_data["stock_data"]
       symbol = parsed_params.get("symbol", "DEMO")
       spot_price = stock_data["current_price"]
       
       # Add stock position
       positions.append(Position(
           symbol=symbol,
           position_type="stock",
           quantity=100,
           current_price=spot_price,
           delta=100.0  # 100 shares = 100 delta
       ))
       
       # Add option position if we have options data
       if market_data.get("options_data"):
           volatility = self._get_volatility_estimate(market_data)
           
           # Create sample call option
           call_inputs = BlackScholesInputs(
               spot_price=spot_price,
               strike_price=spot_price * 1.05,  # 5% OTM call
               time_to_expiry=45/365,
               risk_free_rate=market_data.get("risk_free_rate", 0.05),
               volatility=volatility,
               option_type="call"
           )
           
           call_result = self.black_scholes_agent.calculate_option_price(call_inputs)
           
           positions.append(Position(
               symbol=symbol,
               position_type="option",
               quantity=10,  # 10 contracts
               current_price=call_result.option_price,
               strike=spot_price * 1.05,
               expiry=45/365,
               option_type="call",
               implied_vol=volatility,
               delta=call_result.greeks["delta"] * 10,
               gamma=call_result.greeks["gamma"] * 10,
               theta=call_result.greeks["theta"] * 10,
               vega=call_result.greeks["vega"] * 10
           ))
       
       return positions

    
    def _process_risk_management(self, state: AgentState) -> Dict[str, Any]:
       """Process risk management queries"""
       try:
           parsed_params = state["parsed_parameters"]
           market_data = state["market_data"]
           query = state["user_query"].lower()
           
           # For demo purposes, create a sample portfolio
           # In real implementation, this would come from user's actual portfolio
           sample_positions = self._create_sample_portfolio(market_data, parsed_params)
           
           if not sample_positions:
               return {"error": "Unable to analyze risk without portfolio positions"}
           
           # Create portfolio inputs
           portfolio_inputs = PortfolioInputs(
               positions=sample_positions,
               portfolio_value=sum(abs(pos.quantity * pos.current_price) for pos in sample_positions),
               risk_free_rate=market_data.get("risk_free_rate", 0.05),
               confidence_level=0.95
           )
           
           # Analyze portfolio risk
           risk_result = self.risk_management_agent.analyze_portfolio_risk(portfolio_inputs)
           
           return {
               "analysis_type": "Risk Management",
               "portfolio_summary": risk_result.portfolio_summary,
               "risk_metrics": {
                   "var_1day": risk_result.risk_metrics.var_1day,
                   "var_10day": risk_result.risk_metrics.var_10day,
                   "total_delta": risk_result.risk_metrics.total_delta,
                   "total_gamma": risk_result.risk_metrics.total_gamma,
                   "total_theta": risk_result.risk_metrics.total_theta,
                   "total_vega": risk_result.risk_metrics.total_vega,
                   "concentration_risk": risk_result.risk_metrics.concentration_risk
               },
               "hedge_recommendations": [
                   {
                       "type": hedge.hedge_type.value,
                       "action": hedge.recommended_action,
                       "instrument": hedge.hedge_instrument,
                       "cost": hedge.expected_cost,
                       "risk_reduction": hedge.risk_reduction
                   }
                   for hedge in risk_result.hedge_recommendations
               ],
               "stress_test_results": risk_result.stress_test_results,
               "risk_warnings": risk_result.risk_warnings,
               "action_items": risk_result.action_items
           }
           
       except Exception as e:
           logger.error(f"Risk management analysis failed: {e}")
           return {"error": f"Risk management analysis failed: {str(e)}"}


    def _extract_strategy_strikes(self, parsed_params: Dict[str, Any], spot_price: float, strategy_type: StrategyType) -> List[float]:
        """Extract appropriate strikes for strategy"""
        if "strikes" in parsed_params and parsed_params["strikes"]:
            return parsed_params["strikes"]
        # Generate default strikes based on strategy type
        if strategy_type in [StrategyType.LONG_STRADDLE, StrategyType.SHORT_STRADDLE]:
            return [spot_price]  # ATM straddle
        elif strategy_type in [StrategyType.LONG_STRANGLE, StrategyType.SHORT_STRANGLE]:
            return [spot_price * 0.95, spot_price * 1.05]  # 5% OTM strangle
        elif strategy_type == StrategyType.IRON_CONDOR:
            return [spot_price * 0.90, spot_price * 0.95, spot_price * 1.05, spot_price * 1.10]  # Iron condor strikes
        elif strategy_type == StrategyType.BULL_CALL_SPREAD:
            return [spot_price * 0.98, spot_price * 1.05]  # Bull call spread
        elif strategy_type == StrategyType.BEAR_PUT_SPREAD:
            return [spot_price * 1.02, spot_price * 0.95]  # Bear put spread
        elif strategy_type in [StrategyType.LONG_BUTTERFLY, StrategyType.IRON_BUTTERFLY]:
            return [spot_price * 0.95, spot_price, spot_price * 1.05]  # Butterfly strikes
        else:
            return [spot_price]  # Default to ATM
    

    def _identify_strategy_type(self, query: str) -> Optional[StrategyType]:
        """Identify strategy type from query text"""
        strategy_keywords = {
            "long call": StrategyType.LONG_CALL,
            "long put": StrategyType.LONG_PUT,
            "covered call": StrategyType.COVERED_CALL,
            "protective put": StrategyType.PROTECTIVE_PUT,
            "bull call spread": StrategyType.BULL_CALL_SPREAD,
            "bear put spread": StrategyType.BEAR_PUT_SPREAD,
            "long straddle": StrategyType.LONG_STRADDLE,
            "short straddle": StrategyType.SHORT_STRADDLE,
            "long strangle": StrategyType.LONG_STRANGLE,
            "short strangle": StrategyType.SHORT_STRANGLE,
            "iron condor": StrategyType.IRON_CONDOR,
            "iron butterfly": StrategyType.IRON_BUTTERFLY,
            "long butterfly": StrategyType.LONG_BUTTERFLY,
            "straddle": StrategyType.LONG_STRADDLE,  # Default to long
            "strangle": StrategyType.LONG_STRANGLE,  # Default to long
            "butterfly": StrategyType.LONG_BUTTERFLY,  # Default to long
            "condor": StrategyType.IRON_CONDOR
        }
        
        for keyword, strategy_type in strategy_keywords.items():
            if keyword in query:
                return strategy_type
        
        return None
    
    def _parse_time_to_expiry(self, parsed_params: Dict[str, Any]) -> float:
        """Parse time to expiry from parameters"""
        if "expiry_raw" in parsed_params:
            expiry_raw = parsed_params["expiry_raw"]
            
            if isinstance(expiry_raw, tuple) and len(expiry_raw) == 2:
                amount, unit = expiry_raw
                amount = int(amount)
                
                if unit.startswith("day"):
                    return amount / 365
                elif unit.startswith("week"):
                    return (amount * 7) / 365
                elif unit.startswith("month"):
                    return (amount * 30) / 365
        
        # Default to 30 days
        return 30 / 365
    
    def _get_volatility_estimate(self, market_data: Dict[str, Any]) -> float:
        """Get volatility estimate from market data"""
        # Try to get from volatility analysis first
        if "volatility_data" in market_data:
            vol_data = market_data["volatility_data"]
            if "volatility" in vol_data:
                return vol_data["volatility"]
        
        # Try to get from historical data
        if "historical_data" in market_data:
            hist_data = market_data["historical_data"]
            if "annualized_volatility" in hist_data:
                return hist_data["annualized_volatility"]
        
        # Default volatility
        return 0.25  # 25%
    

    def _process_option_pricing(self, state: AgentState) -> Dict[str, Any]:
        """Process option pricing using appropriate method"""
        try:
            parsed_params = state["parsed_parameters"]
            market_data = state["market_data"]
            query = state["user_query"].lower()
            
            logger.info(f"Processing option pricing with params: {parsed_params}")
            
            # Get symbol
            symbol = parsed_params.get("symbol")
            if not symbol:
                return {"error": "No stock symbol identified in query. Please specify a stock symbol (e.g., AAPL, GOOGL, TSLA)."}
            
            # Check if we have market data, if not use fallback
            if not market_data or not market_data.get("stock_data"):
                logger.warning(f"No market data available for {symbol}, using fallback pricing")
                
                # Use fallback prices for common symbols for demo purposes
                fallback_prices = {
                    'AAPL': 175.0,
                    'GOOGL': 140.0,
                    'GOOG': 140.0,
                    'MSFT': 375.0,
                    'TSLA': 200.0,
                    'AMZN': 145.0,
                    'META': 485.0,
                    'NVDA': 800.0,
                    'SPY': 420.0,
                    'QQQ': 370.0
                }
                
                if symbol in fallback_prices:
                    spot_price = fallback_prices[symbol]
                    logger.info(f"Using fallback price for {symbol}: ${spot_price}")
                    
                    # Create mock market data
                    market_data = {
                        "stock_data": {
                            "current_price": spot_price,
                            "symbol": symbol
                        },
                        "risk_free_rate": 0.05,
                        "dividend_yield": 0.0
                    }
                    
                    # Add to state for future use
                    state["market_data"] = market_data
                    
                else:
                    return {"error": f"Could not fetch market data for {symbol} and no fallback price available. Please try AAPL, GOOGL, MSFT, TSLA, or other major stocks."}
            
            stock_data = market_data["stock_data"]
            spot_price = stock_data["current_price"]
            
            # Extract parameters with better defaults
            strike = parsed_params.get("strike")
            if not strike:
                # If no strike specified, use ATM
                strike = spot_price
                logger.info(f"No strike specified, using ATM: ${strike:.2f}")
            
            option_type = parsed_params.get("option_type", "call")
            
            # Determine time to expiry
            time_to_expiry = self._parse_time_to_expiry(parsed_params)
            logger.info(f"Time to expiry: {time_to_expiry:.4f} years ({time_to_expiry*365:.0f} days)")
            
            # Get volatility (use fallback if no market data)
            volatility = self._get_volatility_estimate(market_data)
            if volatility == 0.25:  # Default fallback
                # Use symbol-specific volatility estimates
                symbol_volatilities = {
                    'AAPL': 0.28,
                    'GOOGL': 0.32,
                    'GOOG': 0.32,
                    'MSFT': 0.25,
                    'TSLA': 0.45,
                    'AMZN': 0.35,
                    'META': 0.40,
                    'NVDA': 0.50,
                    'SPY': 0.18,
                    'QQQ': 0.22
                }
                volatility = symbol_volatilities.get(symbol, 0.30)
            
            logger.info(f"Using volatility: {volatility:.1%}")
            
            # Get other parameters with validation
            risk_free_rate = market_data.get("risk_free_rate", 0.05)
            dividend_yield = market_data.get("dividend_yield", 0.0)
            
            # Validate and cap parameters to ensure they meet BlackScholes constraints
            risk_free_rate = max(-0.05, min(risk_free_rate, 0.15))  # Cap between -5% and 15%
            dividend_yield = max(0.0, min(dividend_yield, 0.25))    # Cap between 0% and 25%
            volatility = max(0.01, min(volatility, 2.0))            # Cap between 1% and 200%
            time_to_expiry = max(1/365, min(time_to_expiry, 5.0))   # Cap between 1 day and 5 years
            
            # Log all inputs after validation
            logger.info(f"Validated option pricing inputs: {symbol} S=${spot_price:.2f}, K=${strike:.2f}, T={time_to_expiry:.4f}, r={risk_free_rate:.4f}, σ={volatility:.4f}, q={dividend_yield:.4f}, type={option_type}")
            
            # Determine exercise style: explicit query keyword overrides an inference
            # from the underlying's quoteType (index options are conventionally
            # European-style regardless of country; single-stock/ETF options are
            # conventionally American-style regardless of country).
            quote_type = market_data.get("stock_data", {}).get("quote_type", "EQUITY")

            if any(word in query for word in ["american", "early exercise"]):
                exercise_style, style_source = "american", "explicit (query)"
            elif "european" in query:
                exercise_style, style_source = "european", "explicit (query)"
            elif quote_type == "INDEX":
                exercise_style, style_source = "european", f"inferred (quoteType={quote_type})"
            else:
                exercise_style, style_source = "american", f"inferred (quoteType={quote_type})"

            logger.info(f"Exercise style: {exercise_style} [{style_source}]")

            # Determine which pricing method to use
            if any(word in query for word in ["barrier", "asian", "exotic"]):
                # Use Monte Carlo for exotic options
                inputs = MonteCarloInputs(
                    spot_price=spot_price,
                    strike_price=strike,
                    time_to_expiry=time_to_expiry,
                    risk_free_rate=risk_free_rate,
                    volatility=volatility,
                    dividend_yield=dividend_yield,
                    option_type=option_type,
                    num_simulations=25000,  # Reduced for faster demo
                    num_steps=int(max(30, time_to_expiry * 252))
                )
                
                # Check for barrier options
                if "barrier" in query:
                    if "up" in query and "out" in query:
                        inputs.barrier_type = "up-and-out"
                        inputs.barrier_level = spot_price * 1.2
                    elif "down" in query and "out" in query:
                        inputs.barrier_type = "down-and-out"
                        inputs.barrier_level = spot_price * 0.8
                
                # Check for Asian options
                if "asian" in query:
                    inputs.asian_type = "arithmetic"
                
                result = self.monte_carlo_agent.calculate_option_price(inputs)
                method = "Monte Carlo Simulation"

            elif exercise_style == "american":
                # Binomial tree for American-style exercise
                inputs = BinomialTreeInputs(
                    spot_price=spot_price,
                    strike_price=strike,
                    time_to_expiry=time_to_expiry,
                    risk_free_rate=risk_free_rate,
                    volatility=volatility,
                    dividend_yield=dividend_yield,
                    option_type=option_type,
                    option_style="american",
                    steps=100
                )
                result = self.binomial_tree_agent.calculate_option_price(inputs)
                method = f"Binomial Tree (American, {style_source})"

            else:
                # Black-Scholes for European-style exercise
                inputs = BlackScholesInputs(
                    spot_price=spot_price,
                    strike_price=strike,
                    time_to_expiry=time_to_expiry,
                    risk_free_rate=risk_free_rate,
                    volatility=volatility,
                    dividend_yield=dividend_yield,
                    option_type=option_type
                )
                result = self.black_scholes_agent.calculate_option_price(inputs)
                method = f"Black-Scholes-Merton (European, {style_source})"
            
            logger.info(f"Option pricing completed using {method}: ${result.option_price:.4f}")
            
            # Format result
            formatted_result = {
                "pricing_method": method,
                "option_price": result.option_price,
                "intrinsic_value": result.intrinsic_value,
                "time_value": result.time_value,
                "moneyness": getattr(result, 'moneyness', strike / spot_price),
                "greeks": result.greeks,
                "inputs": {
                    "spot_price": spot_price,
                    "strike_price": strike,
                    "time_to_expiry": time_to_expiry,
                    "volatility": volatility,
                    "risk_free_rate": risk_free_rate,
                    "dividend_yield": dividend_yield,
                    "option_type": option_type,
                    "symbol": symbol
                },
                "market_data_source": "live" if market_data.get("stock_data", {}).get("data_quality") else "fallback",
                "data_timestamp": market_data.get("stock_data", {}).get("last_updated", "unknown")
            }
            
            return formatted_result
        
        except Exception as e:
            logger.error(f"Option pricing failed: {e}")
            return {"error": f"Option pricing calculation failed: {str(e)}"}

    def _process_greeks_analysis(self, state: AgentState) -> Dict[str, Any]:
        """Process Greeks analysis"""
        try:
            # Use the same parameter extraction as option pricing
            pricing_result = self._process_option_pricing(state)
            
            if "error" in pricing_result:
                return pricing_result
            
            greeks = pricing_result["greeks"]
            inputs = pricing_result["inputs"]
            
            # Add Greeks interpretations
            interpretations = {
                "delta": f"A $1 increase in stock price will change option value by ~${abs(greeks['delta']):.3f}",
                "gamma": f"Delta will change by {greeks['gamma']:.4f} for each $1 stock price move",
                "theta": f"Option loses ~${abs(greeks['theta']):.3f} per day due to time decay",
                "vega": f"A 1% volatility increase changes option value by ~${greeks['vega']:.3f}",
                "rho": f"A 1% interest rate increase changes option value by ~${greeks['rho']:.3f}"
            }
            
            # Risk assessment
            risk_factors = []
            if abs(greeks["delta"]) > 0.7:
                risk_factors.append("High directional risk - sensitive to stock price moves")
            if greeks["gamma"] > 0.1:
                risk_factors.append("High gamma - delta will change rapidly")
            if abs(greeks["theta"]) > 0.1:
                risk_factors.append("High time decay - option value erodes quickly")
            if greeks["vega"] > 50:
                risk_factors.append("High volatility risk - sensitive to volatility changes")
            
            return {
                "greeks": greeks,
                "interpretations": interpretations,
                "risk_factors": risk_factors,
                "option_details": inputs,
                "analysis_type": "Greeks Analysis"
            }
            
        except Exception as e:
            logger.error(f"Greeks analysis failed: {e}")
            return {"error": f"Greeks analysis failed: {str(e)}"}
    
    def _process_volatility_analysis(self, state: AgentState) -> Dict[str, Any]:
        """Process volatility analysis"""
        try:
            parsed_params = state["parsed_parameters"]
            market_data = state["market_data"]
            
            symbol = parsed_params.get("symbol")
            if not symbol or not market_data:
                return {"error": "Symbol and market data required for volatility analysis"}
            
            # Prepare historical data
            price_data = None
            if "historical_data" in market_data:
                hist_data = market_data["historical_data"]["data"]
                price_data = hist_data
            
            # Prepare options data
            options_data = None
            if "options_data" in market_data:
                calls_df = market_data["options_data"]["calls"]
                puts_df = market_data["options_data"]["puts"]
                if not calls_df.empty or not puts_df.empty:
                    options_data = pd.concat([calls_df, puts_df], ignore_index=True)
                    # Add type column
                    options_data['type'] = ['call'] * len(calls_df) + ['put'] * len(puts_df)
            
            # Create volatility inputs
            vol_inputs = VolatilityInputs(
                symbol=symbol,
                price_data=price_data,
                options_data=options_data,
                spot_price=market_data["stock_data"]["current_price"],
                risk_free_rate=market_data.get("risk_free_rate", 0.05),
                dividend_yield=market_data.get("dividend_yield", 0.0),
                analysis_period=30
            )
            
            # Perform volatility analysis
            vol_result = self.volatility_agent.analyze_volatility(vol_inputs)
            
            return {
                "symbol": symbol,
                "historical_volatility": vol_result.historical_volatility,
                "implied_volatility": vol_result.implied_volatility,
                "volatility_statistics": vol_result.volatility_statistics,
                "regime_analysis": vol_result.regime_analysis,
                "volatility_surface": vol_result.volatility_surface,
                "garch_forecast": vol_result.garch_forecast,
                "analysis_type": "Volatility Analysis"
            }
            
        except Exception as e:
            logger.error(f"Volatility analysis failed: {e}")
            return {"error": f"Volatility analysis failed: {str(e)}"}
    
    def _process_educational_query(self, state: AgentState) -> Dict[str, Any]:
        """Process educational queries using knowledge grounding"""
        try:
            grounding = state["knowledge_grounding"]
            
            if grounding and grounding.retrieved_knowledge:
                knowledge = grounding.retrieved_knowledge[0]
                
                return {
                    "explanation": knowledge["content"],
                    "category": knowledge["category"],
                    "supporting_formulas": grounding.supporting_formulas,
                    "confidence": grounding.confidence,
                    "analysis_type": "Educational Content"
                }
            
            return {
                "explanation": "I can help explain options trading concepts, pricing models, and strategies.",
                "message": "Please ask more specific questions about options theory, pricing, or strategies.",
                "analysis_type": "Educational Content"
            }
            
        except Exception as e:
            logger.error(f"Educational query processing failed: {e}")
            return {"error": f"Educational processing failed: {str(e)}"}
    

    def _process_strategy_analysis(self, state: AgentState) -> Dict[str, Any]:
        """Process strategy analysis queries"""
        try:
            parsed_params = state["parsed_parameters"]
            market_data = state["market_data"]
            query = state["user_query"].lower()
            
            if not market_data or not market_data.get("stock_data"):
                return {"error": "Market data required for strategy analysis"}
            
            stock_data = market_data["stock_data"]
            spot_price = stock_data["current_price"]
            
            # Determine strategy type from query
            strategy_type = self._identify_strategy_type(query)
            
            if not strategy_type:
                return {"error": "Unable to identify strategy type from query"}
            
            # Extract strikes
            strikes = self._extract_strategy_strikes(parsed_params, spot_price, strategy_type)
            
            # Create strategy inputs
            strategy_inputs = StrategyInputs(
                strategy_type=strategy_type,
                spot_price=spot_price,
                strikes=strikes,
                expiries=[self._parse_time_to_expiry(parsed_params)] * len(strikes),
                risk_free_rate=market_data.get("risk_free_rate", 0.05),
                volatility=self._get_volatility_estimate(market_data),
                dividend_yield=market_data.get("dividend_yield", 0.0),
                market_outlook=parsed_params.get("market_outlook", "neutral")
            )
            
            # Analyze strategy
            result = self.strategy_agent.analyze_strategy(strategy_inputs)
            
            return {
                "analysis_type": "Strategy Analysis",
                "strategy_name": result.strategy_name,
                "strategy_type": result.strategy_type.value,
                "net_premium": result.net_premium,
                "max_profit": result.max_profit,
                "max_loss": result.max_loss,
                "breakeven_points": result.breakeven_points,
                "profit_probability": result.profit_probability,
                "portfolio_greeks": result.portfolio_greeks,
                "market_bias": result.market_bias,
                "complexity_rating": result.complexity_rating,
                "risk_warnings": result.risk_warnings,
                "management_guidelines": result.management_guidelines,
                "payoff_data": result.payoff_data
            }
            
        except Exception as e:
            logger.error(f"Strategy analysis failed: {e}")
            return {"error": f"Strategy analysis failed: {str(e)}"}


    def _route_to_specialist_node(self, state: AgentState) -> AgentState:
        """Route to appropriate specialist agent for processing"""
        logger.info("Routing to specialist agent...")
        
        try:
            query_type = state["query_type"]
            
            
            if query_type == QueryType.INVALID:
                state["final_result"] = {
                    "error": "Invalid query",
                    "message": "Unable to process this request. Please ask about options trading topics."
                }
                state["confidence_score"] = 0.0
                return state
            # Route to specialized agents
            if query_type == QueryType.OPTION_PRICING:
                result = self._process_option_pricing(state)
            elif query_type == QueryType.GREEKS_ANALYSIS:
                result = self._process_greeks_analysis(state)
            elif query_type == QueryType.VOLATILITY_ANALYSIS:
                result = self._process_volatility_analysis(state)
            elif query_type == QueryType.STRATEGY_ANALYSIS:
                result = self._process_strategy_analysis(state)
            elif query_type == QueryType.RISK_MANAGEMENT:
                result = self._process_risk_management(state)
            else:
                # Fallback for other query types
                result = self._process_educational_query(state)
            
            state["final_result"] = result
            state["current_agent"] = f"specialist_{query_type.value}"
            state["processing_steps"].append(f"Processed by {query_type.value} specialist")
            
            logger.info(f"Specialist processing completed for {query_type.value}")
            
        except Exception as e:
            logger.error(f"Specialist routing failed: {e}")
            state["errors"].append(f"Specialist processing error: {str(e)}")
            state["final_result"] = {"error": str(e)}
        
        return state
    
    def _process_with_specialist(self, state: AgentState) -> Dict[str, Any]:
        """Process with appropriate specialist (mock implementation)"""
        query_type = state["query_type"]
        parsed_params = state["parsed_parameters"]
        market_data = state["market_data"]
        grounding = state["knowledge_grounding"]
        
        if query_type == QueryType.OPTION_PRICING:
            return self._mock_option_pricing(parsed_params, market_data, grounding)
        elif query_type == QueryType.GREEKS_ANALYSIS:
            return self._mock_greeks_analysis(parsed_params, market_data, grounding)
        elif query_type == QueryType.VOLATILITY_ANALYSIS:
            return self._mock_volatility_analysis(parsed_params, market_data, grounding)
        elif query_type == QueryType.STRATEGY_ANALYSIS:
            return self._mock_strategy_analysis(parsed_params, market_data, grounding)
        elif query_type == QueryType.EDUCATIONAL:
            return self._mock_educational_response(state["user_query"], grounding)
        else:
            return {"message": "Specialist processing not yet implemented for this query type"}
    
    def _mock_option_pricing(self, params: Dict, market_data: Dict, grounding: GroundingResult) -> Dict[str, Any]:
        """Mock option pricing calculation"""
        if not market_data.get("stock_data"):
            return {"error": "Market data required for option pricing"}
        
        stock_data = market_data["stock_data"]
        current_price = stock_data["current_price"]
        strike = params.get("strike", current_price)
        option_type = params.get("option_type", "call")
        
        # Simple mock calculation (would use actual Black-Scholes)
        intrinsic = max(0, current_price - strike) if option_type == "call" else max(0, strike - current_price)
        time_value = current_price * 0.02  # Mock time value
        option_price = intrinsic + time_value
        
        return {
            "option_price": round(option_price, 2),
            "intrinsic_value": round(intrinsic, 2),
            "time_value": round(time_value, 2),
            "underlying_price": current_price,
            "strike_price": strike,
            "option_type": option_type,
            "calculation_method": "Black-Scholes (mock)",
            "assumptions": grounding.supporting_formulas if grounding else []
        }
    
    def _mock_greeks_analysis(self, params: Dict, market_data: Dict, grounding: GroundingResult) -> Dict[str, Any]:
        """Mock Greeks calculation"""
        if not market_data.get("stock_data"):
            return {"error": "Market data required for Greeks analysis"}
        
        # Mock Greeks values
        option_type = params.get("option_type", "call")
        
        if option_type == "call":
            greeks = {
                "delta": 0.6,
                "gamma": 0.05,
                "theta": -0.08,
                "vega": 25.0,
                "rho": 15.0
            }
        else:
            greeks = {
                "delta": -0.4,
                "gamma": 0.05,
                "theta": -0.06,
                "vega": 25.0,
                "rho": -12.0
            }
        
        return {
            "greeks": greeks,
            "option_type": option_type,
            "interpretation": {
                "delta": "Price sensitivity to underlying movement",
                "gamma": "Delta sensitivity to underlying movement", 
                "theta": "Time decay per day",
                "vega": "Volatility sensitivity",
                "rho": "Interest rate sensitivity"
            }
        }
    
    def _mock_volatility_analysis(self, params: Dict, market_data: Dict, grounding: GroundingResult) -> Dict[str, Any]:
        """Mock volatility analysis"""
        vol_data = market_data.get("volatility_data", {})
        
        return {
            "historical_volatility": vol_data.get("volatility", 0.25),
            "annualized": True,
            "period_days": vol_data.get("days_used", 30),
            "volatility_interpretation": "Measure of price variability",
            "typical_range": "10% - 80% for most stocks"
        }
    
    def _mock_strategy_analysis(self, params: Dict, market_data: Dict, grounding: GroundingResult) -> Dict[str, Any]:
        """Mock strategy analysis"""
        strategy = params.get("strategy", "straddle")
        current_price = market_data.get("stock_data", {}).get("current_price", 100)
        
        return {
            "strategy_name": strategy,
            "description": f"Analysis of {strategy} strategy",
            "max_profit": "Unlimited" if strategy in ["straddle", "strangle"] else "Limited",
            "max_loss": "Premium paid",
            "breakeven_points": [current_price - 10, current_price + 10],
            "market_outlook": "Neutral with high volatility expectation"
        }
    
    def _mock_educational_response(self, query: str, grounding: GroundingResult) -> Dict[str, Any]:
        """Mock educational response"""
        if grounding and grounding.retrieved_knowledge:
            knowledge = grounding.retrieved_knowledge[0]
            return {
                "explanation": knowledge["content"],
                "category": knowledge["category"],
                "additional_info": "This is educational content based on established options theory"
            }
        
        return {
            "explanation": "Educational content about options trading concepts",
            "message": "Please ask more specific questions about options pricing, strategies, or Greeks"
        }


    def _validate_output_node(self, state: AgentState) -> AgentState:
        """Validate the output from specialist agent"""
        logger.info("Validating output...")
        
        try:
            final_result = state["final_result"]
            query_type = state["query_type"]
            
            if not final_result or "error" in final_result:
                state["output_validation"] = OutputValidationResult(
                    is_valid=False,
                    confidence=0.0,
                    errors=["No valid result generated"]
                )
                return state
            
            # Map query type to validation type
            validation_type_map = {
                QueryType.OPTION_PRICING: "option_price",
                QueryType.GREEKS_ANALYSIS: "greeks",
                QueryType.VOLATILITY_ANALYSIS: "volatility",
                QueryType.STRATEGY_ANALYSIS: "strategy_analysis"
            }
            
            validation_type = validation_type_map.get(query_type, "generic")
            
            # Extract the main result for validation with better context handling
            if validation_type == "option_price":
                result_to_validate = final_result.get("option_price")
                context = {
                    "spot_price": final_result.get("inputs", {}).get("spot_price"),
                    "strike": final_result.get("inputs", {}).get("strike_price"),
                    "option_type": final_result.get("inputs", {}).get("option_type")
                }
            elif validation_type == "greeks":
                result_to_validate = final_result.get("greeks", {})
                context = {
                    "option_type": final_result.get("option_details", {}).get("option_type") or 
                                final_result.get("inputs", {}).get("option_type") or "call"
                }
            elif validation_type == "volatility":
                result_to_validate = final_result.get("historical_volatility", {}).get("annualized_volatility", 0.25)
                context = {}
            else:
                result_to_validate = final_result
                context = {}
            
            # Ensure context has valid values
            if context.get("option_type") is None:
                context["option_type"] = "call"  # Default fallback
            
            validation_result = self.output_validator.validate_output(
                validation_type,
                result_to_validate,
                context
            )
            
            state["output_validation"] = validation_result
            state["current_agent"] = "output_validator"
            state["processing_steps"].append("Output validation completed")
            
            if validation_result.warnings:
                state["warnings"].extend(validation_result.warnings)
            
            if validation_result.errors:
                state["errors"].extend(validation_result.errors)
            
            # Update confidence score
            if state["confidence_score"] == 0:
                state["confidence_score"] = validation_result.confidence
            else:
                state["confidence_score"] = (state["confidence_score"] + validation_result.confidence) / 2
            
            logger.info(f"Output validation: {'✅' if validation_result.is_valid else '❌'}")
            
        except Exception as e:
            logger.error(f"Output validation failed: {e}")
            state["errors"].append(f"Output validation error: {str(e)}")
            # Don't fail the entire process for validation errors
            state["output_validation"] = OutputValidationResult(
                is_valid=True,  # Allow process to continue
                confidence=0.8,
                warnings=[f"Validation error: {str(e)}"]
            )
        
        return state

    def _generate_response_node(self, state: AgentState) -> AgentState:
        """Generate final response message"""
        logger.info("Generating response...")
        
        try:
            final_result = state["final_result"]
            output_validation = state["output_validation"]
            query_type = state["query_type"]
            
            # Build response message
            response_parts = []

            # Add main result
            if final_result and "error" not in final_result:
                formatted_result = self._format_result_for_response(final_result, query_type)
                narrative = self._generate_llm_narrative(
                    state["user_query"], query_type, formatted_result, final_result
                )
                response_parts.append(narrative or formatted_result)

            # Add validation summary if available
            if output_validation:
                response_parts.append(self._format_validation_summary(output_validation))
            
            # Add warnings
            if state["warnings"]:
                response_parts.append("⚠️ Warnings:")
                for warning in state["warnings"][:3]:  # Limit warnings
                    response_parts.append(f"• {warning}")
            
            # Add errors if any
            if state["errors"]:
                response_parts.append("❌ Issues:")
                for error in state["errors"][:2]:  # Limit errors
                    response_parts.append(f"• {error}")
            
            # Add risk disclaimers
            response_parts.extend([
                "",
                "📋 Important Notes:",
                "• This analysis is for educational purposes only",
                "• Past performance does not guarantee future results", 
                "• Consider your risk tolerance before trading"
            ])
            
            state["response_message"] = "\n".join(response_parts)
            state["current_agent"] = "response_generator"
            state["processing_steps"].append("Response generated")
            
            logger.info("Response generation completed")
            
        except Exception as e:
            logger.error(f"Response generation failed: {e}")
            state["response_message"] = f"Error generating response: {str(e)}"

        return state

    def _generate_llm_narrative(
        self,
        user_query: str,
        query_type: "QueryType",
        formatted_result: str,
        final_result: Dict[str, Any]
    ) -> Optional[str]:
        """Ask the configured LLM to explain the already-computed result in plain language.

        The LLM never computes numbers itself — it only narrates the deterministic
        output from the pricing/strategy/risk agents, so a bad LLM response can't
        corrupt the underlying math.
        """
        if not self.llm:
            return None

        try:
            system_prompt = (
                "You are an options trading analyst assistant. You will be given the user's "
                "question and a pre-computed, verified result. Rewrite it as a clear, concise "
                "explanation for the user. Do NOT invent, recompute, or alter any numbers — "
                "use only the figures provided. Do not add new disclaimers; those are appended "
                "separately."
            )
            human_prompt = (
                f"User question: {user_query}\n\n"
                f"Query type: {query_type.value}\n\n"
                f"Computed result (ground truth, do not alter numbers):\n{formatted_result}\n\n"
                f"Raw structured data:\n{json.dumps(final_result, default=str)[:3000]}"
            )

            response = self.llm.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=human_prompt)
            ])

            content = getattr(response, "content", None) or str(response)
            return content.strip() or None

        except Exception as e:
            logger.error(f"LLM narrative generation failed, falling back to template: {e}")
            return None

    def _judge_response_node(self, state: AgentState) -> AgentState:
        """Independent cross-provider check on educational/low-confidence responses.

        Uses OpenAI (a different provider than the one that generated the response)
        to judge whether the response is consistent with established options theory.
        Runs only for educational queries or when grounding confidence is already low —
        running it on every routine pricing query would add cost/latency for cases
        where there's little to hallucinate (a Black-Scholes number is just a number).
        Skips silently (no pipeline failure) if OPENAI_API_KEY isn't configured.
        """
        try:
            query_type = state["query_type"]
            confidence = state.get("confidence_score", 1.0)
            should_judge = (
                query_type == QueryType.EDUCATIONAL or confidence < settings.CONFIDENCE_THRESHOLD
            )

            if not should_judge or not state.get("response_message"):
                return state

            if not settings.OPENAI_API_KEY or settings.OPENAI_API_KEY.startswith("<"):
                logger.debug("Skipping LLM-as-judge check: OPENAI_API_KEY not configured")
                return state

            grounding = state.get("knowledge_grounding")
            grounding_context = "\n".join(
                item["content"] for item in (grounding.retrieved_knowledge if grounding else [])
            ) or "No supporting reference material was retrieved for this query."

            verdict = self._run_llm_judge(state["user_query"], state["response_message"], grounding_context)

            if verdict:
                state["judge_verdict"] = verdict
                if not verdict.get("grounded", True) or verdict.get("concerns"):
                    review_lines = [
                        "",
                        f"🔍 Independent Review (OpenAI, confidence: {verdict.get('confidence', 0):.0%}):"
                    ]
                    for concern in verdict.get("concerns", []):
                        review_lines.append(f"• {concern}")
                    state["response_message"] += "\n" + "\n".join(review_lines)

            state["processing_steps"].append("LLM-as-judge review completed")

        except Exception as e:
            logger.warning(f"LLM-as-judge review failed, continuing without it: {e}")

        return state

    def _run_llm_judge(self, query: str, response_text: str, grounding_context: str) -> Optional[Dict[str, Any]]:
        """Ask an independent OpenAI model to fact-check the response against known theory"""
        try:
            judge_llm = OpenAIProvider().get_llm("validation")

            system_prompt = (
                "You are an independent fact-checker for an options-trading assistant. "
                "You did NOT write the response being checked. Compare it only against "
                "the reference material provided and well-established options theory. "
                "Respond with ONLY a JSON object, no other text, matching exactly: "
                '{"grounded": <bool>, "confidence": <float 0-1>, "concerns": [<string>, ...]}. '
                "\"concerns\" should be empty if you find no issues."
            )
            human_prompt = (
                f"User question: {query}\n\n"
                f"Reference material:\n{grounding_context}\n\n"
                f"Response to check:\n{response_text}"
            )

            result = judge_llm.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=human_prompt)
            ])

            content = getattr(result, "content", None) or str(result)
            match = re.search(r"\{.*\}", content, re.DOTALL)
            if not match:
                logger.warning(f"LLM judge did not return parseable JSON: {content[:200]}")
                return None

            return json.loads(match.group(0))

        except Exception as e:
            logger.warning(f"LLM judge call failed: {e}")
            return None

    # def _format_result_for_response(self, result: Dict[str, Any], query_type: QueryType) -> str:
    #     """Format result based on query type"""
        
    #     if query_type == QueryType.OPTION_PRICING:
    #         return (
    #             f"📊 Option Pricing Analysis:\n"
    #             f"• Option Price: ${result.get('option_price', 'N/A')}\n"
    #             f"• Intrinsic Value: ${result.get('intrinsic_value', 'N/A')}\n"
    #             f"• Time Value: ${result.get('time_value', 'N/A')}\n"
    #             f"• Underlying: ${result.get('underlying_price', 'N/A')}\n"
    #             f"• Strike: ${result.get('strike_price', 'N/A')}\n"
    #             f"• Type: {result.get('option_type', 'N/A').title()}"
    #         )
        
    #     elif query_type == QueryType.GREEKS_ANALYSIS:
    #         greeks = result.get("greeks", {})
    #         return (
    #             f"📈 Greeks Analysis:\n"
    #             f"• Delta: {greeks.get('delta', 'N/A')}\n"
    #             f"• Gamma: {greeks.get('gamma', 'N/A')}\n"
    #             f"• Theta: {greeks.get('theta', 'N/A')}\n"
    #             f"• Vega: {greeks.get('vega', 'N/A')}\n"
    #             f"• Rho: {greeks.get('rho', 'N/A')}"
    #         )
        
    #     elif query_type == QueryType.VOLATILITY_ANALYSIS:
    #         vol = result.get("historical_volatility", 0)
    #         return (
    #             f"📊 Volatility Analysis:\n"
    #             f"• Historical Volatility: {vol:.1%}\n"
    #             f"• Period: {result.get('period_days', 'N/A')} days\n"
    #             f"• Interpretation: {result.get('volatility_interpretation', 'N/A')}"
    #         )
        
    #     elif query_type == QueryType.STRATEGY_ANALYSIS:
    #         return (
    #             f"🎯 Strategy Analysis:\n"
    #             f"• Strategy: {result.get('strategy_name', 'N/A').title()}\n"
    #             f"• Max Profit: {result.get('max_profit', 'N/A')}\n"
    #             f"• Max Loss: {result.get('max_loss', 'N/A')}\n"
    #             f"• Market Outlook: {result.get('market_outlook', 'N/A')}"
    #         )
        
    #     elif query_type == QueryType.EDUCATIONAL:
    #         return (
    #             f"📚 Educational Content:\n"
    #             f"{result.get('explanation', 'No explanation available')}"
    #         )
        
    #     elif query_type == QueryType.STRATEGY_ANALYSIS:
    #        return (
    #            f"🎯 {result.get('strategy_name', 'Strategy')} Analysis:\n"
    #            f"• Net Premium: ${result.get('net_premium', 0):.2f}\n"
    #            f"• Max Profit: {result.get('max_profit', 'N/A')}\n"
    #            f"• Max Loss: {result.get('max_loss', 'N/A')}\n"
    #            f"• Breakeven Points: {[f'${bp:.2f}' for bp in result.get('breakeven_points', [])]}\n"
    #            f"• Probability of Profit: {result.get('profit_probability', 0):.1%}\n"
    #            f"• Market Bias: {result.get('market_bias', 'N/A').title()}\n"
    #            f"• Complexity: {result.get('complexity_rating', 'N/A')}/5\n"
    #            f"• Portfolio Delta: {result.get('portfolio_greeks', {}).get('delta', 0):.2f}"
    #        )
       
    #     elif query_type == QueryType.RISK_MANAGEMENT:
    #         risk_metrics = result.get('risk_metrics', {})
    #         portfolio = result.get('portfolio_summary', {})
            
    #         return (
    #             f"⚠️ Portfolio Risk Analysis:\n"
    #             f"• Portfolio Value: ${portfolio.get('total_portfolio_value', 0):,.0f}\n"
    #             f"• Number of Positions: {portfolio.get('number_of_positions', 0)}\n"
    #             f"• 1-Day VaR: ${risk_metrics.get('var_1day', 0):,.0f}\n"
    #             f"• Portfolio Delta: {risk_metrics.get('total_delta', 0):,.0f}\n"
    #             f"• Portfolio Theta: ${risk_metrics.get('total_theta', 0):,.0f}/day\n"
    #             f"• Top Concentration: {portfolio.get('concentration_top3', 0):.1%}\n"
    #             f"• Hedge Recommendations: {len(result.get('hedge_recommendations', []))}\n"
    #             f"• Risk Warnings: {len(result.get('risk_warnings', []))}"
    #         )
        
    #     else:
    #         return str(result)
    

    def _format_result_for_response(self, result: Dict[str, Any], query_type: QueryType) -> str:
        """Enhanced result formatting"""
        
        if query_type == QueryType.OPTION_PRICING:
            if "error" in result:
                return f"❌ Error: {result['error']}"
            
            symbol = result.get("inputs", {}).get("symbol", "N/A")
            method = result.get("pricing_method", "Unknown")
            
            return (
                f"📊 Option Pricing Analysis for {symbol} ({method}):\n"
                f"• Option Price: ${result.get('option_price', 0):.4f}\n"
                f"• Intrinsic Value: ${result.get('intrinsic_value', 0):.4f}\n"
                f"• Time Value: ${result.get('time_value', 0):.4f}\n"
                f"• Moneyness: {result.get('moneyness', 0):.4f}\n"
                f"• Current Stock Price: ${result.get('inputs', {}).get('spot_price', 0):.2f}\n"
                f"• Strike Price: ${result.get('inputs', {}).get('strike_price', 0):.2f}\n"
                f"• Option Type: {result.get('inputs', {}).get('option_type', 'N/A').upper()}\n"
                f"• Volatility: {result.get('inputs', {}).get('volatility', 0)*100:.1f}%\n"
                f"• Time to Expiry: {result.get('inputs', {}).get('time_to_expiry', 0)*365:.0f} days\n"
                f"• Greeks: Δ={result.get('greeks', {}).get('delta', 0):.4f}, "
                f"Γ={result.get('greeks', {}).get('gamma', 0):.4f}, "
                f"Θ={result.get('greeks', {}).get('theta', 0):.4f}"
            )
        
        elif query_type == QueryType.GREEKS_ANALYSIS:
            if "error" in result:
                return f"❌ Error: {result['error']}"
            
            greeks = result.get("greeks", {})
            interpretations = result.get("interpretations", {})
            return (
                f"📈 Greeks Analysis:\n"
                f"• Delta: {greeks.get('delta', 'N/A'):.4f} - {interpretations.get('delta', '')}\n"
                f"• Gamma: {greeks.get('gamma', 'N/A'):.4f} - {interpretations.get('gamma', '')}\n"
                f"• Theta: {greeks.get('theta', 'N/A'):.4f} - {interpretations.get('theta', '')}\n"
                f"• Vega: {greeks.get('vega', 'N/A'):.4f} - {interpretations.get('vega', '')}\n"
                f"• Rho: {greeks.get('rho', 'N/A'):.4f} - {interpretations.get('rho', '')}"
            )
        
        elif query_type == QueryType.VOLATILITY_ANALYSIS:
            if "error" in result:
                return f"❌ Error: {result['error']}"
            
            hist_vol = result.get("historical_volatility", {})
            impl_vol = result.get("implied_volatility", {})
            return (
                f"📊 Volatility Analysis for {result.get('symbol', 'N/A')}:\n"
                f"• 30-Day Historical Vol: {hist_vol.get('monthly_volatility', 0)*100:.1f}%\n"
                f"• Current Rolling Vol: {hist_vol.get('current_rolling_vol', 0)*100:.1f}%\n"
                f"• Average Implied Vol: {impl_vol.get('statistics', {}).get('mean_implied_vol', 0)*100:.1f}%\n"
                f"• Volatility Regime: {result.get('regime_analysis', {}).get('current_regime', 'Unknown')}\n"
                f"• Vol Percentile: {result.get('volatility_statistics', {}).get('current_volatility_percentile', 'N/A')}%"
            )
        
        elif query_type == QueryType.EDUCATIONAL:
            return (
                f"📚 Educational Content:\n"
                f"{result.get('explanation', 'No explanation available')}"
            )
        
        else:
            return str(result)


    def _format_validation_summary(self, validation: OutputValidationResult) -> str:
        """Format validation summary"""
        status = "✅ Validated" if validation.is_valid else "⚠️ Validation Issues"
        confidence = f"Confidence: {validation.confidence:.1%}"
        return f"{status} ({confidence})"


    def process_query(self, user_query: str) -> Dict[str, Any]:
        """Main entry point for processing user queries"""
        logger.info(f"Processing query: {user_query[:100]}...")
        
        # Initialize state
        initial_state = AgentState(
            user_query=user_query,
            query_type=QueryType.INVALID,
            parsed_parameters={},
            input_validation=None,
            knowledge_grounding=None,
            output_validation=None,
            market_data=None,
            current_agent="router",
            processing_steps=[],
            intermediate_results={},
            errors=[],
            warnings=[],
            final_result=None,
            response_message="",
            confidence_score=0.0,
            messages=[HumanMessage(content=user_query)]
        )
        
        try:
            # Run the workflow
            final_state = self.workflow.invoke(initial_state)
            
            # Determine success based on final result rather than just errors
            has_valid_result = (
                final_state.get("final_result") and 
                "error" not in final_state.get("final_result", {}) and
                final_state.get("query_type") != QueryType.INVALID
            )
            
            # Success if we have a valid result, even with some warnings/validation issues
            success = has_valid_result and (
                len([e for e in final_state["errors"] if "validation" not in e.lower()]) == 0
            )
            
            # Prepare response
            response = {
                "success": success,
                "response": final_state["response_message"],
                "confidence": final_state["confidence_score"],
                "query_type": final_state["query_type"].value,
                "processing_steps": final_state["processing_steps"],
                "result_data": final_state["final_result"],
                "warnings": final_state["warnings"],
                "errors": final_state["errors"],
                "timestamp": datetime.now().isoformat(),
                "visualization_data": self._prepare_visualization_data(final_state)
            }
            
            logger.info(f"Query processing completed. Success: {response['success']}")
            return response
            
        except Exception as e:
            logger.error(f"Query processing failed: {e}")
            return {
                "success": False,
                "response": f"Processing failed: {str(e)}",
                "confidence": 0.0,
                "query_type": "error",
                "errors": [str(e)],
                "timestamp": datetime.now().isoformat()
            }


    def _prepare_visualization_data(self, final_state: AgentState) -> Dict[str, Any]:
        """Enhanced visualization data preparation"""
        viz_data = {}
        
        try:
            final_result = final_state.get("final_result", {})
            query_type = final_state.get("query_type")
            
            logger.info(f"Preparing visualization data for {query_type}")
            
            if query_type == QueryType.OPTION_PRICING and final_result and "error" not in final_result:
                # Prepare Greeks radar chart data
                greeks = final_result.get("greeks", {})
                if greeks:
                    viz_data["greeks_chart"] = {
                        "type": "radar",
                        "data": {
                            "labels": ["Delta", "Gamma", "Theta", "Vega", "Rho"],
                            "values": [
                                abs(greeks.get("delta", 0)),
                                greeks.get("gamma", 0) * 10,  # Scale gamma
                                abs(greeks.get("theta", 0)) * 10,  # Scale theta
                                greeks.get("vega", 0) / 100,  # Scale vega
                                abs(greeks.get("rho", 0)) / 100  # Scale rho
                            ]
                        }
                    }
                
                # Prepare payoff diagram data
                inputs = final_result.get("inputs", {})
                if inputs:
                    viz_data["payoff_diagram"] = self._generate_payoff_data(inputs, final_result)
            
            elif query_type == QueryType.VOLATILITY_ANALYSIS:
                # Prepare volatility charts
                hist_vol = final_result.get("historical_volatility", {})
                viz_data["volatility_chart"] = {
                    "type": "line",
                    "title": f"Volatility Analysis - {final_result.get('symbol', 'Unknown')}",
                    "data": self._generate_volatility_chart_data(final_result)
                }
                
                # Prepare volatility surface if available
                if "volatility_surface" in final_result:
                    viz_data["volatility_surface"] = self._generate_volatility_surface_data(final_result)
            
            logger.info(f"Prepared {len(viz_data)} visualization data sets")
            return viz_data
            
        except Exception as e:
            logger.warning(f"Visualization data preparation failed: {e}")
            return {}


    def _generate_payoff_data(self, inputs: Dict[str, Any], result: Dict[str, Any]) -> Dict[str, Any]:
        """Generate payoff diagram data"""
        try:
            spot_price = inputs.get("spot_price", 100)
            strike_price = inputs.get("strike_price", 100)
            option_type = inputs.get("option_type", "call")
            option_price = result.get("option_price", 0)
            
            # Generate spot price range
            price_range = np.linspace(spot_price * 0.7, spot_price * 1.3, 100)
            
            # Calculate option value at expiration
            if option_type.lower() == "call":
                payoffs = np.maximum(price_range - strike_price, 0) - option_price
            else:
                payoffs = np.maximum(strike_price - price_range, 0) - option_price
            
            return {
                "x": price_range.tolist(),
                "y": payoffs.tolist(),
                "current_price": spot_price,
                "strike_price": strike_price,
                "option_type": option_type,
                "breakeven": self._calculate_breakeven(strike_price, option_price, option_type)
            }
            
        except Exception as e:
            logger.warning(f"Payoff data generation failed: {e}")
            return {}
    
    def _calculate_breakeven(self, strike: float, premium: float, option_type: str) -> float:
        """Calculate breakeven point"""
        if option_type.lower() == "call":
            return strike + premium
        else:
            return strike - premium
    
    def _generate_volatility_chart_data(self, vol_result: Dict[str, Any]) -> Dict[str, Any]:
        """Generate volatility chart data"""
        try:
            hist_vol = vol_result.get("historical_volatility", {})
            impl_vol = vol_result.get("implied_volatility", {})
            
            # Create comparison data
            categories = []
            values = []
            
            if "monthly_volatility" in hist_vol:
                categories.append("30-Day Historical")
                values.append(hist_vol["monthly_volatility"] * 100)
            
            if "quarterly_volatility" in hist_vol:
                categories.append("90-Day Historical")
                values.append(hist_vol["quarterly_volatility"] * 100)
            
            if impl_vol.get("statistics", {}).get("mean_implied_vol"):
                categories.append("Implied Volatility")
                values.append(impl_vol["statistics"]["mean_implied_vol"] * 100)
            
            return {
                "categories": categories,
                "values": values,
                "title": "Volatility Comparison (%)"
            }
            
        except Exception as e:
            logger.warning(f"Volatility chart data generation failed: {e}")
            return {}
    
    def _generate_volatility_surface_data(self, vol_result: Dict[str, Any]) -> Dict[str, Any]:
        """Generate volatility surface visualization data"""
        try:
            surface_data = vol_result.get("volatility_surface", {}).get("surface_by_moneyness", {})
            
            if not surface_data:
                return {}
            
            moneyness_labels = []
            volatilities = []
            
            moneyness_order = ["deep_otm", "otm", "atm", "itm", "deep_itm"]
            
            for moneyness in moneyness_order:
                if moneyness in surface_data:
                    moneyness_labels.append(moneyness.replace("_", " ").title())
                    volatilities.append(surface_data[moneyness]["avg_implied_vol"] * 100)
            
            return {
                "x": moneyness_labels,
                "y": volatilities,
                "title": "Implied Volatility Surface",
                "ylabel": "Implied Volatility (%)"
            }
            
        except Exception as e:
            logger.warning(f"Volatility surface data generation failed: {e}")
            return {}

# Usage example and test
if __name__ == "__main__":
    router = RouterAgent()
    
    test_queries = [
        "Price a call option on AAPL with strike $150, expiry in 30 days",
        "What is delta hedging?",
        "Calculate historical volatility for GOOGL",
        "Explain put-call parity",
        "What's the weather like today?"  # Should be invalid
    ]
    
    for query in test_queries:
        print(f"\n{'='*80}")
        print(f"Query: {query}")
        print('='*80)
        
        response = router.process_query(query)
        
        print(f"Success: {response['success']}")
        print(f"Query Type: {response['query_type']}")
        print(f"Confidence: {response['confidence']:.1%}")
        print(f"\nResponse:")
        print(response['response'])
        
        if response['errors']:
            print(f"\nErrors: {response['errors']}")
        
        if response['warnings']:
            print(f"\nWarnings: {response['warnings'][:2]}")
        
        print(f"\nProcessing Steps: {' -> '.join(response['processing_steps'])}")