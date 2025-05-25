from typing import Dict, List, Any, Optional, TypedDict, Annotated, Sequence
from pydantic import BaseModel, Field
from datetime import datetime
import json
from loguru import logger
from enum import Enum

from langgraph.graph import StateGraph, END
# from langgraph.prebuilt import ToolExecutor
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_community.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
import numpy as np

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
        """Initialize the local LLM for text generation"""
        try:
            logger.info(f"Loading LLM: {settings.HF_MODEL_NAME}")
            
            # Use a more suitable model for financial tasks
            model_name = "microsoft/DialoGPT-medium"
            
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None
            )
            
            # Create pipeline
            pipe = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                max_length=512,
                temperature=0.7,
                do_sample=True,
                device=0 if torch.cuda.is_available() else -1,
                pad_token_id=tokenizer.eos_token_id
            )
            
            llm = HuggingFacePipeline(pipeline=pipe)
            logger.success("LLM loaded successfully")
            return llm
            
        except Exception as e:
            logger.error(f"Failed to load LLM: {e}")
            logger.info("Using fallback text generation")
            return None
            
        except Exception as e:
            logger.error(f"Failed to load LLM: {e}")
            logger.info("Using fallback text generation")
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
        
        # Define the workflow edges
        workflow.set_entry_point("validate_input")
        
        workflow.add_edge("validate_input", "classify_query")
        workflow.add_edge("classify_query", "ground_knowledge") 
        workflow.add_edge("ground_knowledge", "fetch_market_data")
        workflow.add_edge("fetch_market_data", "route_to_specialist")
        workflow.add_edge("route_to_specialist", "validate_output")
        workflow.add_edge("validate_output", "generate_response")
        workflow.add_edge("generate_response", END)
        
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
                # Get current stock data
                stock_result = self.market_data_agent.get_stock_data(symbol)
                if stock_result.success:
                    market_data["stock_data"] = stock_result.data

                    # Get historical data for volatility calculations
                    query_type = state["query_type"]
                    if query_type in [QueryType.OPTION_PRICING, QueryType.VOLATILITY_ANALYSIS, QueryType.GREEKS_ANALYSIS]:
                        hist_result = self.market_data_agent.get_historical_data(symbol, period="1y")
                        if hist_result.success:
                            market_data["historical_data"] = hist_result.data
                        
                        vol_result = self.market_data_agent.calculate_historical_volatility(symbol, days=30)
                        if vol_result.success:
                            market_data["volatility_data"] = vol_result.data
                    
                    # Get options chain if needed
                    if query_type in [QueryType.OPTION_PRICING, QueryType.STRATEGY_ANALYSIS, QueryType.ARBITRAGE_DETECTION]:
                        options_result = self.market_data_agent.get_options_chain(symbol)
                        if options_result.success:
                            market_data["options_data"] = options_result.data
                else:
                    state["warnings"].append(f"Could not fetch market data for {symbol}")
            
            # Always get risk-free rate
            market_data["risk_free_rate"] = self.market_data_agent.get_risk_free_rate()
            
            # Get dividend yield if symbol available
            if symbol:
                market_data["dividend_yield"] = self.market_data_agent.get_dividend_yield(symbol)
            
            state["market_data"] = market_data
            state["current_agent"] = "market_data"
            state["processing_steps"].append("Market data fetched")
            
            logger.info(f"Market data fetched for {symbol if symbol else 'general parameters'}")
            
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
            
            if not market_data or not market_data.get("stock_data"):
                return {"error": "Market data required for option pricing"}
            
            stock_data = market_data["stock_data"]
            spot_price = stock_data["current_price"]
            
            # Extract parameters
            strike = parsed_params.get("strike", spot_price)
            option_type = parsed_params.get("option_type", "call")
            
            # Determine time to expiry
            time_to_expiry = self._parse_time_to_expiry(parsed_params)
            
            # Get volatility
            volatility = self._get_volatility_estimate(market_data)
            
            # Get other parameters
            risk_free_rate = market_data.get("risk_free_rate", 0.05)
            dividend_yield = market_data.get("dividend_yield", 0.0)
            
            # Determine which pricing method to use
            if any(word in query for word in ["american", "early exercise"]):
                # Use binomial tree for American options
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
                method = "Binomial Tree (American)"
                
            elif any(word in query for word in ["barrier", "asian", "exotic"]):
                # Use Monte Carlo for exotic options
                inputs = MonteCarloInputs(
                    spot_price=spot_price,
                    strike_price=strike,
                    time_to_expiry=time_to_expiry,
                    risk_free_rate=risk_free_rate,
                    volatility=volatility,
                    dividend_yield=dividend_yield,
                    option_type=option_type,
                    num_simulations=50000,
                    num_steps=int(max(30, time_to_expiry * 252))
                )
                
                # Check for barrier options
                if "barrier" in query:
                    if "up" in query and "out" in query:
                        inputs.barrier_type = "up-and-out"
                        inputs.barrier_level = spot_price * 1.2  # Default barrier
                    elif "down" in query and "out" in query:
                        inputs.barrier_type = "down-and-out"
                        inputs.barrier_level = spot_price * 0.8
                
                # Check for Asian options
                if "asian" in query:
                    inputs.asian_type = "arithmetic"
                
                result = self.monte_carlo_agent.calculate_option_price(inputs)
                method = "Monte Carlo Simulation"
                
            else:
                # Use Black-Scholes for European options
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
                method = "Black-Scholes-Merton"
            
            # Format result
            return {
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
                    "option_type": option_type
                },
                "market_data_quality": stock_data.get("data_quality", "Unknown"),
                "additional_info": getattr(result, 'early_exercise_premium', None)
            }
            
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
            
            # Extract the main result for validation
            if validation_type == "option_price":
                result_to_validate = final_result.get("option_price")
                context = {
                    "spot_price": final_result.get("underlying_price"),
                    "strike": final_result.get("strike_price"),
                    "option_type": final_result.get("option_type")
                }
            elif validation_type == "greeks":
                result_to_validate = final_result.get("greeks", {})
                context = {"option_type": final_result.get("option_type")}
            elif validation_type == "volatility":
                result_to_validate = final_result.get("historical_volatility")
                context = {}
            else:
                result_to_validate = final_result
                context = {}
            
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
                response_parts.append(self._format_result_for_response(final_result, query_type))
            
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
    
    def _format_result_for_response(self, result: Dict[str, Any], query_type: QueryType) -> str:
        """Format result based on query type"""
        
        if query_type == QueryType.OPTION_PRICING:
            return (
                f"📊 Option Pricing Analysis:\n"
                f"• Option Price: ${result.get('option_price', 'N/A')}\n"
                f"• Intrinsic Value: ${result.get('intrinsic_value', 'N/A')}\n"
                f"• Time Value: ${result.get('time_value', 'N/A')}\n"
                f"• Underlying: ${result.get('underlying_price', 'N/A')}\n"
                f"• Strike: ${result.get('strike_price', 'N/A')}\n"
                f"• Type: {result.get('option_type', 'N/A').title()}"
            )
        
        elif query_type == QueryType.GREEKS_ANALYSIS:
            greeks = result.get("greeks", {})
            return (
                f"📈 Greeks Analysis:\n"
                f"• Delta: {greeks.get('delta', 'N/A')}\n"
                f"• Gamma: {greeks.get('gamma', 'N/A')}\n"
                f"• Theta: {greeks.get('theta', 'N/A')}\n"
                f"• Vega: {greeks.get('vega', 'N/A')}\n"
                f"• Rho: {greeks.get('rho', 'N/A')}"
            )
        
        elif query_type == QueryType.VOLATILITY_ANALYSIS:
            vol = result.get("historical_volatility", 0)
            return (
                f"📊 Volatility Analysis:\n"
                f"• Historical Volatility: {vol:.1%}\n"
                f"• Period: {result.get('period_days', 'N/A')} days\n"
                f"• Interpretation: {result.get('volatility_interpretation', 'N/A')}"
            )
        
        elif query_type == QueryType.STRATEGY_ANALYSIS:
            return (
                f"🎯 Strategy Analysis:\n"
                f"• Strategy: {result.get('strategy_name', 'N/A').title()}\n"
                f"• Max Profit: {result.get('max_profit', 'N/A')}\n"
                f"• Max Loss: {result.get('max_loss', 'N/A')}\n"
                f"• Market Outlook: {result.get('market_outlook', 'N/A')}"
            )
        
        elif query_type == QueryType.EDUCATIONAL:
            return (
                f"📚 Educational Content:\n"
                f"{result.get('explanation', 'No explanation available')}"
            )
        
        elif query_type == QueryType.STRATEGY_ANALYSIS:
           return (
               f"🎯 {result.get('strategy_name', 'Strategy')} Analysis:\n"
               f"• Net Premium: ${result.get('net_premium', 0):.2f}\n"
               f"• Max Profit: {result.get('max_profit', 'N/A')}\n"
               f"• Max Loss: {result.get('max_loss', 'N/A')}\n"
               f"• Breakeven Points: {[f'${bp:.2f}' for bp in result.get('breakeven_points', [])]}\n"
               f"• Probability of Profit: {result.get('profit_probability', 0):.1%}\n"
               f"• Market Bias: {result.get('market_bias', 'N/A').title()}\n"
               f"• Complexity: {result.get('complexity_rating', 'N/A')}/5\n"
               f"• Portfolio Delta: {result.get('portfolio_greeks', {}).get('delta', 0):.2f}"
           )
       
        elif query_type == QueryType.RISK_MANAGEMENT:
            risk_metrics = result.get('risk_metrics', {})
            portfolio = result.get('portfolio_summary', {})
            
            return (
                f"⚠️ Portfolio Risk Analysis:\n"
                f"• Portfolio Value: ${portfolio.get('total_portfolio_value', 0):,.0f}\n"
                f"• Number of Positions: {portfolio.get('number_of_positions', 0)}\n"
                f"• 1-Day VaR: ${risk_metrics.get('var_1day', 0):,.0f}\n"
                f"• Portfolio Delta: {risk_metrics.get('total_delta', 0):,.0f}\n"
                f"• Portfolio Theta: ${risk_metrics.get('total_theta', 0):,.0f}/day\n"
                f"• Top Concentration: {portfolio.get('concentration_top3', 0):.1%}\n"
                f"• Hedge Recommendations: {len(result.get('hedge_recommendations', []))}\n"
                f"• Risk Warnings: {len(result.get('risk_warnings', []))}"
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
            
            # Prepare response
            response = {
                "success": len(final_state["errors"]) == 0,
                "response": final_state["response_message"],
                "confidence": final_state["confidence_score"],
                "query_type": final_state["query_type"].value,
                "processing_steps": final_state["processing_steps"],
                "result_data": final_state["final_result"],
                "warnings": final_state["warnings"],
                "errors": final_state["errors"],
                "timestamp": datetime.now().isoformat()
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
       """Prepare data for visualization components"""
       viz_data = {}
       
       try:
           final_result = final_state.get("final_result", {})
           query_type = final_state.get("query_type")
           
           if query_type == QueryType.OPTION_PRICING and "greeks" in final_result:
               # Prepare Greeks radar chart data
               greeks = final_result["greeks"]
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
               if "inputs" in final_result:
                   inputs = final_result["inputs"]
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