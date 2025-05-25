from typing import Dict, List, Tuple, Optional, Any, Union
from pydantic import BaseModel, Field
import numpy as np
import faiss
import pickle
import json
import os
from sentence_transformers import SentenceTransformer
from loguru import logger
from pathlib import Path

from ...knowledge.options_theory import OPTIONS_THEORY, get_theory_context
from ...config.settings import settings
from ...utils.constants import PRICING_BOUNDS, ERROR_MESSAGES

class GroundingResult(BaseModel):
    """Result of knowledge grounding operation"""
    is_grounded: bool
    confidence: float = Field(ge=0.0, le=1.0)
    retrieved_knowledge: List[Dict[str, Any]] = []
    theory_validation: Dict[str, Any] = {}
    warnings: List[str] = []
    supporting_formulas: List[str] = []

class KnowledgeEntry(BaseModel):
    """Structure for knowledge base entries"""
    id: str
    content: str
    category: str
    subcategory: Optional[str] = None
    formulas: List[str] = []
    assumptions: List[str] = []
    limitations: List[str] = []
    examples: List[str] = []
    embedding: Optional[List[float]] = None

class KnowledgeGroundingAgent:
    def __init__(self):
        self.embedding_model = None
        self.faiss_index = None
        self.knowledge_entries = []
        self.category_mapping = {}
        
        # Initialize embedding model
        self._load_embedding_model()
        
        # Initialize or load knowledge base
        self._initialize_knowledge_base()
    
    def _load_embedding_model(self):
        """Load sentence transformer model for embeddings"""
        try:
            logger.info(f"Loading embedding model: {settings.EMBEDDING_MODEL}")
            self.embedding_model = SentenceTransformer(settings.EMBEDDING_MODEL)
            logger.success("Embedding model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise
    
    def _initialize_knowledge_base(self):
        """Initialize or load existing knowledge base"""
        kb_path = Path(settings.KNOWLEDGE_BASE_PATH)
        faiss_path = Path(settings.FAISS_INDEX_PATH)
        
        # Create directories if they don't exist
        kb_path.mkdir(parents=True, exist_ok=True)
        faiss_path.parent.mkdir(parents=True, exist_ok=True)
        
        entries_file = kb_path / "knowledge_entries.pkl"
        mapping_file = kb_path / "category_mapping.json"
        
        if (faiss_path.with_suffix('.index').exists() and 
            entries_file.exists() and mapping_file.exists()):
            logger.info("Loading existing knowledge base...")
            self._load_knowledge_base()
        else:
            logger.info("Creating new knowledge base...")
            self._create_knowledge_base()
    
    def _create_knowledge_base(self):
        """Create knowledge base from OPTIONS_THEORY"""
        logger.info("Building knowledge base from options theory...")
        
        knowledge_entries = []
        entry_id = 0
        
        # Process Black-Scholes information
        bs_info = OPTIONS_THEORY['black_scholes']
        knowledge_entries.append(KnowledgeEntry(
            id=f"bs_{entry_id}",
            content=f"Black-Scholes Model: {bs_info['description']}. Formula: {bs_info['formula']}",
            category="pricing_models",
            subcategory="black_scholes",
            formulas=[bs_info['formula']],
            assumptions=bs_info['assumptions'],
            limitations=bs_info['limitations']
        ))
        entry_id += 1
        
        # Add parameter explanations
        for param, description in bs_info['parameters'].items():
            knowledge_entries.append(KnowledgeEntry(
                id=f"param_{entry_id}",
                content=f"Black-Scholes parameter {param}: {description}",
                category="parameters",
                subcategory="black_scholes"
            ))
            entry_id += 1
        
        # Process Greeks information
        for greek_name, greek_info in OPTIONS_THEORY['greeks'].items():
            knowledge_entries.append(KnowledgeEntry(
                id=f"greek_{entry_id}",
                content=f"{greek_name.capitalize()}: {greek_info['definition']}. {greek_info['interpretation']}",
                category="greeks",
                subcategory=greek_name,
                examples=[f"Range: {greek_info.get('range', greek_info.get('range_call', 'Variable'))}"]
            ))
            entry_id += 1
        
        # Process option strategies
        for strategy_name, strategy_info in OPTIONS_THEORY['option_strategies'].items():
            content = f"{strategy_name.replace('_', ' ').title()}: {strategy_info['description']}"
            if 'construction' in strategy_info:
                content += f" Construction: {strategy_info['construction']}"
            
            knowledge_entries.append(KnowledgeEntry(
                id=f"strategy_{entry_id}",
                content=content,
                category="strategies",
                subcategory=strategy_name,
                examples=[
                    f"Max Profit: {strategy_info['max_profit']}",
                    f"Max Loss: {strategy_info['max_loss']}"
                ]
            ))
            entry_id += 1
        
        # Add risk management concepts
        risk_concepts = [
            {
                "content": "Delta hedging involves maintaining a delta-neutral portfolio by adjusting positions as the underlying price changes",
                "category": "risk_management",
                "subcategory": "delta_hedging"
            },
            {
                "content": "Implied volatility represents the market's expectation of future volatility embedded in option prices",
                "category": "volatility",
                "subcategory": "implied_volatility"
            },
            {
                "content": "Put-call parity relationship: C - P = S - K*e^(-r*T) for European options",
                "category": "arbitrage",
                "subcategory": "put_call_parity",
                "formulas": ["C - P = S - K*e^(-r*T)"]
            },
            {
                "content": "Time decay (theta) accelerates as options approach expiration, especially for at-the-money options",
                "category": "time_decay",
                "subcategory": "theta_behavior"
            }
        ]
        
        for concept in risk_concepts:
            knowledge_entries.append(KnowledgeEntry(
                id=f"concept_{entry_id}",
                **concept
            ))
            entry_id += 1
        
        # Add validation rules
        validation_rules = [
            {
                "content": f"Option prices must be non-negative and at least intrinsic value",
                "category": "validation",
                "subcategory": "price_bounds"
            },
            {
                "content": f"Volatility should typically be between {PRICING_BOUNDS['min_volatility']*100:.1f}% and {PRICING_BOUNDS['max_volatility']*100:.0f}%",
                "category": "validation", 
                "subcategory": "volatility_bounds"
            },
            {
                "content": f"Call delta ranges from 0 to 1, put delta ranges from -1 to 0",
                "category": "validation",
                "subcategory": "delta_bounds"
            }
        ]
        
        for rule in validation_rules:
            knowledge_entries.append(KnowledgeEntry(
                id=f"validation_{entry_id}",
                **rule
            ))
            entry_id += 1
        
        self.knowledge_entries = knowledge_entries
        
        # Generate embeddings and build FAISS index
        self._build_embeddings_and_index()
        
        # Save knowledge base
        self._save_knowledge_base()
        
        logger.success(f"Knowledge base created with {len(knowledge_entries)} entries")
    
    def _build_embeddings_and_index(self):
        """Generate embeddings and build FAISS index"""
        logger.info("Generating embeddings...")
        
        # Extract content for embedding
        contents = [entry.content for entry in self.knowledge_entries]
        
        # Generate embeddings
        embeddings = self.embedding_model.encode(contents, show_progress_bar=True)
        
        # Store embeddings in entries
        for i, entry in enumerate(self.knowledge_entries):
            entry.embedding = embeddings[i].tolist()
        
        # Build FAISS index
        dimension = embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        self.faiss_index.add(embeddings.astype(np.float32))
        
        # Build category mapping for faster filtering
        self.category_mapping = {}
        for i, entry in enumerate(self.knowledge_entries):
            category = entry.category
            if category not in self.category_mapping:
                self.category_mapping[category] = []
            self.category_mapping[category].append(i)
        
        logger.success(f"FAISS index built with {self.faiss_index.ntotal} vectors")
    
    def _save_knowledge_base(self):
        """Save knowledge base to disk"""
        kb_path = Path(settings.KNOWLEDGE_BASE_PATH)
        faiss_path = Path(settings.FAISS_INDEX_PATH)
        
        # Save FAISS index
        faiss.write_index(self.faiss_index, str(faiss_path.with_suffix('.index')))
        
        # Save knowledge entries
        with open(kb_path / "knowledge_entries.pkl", 'wb') as f:
            pickle.dump(self.knowledge_entries, f)
        
        # Save category mapping
        with open(kb_path / "category_mapping.json", 'w') as f:
            json.dump(self.category_mapping, f, indent=2)
        
        logger.info("Knowledge base saved successfully")
    
    def _load_knowledge_base(self):
        """Load existing knowledge base"""
        kb_path = Path(settings.KNOWLEDGE_BASE_PATH)
        faiss_path = Path(settings.FAISS_INDEX_PATH)
        
        try:
            # Load FAISS index
            self.faiss_index = faiss.read_index(str(faiss_path.with_suffix('.index')))
            
            # Load knowledge entries
            with open(kb_path / "knowledge_entries.pkl", 'rb') as f:
                self.knowledge_entries = pickle.load(f)
            
            # Load category mapping
            with open(kb_path / "category_mapping.json", 'r') as f:
                self.category_mapping = json.load(f)
            
            logger.success(f"Knowledge base loaded with {len(self.knowledge_entries)} entries")
            
        except Exception as e:
            logger.error(f"Failed to load knowledge base: {e}")
            logger.info("Creating new knowledge base...")
            self._create_knowledge_base()
    
    def ground_query(self, query: str, top_k: int = 5, category_filter: Optional[str] = None) -> GroundingResult:
        """Ground query against knowledge base"""
        logger.info(f"Grounding query: {query[:100]}...")
        
        try:
            # Generate query embedding
            query_embedding = self.embedding_model.encode([query])
            faiss.normalize_L2(query_embedding)
            
            # Search FAISS index
            if category_filter and category_filter in self.category_mapping:
                # Filter by category
                candidate_indices = self.category_mapping[category_filter]
                candidate_embeddings = np.array([
                    self.knowledge_entries[i].embedding for i in candidate_indices
                ])
                
                # Manual similarity calculation for filtered search
                similarities = np.dot(query_embedding, candidate_embeddings.T)[0]
                top_indices = np.argsort(similarities)[::-1][:top_k]
                scores = similarities[top_indices]
                indices = [candidate_indices[i] for i in top_indices]
            else:
                # Full search
                scores, indices = self.faiss_index.search(query_embedding.astype(np.float32), top_k)
                scores = scores[0]
                indices = indices[0]
            
            # Build result
            retrieved_knowledge = []
            supporting_formulas = []
            
            for i, (idx, score) in enumerate(zip(indices, scores)):
                if idx < len(self.knowledge_entries):
                    entry = self.knowledge_entries[idx]
                    retrieved_knowledge.append({
                        "content": entry.content,
                        "category": entry.category,
                        "subcategory": entry.subcategory,
                        "score": float(score),
                        "rank": i + 1
                    })
                    
                    # Collect formulas
                    if entry.formulas:
                        supporting_formulas.extend(entry.formulas)
            
            # Calculate grounding confidence
            avg_score = np.mean(scores) if len(scores) > 0 else 0.0
            confidence = min(max(avg_score, 0.0), 1.0)
            
            result = GroundingResult(
                is_grounded=confidence >= settings.CONFIDENCE_THRESHOLD,
                confidence=confidence,
                retrieved_knowledge=retrieved_knowledge,
                supporting_formulas=list(set(supporting_formulas))  # Remove duplicates
            )
            
            # Add warnings for low confidence
            if confidence < settings.CONFIDENCE_THRESHOLD:
                result.warnings.append(
                    f"Low confidence grounding (confidence: {confidence:.2f}). "
                    "Results may not be fully supported by established theory."
                )
            
            logger.info(f"Grounding completed. Confidence: {confidence:.2f}")
            return result
            
        except Exception as e:
            logger.error(f"Grounding failed: {e}")
            return GroundingResult(
                is_grounded=False,
                confidence=0.0,
                warnings=[f"Knowledge grounding failed: {str(e)}"]
            )
    
    def validate_calculation_result(self, calculation_type: str, inputs: Dict[str, Any], result: Any) -> Dict[str, Any]:
        """Validate calculation results against known bounds and relationships"""
        validation = {
            "is_valid": True,
            "warnings": [],
            "bounds_check": True,
            "theory_consistency": True
        }
        
        try:
            if calculation_type == "option_price":
                # Validate option pricing results
                if result <= 0:
                    validation["is_valid"] = False
                    validation["warnings"].append("Option price must be positive")
                
                # Check against intrinsic value
                if "strike" in inputs and "spot_price" in inputs and "option_type" in inputs:
                    if inputs["option_type"].lower() == "call":
                        intrinsic = max(0, inputs["spot_price"] - inputs["strike"])
                    else:
                        intrinsic = max(0, inputs["strike"] - inputs["spot_price"])
                    
                    if result < intrinsic * 0.99:  # Small tolerance for numerical precision
                        validation["is_valid"] = False
                        validation["warnings"].append(
                            f"Option price ({result:.4f}) below intrinsic value ({intrinsic:.4f})"
                        )
            
            elif calculation_type == "greeks":
                # Validate Greeks bounds
                if "delta" in result:
                    delta = result["delta"]
                    if "option_type" in inputs:
                        if inputs["option_type"].lower() == "call" and not (0 <= delta <= 1):
                            validation["warnings"].append(f"Call delta ({delta:.4f}) outside normal range [0,1]")
                        elif inputs["option_type"].lower() == "put" and not (-1 <= delta <= 0):
                            validation["warnings"].append(f"Put delta ({delta:.4f}) outside normal range [-1,0]")
                
                if "gamma" in result and result["gamma"] < 0:
                    validation["warnings"].append("Gamma should be non-negative")
                
                if "vega" in result and result["vega"] < 0:
                    validation["warnings"].append("Vega should be non-negative")
            
            elif calculation_type == "volatility":
                # Validate volatility bounds
                if result < PRICING_BOUNDS["min_volatility"] or result > PRICING_BOUNDS["max_volatility"]:
                    validation["warnings"].append(
                        f"Volatility ({result:.1%}) outside typical range "
                        f"[{PRICING_BOUNDS['min_volatility']:.1%}, {PRICING_BOUNDS['max_volatility']:.1%}]"
                    )
            
            # Check for NaN or infinite values
            if isinstance(result, (int, float)) and (np.isnan(result) or np.isinf(result)):
                validation["is_valid"] = False
                validation["warnings"].append("Calculation produced invalid numerical result")
            
        except Exception as e:
            logger.error(f"Validation error: {e}")
            validation["is_valid"] = False
            validation["warnings"].append(f"Validation failed: {str(e)}")
        
        return validation
    
    def get_theory_explanation(self, topic: str) -> Optional[Dict[str, Any]]:
        """Get detailed theory explanation for a topic"""
        # Direct lookup in OPTIONS_THEORY
        theory_data = get_theory_context(topic)
        if theory_data:
            return theory_data
        
        # Search in knowledge base
        grounding_result = self.ground_query(topic, top_k=3, category_filter="theory")
        if grounding_result.is_grounded:
            return {
                "explanation": grounding_result.retrieved_knowledge[0]["content"],
                "supporting_knowledge": grounding_result.retrieved_knowledge,
                "formulas": grounding_result.supporting_formulas
            }
        
        return None
    
    def get_risk_warnings(self, context: str = "general") -> List[str]:
        """Get appropriate risk warnings for context"""
        base_warnings = OPTIONS_THEORY["risk_warnings"]
        
        # Add context-specific warnings
        context_warnings = {
            "pricing": [
                "Model assumptions may not reflect current market conditions",
                "Actual bid-ask spreads may significantly impact real trading results"
            ],
            "strategy": [
                "Strategy performance depends heavily on market conditions",
                "Transaction costs can significantly impact profitability"
            ],
            "greeks": [
                "Greeks change dynamically with market conditions",
                "Higher-order Greeks become important for large price movements"
            ]
        }
        
        warnings = base_warnings.copy()
        if context in context_warnings:
            warnings.extend(context_warnings[context])
        
        return warnings

# Usage example and test
if __name__ == "__main__":
    # Initialize knowledge grounding agent
    grounding_agent = KnowledgeGroundingAgent()
    
    # Test queries
    test_queries = [
        "How to calculate Black-Scholes option price?",
        "What is delta hedging?",
        "Explain put-call parity",
        "Random non-financial query"
    ]
    
    for query in test_queries:
        print(f"\n{'='*50}")
        print(f"Query: {query}")
        
        result = grounding_agent.ground_query(query)
        print(f"Grounded: {result.is_grounded}")
        print(f"Confidence: {result.confidence:.3f}")
        print(f"Retrieved knowledge: {len(result.retrieved_knowledge)} entries")
        
        if result.retrieved_knowledge:
            for i, knowledge in enumerate(result.retrieved_knowledge[:2]):
                print(f"  {i+1}. [{knowledge['category']}] {knowledge['content'][:100]}...")
        
        if result.supporting_formulas:
            print(f"Formulas: {result.supporting_formulas}")
    
    # Test validation
    print(f"\n{'='*50}")
    print("Testing validation...")
    
    # Valid option price
    validation = grounding_agent.validate_calculation_result(
        "option_price",
        {"spot_price": 100, "strike": 95, "option_type": "call"},
        7.5
    )
    print(f"Valid call price: {validation}")
    
    # Invalid option price (below intrinsic)
    validation = grounding_agent.validate_calculation_result(
        "option_price", 
        {"spot_price": 100, "strike": 95, "option_type": "call"},
        3.0  # Should be at least 5.0
    )
    print(f"Invalid call price: {validation}")