#!/usr/bin/env python3
"""
Options AI Agent - Main Application with Visualization Support
Agentic AI system for options trading analysis with guardrails and interactive charts
"""

import asyncio
import sys
from pathlib import Path
from loguru import logger
from typing import Dict, Any
import json
import webbrowser
import os

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.agents.router.router_agent import RouterAgent
from src.visualization.charts import OptionsVisualization
from src.config.settings import settings

class OptionsAIAgentWithVisualization:
    """Main application class for Options AI Agent with visualization support"""
    
    def __init__(self):
        self.router = None
        self.visualizer = OptionsVisualization()
        self._initialize_logging()
        logger.info("Initializing Options AI Agent with Visualization...")
    
    def _initialize_logging(self):
        """Setup logging configuration"""
        logger.remove()  # Remove default handler
        
        # Console logging
        logger.add(
            sys.stdout,
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
            level="INFO"
        )
        
        # File logging
        Path("logs").mkdir(exist_ok=True)
        logger.add(
            "logs/options_ai.log",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
            level="DEBUG",
            rotation="10 MB",
            retention="1 week"
        )
    
    async def initialize(self):
        """Initialize the agent system"""
        try:
            logger.info("Starting Options AI Agent initialization...")
            
            # Create necessary directories
            Path("logs").mkdir(exist_ok=True)
            Path("outputs").mkdir(exist_ok=True)
            Path("visualizations").mkdir(exist_ok=True)
            
            # Initialize router agent
            logger.info("Initializing Enhanced Router Agent...")
            self.router = RouterAgent()
            
            logger.success("✅ Options AI Agent with Visualization initialized successfully!")
            return True
            
        except Exception as e:
            logger.error(f"❌ Initialization failed: {e}")
            return False
    
    def process_query_with_visualization(self, query: str, generate_charts: bool = True) -> Dict[str, Any]:
        """Process a user query and generate visualizations"""
        if not self.router:
            return {
                "success": False,
                "response": "Agent not initialized. Please run initialize() first.",
                "error": "Agent not initialized"
            }
        
        try:
            # Process the query
            result = self.router.process_query(query)
            
            if not result["success"] or not generate_charts:
                return result
            
            # Generate visualizations if data is available
            viz_data = result.get("visualization_data", {})
            result_data = result.get("result_data", {})
            
            if viz_data or result_data:
                charts = self._generate_charts(viz_data, result_data, result["query_type"])
                result["charts"] = charts
                
                # Save charts to files
                chart_files = self._save_charts_to_files(charts, query)
                result["chart_files"] = chart_files
            
            return result
            
        except Exception as e:
            logger.error(f"Query processing with visualization failed: {e}")
            return {
                "success": False,
                "response": f"An error occurred: {str(e)}",
                "error": str(e)
            }
    
    def _generate_charts(self, viz_data: Dict[str, Any], result_data: Dict[str, Any], query_type: str) -> Dict[str, Any]:
        """Generate appropriate charts based on query type and available data"""
        charts = {}
        
        try:
            # Payoff diagram for option pricing
            if query_type == "option_pricing" and viz_data.get("payoff_diagram"):
                charts["payoff"] = self.visualizer.create_payoff_diagram(
                    viz_data, 
                    "Option Payoff Analysis"
                )
            
            # Greeks radar chart
            if viz_data.get("greeks_chart"):
                charts["greeks"] = self.visualizer.create_greeks_radar_chart(
                    viz_data,
                    "Option Greeks Profile"
                )
            
            # Volatility analysis charts
            if query_type == "volatility_analysis":
                if viz_data.get("volatility_chart"):
                    charts["volatility"] = self.visualizer.create_volatility_comparison_chart(
                        viz_data,
                        f"Volatility Analysis"
                    )
                
                if viz_data.get("volatility_surface"):
                    charts["vol_surface"] = self.visualizer.create_volatility_surface(
                        viz_data,
                        "Implied Volatility Surface"
                    )
            
            # Strategy analysis charts
            if query_type == "strategy_analysis":
                if viz_data.get("strategy_payoff"):
                    charts["strategy_payoff"] = self.visualizer.create_strategy_payoff_chart(
                        viz_data,
                        "Strategy Payoff Analysis"
                    )
                
                if viz_data.get("strategy_greeks"):
                    charts["strategy_greeks"] = self.visualizer.create_greeks_radar_chart(
                        viz_data,
                        "Strategy Risk Profile"
                    )
            
            # Risk management charts
            if query_type == "risk_management":
                if viz_data.get("risk_dashboard"):
                    charts["risk_dashboard"] = self.visualizer.create_risk_dashboard(
                        viz_data,
                        "Portfolio Risk Analysis"
                    )
                
                if viz_data.get("stress_tests"):
                    charts["stress_tests"] = self.visualizer.create_stress_test_chart(
                        viz_data,
                        "Stress Test Results"
                    )
            
            # Monte Carlo convergence
            if result_data.get("convergence_data"):
                charts["convergence"] = self.visualizer.create_monte_carlo_convergence_chart(
                    result_data["convergence_data"],
                    "Monte Carlo Simulation Convergence"
                )
            
            logger.info(f"Generated {len(charts)} visualization charts")
            
        except Exception as e:
            logger.error(f"Chart generation failed: {e}")
        
        return charts
    
    def _save_charts_to_files(self, charts: Dict[str, Any], query: str) -> Dict[str, str]:
        """Save charts to HTML files"""
        chart_files = {}
        
        try:
            # Create safe filename from query
            safe_query = "".join(c for c in query[:50] if c.isalnum() or c in (' ', '-', '_')).rstrip()
            safe_query = safe_query.replace(' ', '_')
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            for chart_name, fig in charts.items():
                filename = f"visualizations/{safe_query}_{chart_name}_{timestamp}.html"
                saved_file = self.visualizer.export_chart_to_html(fig, filename)
                if saved_file:
                    chart_files[chart_name] = saved_file
            
            # Create dashboard if multiple charts
            if len(charts) > 1:
                dashboard_file = f"visualizations/{safe_query}_dashboard_{timestamp}.html"
                saved_dashboard = self.visualizer.save_dashboard_as_html(charts, dashboard_file)
                if saved_dashboard:
                    chart_files["dashboard"] = saved_dashboard
            
        except Exception as e:
            logger.error(f"Chart file saving failed: {e}")
        
        return chart_files
    
    def interactive_mode(self):
        """Run in interactive command-line mode with visualization support"""
        print("\n" + "="*80)
        print("🤖 OPTIONS AI AGENT - Interactive Mode with Visualization")
        print("="*80)
        print("Ask questions about options trading, pricing, strategies, and analysis.")
        print("Charts will be generated automatically and opened in your browser.")
        print("Type 'help' for examples, 'quit' or 'exit' to stop.")
        print("-"*80)
        
        while True:
            try:
                # Get user input
                query = input("\n📝 Your question: ").strip()
                
                if not query:
                    continue
                
                # Handle special commands
                if query.lower() in ['quit', 'exit', 'q']:
                    print("\n👋 Thanks for using Options AI Agent!")
                    break
                
                elif query.lower() == 'help':
                    self._show_help()
                    continue
                
                elif query.lower() == 'examples':
                    self._show_examples()
                    continue
                
                elif query.lower().startswith('viz '):
                    # Toggle visualization
                    viz_query = query[4:].strip()
                    if viz_query:
                        self._process_with_visualization(viz_query)
                    continue
                
                # Process the query
                print("\n🔄 Processing your query...")
                result = self.process_query_with_visualization(query, generate_charts=True)
                
                # Display results
                self._display_result_with_charts(result)
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
    
    def _process_with_visualization(self, query: str):
        """Process query with enhanced visualization"""
        try:
            print(f"\n🎨 Processing with enhanced visualization: {query}")
            result = self.process_query_with_visualization(query, generate_charts=True)
            self._display_result_with_charts(result)
            
            # Auto-open dashboard if available
            if result.get("chart_files", {}).get("dashboard"):
                dashboard_path = os.path.abspath(result["chart_files"]["dashboard"])
                print(f"\n🌐 Opening dashboard in browser...")
                webbrowser.open(f"file://{dashboard_path}")
                
        except Exception as e:
            print(f"❌ Enhanced visualization failed: {e}")
    
    def _show_help(self):
        """Show help information"""
        help_text = """
🆘 HELP - What can I help you with?

📊 OPTION PRICING:
   • "Price a call option on AAPL with strike $150, expiry in 30 days"
   • "What's the value of a put option on GOOGL strike $100?"
   • "Compare American vs European option pricing for MSFT"

📈 GREEKS ANALYSIS:
   • "Calculate delta and gamma for TSLA call option"
   • "What are the Greeks for this option position?"
   • "Show me the risk profile of this trade"

📊 VOLATILITY:
   • "What's the historical volatility of AAPL?"
   • "Calculate implied volatility for this option"
   • "Analyze volatility regime for SPY"

🎯 STRATEGIES:
   • "Analyze a straddle strategy on SPY"
   • "Compare iron condor vs butterfly spread"
   • "Show payoff diagram for covered call"

⚠️ RISK MANAGEMENT:
   • "Analyze portfolio risk and suggest hedges"
   • "Calculate VaR for my options portfolio"
   • "Show stress test results"

🎓 EDUCATION:
   • "What is put-call parity?"
   • "Explain delta hedging"
   • "How does Black-Scholes work?"

🎨 VISUALIZATION COMMANDS:
   • "viz [query]" - Enhanced visualization mode
   • Charts are generated automatically for pricing and analysis
   • Dashboard opens in browser for complex analyses

💡 COMMANDS:
   • help - Show this help
   • examples - Show example queries  
   • viz [query] - Enhanced visualization
   • quit/exit - Exit the program
        """
        print(help_text)
    
    def _show_examples(self):
        """Show example queries"""
        examples = [
            "Price a call option on AAPL strike $160 expiring in 45 days",
            "What is delta hedging and how does it work?",
            "Calculate historical volatility for GOOGL over 30 days",
            "Analyze iron condor strategy on SPY",
            "What are the Greeks for a put option?",
            "Analyze put-call parity for MSFT options",
            "Compare straddle vs strangle strategies",
            "Calculate portfolio risk and suggest hedges",
            "viz Price TSLA barrier option with Monte Carlo",
            "viz Analyze volatility surface for SPY options"
        ]
        
        print("\n💡 EXAMPLE QUERIES:")
        for i, example in enumerate(examples, 1):
            print(f"   {i}. {example}")
    
    def _display_result_with_charts(self, result: Dict[str, Any]):
        """Display query result with chart information"""
        print("\n" + "="*60)
        
        # Status
        status_emoji = "✅" if result['success'] else "❌"
        print(f"{status_emoji} Status: {'SUCCESS' if result['success'] else 'FAILED'}")
        
        # Query type and confidence
        if 'query_type' in result:
            print(f"🏷️  Query Type: {result['query_type'].replace('_', ' ').title()}")
        
        if 'confidence' in result and result['confidence'] > 0:
            confidence_emoji = "🟢" if result['confidence'] > 0.8 else "🟡" if result['confidence'] > 0.6 else "🔴"
            print(f"{confidence_emoji} Confidence: {result['confidence']:.1%}")
        
        print("-"*60)
        
        # Main response
        if result.get('response'):
            print("📋 ANALYSIS:")
            print(result['response'])
        
        # Chart information
        if result.get('chart_files'):
            print(f"\n📊 VISUALIZATIONS GENERATED:")
            chart_files = result['chart_files']
            for chart_type, filename in chart_files.items():
                chart_name = chart_type.replace('_', ' ').title()
                print(f"   • {chart_name}: {filename}")
            
            # Offer to open charts
            if 'dashboard' in chart_files:
                open_dashboard = input(f"\n🌐 Open dashboard in browser? (y/n): ").lower().strip()
                if open_dashboard in ['y', 'yes']:
                    dashboard_path = os.path.abspath(chart_files['dashboard'])
                    webbrowser.open(f"file://{dashboard_path}")
        
        # Additional result data
        if result.get('result_data') and isinstance(result['result_data'], dict):
            if 'option_price' in result['result_data']:
                print(f"\n💰 Calculated Option Price: ${result['result_data']['option_price']}")
        
        # Warnings
        if result.get('warnings'):
            print(f"\n⚠️  WARNINGS:")
            for warning in result['warnings'][:3]:
                print(f"   • {warning}")
        
        # Errors
        if result.get('errors'):
            print(f"\n❌ ERRORS:")
            for error in result['errors'][:2]:
                print(f"   • {error}")
        
        print("="*60)

async def process_batch_queries(agent, batch_file: str, generate_viz: bool = False):
    """Process queries from batch file"""
    try:
        with open(batch_file, 'r') as f:
            queries = json.load(f)
        
        if not isinstance(queries, list):
            queries = [queries] if isinstance(queries, str) else []
        
        results = []
        total_queries = len(queries)
        
        print(f"\n🔄 Processing {total_queries} queries from {batch_file}...")
        
        for i, query in enumerate(queries, 1):
            print(f"\n📝 Query {i}/{total_queries}: {query[:100]}...")
            
            result = agent.process_query_with_visualization(
                query, 
                generate_charts=generate_viz
            )
            
            results.append({
                "query": query,
                "result": result,
                "timestamp": result.get("timestamp"),
                "success": result.get("success", False)
            })
            
            # Show brief status
            status = "✅" if result["success"] else "❌"
            print(f"   {status} {result.get('query_type', 'unknown').replace('_', ' ').title()}")
        
        # Save results
        from datetime import datetime
        output_file = f"outputs/batch_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Generate summary
        successful = sum(1 for r in results if r["success"])
        failed = total_queries - successful
        
        print(f"\n📊 BATCH PROCESSING SUMMARY:")
        print(f"   Total Queries: {total_queries}")
        print(f"   Successful: {successful}")
        print(f"   Failed: {failed}")
        print(f"   Success Rate: {(successful/total_queries)*100:.1f}%")
        print(f"   Results saved to: {output_file}")
        
        # Generate consolidated dashboard if visualizations were created
        if generate_viz:
            await create_batch_dashboard(results, agent)
        
    except Exception as e:
        print(f"❌ Batch processing failed: {e}")
        logger.error(f"Batch processing error: {e}")

async def create_batch_dashboard(results: list, agent):
    """Create consolidated dashboard for batch results"""
    try:
        print("\n🎨 Creating consolidated dashboard...")
        
        # Collect all charts from successful results
        all_charts = {}
        chart_counter = 1
        
        for result_data in results:
            if result_data["success"] and result_data["result"].get("charts"):
                query = result_data["query"][:30] + "..." if len(result_data["query"]) > 30 else result_data["query"]
                
                for chart_type, fig in result_data["result"]["charts"].items():
                    chart_name = f"query_{chart_counter}_{chart_type}"
                    
                    # Update chart title to include query context
                    fig.update_layout(title=f"{fig.layout.title.text}<br><sub>{query}</sub>")
                    all_charts[chart_name] = fig
                
                chart_counter += 1
        
        if all_charts:
            from datetime import datetime
            Path("visualizations/batch_results").mkdir(parents=True, exist_ok=True)
            dashboard_file = f"visualizations/batch_results/batch_dashboard_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
            saved_dashboard = agent.visualizer.save_dashboard_as_html(all_charts, dashboard_file)
            
            if saved_dashboard:
                print(f"✅ Consolidated dashboard saved: {saved_dashboard}")
                
                # Offer to open
                open_dashboard = input("🌐 Open consolidated dashboard in browser? (y/n): ").lower().strip()
                if open_dashboard in ['y', 'yes']:
                    dashboard_path = os.path.abspath(saved_dashboard)
                    webbrowser.open(f"file://{dashboard_path}")
        else:
            print("ℹ️  No charts available for consolidated dashboard")
            
    except Exception as e:
        logger.error(f"Batch dashboard creation failed: {e}")

async def run_comprehensive_tests():
    """Run comprehensive system tests"""
    print("\n🧪 Running Comprehensive Tests...")
    
    test_queries = [
        {
            "query": "Price a call option on AAPL with strike $150, expiry in 30 days",
            "expected_type": "option_pricing",
            "should_have_charts": True
        },
        {
            "query": "What is delta hedging?",
            "expected_type": "educational",
            "should_have_charts": False
        },
        {
            "query": "Calculate Greeks for GOOGL put option strike $100",
            "expected_type": "greeks_analysis", 
            "should_have_charts": True
        },
        {
            "query": "Analyze historical volatility for TSLA over 60 days",
            "expected_type": "volatility_analysis",
            "should_have_charts": True
        },
        {
            "query": "Analyze iron condor strategy on SPY",
            "expected_type": "strategy_analysis",
            "should_have_charts": True
        },
        {
            "query": "Calculate portfolio risk and suggest hedges",
            "expected_type": "risk_management",
            "should_have_charts": True
        },
        {
            "query": "Price American put option on SPY using binomial tree",
            "expected_type": "option_pricing",
            "should_have_charts": True
        },
        {
            "query": "Monte Carlo simulation for barrier option on MSFT",
            "expected_type": "option_pricing",
            "should_have_charts": True
        },
        {
            "query": "What's the weather like today?",
            "expected_type": "invalid",
            "should_have_charts": False
        }
    ]
    
    # Initialize agent
    agent = OptionsAIAgentWithVisualization()
    success = await agent.initialize()
    
    if not success:
        print("❌ Agent initialization failed")
        return
    
    print("✅ Agent initialized successfully")
    print("\n" + "="*80)
    
    test_results = []
    passed = 0
    failed = 0
    
    for i, test in enumerate(test_queries, 1):
        print(f"\n🧪 Test {i}/{len(test_queries)}: {test['query'][:60]}...")
        
        try:
            result = agent.process_query_with_visualization(
                test['query'], 
                generate_charts=test['should_have_charts']
            )
            
            # Check result
            success_status = "✅" if result['success'] else "❌"
            
            # Check query type
            actual_type = result.get('query_type', 'unknown')
            expected_type = test['expected_type']
            type_match = "✅" if actual_type == expected_type else "❌"
            
            # Check charts
            has_charts = bool(result.get('charts'))
            should_have_charts = test['should_have_charts'] and result['success']
            charts_match = "✅" if (has_charts == should_have_charts) or not should_have_charts else "❌"
            
            # Overall test result
            test_passed = (
                result['success'] == (expected_type != 'invalid') and
                actual_type == expected_type and
                (has_charts == should_have_charts or not should_have_charts)
            )
            
            if test_passed:
                passed += 1
            else:
                failed += 1
            
            test_results.append({
                "query": test['query'],
                "expected_type": expected_type,
                "actual_type": actual_type,
                "success": result['success'],
                "confidence": result.get('confidence', 0),
                "has_charts": has_charts,
                "test_passed": test_passed
            })
            
            print(f"   Success: {success_status}")
            print(f"   Type: {actual_type} (expected: {expected_type}) {type_match}")
            print(f"   Confidence: {result.get('confidence', 0):.1%}")
            print(f"   Charts: {has_charts} {charts_match}")
            print(f"   Overall: {'✅ PASS' if test_passed else '❌ FAIL'}")
            
            if not result['success'] and result.get('errors'):
                print(f"   Error: {result['errors'][0] if result['errors'] else 'Unknown error'}")
                
        except Exception as e:
            print(f"   ❌ Test execution failed: {e}")
            failed += 1
            test_results.append({
                "query": test['query'],
                "error": str(e),
                "test_passed": False
            })
    
    # Test summary
    print(f"\n" + "="*80)
    print(f"🧪 TEST SUMMARY")
    print(f"="*80)
    print(f"Total Tests: {len(test_queries)}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Success Rate: {(passed/len(test_queries))*100:.1f}%")
    
    # Detailed results
    print(f"\n📊 DETAILED RESULTS:")
    for i, result in enumerate(test_results, 1):
        status = "✅" if result.get("test_passed", False) else "❌"
        query_short = result["query"][:50] + "..." if len(result["query"]) > 50 else result["query"]
        print(f"   {i}. {status} {query_short}")
        
        if "error" in result:
            print(f"      Error: {result['error']}")
        elif not result.get("test_passed", False):
            expected = result.get("expected_type", "unknown")
            actual = result.get("actual_type", "unknown")
            if expected != actual:
                print(f"      Expected: {expected}, Got: {actual}")
    
    # Save test results
    from datetime import datetime
    test_file = f"outputs/test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    try:
        with open(test_file, 'w') as f:
            json.dump({
                "summary": {
                    "total_tests": len(test_queries),
                    "passed": passed,
                    "failed": failed,
                    "success_rate": (passed/len(test_queries))*100
                },
                "detailed_results": test_results,
                "timestamp": datetime.now().isoformat()
            }, f, indent=2, default=str)
        
        print(f"\n📄 Test results saved to: {test_file}")
        
    except Exception as e:
        print(f"⚠️  Could not save test results: {e}")

async def main():
    """Main application entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Options AI Agent with Visualization")
    parser.add_argument('--test', action='store_true', help='Run tests')
    parser.add_argument('--query', type=str, help='Process a single query')
    parser.add_argument('--viz', action='store_true', help='Generate visualizations')
    parser.add_argument('--batch', type=str, help='Process queries from JSON file')
    
    args = parser.parse_args()
    
    if args.test:
        await run_comprehensive_tests()
        return
    
    # Initialize agent
    agent = OptionsAIAgentWithVisualization()
    success = await agent.initialize()
    
    if not success:
        logger.error("Failed to initialize agent")
        sys.exit(1)
    
    if args.query:
        # Single query mode
        result = agent.process_query_with_visualization(args.query, generate_charts=args.viz)
        agent._display_result_with_charts(result)
        
        # Auto-open dashboard if visualization requested and available
        if args.viz and result.get("chart_files", {}).get("dashboard"):
            dashboard_path = os.path.abspath(result["chart_files"]["dashboard"])
            print(f"\n🌐 Opening dashboard: {dashboard_path}")
            webbrowser.open(f"file://{dashboard_path}")
    
    elif args.batch:
        # Batch mode
        await process_batch_queries(agent, args.batch, args.viz)
    
    else:
        # Interactive mode
        agent.interactive_mode()

if __name__ == "__main__":
    asyncio.run(main())