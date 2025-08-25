#!/usr/bin/env python3
"""
Example: Using the OpenAI Adapter

This example demonstrates how to use the OpenAI adapter programmatically
to run evaluations and generate reports.
"""

import os
from pathlib import Path
from lightweight_evals.config import Config
from lightweight_evals.adapters.openai import OpenAIAdapter
from lightweight_evals.adapters.dummy import DummyAdapter
from lightweight_evals.runner import EvalRunner, RunConfig
from lightweight_evals.scoring import LLMJudge
from lightweight_evals.reporting.report_builder import ReportBuilder


def main():
    """Demonstrate using OpenAI adapter programmatically."""
    print("=== Lightweight Evals: OpenAI Adapter Example ===\n")
    
    # Load configuration
    config = Config()
    
    # Check if OpenAI API key is available
    if not config.openai_api_key:
        print("❌ No OpenAI API key found in environment.")
        print("Please set OPENAI_API_KEY in your .env file or environment variables.")
        print("\nFalling back to dummy adapter for demonstration...\n")
        
        # Use dummy adapter as fallback
        adapter = DummyAdapter(seed=42)
        judge_adapter = DummyAdapter(seed=123)
        print("✅ Using DummyAdapter (no API key required)")
    else:
        print("✅ OpenAI API key found, using OpenAI adapter")
        
        # Initialize OpenAI adapters
        adapter = OpenAIAdapter(
            model=config.default_model,
            api_key=config.openai_api_key
        )
        
        # Use the same adapter for judging (could use different model)
        judge_adapter = OpenAIAdapter(
            model="gpt-4o-mini",  # Could use a different model for judging
            api_key=config.openai_api_key
        )
    
    # Set up the evaluation runner
    runner = EvalRunner()
    
    # Configure the evaluation run
    run_config = RunConfig(
        adapter_name=adapter.name,
        eval_suite="harmlessness",  # Start with just one suite
        seed=42,
        max_tokens=150,
        temperature=0.1,
        output_dir=Path("./example_reports")
    )
    
    print(f"📊 Running {run_config.eval_suite} evaluation...")
    print(f"🤖 Model Adapter: {adapter.name}")
    print(f"⚖️ Judge Adapter: {judge_adapter.name}")
    print(f"🌱 Seed: {run_config.seed}")
    print(f"🎯 Max Tokens: {run_config.max_tokens}")
    print(f"🌡️ Temperature: {run_config.temperature}\n")
    
    try:
        # Run the evaluation
        result = runner.run_eval(
            adapter=adapter,
            suite_name=run_config.eval_suite,
            config=run_config,
            judge_adapter=judge_adapter
        )
        
        print("✅ Evaluation completed successfully!\n")
        
        # Display results summary
        stats = result.summary_stats
        print("📈 Results Summary:")
        print(f"   Pass Rate: {stats['pass_rate']:.1%}")
        print(f"   Passed: {stats['passed_items']}/{stats['total_items']}")
        
        if stats.get('average_scores'):
            print("   Average Scores:")
            for score_name, score_value in stats['average_scores'].items():
                print(f"     {score_name}: {score_value:.2f}")
        
        print()
        
        # Show some example results
        print("🔍 Sample Results:")
        for i, eval_result in enumerate(result.eval_results[:3]):  # Show first 3
            print(f"\n   {i+1}. Item: {eval_result.item_id}")
            print(f"      Prompt: {eval_result.prompt[:60]}...")
            print(f"      Response: {eval_result.response[:80]}...")
            print(f"      Passed: {'✅' if eval_result.passed else '❌'}")
            if eval_result.notes:
                print(f"      Notes: {eval_result.notes[:60]}...")
        
        if len(result.eval_results) > 3:
            print(f"   ... and {len(result.eval_results) - 3} more results")
        
        # Save results
        json_path = runner.save_results(result)
        print(f"\n💾 Results saved to: {json_path}")
        
        # Generate reports
        report_builder = ReportBuilder()
        
        # HTML report
        html_path = run_config.output_dir / f"example_report_{result.timestamp}.html"
        report_builder.generate_html_report(result, html_path)
        print(f"📄 HTML report: {html_path}")
        
        # Markdown report
        md_path = run_config.output_dir / f"example_report_{result.timestamp}.md"
        report_builder.generate_markdown_report(result, md_path)
        print(f"📝 Markdown report: {md_path}")
        
        print(f"\n🌐 Open the HTML report in your browser:")
        print(f"   open {html_path}")
        
    except Exception as e:
        print(f"❌ Error during evaluation: {e}")
        return
    
    # Demonstrate running multiple suites
    print(f"\n{'='*50}")
    print("🚀 Running Multiple Evaluation Suites")
    print(f"{'='*50}\n")
    
    try:
        # Run all three suites
        suite_names = ["harmlessness", "robustness", "consistency"]
        print(f"📊 Running {len(suite_names)} evaluation suites...")
        
        all_results = runner.run_multiple_suites(
            adapter=adapter,
            suite_names=suite_names,
            config=run_config,
            judge_adapter=judge_adapter
        )
        
        print("✅ All evaluations completed!\n")
        
        # Summary of all results
        print("📈 Overall Summary:")
        total_items = sum(r.summary_stats['total_items'] for r in all_results)
        total_passed = sum(r.summary_stats['passed_items'] for r in all_results)
        overall_pass_rate = total_passed / total_items if total_items > 0 else 0
        
        print(f"   Overall Pass Rate: {overall_pass_rate:.1%}")
        print(f"   Total Items: {total_passed}/{total_items}")
        
        print("\n   By Suite:")
        for result in all_results:
            stats = result.summary_stats
            print(f"     {result.config.eval_suite.title()}: {stats['pass_rate']:.1%} ({stats['passed_items']}/{stats['total_items']})")
        
        # Save all results
        print(f"\n💾 Saved {len(all_results)} result files to {run_config.output_dir}")
        
    except Exception as e:
        print(f"❌ Error during multi-suite evaluation: {e}")


if __name__ == "__main__":
    main()