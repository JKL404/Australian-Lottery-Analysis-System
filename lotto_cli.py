#!/usr/bin/env python3
"""
Lottery Analysis CLI Tool
"""
import argparse
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import pandas as pd
import numpy as np
import random
import os
from lotto_core import LOTTERY_TYPES
from lotto_data_manager import LottoDataManager
from lotto_scraper import UniversalLottoScraper
from lotto_analytics import (
    enhanced_analysis, 
    advanced_prediction_engine,
    evaluate_prediction_methods,
    simple_frequency_model,
    time_weighted_frequency_model,
    pattern_based_predictions,
    generate_random_predictions,
    generate_advanced_visualizations
)
import json

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('LottoCLI')


def main():
    parser = argparse.ArgumentParser(description='Lottery Analysis System')
    parser.add_argument('--lottery', required=True, choices=LOTTERY_TYPES.keys(),
                        help='Lottery type to analyze')
    parser.add_argument('--action', required=True, 
                        choices=['scrape', 'analyze', 'predict', 'import', 'batch', 'evaluate'],
                        help='Action to perform')
    parser.add_argument('--years', type=int, default=5,
                        help='Years of historical data to process')
    parser.add_argument('--input-dir', default='results/historical',
                        help='Directory with existing CSV/JSON files')
    parser.add_argument('--output-dir', default='results/predictions',
                        help='Output directory for predictions')
    parser.add_argument('--format', choices=['csv', 'json'], default='csv',
                       help='Output format for predictions')
    parser.add_argument('--visualize', action='store_true',
                       help='Generate visualizations during analysis')
    parser.add_argument('--interactive', action='store_true',
                       help='Generate interactive visualizations (requires plotly)')
    parser.add_argument('--prediction-count', type=int, default=10,
                       help='Number of predictions to generate')
    parser.add_argument('--model', 
                       choices=['frequency', 'time_weighted', 'pattern', 'advanced'],
                       default='advanced',
                       help='Prediction model to use (advanced uses all methods)')
    parser.add_argument('--skip-evaluation', action='store_true',
                       help='Skip evaluation step in batch processing')
    parser.add_argument('--output-report', action='store_true',
                       help='Generate a detailed PDF report with analysis results')

    args = parser.parse_args()
    manager = LottoDataManager(output_root="results")
    
    try:
        if args.action == 'scrape':
            handle_scrape(args, manager)
        elif args.action == 'analyze':
            handle_analysis(args, manager)
        elif args.action == 'predict':
            handle_prediction(args, manager)
        elif args.action == 'import':
            handle_import(args, manager)
        elif args.action == 'batch':
            handle_batch(args, manager)
        elif args.action == 'evaluate':
            handle_evaluate(args, manager)
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        print(f"❌ Error: {str(e)}")


def handle_scrape(args, manager):
    """Handle data scraping and merging"""
    print(f"🔄 Scraping {args.lottery} data for last {args.years} years")
    scraper = UniversalLottoScraper(args.lottery)
    data = scraper.scrape(years_back=args.years)
    
    if not data:
        print("❌ No data found or scraping failed")
        return
        
    print(f"💾 Saving {len(data)} results...")
    saved_files = manager.save_dataset(data, args.lottery)
    
    print("🔗 Merging historical data...")
    merged = manager.merge_historical_data(args.lottery)
    
    if merged is not None and not merged.empty:
        print(f"✅ Total {len(merged)} records available after merge")
    else:
        print("⚠️ No data available after merge")


def handle_analysis(args, manager):
    """Handle data analysis with existing files"""
    # Corrected path to look in lottery-specific directory
    input_dir = Path(args.input_dir) / args.lottery
    files = list(input_dir.glob("combined*.csv"))
    
    if not files:
        print(f"⚠️ No combined data found for {args.lottery}, attempting to merge...")
        merged = manager.merge_historical_data(args.lottery)
        
        if merged is None or merged.empty:
            print(f"❌ No data available for {args.lottery}")
            return
            
        # Check again after merge
        files = list(input_dir.glob("combined*.csv"))
        if not files:
            print("❌ Failed to create combined data file")
            return
    
    latest_file = max(files, key=lambda f: f.stat().st_mtime)
    print(f"📊 Analyzing {latest_file}...")
    
    # Load and perform analysis
    df = pd.read_csv(latest_file)
    config = LOTTERY_TYPES[args.lottery]
    
    # Use enhanced analysis from imported module
    visualize_type = "interactive" if args.visualize and getattr(args, 'interactive', False) else args.visualize
    results = enhanced_analysis(df, args.lottery, config, visualize=visualize_type)
    
    # Print selected results to console
    print(f"\n🔍 {args.lottery.replace('_', ' ').title()} Analysis")
    print(f"📅 Date Range: {results['basic_stats']['date_range'][0]} to {results['basic_stats']['date_range'][1]}")
    print(f"🎰 Total Draws: {results['basic_stats']['draws']}")
    
    print("\n🏆 Top 10 Frequent Main Numbers:")
    top_nums = sorted(results['frequencies'].items(), key=lambda x: x[1], reverse=True)[:10]
    for num, freq in top_nums:
        print(f"Number {num}: {freq} times")
    
    if 'supp_frequencies' in results:
        print("\n🏅 Top 5 Frequent Supplementary Numbers:")
        top_supps = sorted(results['supp_frequencies'].items(), key=lambda x: x[1], reverse=True)[:5]
        for num, freq in top_supps:
            print(f"Number {num}: {freq} times")
    
    print(f"\n📊 Statistics:")
    print(f"Most drawn main number: {results['basic_stats']['most_frequent']['number']} ({results['basic_stats']['most_frequent']['count']} times)")
    print(f"Least drawn main number: {results['basic_stats']['least_frequent']['number']} ({results['basic_stats']['least_frequent']['count']} times)")
    
    print("\n🎲 Randomness Testing:")
    print(f"Chi-Square Statistic: {results['randomness_tests']['chi2_stat']:.2f}")
    print(f"P-Value: {results['randomness_tests']['p_value']:.4f}")
    print(f"Is Random: {'Yes' if results['randomness_tests']['is_random'] else 'No'}")
    
    print("\n👫 Common Number Pairs:")
    for pair_str, count in list(results['common_pairs'].items())[:5]:
        print(f"{pair_str}: {count} times")
    
    # If time patterns analysis was performed
    if 'time_patterns' in results:
        print("\n📈 Time Series Analysis:")
        print(f"Trend direction: {results['time_patterns'].get('trend_direction', 'unknown')}")
        print(f"Seasonality detected: {'Yes' if results['time_patterns'].get('has_seasonality', False) else 'No'}")
    
    print("\n🔍 Number Cluster Analysis:")
    print("Number groups that frequently appear together:")
    for i, cluster_numbers in enumerate(results['clusters']['cluster_groups']):
        print(f"Group {i+1}: {cluster_numbers}")
    
    print(f"\n✅ Analysis complete - detailed report saved to results/analysis/reports/")


def handle_prediction(args, manager):
    """Generate and save predictions using advanced methods"""
    print(f"🔮 Generating predictions for {args.lottery}")
    config = LOTTERY_TYPES[args.lottery]
    
    # Load historical data
    historical = manager.merge_historical_data(args.lottery)
    
    # Generate predictions with advanced engine
    predictions = advanced_prediction_engine(historical, config, args.prediction_count)
    
    # Print summary of prediction methods used
    methods_used = {}
    for pred in predictions:
        model = pred.get("model", "unknown")
        if model not in methods_used:
            methods_used[model] = 0
        methods_used[model] += 1
    
    print("🧠 Prediction methods used:")
    for method, count in methods_used.items():
        print(f" - {method.replace('_', ' ').title()}: {count} predictions")
    
    # Save predictions with model information included
    output_file = save_predictions_with_model(manager, predictions, args.lottery, format=args.format)
    print(f"✅ Predictions saved to {output_file}")


def save_predictions_with_model(manager, predictions, lottery_type, format="csv"):
    """Save predictions with model information included"""
    try:
        config = LOTTERY_TYPES[lottery_type]
        
        # Add validation
        for pred in predictions:
            if len(pred["main"]) != config.main_numbers:
                raise ValueError(f"{config.name} requires exactly {config.main_numbers} main numbers")
            if len(pred["supp"]) != config.supplementary:
                raise ValueError(f"{config.name} requires exactly {config.supplementary} supplementary numbers")
        
        filename = f"{lottery_type}_predictions_{datetime.now().strftime('%Y%m%d')}.{format}"
        path = Path(manager.output_root) / "predictions" / filename
        
        # Ensure the predictions directory exists
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Delete the file if it already exists
        if path.exists():
            path.unlink()
            logger.info(f"Deleted existing file: {path}")
        
        # Include model information in the record
        records = [{
            "generated": datetime.now().isoformat(),
            "combination": i+1,
            "main_numbers": " ".join(map(str, pred["main"])),
            "supplementary": " ".join(map(str, pred["supp"])) if pred["supp"] else "",
            "probability": pred["probability"],
            "confidence_score": pred.get("confidence_score", None),
            "prediction_method": pred.get("model", "unknown").replace("_", " ").title()
        } for i, pred in enumerate(predictions)]

        if format == "csv":
            pd.DataFrame(records).to_csv(path, index=False)
        elif format == "json":
            with open(path, "w") as f:
                json.dump(records, f, indent=2)
        else:
            raise ValueError(f"Unsupported format: {format}")

        logger.info(f"Saved {len(predictions)} predictions to {path}")
        return path
    except Exception as e:
        logger.error(f"Error saving predictions: {str(e)}")
        raise


def handle_import(args, manager):
    """Import existing historical data"""
    source = Path(f"historical_data/{args.lottery}.csv")
    if not source.exists():
        raise FileNotFoundError(f"Source file {source} not found")
    
    print(f"📥 Importing historical data from {source}")
    df = pd.read_csv(source)
    
    # Convert to universal format with proper type handling
    data = []
    for _, row in df.iterrows():
        try:
            # Handle MAIN NUMBERS regardless of type
            main_str = str(row["MAIN NUMBERS"])
            main_numbers = [int(n) for n in main_str.split() if n.strip()] if ' ' in main_str else [int(main_str)]
            
            # Handle SUPP regardless of type
            supp_str = str(row["SUPP"])
            supp_numbers = [int(n) for n in supp_str.split() if n.strip()] if ' ' in supp_str else [int(supp_str)]
            
            data.append({
                "date": row["DRAW DATE"],
                "draw": row["DRAW ID"],
                "main_numbers": main_numbers,
                "supplementary": supp_numbers
            })
        except Exception as e:
            print(f"⚠️ Error processing row: {e}")
            continue

    if data:
        manager.save_dataset(data, args.lottery)
        print(f"✅ Imported {len(data)} historical records")
    else:
        print("❌ No data could be imported")


def handle_evaluate(args, manager):
    """Evaluate prediction methods using historical data"""
    print(f"🔬 Evaluating prediction methods for {args.lottery}...")
    # Load historical data
    historical = manager.merge_historical_data(args.lottery)
    
    if historical is None or historical.empty:
        print(f"❌ No historical data available for {args.lottery}")
        return
        
    config = LOTTERY_TYPES[args.lottery]
    results = evaluate_prediction_methods(historical, config)
    
    if "insufficient_data" in results:
        print("⚠️ Insufficient historical data for a reliable evaluation")
        return
        
    print(f"\n📊 Prediction Method Evaluation for {args.lottery}")
    for method, stats in results.items():
        print(f"\n🔸 {method.replace('_', ' ').title()} Method:")
        print(f"  Average matches: {stats['avg_matches']:.2f}")
        print(f"  Maximum matches: {stats['max_matches']}")
        print(f"  Full matches: {stats['full_matches']}")
        print("  Match distribution:")
        for i, count in stats['match_distribution'].items():
            print(f"    {i} matches: {count} times")
    
    # Calculate overall best method
    best_method = max(results.items(), key=lambda x: x[1]['avg_matches'])
    print(f"\n🏆 Best performing method: {best_method[0].replace('_', ' ').title()} with {best_method[1]['avg_matches']:.2f} average matches")
    
    # Save results to file
    output_dir = Path(f"results/analysis/evaluation")
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d")
    
    with open(output_dir / f"{args.lottery}_evaluation_{timestamp}.txt", "w") as f:
        f.write(f"Prediction Method Evaluation for {args.lottery}\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        for method, stats in results.items():
            f.write(f"\n{method.replace('_', ' ').title()} Method:\n")
            f.write(f"  Average matches: {stats['avg_matches']:.2f}\n")
            f.write(f"  Maximum matches: {stats['max_matches']}\n")
            f.write(f"  Full matches: {stats['full_matches']}\n")
            f.write("  Match distribution:\n")
            for i, count in stats['match_distribution'].items():
                f.write(f"    {i} matches: {count} times\n")
    
    print(f"📝 Evaluation results saved to {output_dir / f'{args.lottery}_evaluation_{timestamp}.txt'}")


def handle_batch(args, manager):
    """Handle batch processing - scrape, analyze, predict in sequence"""
    print(f"🔄 Starting batch processing for {args.lottery}")
    
    # Step 0: Import historical data if available
    print("\n📥 STEP 0: Importing historical data...")
    try:
        source = Path(f"historical_data/{args.lottery}.csv")
        if source.exists():
            # Use the existing handle_import function
            import_args = argparse.Namespace()
            import_args.lottery = args.lottery
            handle_import(import_args, manager)
        else:
            print(f"⚠️ No historical data file found at {source}")
    except Exception as e:
        print(f"⚠️ Error importing historical data: {str(e)}")
    
    # Step 1: Scrape recent data
    print("\n📥 STEP 1: Scraping recent data...")
    scraper = UniversalLottoScraper(args.lottery)
    data = scraper.scrape(years_back=args.years)
    
    if data:
        print(f"💾 Saving {len(data)} results...")
        saved_files = manager.save_dataset(data, args.lottery)
    else:
        print("⚠️ No new data scraped, continuing with existing data")
    
    # Step 2: Merge and analyze
    print(f"\n📊 STEP 2: Analyzing historical data...")
    
    # Create the output directory before merging
    Path(f"results/historical/{args.lottery}").mkdir(parents=True, exist_ok=True)
    
    merged = manager.merge_historical_data(args.lottery)
    
    if merged is None or merged.empty:
        print(f"❌ No data available for {args.lottery}, batch processing stopped")
        return
    
    config = LOTTERY_TYPES[args.lottery]
    print(f"📈 Running enhanced analysis on {len(merged)} records")
    
    # Run analysis with visualizations enabled
    visualize_type = "interactive" if getattr(args, 'interactive', False) else True
    results = enhanced_analysis(merged, args.lottery, config, visualize=visualize_type)
    
    # Step 3: Generate predictions
    print(f"\n🔮 STEP 3: Generating predictions...")
    predictions = advanced_prediction_engine(merged, config, args.prediction_count)
    
    # Print prediction summary
    methods_used = {}
    for pred in predictions:
        model = pred.get("model", "unknown")
        if model not in methods_used:
            methods_used[model] = 0
        methods_used[model] += 1
    
    print("Prediction methods used:")
    for method, count in methods_used.items():
        print(f" - {method.replace('_', ' ').title()}: {count} predictions")
    
    # Save predictions in both formats
    for fmt in ['csv', 'json']:
        output_file = save_predictions_with_model(manager, predictions, args.lottery, format=fmt)
        print(f"✅ Predictions saved to {output_file}")
    
    # Step 4: Evaluate prediction methods
    print(f"\n🔬 STEP 4: Evaluating prediction methods...")
    if len(merged) >= 50:
        results = evaluate_prediction_methods(merged, config)
        
        best_method = max(results.items(), key=lambda x: x[1]['avg_matches'])
        print(f"🏆 Best performing method: {best_method[0].replace('_', ' ').title()} with {best_method[1]['avg_matches']:.2f} average matches")
    else:
        print("⚠️ Insufficient data for reliable evaluation")
    
    print(f"\n✅ Batch processing for {args.lottery} completed successfully")
    print(f"📁 Results saved to the 'results' directory")


if __name__ == "__main__":
    main()
