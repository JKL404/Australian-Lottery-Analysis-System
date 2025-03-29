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

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('LottoCLI')


def main():
    parser = argparse.ArgumentParser(description='Lottery Analysis System')
    parser.add_argument('--lottery', required=True, choices=LOTTERY_TYPES.keys(),
                        help='Lottery type to analyze')
    parser.add_argument('--action', required=True, choices=['scrape', 'analyze', 'predict', 'import', 'batch'],
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
    parser.add_argument('--samples', type=int, default=10,
                       help='Number of predictions to generate')

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
    perform_analysis(df, args.lottery, visualize=args.visualize)


def perform_analysis(df, lottery_type, visualize=True, save_report=True):
    """Core analysis logic"""
    config = LOTTERY_TYPES[lottery_type]
    
    print(f"\n🔍 {lottery_type.replace('_', ' ').title()} Analysis")
    print(f"📅 Date Range: {df['date'].min()} to {df['date'].max()}")
    print(f"🎰 Total Draws: {len(df)}")
    
    # Frequency analysis
    main_numbers = df['main_numbers'].str.split().explode().astype(int)
    print("\n🏆 Top 10 Frequent Main Numbers:")
    main_freq = main_numbers.value_counts().head(10)
    print(main_freq)
    
    if config.supplementary > 0:
        supp_numbers = df['supplementary'].str.split().explode().astype(int)
        print("\n🏅 Top 5 Frequent Supplementary Numbers:")
        supp_freq = supp_numbers.value_counts().head(5)
        print(supp_freq)
    
    # Pattern analysis
    print("\n📈 Recent Trends (Last 10 Draws):")
    recent = df.head(10)
    print(recent[['date', 'draw', 'main_numbers']].to_string(index=False))
    
    # Number distribution analysis
    main_range = range(config.main_range[0], config.main_range[1] + 1)
    all_main_counts = main_numbers.value_counts().reindex(main_range, fill_value=0)
    
    print("\n📊 Statistics:")
    print(f"Most drawn main number: {all_main_counts.idxmax()} ({all_main_counts.max()} times)")
    print(f"Least drawn main number: {all_main_counts.idxmin()} ({all_main_counts.min()} times)")
    
    # Time analysis - draw frequency over time
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df['year'] = df['date'].dt.year
        yearly_counts = df.groupby('year').size()
        print("\n📆 Draws per Year:")
        print(yearly_counts)
    
    # Adjacent pairs analysis
    print("\n👫 Common Number Pairs:")
    pairs = []
    for i, row in df.iterrows():
        numbers = [int(n) for n in row['main_numbers'].split() if n.strip()]
        for j in range(len(numbers)):
            for k in range(j+1, len(numbers)):
                pairs.append(tuple(sorted([numbers[j], numbers[k]])))
    
    pair_counts = pd.Series(pairs).value_counts().head(5)
    print(pair_counts)
    
    # Generate visualizations
    if visualize:
        generate_visualizations(df, lottery_type, all_main_counts, main_freq, supp_freq if config.supplementary > 0 else None)

    # Create a report string with all the output
    report = []
    report.append(f"\n🔍 {lottery_type.replace('_', ' ').title()} Analysis")
    report.append(f"📅 Date Range: {df['date'].min()} to {df['date'].max()}")
    report.append(f"🎰 Total Draws: {len(df)}")
    report.append("\n🏆 Top 10 Frequent Main Numbers:")
    report.append(str(main_freq))
    if config.supplementary > 0:
        report.append("\n🏅 Top 5 Frequent Supplementary Numbers:")
        report.append(str(supp_freq))
    report.append("\n📈 Recent Trends (Last 10 Draws):")
    report.append(recent[['date', 'draw', 'main_numbers']].to_string(index=False))
    report.append("\n📊 Statistics:")
    report.append(f"Most drawn main number: {all_main_counts.idxmax()} ({all_main_counts.max()} times)")
    report.append(f"Least drawn main number: {all_main_counts.idxmin()} ({all_main_counts.min()} times)")
    if 'date' in df.columns:
        report.append("\n📆 Draws per Year:")
        report.append(str(yearly_counts))
    report.append("\n👫 Common Number Pairs:")
    report.append(str(pair_counts))
    
    # Print to console
    print("\n".join(report))
    
    # Save to file
    if save_report:
        output_dir = Path(f"results/analysis/reports")
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d")
        with open(output_dir / f"{lottery_type}_analysis_{timestamp}.txt", "w") as f:
            f.write("\n".join(report))


def generate_visualizations(df, lottery_type, all_counts, main_freq, supp_freq):
    """Generate and save visualizations"""
    output_dir = Path(f"results/analysis/frequency_reports")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set style
    sns.set(style="whitegrid")
    
    # 1. Number frequency heatmap
    plt.figure(figsize=(12, 6))
    all_counts_df = all_counts.reset_index()
    all_counts_df.columns = ['Number', 'Frequency']
    
    # Reshape for heatmap
    grid_size = int(np.ceil(np.sqrt(len(all_counts_df))))
    heatmap_data = np.zeros((grid_size, grid_size))
    
    for i, row in all_counts_df.iterrows():
        num = int(row['Number'])
        freq = row['Frequency']
        r = (num - 1) // grid_size
        c = (num - 1) % grid_size
        if r < grid_size and c < grid_size:
            heatmap_data[r, c] = freq
    
    ax = sns.heatmap(heatmap_data, annot=True, fmt="g", cmap="YlGnBu")
    plt.title(f"{lottery_type.replace('_', ' ').title()} Number Frequency Heatmap")
    plt.savefig(output_dir / f"{lottery_type}_heatmap.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Top numbers bar chart
    plt.figure(figsize=(10, 6))
    sns.barplot(x=main_freq.index, y=main_freq.values)
    plt.title(f"Top {len(main_freq)} Most Frequent Numbers")
    plt.xlabel("Number")
    plt.ylabel("Frequency")
    plt.savefig(output_dir / f"{lottery_type}_top_numbers.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Number distribution
    plt.figure(figsize=(12, 6))
    sns.histplot(data=all_counts_df, x='Number', weights='Frequency', bins=len(all_counts_df))
    plt.title(f"{lottery_type.replace('_', ' ').title()} Number Distribution")
    plt.xlabel("Number")
    plt.ylabel("Frequency")
    plt.savefig(output_dir / f"{lottery_type}_distribution.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Time series analysis if date column exists
    if 'date' in df.columns:
        try:
            df['date'] = pd.to_datetime(df['date'])
            df['year_month'] = df['date'].dt.to_period('M')
            monthly_counts = df.groupby('year_month').size()
            
            plt.figure(figsize=(12, 6))
            monthly_counts.plot(kind='line')
            plt.title(f"{lottery_type.replace('_', ' ').title()} Draws per Month")
            plt.xlabel("Month")
            plt.ylabel("Number of Draws")
            plt.savefig(output_dir / f"{lottery_type}_time_series.png", dpi=300, bbox_inches='tight')
            plt.close()
        except Exception as e:
            logger.warning(f"Error creating time series chart: {str(e)}")
    
    print(f"✅ Visualizations saved to {output_dir}")


def handle_prediction(args, manager):
    """Generate and save predictions"""
    print(f"🔮 Generating predictions for {args.lottery}")
    config = LOTTERY_TYPES[args.lottery]
    
    # Load historical data
    historical = manager.merge_historical_data(args.lottery)
    
    # Generate predictions (improved logic)
    predictions = generate_predictions(historical, config)
    
    # Save predictions
    output_file = manager.save_predictions(predictions, args.lottery, format=args.format)
    print(f"✅ Predictions saved to {output_file}")


def generate_predictions(data, config):
    """Fixed prediction logic that avoids numpy boolean issues"""
    if data.empty:
        logger.warning("Empty dataset for predictions")
        return generate_random_predictions(config)
        
    try:
        # Process main numbers
        main_numbers = data['main_numbers'].str.strip().str.split()
        main_numbers = main_numbers.explode().astype(int)
        main_counts = main_numbers.value_counts()
        main_probs = main_counts / main_counts.sum()
        
        # Process supplementary numbers
        supp_probs = None
        if config.supplementary > 0:
            supp_numbers = data['supplementary'].str.strip().str.split()
            supp_numbers = supp_numbers.explode().astype(int)
            supp_counts = supp_numbers.value_counts()
            supp_probs = supp_counts / supp_counts.sum()
        
        # Generate predictions
        predictions = []
        for i in range(10):
            # Select main numbers
            main_nums = []
            available_nums = list(main_probs.index)
            available_probs = list(main_probs.values)
            
            for _ in range(config.main_numbers):
                if not available_nums:
                    break
                total = sum(available_probs)
                if total == 0:
                    break
                normalized_probs = [p/total for p in available_probs]
                chosen_idx = np.random.choice(len(available_nums), p=normalized_probs)
                main_nums.append(available_nums[chosen_idx])
                available_nums.pop(chosen_idx)
                available_probs.pop(chosen_idx)
            
            # Select supplementary numbers
            supp_nums = []
            if config.supplementary > 0 and supp_probs is not None and len(supp_probs) > 0:
                available_nums = list(supp_probs.index)
                available_probs = list(supp_probs.values)
                
                for _ in range(config.supplementary):
                    if not available_nums:
                        break
                    total = sum(available_probs)
                    if total == 0:
                        break
                    normalized_probs = [p/total for p in available_probs]
                    chosen_idx = np.random.choice(len(available_nums), p=normalized_probs)
                    supp_nums.append(available_nums[chosen_idx])
                    available_nums.pop(chosen_idx)
                    available_probs.pop(chosen_idx)
            
            # Calculate probabilities manually without numpy
            prob = 1.0
            for n in main_nums:
                prob *= main_probs.get(n, 0.0001)
            for n in supp_nums:
                if supp_probs is not None:
                    prob *= supp_probs.get(n, 0.0001)
            
            predictions.append({
                "main": sorted(main_nums),
                "supp": sorted(supp_nums),
                "probability": float(prob),
                "confidence_score": float(np.log(max(prob, 1e-10)))
            })
        
        return predictions
        
    except Exception as e:
        logger.error(f"Error in prediction: {str(e)}")
        return generate_random_predictions(config)


def generate_random_predictions(config):
    """Fallback random predictions"""
    return [{
        "main": sorted(random.sample(range(config.main_range[0], config.main_range[1]+1), config.main_numbers)),
        "supp": sorted(random.sample(range(config.supp_range[0], config.supp_range[1]+1), config.supplementary)),
        "probability": 0.0001,
        "confidence_score": -9.0
    } for _ in range(10)]


def handle_import(args, manager):
    """Import existing historical data"""
    # Fix the path to look in historical_data directory
    source = Path(f"historical_data/{args.lottery}.csv")
    if not source.exists():
        raise FileNotFoundError(f"Source file {source} not found")
    
    print(f"📥 Importing historical data from {source}")
    df = pd.read_csv(source)
    
    # Convert to universal format
    data = [{
        "date": row["DRAW DATE"],
        "draw": row["DRAW ID"],
        "main_numbers": list(map(int, row["MAIN NUMBERS"].strip().split())),
        "supplementary": list(map(int, row["SUPP"].strip().split()))
    } for _, row in df.iterrows()]

    manager.save_dataset(data, args.lottery)
    print(f"✅ Imported {len(data)} historical records")


if __name__ == "__main__":
    main() 