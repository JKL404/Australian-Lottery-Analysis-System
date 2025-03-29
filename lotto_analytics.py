"""
Advanced lottery analysis and prediction algorithms
Provides statistical analysis, pattern detection, and enhanced prediction methods
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import random
from datetime import datetime
from pathlib import Path
from scipy.stats import chisquare
from sklearn.cluster import KMeans

try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('LottoAnalytics')

def test_draw_randomness(df, lottery_type, config):
    """Statistical tests to check if draws show any non-random patterns"""
    # Chi-square test for uniform distribution
    main_numbers = df['main_numbers'].str.split().explode().astype(int)
    observed = main_numbers.value_counts().sort_index()
    n = len(df) * config.main_numbers
    expected = pd.Series([n / (config.main_range[1] - config.main_range[0] + 1)] * 
                         (config.main_range[1] - config.main_range[0] + 1),
                         index=range(config.main_range[0], config.main_range[1] + 1))
    
    # Run chi-square test
    chi2_stat, p_value = chisquare(observed, expected)
    
    return {
        'chi2_stat': chi2_stat,
        'p_value': p_value,
        'is_random': p_value > 0.05
    }

def find_number_clusters(df, lottery_type, config):
    """Find clusters of numbers that frequently appear together"""
    range_size = config.main_range[1] - config.main_range[0] + 1
    cooccurrence = np.zeros((range_size, range_size))
    
    for _, row in df.iterrows():
        numbers = [int(n) for n in row['main_numbers'].split() if n.strip()]
        for i in range(len(numbers)):
            for j in range(i+1, len(numbers)):
                idx1 = numbers[i] - config.main_range[0]
                idx2 = numbers[j] - config.main_range[0]
                cooccurrence[idx1, idx2] += 1
                cooccurrence[idx2, idx1] += 1
    
    # Apply clustering to identify number groups
    n_clusters = min(3, range_size // 2)  # Choose appropriate number of clusters
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    
    # Normalize and reshape for clustering
    norm_cooccurrence = cooccurrence / cooccurrence.max()
    clusters = kmeans.fit_predict(norm_cooccurrence)
    
    # Extract number groups
    cluster_groups = []
    for i in range(n_clusters):
        cluster_indices = np.where(clusters == i)[0]
        cluster_numbers = [idx + config.main_range[0] for idx in cluster_indices]
        cluster_groups.append(cluster_numbers)
    
    return {
        'clusters': clusters,
        'cooccurrence_matrix': cooccurrence,
        'cluster_groups': cluster_groups
    }

def analyze_time_patterns(df, lottery_type, config):
    """Analyze time-based patterns in the lottery data"""
    try:
        from statsmodels.tsa.seasonal import seasonal_decompose
        tslib_available = True
    except ImportError:
        tslib_available = False
    
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    
    # For simplicity, analyze frequency of draws per month
    df['year_month'] = df['date'].dt.to_period('M')
    monthly_counts = df.groupby('year_month').size()
    
    # If we have enough data points and statsmodels is available
    if len(monthly_counts) >= 12 and tslib_available:  # Need at least a year of data
        try:
            # Create a regular datetime index
            month_start_dates = pd.date_range(
                start=monthly_counts.index.min().to_timestamp(),
                end=monthly_counts.index.max().to_timestamp(),
                freq='MS'
            )
            
            # Reindex with the regular date range
            reindexed_counts = pd.Series(
                index=month_start_dates,
                data=[monthly_counts.get(month.to_period('M'), 0) 
                      for month in month_start_dates]
            )
            
            # Now decompose with the fixed index
            result = seasonal_decompose(reindexed_counts, model='additive')
            
            # Check if there's a trend
            trend = result.trend.dropna()
            trend_direction = "increasing" if trend.iloc[-1] > trend.iloc[0] else "decreasing"
            
            # Check for seasonality
            season = result.seasonal.dropna()
            has_seasonality = season.std() > (reindexed_counts.mean() * 0.05)
            
            return {
                'trend_direction': trend_direction,
                'has_seasonality': has_seasonality
            }
        except Exception as e:
            logger.warning(f"Error in time series analysis: {e}")
            # Fallback if decomposition fails
            return {
                'trend_direction': 'unknown',
                'has_seasonality': False,
                'error': str(e)
            }
    else:
        if not tslib_available:
            reason = "statsmodels not available"
        else:
            reason = "insufficient data"
            
        return {
            'trend_direction': 'unknown',
            'has_seasonality': False,
            'reason': reason
        }

def calculate_conditional_probabilities(df, lottery_type, config):
    """Calculate conditional probabilities for numbers"""
    conditional_probs = {}
    
    # Count occurrences of number A followed by number B in the next draw
    prev_draw = None
    
    for _, row in df.sort_values('date').iterrows():
        current_numbers = set([int(n) for n in row['main_numbers'].split() if n.strip()])
        
        if prev_draw is not None:
            for num in prev_draw:
                if num not in conditional_probs:
                    conditional_probs[num] = {}
                
                for next_num in current_numbers:
                    if next_num not in conditional_probs[num]:
                        conditional_probs[num][next_num] = 0
                    conditional_probs[num][next_num] += 1
        
        prev_draw = current_numbers
    
    # Convert to probability
    for num, follows in conditional_probs.items():
        total = sum(follows.values())
        conditional_probs[num] = sorted([(next_num, count/total) 
                                        for next_num, count in follows.items()],
                                        key=lambda x: x[1], reverse=True)
    
    return conditional_probs

def generate_correlation_heatmap(df, lottery_type, config, output_dir=None):
    """Generate heatmap showing correlations between numbers"""
    if output_dir is None:
        output_dir = Path(f"results/analysis/frequency_reports")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create co-occurrence matrix
    range_size = config.main_range[1] - config.main_range[0] + 1
    cooccurrence = np.zeros((range_size, range_size))
    
    for _, row in df.iterrows():
        numbers = [int(n) for n in row['main_numbers'].split() if n.strip()]
        for i in range(len(numbers)):
            for j in range(i+1, len(numbers)):
                idx1 = numbers[i] - config.main_range[0]
                idx2 = numbers[j] - config.main_range[0]
                cooccurrence[idx1, idx2] += 1
                cooccurrence[idx2, idx1] += 1
    
    # Normalize for heatmap
    max_val = cooccurrence.max()
    if max_val > 0:
        cooccurrence = cooccurrence / max_val
    
    # Generate heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(cooccurrence, cmap="YlGnBu", 
                xticklabels=range(config.main_range[0], config.main_range[1]+1),
                yticklabels=range(config.main_range[0], config.main_range[1]+1))
    plt.title(f"{lottery_type.replace('_', ' ').title()} Number Correlation Heatmap")
    plt.xlabel("Number")
    plt.ylabel("Number")
    plt.savefig(output_dir / f"{lottery_type}_correlation_heatmap.png", dpi=300, bbox_inches='tight')
    plt.close()

def generate_advanced_visualizations(df, lottery_type, config, output_dir=None):
    """Create interactive visualizations with Plotly if available"""
    if not PLOTLY_AVAILABLE:
        logger.warning("Plotly not installed. Interactive visualizations not available.")
        return False
        
    if output_dir is None:
        output_dir = Path(f"results/analysis/interactive")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Number frequency
    main_numbers = df['main_numbers'].str.split().explode().astype(int)
    freq_df = main_numbers.value_counts().reset_index()
    freq_df.columns = ['Number', 'Frequency']
    
    fig = px.bar(freq_df, x='Number', y='Frequency',
                title=f"{lottery_type.replace('_', ' ').title()} Number Frequency")
    fig.write_html(str(output_dir / f"{lottery_type}_frequency.html"))
    
    # Time series visualization
    if 'date' in df.columns:
        df = df.copy()
        df['date'] = pd.to_datetime(df['date'])
        df['year_month'] = df['date'].dt.to_period('M').astype(str)
        
        monthly_counts = df.groupby('year_month').size().reset_index()
        monthly_counts.columns = ['Month', 'Draws']
        
        fig = px.line(monthly_counts, x='Month', y='Draws',
                    title=f"{lottery_type.replace('_', ' ').title()} Monthly Draw Counts")
        fig.write_html(str(output_dir / f"{lottery_type}_time_series.html"))
    
    # Number pair correlation heatmap
    range_size = config.main_range[1] - config.main_range[0] + 1
    
    # Create correlation matrix
    cooccurrence = np.zeros((range_size, range_size))
    for _, row in df.iterrows():
        numbers = [int(n) for n in row['main_numbers'].split() if n.strip()]
        for i in range(len(numbers)):
            for j in range(i+1, len(numbers)):
                idx1 = numbers[i] - config.main_range[0]
                idx2 = numbers[j] - config.main_range[0]
                cooccurrence[idx1, idx2] += 1
                cooccurrence[idx2, idx1] += 1
    
    # Create labels for the heatmap
    labels = list(range(config.main_range[0], config.main_range[1]+1))
    
    # Create interactive heatmap
    fig = go.Figure(data=go.Heatmap(
        z=cooccurrence,
        x=labels,
        y=labels,
        colorscale='Viridis',
        hoverongaps=False))
    
    fig.update_layout(
        title=f"{lottery_type.replace('_', ' ').title()} Number Pair Correlation",
        xaxis_title="Number",
        yaxis_title="Number")
    
    fig.write_html(str(output_dir / f"{lottery_type}_correlation_heatmap.html"))
    return True

# Prediction methods
def simple_frequency_model(data, config, prediction_count):
    """Basic frequency-based prediction model"""
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
    for i in range(prediction_count):
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
        
        # Calculate probability score
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
            "confidence_score": float(np.log(max(prob, 1e-10))),
            "model": "frequency"
        })
    
    return predictions

def time_weighted_frequency_model(data, config, prediction_count=10):
    """Prediction model giving higher weight to recent draws"""
    if data.empty:
        return generate_random_predictions(config, prediction_count)
    
    # Add recency weight to each draw
    data = data.copy()
    data['date'] = pd.to_datetime(data['date'])
    latest_date = data['date'].max()
    
    # Calculate days since latest draw
    data['days_ago'] = (latest_date - data['date']).dt.days
    
    # Calculate time weight (more recent = higher weight)
    max_days = max(data['days_ago'].max(), 1)
    data['time_weight'] = 1 - (data['days_ago'] / (max_days + 1))
    
    # Weight numbers by recency
    weighted_main_numbers = []
    for _, row in data.iterrows():
        weight = row['time_weight']
        numbers = [int(n) for n in row['main_numbers'].split() if n.strip()]
        weighted_main_numbers.extend([(num, weight) for num in numbers])
    
    weighted_supp_numbers = []
    if config.supplementary > 0:
        for _, row in data.iterrows():
            weight = row['time_weight']
            numbers = [int(n) for n in row['supplementary'].split() if n.strip()]
            weighted_supp_numbers.extend([(num, weight) for num in numbers])
    
    # Calculate weighted probabilities
    main_weights = {}
    for num, weight in weighted_main_numbers:
        if num not in main_weights:
            main_weights[num] = 0
        main_weights[num] += weight
    
    supp_weights = {}
    for num, weight in weighted_supp_numbers:
        if num not in supp_weights:
            supp_weights[num] = 0
        supp_weights[num] += weight
    
    # Normalize to probabilities
    main_total = sum(main_weights.values())
    main_probs = {k: v/main_total for k, v in main_weights.items()}
    
    supp_probs = {}
    if supp_weights:
        supp_total = sum(supp_weights.values())
        supp_probs = {k: v/supp_total for k, v in supp_weights.items()}
    
    # Generate predictions with weighted sampling
    predictions = []
    for _ in range(prediction_count):
        # Sample main numbers
        main_nums = []
        remaining_main_probs = main_probs.copy()
        
        for _ in range(config.main_numbers):
            if not remaining_main_probs:
                break
                
            # Normalize remaining probabilities
            total = sum(remaining_main_probs.values())
            if total == 0:
                break
                
            # Weighted random choice
            r = random.random() * total
            cumsum = 0
            for num, prob in remaining_main_probs.items():
                cumsum += prob
                if r <= cumsum:
                    main_nums.append(num)
                    del remaining_main_probs[num]
                    break
        
        # Sample supplementary numbers
        supp_nums = []
        if config.supplementary > 0 and supp_probs:
            remaining_supp_probs = supp_probs.copy()
            
            for _ in range(config.supplementary):
                if not remaining_supp_probs:
                    break
                    
                # Normalize remaining probabilities
                total = sum(remaining_supp_probs.values())
                if total == 0:
                    break
                    
                # Weighted random choice
                r = random.random() * total
                cumsum = 0
                for num, prob in remaining_supp_probs.items():
                    cumsum += prob
                    if r <= cumsum:
                        supp_nums.append(num)
                        del remaining_supp_probs[num]
                        break
        
        # Calculate overall probability
        prob = 1.0
        for n in main_nums:
            prob *= main_probs.get(n, 0.0001)
        for n in supp_nums:
            prob *= supp_probs.get(n, 0.0001) if supp_probs else 0.0001
        
        predictions.append({
            "main": sorted(main_nums),
            "supp": sorted(supp_nums),
            "probability": float(prob),
            "confidence_score": float(np.log(max(prob, 1e-10))),
            "model": "time_weighted"
        })
    
    return predictions

def pattern_based_predictions(data, config, prediction_count=10):
    """Generate predictions based on patterns in recent draws"""
    if len(data) < 5:  # Need enough historical data
        return simple_frequency_model(data, config, prediction_count)
    
    # Analyze recent trends
    recent_data = data.head(20)  # Last 20 draws
    
    # Check for trend in sum of numbers
    recent_data['sum'] = recent_data['main_numbers'].apply(
        lambda x: sum([int(n) for n in x.split() if n.strip()])
    )
    
    avg_sum = recent_data['sum'].mean()
    std_sum = recent_data['sum'].std()
    
    # Analyze hot and cold numbers in recent draws
    recent_numbers = recent_data['main_numbers'].str.split().explode().astype(int)
    hot_numbers = recent_numbers.value_counts().head(config.main_numbers)
    
    # Get overall frequencies for comparison
    all_numbers = data['main_numbers'].str.split().explode().astype(int)
    all_freqs = all_numbers.value_counts()
    
    # Find numbers that are trending up (higher frequency in recent draws)
    trending_numbers = []
    for num in range(config.main_range[0], config.main_range[1]+1):
        recent_freq = recent_numbers.value_counts().get(num, 0) / len(recent_data)
        overall_freq = all_freqs.get(num, 0) / len(data)
        
        if recent_freq > overall_freq * 1.5:  # 50% more frequent recently
            trending_numbers.append(num)
    
    # Find cold numbers that might be "due"
    cold_numbers = []
    for num in range(config.main_range[0], config.main_range[1]+1):
        if num not in recent_numbers.values and num in all_numbers.values:
            cold_numbers.append(num)
    
    # Generate predictions with both hot and cold numbers
    predictions = []
    for i in range(prediction_count):
        main_nums = []
        
        # Mix strategy based on position in the prediction set
        if i % 3 == 0:  # One third hot numbers based
            # Use 3-4 hot numbers
            hot_count = min(len(hot_numbers), random.randint(3, 4))
            for j in range(hot_count):
                if j < len(hot_numbers.index):
                    main_nums.append(hot_numbers.index[j])
            
            # Fill remaining with random selection from historical frequencies
            remaining = config.main_numbers - len(main_nums)
            if remaining > 0:
                available = [n for n in range(config.main_range[0], config.main_range[1]+1) 
                           if n not in main_nums]
                weights = [all_freqs.get(n, 1) for n in available]
                total = sum(weights)
                if total > 0:
                    probs = [w/total for w in weights]
                    main_nums.extend(np.random.choice(available, size=remaining, p=probs))
        
        elif i % 3 == 1:  # One third trending numbers based
            # Use 2-3 trending numbers
            trending_count = min(len(trending_numbers), random.randint(2, 3))
            if trending_count > 0:
                main_nums.extend(random.sample(trending_numbers, trending_count))
            
            # Fill remaining with general frequency
            remaining = config.main_numbers - len(main_nums)
            if remaining > 0:
                available = [n for n in range(config.main_range[0], config.main_range[1]+1) 
                           if n not in main_nums]
                weights = [all_freqs.get(n, 1) for n in available]
                total = sum(weights)
                if total > 0:
                    probs = [w/total for w in weights]
                    main_nums.extend(np.random.choice(available, size=remaining, p=probs))
        
        else:  # One third with cold "due" numbers
            # Use 2-3 cold numbers
            cold_count = min(len(cold_numbers), random.randint(2, 3))
            if cold_count > 0:
                main_nums.extend(random.sample(cold_numbers, cold_count))
            
            # Fill remaining with hot numbers
            hot_count = min(len(hot_numbers), config.main_numbers - len(main_nums))
            for j in range(hot_count):
                if j < len(hot_numbers.index) and hot_numbers.index[j] not in main_nums:
                    main_nums.append(hot_numbers.index[j])
            
            # If still not enough, add from general pool
            remaining = config.main_numbers - len(main_nums)
            if remaining > 0:
                available = [n for n in range(config.main_range[0], config.main_range[1]+1) 
                           if n not in main_nums]
                main_nums.extend(random.sample(available, remaining))
        
        # Ensure exactly the right number of main numbers
        main_nums = sorted(main_nums[:config.main_numbers])
        while len(main_nums) < config.main_numbers:
            available = [n for n in range(config.main_range[0], config.main_range[1]+1) 
                       if n not in main_nums]
            main_nums.append(random.choice(available))
        
        # Generate supplementary numbers
        supp_nums = []
        if config.supplementary > 0:
            recent_supps = recent_data['supplementary'].str.split().explode().astype(int)
            supp_freqs = recent_supps.value_counts()
            
            available_supps = list(range(config.supp_range[0], config.supp_range[1]+1))
            weights = [supp_freqs.get(n, 1) for n in available_supps]
            total = sum(weights)
            probs = [w/total for w in weights]
            
            supp_nums = list(np.random.choice(
                available_supps, 
                size=config.supplementary, 
                p=probs,
                replace=False
            ))
        
        # Calculate probability (approximate based on historical frequencies)
        prob = 1.0
        for n in main_nums:
            freq = all_freqs.get(n, 1) / len(data)
            prob *= max(freq, 0.0001)
        
        predictions.append({
            "main": sorted(main_nums),
            "supp": sorted(supp_nums),
            "probability": float(prob),
            "confidence_score": float(np.log(max(prob, 1e-10))),
            "model": "pattern_based"
        })
    
    return predictions

def generate_random_predictions(config, prediction_count=10):
    """Fallback random predictions"""
    return [{
        "main": sorted(random.sample(range(config.main_range[0], config.main_range[1]+1), config.main_numbers)),
        "supp": sorted(random.sample(range(config.supp_range[0], config.supp_range[1]+1), config.supplementary)),
        "probability": 0.0001,
        "confidence_score": -9.0,
        "model": "random"
    } for _ in range(prediction_count)]

def advanced_prediction_engine(data, config, prediction_count=10):
    """Advanced prediction system using multiple models"""
    if data.empty:
        logger.warning("Empty dataset for predictions")
        return generate_random_predictions(config, prediction_count)
    
    try:
        # Use three different prediction methods
        predictions = []
        
        # 1. Time-weighted frequency model (recent draws have higher weight)
        time_weighted_preds = time_weighted_frequency_model(data, config, prediction_count // 3)
        predictions.extend(time_weighted_preds)
        
        # 2. Pattern-based model using trends
        pattern_preds = pattern_based_predictions(data, config, prediction_count // 3)
        predictions.extend(pattern_preds)
        
        # 3. Standard frequency model for remaining predictions
        simple_preds = simple_frequency_model(data, config, 
                                             prediction_count - len(predictions))
        predictions.extend(simple_preds)
        
        # Ensure we have exactly the requested number of predictions
        return predictions[:prediction_count]
    
    except Exception as e:
        logger.error(f"Error in prediction: {str(e)}")
        return generate_random_predictions(config, prediction_count)

def evaluate_prediction_methods(historical_data, config, methods=['frequency', 'pattern', 'time_weighted']):
    """Evaluate different prediction methods using historical data"""
    if len(historical_data) < 50:
        logger.warning("Not enough historical data for proper evaluation")
        return {"insufficient_data": True}
    
    results = {}
    
    # Use past data to make predictions and test against known outcomes
    test_size = min(50, len(historical_data) // 4)
    train_data = historical_data.iloc[test_size:]
    test_data = historical_data.iloc[:test_size]
    
    for method in methods:
        hits = []
        for i in range(len(test_data)):
            # Create training set up to this point
            train_subset = pd.concat([train_data, test_data.iloc[i+1:]])
            
            # Generate predictions
            if method == 'frequency':
                preds = simple_frequency_model(train_subset, config, 10)
            elif method == 'pattern':
                preds = pattern_based_predictions(train_subset, config, 10)
            elif method == 'time_weighted':
                preds = time_weighted_frequency_model(train_subset, config, 10)
            
            # Test against actual draw
            actual_draw = test_data.iloc[i]
            actual_main = set([int(n) for n in actual_draw['main_numbers'].split()])
            
            # Calculate matches for each prediction
            for pred in preds:
                pred_main = set(pred['main'])
                matches = len(actual_main.intersection(pred_main))
                hits.append(matches)
        
        results[method] = {
            'avg_matches': sum(hits) / len(hits),
            'max_matches': max(hits),
            'full_matches': hits.count(config.main_numbers),
            'match_distribution': {i: hits.count(i) for i in range(config.main_numbers + 1)}
        }
    
    return results

def generate_detailed_report(results, lottery_type, df, config, output_dir=None):
    """Create a detailed text report of analysis results"""
    if output_dir is None:
        output_dir = Path(f"results/analysis/reports")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d")
    report_path = output_dir / f"{lottery_type}_detailed_analysis_{timestamp}.txt"
    
    with open(report_path, "w") as f:
        # Title and basic information
        f.write(f"{'='*80}\n")
        f.write(f"{lottery_type.replace('_', ' ').title()} Lottery Analysis Report\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"{'='*80}\n\n")
        
        # Dataset info
        f.write(f"Dataset Information:\n")
        f.write(f"{'_'*40}\n")
        f.write(f"Date Range: {results['basic_stats']['date_range'][0]} to {results['basic_stats']['date_range'][1]}\n")
        f.write(f"Total Draws: {results['basic_stats']['draws']}\n")
        f.write(f"Lottery Configuration: {config.main_numbers} main numbers from {config.main_range}, ")
        f.write(f"{config.supplementary} supplementary from {config.supp_range}\n\n")
        
        # Frequency Analysis
        f.write(f"Number Frequency Analysis:\n")
        f.write(f"{'_'*40}\n")
        
        # Main numbers
        f.write(f"Top 20 Main Numbers by Frequency:\n")
        top_nums = sorted(results['frequencies'].items(), key=lambda x: x[1], reverse=True)[:20]
        max_freq = top_nums[0][1] if top_nums else 0
        for num, freq in top_nums:
            # Create a visual bar using asterisks
            bar = '*' * int(50 * freq / max_freq) if max_freq > 0 else ''
            f.write(f"Number {num:2d}: {freq:3d} times {bar}\n")
        f.write("\n")
        
        # Supplementary numbers if applicable
        if 'supp_frequencies' in results:
            f.write(f"Top 10 Supplementary Numbers by Frequency:\n")
            top_supps = sorted(results['supp_frequencies'].items(), key=lambda x: x[1], reverse=True)[:10]
            max_freq = top_supps[0][1] if top_supps else 0
            for num, freq in top_supps:
                bar = '*' * int(50 * freq / max_freq) if max_freq > 0 else ''
                f.write(f"Number {num:2d}: {freq:3d} times {bar}\n")
            f.write("\n")
        
        # Statistical Analysis
        f.write(f"Statistical Analysis:\n")
        f.write(f"{'_'*40}\n")
        f.write(f"Most frequently drawn main number: {results['basic_stats']['most_frequent']['number']} ")
        f.write(f"({results['basic_stats']['most_frequent']['count']} times)\n")
        f.write(f"Least frequently drawn main number: {results['basic_stats']['least_frequent']['number']} ")
        f.write(f"({results['basic_stats']['least_frequent']['count']} times)\n\n")
        
        # Randomness Testing
        f.write(f"Randomness Testing Results:\n")
        f.write(f"Chi-Square Statistic: {results['randomness_tests']['chi2_stat']:.2f}\n")
        f.write(f"P-Value: {results['randomness_tests']['p_value']:.4f}\n")
        interpretation = "The distribution appears to be random (null hypothesis not rejected)" \
            if results['randomness_tests']['is_random'] else \
            "The distribution shows significant deviation from randomness"
        f.write(f"Interpretation: {interpretation}\n\n")
        
        # Number Patterns and Clusters
        f.write(f"Number Patterns and Clusters:\n")
        f.write(f"{'_'*40}\n")
        f.write(f"Number groups that frequently appear together:\n")
        for i, cluster_numbers in enumerate(results['clusters']['cluster_groups']):
            f.write(f"Group {i+1}: {cluster_numbers}\n")
        f.write("\n")
        
        # Common Pairs
        f.write(f"Most Common Number Pairs:\n")
        for pair_str, count in list(results['common_pairs'].items())[:10]:
            f.write(f"{pair_str}: {count} times\n")
        f.write("\n")
        
        # Time Series Analysis
        if 'time_patterns' in results:
            f.write(f"Temporal Analysis:\n")
            f.write(f"{'_'*40}\n")
            f.write(f"Trend direction: {results['time_patterns'].get('trend_direction', 'unknown')}\n")
            f.write(f"Seasonality detected: {'Yes' if results['time_patterns'].get('has_seasonality', False) else 'No'}\n\n")
        
        # Recent Draws
        f.write(f"Recent Draw History (Last 10 Draws):\n")
        f.write(f"{'_'*40}\n")
        recent = df.head(10)
        for _, row in recent.iterrows():
            f.write(f"Date: {row['date']}, Draw: {row['draw']}, ")
            f.write(f"Main: {row['main_numbers']}, Supp: {row['supplementary']}\n")
        
        # Recommendations
        f.write(f"\n\nAnalysis Conclusions:\n")
        f.write(f"{'_'*40}\n")
        
        # Add some conclusions based on the data
        if not results['randomness_tests']['is_random']:
            f.write("• The draw distribution shows some non-random patterns that might be exploitable\n")
        else:
            f.write("• The draw distribution appears random, consistent with expected lottery behavior\n")
        
        # High/low number balance
        main_nums = sorted(results['frequencies'].items(), key=lambda x: x[0])
        mid_point = (config.main_range[1] + config.main_range[0]) // 2
        low_nums_freq = sum(freq for num, freq in main_nums if num <= mid_point)
        high_nums_freq = sum(freq for num, freq in main_nums if num > mid_point)
        total_freq = low_nums_freq + high_nums_freq
        
        if abs(low_nums_freq/total_freq - 0.5) > 0.05:  # More than 5% deviation
            if low_nums_freq > high_nums_freq:
                f.write("• Lower numbers (1-22) appear more frequently than higher numbers\n")
            else:
                f.write("• Higher numbers (23-45) appear more frequently than lower numbers\n")
        else:
            f.write("• Lower and higher numbers appear with similar frequencies\n")
            
        # Check if numbers in the same cluster appear together more often
        f.write("• Consider using numbers from the same cluster groups for better correlation\n")
        
        # Add disclaimer
        f.write(f"\n{'='*80}\n")
        f.write("DISCLAIMER: This analysis is for entertainment purposes only. Lottery outcomes\n")
        f.write("are random and no prediction system can guarantee winnings.\n")
        f.write(f"{'='*80}\n")
    
    return report_path

def enhanced_analysis(df, lottery_type, config, visualize=True, save_report=True):
    """Enhanced lottery analysis with advanced statistical methods"""
    results = {
        'basic_stats': {},
        'randomness_tests': {},
        'patterns': {},
        'clusters': {},
        'conditional_probs': {},
    }
    
    # Basic statistics
    results['basic_stats']['draws'] = len(df)
    results['basic_stats']['date_range'] = (df['date'].min(), df['date'].max())
    
    # Frequency analysis
    main_numbers = df['main_numbers'].str.split().explode().astype(int)
    # Convert to standard Python types for JSON serialization
    results['frequencies'] = {int(k): int(v) for k, v in main_numbers.value_counts().to_dict().items()}
    
    if config.supplementary > 0:
        supp_numbers = df['supplementary'].str.split().explode().astype(int)
        results['supp_frequencies'] = {int(k): int(v) for k, v in supp_numbers.value_counts().to_dict().items()}
    
    # Number distribution analysis
    main_range = range(config.main_range[0], config.main_range[1] + 1)
    all_main_counts = main_numbers.value_counts().reindex(main_range, fill_value=0)
    
    results['basic_stats']['most_frequent'] = {
        'number': int(all_main_counts.idxmax()),
        'count': int(all_main_counts.max())
    }
    
    results['basic_stats']['least_frequent'] = {
        'number': int(all_main_counts.idxmin()),
        'count': int(all_main_counts.min())
    }
    
    # Add randomness testing
    randomness_tests = test_draw_randomness(df, lottery_type, config)
    # Convert numpy types to standard Python types
    results['randomness_tests'] = {
        'chi2_stat': float(randomness_tests['chi2_stat']),
        'p_value': float(randomness_tests['p_value']),
        'is_random': bool(randomness_tests['is_random'])
    }
    
    # Add number cluster analysis
    cluster_results = find_number_clusters(df, lottery_type, config)
    results['clusters'] = {
        'cluster_groups': [[int(n) for n in group] for group in cluster_results['cluster_groups']]
    }
    
    # Time analysis 
    if 'date' in df.columns:
        df_copy = df.copy()
        df_copy['date'] = pd.to_datetime(df_copy['date'])
        df_copy['year'] = df_copy['date'].dt.year
        yearly_counts = df_copy.groupby('year').size()
        results['yearly_counts'] = {int(k): int(v) for k, v in yearly_counts.to_dict().items()}
        
        # Add time series analysis if enough data
        if len(df) > 50:
            time_patterns = analyze_time_patterns(df, lottery_type, config)
            results['time_patterns'] = {
                'trend_direction': time_patterns.get('trend_direction', 'unknown'),
                'has_seasonality': bool(time_patterns.get('has_seasonality', False))
            }
    
    # Common pairs analysis
    pairs = []
    for _, row in df.iterrows():
        numbers = [int(n) for n in row['main_numbers'].split() if n.strip()]
        for j in range(len(numbers)):
            for k in range(j+1, len(numbers)):
                pairs.append(tuple(sorted([numbers[j], numbers[k]])))
    
    pair_counts = pd.Series(pairs).value_counts()
    results['common_pairs'] = {str(pair): int(count) for pair, count in pair_counts.head(10).items()}
    
    # Conditional probability analysis - skip for now to simplify
    # results['conditional_probs'] = calculate_conditional_probabilities(df, lottery_type, config)
    
    # Generate visualizations if requested
    if visualize:
        if visualize == "interactive":
            generate_advanced_visualizations(df, lottery_type, config)
        else:
            # Create base directory for visualizations
            output_dir = Path(f"results/analysis/frequency_reports")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Number frequency visualization
            plt.figure(figsize=(12, 6))
            sns.barplot(x=all_main_counts.index, y=all_main_counts.values)
            plt.title(f"{lottery_type.replace('_', ' ').title()} Number Frequency")
            plt.xlabel("Number")
            plt.ylabel("Frequency")
            plt.xticks(rotation=90)
            plt.tight_layout()
            plt.savefig(output_dir / f"{lottery_type}_frequency.png", dpi=300)
            plt.close()
            
            # Generate correlation heatmap
            generate_correlation_heatmap(df, lottery_type, config, output_dir)
    
    # Create comprehensive reports if requested
    if save_report:
        output_dir = Path(f"results/analysis/reports")
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d")
        
        # Full JSON results
        with open(output_dir / f"{lottery_type}_analysis_{timestamp}.json", "w") as f:
            import json
            json.dump(results, f, indent=2)
        
        # Generate detailed text report
        txt_report_path = generate_detailed_report(results, lottery_type, df, config, output_dir)
        logger.info(f"Detailed text report saved to {txt_report_path}")
    
    return results
