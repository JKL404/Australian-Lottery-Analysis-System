
# Australian Lottery Analysis System

A comprehensive tool for analyzing lottery data, visualizing patterns, and generating predictions for various Australian lottery games.

## 📋 Features

- **Multi-Lottery Support**: Analyzes Saturday Lotto, Powerball, Oz Lotto, Monday Lotto, and Wednesday Lotto
- **Data Collection**: Scrapes historical lottery data from official sources
- **Robust Analysis**: Frequency analysis, pattern detection, and trend visualization
- **Prediction Generation**: Statistical models to generate potential number combinations
- **Visualization**: Charts and heatmaps showing number frequencies and patterns
- **Data Management**: Import, export, and merge historical lottery datasets

## 🚀 Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/lottery-analysis.git
cd lottery-analysis

# Install dependencies
pip install -r requirements.txt
```

## 📁 Project Structure

```
lotto_analysis/
├── lotto_cli.py          # Command-line interface
├── lotto_core.py         # Core lottery definitions and validators
├── lotto_data_manager.py # Data processing and storage
├── lotto_scraper.py      # Web scraping functionality
├── lotto_analytics.py    # Advanced analysis and prediction algorithms
├── historical/           # Historical lottery data
│   ├── monday_lotto.csv
│   ├── oz_lotto.csv
│   ├── powerball.csv
│   ├── saturday_lotto.csv
│   └── wednesday_lotto.csv
└── results/              # Generated outputs
    ├── historical/       # Processed historical data
    ├── predictions/      # Generated number predictions
    └── analysis/         # Analysis reports and visualizations
```

## 🔧 Usage

### Import Historical Data
```bash
python lotto_cli.py --lottery saturday_lotto --action import
```

### Scr ape Recent Data
```bash
python lotto_cli.py --lottery saturday_lotto --action scrape --years 5
```

### Analyze Lottery Patterns
```bash
# Text-only analysis output to console
python lotto_cli.py --lottery saturday_lotto --action analyze

# Generate visualizations (charts, heatmaps, etc.)
python lotto_cli.py --lottery saturday_lotto --action analyze --visualize

# Generate interactive visualizations
python lotto_cli.py --lottery saturday_lotto --action analyze --visualize --interactive
```

### Generate Predictions
```bash
# Default 10 predictions in CSV format
python lotto_cli.py --lottery saturday_lotto --action predict

# Custom number of predictions in JSON format
python lotto_cli.py --lottery saturday_lotto --action predict --prediction-count 20 --format json

# Use specific prediction model
python lotto_cli.py --lottery saturday_lotto --action predict --model time_weighted
```

### Evaluate Prediction Models
```bash
# Compare performance of different prediction models
python lotto_cli.py --lottery saturday_lotto --action evaluate
```

### Batch Processing
```bash
# Run complete workflow: scrape, analyze, predict, evaluate
python lotto_cli.py --lottery saturday_lotto --action batch

# Historical Data + With Latest Data (last 2 Years)
python lotto_cli.py --lottery powerball --action batch --years 2
```

## 📊 Analysis Features

- Number frequency analysis
- Hot and cold number identification
- Number pair correlation detection
- Time-based pattern analysis
- Visualization through heatmaps and charts
- Statistical randomness testing
- Number cluster detection using machine learning
- Time-series decomposition for trend analysis

## 📈 Results

### Analysis Results
- **Detailed Text Reports**: Comprehensive analysis with statistics, patterns, and recommendations (`results/analysis/reports/lottery_type_detailed_analysis_YYYYMMDD.txt`)
- **Frequency Reports**: Visualizations of number frequencies as heatmaps and bar charts (`results/analysis/frequency_reports/`)
- **Temporal Analysis**: Time-series charts showing draw patterns and trends (`results/analysis/temporal_analysis/`)
- **Number Distribution**: Charts showing the distribution of drawn numbers across the possible range
- **Common Pairs**: Reports on frequently occurring number pairs
- **Interactive Visualizations**: Web-based interactive charts for detailed exploration (`results/analysis/interactive/`)
- **JSON Data**: Machine-readable structured data for further analysis (`results/analysis/reports/lottery_type_analysis_YYYYMMDD.json`)

### Prediction Results
- **CSV Format**: Structured predictions with probabilities, confidence scores, and prediction method (`results/predictions/lottery_type_YYYYMMDD.csv`)
- **JSON Format**: Detailed prediction data with metadata (`results/predictions/lottery_type_YYYYMMDD.json`)
- **Model Evaluation**: Performance reports comparing prediction methods (`results/analysis/evaluation/`)

### Historical Data
- **Combined Data**: Merged historical datasets (`results/historical/combined_results.csv`)
- **Individual Files**: Processed lottery-specific data files (`results/historical/individual/`)

## 🧠 Prediction Methods

- **Frequency-Based**: Traditional prediction based on historical frequency
- **Time-Weighted**: Enhanced predictions giving more weight to recent draws
- **Pattern-Based**: Predictions leveraging detected number patterns and clusters
- **Advanced (Ensemble)**: Combination of multiple models for optimal results

## ⚙️ Configuration

Each lottery type is configured with appropriate parameters:
- Number of selections required
- Range of possible numbers
- Supplementary numbers
- Draw schedule

## 🎯 Advanced Features

### Interactive Visualizations
Generate interactive web-based charts and heatmaps that allow for detailed exploration of lottery patterns.

### Statistical Testing
Apply chi-square tests and other statistical methods to validate the randomness of lottery draws.

### Model Evaluation
Compare the performance of different prediction models against historical data to determine the most effective approach.

### NOTE:
While running this app, it will open the Chrome browser to fetch the latest data from the lottery website. For some regions, it may display data that is not available in your state (especially Queensland). In such cases, you can quickly switch to another region (New South Wales) within 2-3 seconds of the browser loading.

---

⚠️ **Disclaimer**: This tool is for entertainment and educational purposes only. Lottery outcomes are random and no prediction system can guarantee winnings.
