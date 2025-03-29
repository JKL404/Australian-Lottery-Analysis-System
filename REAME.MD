
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

### Scrape Recent Data
```bash
python lotto_cli.py --lottery saturday_lotto --action scrape --years 5
```

### Analyze Lottery Patterns
```bash
python lotto_cli.py --lottery saturday_lotto --action analyze --visualize
```

### Generate Predictions
```bash
python lotto_cli.py --lottery saturday_lotto --action predict --samples 20
```

### Batch Processing
```bash
python lotto_cli.py --lottery saturday_lotto --action batch
```

## 📊 Analysis Features

- Number frequency analysis
- Hot and cold number identification
- Number pair correlation detection
- Time-based pattern analysis
- Visualization through heatmaps and charts

## ⚙️ Configuration

Each lottery type is configured with appropriate parameters:
- Number of selections required
- Range of possible numbers
- Supplementary numbers
- Draw schedule

## 📝 License

[Your License]

## 📧 Contact

[Your Contact Information]

---

⚠️ **Disclaimer**: This tool is for entertainment and educational purposes only. Lottery outcomes are random and no prediction system can guarantee winnings.
