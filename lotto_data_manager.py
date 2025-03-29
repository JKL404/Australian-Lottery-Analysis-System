"""
Centralized data management for lottery results
Handles: CSV/JSON formatting, historical data merging, and prediction storage
"""
import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime, timedelta
import os
import logging
from lotto_core import LOTTERY_TYPES, LotteryValidator

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('LottoDataManager')

class LottoDataManager:
    def __init__(self, output_root="results"):
        self.output_root = Path(output_root)
        self.validator = None
        self._setup_dirs()
        
    def _setup_dirs(self):
        """Create standardized directory structure"""
        dirs = ["historical/individual", "predictions", "analysis/frequency_reports", 
                "analysis/temporal_analysis", "analysis/patterns"]
        for d in dirs:
            (self.output_root / d).mkdir(parents=True, exist_ok=True)

    def save_dataset(self, data, lottery_type, dataset_type="historical"):
        """Save data in multiple formats with validation"""
        try:
            self.validator = LotteryValidator(lottery_type)
            config = LOTTERY_TYPES[lottery_type]
            
            # Validate all entries first
            validated = [self._validate_and_format(d, config) for d in data]
            
            # Create filename with date range
            dates = [d.get("date", "unknown") for d in validated if d]
            if not dates or all(d == "unknown" for d in dates):
                filename = f"{lottery_type}_{datetime.now().strftime('%Y%m%d')}"
            else:
                filename = f"{lottery_type}_{min(dates)}_{max(dates)}"
            
            # Save in both formats
            json_path = self._save_json(validated, filename, lottery_type)
            csv_path = self._save_csv(validated, filename, config, lottery_type)
            
            logger.info(f"Saved {len(validated)} records to {json_path} and {csv_path}")
            return f"{filename}.*"
        except Exception as e:
            logger.error(f"Error saving dataset: {str(e)}")
            raise

    def _validate_and_format(self, entry, config):
        """Validate and standardize data entry"""
        try:
            if self.validator.validate_draw(entry):
                return {
                    "lottery_type": config.name,
                    "date": entry["date"],
                    "draw": entry["draw"],
                    "main_numbers": sorted(entry["main_numbers"]),
                    "supplementary": sorted(entry["supplementary"]),
                    "source": "ozlotteries.com"
                }
            return None
        except Exception as e:
            logger.warning(f"Invalid entry skipped: {str(e)}")
            return None

    def _save_json(self, data, filename, lottery_type):
        try:
            # Create subfolder for lottery type inside 'individual'
            path = self.output_root / "historical" / "individual" / lottery_type / f"{filename}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(path, "w") as f:
                json.dump([d for d in data if d], f, indent=2)
            return path
        except Exception as e:
            logger.error(f"Error saving JSON: {str(e)}")
            raise

    def _save_csv(self, data, filename, config, lottery_type):
        try:
            # Create subfolder for lottery type inside 'individual'
            path = self.output_root / "historical" / "individual" / lottery_type / f"{filename}.csv"
            path.parent.mkdir(parents=True, exist_ok=True)
            
            # Filter out None values
            valid_data = [d for d in data if d]
            if not valid_data:
                logger.warning("No valid data to save to CSV")
                return None
                
            df = pd.DataFrame(valid_data)
            
            # Convert dates to a standard format with flexible parsing
            if 'date' in df.columns:
                try:
                    # Use 'mixed' format to handle different date formats
                    df['date'] = pd.to_datetime(df['date'], format='mixed', dayfirst=True)
                    df['date'] = df['date'].dt.strftime("%Y-%m-%d")
                except Exception as e:
                    logger.warning(f"Error standardizing dates, trying with custom parsing: {str(e)}")
                    # Fallback: Try to parse dates manually
                    parsed_dates = []
                    for date_str in df['date']:
                        try:
                            # Try multiple date formats
                            date_formats = [
                                "%A %d %B %Y",  # Monday 24 April 2023
                                "%A %d %b %Y",  # Monday 24 Apr 2023
                                "%d %b %Y",      # 24 Apr 2023
                                "%d %B %Y",      # 24 April 2023
                                "%Y-%m-%d"       # 2023-04-24
                            ]
                            
                            date_obj = None
                            for fmt in date_formats:
                                try:
                                    date_obj = datetime.strptime(date_str, fmt)
                                    break
                                except ValueError:
                                    continue
                            
                            if date_obj:
                                parsed_dates.append(date_obj.strftime("%Y-%m-%d"))
                            else:
                                parsed_dates.append(date_str)  # Keep original if parsing fails
                                
                        except Exception:
                            parsed_dates.append(date_str)  # Keep original if any error
                    
                    df['date'] = parsed_dates
            
            df["main_numbers"] = df["main_numbers"].apply(
                lambda x: " ".join(map(str, sorted(x))) + " "
            )
            df["supplementary"] = df["supplementary"].apply(
                lambda x: " ".join(map(str, sorted(x))) + " "
            )
            
            df.to_csv(path, index=False)
            return path
        except Exception as e:
            logger.error(f"Error saving CSV: {str(e)}")
            raise

    def merge_historical_data(self, lottery_type):
        """Combine all historical files for a lottery type"""
        try:
            config = LOTTERY_TYPES[lottery_type]
            all_files = []
            
            # Modified file collection to exclude temporary files
            search_paths = [
                (self.output_root / "historical" / lottery_type, "*.csv"),
                (self.output_root / "historical" / "individual" / lottery_type, "*.csv"),
            ]
            
            # Make sure directories exist
            for path, _ in search_paths:
                path.mkdir(parents=True, exist_ok=True)
            
            # Exclude temporary/backup files
            excluded_patterns = ['combined_20', 'historical_imported']
            all_files = [
                f for path, pattern in search_paths
                for f in path.glob(pattern) 
                if not any(p in f.name for p in excluded_patterns)
            ]
            
            # Prioritize combined.csv if exists
            combined_path = self.output_root / "historical" / lottery_type / "combined.csv"
            if combined_path.exists():
                all_files = [f for f in all_files if f.name != "combined.csv"]
                all_files.insert(0, combined_path)
            
            # Try import if no files found
            if not all_files:
                logger.warning(f"No historical data files found for {lottery_type}")
                source = Path(f"historical_data/{lottery_type}.csv")
                if source.exists():
                    logger.info(f"Found source file {source}, importing...")
                    return self._import_source_file(source, lottery_type)
                return pd.DataFrame()
            
            # Load and merge all files
            dfs = []
            for f in all_files:
                try:
                    df = pd.read_csv(f)
                    logger.info(f"Loaded {len(df)} records from {f}")
                    dfs.append(df)
                except Exception as e:
                    logger.warning(f"Error reading {f}: {str(e)}")
            
            if not dfs:
                logger.warning("No valid CSV files found")
                return pd.DataFrame()
            
            # Modified duplicate removal with enhanced validation
            combined = pd.concat(dfs, ignore_index=True)
            
            # Clean and validate data
            combined = combined.dropna(subset=['main_numbers', 'supplementary'])
            combined['date'] = pd.to_datetime(
                combined['date'],
                format='mixed',
                dayfirst=True,
                errors='coerce'
            )
            
            # Create unique hash for each draw
            combined['data_hash'] = combined.apply(
                lambda x: hash(
                    (x['date'], x['draw'], x['main_numbers'], x['supplementary'])
                ),
                axis=1
            )
            
            # Final clean dataset
            combined = combined.sort_values('date', ascending=False)
            combined = combined.drop_duplicates(
                subset=['data_hash', 'date', 'draw'], 
                keep='first'
            ).drop(columns=['data_hash'])
            
            if not combined.empty:
                # Save only the canonical combined file
                combined_path = self.output_root / "historical" / lottery_type / "combined.csv"
                combined.to_csv(combined_path, index=False)
                logger.info(f"Merged {len(combined)} valid records")
                return combined
            
            logger.error("No valid data remaining after merge")
            return None
        except Exception as e:
            logger.error(f"Error merging data: {str(e)}")
            return pd.DataFrame()

    def _import_source_file(self, source_path, lottery_type):
        """Import from a source file in a different format"""
        try:
            df = pd.read_csv(source_path)
            
            # Create lottery-specific directory
            lottery_dir = self.output_root / "historical" / lottery_type
            lottery_dir.mkdir(parents=True, exist_ok=True)
            
            # Format conversion based on observed source format
            data = []
            for _, row in df.iterrows():
                try:
                    # Try multiple date formats for more robust parsing
                    date_str = row["DRAW DATE"].strip() if "DRAW DATE" in row else row.get("date", "unknown")
                    date_obj = None
                    
                    # Try different date formats
                    date_formats = [
                        "%A %d %B %Y",  # Monday 24 April 2023
                        "%A %d %b %Y",  # Monday 24 Apr 2023
                        "%d %b %Y",     # 24 Apr 2023
                        "%d %B %Y",     # 24 April 2023
                        "%Y-%m-%d"      # 2023-04-24
                    ]
                    
                    for fmt in date_formats:
                        try:
                            date_obj = datetime.strptime(date_str, fmt)
                            break
                        except ValueError:
                            continue
                    
                    if date_obj is None:
                        logger.warning(f"Could not parse date: {date_str}")
                        formatted_date = "unknown"
                    else:
                        formatted_date = date_obj.strftime("%Y-%m-%d")
                    
                    entry = {
                        "date": formatted_date,
                        "draw": str(row["DRAW ID"]) if "DRAW ID" in row else str(row.get("draw", "0")),
                        "main_numbers": list(map(int, row["MAIN NUMBERS"].strip().split()))
                                         if "MAIN NUMBERS" in row else [],
                        "supplementary": list(map(int, row["SUPP"].strip().split()))
                                         if "SUPP" in row else []
                    }
                    data.append(entry)
                except Exception as e:
                    logger.warning(f"Error processing row: {str(e)}")
            
            # Simple filename without dates/times
            filename = f"historical_imported"
            
            # Save to lottery-specific directory
            json_path = lottery_dir / f"{filename}.json"
            with open(json_path, "w") as f:
                json.dump([d for d in data if d], f, indent=2)
            
            csv_path = lottery_dir / f"{filename}.csv"
            valid_data = [d for d in data if d]
            if valid_data:
                df = pd.DataFrame(valid_data)
                df['date'] = pd.to_datetime(df['date']).dt.strftime("%Y-%m-%d")
                df["main_numbers"] = df["main_numbers"].apply(
                    lambda x: " ".join(map(str, sorted(x))) + " "
                )
                df["supplementary"] = df["supplementary"].apply(
                    lambda x: " ".join(map(str, sorted(x))) + " "
                )
                df.to_csv(csv_path, index=False)
            
            logger.info(f"Imported {len(data)} records from {source_path} to {csv_path}")
            
            return self.merge_historical_data(lottery_type)
        except Exception as e:
            logger.error(f"Error importing source file: {str(e)}")
            return pd.DataFrame()

    def save_predictions(self, predictions, lottery_type, format="csv"):
        """Store prediction results with metadata"""
        try:
            # Add validation for all lottery types
            config = LOTTERY_TYPES[lottery_type]
            for pred in predictions:
                if len(pred["main"]) != config.main_numbers:
                    raise ValueError(f"{config.name} requires exactly {config.main_numbers} main numbers")
                if len(pred["supp"]) != config.supplementary:
                    raise ValueError(f"{config.name} requires exactly {config.supplementary} supplementary numbers")
                
            filename = f"{lottery_type}_predictions_{datetime.now().strftime('%Y%m%d')}.{format}"
            path = self.output_root / "predictions" / filename
            
            records = [{
                "generated": datetime.now().isoformat(),
                "combination": i+1,
                "main_numbers": " ".join(map(str, pred["main"])),
                "supplementary": " ".join(map(str, pred["supp"])) if pred["supp"] else "",
                "probability": pred["probability"],
                "confidence_score": pred.get("confidence_score", None)
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

# Example usage
if __name__ == "__main__":
    # Initialize with scraper results
    manager = LottoDataManager()
    
    # Sample data from scraper
    sample_data = [{
        "date": "2023-05-01",
        "draw": "4282",
        "main_numbers": [40, 37, 6, 39, 29, 26],
        "supplementary": [20, 38]
    }]
    
    # Save and merge
    manager.save_dataset(sample_data, "saturday_lotto")
    merged = manager.merge_historical_data("saturday_lotto")
    
    # Save predictions
    predictions = [{
        "main": [5, 10, 15, 20, 25, 30],
        "supp": [35, 40],
        "probability": 0.00015
    }]
    manager.save_predictions(predictions, "saturday_lotto")
