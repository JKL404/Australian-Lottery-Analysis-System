"""
Core lottery configuration and validation system
Handles all lottery types: Saturday Lotto, Powerball, Oz Lotto, etc.
"""
from dataclasses import dataclass
from datetime import datetime
import json
import logging
import pandas as pd
import argparse
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('LotteryCore')

@dataclass
class LotteryConfig:
    name: str
    base_url: str
    main_numbers: int
    supplementary: int
    main_range: tuple
    supp_range: tuple
    date_format: str = "%A %d %b %Y"
    result_pattern: str = r"(\d{1,2}\s+\w+\s+\d{4}).*Draw\s+(\d+)"
    draw_day: str = None  # Day of week the draw occurs


LOTTERY_TYPES = {
    "saturday_lotto": LotteryConfig(
        name="Saturday Lotto",
        base_url="https://www.ozlotteries.com/saturday-lotto/results",
        main_numbers=6,
        supplementary=2,
        main_range=(1, 45),
        supp_range=(1, 45),
        result_pattern=r"Saturday (\d{1,2} \w+ \d{4}).*Draw\s+(\d+)",
        date_format="%A %d %B %Y",
        draw_day="Saturday"
    ),
    "monday_lotto": LotteryConfig(
        name="Monday Lotto",
        base_url="https://www.ozlotteries.com/monday-lotto/results",
        main_numbers=6,
        supplementary=2,
        main_range=(1, 45),
        supp_range=(1, 45),
        date_format="%A %d %B %Y",
        draw_day="Monday"
    ),
    "wednesday_lotto": LotteryConfig(
        name="Wednesday Lotto",
        base_url="https://www.ozlotteries.com/wednesday-lotto/results",
        main_numbers=6,
        supplementary=2,
        main_range=(1, 45),
        supp_range=(1, 45),
        result_pattern=r"Wednesday (\d{1,2} \w+ \d{4}).*Draw\s+(\d+)",
        date_format="%A %d %B %Y",
        draw_day="Wednesday"
    ),
    "powerball": LotteryConfig(
        name="Powerball",
        base_url="https://www.ozlotteries.com/powerball/results",
        main_numbers=7,
        supplementary=1,
        main_range=(1, 35),
        supp_range=(1, 20),
        result_pattern=r"Thursday (\d{1,2} \w+ \d{4}).*Draw\s+(\d+)",
        date_format="%A %d %B %Y",
        draw_day="Thursday"
    ),
    "oz_lotto": LotteryConfig(
        name="Oz Lotto",
        base_url="https://www.ozlotteries.com/oz-lotto/results",
        main_numbers=7,
        supplementary=2,
        main_range=(1, 45),
        supp_range=(1, 45),
        result_pattern=r"Tuesday (\d{1,2} \w+ \d{4}).*Draw\s+(\d+)",
        date_format="%A %d %B %Y",
        draw_day="Tuesday"
    )
}


class LotteryValidator:
    def __init__(self, lottery_type: str):
        if lottery_type not in LOTTERY_TYPES:
            raise ValueError(f"Invalid lottery type. Choose from: {', '.join(LOTTERY_TYPES.keys())}")
        self.config = LOTTERY_TYPES[lottery_type]
    
    def validate_draw(self, draw: dict) -> bool:
        """Validate a single draw against lottery rules"""
        try:
            self._validate_numbers(draw['main_numbers'], 'main')
            self._validate_numbers(draw['supplementary'], 'supplementary')
            self._validate_date_format(draw['date'])
            
            # Validate draw day if configured
            if self.config.draw_day and 'date' in draw:
                self._validate_draw_day(draw['date'])
                
            return True
        except Exception as e:
            logger.warning(f"Validation failed: {str(e)}")
            return False
    
    def _validate_numbers(self, numbers: list, number_type: str):
        """Validate number counts and ranges"""
        expected_count = self.config.main_numbers if number_type == 'main' else self.config.supplementary
        valid_range = self.config.main_range if number_type == 'main' else self.config.supp_range
        
        if len(numbers) != expected_count:
            raise ValueError(f"Expected {expected_count} {number_type} numbers, got {len(numbers)}")
        
        if any(not (valid_range[0] <= n <= valid_range[1]) for n in numbers):
            raise ValueError(f"{number_type} numbers out of range {valid_range}")
        
        # Add duplicate check
        if len(numbers) != len(set(numbers)):
            raise ValueError(f"Duplicate numbers found in {number_type}")

    def _validate_date_format(self, date_str: str):
        """Validate and normalize date format"""
        # First try ISO format
        try:
            datetime.strptime(date_str, "%Y-%m-%d")
            return True
        except ValueError:
            pass
            
        # Try configured format
        try:
            datetime.strptime(date_str, self.config.date_format)
            return True
        except ValueError:
            pass
            
        # Try alternative formats
        for fmt in ["%d %B %Y", "%d %b %Y", "%A %d %B %Y", "%A %d %b %Y"]:
            try:
                datetime.strptime(date_str, fmt)
                return True
            except ValueError:
                continue
                
        raise ValueError(f"Invalid date format: {date_str}. Expected format: {self.config.date_format}")
    
    def _validate_draw_day(self, date_str: str):
        """Validate draw day using config without hardcoded patterns"""
        if not self.config.draw_day:
            return True
        
        try:
            # Try multiple date formats dynamically
            date_formats = [
                self.config.date_format,
                "%A %d %b %Y",  # Abbreviated month
                "%d %B %Y",     # Without weekday
                "%d %b %Y"      # Abbreviated without weekday
            ]
            
            date_obj = None
            for fmt in date_formats:
                try:
                    date_obj = datetime.strptime(date_str, fmt)
                    break
                except ValueError:
                    continue
                
            if not date_obj:
                return False
            
            actual_day = date_obj.strftime('%A')
            return actual_day == self.config.draw_day
        
        except Exception as e:
            logger.warning(f"Draw day validation error: {str(e)}")
            return False

    def normalize_date(self, date_str: str) -> str:
        """Convert any valid date format to ISO format (YYYY-MM-DD)"""
        try:
            # Try ISO format first
            date_obj = datetime.strptime(date_str, "%Y-%m-%d")
            return date_str
        except ValueError:
            # Try lottery-specific format
            try:
                date_obj = datetime.strptime(date_str, self.config.date_format)
                return date_obj.strftime("%Y-%m-%d")
            except ValueError:
                # Try alternative formats
                for fmt in ["%d %B %Y", "%d %b %Y", "%A %d %B %Y", "%A %d %b %Y"]:
                    try:
                        date_obj = datetime.strptime(date_str, fmt)
                        return date_obj.strftime("%Y-%m-%d")
                    except ValueError:
                        continue
                        
        # If we get here, all attempts failed
        raise ValueError(f"Could not parse date: {date_str}")
