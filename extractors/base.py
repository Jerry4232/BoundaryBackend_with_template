
# utils
from dataclasses import dataclass
from datetime import datetime

@dataclass
class DateRange:
    """
    A class to represent a date range.

    # Construct example:
    date_range = DateRange(
        start=datetime(2022, 1, 1), 
        end=datetime(2022, 1, 31))
    """
    start: datetime
    end: datetime

    def __post_init__(self):
        if not isinstance(self.start, datetime) or not isinstance(self.end, datetime):
            raise TypeError("Both start and end must be datetime objects.")
        if self.start >= self.end:
            raise ValueError("Start date must be earlier than end date.")
        

# geo_backend/extractors/base.py

from abc import ABC, abstractmethod
import os
import logging
# from geobackend.utils import DateRange

logger = logging.getLogger(__name__)
logger.info = print  # For simplicity, redirect logger info to print`

class GeospatialDataExtractorBase(ABC):
    def __init__(self, aoi_bounds: list[float], output_dir: str ="extracted_data"):
        self.aoi_bounds = aoi_bounds  # [minx, miny, maxx, maxy]
        if len(aoi_bounds) != 4:
            raise ValueError("AOI bounds must be a list of four floats: [minx, miny, maxx, maxy]")
        if not all(isinstance(coord, (int, float)) for coord in aoi_bounds):
            raise TypeError("AOI bounds must contain only numeric values.")
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    @abstractmethod
    def fetch(self, date_range: DateRange) -> dict:
        pass

    @abstractmethod
    def preprocess(self, raw_data: dict) -> dict:
        pass

    @abstractmethod
    def validate(self, processed_data: dict) -> dict:
        pass

    @abstractmethod
    def store(self, validated_data: dict) -> None:
        pass

    def run(self, date_range: DateRange) -> dict:
        raw = self.fetch(date_range)
        processed = self.preprocess(raw)
        validated = self.validate(processed)
        self.store(validated)
        return validated
