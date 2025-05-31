from .extractors.landsat_extractor import LandsatExtractor, DateRange
from datetime import datetime
import os

# temporary local development environment variables
from dotenv import load_dotenv
load_dotenv()
# Ensure that the environment variables are set for Landsat API access

LANDSAT_USERNAME = os.getenv("LANDSAT_USERNAME")
LANDSAT_API_KEY = os.getenv("LANDSAT_API_KEY")

if not LANDSAT_API_KEY:
    raise EnvironmentError("Missing LANDSAT_API_KEY in environment variables.")


def run_landsat_extractor():
    """Run the Landsat data extractor with specified area of interest and user credentials."""
    aoi_bounds = [-120.0, 35.0, -119.0, 36.0]
    print(f"Using username: {LANDSAT_USERNAME} and token: {LANDSAT_API_KEY}")
    extractor = LandsatExtractor(aoi_bounds, LANDSAT_USERNAME, LANDSAT_API_KEY)
    extractor.run(date_range=DateRange(
        start=datetime(2022, 1, 1),
        end=datetime(2022, 1, 31)
    ))

if __name__ == "__main__":
    run_landsat_extractor()
    print("Landsat data extraction completed.")
