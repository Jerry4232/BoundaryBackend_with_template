# geo_backend/geo_extractor.py

import os
import json
import argparse
import geopandas as gpd
from pathlib import Path
from dotenv import load_dotenv
from datetime import datetime, timedelta
import logging
from typing import Dict

from .extractors import *

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GeospatialDataExtractor:
    def __init__(self, aoi_shapefile: str, output_dir: str = "extracted_data", target_crs: str = "EPSG:4326"):
        self.aoi_shapefile = aoi_shapefile
        self.output_dir = Path(output_dir)
        self.target_crs = target_crs

        logger.info(f"Loading AOI from: {self.aoi_shapefile}")
        self.aoi_gdf = gpd.read_file(self.aoi_shapefile)
        if self.aoi_gdf.crs is None:
            logger.warning("AOI file has no CRS defined. Assuming EPSG:4326 (WGS84).")
            self.aoi_gdf.set_crs(self.target_crs, inplace=True)
        elif self.aoi_gdf.crs.to_string() != self.target_crs:
            logger.info(f"Reprojecting AOI from {self.aoi_gdf.crs.to_string()} to {self.target_crs}")
            self.aoi_gdf = self.aoi_gdf.to_crs(self.target_crs)

        self.aoi_bounds = self.aoi_gdf.total_bounds.tolist()
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.extractor_map = {
            "landsat": self._init_landsat,
            "sentinel": self._init_copernicus,
        }

    def _init_landsat(self):
        username = os.getenv("LANDSAT_USERNAME")
        token = os.getenv("LANDSAT_API_KEY")
        return LandsatExtractor(aoi_bounds=self.aoi_bounds, username=username, token=token, output_dir=str(self.output_dir))

    def _init_copernicus(self):
        username = os.getenv("COPERNICUS_USERNAME")
        password = os.getenv("COPERNICUS_PASSWORD")
        return CopernicusExtractor(aoi_bounds=self.aoi_bounds, username=username, password=password, output_dir=str(self.output_dir))

    def run(self, data_types: list[str], date_range: DateRange) -> dict:
        results = {}
        logger.info("Starting data extraction for selected types...")
        for dtype in data_types:
            extractor_fn = self.extractor_map.get(dtype)
            if extractor_fn is None:
                logger.warning(f"Unknown data type: {dtype}")
                continue
            try:
                extractor = extractor_fn()
                logger.info(f"Running extractor for: {dtype}")
                result = extractor.run(date_range)
                results[dtype] = result
            except Exception as e:
                logger.error(f"Extraction failed for {dtype}: {str(e)}")
                results[dtype] = {
                    "source": dtype,
                    "dataset": dtype,
                    "status": "error",
                    "message": f"Failed to extract {dtype} data",
                    "products_found": 0,
                    "products": [],
                    "error": str(e)
                }
        report_path = self.create_extraction_report(results)
        results["report_path"] = report_path

        logger.info("Full extraction completed!")
        return results

    def create_extraction_report(self, extraction_results: Dict) -> str:
        """
        Create a comprehensive report of extraction results

        Args:
            extraction_results: Dictionary containing extraction results

        Returns:
            Path to the generated report
        """
        try:
            report_path = self.output_dir / "extraction_report.md"

            with open(report_path, 'w') as f:
                f.write("# Geospatial Data Extraction Report\n\n")
                f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                f.write(f"**AOI Bounds:** {self.aoi_bounds}\n")
                f.write(f"**Target CRS:** {self.target_crs}\n")

                f.write("## Extraction Results\n\n")
                for data_type, results in extraction_results.items():
                    f.write(f"### {data_type.upper()}\n")
                    if isinstance(results, dict):
                        for key, value in results.items():
                            f.write(f"- **{key}:** {value}\n")
                    else:
                        f.write(f"- {results}\n")
                    f.write("\n")

                f.write("## Files Generated\n\n")
                for file_path in self.output_dir.glob("*"):
                    if file_path.is_file():
                        f.write(f"- {file_path.name}\n")

            logger.info(f"Extraction report saved to {report_path}")
            return str(report_path)

        except Exception as e:
            logger.error(f"Error creating extraction report: {str(e)}")
            return ""


def main():
    parser = argparse.ArgumentParser(description='Geospatial Data Extraction and Resampling')
    parser.add_argument('aoi_shapefile', help='Path to AOI shapefile')
    parser.add_argument('--output-dir', default='extracted_data', help='Output directory')
    parser.add_argument('--target-crs', default='EPSG:4326', help='Target CRS')
    parser.add_argument('--target-resolution', type=float, default=30.0, help='Target resolution in meters')
    parser.add_argument('--data-types', nargs='+', choices=['sentinel', 'landsat'], help='Data types to extract')

    args = parser.parse_args()

    extractor = GeospatialDataExtractor(
        aoi_shapefile=args.aoi_shapefile,
        output_dir=args.output_dir,
        target_crs=args.target_crs
    )

    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    date_range = DateRange(start=start_date, end=end_date)

    results = extractor.run(data_types=args.data_types, date_range=date_range)

    report_path = Path(args.output_dir) / "extraction_summary.json"
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Extraction completed! Summary saved to: {report_path}")

if __name__ == "__main__":
    main()
