# geo_backend/extractors/landsat_extractor.py

import requests
import json
import os
import pandas as pd
from datetime import datetime
from typing import Tuple

from .base import GeospatialDataExtractorBase, logger, DateRange

class LandsatExtractor(GeospatialDataExtractorBase):
    def __init__(self, aoi_bounds: list[float], username: str, token: str, output_dir: str="extracted_data"):
        super().__init__(aoi_bounds, output_dir)
        self.username = username
        self.token = token
        self.service_url = "https://m2m.cr.usgs.gov/api/api/json/stable/"
        self.dataset = "landsat_ot_c2_l2"

        # USER LOGIN
        login_payload = {"username": self.username, "token": self.token}
        login_response = requests.post(self.service_url + "login-token", json=login_payload)
        if login_response.status_code == 200:
            self.api_key = login_response.json()["data"]
            logger.info("USGS login successful.")
        else:
            raise RuntimeError(f"USGS login failed: {login_response.text}")

    def fetch(self, date_range: DateRange) -> dict:
        logger.info("Fetching Landsat scenes from USGS M2M API...")
        spatial_filter = {
            "filterType": "mbr",
            "lowerLeft": {"latitude": self.aoi_bounds[1], "longitude": self.aoi_bounds[0]},
            "upperRight": {"latitude": self.aoi_bounds[3], "longitude": self.aoi_bounds[2]}
        }

        temporal_filter = {
            "start": date_range.start.strftime("%Y-%m-%d"),
            "end": date_range.end.strftime("%Y-%m-%d")
        }
        payload = {
            "datasetName": self.dataset,
            "spatialFilter": spatial_filter,
            "temporalFilter": temporal_filter
        }

        headers = {"X-Auth-Token": self.api_key}
        response = requests.post(self.service_url + "scene-search", json=payload, headers=headers)
        result = response.json()
        output_path = os.path.join(self.output_dir, "scene_search_result.json")
        logger.info(f"Saving raw data to {output_path}")

        with open(output_path, "w") as f:
            json.dump(result, f, indent=2)
        return result

    def preprocess(self, raw_data):
        logger.info("Preprocessing Landsat scene metadata...")
        scene_results = raw_data["data"]["results"]
        entries = []
        for scene in scene_results:
            scene_id = scene.get("displayId")
            metadata = {item["fieldName"]: item["value"] for item in scene["metadata"]}
            entries.append({"id": scene_id, **metadata})

        df = pd.DataFrame(entries)
        csv_path = os.path.join(self.output_dir, "scene_search_result.csv")
        df.to_csv(csv_path, index=False)
        return df

    def validate(self, processed_data):
        logger.info("Validating Landsat scene metadata...")
        # Simple check: ensure non-empty dataframe with expected columns
        required_cols = ["id"]
        if all(col in processed_data.columns for col in required_cols) and not processed_data.empty:
            return processed_data
        else:
            raise ValueError("Validation failed: Required columns missing or no data.")

    def store(self, validated_data):
        logger.info("Storing validated Landsat metadata...")
        validated_data.to_csv(os.path.join(self.output_dir, "validated_landsat_data.csv"), index=False)
