# geo_backend/extractors/copernicus_extractor.py

import os
import json
import requests
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import List, Dict

from .base import GeospatialDataExtractorBase, DateRange, logger

class CopernicusExtractor(GeospatialDataExtractorBase):
    def __init__(self, aoi_bounds: List[float], username: str, password: str, output_dir: str = "extracted_data"):
        super().__init__(aoi_bounds, output_dir)
        self.username = username
        self.password = password

        self.base_url = "https://catalogue.dataspace.copernicus.eu/odata/v1/"
        self.download_url = "https://zipper.dataspace.copernicus.eu/odata/v1/"
        self.auth_url = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
        self.user_guide = 'https://documentation.dataspace.copernicus.eu/'
        self.token = None

        self.authenticate()

    def authenticate(self):
        logger.info("Authenticating with Copernicus Data Space...")
        auth_data = {
            'grant_type': 'password',
            'username': self.username,
            'password': self.password,
            'client_id': 'cdse-public'
        }
        response = requests.post(
            self.auth_url,
            data=auth_data,
            headers={'Content-Type': 'application/x-www-form-urlencoded'}
        )
        if response.status_code == 200:
            self.token = response.json().get("access_token")
            if self.token:
                self.session = requests.Session()
                self.session.headers.update({"Authorization": f"Bearer {self.token}"})
                logger.info("Authentication successful.")
        else:
            raise RuntimeError(f"Authentication failed: {response.status_code} - {response.text}")

    def fetch(self, date_range: DateRange) -> Dict:
        satellite = "S2"
        bbox = self.aoi_bounds
        bbox_wkt = f"POLYGON(({bbox[0]} {bbox[1]}, {bbox[2]} {bbox[1]}, {bbox[2]} {bbox[3]}, {bbox[0]} {bbox[3]}, {bbox[0]} {bbox[1]}))"

        query_filter = (
            f"(ContentDate/Start ge {date_range.start.strftime('%Y-%m-%d')}T00:00:00.000Z and "
            f"ContentDate/Start le {date_range.end.strftime('%Y-%m-%d')}T23:59:59.999Z) and "
            f"OData.CSC.Intersects(area=geography'SRID=4326;{bbox_wkt}') and contains(Name,'{satellite}') and "
            f"contains(Name,'L2A') and Attributes/OData.CSC.DoubleAttribute/any(att:att/Name eq 'cloudCover' and att/OData.CSC.DoubleAttribute/Value le 20)"
        )

        url = self.base_url + "Products"
        response = self.session.get(url, params={"$filter": query_filter, "$top": "50"}, timeout=60)

        if response.status_code != 200:
            return {
                "source": "Copernicus Data Space",
                "dataset": satellite,
                "status": "error",
                "message": f"HTTP {response.status_code}",
                "products_found": 0,
                "products": [],
                "error": response.text
            }

        data = response.json().get("value", [])
        products = []
        for p in data:
            info = {
                "id": p.get("Id"),
                "name": p.get("Name"),
                "size": p.get("ContentLength"),
                "ingestion_date": p.get("IngestionDate"),
                "content_date": p.get("ContentDate", {}).get("Start"),
                "footprint": p.get("Footprint"),
                "online": p.get("Online"),
                "download_url": f"{self.download_url}Products({p.get('Id')})/$value"
            }
            products.append(info)

        json_path = Path(self.output_dir) / "copernicus_s2_products.json"
        csv_path = Path(self.output_dir) / "copernicus_s2_products.csv"
        with open(json_path, "w") as f:
            json.dump(products, f, indent=2)

        df = pd.DataFrame(products)
        df.to_csv(csv_path, index=False)

        return {
            "source": "Copernicus Data Space",
            "dataset": satellite,
            "status": "success",
            "message": f"Found {len(products)} products",
            "products_found": len(products),
            "products": products[:5],
            "json_path": str(json_path),
            "csv_path": str(csv_path),
            "bounds": self.aoi_bounds
        }

    def preprocess(self, raw_data: dict) -> dict:
        logger.info("Preprocessing is handled during fetch for Copernicus.")
        return raw_data

    def validate(self, processed_data: dict) -> dict:
        logger.info("Validating Copernicus data output...")
        if processed_data.get("status") == "success" and processed_data.get("products_found", 0) > 0:
            return processed_data
        raise ValueError("Validation failed: No products or fetch error.")

    def store(self, validated_data: dict) -> None:
        logger.info("Copernicus data already stored during fetch.")

    def run(self, date_range: DateRange) -> dict:
        raw = self.fetch(date_range)
        processed = self.preprocess(raw)
        validated = self.validate(processed)
        self.store(validated)
        return validated
