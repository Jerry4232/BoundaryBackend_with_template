
import os
import sys
import json
import warnings
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional, Union
import concurrent.futures
from functools import partial

# Core geospatial libraries
import geopandas as gpd
import rasterio
from rasterio.mask import mask
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.merge import merge
import xarray as xr
import rioxarray
import numpy as np
import pandas as pd
from shapely.geometry import box, Polygon
import pyproj
from pyproj import CRS, Transformer

# API and web libraries
import requests
from urllib.parse import urljoin
from owslib.wcs import WebCoverageService
from owslib.wms import WebMapService

# Visualization
import matplotlib.pyplot as plt
import folium

# Suppress warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GeospatialDataExtractor:
    """
    Main class for extracting and resampling geospatial data from multiple sources
    """
            
    def _load_aoi(self):
        """Loads AOI from a shapefile and sets internal variables for geometry and bounds."""
        try:
            self.aoi_gdf = gpd.read_file("geo_backend\\" + self.aoi_shapefile)
            
            if self.aoi_gdf.empty:
                raise ValueError("AOI shapefile is empty or invalid.")

            # Reproject to target CRS if needed
            if self.aoi_gdf.crs is None:
                self.aoi_gdf.set_crs(self.target_crs, inplace=True)
            elif self.aoi_gdf.crs.to_string() != self.target_crs:
                self.aoi_gdf = self.aoi_gdf.to_crs(self.target_crs)

            self.aoi_bounds = self.aoi_gdf.total_bounds
            logger.info(f"AOI loaded successfully. Bounds: {self.aoi_bounds}")
        
        except Exception as e:
            logger.error(f"Failed to load AOI: {str(e)}")
            raise

    def __init__(self, aoi_shapefile: str, output_dir: str = "extracted_data", 
                 target_crs: str = "EPSG:4326", target_resolution: float = 30.0):
        """
        Initialize the data extractor
        
        Args:
            aoi_shapefile: Path to AOI shapefile
            output_dir: Output directory for extracted data
            target_crs: Target coordinate reference system
            target_resolution: Target resolution in meters
        """
        self.aoi_shapefile = aoi_shapefile
        self.output_dir = Path(output_dir)
        self.target_crs = target_crs
        self.target_resolution = target_resolution
        self.aoi_gdf = None
        self.aoi_bounds = None
        self.session = requests.Session()
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load AOI
        self._load_aoi()
        
        # API configurations
        self.api_configs = {
            'copernicus': {
                'base_url': 'https://catalogue.dataspace.copernicus.eu/odata/v1/',
                'download_url': 'https://zipper.dataspace.copernicus.eu/odata/v1/',
                'auth_url': 'https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token',
                'user_guide': 'https://documentation.dataspace.copernicus.eu/'
            },
            'usgs': {
                'base_url': 'https://m2m.cr.usgs.gov/api/api/json/stable/',
                'login_url': 'https://ers.cr.usgs.gov/login'
            },
            'nasa': {
                'base_url': 'https://cmr.earthdata.nasa.gov/search/',
                'data_url': 'https://e4ftl01.cr.usgs.gov/'
            },
            'opentopo': {
                'base_url': 'https://portal.opentopography.org/API/globaldem'
            }
        }
    
    def authenticate_copernicus(self, username: str = None, password: str = None) -> bool:
        """
        Authenticate with Copernicus Data Space Ecosystem
        
        Args:
            username: Copernicus username (if None, looks for env variable)
            password: Copernicus password (if None, looks for env variable)
            
        Returns:
            True if authentication successful
        """
        try:
            # Get credentials from environment if not provided
            if username is None:
                username = os.getenv('COPERNICUS_USERNAME')
            if password is None:
                password = os.getenv('COPERNICUS_PASSWORD')
            
            if not username or not password:
                logger.warning("Copernicus credentials not provided. Set COPERNICUS_USERNAME and COPERNICUS_PASSWORD environment variables.")
                return False
            
            # OAuth2 authentication
            auth_data = {
                'grant_type': 'password',
                'username': username,
                'password': password,
                'client_id': 'cdse-public'
            }
            
            response = self.session.post(
                self.api_configs['copernicus']['auth_url'],
                data=auth_data,
                headers={'Content-Type': 'application/x-www-form-urlencoded'}
            )
            
            if response.status_code == 200:
                token_data = response.json()
                access_token = token_data.get('access_token')
                
                if access_token:
                    # Add token to session headers
                    self.session.headers.update({
                        'Authorization': f'Bearer {access_token}'
                    })
                    logger.info("Successfully authenticated with Copernicus Data Space")
                    return True
            
            logger.error(f"Copernicus authentication failed: {response.status_code}")
            return False
            
        except Exception as e:
            logger.error(f"Error authenticating with Copernicus: {str(e)}")
            return False
        """Load and validate AOI shapefile"""
        try:
            self.aoi_gdf = gpd.read_file(self.aoi_shapefile)
            
            # Ensure AOI is in target CRS
            if self.aoi_gdf.crs != self.target_crs:
                self.aoi_gdf = self.aoi_gdf.to_crs(self.target_crs)
            
            # Get bounds
            self.aoi_bounds = self.aoi_gdf.total_bounds
            logger.info(f"AOI loaded successfully. Bounds: {self.aoi_bounds}")
            
        except Exception as e:
            logger.error(f"Error loading AOI: {str(e)}")
            raise
    
    def visualize_aoi(self, save_map: bool = True):
        """Create an interactive map of the AOI"""
        try:
            # Calculate center point
            center_lat = (self.aoi_bounds[1] + self.aoi_bounds[3]) / 2
            center_lon = (self.aoi_bounds[0] + self.aoi_bounds[2]) / 2
            
            # Create folium map
            m = folium.Map(location=[center_lat, center_lon], zoom_start=10)
            
            # Add AOI to map
            folium.GeoJson(
                self.aoi_gdf.to_json(),
                style_function=lambda x: {
                    'fillColor': 'blue',
                    'color': 'black',
                    'weight': 2,
                    'fillOpacity': 0.3
                }
            ).add_to(m)
            
            if save_map:
                map_path = self.output_dir / "aoi_map.html"
                m.save(str(map_path))
                logger.info(f"AOI map saved to {map_path}")
            
            return m
            
        except Exception as e:
            logger.error(f"Error creating AOI visualization: {str(e)}")
            return None
    
    def extract_sentinel_data(self, satellite: str = "S2", date_range: Tuple[str, str] = None,
                            cloud_cover: int = 20, download_products: bool = False) -> Dict:
        """
        Extract Sentinel satellite data using Copernicus Data Space API
        
        Args:
            satellite: Satellite type (S1, S2, S3, S5P)
            date_range: Tuple of start and end dates (YYYY-MM-DD)
            cloud_cover: Maximum cloud cover percentage
            download_products: Whether to download actual product files
        """
        try:
            if date_range is None:
                end_date = datetime.now()
                start_date = end_date - timedelta(days=30)
                date_range = (start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
            
            # Construct spatial filter (WKT format)
            bbox_wkt = f"POLYGON(({self.aoi_bounds[0]} {self.aoi_bounds[1]}, {self.aoi_bounds[2]} {self.aoi_bounds[1]}, {self.aoi_bounds[2]} {self.aoi_bounds[3]}, {self.aoi_bounds[0]} {self.aoi_bounds[3]}, {self.aoi_bounds[0]} {self.aoi_bounds[1]}))"
            
            # Construct search query with proper OData syntax
            query_params = {
                '$filter': f"(ContentDate/Start ge {date_range[0]}T00:00:00.000Z and ContentDate/Start le {date_range[1]}T23:59:59.999Z) and OData.CSC.Intersects(area=geography'SRID=4326;{bbox_wkt}') and contains(Name,'{satellite}')",
                '$orderby': 'ContentDate/Start desc',
                '$top': '50'
            }
            
            # Add cloud cover filter for optical satellites
            if satellite in ['S2', 'S3']:
                cloud_filter = f" and Attributes/OData.CSC.DoubleAttribute/any(att:att/Name eq 'cloudCover' and att/OData.CSC.DoubleAttribute/Value le {cloud_cover})"
                query_params['$filter'] += cloud_filter
            
            # Add product type filters for specific satellites
            product_type_filters = {
                'S1': 'GRD',  # Ground Range Detected
                'S2': 'L2A',  # Level 2A (atmospherically corrected)
                'S3': 'OL_1_EFR',  # Ocean and Land Full Resolution
                'S5P': 'L2'   # Level 2 products
            }
            
            if satellite in product_type_filters:
                type_filter = f" and contains(Name,'{product_type_filters[satellite]}')"
                query_params['$filter'] += type_filter
            
            logger.info(f"Searching Copernicus for {satellite} products...")
            logger.debug(f"Query filter: {query_params['$filter']}")
            
            response = self.session.get(
                self.api_configs['copernicus']['base_url'] + 'Products',
                params=query_params,
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                products = data.get('value', [])
                logger.info(f"Found {len(products)} {satellite} products")
                
                # Enhanced product information extraction
                processed_products = []
                for product in products:
                    product_info = {
                        'id': product.get('Id'),
                        'name': product.get('Name'),
                        'size': product.get('ContentLength'),
                        'ingestion_date': product.get('IngestionDate'),
                        'content_date': product.get('ContentDate', {}).get('Start'),
                        'footprint': product.get('Footprint'),
                        'online': product.get('Online', False),
                        'download_url': f"{self.api_configs['copernicus']['download_url']}Products({product.get('Id')})/$value"
                    }
                    
                    # Extract attributes (cloud cover, etc.)
                    attributes = product.get('Attributes', [])
                    for attr in attributes:
                        if attr.get('Name') == 'cloudCover':
                            product_info['cloud_cover'] = attr.get('Value')
                        elif attr.get('Name') == 'platformSerialIdentifier':
                            product_info['platform'] = attr.get('Value')
                    
                    processed_products.append(product_info)
                
                # Save enhanced metadata
                metadata_path = self.output_dir / f"{satellite}_copernicus_products.json"
                with open(metadata_path, 'w') as f:
                    json.dump(processed_products, f, indent=2)
                
                # Create summary CSV
                if processed_products:
                    df = pd.DataFrame(processed_products)
                    csv_path = self.output_dir / f"{satellite}_copernicus_products.csv"
                    df.to_csv(csv_path, index=False)
                    logger.info(f"Product summary saved to {csv_path}")
                
                result = {
                    'source': 'Copernicus Data Space Ecosystem',
                    'satellite': satellite,
                    'products_found': len(processed_products),
                    'date_range': date_range,
                    'cloud_cover_max': cloud_cover,
                    'metadata_path': str(metadata_path),
                    'products': processed_products[:5] if processed_products else []  # First 5 for preview
                }
                
                # Download products if requested
                if download_products and processed_products:
                    result['downloads'] = self._download_copernicus_products(
                        processed_products[:3],  # Limit to first 3 products
                        satellite
                    )
                
                return result
                
            else:
                logger.error(f"Error searching Copernicus {satellite} data: {response.status_code}")
                logger.error(f"Response: {response.text}")
                return {
                    'error': f"HTTP {response.status_code}",
                    'satellite': satellite,
                    'message': 'Failed to search Copernicus catalogue'
                }
                
        except Exception as e:
            logger.error(f"Error extracting Copernicus Sentinel data: {str(e)}")
            return {
                'error': str(e),
                'satellite': satellite,
                'message': 'Exception during Copernicus data extraction'
            }
    
    def _download_copernicus_products(self, products: List[Dict], satellite: str) -> List[Dict]:
        """
        Download Copernicus products (requires authentication)
        
        Args:
            products: List of product dictionaries
            satellite: Satellite identifier
            
        Returns:
            List of download results
        """
        downloads = []
        
        for product in products:
            try:
                product_name = product.get('name', 'unknown')
                download_url = product.get('download_url')
                
                if not download_url:
                    continue
                
                logger.info(f"Attempting to download: {product_name}")
                
                # Note: This requires proper authentication with Copernicus
                # Users need to implement OAuth2 or basic auth here
                response = self.session.get(
                    download_url,
                    stream=True,
                    timeout=300
                )
                
                if response.status_code == 200:
                    # Save product
                    output_path = self.output_dir / f"{satellite}_{product_name}.zip"
                    
                    with open(output_path, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)
                    
                    downloads.append({
                        'product_name': product_name,
                        'status': 'success',
                        'path': str(output_path),
                        'size_mb': output_path.stat().st_size / (1024*1024)
                    })
                    
                    logger.info(f"Downloaded: {output_path}")
                    
                elif response.status_code == 401:
                    downloads.append({
                        'product_name': product_name,
                        'status': 'authentication_required',
                        'message': 'Copernicus Data Space requires authentication'
                    })
                    logger.warning("Authentication required for Copernicus downloads")
                    break  # No point trying other products
                    
                else:
                    downloads.append({
                        'product_name': product_name,
                        'status': 'failed',
                        'error_code': response.status_code
                    })
                    
            except Exception as e:
                downloads.append({
                    'product_name': product.get('name', 'unknown'),
                    'status': 'error',
                    'error': str(e)
                })
                logger.error(f"Error downloading {product.get('name')}: {str(e)}")
        
        return downloads
    
    def extract_landsat_data(self, collection: str = "landsat_ot_c2_l2", 
                           date_range: Tuple[str, str] = None) -> Dict:
        """
        Extract Landsat data using USGS M2M API
        Note: Requires USGS account credentials
        """
        try:
            # This is a placeholder for USGS M2M API implementation
            # Full implementation requires authentication and proper API calls
            
            logger.info("Landsat extraction requires USGS M2M API authentication")
            logger.info("Please implement authentication and search logic")
            
            # Return placeholder structure
            return {
                'collection': collection,
                'date_range': date_range,
                'status': 'requires_authentication',
                'note': 'Implement USGS M2M API authentication'
            }
            
        except Exception as e:
            logger.error(f"Error extracting Landsat data: {str(e)}")
            return {}
    
    def extract_dem_data(self, dataset: str = "SRTM30") -> Dict:
        """
        Extract DEM data from OpenTopography
        
        Args:
            dataset: DEM dataset (SRTMGL30, SRTMGL1, ALOS, etc.)
        """
        try:
            # OpenTopography API parameters
            params = {
                'demtype': dataset,
                'south': self.aoi_bounds[1],
                'north': self.aoi_bounds[3],
                'west': self.aoi_bounds[0],
                'east': self.aoi_bounds[2],
                'outputFormat': 'GTiff'
            }
            
            response = self.session.get(
                self.api_configs['opentopo']['base_url'],
                params=params
            )
            
            if response.status_code == 200:
                # Save DEM file
                dem_path = self.output_dir / f"dem_{dataset}.tif"
                with open(dem_path, 'wb') as f:
                    f.write(response.content)
                
                logger.info(f"DEM data saved to {dem_path}")
                
                # Process and resample DEM
                processed_dem = self._process_raster(str(dem_path), f"dem_{dataset}_processed.tif")
                
                return {
                    'dataset': dataset,
                    'original_path': str(dem_path),
                    'processed_path': processed_dem,
                    'bounds': self.aoi_bounds.tolist()
                }
            else:
                logger.error(f"Error downloading DEM: {response.status_code}")
                return {}
                
        except Exception as e:
            logger.error(f"Error extracting DEM data: {str(e)}")
            return {}
    
    def extract_bathymetry_data(self) -> Dict:
        """
        Extract bathymetry data from GEBCO or other sources
        """
        try:
            # GEBCO WCS service
            wcs_url = "https://www.gebco.net/data_and_products/gebco_web_services/web_map_service/mapserv"
            
            try:
                wcs = WebCoverageService(wcs_url, version='1.0.0')
                
                # Get available coverages
                coverages = list(wcs.contents.keys())
                logger.info(f"Available bathymetry coverages: {coverages}")
                
                return {
                    'source': 'GEBCO',
                    'service_url': wcs_url,
                    'available_coverages': coverages,
                    'status': 'service_available'
                }
                
            except Exception as wcs_error:
                logger.warning(f"WCS service error: {wcs_error}")
                
                # Fallback to direct data download if available
                return {
                    'source': 'GEBCO',
                    'status': 'manual_download_required',
                    'note': 'Download GEBCO grid manually from https://download.gebco.net/'
                }
                
        except Exception as e:
            logger.error(f"Error extracting bathymetry data: {str(e)}")
            return {}
    
    def extract_sar_data(self, platform: str = "S1") -> Dict:
        """
        Extract SAR data from ASF or Copernicus
        
        Args:
            platform: SAR platform (S1 for Sentinel-1, R1 for RADARSAT-1)
        """
        try:
            if platform == "S1":
                # Use Copernicus for Sentinel-1
                return self.extract_sentinel_data(satellite="S1")
            
            elif platform == "R1":
                # RADARSAT-1 data (historical, may require special access)
                logger.info("RADARSAT-1 data requires access through ASF or CSA")
                return {
                    'platform': platform,
                    'status': 'requires_special_access',
                    'note': 'Contact ASF DAAC or CSA for RADARSAT-1 data'
                }
            
        except Exception as e:
            logger.error(f"Error extracting SAR data: {str(e)}")
            return {}
    
    def _process_raster(self, input_path: str, output_filename: str) -> str:
        """
        Process and resample raster to target specifications
        
        Args:
            input_path: Path to input raster
            output_filename: Output filename
            
        Returns:
            Path to processed raster
        """
        try:
            output_path = self.output_dir / output_filename
            
            with rasterio.open(input_path) as src:
                # Clip to AOI
                aoi_geom = [mapping(geom) for geom in self.aoi_gdf.geometry]
                
                # Mask raster with AOI
                clipped_data, clipped_transform = mask(src, aoi_geom, crop=True)
                
                # Update metadata
                clipped_meta = src.meta.copy()
                clipped_meta.update({
                    "driver": "GTiff",
                    "height": clipped_data.shape[1],
                    "width": clipped_data.shape[2],
                    "transform": clipped_transform,
                    "crs": self.target_crs
                })
                
                # Write clipped raster
                with rasterio.open(output_path, "w", **clipped_meta) as dest:
                    dest.write(clipped_data)
            
            logger.info(f"Processed raster saved to {output_path}")
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Error processing raster: {str(e)}")
            return input_path
    
    def resample_to_common_grid(self, input_rasters: List[str], 
                               output_dir: str = None) -> List[str]:
        """
        Resample all rasters to a common grid
        
        Args:
            input_rasters: List of input raster paths
            output_dir: Output directory for resampled rasters
            
        Returns:
            List of resampled raster paths
        """
        try:
            if output_dir is None:
                output_dir = self.output_dir / "resampled"
            else:
                output_dir = Path(output_dir)
            
            output_dir.mkdir(parents=True, exist_ok=True)
            
            resampled_paths = []
            
            for raster_path in input_rasters:
                try:
                    # Generate output filename
                    input_name = Path(raster_path).stem
                    output_path = output_dir / f"{input_name}_resampled.tif"
                    
                    with rasterio.open(raster_path) as src:
                        # Calculate transform for target resolution
                        transform, width, height = calculate_default_transform(
                            src.crs, CRS.from_string(self.target_crs),
                            src.width, src.height, *src.bounds,
                            resolution=self.target_resolution
                        )
                        
                        # Update metadata
                        kwargs = src.meta.copy()
                        kwargs.update({
                            'crs': self.target_crs,
                            'transform': transform,
                            'width': width,
                            'height': height
                        })
                        
                        with rasterio.open(output_path, 'w', **kwargs) as dst:
                            for i in range(1, src.count + 1):
                                reproject(
                                    source=rasterio.band(src, i),
                                    destination=rasterio.band(dst, i),
                                    src_transform=src.transform,
                                    src_crs=src.crs,
                                    dst_transform=transform,
                                    dst_crs=CRS.from_string(self.target_crs),
                                    resampling=Resampling.bilinear
                                )
                    
                    resampled_paths.append(str(output_path))
                    logger.info(f"Resampled {raster_path} to {output_path}")
                    
                except Exception as e:
                    logger.error(f"Error resampling {raster_path}: {str(e)}")
                    continue
            
            return resampled_paths
            
        except Exception as e:
            logger.error(f"Error in resampling process: {str(e)}")
            return []
    
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
                f.write(f"**Target Resolution:** {self.target_resolution}m\n\n")
                
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
    
    def run_full_extraction(self, data_types: List[str] = None) -> Dict:
        """
        Run full data extraction for specified data types
        
        Args:
            data_types: List of data types to extract. If None, extracts all available.
            
        Returns:
            Dictionary containing extraction results
        """
        if data_types is None:
            data_types = ['sentinel_s2', 'sentinel_s1', 'landsat', 'dem', 'bathymetry']
        
        results = {}
        
        logger.info("Starting full data extraction...")
        
        # Create AOI visualization
        self.visualize_aoi()
        
        # Extract data based on specified types
        if 'sentinel_s2' in data_types:
            logger.info("Extracting Sentinel-2 data...")
            results['sentinel_s2'] = self.extract_sentinel_data(satellite='S2')
        
        if 'sentinel_s1' in data_types:
            logger.info("Extracting Sentinel-1 SAR data...")
            results['sentinel_s1'] = self.extract_sentinel_data(satellite='S1')
        
        if 'sentinel_s5p' in data_types:
            logger.info("Extracting Sentinel-5P data...")
            results['sentinel_s5p'] = self.extract_sentinel_data(satellite='S5P')
        
        if 'landsat' in data_types:
            logger.info("Extracting Landsat data...")
            results['landsat'] = self.extract_landsat_data()
        
        if 'dem' in data_types:
            logger.info("Extracting DEM data...")
            results['dem'] = self.extract_dem_data()
        
        if 'bathymetry' in data_types:
            logger.info("Extracting bathymetry data...")
            results['bathymetry'] = self.extract_bathymetry_data()
        
        if 'sar' in data_types:
            logger.info("Extracting additional SAR data...")
            results['sar'] = self.extract_sar_data()
        
        # Create extraction report
        report_path = self.create_extraction_report(results)
        results['report_path'] = report_path
        
        logger.info("Full extraction completed!")
        return results


def main():
    """
    Main function to run the extraction script
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Geospatial Data Extraction and Resampling')
    parser.add_argument('aoi_shapefile', help='Path to AOI shapefile')
    parser.add_argument('--output-dir', default='extracted_data', help='Output directory')
    parser.add_argument('--target-crs', default='EPSG:4326', help='Target CRS')
    parser.add_argument('--target-resolution', type=float, default=30.0, help='Target resolution in meters')
    parser.add_argument('--data-types', nargs='+', 
                       choices=['sentinel_s1', 'sentinel_s2', 'sentinel_s5p', 'landsat', 'dem', 'bathymetry', 'sar'],
                       help='Data types to extract')
    
    args = parser.parse_args()
    
    # Initialize extractor
    extractor = GeospatialDataExtractor(
        aoi_shapefile=args.aoi_shapefile,
        output_dir=args.output_dir,
        target_crs=args.target_crs,
        target_resolution=args.target_resolution
    )
    
    # Run extraction
    results = extractor.run_full_extraction(data_types=args.data_types)
    
    print(f"\nExtraction completed! Results saved to: {args.output_dir}")
    print(f"Report available at: {results.get('report_path', 'N/A')}")


if __name__ == "__main__":
    main()
