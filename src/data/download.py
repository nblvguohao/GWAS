"""
Data download utilities for PlantHGNN
Downloads datasets from CropGS-Hub and biological networks
"""

import os
import requests
import argparse
from pathlib import Path
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataDownloader:
    """Download datasets and biological networks"""
    
    def __init__(self, data_dir="data/raw"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.cropgs_dir = self.data_dir / "cropgs"
        self.networks_dir = self.data_dir / "networks"
        self.annotations_dir = self.data_dir / "annotations"
        
        for d in [self.cropgs_dir, self.networks_dir, self.annotations_dir]:
            d.mkdir(parents=True, exist_ok=True)
    
    def download_file(self, url, output_path, desc=None):
        """Download a file with progress bar"""
        if os.path.exists(output_path):
            logger.info(f"File already exists: {output_path}")
            return
        
        logger.info(f"Downloading {url} to {output_path}")
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        
        with open(output_path, 'wb') as f, tqdm(
            desc=desc or os.path.basename(output_path),
            total=total_size,
            unit='B',
            unit_scale=True
        ) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                pbar.update(len(chunk))
    
    def download_cropgs_dataset(self, dataset_name):
        """
        Download dataset from CropGS-Hub
        Available: rice469, maize282, soybean999, wheat599
        """
        logger.info(f"Downloading CropGS dataset: {dataset_name}")
        
        # CropGS-Hub URLs (placeholder - update with actual URLs)
        urls = {
            'rice469': {
                'genotype': 'https://iagr.genomics.cn/CropGS/data/rice469_genotype.csv',
                'phenotype': 'https://iagr.genomics.cn/CropGS/data/rice469_phenotype.csv',
            },
            'maize282': {
                'genotype': 'https://iagr.genomics.cn/CropGS/data/maize282_genotype.csv',
                'phenotype': 'https://iagr.genomics.cn/CropGS/data/maize282_phenotype.csv',
            },
            'soybean999': {
                'genotype': 'https://iagr.genomics.cn/CropGS/data/soybean999_genotype.csv',
                'phenotype': 'https://iagr.genomics.cn/CropGS/data/soybean999_phenotype.csv',
            },
            'wheat599': {
                'genotype': 'https://iagr.genomics.cn/CropGS/data/wheat599_genotype.csv',
                'phenotype': 'https://iagr.genomics.cn/CropGS/data/wheat599_phenotype.csv',
            }
        }
        
        if dataset_name not in urls:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        
        dataset_dir = self.cropgs_dir / dataset_name
        dataset_dir.mkdir(exist_ok=True)
        
        for file_type, url in urls[dataset_name].items():
            output_path = dataset_dir / f"{dataset_name}_{file_type}.csv"
            try:
                self.download_file(url, output_path, desc=f"{dataset_name} {file_type}")
            except Exception as e:
                logger.warning(f"Failed to download {file_type}: {e}")
                logger.info(f"Please manually download from CropGS-Hub: https://iagr.genomics.cn/CropGS")
    
    def download_string_network(self, species='oryza_sativa'):
        """
        Download STRING protein-protein interaction network
        Species: oryza_sativa (rice), zea_mays (maize), glycine_max (soybean)
        """
        logger.info(f"Downloading STRING network for {species}")
        
        species_ids = {
            'oryza_sativa': '4530',
            'zea_mays': '4577',
            'glycine_max': '3847',
            'triticum_aestivum': '4565'
        }
        
        if species not in species_ids:
            raise ValueError(f"Unknown species: {species}")
        
        species_id = species_ids[species]
        url = f"https://stringdb-downloads.org/download/protein.links.v12.0/{species_id}.protein.links.v12.0.txt.gz"
        
        output_path = self.networks_dir / f"{species}_string_v12.txt.gz"
        try:
            self.download_file(url, output_path, desc=f"STRING {species}")
        except Exception as e:
            logger.warning(f"Failed to download STRING network: {e}")
            logger.info(f"Please manually download from: https://string-db.org/")
    
    def download_go_annotations(self, species='rice'):
        """Download GO annotations for plant species"""
        logger.info(f"Downloading GO annotations for {species}")
        
        go_urls = {
            'rice': 'http://current.geneontology.org/annotations/osa.gaf.gz',
            'maize': 'http://current.geneontology.org/annotations/zma.gaf.gz',
            'soybean': 'http://current.geneontology.org/annotations/gma.gaf.gz',
        }
        
        if species not in go_urls:
            logger.warning(f"GO annotations not available for {species}")
            return
        
        output_path = self.annotations_dir / f"{species}_go.gaf.gz"
        try:
            self.download_file(go_urls[species], output_path, desc=f"GO {species}")
        except Exception as e:
            logger.warning(f"Failed to download GO annotations: {e}")
    
    def download_planttfdb(self):
        """Download PlantTFDB transcription factor data"""
        logger.info("Downloading PlantTFDB data")
        logger.info("Please manually download from: http://planttfdb.gao-lab.org/download.php")
        logger.info("Required files:")
        logger.info("  - TF-target regulatory relationships")
        logger.info("  - TF family classifications")


def main():
    parser = argparse.ArgumentParser(description='Download PlantHGNN datasets')
    parser.add_argument('--dataset', nargs='+', 
                       choices=['rice469', 'maize282', 'soybean999', 'wheat599', 'all'],
                       help='Dataset(s) to download')
    parser.add_argument('--networks', action='store_true',
                       help='Download biological networks')
    parser.add_argument('--data-dir', default='data/raw',
                       help='Output directory')
    
    args = parser.parse_args()
    
    downloader = DataDownloader(args.data_dir)
    
    if args.dataset:
        datasets = ['rice469', 'maize282', 'soybean999', 'wheat599'] if 'all' in args.dataset else args.dataset
        for dataset in datasets:
            downloader.download_cropgs_dataset(dataset)
    
    if args.networks:
        for species in ['oryza_sativa', 'zea_mays', 'glycine_max']:
            downloader.download_string_network(species)
        
        for species in ['rice', 'maize', 'soybean']:
            downloader.download_go_annotations(species)
        
        downloader.download_planttfdb()
    
    logger.info("Download complete!")
    logger.info(f"Data saved to: {downloader.data_dir}")


if __name__ == '__main__':
    main()
