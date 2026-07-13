#!/usr/bin/env python3
"""
Download CTD data files from Bio2RDF Release 3
Files will be used to extract 1M triple knowledge graph
"""

import os
import requests
import gzip
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CTDDataDownloader:
    def __init__(self, base_url="https://download.bio2rdf.org/files/release/3/ctd", data_dir="data/ctd_source"):
        self.base_url = base_url
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Files to download for 1M triple extraction
        # Prioritizing files with chemical-disease-gene interactions
        self.files_to_download = [
            ("ctd_chemicals_diseases.nq.gz", 183935177),  # ~183 MB - Chemical-Disease associations
            ("ctd_chem_gene_ixns.nq.gz", 98690781),        # ~98 MB - Chemical-Gene interactions
            ("ctd_genes_diseases.nq.gz", 2727935786),     # ~2.7 GB - Gene-Disease associations (large!)
            ("ctd_chemicals.nq.gz", 15521129),             # ~15 MB - Chemical entities
            ("ctd_diseases.nq.gz", 1583168),               # ~1.5 MB - Disease entities
            ("ctd_genes.nq.gz", 136443491),                # ~136 MB - Gene entities
        ]

    def download_file(self, filename, expected_size):
        """Download a single file with progress reporting"""
        url = f"{self.base_url}/{filename}"
        output_path = self.data_dir / filename

        # Skip if already downloaded
        if output_path.exists():
            actual_size = output_path.stat().st_size
            if actual_size == expected_size:
                logger.info(f"✓ Already downloaded: {filename} ({actual_size:,} bytes)")
                return str(output_path)
            else:
                logger.warning(f"File exists but size mismatch: {filename}. Re-downloading...")

        logger.info(f"Downloading: {filename} ({expected_size:,} bytes)")

        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()

            downloaded = 0
            chunk_size = 8192

            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)

                        # Progress update every 10MB
                        if downloaded % (10 * 1024 * 1024) == 0:
                            progress = (downloaded / expected_size) * 100 if expected_size > 0 else 0
                            logger.info(f"  Progress: {downloaded:,} / {expected_size:,} bytes ({progress:.1f}%)")

            final_size = output_path.stat().st_size
            logger.info(f"✓ Downloaded: {filename} ({final_size:,} bytes)")
            return str(output_path)

        except Exception as e:
            logger.error(f"✗ Failed to download {filename}: {e}")
            if output_path.exists():
                output_path.unlink()
            return None

    def download_essential_files(self, skip_large_genes_diseases=True):
        """
        Download essential CTD files for 1M triple extraction

        Args:
            skip_large_genes_diseases: Skip the 2.7GB genes_diseases file initially
        """
        logger.info("Starting CTD data download...")
        logger.info(f"Download directory: {self.data_dir.absolute()}")

        downloaded_files = []

        for filename, size in self.files_to_download:
            # Skip the huge genes-diseases file unless explicitly requested
            if skip_large_genes_diseases and filename == "ctd_genes_diseases.nq.gz":
                logger.info(f"Skipping large file: {filename} ({size:,} bytes)")
                logger.info("  You can download it later if needed for more triples")
                continue

            result = self.download_file(filename, size)
            if result:
                downloaded_files.append(result)

        logger.info(f"\n✓ Downloaded {len(downloaded_files)} files")
        return downloaded_files

    def get_file_stats(self):
        """Get statistics about downloaded files"""
        stats = {}
        for filename, _ in self.files_to_download:
            file_path = self.data_dir / filename
            if file_path.exists():
                stats[filename] = {
                    'size': file_path.stat().st_size,
                    'exists': True
                }
            else:
                stats[filename] = {'exists': False}

        return stats

if __name__ == "__main__":
    downloader = CTDDataDownloader()

    print("\n" + "="*70)
    print("CTD Data Downloader")
    print("="*70)

    # Check existing files
    stats = downloader.get_file_stats()
    print("\nChecking existing files...")
    for filename, info in stats.items():
        if info['exists']:
            print(f"  ✓ {filename}: {info['size']:,} bytes")
        else:
            print(f"  ✗ {filename}: Not downloaded")

    # Download files
    print("\n" + "="*70)
    response = input("\nDownload missing files? (y/n): ").strip().lower()

    if response == 'y':
        downloaded = downloader.download_essential_files(skip_large_genes_diseases=True)

        print("\n" + "="*70)
        print("Download Summary")
        print("="*70)
        print(f"Files downloaded: {len(downloaded)}")

        total_size = sum(Path(f).stat().st_size for f in downloaded if Path(f).exists())
        print(f"Total size: {total_size:,} bytes ({total_size / (1024**2):.1f} MB)")

        print("\nNext steps:")
        print("1. Run scripts/extract_1m_triples.py to extract 1M triple graph")
        print("2. Verify the extracted graph with scripts/verify_initial_graph.py")
    else:
        print("Download cancelled.")
