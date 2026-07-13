#!/usr/bin/env python3
"""
Phase 3B: Query PubTator 3.0 API for CTD Entities

Query PubTator 3.0 API for each CTD entity to find all documents containing that entity.
This guarantees 100% coverage of the initial CTD knowledge graph.

WARNING: This script queries 29,954 entities with rate limiting (0.5s per entity).
Expected runtime: 2-4 hours

API Documentation: https://www.ncbi.nlm.nih.gov/research/pubtator3/
"""

import requests
import json
import logging
import sys
import time
from typing import Dict, List, Set
from pathlib import Path
from collections import defaultdict

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/pubtator_api_queries.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class PubTatorAPIQuerier:
    """Query PubTator 3.0 API for entity-document associations"""

    BASE_URL = "https://www.ncbi.nlm.nih.gov/research/pubtator3-api"
    SEARCH_ENDPOINT = f"{BASE_URL}/search/"

    def __init__(self, rate_limit_delay=0.5, checkpoint_interval=100):
        """
        Initialize PubTator API querier

        Args:
            rate_limit_delay: Delay between requests (seconds) to respect rate limits
            checkpoint_interval: Save progress every N entities
        """
        self.rate_limit_delay = rate_limit_delay
        self.checkpoint_interval = checkpoint_interval
        self.session = requests.Session()
        self.checkpoint_dir = Path("data/checkpoints")
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def query_entity(self, entity_id, entity_type):
        """
        Query PubTator 3.0 API for documents containing a specific entity

        Args:
            entity_id: Normalized entity ID (e.g., 'D001241', '1950')
            entity_type: 'chemical', 'disease', or 'gene'

        Returns:
            List of PMIDs containing the entity
        """

        # Format query according to PubTator API
        # Chemicals/Diseases use MeSH format: @CHEMICAL_MESH:D001241
        # Genes use NCBI format: @GENE_1950
        if entity_type.lower() == 'chemical':
            query_text = f"@CHEMICAL_MESH:{entity_id}"
        elif entity_type.lower() == 'disease':
            query_text = f"@DISEASE_MESH:{entity_id}"
        elif entity_type.lower() == 'gene':
            query_text = f"@GENE_{entity_id}"
        else:
            logger.warning(f"Unknown entity type: {entity_type}")
            return []

        try:
            # Call PubTator search API
            params = {'text': query_text}
            response = self.session.get(self.SEARCH_ENDPOINT, params=params, timeout=30)
            response.raise_for_status()

            data = response.json()

            # Extract PMIDs from response
            pmids = []
            if 'results' in data:
                for result in data['results']:
                    pmid = result.get('pmid') or result.get('PMID')
                    if pmid:
                        pmids.append(str(pmid))

            # Enhanced logging for progress tracking
            if len(pmids) > 0:
                logger.debug(f"Query {query_text}: found {len(pmids)} PMIDs")

            time.sleep(self.rate_limit_delay)  # Respect rate limits

            return pmids

        except requests.RequestException as e:
            logger.error(f"API query failed for {query_text}: {e}")
            return []

    def load_checkpoint(self, checkpoint_file):
        """Load progress from checkpoint file"""
        if checkpoint_file.exists():
            logger.info(f"Loading checkpoint from {checkpoint_file}")
            with open(checkpoint_file, 'r') as f:
                return json.load(f)
        return {}

    def save_checkpoint(self, entity_pmids, checkpoint_file, processed_count):
        """Save progress to checkpoint file"""
        checkpoint_data = {
            'processed_count': processed_count,
            'entity_pmids': {k: list(v) for k, v in entity_pmids.items()}
        }
        with open(checkpoint_file, 'w') as f:
            json.dump(checkpoint_data, f)
        logger.info(f"Checkpoint saved: {processed_count} entities processed")

    def query_all_entities(self, entity_inventory, resume_from_checkpoint=True):
        """
        Query PubTator for all CTD entities

        Args:
            entity_inventory: Dict with 'chemicals', 'diseases', 'genes' keys
            resume_from_checkpoint: Resume from last checkpoint if available

        Returns:
            Dict mapping entity_id → set of PMIDs
        """

        entity_pmids = {}  # {entity_id: {set of PMIDs}}
        total_entities = entity_inventory['summary']['total_entities']

        # Try to resume from checkpoint
        checkpoint_file = self.checkpoint_dir / "phase3b_progress.json"
        processed = 0

        if resume_from_checkpoint and checkpoint_file.exists():
            checkpoint_data = self.load_checkpoint(checkpoint_file)
            entity_pmids = {k: set(v) for k, v in checkpoint_data.get('entity_pmids', {}).items()}
            processed = checkpoint_data.get('processed_count', 0)
            logger.info(f"Resuming from checkpoint: {processed}/{total_entities} entities already processed")

        # Query chemicals
        logger.info(f"\nQuerying chemicals...")
        chem_count = 0
        for chem_id in entity_inventory['chemicals'].keys():
            entity_key = f"CHEM_{chem_id}"

            # Skip if already processed
            if entity_key in entity_pmids:
                chem_count += 1
                continue

            pmids = self.query_entity(chem_id, 'chemical')
            entity_pmids[entity_key] = set(pmids)

            processed += 1
            chem_count += 1

            if processed % self.checkpoint_interval == 0:
                # Calculate statistics
                total_pmids = len(set().union(*[entity_pmids.get(k, set()) for k in entity_pmids]))
                entities_with_pmids = sum(1 for v in entity_pmids.values() if len(v) > 0)
                avg_pmids = sum(len(v) for v in entity_pmids.values()) / len(entity_pmids) if entity_pmids else 0

                logger.info(f"Progress: {processed}/{total_entities} entities ({processed/total_entities*100:.1f}%)")
                logger.info(f"  Chemicals processed: {chem_count}/{len(entity_inventory['chemicals'])}")
                logger.info(f"  Entities with PMIDs: {entities_with_pmids}/{processed}")
                logger.info(f"  Total unique PMIDs found: {total_pmids:,}")
                logger.info(f"  Average PMIDs per entity: {avg_pmids:.1f}")

                self.save_checkpoint(entity_pmids, checkpoint_file, processed)

        logger.info(f"Chemicals complete: {chem_count}/{len(entity_inventory['chemicals'])}")

        # Query diseases
        logger.info(f"\nQuerying diseases...")
        disease_count = 0
        for disease_id in entity_inventory['diseases'].keys():
            entity_key = f"DIS_{disease_id}"

            # Skip if already processed
            if entity_key in entity_pmids:
                disease_count += 1
                continue

            pmids = self.query_entity(disease_id, 'disease')
            entity_pmids[entity_key] = set(pmids)

            processed += 1
            disease_count += 1

            if processed % self.checkpoint_interval == 0:
                # Calculate statistics
                total_pmids = len(set().union(*[entity_pmids.get(k, set()) for k in entity_pmids]))
                entities_with_pmids = sum(1 for v in entity_pmids.values() if len(v) > 0)
                avg_pmids = sum(len(v) for v in entity_pmids.values()) / len(entity_pmids) if entity_pmids else 0

                logger.info(f"Progress: {processed}/{total_entities} entities ({processed/total_entities*100:.1f}%)")
                logger.info(f"  Diseases processed: {disease_count}/{len(entity_inventory['diseases'])}")
                logger.info(f"  Entities with PMIDs: {entities_with_pmids}/{processed}")
                logger.info(f"  Total unique PMIDs found: {total_pmids:,}")
                logger.info(f"  Average PMIDs per entity: {avg_pmids:.1f}")

                self.save_checkpoint(entity_pmids, checkpoint_file, processed)

        logger.info(f"Diseases complete: {disease_count}/{len(entity_inventory['diseases'])}")

        # Query genes
        logger.info(f"\nQuerying genes...")
        gene_count = 0
        for gene_id in entity_inventory['genes'].keys():
            entity_key = f"GENE_{gene_id}"

            # Skip if already processed
            if entity_key in entity_pmids:
                gene_count += 1
                continue

            pmids = self.query_entity(gene_id, 'gene')
            entity_pmids[entity_key] = set(pmids)

            processed += 1
            gene_count += 1

            if processed % self.checkpoint_interval == 0:
                # Calculate statistics
                total_pmids = len(set().union(*[entity_pmids.get(k, set()) for k in entity_pmids]))
                entities_with_pmids = sum(1 for v in entity_pmids.values() if len(v) > 0)
                avg_pmids = sum(len(v) for v in entity_pmids.values()) / len(entity_pmids) if entity_pmids else 0

                logger.info(f"Progress: {processed}/{total_entities} entities ({processed/total_entities*100:.1f}%)")
                logger.info(f"  Genes processed: {gene_count}/{len(entity_inventory['genes'])}")
                logger.info(f"  Entities with PMIDs: {entities_with_pmids}/{processed}")
                logger.info(f"  Total unique PMIDs found: {total_pmids:,}")
                logger.info(f"  Average PMIDs per entity: {avg_pmids:.1f}")

                self.save_checkpoint(entity_pmids, checkpoint_file, processed)

        logger.info(f"Genes complete: {gene_count}/{len(entity_inventory['genes'])}")

        logger.info(f"\nCompleted: {processed}/{total_entities} entities queried")

        # Final checkpoint
        self.save_checkpoint(entity_pmids, checkpoint_file, processed)

        return entity_pmids

    def save_entity_pmid_mapping(self, entity_pmids, output_file):
        """Save entity→PMID mapping for Phase 4"""

        # Convert sets to lists for JSON serialization
        mapping = {k: list(v) for k, v in entity_pmids.items()}

        # Calculate statistics
        total_pmids = set()
        for pmids in entity_pmids.values():
            total_pmids.update(pmids)

        with open(output_file, 'w') as f:
            json.dump(mapping, f, indent=2)

        logger.info(f"\nSaved entity→PMID mapping to {output_file}")
        logger.info(f"Total entities: {len(mapping):,}")
        logger.info(f"Total unique PMIDs: {len(total_pmids):,}")

        # Calculate coverage statistics
        entities_with_pmids = sum(1 for pmids in mapping.values() if len(pmids) > 0)
        entities_without_pmids = len(mapping) - entities_with_pmids

        logger.info(f"\nCoverage Statistics:")
        logger.info(f"  Entities with ≥1 PMID: {entities_with_pmids:,} ({entities_with_pmids/len(mapping)*100:.1f}%)")
        logger.info(f"  Entities with 0 PMIDs: {entities_without_pmids:,}")

        return {
            'total_entities': len(mapping),
            'total_unique_pmids': len(total_pmids),
            'entities_with_pmids': entities_with_pmids,
            'entities_without_pmids': entities_without_pmids
        }


def main():
    """Main execution for Phase 3B"""

    logger.info("=" * 80)
    logger.info("PHASE 3B: Query PubTator 3.0 API for CTD Entities")
    logger.info("=" * 80)

    # Check if entity inventory exists
    inventory_file = "data/ctd_entities_inventory.json"
    if not Path(inventory_file).exists():
        logger.error(f"Entity inventory not found: {inventory_file}")
        logger.error("Please run Phase 3A first: python scripts/extract_ctd_entities.py")
        return 1

    # Load entity inventory
    with open(inventory_file, 'r') as f:
        inventory = json.load(f)

    total_entities = inventory['summary']['total_entities']
    logger.info(f"\nEntities to query:")
    logger.info(f"  Chemicals: {inventory['summary']['total_chemicals']:,}")
    logger.info(f"  Diseases: {inventory['summary']['total_diseases']:,}")
    logger.info(f"  Genes: {inventory['summary']['total_genes']:,}")
    logger.info(f"  TOTAL: {total_entities:,}")

    # Estimate time
    rate_limit = 0.5  # seconds per query
    estimated_time_hours = (total_entities * rate_limit) / 3600
    estimated_time_mins = (total_entities * rate_limit) / 60

    logger.info("\n" + "=" * 80)
    logger.info("EXECUTION DETAILS")
    logger.info("=" * 80)
    logger.info(f"Rate limit: {rate_limit} seconds per entity")
    logger.info(f"Estimated time: {estimated_time_hours:.1f} hours ({estimated_time_mins:.0f} minutes)")
    logger.info(f"Checkpoint interval: Every 100 entities")
    logger.info(f"Checkpoint location: data/checkpoints/phase3b_progress.json")
    logger.info(f"Log file: logs/pubtator_api_queries.log")
    logger.info("\nPROGRESS TRACKING:")
    logger.info("  - Progress updates every 100 entities")
    logger.info("  - Statistics: entities with PMIDs, unique PMIDs, averages")
    logger.info("  - Safe to interrupt (Ctrl+C) - will resume from last checkpoint")
    logger.info("\nMONITORING:")
    logger.info(f"  - Watch log: tail -f logs/pubtator_api_queries.log")
    logger.info(f"  - Check checkpoint: cat data/checkpoints/phase3b_progress.json | python3 -m json.tool | head")
    logger.info("=" * 80 + "\n")

    # Confirm before proceeding
    response = input("Proceed with API queries? (yes/no): ")
    if response.lower() not in ['yes', 'y']:
        logger.info("Aborted by user.")
        return 0

    # Query PubTator
    querier = PubTatorAPIQuerier(rate_limit_delay=0.5, checkpoint_interval=100)
    entity_pmids = querier.query_all_entities(inventory, resume_from_checkpoint=True)

    # Save mapping
    stats = querier.save_entity_pmid_mapping(entity_pmids, "data/entity_pmid_mapping.json")

    logger.info("\n" + "=" * 80)
    logger.info("PHASE 3B COMPLETE")
    logger.info("=" * 80)
    logger.info("\nNext Steps:")
    logger.info("  Phase 4: Iterative document filtering")
    logger.info("  Run: python scripts/iterative_document_filtering.py")

    return 0


if __name__ == "__main__":
    sys.exit(main())
