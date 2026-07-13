#!/usr/bin/env python3
"""
Phase 3A: Extract & Normalize CTD Entities

Extract all unique entities from the 1M-triple CTD knowledge graph
and normalize them for PubTator API queries.

This is the corrected approach that will query PubTator API for each entity.
"""

from rdflib import Graph, Namespace, RDF, RDFS
import json
import logging
import sys
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/extract_ctd_entities.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class CTDEntityExtractor:
    """Extract and normalize CTD entities for PubTator API queries"""

    def __init__(self, kg_file):
        """
        Load CTD knowledge graph

        Args:
            kg_file: Path to CTD RDF graph file (Turtle format)
        """
        logger.info(f"Loading CTD knowledge graph from {kg_file}")
        self.kg = Graph()
        self.kg.parse(kg_file, format='turtle')
        logger.info(f"Loaded {len(self.kg):,} triples")

        # Define namespaces
        self.MESH = Namespace("http://bio2rdf.org/mesh:")
        self.NCBIGENE = Namespace("http://bio2rdf.org/ncbigene:")
        self.CTD = Namespace("http://bio2rdf.org/ctd:")
        self.CTD_VOCAB = Namespace("http://bio2rdf.org/ctd_vocabulary:")
        self.RDF = RDF
        self.RDFS = RDFS

    def extract_all_entities(self):
        """
        Extract all Chemical, Disease, and Gene entities from KG

        Returns:
            Dictionary with entity types as keys, {normalized_id: uri} mappings as values
        """
        entities = {
            'chemicals': {},      # ID → URI mapping
            'diseases': {},
            'genes': {}
        }

        # Query 1: Extract chemicals
        query_chemicals = """
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        PREFIX ctd: <http://bio2rdf.org/ctd_vocabulary:>

        SELECT DISTINCT ?entity
        WHERE {
          ?entity rdf:type ctd:Chemical .
        }
        """

        logger.info("Extracting Chemical entities...")
        for row in self.kg.query(query_chemicals):
            uri = str(row.entity)
            # Extract ID from URI: http://bio2rdf.org/mesh:D001241 → D001241
            if 'mesh:' in uri:
                chem_id = uri.split('mesh:')[-1]
            elif 'ctd:' in uri:
                chem_id = uri.split('ctd:')[-1]
            else:
                continue
            entities['chemicals'][chem_id] = uri

        logger.info(f"  Found {len(entities['chemicals']):,} unique chemicals")

        # Query 2: Extract diseases
        query_diseases = """
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        PREFIX ctd: <http://bio2rdf.org/ctd_vocabulary:>

        SELECT DISTINCT ?entity
        WHERE {
          ?entity rdf:type ctd:Disease .
        }
        """

        logger.info("Extracting Disease entities...")
        for row in self.kg.query(query_diseases):
            uri = str(row.entity)
            # Extract ID: http://bio2rdf.org/mesh:D009203 → D009203
            if 'mesh:' in uri:
                disease_id = uri.split('mesh:')[-1]
            elif 'ctd:' in uri:
                disease_id = uri.split('ctd:')[-1]
            else:
                continue
            entities['diseases'][disease_id] = uri

        logger.info(f"  Found {len(entities['diseases']):,} unique diseases")

        # Query 3: Extract genes
        query_genes = """
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        PREFIX ctd: <http://bio2rdf.org/ctd_vocabulary:>

        SELECT DISTINCT ?entity
        WHERE {
          ?entity rdf:type ctd:Gene .
        }
        """

        logger.info("Extracting Gene entities...")
        for row in self.kg.query(query_genes):
            uri = str(row.entity)
            # Extract ID: http://bio2rdf.org/ncbigene:1950 → 1950
            if 'ncbigene:' in uri:
                gene_id = uri.split('ncbigene:')[-1]
            else:
                continue
            entities['genes'][gene_id] = uri

        logger.info(f"  Found {len(entities['genes']):,} unique genes")

        total = sum(len(v) for v in entities.values())
        logger.info(f"\nTotal unique entities: {total:,}")

        return entities

    def save_entities_inventory(self, entities, output_file):
        """
        Save entity inventory for Phase 3B API queries

        Args:
            entities: Dictionary from extract_all_entities()
            output_file: Path to save JSON inventory

        Returns:
            Inventory dictionary
        """
        inventory = {
            'summary': {
                'total_chemicals': len(entities['chemicals']),
                'total_diseases': len(entities['diseases']),
                'total_genes': len(entities['genes']),
                'total_entities': sum(len(v) for v in entities.values())
            },
            'chemicals': entities['chemicals'],
            'diseases': entities['diseases'],
            'genes': entities['genes']
        }

        # Create output directory if needed
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w') as f:
            json.dump(inventory, f, indent=2)

        logger.info(f"\nSaved entity inventory to {output_file}")
        logger.info(f"Summary statistics:")
        for key, value in inventory['summary'].items():
            logger.info(f"  {key}: {value:,}")

        return inventory


def main():
    """Main execution for Phase 3A"""

    logger.info("=" * 80)
    logger.info("PHASE 3A: Extract & Normalize CTD Entities")
    logger.info("=" * 80)

    # Check if CTD graph exists
    kg_file = "data/ctd_1m_triples.ttl"
    if not Path(kg_file).exists():
        logger.error(f"CTD graph file not found: {kg_file}")
        logger.error("Please run Phase 1-2 first!")
        return 1

    # Extract entities
    extractor = CTDEntityExtractor(kg_file)
    entities = extractor.extract_all_entities()

    # Save inventory
    inventory = extractor.save_entities_inventory(entities, "data/ctd_entities_inventory.json")

    logger.info("\n" + "=" * 80)
    logger.info("PHASE 3A COMPLETE")
    logger.info("=" * 80)
    logger.info("\nNext Steps:")
    logger.info("  Phase 3B: Query PubTator API for each entity")
    logger.info("  Run: python scripts/pubtator_api_queries.py")

    return 0


if __name__ == "__main__":
    sys.exit(main())
