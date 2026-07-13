#!/usr/bin/env python3
"""
Extract exactly 1M triples from CTD Bio2RDF data files
Uses proper RDFLib methods for parsing N-Quads format
Based on RDFLib documentation: https://rdflib.readthedocs.io/
"""

import gzip
import logging
from pathlib import Path
from rdflib import Graph, ConjunctiveGraph, Namespace
from collections import defaultdict
import time
import io

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CTDGraphExtractor:
    def __init__(self, source_dir="data/ctd_source", output_dir="data"):
        self.source_dir = Path(source_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Use regular Graph (will merge all triples from different graphs)
        self.final_graph = Graph()
        self.stats = defaultdict(int)

        # Bind namespaces
        self.CTD = Namespace("http://bio2rdf.org/ctd_vocabulary:")
        self.MESH = Namespace("http://bio2rdf.org/mesh:")
        self.NCBIGENE = Namespace("http://bio2rdf.org/ncbigene:")

        self.final_graph.bind('ctd', self.CTD)
        self.final_graph.bind('mesh', self.MESH)
        self.final_graph.bind('ncbigene', self.NCBIGENE)

    def load_triples_from_nquads_streaming(self, filepath, max_triples=None):
        """
        Load triples from N-Quads file using streaming approach

        This method:
        1. Reads N-Quads in chunks
        2. Parses each chunk with RDFLib's nquads parser
        3. Merges triples into final graph (ignoring graph context)
        4. Stops when reaching max_triples limit

        Based on best practices from:
        - https://rdflib.readthedocs.io/en/stable/
        - https://skeptric.com/streaming-nquad-rdf/
        """
        logger.info(f"Loading triples from {filepath.name}...")
        if max_triples:
            logger.info(f"  Target: {max_triples:,} triples")

        loaded = 0
        start_time = time.time()
        chunk_size = 10000  # Process 10K lines at a time

        try:
            with gzip.open(filepath, 'rt', encoding='utf-8') as f:
                chunk_lines = []

                for line in f:
                    line = line.strip()

                    # Skip empty lines and comments
                    if not line or line.startswith('#'):
                        continue

                    chunk_lines.append(line + '\n')

                    # Process chunk when it reaches chunk_size
                    if len(chunk_lines) >= chunk_size:
                        loaded += self._process_nquads_chunk(chunk_lines)
                        chunk_lines = []

                        if loaded % 10000 == 0:
                            elapsed = time.time() - start_time
                            rate = loaded / elapsed if elapsed > 0 else 0
                            logger.info(f"  Loaded {loaded:,} triples ({rate:.0f} triples/sec)")

                        # Check if we've reached target
                        if max_triples and loaded >= max_triples:
                            logger.info(f"  ✓ Reached target of {max_triples:,} triples")
                            break

                # Process remaining lines
                if chunk_lines and (not max_triples or loaded < max_triples):
                    loaded += self._process_nquads_chunk(chunk_lines)

            elapsed = time.time() - start_time
            logger.info(f"  ✓ Loaded {loaded:,} triples in {elapsed:.1f} seconds")
            return loaded

        except Exception as e:
            logger.error(f"Error loading triples from {filepath}: {e}")
            import traceback
            traceback.print_exc()
            return loaded

    def _process_nquads_chunk(self, lines):
        """
        Process a chunk of N-Quad lines using RDFLib's nquads parser
        Returns number of triples added to final_graph
        """
        # Create a temporary ConjunctiveGraph for parsing
        temp_graph = ConjunctiveGraph()

        try:
            # Join lines into a single string and parse
            nquads_data = ''.join(lines)
            temp_graph.parse(data=nquads_data, format='nquads')

            # Merge all triples from temp_graph into final_graph
            # (this flattens the named graphs into a single graph)
            initial_len = len(self.final_graph)

            for s, p, o in temp_graph:
                self.final_graph.add((s, p, o))

            return len(self.final_graph) - initial_len

        except Exception as e:
            logger.error(f"Error parsing chunk: {e}")
            return 0

    def extract_1m_triple_graph(self, output_file="ctd_1m_triples.ttl", target_triples=1000000):
        """
        Extract a balanced 1M-triple knowledge graph from CTD data

        Strategy:
        1. Chemical-Disease associations (~300K triples)
        2. Chemical-Gene interactions (~300K triples)
        3. Chemical entities (~200K triples)
        4. Gene entities (~150K triples)
        5. Disease entities (~50K triples)

        Total: ~1,000,000 triples
        """
        logger.info("="*70)
        logger.info("Extracting 1M-Triple CTD Knowledge Graph")
        logger.info("="*70)

        # File loading plan with triple allocations
        loading_plan = [
            ("ctd_chemicals_diseases.nq.gz", 300000, "Chemical-Disease associations"),
            ("ctd_chem_gene_ixns.nq.gz", 300000, "Chemical-Gene interactions"),
            ("ctd_chemicals.nq.gz", 200000, "Chemical entities"),
            ("ctd_genes.nq.gz", 150000, "Gene entities"),
            ("ctd_diseases.nq.gz", 50000, "Disease entities"),
        ]

        total_loaded = 0

        for filename, allocation, description in loading_plan:
            filepath = self.source_dir / filename

            if not filepath.exists():
                logger.warning(f"✗ File not found: {filename}")
                logger.warning(f"  Run download_ctd_data.py first to download CTD data")
                continue

            logger.info(f"\n[{description}]")
            logger.info(f"File: {filename}")
            logger.info(f"Allocation: {allocation:,} triples")

            # Calculate remaining budget
            current_size = len(self.final_graph)
            remaining = target_triples - current_size
            actual_allocation = min(allocation, remaining)

            if actual_allocation <= 0:
                logger.info(f"✓ Target reached! Skipping remaining files.")
                break

            # Load triples using proper N-Quads streaming
            loaded = self.load_triples_from_nquads_streaming(filepath, max_triples=actual_allocation)
            total_loaded += loaded

            current_total = len(self.final_graph)
            logger.info(f"Graph size: {current_total:,} triples")

            # Check if we've reached the target
            if current_total >= target_triples:
                logger.info(f"✓ Reached target of {target_triples:,} triples!")
                break

        # Final statistics
        final_count = len(self.final_graph)
        logger.info("\n" + "="*70)
        logger.info("Extraction Complete")
        logger.info("="*70)
        logger.info(f"Total triples in graph: {final_count:,}")
        logger.info(f"Target was: {target_triples:,}")

        if final_count < target_triples:
            logger.warning(f"⚠ Graph has fewer triples than target ({final_count:,} < {target_triples:,})")
            logger.warning("  Consider downloading additional CTD files")

        # Save the graph
        output_path = self.output_dir / output_file
        logger.info(f"\nSaving graph to {output_path}...")

        # Save in multiple formats
        logger.info("  Saving as Turtle (.ttl)...")
        self.final_graph.serialize(destination=str(output_path), format='turtle')

        logger.info("  Saving as RDF/XML (.rdf)...")
        rdf_path = str(output_path).replace('.ttl', '.rdf')
        self.final_graph.serialize(destination=rdf_path, format='xml')

        logger.info("  Saving as N-Triples (.nt)...")
        nt_path = str(output_path).replace('.ttl', '.nt')
        self.final_graph.serialize(destination=nt_path, format='nt')

        logger.info(f"\n✓ Saved graph in 3 formats:")
        logger.info(f"  - {output_path}")
        logger.info(f"  - {rdf_path}")
        logger.info(f"  - {nt_path}")

        return self.final_graph

if __name__ == "__main__":
    print("\n" + "="*70)
    print("CTD Knowledge Graph Extractor - 1M Triples (Fixed)")
    print("="*70)

    extractor = CTDGraphExtractor()

    # Check if source files exist
    source_dir = Path("data/ctd_source")
    if not source_dir.exists() or not list(source_dir.glob("*.nq.gz")):
        print("\n✗ No CTD source files found!")
        print(f"  Expected directory: {source_dir.absolute()}")
        print("\nPlease run: python scripts/download_ctd_data.py")
        exit(1)

    # List available files
    print(f"\nSource directory: {source_dir.absolute()}")
    print("Available files:")
    for filepath in sorted(source_dir.glob("*.nq.gz")):
        size_mb = filepath.stat().st_size / (1024**2)
        print(f"  ✓ {filepath.name} ({size_mb:.1f} MB)")

    # Extract graph
    print("\n" + "="*70)
    graph = extractor.extract_1m_triple_graph()

    print("\n" + "="*70)
    print(f"✓ SUCCESS: Extracted {len(graph):,} triples")
    print("="*70)
