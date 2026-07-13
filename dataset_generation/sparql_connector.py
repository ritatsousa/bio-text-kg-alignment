from SPARQLWrapper import SPARQLWrapper, JSON, RDF
import rdflib
from rdflib import Graph
import time
import logging

logger = logging.getLogger(__name__)

class CTDSPARQLConnector:
    def __init__(self, endpoint="https://bio2rdf.org/sparql"):
        """
        Initialize SPARQL connector to Bio2RDF endpoint
        Default uses main Bio2RDF endpoint: https://bio2rdf.org/sparql
        Alternative endpoints to try:
        - http://ctd.bio2rdf.org/sparql (CTD-specific, if available)
        - http://cu.ctd.bio2rdf.org/sparql (from documentation)
        """
        self.endpoint = endpoint
        self.sparql = SPARQLWrapper(self.endpoint)
        self.sparql.setTimeout(300)  # 5 minutes timeout
        logger.info(f"Initialized SPARQL connector to: {self.endpoint}")

    def query_json(self, query_string):
        """Execute SPARQL query and return JSON results"""
        self.sparql.setQuery(query_string)
        self.sparql.setReturnFormat(JSON)
        try:
            results = self.sparql.query().convert()
            return results
        except Exception as e:
            logger.error(f"Query error: {e}")
            return None

    def query_rdf(self, query_string):
        """Execute CONSTRUCT query and return RDF graph"""
        self.sparql.setQuery(query_string)
        self.sparql.setReturnFormat(RDF)
        try:
            logger.info("Executing SPARQL CONSTRUCT query...")
            results = self.sparql.query().convert()
            g = Graph()
            g.parse(data=results.serialize(format='xml'), format='xml')
            logger.info(f"Query returned {len(g)} triples")
            return g
        except Exception as e:
            logger.error(f"RDF Query error: {e}")
            return None

    def count_triples(self, query_pattern):
        """Count triples matching a SPARQL pattern"""
        count_query = f"""
        SELECT (COUNT(*) as ?count) WHERE {{
          {query_pattern}
        }}
        """
        results = self.query_json(count_query)
        if results and results['results']['bindings']:
            return int(results['results']['bindings'][0]['count']['value'])
        return 0

    def test_connection(self):
        """Test the SPARQL endpoint connection"""
        test_query = """
        SELECT (COUNT(*) as ?count) WHERE {
          ?s ?p ?o .
        } LIMIT 1
        """
        try:
            logger.info("Testing SPARQL endpoint connection...")
            results = self.query_json(test_query)
            if results:
                logger.info("✓ SPARQL endpoint connection successful")
                return True
            else:
                logger.error("✗ SPARQL endpoint connection failed")
                return False
        except Exception as e:
            logger.error(f"✗ Connection test failed: {e}")
            return False

if __name__ == "__main__":
    # Test the connector
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    connector = CTDSPARQLConnector()

    # Test connection
    if connector.test_connection():
        print("\n✓ SPARQL Connector is working correctly!")

        # Try a simple query to get some CTD chemicals
        print("\nTesting a simple query to get CTD chemicals...")
        query = """
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        PREFIX ctd: <http://bio2rdf.org/ctd_vocabulary:>

        SELECT ?chemical ?label
        WHERE {
          ?chemical rdf:type ctd:Chemical .
          ?chemical rdfs:label ?label .
        }
        LIMIT 5
        """

        results = connector.query_json(query)
        if results:
            print("\nSample CTD chemicals:")
            for binding in results['results']['bindings']:
                chemical = binding['chemical']['value']
                label = binding['label']['value']
                print(f"  {label} ({chemical})")
        else:
            print("\n✗ Query failed or returned no results")
    else:
        print("\n✗ SPARQL Connector test failed")
