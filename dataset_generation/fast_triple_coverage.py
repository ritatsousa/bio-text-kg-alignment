#!/usr/bin/env python3
"""
Fast Triple Coverage Analysis - Direct Turtle File Parsing

Bypasses slow RDFLib parsing by directly reading Turtle file line-by-line.
Provides detailed logging at each step.
"""

import json
import sys
import re
from collections import defaultdict
from datetime import datetime

def log(msg):
    """Print with timestamp"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {msg}", flush=True)

log("="*80)
log("FAST TRIPLE COVERAGE ANALYSIS")
log("="*80)

# Step 1: Load entity→PMID mapping
log("\n[STEP 1/3] Loading entity→PMID mapping...")
with open('data/entity_pmid_mapping.json', 'r') as f:
    entity_pmids = json.load(f)
log(f"  Loaded {len(entity_pmids):,} entity mappings")

# Step 2: Build covered entity set
log("\n[STEP 2/3] Building covered entity set...")
covered_entities = set()
entities_with_pmids = 0

for entity_key, pmids in entity_pmids.items():
    if len(pmids) > 0:
        entities_with_pmids += 1
        entity_id = entity_key.split('_', 1)[1]

        # Add all possible URI formats
        covered_entities.add(f"mesh:{entity_id}")
        covered_entities.add(f"ctd:{entity_id}")
        covered_entities.add(f"ncbigene:{entity_id}")
        covered_entities.add(entity_id)

log(f"  Entities with PMIDs: {entities_with_pmids:,}")
log(f"  Covered entity patterns: {len(covered_entities):,}")

# Step 3: Parse Turtle file and count triple coverage
log("\n[STEP 3/3] Parsing CTD knowledge graph...")
log("  Reading: data/ctd_1m_triples.ttl")
log("  Using direct line-by-line scan for entity IDs...")

total_triples = 0
covered_triples = 0
uncovered_triples = 0
predicate_stats = defaultdict(lambda: {'total': 0, 'covered': 0})

log_interval = 100000
start_time = datetime.now()

log("  Starting analysis...")
log("  (Progress updates every 100,000 triples)")

# Parse by looking for predicate patterns in each triple statement
current_subject = None
line_count = 0

with open('data/ctd_1m_triples.ttl', 'r') as f:
    for line in f:
        line_count += 1
        line = line.strip()

        # Skip comments, prefixes, empty lines
        if not line or line.startswith('#') or line.startswith('@prefix'):
            continue

        # New subject (starts with < and contains >)
        if line.startswith('<') and '> a ' in line:
            current_subject = line.split('>')[0] + '>'
            # Check if subject covered
            subject_covered = any(ent_id in current_subject for ent_id in covered_entities)
            continue

        # Predicate-object lines
        # Look for patterns like: ctd:chemical mesh:C000516 ;
        # or: ctd:gene ncbigene:10908 ;
        if current_subject and (' ' in line):
            parts = line.split()
            if len(parts) >= 2:
                predicate = parts[0]
                obj = parts[1].rstrip(' ;.,')

                # Extract predicate name
                pred_name = predicate.split(':')[-1]

                # Count as a triple
                total_triples += 1

                # Check coverage
                subject_covered = any(ent_id in current_subject for ent_id in covered_entities)
                object_covered = any(ent_id in obj for ent_id in covered_entities)
                is_covered = subject_covered or object_covered

                if is_covered:
                    covered_triples += 1
                else:
                    uncovered_triples += 1

                # Track by predicate
                predicate_stats[pred_name]['total'] += 1
                if is_covered:
                    predicate_stats[pred_name]['covered'] += 1

                # Progress logging
                if total_triples % log_interval == 0:
                    elapsed = (datetime.now() - start_time).total_seconds()
                    rate = total_triples / elapsed if elapsed > 0 else 0
                    coverage_pct = (covered_triples / total_triples * 100) if total_triples > 0 else 0
                    remaining = max(1006800 - total_triples, 0)
                    eta_seconds = remaining / rate if rate > 0 else 0

                    log(f"  Progress: {total_triples:,} triples processed ({coverage_pct:.1f}% covered)")
                    log(f"    Lines read: {line_count:,} | Rate: {rate:.0f} triples/sec | ETA: {eta_seconds/60:.1f} min")

log(f"\n  Completed parsing: {total_triples:,} total triples found from {line_count:,} lines")

# Calculate results
coverage_pct = (covered_triples / total_triples * 100) if total_triples > 0 else 0

log("\n" + "="*80)
log("RESULTS: TRIPLE-LEVEL COVERAGE")
log("="*80)

print(f"\nTotal triples:           {total_triples:,}")
print(f"Covered triples:         {covered_triples:,}")
print(f"Uncovered triples:       {uncovered_triples:,}")
print(f"\n{'='*80}")
print(f"COVERAGE PERCENTAGE:     {coverage_pct:.2f}%")
print(f"{'='*80}")

# Visual bar
bar_length = 50
covered_bar = int(bar_length * coverage_pct / 100)
uncovered_bar = bar_length - covered_bar
print(f"\n[{'█' * covered_bar}{'░' * uncovered_bar}] {coverage_pct:.1f}%")

# Coverage by predicate type
print(f"\n{'='*80}")
print("COVERAGE BY RELATION TYPE")
print(f"{'='*80}\n")

sorted_predicates = sorted(
    [(k, v['covered'], v['total'], v['covered']/v['total']*100 if v['total'] > 0 else 0)
     for k, v in predicate_stats.items()],
    key=lambda x: x[2],  # Sort by total count
    reverse=True
)

print(f"{'Predicate':<40} {'Covered':>10} {'Total':>10} {'Coverage':>10}")
print("-"*80)

for pred, covered, total, pct in sorted_predicates[:20]:
    print(f"{pred:<40} {covered:>10,} {total:>10,} {pct:>9.1f}%")

# Save results
results = {
    'total_triples': total_triples,
    'covered_triples': covered_triples,
    'uncovered_triples': uncovered_triples,
    'coverage_percentage': coverage_pct,
    'predicate_coverage': {
        k: {
            'covered': v['covered'],
            'total': v['total'],
            'percentage': v['covered']/v['total']*100 if v['total'] > 0 else 0
        }
        for k, v in predicate_stats.items()
    }
}

with open('data/triple_coverage_analysis.json', 'w') as f:
    json.dump(results, f, indent=2)

log(f"\n{'='*80}")
log(f"✓ Results saved to: data/triple_coverage_analysis.json")
log(f"{'='*80}")

# Summary
print(f"\nSUMMARY:")
print(f"--------")
print(f"Out of {total_triples:,} triples in the CTD knowledge graph,")
print(f"{covered_triples:,} triples ({coverage_pct:.1f}%) have at least one entity")
print(f"with text evidence in PubTator.")
print(f"\nThis means {coverage_pct:.1f}% of the knowledge graph is grounded")
print(f"in PubMed literature according to PubTator annotations.")

if coverage_pct >= 90:
    print(f"\n✅ EXCELLENT coverage - vast majority of KG is literature-grounded")
elif coverage_pct >= 70:
    print(f"\n✅ GOOD coverage - most of KG is literature-grounded")
elif coverage_pct >= 50:
    print(f"\n⚠️  MODERATE coverage - about half of KG is literature-grounded")
else:
    print(f"\n⚠️  LIMITED coverage - less than half of KG is literature-grounded")

log(f"\nAnalysis completed successfully!")
print()
