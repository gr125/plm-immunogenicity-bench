import pandas as pd
import numpy as np
import requests
import xml.etree.ElementTree as ET
from time import sleep
import re
import pickle
from pathlib import Path
from collections import defaultdict

# Define canonical rank hierarchy
RANK_HIERARCHY = {
    'root': 0,
    'acellular root': 1,
    'domain': 10,
    'superkingdom': 10,
    'kingdom': 20,
    'subkingdom': 25,
    'phylum': 30,
    'subphylum': 35,
    'superclass': 38,
    'class': 40,
    'subclass': 45,
    'infraclass': 48,
    'cohort': 50,
    'superorder': 55,
    'order': 60,
    'suborder': 65,
    'infraorder': 68,
    'parvorder': 70,
    'superfamily': 75,
    'family': 80,
    'subfamily': 85,
    'tribe': 88,
    'subtribe': 90,
    'genus': 100,
    'subgenus': 105,
    'species group': 110,
    'species subgroup': 115,
    'species': 120,
    'subspecies': 125,
    'varietas': 130,
    'forma': 135,
    'strain': 140,
    'serotype': 145,
    'serogroup': 145,
    'isolate': 150,
    'clade': 999,
    'no rank': 1000
}

def parse_taxon_string(taxon_string):
    """
    Extract NCBI Taxonomy IDs from a string like '{NCBITaxon:1,NCBITaxon:10239,...}'
    
    Parameters:
    -----------
    taxon_string : str
        String containing taxonomy identifiers
    
    Returns:
    --------
    list of int: NCBI Taxonomy IDs
    """
    if pd.isna(taxon_string):
        return []
    
    # Extract all NCBITaxon IDs
    taxon_ids = re.findall(r'NCBITaxon:(\d+)', str(taxon_string))
    return [int(tid) for tid in taxon_ids]

def get_ncbi_taxonomy(taxon_id):
    """
    Query NCBI Taxonomy database for a given taxon ID
    
    Parameters:
    -----------
    taxon_id : int
        NCBI Taxonomy ID
    
    Returns:
    --------
    dict: {'taxon_id': int, 'rank': str, 'name': str} or None if error
    """
    try:
        url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
        params = {
            'db': 'taxonomy',
            'id': taxon_id,
            'retmode': 'xml'
        }
        
        response = requests.get(url, params=params, timeout=10)
        
        if response.status_code == 200:
            root = ET.fromstring(response.content)
            
            # Extract rank and scientific name
            rank = root.find('.//Rank')
            sci_name = root.find('.//ScientificName')
            
            return {
                'taxon_id': taxon_id,
                'rank': rank.text if rank is not None else 'no rank',
                'name': sci_name.text if sci_name is not None else 'Unknown'
            }
        else:
            print(f"Error fetching taxon {taxon_id}: HTTP {response.status_code}")
            return None
            
    except Exception as e:
        print(f"Error fetching taxon {taxon_id}: {e}")
        return None

def load_taxonomy_cache(cache_file='taxonomy_cache.pkl'):
    """
    Load taxonomy cache from file, create empty cache if file doesn't exist
    
    Parameters:
    -----------
    cache_file : str
        Path to cache file
    
    Returns:
    --------
    dict: Cache dictionary (empty if file doesn't exist)
    """
    cache_path = Path(cache_file)
    
    if cache_path.exists():
        try:
            with open(cache_file, 'rb') as f:
                cache = pickle.load(f)
            print(f"Loaded {len(cache)} entries from cache: {cache_file}")
            return cache
        except Exception as e:
            print(f"Error loading cache file: {e}")
            print("Creating new empty cache...")
            return {}
    else:
        print(f"Cache file not found: {cache_file}")
        print("Creating new empty cache...")
        
        # Create directory if it doesn't exist
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        
        return {}

def save_taxonomy_cache(cache, cache_file='taxonomy_cache.pkl'):
    """
    Save taxonomy cache to file
    
    Parameters:
    -----------
    cache : dict
        Cache dictionary to save
    cache_file : str
        Path to cache file
    """
    try:
        cache_path = Path(cache_file)
        
        # Create directory if it doesn't exist
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(cache_file, 'wb') as f:
            pickle.dump(cache, f)
        print(f"Saved {len(cache)} entries to cache: {cache_file}")
    except Exception as e:
        print(f"Error saving cache file: {e}")

def fetch_taxonomies_with_caching(taxon_ids, cache=None, cache_file='taxonomy_cache.pkl'):
    """
    Fetch taxonomy information for a list of taxon IDs with caching
    
    Parameters:
    -----------
    taxon_ids : list of int
        NCBI Taxonomy IDs to fetch
    cache : dict, optional
        Existing cache dictionary
    cache_file : str
        Path to cache file for persistence
    
    Returns:
    --------
    list of dict: Taxonomy information for each ID
    dict: Updated cache
    """
    if cache is None:
        cache = load_taxonomy_cache(cache_file)
    
    results = []
    new_fetches = 0
    
    for tid in taxon_ids:
        if tid in cache:
            results.append(cache[tid])
        else:
            print(f"Fetching taxon {tid} from NCBI...")
            info = get_ncbi_taxonomy(tid)
            if info:
                cache[tid] = info
                results.append(info)
                new_fetches += 1
                sleep(0.35)  # Rate limit: max 3 requests/second
            else:
                # Store failed lookup to avoid retrying
                cache[tid] = None
    
    if new_fetches > 0:
        save_taxonomy_cache(cache, cache_file)
        print(f"Fetched {new_fetches} new taxonomies")
    
    # Filter out None values
    return [r for r in results if r is not None], cache

def get_rank_level(rank):
    """Get numerical level for a rank"""
    rank_lower = rank.lower().strip()
    return RANK_HIERARCHY.get(rank_lower, 1000)

def organize_taxonomy_by_hierarchy(taxon_list):
    """
    Organize taxonomy entries by their hierarchical rank level
    
    Parameters:
    -----------
    taxon_list : list of dict
        Each dict has 'taxon_id', 'rank', and 'name'
    
    Returns:
    --------
    dict with standardized rank columns
    """
    result = {
        'domain': None,
        'kingdom': None,
        'phylum': None,
        'class': None,
        'order': None,
        'family': None,
        'genus': None,
        'species': None,
        'strain': None,
        'species_group': None,
        'subspecies': None,
        'most_specific': None
    }
    
    if not taxon_list:
        return result
    
    # Sort by rank hierarchy
    sorted_taxa = sorted(taxon_list, key=lambda x: get_rank_level(x['rank']))
    
    for taxon in sorted_taxa:
        rank = taxon['rank'].lower().strip()
        name = taxon['name']
        
        # Skip root
        if rank == 'root' or rank == 'no rank' and name == 'root':
            continue
        
        # Map to standardized columns
        if rank in ['domain', 'superkingdom', 'acellular root']:
            result['domain'] = name
        elif rank == 'kingdom':
            result['kingdom'] = name
        elif rank == 'phylum':
            result['phylum'] = name
        elif rank == 'class':
            result['class'] = name
        elif rank == 'order':
            result['order'] = name
        elif rank in ['family', 'superfamily']:
            if result['family'] is None:  # Don't overwrite family with superfamily
                result['family'] = name
        elif rank == 'genus':
            result['genus'] = name
        elif rank == 'species':
            result['species'] = name
        elif rank == 'subspecies':
            result['subspecies'] = name
        elif rank == 'species group':
            result['species_group'] = name
        elif rank == 'strain':
            result['strain'] = name
        elif rank == 'no rank':
            # For 'no rank', try to infer from position in hierarchy
            if result['species'] is None:
                # Might be a species-level designation
                result['species'] = name
    
    # Set most specific (excluding root)
    non_root_taxa = [t for t in sorted_taxa if not (t['rank'].lower() == 'root' or 
                                                     (t['rank'].lower() == 'no rank' and t['name'] == 'root'))]
    if non_root_taxa:
        result['most_specific'] = non_root_taxa[-1]['name']
    
    return result

def process_taxonomy_column(df, taxon_column='source_organism_taxon', 
                           cache_file='taxonomy_cache.pkl',
                           batch_size=100):
    """
    Complete pipeline: parse taxonomy strings, fetch data, and standardize columns
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with taxonomy column
    taxon_column : str
        Name of column containing taxonomy strings like '{NCBITaxon:1,NCBITaxon:10239,...}'
    cache_file : str
        Path to cache file for persistent storage (will be created if doesn't exist)
    batch_size : int
        Number of rows to process before saving cache
    
    Returns:
    --------
    pandas.DataFrame with added standardized taxonomy columns
    """
    
    print(f"Processing {len(df)} rows...")
    print(f"Using cache file: {cache_file}")
    
    # Load cache (creates empty cache if file doesn't exist)
    cache = load_taxonomy_cache(cache_file)
    
    # Initialize standardized columns
    std_columns = ['tax_domain', 'tax_kingdom', 'tax_phylum', 'tax_class', 
                   'tax_order', 'tax_family', 'tax_genus', 'tax_species',
                   'tax_strain', 'tax_species_group', 'tax_subspecies', 
                   'tax_most_specific']
    
    for col in std_columns:
        df[col] = None
    
    # Collect all unique taxon IDs first
    print("Parsing taxonomy strings...")
    all_taxon_ids = set()
    row_taxon_ids = []
    
    for idx, row in df.iterrows():
        taxon_ids = parse_taxon_string(row[taxon_column])
        row_taxon_ids.append(taxon_ids)
        all_taxon_ids.update(taxon_ids)
    
    print(f"Found {len(all_taxon_ids)} unique taxon IDs")
    
    # Fetch all unique taxonomies
    print("Fetching taxonomy information...")
    all_taxon_list, cache = fetch_taxonomies_with_caching(
        list(all_taxon_ids), 
        cache=cache, 
        cache_file=cache_file
    )
    
    # Build lookup dict
    taxon_lookup = {t['taxon_id']: t for t in all_taxon_list}
    
    # Process each row
    print("Organizing taxonomies...")
    for idx, taxon_ids in enumerate(row_taxon_ids):
        if idx % 100 == 0:
            print(f"Processed {idx}/{len(df)} rows...")
        
        # Get taxonomy info for this row
        taxon_list = [taxon_lookup[tid] for tid in taxon_ids if tid in taxon_lookup]
        
        if taxon_list:
            organized = organize_taxonomy_by_hierarchy(taxon_list)
            
            df.at[idx, 'tax_domain'] = organized['domain']
            df.at[idx, 'tax_kingdom'] = organized['kingdom']
            df.at[idx, 'tax_phylum'] = organized['phylum']
            df.at[idx, 'tax_class'] = organized['class']
            df.at[idx, 'tax_order'] = organized['order']
            df.at[idx, 'tax_family'] = organized['family']
            df.at[idx, 'tax_genus'] = organized['genus']
            df.at[idx, 'tax_species'] = organized['species']
            df.at[idx, 'tax_strain'] = organized['strain']
            df.at[idx, 'tax_species_group'] = organized['species_group']
            df.at[idx, 'tax_subspecies'] = organized['subspecies']
            df.at[idx, 'tax_most_specific'] = organized['most_specific']
    
    print("Done!")
    
    # Print summary statistics
    print("\n=== Summary Statistics ===")
    for col in std_columns:
        non_null = df[col].notna().sum()
        if non_null > 0:
            print(f"{col}: {non_null}/{len(df)} rows ({100*non_null/len(df):.1f}%)")
            print(f"  Unique values: {df[col].nunique()}")
            print(f"  Top 3: {df[col].value_counts().head(3).to_dict()}")
            print()
    
    return df

# ============================================================================
# CLI
# ============================================================================

def main(argv=None):
    """Resolve the taxon column of the annotated table into 12 named ranks.

    Reads data.annotated, writes data.taxonomy -- the input to every downstream
    script. Lookups are cached in cache.taxonomy, so a rerun costs no NCBI
    round-trips for taxa already seen.

        python -m antigen_embedding.data.taxonomy
    """
    import argparse

    from ..config import add_config_args, config_from_args
    from ..io import write_table

    parser = argparse.ArgumentParser(description=main.__doc__)
    add_config_args(parser)
    parser.add_argument("--taxon-column", default="source_organism_iri_search",
                        help="column holding the {NCBITaxon:...} identifier set")
    args = parser.parse_args(argv)
    cfg = config_from_args(args)

    src = cfg.path("data.annotated")
    dst = cfg.path("data.taxonomy")
    cache = cfg.path("cache.taxonomy")

    print(f"reading  {src}")
    df = pd.read_csv(src)
    df = process_taxonomy_column(df, taxon_column=args.taxon_column,
                                 cache_file=str(cache))
    out = write_table(df, dst)
    print(f"wrote    {out}  ({len(df)} rows)")

    print("\nDomain distribution:")
    print(df["tax_domain"].value_counts())
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
