#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 11 13:47:09 2025

@author: icarri
"""

import os
import requests
import pandas as pd
import io
import time
import csv
from io import StringIO
from Bio import Entrez, SeqIO
from numpy.core.defchararray import find
import matplotlib.pyplot as plt

path='/Users/icarri/Documents/Posdoc_LJI/Projects/PLM_Benchmark/'

def API_query(endpoint, query_params, base_uri='https://cedar-api.iedb.org/'):
    """
    Execute a query against the CEDAR/IEDB API with pagination support.

    Parameters:
    -----------
    endpoint : str
        The API endpoint to query
    query_params : dict
        Dictionary of query parameters
    base_uri : str
        Base URI for the CEDAR API

    Returns:
    --------
    pd.DataFrame
        Combined results from all paginated requests
    """

    url = os.path.join(base_uri, endpoint)
    df = pd.DataFrame()

    # set the offset to 0
    query_params['offset'] = 0

    # loop through the pages of results
    # API only allows pulling 10,000 entries at a time
    while(True):
        print('Fetching offset: %i' % query_params['offset'])
        r = requests.get(
            url,
            params=query_params,
            headers={'accept': 'text/csv', 'Prefer': 'count=exact'}
        )

        try:
            # Parse CSV response and append to existing DataFrame
            df = pd.concat([df, pd.read_csv(io.StringIO(r.content.decode('utf-8')))])
            query_params['offset'] += 10000
        except pd.errors.EmptyDataError:
            # No more data available
            break

        # Rate limiting: pause between requests to not overload the server
        time.sleep(1)

    return df


def fetch_genpept(acc: str):
    """Fetch protein sequence from NCBI (GenPept)."""
    try:
        handle = Entrez.efetch(db="protein", id=acc, rettype="fasta", retmode="text")
        fasta_data = handle.read()
        handle.close()
        if fasta_data.strip():
            record = SeqIO.read(StringIO(fasta_data), "fasta")
            return str(record.seq)
    except Exception as e:
        print(f"Error fetching GenPept {acc}: {e}")
    return None


def fetch_uniprot(acc: str):
    """Fetch protein sequence from UniProt (FASTA)."""
    acc = acc.split(".")[0]  # UniProt ignores version suffix (.1, .2, …)
    url = f"https://rest.uniprot.org/uniprotkb/{acc}.fasta"
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            fasta_data = response.text
            record = SeqIO.read(StringIO(fasta_data), "fasta")
            return str(record.seq)
        else:
            print(f"UniProt fetch failed for {acc}, status {response.status_code}")
    except Exception as e:
        print(f"Error fetching UniProt {acc}: {e}")
    return None


def get_sequence(source: str):
    """Get protein sequence depending on source type."""
    if source.startswith("GENPEPT:"):
        return fetch_genpept(source.split(":")[1])
    elif source.startswith("UNIPROT:"):
        return fetch_uniprot(source.split(":")[1])
    else:
        return None

# Required by NCBI
Entrez.email = "ibelcarri@gmail.com"  
Entrez.api_key = None                   

# Avoid permission issue by setting cache directory
os.makedirs("biopython_cache", exist_ok=True)
Entrez.cache = "biopython_cache"

#%% search

#url='https://cedar-api.iedb.org/'
url='https://query-api.iedb.org/'

endpoint='tcell_search'

search_params={
    # Query filters
    'host_organism_iri': 'eq.NCBITaxon:9606',
    'structure_type':'eq.Linear peptide',
    'reference_date':'gte.2010',
    'or':'(mhc_allele_resolution.eq.1 chain,mhc_allele_resolution.eq.2 chain)',

    # Pagination
    'order': 'tcell_id',
    'offset': 0,

    # Column selection
    #'select':'tcell_id',
    'select':'linear_sequence,e_modification,linear_sequence_length,qualitative_measure,mhc_class,mhc_restriction,parent_source_antigen_iri,curated_source_antigen,source_organism_iri_search,source_organism_name,disease_iri_search,disease_names,assay_names',
}

# Execute epitope search
tcell_df = API_query(endpoint, search_params, url)

#%% export

# tcell_ids = ','.join(map(str,set(tcell_df['tcell_id'].to_list())))

# endpoint='tcell_export'

# search_params = {
#     # Query filter
#     'assay_id': 'in.(' + tcell_ids + ')',

#     # Pagination
#     'order': 'assay_id',
#     'offset': 0,
# }

# # Execute final query
# tcell_assays_df = API_query(endpoint, search_params, url)

del(endpoint, search_params, url)

#%% filter 

# remove modified epitopes
tcell_df = tcell_df[tcell_df['e_modification'].isnull()]
tcell_df.drop(columns=['e_modification'], inplace=True)

# there is a source protein
tcell_df = tcell_df[~tcell_df['curated_source_antigen'].isnull()]

# remove long epitopes
cond_I  = (tcell_df["mhc_class"] == "I")  & (tcell_df["linear_sequence_length"].between(8, 12))
cond_II = (tcell_df["mhc_class"] == "II") & (tcell_df["linear_sequence_length"].between(12, 25))
tcell_df = tcell_df[cond_II | cond_I]

del(cond_I, cond_II)

#%% format

# numeric target values
tcell_df['immune_response'] = tcell_df['qualitative_measure'].apply(lambda x: 0 if x == 'Negative' else 1)
# assay

tcell_df['assay_response_measured'] = tcell_df["assay_names"].str.split('|', expand=True)[0]
tcell_df['assay_method'] = tcell_df["assay_names"].str.split('|', expand=True)[1]

tcell_df.drop(columns=['assay_names'], inplace=True)

# source protein

def parse_antigen(row):
    # strip parentheses at start/end
    clean = row.strip("()")
    # parse as CSV
    parsed = next(csv.reader(StringIO(clean)))
    return parsed

tcell_df["curated_source_antigen"] = tcell_df["curated_source_antigen"].apply(parse_antigen)

# get id, start, end
tcell_df['curated_source_antigen_iri'] = tcell_df["curated_source_antigen"].str[2]
#tcell_df[['prot_db', 'prot_id']] = tcell_df["curated_source_antigen"].str[2].str.split(':', expand=True)
tcell_df['curated_source_antigen_start'] = tcell_df["curated_source_antigen"].str[3]
tcell_df['curated_source_antigen_end']   = tcell_df["curated_source_antigen"].str[4]

# remove nan which might be edge cases
tcell_df = tcell_df.loc[tcell_df['curated_source_antigen_iri'] != '']
tcell_df = tcell_df.loc[tcell_df['curated_source_antigen_start'] != '']

tcell_df.drop(columns=['curated_source_antigen'], inplace=True)

# save dataset
tcell_df.to_csv(path + 'data_ungroup.csv')

#%% group by epitope

tcell_df = tcell_df.groupby(['linear_sequence', 'linear_sequence_length',
                              'mhc_class',  'mhc_restriction', 
                              #'prot_db', 'prot_id',
                              'curated_source_antigen_iri',
                              'curated_source_antigen_start', 'curated_source_antigen_end',
                              ], as_index=False)['immune_response'].max()

#%% get antigens

# get unique prot ids to avoid retrieving the same data
protein_df = pd.DataFrame(columns=['curated_source_antigen_iri','protein_sequence'])
protein_df['curated_source_antigen_iri'] = tcell_df["curated_source_antigen_iri"].unique()

# apply function
protein_df["protein_sequence"] = protein_df["curated_source_antigen_iri"].apply(get_sequence)
protein_df.to_csv(path + 'protein_sequences.csv')

# remove nans
protein_df = protein_df[~protein_df['protein_sequence'].isnull()]

# merge
tcell_df = tcell_df.merge(protein_df, on="curated_source_antigen_iri")

# search sequence 
peps = tcell_df["linear_sequence"].values.astype(str)
prot = tcell_df["protein_sequence"].values.astype(str)

tcell_df = tcell_df.assign(check=find(prot, peps))
tcell_df = tcell_df.loc[tcell_df['check'] > 0]
tcell_df.drop(columns=['check'], inplace=True)

# drop duplicates
tcell_df.drop(columns=['curated_source_antigen_iri'], inplace=True)
tcell_df.drop_duplicates(inplace=True)

# save dataset
tcell_df.to_csv(path + 'data.csv')

#%% plot

variable = 'immune_response'
categories = ['Positives', 'Negatives']

values = [len(tcell_df.loc[tcell_df[variable] == 1]), 
          len(tcell_df.loc[tcell_df[variable] == 0])]
          
plt.bar(categories, values)
plt.ylabel('Counts')
plt.xlabel(variable)
plt.show()

variable = 'mhc_class'
categories = ['I', 'II']

values = [len(tcell_df.loc[tcell_df[variable] == 'I']), 
          len(tcell_df.loc[tcell_df[variable] == 'II'])]
 
plt.bar(categories, values)
plt.ylabel('Counts')
plt.xlabel(variable)
plt.show()

