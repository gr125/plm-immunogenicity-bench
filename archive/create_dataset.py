#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 15:25:50 2025

@author: icarri

https://github.com/IEDB/IQ-API-use-cases/blob/master/python/use_case_1h.ipynb

"""

import os
import pandas as pd
import requests
import time
import io

path = '/Users/icarri/Documents/Posdoc_LJI/Projects/Self-similarity/data/'

#%% IEDB API
    
def iq_query(endpoint, query_params, base_uri='https://query-api.iedb.org/'):

    url = os.path.join(base_uri, endpoint)
    df = pd.DataFrame()
    
    # set the offset to 0
    query_params['offset'] = 0
    
    # loop through the pages of results
    # API only allows pulling 10,000 entries at a time
    while(True):
        print('Fetching offset: %i' % query_params['offset'])
        r = requests.get(url, params=query_params, headers={'accept': 'text/csv', 'Prefer': 'count=exact'})
        try:
            df = pd.concat([df, pd.read_csv(io.StringIO(r.content.decode('utf-8')))])
            query_params['offset'] += 10000
            #break###
        except pd.errors.EmptyDataError:
            break
        
        # sleep for 1 second between calls so as not to overload the server
        time.sleep(1)
        
    return df 

#%% UniProt API



        
#%% MHCLE: thymus and human proteome

# # explore columns to filter data
# iedb_url='https://query-api.iedb.org'
# table = pd.json_normalize(requests.get(os.path.join(iedb_url, 'mhc_export')).json())
# table = table.columns


# query: healthy


# download data using search endpoint (to filter by healthy)

search_params={
               # query
               'structure_type':'eq.Linear peptide',
               'source_organism_iri_search':'ov.{NCBITaxon:9606}',
               'host_organism_iri_search':'ov.{NCBITaxon:9606}',
               'qualitative_measure':'in.(Positive,Positive-Low,Positive-Intermediate,Positive-High)',
               'disease_iri_search':'ov.{ONTIE:0003423}',
               #'elution_id':'eq.22791522',
               # page
               'offset': 0,
               'order': 'elution_id',
               # retrieve
               'select':'elution_id,linear_sequence',
               }
mhc_search = iq_query('mhc_search', search_params)

# save output
mhc_search.to_csv(path + 'query/mhc_search.csv', index=False)


# download data using export endpoint (to filter by APC)

search_params={
               # query
               #'epitope__source_organism_iri':'eq.http://purl.obolibrary.org/obo/NCBITaxon_9606',
               #'host__iri':'eq.http://purl.obolibrary.org/obo/NCBITaxon_9606',
               'epitope__species':'eq.Homo sapiens',
               'host__name': 'eq.Homo sapiens (human)',
               'assay__qualitative_measurement':'in.(Positive,Positive-Low,Positive-Intermediate,Positive-High)',
               #'assay_id':'eq.22791522',
               # page
               'offset': 0,
               'order': 'assay_id',
               # retrieve
               #'select':'assay_id,epitope__name, epitope__source_molecule_iri, epitope__species, antigen_presenting_cell__name, mhc_restriction__name,reference__title'
               }
mhc_export = iq_query('mhc_export', search_params)

# save output
mhc_export.to_csv(path + 'query/mhc_export.csv', index=False)

#TODO: merge datasets

#df = mhc_search.merge(mhc_export, left_on='elution_id', right_on='assay_id', how='left')

#TODO: process datasets 
#add columns
#check ids
#check nans


# df.rename(columns={'epitope__name':'sequence', 
#                    'epitope__molecule_parent_iri':'uniprot_id', 
#                    #'epitope__source_molecule_iri':'uniprot_id', 
#                    'epitope__species':'species', 
#                    #'antigen_presenting_cell__name':, 
#                    'mhc_restriction__name':'mhc_restriction',
#                    #'reference__title':
#                        }, inplace=True)
# df.drop(columns=['antigen_presenting_cell__name', 'reference__title'], inplace=True)

# df['type'] = 'Thymus MHCLE'
# df['immune_response'] = 0


#TODO: Get protein sequence
source_proteins = pd.read_csv(path + 'query/source_202508011237.csv', usecols=['iri', 'sequence'])
#df = df.merge(source_proteins, left_on='epitope__source_molecule_iri', right_on='iri', how='left')

#TODO: check substring
# df['substring'] = df.apply(lambda row: row['protein_sequence'][int(row['epitope__starting_position'])-1:int(row['epitope__ending_position'])], axis=1)
# df['mismatches'] = df.apply(lambda row: sum(c1!=c2 for c1,c2 in zip(row['sequence'],row['substring'])), axis=1)

del(search_params, mhc_search, mhc_export, source_proteins)

#%% Antigens

# download data using search endpoint (to filter by disease)

#TODO: review ontology to filter data
# https://ontology.iedb.org/disease-tree/ONTIE:0003543
# there are cancer antigens outside the filtering criteria in the IEDB

search_params={
               # query
               'structure_type':'eq.Linear peptide',
               'host_organism_iri_search':'ov.{NCBITaxon:9606}',
               #'qualitative_measure':'in.(Positive,Positive-Low,Positive-Intermediate,Positive-High)',
               
               #'disease_iri_search': 'ov.{DOID:417}', # autoimmune
               #'disease_iri_search': 'ov.{ONTIE:0003421}', # transplant
               #'disease_iri_search': 'ov.{DOID:14566}', # neoplasm ~2000 more than cancer
               #'disease_iri_search': 'ov.{DOID:1205}', # allergic
               #'disease_iri_search': 'ov.{DOID:0050117}', # infectious
             
               #'tcell_id':'eq.1773384',
               
               # page
               'offset': 0,
               'order': 'tcell_id',
               # retrieve
               #'select':'tcell_id,linear_sequence',
               }
tcell_search = iq_query('tcell_search', search_params)

# save output
#tcell_search.to_csv(path + 'query/tcell_search.csv', index=False)

#del(search_params, tcell_search)

#%% CEDAR Neoantigens



#%% Uniprot: Human REF



#%% Microbiome 



#%% 


# columns needed

# sequence
# uniprot_id
# protein_sequence
# mhc_restriction
# type
# species
# immune_response

























