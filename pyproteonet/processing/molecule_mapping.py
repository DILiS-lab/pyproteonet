from io import StringIO
import requests

from pybiomart import Server

class BiomartMapper:

    def __init__(self, from_attribute:str, to_attribute:str='external_gene_name', dataset:str='hsapiens_gene_ensembl',
                 mart:str='ENSEMBL_MART_ENSEMBL', server_url:str='http://www.ensembl.org', skip_missing:bool=True):
        server = Server(host=server_url)
        dataset = (server.marts[mart].datasets[dataset])
        data_array = dataset.query(attributes=[from_attribute, to_attribute]).to_numpy()
        self.mapping = dict()
        for data_ in data_array:
            self.mapping[data_[0]] = data_[1]
        self.skip_missing=skip_missing

    def __call__(self, gene_series):
        gene_series = gene_series.str.split('.',expand=True).loc[:,0].map(self.mapping)
        if self.skip_missing:
            gene_series = gene_series[~gene_series.isna()]
        else:
            gene_series[gene_series.isna()] = 'NaN_Gene'
        return gene_series.astype(str)