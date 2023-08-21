from typing import Optional, List
from pathlib import Path
from string import Template

from IPython.core.display import HTML

from ..data.molecule_graph import MoleculeGraph


el_grapho_path = Path(__file__).parents[0]/ 'static/ElGrapho.min.js'
template_path = Path(__file__).parents[0] / 'static/template3.html'

def plot_graph_notebook(graph: MoleculeGraph, label_nodes: Optional[List] = None):
    nodes = graph.nodes
    nodes['label'] = nodes.index
    nodes = nodes[['label', 'type']].rename(columns={'type':'group'})
    if label_nodes is not None:
        nodes.loc[~(nodes.index.isin(label_nodes)), 'label'] = ''
    else:
        nodes['label'] = ''
    nodes_json = nodes.to_json(orient='records')
    edges_json = graph.edges.rename(columns={'source_node':'from','destination_node':'to'}).to_json(orient='records')
    graph_json = f'{{nodes: {nodes_json}, edges: {edges_json}}}'

    from string import Template
    with open(template_path, 'r', encoding="utf-8") as f:
        template = f.read()
    template = Template(template)
    template = template.substitute({'graph_json': graph_json})
    with open(el_grapho_path, 'r', encoding="utf-8") as f:
        el_grapho_js = f.read()
    html = f"<script>{el_grapho_js}</script>{template}"
    return HTML(html)