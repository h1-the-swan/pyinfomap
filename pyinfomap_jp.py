import sys, os, time
from datetime import datetime
from timeit import default_timer as timer
import itertools

import networkx as nx
from pyinfomap import Clustering

TAU = 0.15
PAGE_RANK = 'page_rank'

import logging
logging.basicConfig(format='%(asctime)s %(name)s.%(lineno)d %(levelname)s : %(message)s',
        datefmt="%H:%M:%S",
        level=logging.INFO)
# logger = logging.getLogger(__name__)
logger = logging.getLogger('__main__').getChild(__name__)

class PyInfomap(object):

    """Docstring for PyInfomap. """

    def __init__(self, filename=None, graph=None, clustering=None):
        self.filename = filename
        self.graph = graph
        self.clustering = clustering
        self.current_mdl = 999
        self.num_mdl_calculations = 0
        if self.filename and not self.graph:
            self.load_and_process_graph(self.filename)

    def process_graph(self, graph):
        """normalize edge weights, compute pagerank, store as node data

        :graph: networkx graph
        :returns: modified networkx graph

        """
        for node in graph:
            edges = graph.edges(node, data=True)
            total_weight = sum([data['weight'] for (_, _, data) in edges])
            for (_, _, data) in edges:
                data['weight'] = data['weight'] / total_weight
        # Get its PageRank, alpha is 1-tau where [RAB2009 says \tau=0.15]
        page_ranks = nx.pagerank(graph, alpha=1-TAU)
        for (node, page_rank) in page_ranks.items():
            graph.node[node][PAGE_RANK] = page_rank
        return graph

    def load_and_process_graph(self, filename):
        """Load the graph, normalize edge weights, compute pagerank, and store all
        this back in node data.

        :filename: pajek filename (.net)
        :returns: networkx graph object

        """
        graph = nx.DiGraph(nx.read_pajek(filename))
        logger.info("Loaded a graph ({} nodes, {} edges)".format(graph.number_of_nodes(), graph.number_of_edges()))
        graph = self.process_graph(graph)
        self.graph = graph

        if self.graph and not self.clustering:
            self.initial_clustering()
        return graph

    def initial_clustering(self):
        """Default initial clustering puts each node in its own cluster
        :returns: TODO

        """
        modules = [[x] for x in self.graph.nodes()]
        self.clustering = Clustering(self.graph, modules)
        self.current_mdl = self.get_mdl()

    def get_mdl(self):
        """get MDL (map equation) for the current graph/clustering

        """
        self.num_mdl_calculations += 1
        return self.clustering.get_mdl()

    def optimize_mdl(self):
        """attempt to optimize the map equation
        :returns: TODO

        """
        self.try_move_each_node()

    def try_change_node_module(self, node, new_module_id):
        """Change a node's module

        :node: node id to change
        :new_module_id: module_id for the new module
        :returns: new Clustering, new copy of graph

        """
        module_config = []
        for m in self.clustering.modules:
            this_module_nodes = set(m.nodes)
            if node in this_module_nodes:
                this_module_nodes.remove(node)
            if m.module_id == new_module_id:
                this_module_nodes.add(node)
            if this_module_nodes:
                module_config.append(this_module_nodes)
        new_graph = self.graph
        new_graph.node[node]['module_id'] = new_module_id
        new_clustering = Clustering(self.graph, module_config)
        return new_clustering, new_graph


    def try_move_each_node_once(self, current_graph=None, improvement=False):
        """Try to move each node into the module of its neighbor.
        As in the first phase of the Louvain method

        NOTE: as the clustering changes, module_ids do NOT remain constant.
        So make sure to check for module_id often

        :returns: improvement (bool): True if the MDL was successfully reduced from
                                        what it was at the start of the loop

        """
        if not current_graph:
            current_graph = self.graph

        for node in current_graph.nodes_iter():
            node_module_id = current_graph.node[node]['module_id']
            for _, nbr in nx.edges_iter(current_graph, node):
                nbr_module_id = current_graph.node[nbr]['module_id']
                if node_module_id != nbr_module_id:
                    new_clustering, new_graph = self.try_change_node_module(node, nbr_module_id)
                    new_mdl = new_clustering.get_mdl()
                    self.num_mdl_calculations += 1
                    if new_mdl < self.current_mdl:
                        logger.debug('updating best MDL: {:.4f} -> {:.4f}'.format(self.current_mdl, new_mdl))
                        self.graph = new_graph
                        self.clustering = new_clustering
                        self.current_mdl = new_mdl
                        improvement = True
        return improvement

    def try_move_each_node_repeatedly(self, current_graph=None):
        """Try to move each node into the module of its neighbor.
        As in the first phase of the Louvain method
        :returns: TODO

        :current_graph: TODO
        :returns: TODO

        """
        if not current_graph:
            current_graph = self.graph

        i = 0
        while True:
            i += 1
            improvement = False
            #old_mdl = self.current_mdl
            logger.debug('looping through each node: attempt {}'.format(i))
            improvement = self.try_move_each_node_once(current_graph)
            if not improvement:
                break

    def condense_graph(self, graph):
        """Make a new graph, where each node represents a cluster of the input graph
        and edges mean that a node in the cluster links to a node in the other cluster
        (weighted by number of these connections).

        This is known as a *quotient graph*

        NetworkX has a function nx.quotient_graph, but this does not deal with weighted edges
        So here is a modified version

        :graph: input graph to condense
        :returns: new graph

        """
        from networkx.algorithms.minors import equivalence_classes
        # define a node relation function to condense nodes
        module_dict = {n: graph.node[n]['module_id'] for n in graph.nodes_iter()}
        same_module = lambda u, v: module_dict[u] == module_dict[v]


        # make a new graph of the same type as the input graph
        H = type(graph)()
        # Compute the blocks of the partition on the nodes of G induced by the
        # equivalence relation R.
        H.add_nodes_from(equivalence_classes(graph, same_module))
        block_pairs = itertools.permutations(H, 2) if H.is_directed() else itertools.combinations(H, 2)
        for b, c in block_pairs:
            for u, v in itertools.product(b, c):
                if graph.has_edge(u, v):
                    weight = graph[u][v].get('weight', 1.0)
                    if H.has_edge(b, c):
                        H[b][c]['weight'] += weight
                    else:
                        H.add_edge(b, c, weight=weight)
        return H


        
def test(fname='2009_figure3ab.net'):
    t = PyInfomap(fname)
    logger.debug('Initial MDL: {}'.format(t.current_mdl))
    t.try_move_each_node_repeatedly()
    logger.debug('final MDL: {}'.format(t.current_mdl))
    for m in t.clustering.modules:
        logger.debug("moduleid {}: nodes: {}".format(m.module_id, m.nodes))
    logger.debug("number of MDL calculations performed: {}".format(t.num_mdl_calculations))

    logger.debug("")
    H = t.condense_graph(t.graph)
    H = t.process_graph(H)
    for n in H.edges_iter(data=True):
        logger.debug("{}".format(n))
    modules = [[n] for n in H.nodes()]
    c = Clustering(H, modules)
    logger.debug(c.get_mdl())
    ### SOMETHING IS WRONG, the new graph has higher MDL than the old (expanded) graph


def main(args):
    test(args.filename)

if __name__ == "__main__":
    total_start = timer()
    logger = logging.getLogger(__name__)
    logger.info(" ".join(sys.argv))
    logger.info( '{:%Y-%m-%d %H:%M:%S}'.format(datetime.now()) )
    import argparse
    parser = argparse.ArgumentParser(description="python infomap")
    parser.add_argument("filename", nargs='?', default='2009_figure3ab.net', help="filename for pajek network (.net)")
    parser.add_argument("--debug", action='store_true', help="output debugging info")
    global args
    args = parser.parse_args()
    if args.debug:
        logger.setLevel(logging.DEBUG)
        logger.debug('debug mode is on')
    else:
        logger.setLevel(logging.INFO)
    main(args)
    total_end = timer()
    logger.info('all finished. total time: {:.2f} seconds'.format(total_end-total_start))
