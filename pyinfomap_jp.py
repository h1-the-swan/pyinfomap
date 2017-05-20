import sys, os, time
from datetime import datetime
from timeit import default_timer as timer

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

    def get_mdl(self):
        """get MDL (map equation) for the current graph/clustering

        """
        return self.clustering.get_mdl()

    def optimize_mdl(self):
        """attempt to optimize the map equation
        :returns: TODO

        """
        self.try_move_each_node()

    def try_move_each_node(self, current_graph):
        """Try to move each node into the module of its neighbor.
        As in the first phase of the Louvain method
        :returns: TODO

        """
        for node in current_graph.nodes_iter():
            pass
        


def main(args):
    pass

if __name__ == "__main__":
    total_start = timer()
    logger = logging.getLogger(__name__)
    logger.info(" ".join(sys.argv))
    logger.info( '{:%Y-%m-%d %H:%M:%S}'.format(datetime.now()) )
    import argparse
    parser = argparse.ArgumentParser(description="python infomap")
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
