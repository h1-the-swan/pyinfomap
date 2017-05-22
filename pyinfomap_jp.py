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

def flatten(container):
    """TODO: Docstring for flatten.
    :returns: TODO

    """
    for i in container:
        if isinstance(i, (list, tuple, set, frozenset)):
            for j in flatten(i):
                yield j
        else:
            yield i

class PyInfomap(object):

    """Docstring for PyInfomap. """

    def __init__(self, filename=None, graph=None, clustering=None):
        self.filename = filename
        self.graph = graph
        self.clustering = clustering
        self.current_mdl = 999
        self.num_mdl_calculations = 0
        self.meta_graph = None
        self.meta_clustering = None
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
                data['orig_weight'] = data['weight']
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

    def try_change_node_module(self, graph, clustering, node, new_module_id):
        """Change a node's module

        :node: node id to change
        :new_module_id: module_id for the new module
        :returns: new Clustering, new copy of graph

        """
        module_config = []
        for m in clustering.modules:
            this_module_nodes = set(m.nodes)
            if node in this_module_nodes:
                this_module_nodes.remove(node)
            if m.module_id == new_module_id:
                this_module_nodes.add(node)
            if this_module_nodes:
                module_config.append(this_module_nodes)
        # logger.debug(module_config)
        new_graph = graph.copy()
        new_clustering = Clustering(new_graph, module_config)
        for m in new_clustering.modules:
            for module_node in m.nodes:
                new_graph.node[module_node]['module_id'] = m.module_id
        return new_clustering, new_graph



    def get_mdl_from_meta_clustering(self, meta_clustering):
        """TODO: Docstring for get_mdl_from_meta_clustering.

        :meta_clustering: TODO
        :returns: TODO

        """
        current_graph = meta_clustering.graph
        if nx.is_isomorphic(current_graph, self.graph):
            # this is not actually a meta clustering.
            # just get the MDL normally
            clustering = meta_clustering
        else:
            module_config = []
            for m in meta_clustering.modules:
                for meta_node in m.nodes:
                    this_module = list(flatten(meta_node))
                # logger.debug("this_module: {}".format(this_module))
                module_config.append(this_module)
            # module_config = current_graph.nodes()  # list of sets of nodes in original graph
            # logger.debug(module_config)
            clustering = Clustering(self.graph, module_config)

        self.num_mdl_calculations += 1
        return clustering.get_mdl()

    def try_move_each_node_once(self, current_graph=None, current_clustering=None, improvement=False):
        """Try to move each node into the module of its neighbor.
        As in the first phase of the Louvain method

        NOTE: as the clustering changes, module_ids do NOT remain constant.
        So make sure to check for module_id often

        :returns: improvement (bool): True if the MDL was successfully reduced from
                                        what it was at the start of the loop

        """
        if not current_graph:
            current_graph = self.graph
        if not current_clustering:
            current_clustering = self.clustering

        for node in current_graph.nodes_iter():
            node_module_id = current_graph.node[node]['module_id']
            for _, nbr in nx.edges_iter(current_graph, node):
                nbr_module_id = current_graph.node[nbr]['module_id']
                if node_module_id != nbr_module_id:
                    new_clustering, new_graph = self.try_change_node_module(current_graph, current_clustering, node, nbr_module_id)
                    new_mdl = self.get_mdl_from_meta_clustering(new_clustering)
                    # logger.debug("moved node {}. new MDL {:.4f}".format(node, new_mdl))
                    if new_mdl < self.current_mdl:
                        logger.debug('updating best MDL: {:.4f} -> {:.4f}'.format(self.current_mdl, new_mdl))
                        # self.graph = new_graph
                        current_graph = new_graph
                        # self.clustering = new_clustering
                        current_clustering = new_clustering
                        node_module_id = current_graph.node[node]['module_id']
                        self.current_mdl = new_mdl
                        improvement = True
        return current_graph, current_clustering, improvement

    def try_move_each_node_repeatedly(self, current_graph=None, current_clustering=None):
        """Try to move each node into the module of its neighbor.
        As in the first phase of the Louvain method
        :returns: TODO

        :current_graph: TODO
        :returns: TODO

        """
        if not current_graph:
            current_graph = self.graph
        if not current_clustering:
            current_clustering = self.clustering

        i = 0
        improvement_overall = False
        while True:
            i += 1
            improvement = False
            #old_mdl = self.current_mdl
            logger.debug('looping through each node: attempt {}'.format(i))
            current_graph, current_clustering, improvement = self.try_move_each_node_once(current_graph, current_clustering)
            if not improvement:
                break
            else:
                improvement_overall = True
        return current_graph, current_clustering, improvement_overall

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
        # logger.debug(H.number_of_nodes())
        # logger.debug(H.nodes(data=True))
        block_pairs = itertools.permutations(H, 2) if H.is_directed() else itertools.combinations(H, 2)
        for b, c in block_pairs:
            # self links
            for block in [b, c]:
                for u, v in itertools.permutations(block, 2):
                    if graph.has_edge(u, v):
                        # add self link to H
                        weight = graph[u][v].get('orig_weight', 1.0)
                        if H.has_edge(block, block):
                            H[block][block]['weight'] += weight
                        else:
                            H.add_edge(block, block, weight=weight)

            # links between blocks
            for u, v in itertools.product(b, c):
                if graph.has_edge(u, v):
                    weight = graph[u][v].get('orig_weight', 1.0)
                    if H.has_edge(b, c):
                        H[b][c]['weight'] += weight
                    else:
                        H.add_edge(b, c, weight=weight)
        return H

    def find_best_partition(self):
        """TODO: Docstring for find_best_partition.
        :returns: TODO

        """
        current_graph = self.graph
        current_clustering = self.clustering
        improvement = False
        num_passes = 0
        while True:
            current_graph, current_clustering, improvement = self.try_move_each_node_repeatedly(current_graph, current_clustering)
            output_clustering(current_clustering)
            if not improvement:
                break

            current_graph = self.condense_graph(current_graph)
            current_graph = self.process_graph(current_graph)
            modules_init = [[x] for x in current_graph.nodes()]
            current_clustering = Clustering(current_graph, modules_init)
            num_passes += 1
            logger.debug("pass {} complete".format(num_passes))

        if nx.is_isomorphic(current_graph, self.graph):
            self.clustering = current_clustering
        else:
            # this is a meta clustering. recover the original graph
            module_config = []
            for m in current_clustering.modules:
                for meta_node in m.nodes:
                    this_module = list(flatten(meta_node))
                # logger.debug("this_module: {}".format(this_module))
                module_config.append(this_module)
            # module_config = current_graph.nodes()  # list of sets of nodes in original graph
            self.clustering = Clustering(self.graph, module_config)

        for m in self.clustering.modules:
            for node in m.nodes:
                self.graph.node[node]['module_id'] = m.module_id
                    

def output_clustering(clustering):
    for m in clustering.modules:
        logger.debug("moduleid {}: nodes: {}".format(m.module_id, m.nodes))


def check_module_id_mismatch(graph, clustering):
    err = False
    for m in clustering.modules:
        module_nodes = m.nodes
        for module_node in module_nodes:
            if graph.node[module_node]['module_id'] != m.module_id:
                logger.debug('node {} module id mismatch'.format(module_node))
                err = True
    if not err:
        logger.debug("no module id mismatches")
        
def test(fname='2009_figure3ab.net'):
    t = PyInfomap(fname)
    logger.debug('Initial MDL: {}'.format(t.current_mdl))
    # t.try_move_each_node_repeatedly()
    t.find_best_partition()
    logger.debug('final MDL: {}'.format(t.current_mdl))
    # for m in t.clustering.modules:
    #     logger.debug("moduleid {}: nodes: {}".format(m.module_id, m.nodes))
    # for n in t.graph.nodes(data=True):
    #     logger.debug(n)
    check_module_id_mismatch(t.graph, t.clustering)

            
    logger.debug("number of MDL calculations performed: {}".format(t.num_mdl_calculations))




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
