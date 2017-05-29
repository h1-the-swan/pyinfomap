# Python implementation of Infomap
# Module and Clustering classes by Daniel Halperin 2013
# Everything below that including PyInfomap class by Jason Portenoy 2017


import sys, os, time
from math import log
from datetime import datetime
from timeit import default_timer as timer
import itertools

import networkx as nx
import numpy as np
# from pyinfomap import Clustering

TAU = 0.15
PAGE_RANK = 'page_rank'
MODULE_ID = 'module_id'

import logging
logging.basicConfig(format='%(asctime)s %(name)s.%(lineno)d %(levelname)s : %(message)s',
        datefmt="%H:%M:%S",
        level=logging.INFO)
# logger = logging.getLogger(__name__)
logger = logging.getLogger('__main__').getChild(__name__)

def log2(prob):
    "Returns the log of prob in base 2"
    return log(prob, 2)

def entropy1(prob):
    """Half of the entropy function, as used in the InfoMap paper.
    entropy1(p) = p * log2(p)
    """
    if prob == 0:
        return 0
    return prob * log2(prob)

class Module:
    """Stores the information about a single module"""
    def __init__(self, module_id, nodes, graph):
        self.module_id = module_id
        self.nodes = frozenset(nodes)
        self.graph = graph
        self.prop_nodes = 1 - float(len(self.nodes)) / len(graph)
        # Set the module_id for every node
        for node in nodes:
            graph.node[node][MODULE_ID] = module_id
        # Compute the total PageRank
        self.total_pr = sum([graph.node[node][PAGE_RANK] for node in nodes])
        # Compute q_out, the exit probability of this module
        # .. Left half: tau * (n - n_i) / n * sum{alpha in i}(p_alpha)
        self.q_out = self.total_pr * TAU * self.prop_nodes
        # .. Right half: (1-tau) * sum{alpha in i}(sum{beta not in i}
        #                  p_alpha weight_alpha,beta)
        # This is what's in [RAB2009 eq. 6]. But it's apparently wrong if
        # node alpha has no out-edges, which is not in the paper.
        # ..
        # Implementing it with Seung-Hee's correction about dangling nodes
        for node in self.nodes:
            edges = graph.edges(node, data=True)
            page_rank = graph.node[node][PAGE_RANK]
            if len(edges) == 0:
                self.q_out += page_rank * self.prop_nodes * (1 - TAU)
                continue
            for (_, dest, data) in edges:
                if dest not in self.nodes:
                    self.q_out += page_rank * data['weight'] * (1 - TAU)
        self.q_plus_p = self.q_out + self.total_pr

    def get_codebook_length(self):
        "Computes module codebook length according to [RAB2009, eq. 3]"
        first = -entropy1(self.q_out / self.q_plus_p)
        second = -sum( \
                [entropy1(self.graph.node[node][PAGE_RANK]/self.q_plus_p) \
                    for node in self.nodes])
        return (self.q_plus_p) * (first + second)


class Clustering:
    "Stores a clustering of the graph into modules"
    def __init__(self, graph, modules):
        self.graph = graph
        self.total_pr_entropy = sum([entropy1(graph.node[node][PAGE_RANK]) \
                for node in graph])

        self.modules = [Module(module_id, module, graph) \
                for (module_id, module) in enumerate(modules)]

    def get_mdl(self):
        "Compute the MDL of this clustering according to [RAB2009, eq. 4]"
        total_qout = 0
        total_qout_entropy = 0
        total_both_entropy = 0
        for mod in self.modules:
            q_out = mod.q_out
            total_qout += q_out
            total_qout_entropy += entropy1(q_out)
            total_both_entropy += entropy1(mod.q_plus_p)
        term1 = entropy1(total_qout)
        term2 = -2 * total_qout_entropy
        term3 = -self.total_pr_entropy
        term4 = total_both_entropy
        return term1 + term2 + term3 + term4

    def get_index_codelength(self):
        "Compute the index codebook length according to [RAB2009, eq. 2]"
        if len(self.modules) == 1:
            return 0
        total_q = sum([mod.q_out for mod in self.modules])
        entropy = -sum([entropy1(mod.q_out / total_q) for mod in self.modules])
        return total_q * entropy

    def get_module_codelength(self):
        "Compute the module codebook length according to [RAB2009, eq. 3]"
        return sum([mod.get_codebook_length() for mod in self.modules])
def flatten(container):
    """flattens nested lists or sets

    """
    for i in container:
        if isinstance(i, (list, tuple, set, frozenset)):
            for j in flatten(i):
                yield j
        else:
            yield i

class PyInfomap(object):

    """Class for performing Infomap clustering on a graph. Created by Jason Portenoy"""

    def __init__(self, filename=None, graph=None, clustering=None, seed=None):
        self.filename = filename
        self.seed = seed
        self.randomstate = np.random.RandomState(seed)
        self.graph = graph
        self.clustering = clustering
        self.initial_mdl = 999
        self.current_mdl = 999
        self.num_mdl_calculations = 0
        self.meta_graph = None
        self.meta_clustering = None
        self.start_time = 0
        self.end_time = 0
        if self.filename and not self.graph:
            self.load_and_process_graph(self.filename)

    def process_graph(self, graph):
        """normalize edge weights, compute pagerank, store as node data

        :graph: networkx graph
        :returns: modified networkx graph

        """
        for node in graph:
            edges = graph.edges(node, data=True)
            try:
                total_weight = sum([data['weight'] for (_, _, data) in edges])
            except KeyError:
                # unweighted
                total_weight = len(edges)
            for (_, _, data) in edges:
                edge_weight = float(data.get('weight', 1.))
                data['orig_weight'] = edge_weight
                data['weight'] = edge_weight / total_weight
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
            self.initial_mdl = self.current_mdl
        return graph

    def initial_clustering(self):
        """Default initial clustering puts each node in its own cluster

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
        """Calculate the MDL from a meta clustering. Works on the original graph too

        :meta_clustering: Clustering object
        :returns: MDL (float)

        """
        current_graph = meta_clustering.graph
        if nx.is_isomorphic(current_graph, self.graph):
            # this is not actually a meta clustering.
            # just get the MDL normally
            clustering = meta_clustering
        else:
            module_config = []
            for m in meta_clustering.modules:
                # for meta_node in m.nodes:
                #     this_module = list(flatten(meta_node))
                # logger.debug("this_module: {}".format(this_module))
                this_module = list(flatten(m.nodes))
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

        #for node in current_graph.nodes_iter():
        for node in self.randomstate.permutation(current_graph.nodes()):
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
                        logger.debug(output_clustering(new_clustering))
                        improvement = True
        return current_graph, current_clustering, improvement

    def try_move_each_node_repeatedly(self, current_graph=None, current_clustering=None):
        """Try to move each node into the module of its neighbor.
        As in the first phase of the Louvain method

        :current_graph: graph object. the current graph to operate on
        :returns: 3-tuple: the modified current graph, the new clustering, whether there was an improvement (boolean)

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

        This corresponds to phase 2 of the Louvain method

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
        """Perform Infomap clustering to try to find the best partition

        """
        self.start_time = timer()
        current_graph = self.graph
        current_clustering = self.clustering
        improvement = False
        num_passes = 0
        while True:
            current_graph, current_clustering, improvement = self.try_move_each_node_repeatedly(current_graph, current_clustering)
            logger.debug(output_clustering(current_clustering))
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
        self.end_time = timer()

    def output_tree(self, graph=None, clustering=None, outstream=None):
        """Output the clustering in .tree format

        :graph: graph object. Defaults to the instance's graph (self.graph)
        :clustering: Clustering object. Defaults to the instance's clustering (self.clustering)
        :outstream: stream to output to. Defaults to standard out. If string, it will open a file and write to it (it will overwrite any file with the same name)

        """
        if not graph:
            graph = self.graph
        if not clustering:
            clustering = self.clustering
        opened_file = False
        if not outstream:
            outstream = sys.stdout
        elif isinstance(outstream, str):
            outstream = open(outstream, 'w')
            opened_file = True

        output_argv = " ".join(sys.argv)
        runtime = self.end_time - self.start_time
        outstream.write("# '{}' -> {} nodes partitioned in {:.0f}s from codelength {:.9f} in one level to codelength {:.9f} in 2 levels.\n".format(output_argv, graph.number_of_nodes(), runtime, self.initial_mdl, self.current_mdl))
        outstream.write("# path flow name node:\n")
        
        modules = clustering.modules
        for module_idx, module in enumerate(modules):
            nodes = list(flatten(module.nodes))
            for node_idx, node_name in enumerate(nodes):
                flow = graph.node[node_name][PAGE_RANK]
                node_id = graph.node[node_name]['id']
                outstream.write("{}:{} {:.7f} \"{}\" {}\n".format(module_idx, node_idx, flow, node_name, node_id))

        if opened_file:
            outstream.close()
                    

def output_clustering(clustering):
    lines = []
    for m in clustering.modules:
        #logger.debug("moduleid {}: nodes: {}".format(m.module_id, m.nodes))
        lines.append("moduleid {}: nodes: {}".format(m.module_id, m.nodes))
    return '\n'.join(lines)


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
        
def test(fname='2009_figure3ab.net', seed=None):
    t = PyInfomap(fname, seed=seed)
    logger.info('Initial MDL: {}'.format(t.current_mdl))
    # t.try_move_each_node_repeatedly()
    t.find_best_partition()
    t.output_tree()
    logger.info('final MDL: {}'.format(t.current_mdl))
    # for m in t.clustering.modules:
    #     logger.debug("moduleid {}: nodes: {}".format(m.module_id, m.nodes))
    # for n in t.graph.nodes(data=True):
    #     logger.debug(n)
    check_module_id_mismatch(t.graph, t.clustering)

            
    logger.debug("number of MDL calculations performed: {}".format(t.num_mdl_calculations))




def main(args):
    p = PyInfomap(args.filename, seed=args.seed)
    logger.info('Initial MDL: {}'.format(p.current_mdl))
    logger.info('Attempting two-level partition...')
    p.find_best_partition()
    logger.info('done.')
    logger.info('final MDL: {}'.format(p.current_mdl))
    logger.debug("number of MDL calculations performed: {}".format(p.num_mdl_calculations))
    
    if args.outdir:
        outfname = os.path.split(args.filename)[1]
        outfname = os.path.splitext(outfname)[0]
        outfname = "{}.tree".format(outfname)
        outfname = os.path.join(args.outdir, outfname)
    else:
        outfname = None

    p.output_tree(p.graph, p.clustering, outfname)

if __name__ == "__main__":
    total_start = timer()
    logger = logging.getLogger(__name__)
    logger.info(" ".join(sys.argv))
    logger.info( '{:%Y-%m-%d %H:%M:%S}'.format(datetime.now()) )
    import argparse
    parser = argparse.ArgumentParser(description="python infomap")
    parser.add_argument("filename", nargs='?', default='2009_figure3ab.net', help="filename for pajek network (.net)")
    parser.add_argument("outdir", nargs='?', default=None, help="output directory (default standard out)")
    parser.add_argument("--seed", type=int, default=None, help="random seed. If not specified, use numpy defaults.")
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
