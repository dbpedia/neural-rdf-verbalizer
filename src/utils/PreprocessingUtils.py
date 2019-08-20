"""
Utils that are used in the preprocessing pipeline
to convert source triples into graphs and crate tensors
"""
import networkx as nx
import numpy as np


def PreProcessRolesModel(path):
    """
    The preprocessing function that takes in a set of RDF triples
    and converts that into the graph dataset we will be using for
    our models using the Role based processing scheme.
    It loads the input from the disk, and iterates through each line
    of RDF triples

    ex - Let the triple set be
    " Dwarak | Loves | Physics <TSP> Dwarak | lives_in | India "
    There could be sets from one to seven triples.
    We intially extract each triple and create a triple list.
    triple 1 = Dwarak | loves | physics
    triple 2 = Dwaral | lives_in | India

    We create a Networkx Multi-Di graph objet with the subject
    and predicate of the triple as nodes and the predicate as
    the edge. After doing this for all the triples in the triple
    list, we can get the adjacency matrix of the graph as a numpy
    array. We also give each node a role.
    Subject - appears on the left side of the triple
    Object - appears on the right side of the triple
    Bridge - appears on both sides
    Predicate - forms an edge between nodes.

    This way we impart the structural information of the triple set
    into the models inputs.

    :param path: Path to the triple source file
    :type path: str
    :return: Adjacency matrix, nodes, edges, roles
    :rtype:Numpy matrix, lists

    """
    adj = []
    degree_mat = []
    tensor = []
    train_nodes = []
    roles = []
    edges = []
    dest = open(path, 'r')
    count = 0
    for line in dest:
        g = nx.MultiDiGraph()
        temp_edge = []
        triple_list = line.split('< TSP >')
        for l in triple_list:
            l = l.strip().split(' | ')
            g.add_edge(l[0], l[1])
            g.add_edge(l[1], l[2])
            g.add_edge(l[0], l[2])
            temp_edge.append(l[1])
        edges.append(temp_edge)
        train_nodes.append(list(g.nodes()))

        # set roles
        roles_ = []
        for node in list(g.nodes()):
            role = ''
            for l in triple_list:
                l = l.strip().split(' | ')

                if l[0] == node:
                    if role == 'object':
                        role = 'bridge'
                    else:
                        role = 'subject'
                elif l[1] == node:
                    role = 'predicate'
                elif l[2] == node:
                    if role == 'subject':
                        role = 'bridge'
                    else:
                        role = 'object'
            roles_.append(role)
        roles.append(roles_)

        array = nx.to_numpy_matrix(g)
        result = np.zeros((16, 16))

        result[:array.shape[0], :array.shape[1]] = array

        result += np.identity(16)

        adj.append(result)
        diag = np.sum(result, axis=0)
        D = np.matrix(np.diag(diag))
        degree_mat.append(D)
        result = D ** -1 * result

    dest.close()

    return adj, train_nodes, roles, edges


def PreProcess(path, lang):
    """
    The preprocessing function that takes in a set of RDF triples
    and converts that into the graph dataset we will be using for
    our models using the reification scheme.
    It loads the input from the disk, and iterates through each line
    of RDF triples

    ex - Let the triple set be
    " Dwarak | Loves | Physics <TSP> Dwarak | lives_in | India "
    There could be sets from one to seven triples.
    We intially extract each triple and create a triple list.
    triple 1 = Dwarak | loves | physics
    triple 2 = Dwaral | lives_in | India

    Then we create a Networkx Mutli-Di Graph object that creates
    a graph with all entities in the triples as nodes and describes the
    edge between them as connection of nodes.
    node 1 - Dwarak, node2 - loves, node3 - Physics
    Label of edge between node1 - node2 - A_ZERO
    Label of edge between node3 - node3 - A_ONE

    This way we impart the structural information of the triple set
    into the models inputs.

    :param path: The path to the RDF triple source file
    :type path: str
    :param lang: The language on which we are operating
    :type lang: str
    :return: nodes_list, node1 of edges, node2 of edges, and edge labels
    :rtype:list

    """
    nodes = []
    labels = []
    node1 = []
    node2 = []
    dest = open(path, 'r')
    lang = '<' + lang + '>'
    for line in dest:
        g = nx.MultiDiGraph()
        temp_label = []
        temp_node1 = []
        temp_node2 = []
        triple_list = line.split('<TSP>')
        # triple_list = triple_list[:-1]
        for l in triple_list:
            l = l.strip().split(' | ')
            # l = [lang+' '+x for x in l]
            g.add_edge(l[0], l[1], label='A_ZERO')
            # g.add_edge(l[1], l[0])
            g.add_edge(l[1], l[2], label='A_ONE')
            # g.add_edge(l[2], l[1])
        node_list = list(g.nodes())
        node_list.insert(0, lang)
        # print(node_list)
        nodes.append(node_list)
        edge_list = list(g.edges.data())
        for edge in edge_list:
            temp_node1.append(edge[0])
            temp_node2.append(edge[1])
            label = (edge[2]['label'])
            temp_label.append(label)
        node1.append(temp_node1)
        node2.append(temp_node2)
        labels.append(temp_label)

    dest.close()

    return nodes, labels, node1, node2
