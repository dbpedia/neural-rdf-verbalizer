"""
Utils that are used in the preprocessing pipeline
to convert source triples into graphs and crate tensors
"""
import numpy as np
import networkx as nx

def PreProcessRolesModel(path):
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
        result = D**-1 * result

    dest.close()

    return adj, train_nodes, roles, edges

def PreProcess(path, lang):
    nodes = []
    labels = []
    node1 = []
    node2 = []
    dest = open(path, 'r')
    lang = '<'+lang+'>'
    for line in dest:
        g = nx.MultiDiGraph()
        temp_label = []
        temp_node1 = []
        temp_node2 = []
        triple_list = line.split('<TSP>')
        #triple_list = triple_list[:-1]
        for l in triple_list:
            l = l.strip().split(' | ')
            #l = [lang+' '+x for x in l]
            g.add_edge(l[0], l[1], label='A_ZERO')
            #g.add_edge(l[1], l[0])
            g.add_edge(l[1], l[2], label='A_ONE')
            #g.add_edge(l[2], l[1])
        node_list = list(g.nodes())
        #node_list.append(lang)
        #print(node_list)
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