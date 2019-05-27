""" Script to preprocess the .triple files to 
create graphs using the networkx library, and
save the adjacency matrix as numpy array 
"""
import numpy as np 
import networkx as nx 
import argparse 

parser = argparse.ArgumentParser(description="preprocessor parser")
parser.add_argument(
'--path', type=str, required=True, help='Path to source.lex file')
parser.add_argument(
'--opt', type=str, required=True, help='Adjacency processing or feature')
args = parser.parse_args()

def pre_process(path):
    dest = open(path, 'r')
    count =0
    for line in dest:
        g = nx.MultiDiGraph()
        
        triple_list = line.split('< TSP >')
        for l in triple_list:
            l = l.strip().split(' | ')
            print(l)
            g.add_edge(l[0], l[2], label=l[1])
        nodes.append(list(g.nodes))
        array = nx.to_numpy_array(g)
        print(array)
        result = np.zeros((16,16))
        result[:array.shape[0],:array.shape[1]] = array
        tensor.append(result)
    
    dest.close()


if __name__ == '__main__':
    if args.opt == 'adj':
        tensor = [] 
        nodes = []
        pre_process(args.path)
        tensor = np.array(tensor)
        print(tensor.shape)
        np.save('data/graph_adj', tensor)
        np.save('data/graph_nodes', nodes)
    else:
        nodes = np.load('data/graph_nodes.npy')
        ebmeddings = [] 
        
