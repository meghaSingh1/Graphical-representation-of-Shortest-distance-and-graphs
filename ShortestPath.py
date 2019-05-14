import networkx as nx
import matplotlib.pyplot as plt
import  numpy as np
import tkinter

g = nx.Graph()
g2 = nx.Graph()
exc = 0

#graph = {'a': {'b': 10, 'c': 3}, 'b': {'c': 1, 'd': 2}, 'c': {'b': 4, 'd': 8, 'e': 2}, 'd': {'e': 7}, 'e': {'d': 9}}
graph ={}
try:
    n = int(input('Enter the number of nodes present in graph'))
    if n <=0:
        exc =1
        raise ValueError("You entered negative or zero edge...Please Enter positive edge only")
except ValueError as ve:
    print(ve)

for i in range(n):
    if exc ==0:
        node = input('Enter the name of NODE')
    try:
        n2 = int(input('Enter the number of node connected with it.Please enter positive integers only.'))
        if n2 <= 0:
            exc = 1
            raise ValueError("you entered non positive ")
        if type(n2) != int:
            exc =1
            raise ValueError()
    except ValueError:
        print("That's not a valid input. we are looking for an integer.")
    except ValueError as ve:
        print(ve)
    slist ={}
    if exc ==0:
        for j in range(n2):
            snode= input("Enter the name of    CONNECTED node")
            wt = int(input('Enter the weight of this edge'))
            slist[snode] = wt
        graph[node] = slist


def get_coordinates_in_circle(n):
    thetas = [2* np.pi*(float(i)/n) for i in range(n)]
    returnlist = [(np.cos(theta), np.sin(theta)) for theta in thetas]
    return returnlist

def plotgraph(graph):
    for key in graph.keys():
        g.add_node(key)
    for keynode,nodelist in graph.items():
        for cnode,cweight in nodelist.items():
            g.add_edge(keynode,cnode, weight = cweight )
    #weight = nx.get_edge_attributes(g, 'weight')

    #len(dps_2211)) ,
    fixed_nodes = [n for n in g.nodes() if n in graph.keys()]
    circular_positions = get_coordinates_in_circle(len(fixed_nodes))
    pos = {}
    for i, p in enumerate(fixed_nodes):
        pos[p] = circular_positions[i]

    #colours = get_node_colours(g,"gender")
    pos = nx.spring_layout(g,pos = pos ,fixed =fixed_nodes)
    nx.draw_networkx(g,pos,cmap=plt.get_cmap('jet'),node_size=500,alpha=0.8,with_labels=True)
    #nx.draw_networkx_edges(g,pos,alpha=0.02)
    #nx.draw(g,pos, with_labels=True)
    edge_labels = dict([((u, v,), d['weight'])
                        for u, v, d in g.edges(data=True)])
    nx.draw_networkx_edge_labels(g, pos, edge_labels=edge_labels)
    plt.figure(1)
    plt.show()

    #nx.draw_networkx_edge_labels(weight,)

    print(nx.info(g))
if exc ==0:
    plotgraph(graph)

def dijkstra(graph, start, goal):
    shortest_distance = {}
    predecessor = {}
    unseenNodes = graph
    infinity = 9999999
    path = []
    for node in unseenNodes:
        shortest_distance[node] = infinity
    shortest_distance[start] = 0
    while unseenNodes:
        minNode = None
        for node in unseenNodes:
            if minNode is None:
                minNode = node
            elif shortest_distance[node] < shortest_distance[minNode]:
                minNode = node

        for childNode, weight in graph[minNode].items():
            if weight + shortest_distance[minNode] < shortest_distance[childNode]:
                shortest_distance[childNode] = weight + shortest_distance[minNode]
                predecessor[childNode] = minNode
        unseenNodes.pop(minNode)

    currentNode = goal
    while currentNode != start:
        try:
            path.insert(0, currentNode)
            currentNode = predecessor[currentNode]
        except KeyError:
            print('Path not reachable')
            break
    path.insert(0, start)
    if shortest_distance[goal] != infinity:
        print('Shortest distance is ' + str(shortest_distance[goal]))
        print('And the path is ' + str(path))

    for nodes in path:
        g2.add_node(nodes)
    temp = path[0]
    for edges in path:
        g2.add_edge(temp, edges)
        temp = edges

    plt.figure(2)
    nx.draw_networkx(g2, node_size=500, alpha=0.8, with_labels=True)
    #nx.draw_networkx_edges(g2, pos, alpha=0.02)
    #plotgraph(graph)
    # print(shortest_distance)
    plt.show()
if exc ==0:
    x = input("Enter the start node for shortest distance ")
    y = input("Enter the end node for shortest distance ")

if exc ==0:
    dijkstra(graph, x, y)

def mst_prim(g):
    """Return a minimum cost spanning tree of the connected graph g."""
    mst = Graph()  # create new Graph object to hold the MST

    # if graph is empty
    if not g:
        return mst

    # nearest_neighbour[v] is the nearest neighbour of v that is in the MST
    # (v is a vertex outside the MST and has at least one neighbour in the MST)
    nearest_neighbour = {}
    # smallest_distance[v] is the distance of v to its nearest neighbour in the MST
    # (v is a vertex outside the MST and has at least one neighbour in the MST)
    smallest_distance = {}
    # v is in unvisited iff v has not been added to the MST
    unvisited = set(g)

    u = next(iter(g))  # select any one vertex from g
    mst.add_vertex(u.get_key())  # add a copy of it to the MST
    unvisited.remove(u)

    # for each neighbour of vertex u
    for n in u.get_neighbours():
        if n is u:
            # avoid self-loops
            continue
        # update dictionaries
        nearest_neighbour[n] = mst.get_vertex(u.get_key())
        smallest_distance[n] = u.get_weight(n)

    # loop until smallest_distance becomes empty
    while (smallest_distance):
        # get nearest vertex outside the MST
        outside_mst = min(smallest_distance, key=smallest_distance.get)
        # get the nearest neighbour inside the MST
        inside_mst = nearest_neighbour[outside_mst]

        # add a copy of the outside vertex to the MST
        mst.add_vertex(outside_mst.get_key())
        # add the edge to the MST
        mst.add_edge(outside_mst.get_key(), inside_mst.get_key(),
                     smallest_distance[outside_mst])
        mst.add_edge(inside_mst.get_key(), outside_mst.get_key(),
                     smallest_distance[outside_mst])

        # now that outside_mst has been added to the MST, remove it from our
        # dictionaries and the set unvisited
        unvisited.remove(outside_mst)
        del smallest_distance[outside_mst]
        del nearest_neighbour[outside_mst]

        # update dictionaries
        for n in outside_mst.get_neighbours():
            if n in unvisited:
                if n not in smallest_distance:
                    smallest_distance[n] = outside_mst.get_weight(n)
                    nearest_neighbour[n] = mst.get_vertex(outside_mst.get_key())
                else:
                    if smallest_distance[n] > outside_mst.get_weight(n):
                        smallest_distance[n] = outside_mst.get_weight(n)
                        nearest_neighbour[n] = mst.get_vertex(outside_mst.get_key())

    return mst

'''
mst = mst_prim(graph)
print('Minimum Spanning Tree:')
plotgraph(mst)
'''