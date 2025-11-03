import os
# os.chdir("/home/bjabbar/pyfiles")
os.chdir("C:\\Users\\bjabbar\\.spyder-py3")
import numpy as np
import math
from collections import defaultdict
import sys
import time
import copy
import heapq
from collections import deque
#%% Read the data
bus_data = {}
gend_data = {}
branch_data = {}
candidate_data={}
with open("6717busv13.dat", 'r') as file:
    lines = file.read().splitlines()

current_param = None
column_names = None
column_order = None

for line in lines:
    if line.startswith("param:"):
        current_param = line.split()[1].strip()
        if current_param == "bus_num":
            column_names = ["bus_type","bus_Pd","bus_Qd","bus_Gs","bus_Bs","bus_area","bus_Vm",
                            "bus_Va","bus_baseKv","bus_zone","bus_Vmax","bus_Vmin:="]
        elif current_param == "GEND:":
            column_names = ["genD_bus","genD_Pg","genD_Qg","genD_Qmax",
                            "genD_Qmin","genD_Vg","genD_mBase","genD_status",
                            "genD_Pmax","genD_Pmin","CG/MWh","genC_n"]
        elif current_param == "BRANCH:":
            column_names = ["branch_fbus","branch_tbus","branch_r","branch_x",
                            "branch_scpt","branch_b","branch_rateA","branch_rateB",
                            "branch_rateC","branch_ratio","branch_angle","branch_status",
                            "angmin", "angmax", "Fmax", "IC(Euro)"]
        elif current_param == "candidate":
            column_names = ["branch_fbus","branch_tbus","branch_x","branch_scpt","Fmax","IC", "max_lines"]
        column_order = line.split()[2:]

        continue

    # Skip empty lines
    if not line.strip():
        continue

    # Split the line into fields
    fields = line.split()

    # Check if we have a valid parameter and fields
    if current_param and column_order and len(fields) >= 6:  # Assuming each line has 14 columns
        # Extract the key from the first column
        key = int(fields[0])

        # Create a dictionary for the current line using the column order
        data_dict = {column_name: float(val) if "." in val else int(val) for
                     column_name, val in zip(column_order, fields[1:])}

        # Store the data dictionary with the key as the dictionary key
        if current_param == "bus_num":
            bus_data[key] = data_dict
        elif current_param == "GEND:":
            gend_data[key] = data_dict
        elif current_param == "BRANCH:":
            branch_data[key] = data_dict
        elif current_param == "candidate":
            candidate_data[key] = data_dict

Sbase=100
def round_up(value, decimal_places):
    multiplier = 10 ** decimal_places
    return math.ceil(value * multiplier) / multiplier


def adj_list(branch_data,Sbase):
    adj_all = defaultdict(lambda: defaultdict(float))    
    adj_exist = defaultdict(lambda: defaultdict(float)) 
    for info in branch_data.values():
        u = info['branch_fbus']
        v = info['branch_tbus']
        w = info['branch_x'] * info['Fmax'] / Sbase 
        adj_all[u][v] += w
        adj_all[v][u] += w
        if info['branch_status']:
            adj_exist[u][v] += w
            adj_exist[v][u] += w
    
    adj_all = {u: list(v_and_w.items()) for u, v_and_w in adj_all.items()}
    adj_exist = {u: list(v_and_w.items()) for u, v_and_w in adj_exist.items()}
    return adj_all, adj_exist


adj_all_lines,adj_existing_lines=adj_list(branch_data,1)


def edge_list(branch_data,Sbase):
    all_edges = defaultdict(float)  
    existing_edges = defaultdict(float)   
    for info in branch_data.values():
        u = info['branch_fbus']
        v = info['branch_tbus']
        w = info['branch_x'] * info['Fmax'] * Sbase 

        all_edges[(u, v)] += w

        # Existing lines
        if info['branch_status']:
            existing_edges[(u, v)] += w

    # Convert to original structures
    edge_list1 = [[u, v, w] for (u, v), w in all_edges.items()]
    edge_list0 = [[u, v, w] for (u, v), w in existing_edges.items()]
    
    return edge_list0, edge_list1


def dijkstra_adj(adjacency_list, source,num_vertices):
    vertices = list(range(1, num_vertices + 1))
    
    # Initialize distances and predecessors for all vertices.
    # Every vertex an initial distance of infinity.
    d = {vertex: float('inf') for vertex in vertices}
    d[source] = 0
    pred = {vertex: None for vertex in vertices}
    
    # Use a min-heap 
    heap = [(0, source)]  # (distance, vertex)
    visited = set()
    
    while heap:
        current_distance, u = heapq.heappop(heap)
        if u in visited:
            continue
        visited.add(u)
        
        for neighbor, weight in adjacency_list.get(u, []):
            if neighbor not in visited:
                distance = d[u] + weight
                if distance < d[neighbor]:
                    d[neighbor] = distance
                    pred[neighbor] = u
                    heapq.heappush(heap, (d[neighbor], neighbor))
    
    # Shortest path for each vertex
    paths = {}
    for vertex in vertices:
        if d[vertex] < float('inf'):
            path = []
            current = vertex
            while current is not None:
                path.append(current)
                current = pred[current]
            path.reverse()  # Reverse to get the path from source to the vertex.
        else:
            path = []
        paths[vertex] = path
    
    return d, paths

def adj(adj_list,node):
    ls=[]
    for neighbor, weight in adj_list[node]:
        ls.append(neighbor)
    return ls



def is_connected(adj_ls, i, j):
    if i not in adj_ls or j not in adj_ls:
        return False
    
    visited = set()
    queue = deque([i])  # Use a queue for BFS
    
    while queue:
        node = queue.popleft()
        if node == j:
            return True  # Path exists
        
        if node not in visited:
            visited.add(node)
            # Add unvisited neighbors to the queue
            queue.extend(neighbor[0] for neighbor in adj_ls[node] if neighbor[0] not in visited)
    
    return False    

def has_edge(edgelist,i,j):
    if (i,j) in edgelist:
        return True
    else:
        return False

def edge_weight(adj_list, i, j):
    if i in adj_list:
        for neighbor, weight in adj_list[i]:
            if neighbor == j:
                return weight  # Return the weight if the neighbor matches j
    return float('inf')


edge_existing_lines,edge_all_lines=edge_list(branch_data,Sbase)
num_vertices=len(set(adj_all_lines) | {v for edges in adj_all_lines.values() for v, _ in edges})

def LSPC(adj_existing_lines,adj_all_lines,edge_existing_lines,edge_all_lines,num_vertices):
    #%% Distance list 
    num_vertices=len(set(adj_all_lines) | {v for edges in adj_all_lines.values() for v, _ in edges})
    dist_list=[{i: float('inf') for i in range(1, num_vertices + 1)}]
    start_time = time.time()
    for bus in range(1,num_vertices + 1):
        ds,_= dijkstra_adj(adj_existing_lines, bus,num_vertices)
        dist_list.append(ds)
    print(' Execution time is {} seconds'.format(round(time.time() - start_time,5)))
    print('SPP Done!')
    res_spp=[]
    res_spp.append(round(time.time() - start_time,5))
    np.savetxt('results0.txt', res_spp, delimiter='\t', fmt='%.4f')
    
    #%% LSPC Phase I
    
    dist_list1 = copy.deepcopy(dist_list)
    count=0
    start_time = time.time()
    for bus_i in range(1, num_vertices + 1):
        count+=1
        if count%10==0:
            print(count)
            print(' Execution time is {} seconds'.format(round(time.time() - start_time,5)))
        for bus_j in range(bus_i+1, num_vertices + 1):
            start_node = bus_i
            end_node = bus_j
            lsp=0
            if dist_list1[start_node][end_node]==float('inf'):
                sn=adj(adj_all_lines,start_node)
                en=adj(adj_all_lines,end_node)
                ng=[]
                try:
                    sn.remove(end_node)
                except:
                    pass
                try:
                    en.remove(start_node)
                except:
                    pass
                
                for i in sn:
                    for j in en:
                        ng.append([i,j])
    
            
                adjacency_bar= {}
                for u, v, weight in edge_all_lines:
                    if u not in adjacency_bar :
                        adjacency_bar[u] = []
                    if u!=start_node and u!=end_node:
                        adjacency_bar[u].append((v, weight))
                    
                    if v not in adjacency_bar:
                        adjacency_bar[v] = []
                    if v!=start_node and v!=end_node:
                        adjacency_bar[v].append((u, weight))
                
                for i in ng[:]:
                    if not is_connected(adjacency_bar,i[0],i[1]):
                        ng.remove(i)
                
                # (i,n_j) replaces (n_i,n_j) if needed
                repl=[]
                remv=[]
                for i in ng[:]:
                    if has_edge(edge_existing_lines,start_node, i[0]):
                    
                        if [start_node,i[1]] not in ng and [start_node,i[1]] not in repl:
                            repl.append([start_node,i[1]])
                        remv.append(i)
                for i in repl:
                    ng.append(i)
                for i in remv:
                    ng.remove(i)
                    
                repl=[]
                remv=[]
                for j in ng[:]:
                    if has_edge(edge_existing_lines,end_node, j[1]):
                    
                        if [j[0],end_node] not in ng and [j[0],end_node] not in repl:
                            repl.append([j[0],end_node])
                        remv.append(j)
                for i in repl:
                    ng.append(i)
                for i in remv:
                    ng.remove(i)
                
                
                # Consider only the required (n_i,n_j)
                for i in ng[:]:
                    if i[0] != start_node and is_connected(adj_existing_lines,start_node,i[1]):
                        ng.remove(i)
                        if [start_node,i[1]] not in ng:
                            ng.append([start_node,i[1]])
                for i in ng[:]:
                    if i[1] != end_node and is_connected(adj_existing_lines,end_node,i[0]):
                        ng.remove(i)
                        if [i[0],end_node] not in ng:
                            ng.append([i[0],end_node])

                cn=0  
                for i in ng[:]:
                    if is_connected(adj_existing_lines, i[0], i[1]):
                        cn+=1
                if cn==len(ng):
                    if any((sublist[:2] == [end_node, start_node] or sublist[:2] == [start_node,end_node]) for sublist in edge_all_lines) and [start_node,end_node] not in ng:
                        ng.append([start_node,end_node])
    
                    lsp=0
                    for i in ng[:]:
                        if i==[start_node,end_node]:
                            lsp=max(edge_weight(adj_all_lines, start_node,end_node),lsp)
                        elif i[0]==start_node:
                            lsp=max(lsp,dist_list[start_node][i[1]] + edge_weight(adj_all_lines, i[1],end_node))
                        elif i[1]==end_node:
                            lsp=max(lsp, edge_weight(adj_all_lines, start_node,i[0]) + dist_list[end_node][i[0]])
                        else:
                            lsp=max(lsp, edge_weight(adj_all_lines, start_node,i[0]) + dist_list[i[0]][i[1]] + edge_weight(adj_all_lines, i[1],end_node))
        
                    dist_list1[bus_i][bus_j]=round_up(lsp,5)
                    dist_list1[bus_j][bus_i]=round_up(lsp,5)
    
                
    print(' Execution time is {} seconds'.format(round(time.time() - start_time,5)))
    print('LSPC-I Done!')
