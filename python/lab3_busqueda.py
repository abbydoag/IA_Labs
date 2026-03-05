import heapq
import networkx as nx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import deque



#Uniform Cost Search (Desinformada)
class Node:
    def __init__(self, estado, parent=None, action=None, cost=0, depth=0, heuristica=0):
        self.estado = estado
        self.parent = parent
        self.action = action
        self.cost = cost
        self.depth = depth
        self.heuristica = heuristica
        self.f = cost + heuristica

    def __str__(self):
        return f"Estado: {self.estado}\n\tCosto:{self.cost}    Heurística:{self.heuristica}"
    
    def __repr__(self):
        return self.__str__()

class FIFO: #añado al final, extrae del inico
    def __init__(self):
        self.items = deque()
        self.name = 'FIFO'
    def empty(self):
        return len(self.items) == 0
    def top(self):
        if not self.empty():
            return self.items[0]
        return None
    def pop(self):
        if not self.empty():
            return self.items.popleft()
        return None
    def add(self, element):
        self.items.append(element)
        return list(self.items)
    
    def __str__(self):
        return f"FIFO: [{', '.join(str(item) for item in self.items)}]"
    def __len__(self):
        return len(self.items)
    

class LIFO: #añadir y extraer del final
    def __init__(self):
        self.items= []
        self.name = 'LIFO'
    def empty(self):
        return len(self.items) == 0 
    def top(self):
        if not self.empty():
            return self.items[-1] #final/ultimo
        return None
    def pop(self):
        if not self.empty():
            return self.items.pop() #ultimo
        return None
    def add(self, elemento):
        self.items.append(elemento)
        return self.items
    
    def __str__(self):
        return f"LIFO: [{', '.join(str(item) for item in self.items)}]"
    def __len__(self):
        return len(self.items)

class Priority:
    #x>a, a mayor prioridad
    def __init__(self):
        self.items = []  #prioridad, contador, elemento
        self.contador = 0 #si proridad =
        self.nombre = "PRIORITY"
    def empty(self):
        return len(self.items) == 0
    def top(self):
        if not self.empty():
            return self.items[0][2]
        return None
    def pop(self):
        if not self.empty():
            prioridad, contador, elemento = heapq.heappop(self.items)
            return elemento
        return None
    def add(self, elemento, prioridad=None):
        """
        Uniform Cost Search prioridad: costo acumulado
        A* prioridad: f = g + h (self.f)
        """
        if prioridad is None:
            if hasattr(elemento, 'f'):
                prioridad = elemento.f
            else:
                prioridad = elemento.costo
        
        heapq.heappush(self.items, (prioridad, self.contador, elemento))
        self.contador += 1
        return [(p, c, e.estado) for p, c, e in self.items]
    
    def __str__(self):
        elementos = [(p, e.estado) for p, c, e in self.items]
        return f"PRIORITY: [{elementos}]"   
    def __len__(self):
        return len(self.items)
    

def datos_excel(archivo_costos="funcion_de_costo.xlsx", archivo_heuristicas="heuristica.xlsx"):
    print("="*60)
    print("CARGANDO DATOS")
    print("="*60)
    df_costos = pd.read_excel(archivo_costos)
    df_heuristicas = pd.read_excel(archivo_heuristicas)
    #grafo construido
    grafo = {}
    for _, row in df_costos.iterrows():
        origen = row['origen']
        destino = row['destino']
        costo = row['costo']
        
        if origen not in grafo:
            grafo[origen] = []
        grafo[origen].append((destino, costo))
    
    print(f"Grafo construido: {len(grafo)} nodos")

    heuristicas = {}
    for _, row in df_heuristicas.iterrows():
        estado = row['estado']
        heuristica = row['heuristica']
        heuristicas[estado] = heuristica
    
    print(f"Heuristicas cargadas: {len(heuristicas)}")
    
    return grafo, heuristicas

#ALGORITMOS
def uniform_cost_search(grafo, inicio, objetivo, heuristica=None):
    #prioridad = costo
    print(f"\n{"-"*60}")
    print("UNIFORM COST SEARCH")
    print(f"\n{"-"*60}")

    nodo_inicial = Node(inicio, costo=0)

    #inicio frontera con priority
    frontera = Priority()
    frontera.add(nodo_inicial, prioridad=0)

    #nodos explorados
    nodos_explorados = set()
    iteraciones = 1
    nodos_expandidos = 0

    print(f"\nIteracion: {iteraciones}")
    print(f"Frontera: {frontera}")
    print(f"Explorados: {nodos_explorados}")

    while not frontera.empty():
        #nodo con costo mas peuqeño
        nodo_actual = frontera.pop()
        nodos_expandidos += 1
        print(f"\nExpandido: {nodo_actual.estado}  (Costo: {nodo_actual.costo} ")
        #verificar
        if nodo_actual.estado == objetivo:
            print("Objetivo localizado")
            return reconstruir_camino(nodo_actual), nodo_actual.costo, nodos_expandidos
        
        nodos_explorados.add(nodo_actual.estado) #ya se exploro
        #expansion
        if nodo_actual.estado in grafo:
            for vecino, costo_accion in grafo[nodo_actual.estado]:
                if vecino not in nodos_explorados:
                    nuevo_costo = nodo_actual.costo + costo_accion
                    nuevo_nodo = Node(
                        estado=vecino,
                        padre=nodo_actual,
                        accion=f"{nodo_actual.estado}→{vecino}",
                        costo=nuevo_costo,
                        profundidad=nodo_actual.profundidad + 1
                    )
                    frontera.add(nuevo_nodo, prioridad=nuevo_costo)
        iteracion += 1
        if iteracion <= 5: #fragmento
            print(f"\nIteracion {iteracion}:")
            print(f"Frontera: {frontera}")
            print(f"Explorados: {nodos_explorados}")

    print("»»» No se encontró una solución")
    return None, None, nodos_expandidos

def a_star(grafo, inicio, objetivo, heuristica):
    #prioridad: f = g+h
    print(f"\n{"-"*60}")
    print("\t\tA*")
    print(f"\n{"-"*60}")
     # Obtener heurística inicial
    h_inicial = heuristica.get(inicio, 0)
    nodo_inicial = Node(inicio, costo=0, heuristica=h_inicial)
    
    #iniio frontera
    frontera = Priority()
    frontera.add(nodo_inicial)
    
    explorados = set()
    
    iteracion = 1
    nodos_expandidos = 0
    
    print(f"\nIteracion {iteracion}:")
    print(f"Frontera: {frontera}")
    print(f"Explorados: {explorados}")
    
    while not frontera.empty():
        #nodo menor, con f
        nodo_actual = frontera.pop()
        nodos_expandidos += 1
        
        print(f"\nExpandido: {nodo_actual.estado} (g={nodo_actual.costo}, h={nodo_actual.heuristica}, f={nodo_actual.f})")
        
        if nodo_actual.estado == objetivo:
            print(f"\nObjetivo localizado")
            return reconstruir_camino(nodo_actual), nodo_actual.costo, nodos_expandidos
        
        # Marcar como explorado
        explorados.add(nodo_actual.estado)
        
        # Expandir nodo
        if nodo_actual.estado in grafo:
            for vecino, costo_accion in grafo[nodo_actual.estado]:
                if vecino not in explorados:
                    nuevo_costo = nodo_actual.costo + costo_accion
                    h_vecino = heuristica.get(vecino, 0)
                    nuevo_nodo = Node(
                        estado=vecino,
                        padre=nodo_actual,
                        accion=f"{nodo_actual.estado}→{vecino}",
                        costo=nuevo_costo,
                        profundidad=nodo_actual.profundidad + 1,
                        heuristica=h_vecino
                    )
                    frontera.add(nuevo_nodo)  # Usa nodo f default
        
        iteracion += 1
        if iteracion <= 5:  # Mostrar solo primeras iteraciones
            print(f"\nIteracion {iteracion}:")
            print(f"Frontera: {frontera}")
            print(f"Explorados: {explorados}")
    
    print("»»» No se encontró solución")
    return None, None, nodos_expandidos

def reconstruir_camino(nodo):
    #Reconstruir camino de inicio a fin
    camino = []
    while nodo:
        camino.append(nodo.estado)
        nodo = nodo.padre
    camino.reverse()
    return camino


#GRAFICAS
def graficas(grafo, camino_ucs, costo_ucs, camino_astar, costo_astar):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    G = nx.DiGraph()
    for origen, destinos in grafo.items():
        for destino, costo in destinos:
            G.add_edge(origen, destino, weight=costo)
    
    pos = nx.spring_layout(G, k=2, seed=42)
    
    node_color_ucs = ['red' if node in camino_ucs else 'lightgray' for node in G.nodes()]
    node_color_astar = ['blue' if node in camino_astar else 'lightgray' for node in G.nodes()]
    
    #UNiform cost search
    ax1.set_title(f"Uniform Cost Search\nCosto: {costo_ucs} min | Nodos: {len(camino_ucs)}", 
                  fontsize=14, fontweight='bold')
    nx.draw(G, pos, with_labels=True, node_color=node_color_ucs, node_size=2000, font_size=8, font_weight='bold', 
            arrows=True, arrowstyle='->', ax=ax1)
    
    #A*
    ax2.set_title(f"A* Search\nCosto: {costo_astar} min | Nodos: {len(camino_astar)}", 
                  fontsize=14, fontweight='bold')
    nx.draw(G, pos, with_labels=True, node_color=node_color_astar, node_size=2000, font_size=8, font_weight='bold', 
            arrows=True, arrowstyle='->', ax=ax2)
    
    plt.tight_layout()
    plt.savefig('comparacion_busqueda.png', dpi=150, bbox_inches='tight')
    plt.show()