import heapq
import networkx as nx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import deque

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
        return f"{self.estado}(c={self.cost},h={self.heuristica})"
    
    def __repr__(self):
        return self.__str__()


# ============================================
# ESTRUCTURAS DE COLA
# ============================================
class FIFO:
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
        return f"FIFO[{', '.join(str(item) for item in self.items)}]"
    
    def __len__(self):
        return len(self.items)


class LIFO:
    def __init__(self):
        self.items = []
        self.name = 'LIFO'
    
    def empty(self):
        return len(self.items) == 0 
    
    def top(self):
        if not self.empty():
            return self.items[-1]
        return None
    
    def pop(self):
        if not self.empty():
            return self.items.pop()
        return None
    
    def add(self, elemento):
        self.items.append(elemento)
        return self.items
    
    def __str__(self):
        return f"LIFO[{', '.join(str(item) for item in self.items)}]"
    
    def __len__(self):
        return len(self.items)


class Priority:
    def __init__(self):
        self.items = []
        self.contador = 0
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
        if prioridad is None:
            if hasattr(elemento, 'f'):
                prioridad = elemento.f
            else:
                prioridad = elemento.cost
        
        heapq.heappush(self.items, (prioridad, self.contador, elemento))
        self.contador += 1
        return [(p, c, e.estado) for p, c, e in self.items]
    
    def __str__(self):
        elementos = [(p, e.estado) for p, c, e in self.items]
        return f"PRIORITY{elementos}"   
    
    def __len__(self):
        return len(self.items)


# ============================================
# CARGA DE DATOS DESDE EXCEL
# ============================================
def datos_excel(archivo_costos='funcion_de_costo.xlsx', archivo_heuristicas='heuristica.xlsx'):
    print("="*60)
    print("CARGANDO DATOS")
    print("="*60)
    
    df_costos = pd.read_excel(archivo_costos)
    print(f"Costos Columnas: {list(df_costos.columns)}")
    
    df_heuristicas = pd.read_excel(archivo_heuristicas)
    print(f"Heuristicas Columnas: {list(df_heuristicas.columns)}")

    # Construir grafo
    grafo = {}
    for _, row in df_costos.iterrows():
        origen = row['Origen']
        destino = row['Destino']
        costo = row['Cost']
        
        if origen not in grafo:
            grafo[origen] = []
        grafo[origen].append((destino, costo))
    
    print(f"Grafo construido con {len(grafo)} nodos")
    print("\nConexiones encontradas:")
    for origen in sorted(grafo.keys()):
        print(f"\t{origen} → {grafo[origen]}")

    # Cargar heurísticas
    heuristicas = {}
    for _, row in df_heuristicas.iterrows():
        actividad = row['Activity']
        heuristica = row['Recovery time after burning 300cal (minutes)']
        heuristicas[actividad] = heuristica
    
    print(f"Heuristicas cargadas: {len(heuristicas)}")
    print("\nValores heuristicos (tiempo recuperacion):")
    for actividad in sorted(heuristicas.keys()):
        print(f"   {actividad}: {heuristicas[actividad]}")

    # Agregar Stretching si no existe
    if 'Stretching' not in grafo:
        grafo['Stretching'] = []
        print("\n✓ Nodo 'Stretching' agregado")
    
    # Conectar ejercicios finales a Stretching
    ejercicios_finales = []
    for ejercicio in grafo.keys():
        if ejercicio != 'Stretching' and len(grafo[ejercicio]) == 0:
            ejercicios_finales.append(ejercicio)
    
    if ejercicios_finales:
        print(f"\nConectando ejercicios finales a Stretching:")
        for ejercicio in ejercicios_finales:
            costo_estimado = heuristicas.get(ejercicio, 10)
            grafo[ejercicio].append(('Stretching', costo_estimado))
            print(f"      {ejercicio} → Stretching (costo={costo_estimado})")
    
    return grafo, heuristicas


def reconstruir_camino(nodo):
    camino = []
    while nodo:
        camino.append(nodo.estado)
        nodo = nodo.parent
    camino.reverse()
    return camino


#BREADHT
def breadth_first_search(grafo, inicio, objetivo, heuristicas=None):
    print(f"\n{'-'*60}")
    print("1. BREADTH FIRST SEARCH (BFS)")
    print(f"{'-'*60}")
    
    nodo_inicial = Node(inicio, cost=0)
    frontera = FIFO()
    frontera.add(nodo_inicial)
    explorados = set()
    iteracion = 1
    nodos_expandidos = 0

    print(f"\nIteración {iteracion}:")
    print(f"Frontera: {frontera}")
    print(f"Explorados: {explorados}")

    while not frontera.empty():
        nodo_actual = frontera.pop()
        nodos_expandidos += 1
        
        print(f"\n► Expandido: {nodo_actual.estado} (profundidad={nodo_actual.depth})")
        
        if nodo_actual.estado == objetivo:
            print(f"\n✅ OBJETIVO ENCONTRADO!")
            camino = reconstruir_camino(nodo_actual)
            return camino, nodo_actual.cost, nodos_expandidos, "BFS"
        
        explorados.add(nodo_actual.estado)
        
        if nodo_actual.estado in grafo:
            for vecino, costo_accion in grafo[nodo_actual.estado]:
                if vecino not in explorados:
                    nuevo_nodo = Node(
                        estado=vecino,
                        parent=nodo_actual,
                        action=f"{nodo_actual.estado}→{vecino}",
                        cost=nodo_actual.cost + costo_accion,
                        depth=nodo_actual.depth + 1
                    )
                    frontera.add(nuevo_nodo)
        
        iteracion += 1
        if iteracion <= 5:
            print(f"\nIteración {iteracion}:")
            print(f"Frontera: {frontera}")
            print(f"Explorados: {explorados}")

    print(">>> No se encontró solución")
    return None, None, nodos_expandidos, "BFS"


# DEPTH
def depth_first_search(grafo, inicio, objetivo, heuristicas=None):
    print(f"\n{'-'*60}")
    print("DEPTH FIRST SEARCH")
    print(f"{'-'*60}")
    
    nodo_inicial = Node(inicio, cost=0)
    frontera = LIFO()
    frontera.add(nodo_inicial)

    explorados = set()
    iteracion = 1
    nodos_expandidos = 0

    print(f"\nIteracion {iteracion}:")
    print(f"Frontera: {frontera}")
    print(f"Explorados: {explorados}")

    while not frontera.empty():
        nodo_actual = frontera.pop()
        nodos_expandidos += 1
        
        print(f"\nExpandido: {nodo_actual.estado} (profundidad={nodo_actual.depth})")
        
        if nodo_actual.estado == objetivo:
            print(f"\nObjetivo localizado")
            camino = reconstruir_camino(nodo_actual)
            return camino, nodo_actual.cost, nodos_expandidos, "DFS"
        
        explorados.add(nodo_actual.estado)
        
        if nodo_actual.estado in grafo:
            for vecino, costo_accion in reversed(grafo[nodo_actual.estado]):
                if vecino not in explorados:
                    nuevo_nodo = Node(
                        estado=vecino,
                        parent=nodo_actual,
                        action=f"{nodo_actual.estado}→{vecino}",
                        cost=nodo_actual.cost + costo_accion,
                        depth=nodo_actual.depth + 1
                    )
                    frontera.add(nuevo_nodo)
        
        iteracion += 1
        if iteracion <= 5:
            print(f"\nIteracion {iteracion}:")
            print(f"Frontera: {frontera}")
            print(f"Explorados: {explorados}")

    print(">>> No se encontro solución")
    return None, None, nodos_expandidos, "DFS"


#UNIFORM SEARCH
def uniform_cost_search(grafo, inicio, objetivo, heuristicas=None):
    print(f"\n{'-'*60}")
    print("UNIFORM COST SEARCH (UCS)")
    print(f"{'-'*60}")

    nodo_inicial = Node(inicio, cost=0)
    frontera = Priority()
    frontera.add(nodo_inicial, prioridad=0)

    explorados = set()
    iteracion = 1
    nodos_expandidos = 0

    print(f"\nIteracion {iteracion}:")
    print(f"Frontera: {frontera}")
    print(f"Explorados: {explorados}")

    while not frontera.empty():
        nodo_actual = frontera.pop()
        nodos_expandidos += 1
        
        print(f"\nExpandido: {nodo_actual.estado} (costo={nodo_actual.cost})")
        
        if nodo_actual.estado == objetivo:
            print(f"\nObjetivo localizado")
            camino = reconstruir_camino(nodo_actual)
            return camino, nodo_actual.cost, nodos_expandidos, "UCS"
        
        explorados.add(nodo_actual.estado)
        
        if nodo_actual.estado in grafo:
            for vecino, costo_accion in grafo[nodo_actual.estado]:
                if vecino not in explorados:
                    nuevo_costo = nodo_actual.cost + costo_accion
                    nuevo_nodo = Node(
                        estado=vecino,
                        parent=nodo_actual,
                        action=f"{nodo_actual.estado}→{vecino}",
                        cost=nuevo_costo,
                        depth=nodo_actual.depth + 1
                    )
                    frontera.add(nuevo_nodo, prioridad=nuevo_costo)
        
        iteracion += 1
        if iteracion <= 5:
            print(f"\nIteracion {iteracion}:")
            print(f"Frontera: {frontera}")
            print(f"Explorados: {explorados}")

    print(">>> No se encontro solucion")
    return None, None, nodos_expandidos, "UCS"


#GREDDY
def greedy_best_first_search(grafo, inicio, objetivo, heuristicas):
    print(f"\n{'-'*60}")
    print("GREEDY BEST-FIRST SEARCH")
    print(f"{'-'*60}")
    
    h_inicial = heuristicas.get(inicio, 0)
    nodo_inicial = Node(inicio, cost=0, heuristica=h_inicial)
    
    frontera = Priority()
    frontera.add(nodo_inicial, prioridad=h_inicial)
    
    explorados = set()
    iteracion = 1
    nodos_expandidos = 0
    
    print(f"\nIteracion {iteracion}:")
    print(f"Frontera: {frontera}")
    print(f"Explorados: {explorados}")
    
    while not frontera.empty():
        nodo_actual = frontera.pop()
        nodos_expandidos += 1
        
        print(f"\n► Expandido: {nodo_actual.estado} (h={nodo_actual.heuristica})")
        
        if nodo_actual.estado == objetivo:
            print(f"\nObjetivo localizado")
            camino = reconstruir_camino(nodo_actual)
            return camino, nodo_actual.cost, nodos_expandidos, "Greedy"
        
        explorados.add(nodo_actual.estado)
        
        if nodo_actual.estado in grafo:
            for vecino, costo_accion in grafo[nodo_actual.estado]:
                if vecino not in explorados:
                    h_vecino = heuristicas.get(vecino, 0)
                    nuevo_nodo = Node(
                        estado=vecino,
                        parent=nodo_actual,
                        action=f"{nodo_actual.estado}→{vecino}",
                        cost=nodo_actual.cost + costo_accion,
                        depth=nodo_actual.depth + 1,
                        heuristica=h_vecino
                    )
                    frontera.add(nuevo_nodo, prioridad=h_vecino)
        
        iteracion += 1
        if iteracion <= 5:
            print(f"\nIteración {iteracion}:")
            print(f"Frontera: {frontera}")
            print(f"Explorados: {explorados}")
    
    print(">>> No se encontro solución")
    return None, None, nodos_expandidos, "Greedy"


#A*
def a_star_search(grafo, inicio, objetivo, heuristicas):
    print(f"\n{'-'*60}")
    print("5. A* SEARCH")
    print(f"{'-'*60}")
    
    h_inicial = heuristicas.get(inicio, 0)
    nodo_inicial = Node(inicio, cost=0, heuristica=h_inicial)
    
    frontera = Priority()
    frontera.add(nodo_inicial)
    
    explorados = set()
    iteracion = 1
    nodos_expandidos = 0
    
    print(f"\nIteración {iteracion}:")
    print(f"Frontera: {frontera}")
    print(f"Explorados: {explorados}")
    
    while not frontera.empty():
        nodo_actual = frontera.pop()
        nodos_expandidos += 1
        
        print(f"\nExpandido: {nodo_actual.estado} (g={nodo_actual.cost}, h={nodo_actual.heuristica}, f={nodo_actual.f})")
        
        if nodo_actual.estado == objetivo:
            print(f"\nObjetivo localizado")
            camino = reconstruir_camino(nodo_actual)
            return camino, nodo_actual.cost, nodos_expandidos, "A*"
        
        explorados.add(nodo_actual.estado)
        
        if nodo_actual.estado in grafo:
            for vecino, costo_accion in grafo[nodo_actual.estado]:
                if vecino not in explorados:
                    nuevo_costo = nodo_actual.cost + costo_accion
                    h_vecino = heuristicas.get(vecino, 0)
                    nuevo_nodo = Node(
                        estado=vecino,
                        parent=nodo_actual,
                        action=f"{nodo_actual.estado}→{vecino}",
                        cost=nuevo_costo,
                        depth=nodo_actual.depth + 1,
                        heuristica=h_vecino
                    )
                    frontera.add(nuevo_nodo)
        
        iteracion += 1
        if iteracion <= 5:
            print(f"\nIteracion {iteracion}:")
            print(f"Frontera: {frontera}")
            print(f"Explorados: {explorados}")
    
    print(">>> No se encontró solución")
    return None, None, nodos_expandidos, "A*"


#GRAFICAS
def grafica_grafo(grafo, heuristicas):
    plt.figure(figsize=(16, 10))
    
    G = nx.DiGraph()
    for origen, destinos in grafo.items():
        for destino, costo in destinos:
            G.add_edge(origen, destino, weight=costo)
    
    pos = nx.spring_layout(G, k=3, iterations=100, seed=42, scale=2)
    
    node_colors = []
    for node in G.nodes():
        if node == 'Stretching':
            node_colors.append('lightgreen')
        elif 'Warm-up' in node:
            node_colors.append('lightblue')
        else:
            node_colors.append('lightgray')
    
    nx.draw(G, pos, with_labels=True, node_color=node_colors, node_size=2500, font_size=9, font_weight='bold',
            arrows=True, arrowstyle='->', arrowsize=15)
    
    edge_labels = {(u, v): f"{d['weight']}" for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=8)
    nx.draw_networkx_edges(G, pos, edge_color='gray', arrows=True, arrowstyle='->', arrowsize=20, width=1.5, alpha=0.7,
                          connectionstyle='arc3,rad=0.1')
    
    plt.title("Grafo Completo de Ejercicios", fontsize=14, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('grafo_completo.png', dpi=150, bbox_inches='tight')
    plt.show()


def comparacion_algoritmos(grafo, resultados):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    colores = {
        'BFS': 'blue',
        'DFS': 'green',
        'UCS': 'orange',
        'Greedy': 'purple',
        'A*': 'red'
    }
    
    G = nx.DiGraph()
    for origen, destinos in grafo.items():
        for destino, costo in destinos:
            G.add_edge(origen, destino, weight=costo)
    
    pos = nx.spring_layout(G, k=2, seed=42)
    
    for idx, (nombre, res) in enumerate(resultados.items()):
        if idx < 5 and res['camino']:
            ax = axes[idx]
            
            node_colors = []
            for node in G.nodes():
                if node in res['camino']:
                    node_colors.append(colores[nombre])
                elif node == 'Stretching':
                    node_colors.append('lightgreen')
                else:
                    node_colors.append('lightgray')
            
            nx.draw(G, pos, with_labels=True, node_color=node_colors, 
                   node_size=2000, font_size=8, font_weight='bold',
                   arrows=True, arrowstyle='->', ax=ax)
            
            ax.set_title(f"{nombre}\nCosto: {res['costo']} min | Nodos: {res['nodos']}",
                        fontsize=12, fontweight='bold')
            ax.axis('off')
    
    axes[5].axis('off')
    plt.suptitle("COMPARACION ALGORITMOS", fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('comparacion_algoritmos.png', dpi=150, bbox_inches='tight')
    plt.show()


def tabla_comparacion(resultados):
    print("\n" + "="*90)
    print("TABLA COMPARACION")
    print("="*90)
    print(f"{'Algoritmo':<10} {'Tipo':<12} {'Costo':<8} {'Nodos':<8} {'Camino encontrado'}")
    print("-"*90)
    
    tipos = {
        'BFS': 'No informada',
        'DFS': 'No informada',
        'UCS': 'No informada',
        'Greedy': 'Informada',
        'A*': 'Informada'
    }
    
    for nombre, res in resultados.items():
        if res['camino']:
            camino_str = ' → '.join(res['camino'])
            print(f"{nombre:<10} {tipos[nombre]:<12} {res['costo']:<8} {res['nodos']:<8} {camino_str}")
        else:
            print(f"{nombre:<10} {tipos[nombre]:<12} {'N/A':<8} {res['nodos']:<8} {'No encontrado'}")

def main():
    print("\n" + "="*90)
    print("ALGORITMOS BUSQUEDA")
    print("="*90)
    
    archivo_costos = 'funcion_de_costo.xlsx'
    archivo_heuristicas = 'heuristica.xlsx'
    
    print(f"\nArchivo de costos: {archivo_costos}")
    print(f"Archivo de heurísticas: {archivo_heuristicas}")
    
    try:
        grafo, heuristicas = datos_excel(archivo_costos, archivo_heuristicas)
    except FileNotFoundError as e:
        print(f"\nError: No se encontro archivo - {e}")
        return
    except Exception as e:
        print(f"\nError inesperado: {e}")
        return
    
    #inicio y objetivo
    inicio = 'Warm-up activities'
    objetivo = 'Stretching'
    
    print(f"\nBuscando rutina: {inicio} → {objetivo}")
    
    if inicio not in grafo:
        print(f"\nError: El nodo inicial '{inicio}' no está en el grafo")
        return
    
    resultados = {}
    
    #BFS
    camino, costo, nodos, nombre = breadth_first_search(grafo, inicio, objetivo)
    resultados[nombre] = {'camino': camino, 'costo': costo, 'nodos': nodos}
    
    #DFS
    camino, costo, nodos, nombre = depth_first_search(grafo, inicio, objetivo)
    resultados[nombre] = {'camino': camino, 'costo': costo, 'nodos': nodos}
    
    #UCS
    camino, costo, nodos, nombre = uniform_cost_search(grafo, inicio, objetivo)
    resultados[nombre] = {'camino': camino, 'costo': costo, 'nodos': nodos}
    
    #Greedy
    camino, costo, nodos, nombre = greedy_best_first_search(grafo, inicio, objetivo, heuristicas)
    resultados[nombre] = {'camino': camino, 'costo': costo, 'nodos': nodos}
    
    #A*
    camino, costo, nodos, nombre = a_star_search(grafo, inicio, objetivo, heuristicas)
    resultados[nombre] = {'camino': camino, 'costo': costo, 'nodos': nodos}
    
    tabla_comparacion(resultados)
    
    if all(r['camino'] for r in resultados.values() if r['camino'] is not None):
        grafica_grafo(grafo, heuristicas)
        comparacion_algoritmos(grafo, resultados)
        print("Graficas guardadas: grafo_completo.png, comparacion_algoritmos.png")
    

if __name__ == "__main__":
    main()