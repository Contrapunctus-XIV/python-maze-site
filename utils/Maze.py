from __future__ import annotations
from collections import deque
from typing import Any, Generator
from PIL import Image, ImageDraw
from disjoint_set import DisjointSet
from io import BytesIO
from .HashXList import HashXList
from enum import Enum
import random, os, sys, base64

sys.setrecursionlimit(10**6)

DELTAS = [(-1,0), (0,-1), (1,0), (0,1)]

class Maze:
    """
    Classe représentant un labyrinthe parfait (carré et comprenant un unique chemin entre deux cases quelconques distinctes).

    Attributs :
    - h [int] -> height
    - w [int] -> width
    - walls [tuple[tuple[int]]] : set des murs représentés par des tuples contenant les coordonnées des cases adjacentes
    - generator [callable] : fonction de génération initialisée par Maze.set_generation_method
    
    Méthodes de base de la SDD Labyrinthe :
    - has_wall : renvoie True si le mur délimité par les deux coordonnées passées en entrée est actif, False sinon.
    - remove_wall : rend inactif (retire) le mur passé en entrée.
    - add_wall : rend actif (ajoute) le mur passé en entrée.
    - init_walls : rend actifs tous les murs du labyrinthe.
    
    Une case est représentée par un tuple constitué de deux coordonnées (tuple[int]) décrivant dans l'ordre son
    abscisse et son ordonnée, la case (0,0) étant celle située en haut à gauche.
    Les coordonnées sont toujours positives.
    
    On considère l'entrée du labyrinthe comme étant la case en haut à gauche et la fin comme étant celle en bas à droite.
    
    Un mur est représentant par un tuple des coordonnées des deux cases qui le délimitent soit : tuple[tuple[int]].
    
    Les bordures ne sont pas retirables.
    
    L'algorithme de génération est à spécifier grâce à la fonction Maze.set_generation_method.
    """
    def __init__(self, height: int, width: int):
        """
        Args:
            height (int): hauteur du labyrinthe (en nombre de cases)
            width (int): largeur du labyrinthe (en nombre de cases)
        """
        assert height > 1 and width > 1
        
        self.h = height
        self.w = width
        self.generator = None

    def init_walls(self) -> None:
        """
        Initialise l'attribut walls avec tous les murs possibles.
        """

        self.walls = set()

        for x in range(self.w):
            for y in range(self.h):
                if x < self.w - 1:
                    self.walls.add(((x,y), (x+1,y)))
                if y < self.h - 1:
                    self.walls.add(((x,y), (x,y+1)))

    def has_wall(self, wall: tuple[tuple[int]]) -> bool:
        """Renvoie l'état (ouvert ou fermé) du mur situé entre les deux cases passées en entrée.

        Args:
            wall (tuple[tuple[int]]): le mur

        Returns:
            bool: true si le mur est actif, false s'il est retiré.
        """

        (x0,y0), (x1,y1) = wall

        if x1 > x0 or y1 > y0: # réarrangement des coordonnées (la plus faible en premier) pour que cela corresponde au format de Maze.walls
            ref = ((x0,y0), (x1,y1))
        else:
            ref = ((x1,y1), (x0,y0))
            
        if y0 == y1 and (ref[0][0] < 0 or ref[1][0] >= self.w): # bordure verticale toujours présente
            return True

        if x0 == x1 and (ref[0][1] < 0 or ref[1][1] >= self.h): # bordure horizontale toujours présente
            return True

        return ref in self.walls

    def remove_wall(self, wall: tuple[tuple[int]]) -> None:
        """
        Retire le mur passé en entrée.

        Args:
            wall (tuple[tuple[int]]): le mur
        """
        (x0, y0), (x1, y1) = wall

        if x1 > x0 or y1 > y0: # réarrangement des coordonnées (la plus faible en premier) pour que cela corresponde au format de Maze.walls
            ref = ((x0, y0), (x1, y1))
        else:
            ref = ((x1, y1), (x0, y0))

        self.walls.remove(ref)
        
    def add_wall(self, wall: tuple[tuple[int]]) -> None:
        """
        Ajoute le mur passé en entrée.

        Args:
            wall (tuple[tuple[int]]): le mur
        """

        (x0, y0), (x1, y1) = wall

        if x1 > x0 or y1 > y0: # réarrangement des coordonnées (la plus faible en premier) pour que cela corresponde au format de Maze.walls
            ref = ((x0, y0), (x1, y1))
        else:
            ref = ((x1, y1), (x0, y0))

        self.walls.add(ref)
        
    def set_generation_method(self, method: GenerationMethods) -> None:
        """
        Permet de spécifier la méthode de génération souhaitée.
        
        Args:
            method (GenerationMethods): méthode de génération du labyrinthe
        """

        assert method in GenerationMethods, 'Please specify a valid generation method.'
        
        self.generator = method.value[2]
        
    def generate(self) -> None:
        """
        Appelle la méthode de génération spécifiée.
        """
        assert self.generator is not None, 'Please specify a generation method.'
        self.generator(self)

    def generate_kruskal(self) -> None:
        """
        Implémente l'algorithme de génération KRUSKAL basé sur les DisjointSet.
        """
        self.init_walls()
        walls = list(self.walls)
        random.shuffle(walls)

        disjoint_set = DisjointSet([(x,y) for y in range(self.h) for x in range(self.w)])
        # deux cases sont dans un même set s'il existe un chemin entre les deux
        # au départ, toutes les cases ont leur propre set

        while len(walls) > 0:
            coords0, coords1 = walls.pop() # choix d'un mur au hasard

            if not disjoint_set.connected(coords0, coords1): # si pas de chemin entre les deux cases délimitées par le mur
                self.remove_wall((coords0, coords1))
                disjoint_set.union(coords0, coords1) # retrait du mur et mise à jour du DisjointSet
                
    def generate_wilson(self) -> None:
        """
        Implémente l'algorithme de génération WILSON basé sur les marches aléatoires.
        """
        
        self.init_walls()
        start = (random.randrange(self.w), random.randrange(self.h)) # case de départ choisie aléatoirement
        visited = set([start]) # cases marquées comme inclus au labyrinthe
        visited_count = 1
        
        def find_and_remove(stack: deque, x: Any) -> bool:
            """
            Si x est un élément de la pile, renvoie True et retire récursivement tous les éléments
            entre le haut de la pile et cet élément (inclus).
            
            Args:
                stack (deque): la pile
                x (Any): l'élément à rechercher
            """
            if not stack:
                return False
            
            val = stack.pop()
            if val == x:
                stack.append(val)
                return True
            
            found = find_and_remove(stack, x)
            if not found:
                stack.append(val)

            return found
        
        while visited_count < self.w * self.h: # tant que toutes les cases ne sont pas marquées
            current = (random.randrange(self.w), random.randrange(self.h)) # case aléatoire
            path = deque([current]) # initialisation de la marche aléatoire
            while current not in visited: # tant qu'une case visitée n'est pas rencontrée
                x0,y0 = current
                neighbours = [
                    (x0-1,y0),
                    (x0+1,y0),
                    (x0,y0-1),
                    (x0,y0+1)
                ]
                
                x1,y1 = -1,-1
                
                while not (0 <= x1 < self.w and 0 <= y1 < self.h):
                    i = random.randrange(len(neighbours))
                    neighbours[i], neighbours[-1] = neighbours[-1], neighbours[i]
                    x1,y1 = neighbours.pop()
                    
                if find_and_remove(path, (x1,y1)): # si une boucle dans le chemin se crée
                    current = path.pop() # retour à la valeur d'où la boucle commence
                else:
                    current = (x1,y1)
                    
                path.append(current)
                                        
            while path: # à la fin de la marche, on enlève les murs parcourus et on marque les cases
                first = path.popleft()
                visited.add(first)
                if path:
                    second = path.popleft()
                    self.remove_wall((first, second))
                    path.appendleft(second)
                    visited_count += 1
                    
    def generate_prim(self) -> None:
        """
        Implémente l'algorithme de génération PRIM basé sur un set de murs entre les cases marquées et non marquées.
        """
        
        def get_walls(coords: tuple[int]) -> Generator[tuple[tuple[int]], None, None]:
            """
            Renvoie un générateur des murs dont la case est un délimitant.
            
            Args:
                coords (tuple[int]): case
            """
            x, y = coords

            for dx, dy in DELTAS:
                new_x, new_y = x + dx, y + dy
                if 0 <= new_x < self.w and 0 <= new_y < self.h:
                    yield (coords, (new_x, new_y))
                    
        self.init_walls()
        start = (random.randrange(self.w), random.randrange(self.h)) # coordonnées de départ aléatoires
        marked_count = 1 # nombre de cases marquées
        frontier = HashXList(get_walls(start)) # ensemble de murs délimités par une case marquée et une non marquée (dans cet ordre)

        while marked_count < self.h * self.w: # tant que toutes les cases ne sont pas marquées
            current, unmarked = frontier.random() # choix d'un mur de la frontière au hasard
                
            self.remove_wall((current, unmarked))
            
            for c0, c1 in get_walls(unmarked): # pour chaque mur adjacent à la case non marquée
                if (c1, c0) in frontier:
                    frontier.remove((c1, c0)) # retrait des murs délimités par deux cases marquées
                else:
                    frontier.add((c0, c1)) # ajout des murs dont l'autre case est non marquée
                    
            marked_count += 1
            
    def generate_backtracking(self) -> None:
        """
        Implémente l'algorithme de génération RECURSIVE BACKTRACKING basé sur la récursivité.
        """
                
        def get_unmarked_walls(coords: tuple[int]) -> list[tuple[tuple[int]]]:
            """
            Renvoie tous les murs non marqués dont la case est un délimitant.
            
            Args:
                coords (tuple[int]): la case
            """
            x, y = coords

            random.shuffle(DELTAS)
                    
            return [(coords, (new_x, new_y)) for dx, dy in DELTAS
                    if 0 <= (new_x := x + dx) < self.w
                    and 0 <= (new_y := y + dy) < self.h
                    and (new_x, new_y) not in marked]
                    
        self.init_walls()
        start = (random.randrange(self.w), random.randrange(self.h)) # case de départ choisie aléatoirement
        marked = set() # ensemble des cases marquées
        
        def carve(current: tuple[int]) -> None:
            """
            Marque la case et pour chaque mur non marqué le retire et s'applique récursivement sur la nouvelle case.
            
            Args:
                current (tuple[int]): la case
            """
            
            marked.add(current)
            walls = get_unmarked_walls(current)
            
            while len(walls) > 0:
                _, destination = walls[0]
                self.remove_wall((current, destination))
                carve(destination)
                walls = get_unmarked_walls(current)
                
        carve(start)

    def generate_hunt_and_kill(self) -> None:
        """
        Implémente l'algorithme de génération HUNT & KILL basé sur les marches aléatoires.
        """
        self.init_walls()
        
        def get_random_unmarked_wall(coords: tuple[int]) -> tuple[int] | None:
            """
            Renvoie un mur aléatoire parmi tous les murs de la case dont l'autre case adjacente
            n'est pas marquée, None s'il n'y en a pas.
            
            Args:
                coords (tuple[int]): la case
            """
            x, y = coords
            random.shuffle(DELTAS)

            for dx, dy in DELTAS:
                new_x, new_y = x + dx, y + dy
                if 0 <= new_x < self.w and 0 <= new_y < self.h and (new_x, new_y) not in marked:
                    return (coords, (new_x, new_y))
                
            return None

        start = (random.randrange(self.w), random.randrange(self.h)) # case de départ choisie aléatoirement
        marked = set([start])
        marked_count = 1

        def kill(start: tuple[int]) -> None:
            """
            Effectue une marche aléatoire partant de la case passée en entrée et passant par des cases non-marquées,
            jusqu'à ce que toutes les cases adjacentes soient marquées.
            
            Args:
                start (tuple[int]): la case
            """
            nonlocal marked_count

            unmarked_wall = get_random_unmarked_wall(start)
            while unmarked_wall is not None: # tant qu'il y a un mur menant à une case non marquée
                self.remove_wall(unmarked_wall) # retrait du mur et marquage de la nouvelle case
                marked.add(unmarked_wall[1])
                marked_count += 1
                unmarked_wall = get_random_unmarked_wall(unmarked_wall[1])
                
        def hunt() -> None:
            """
            Pour chaque case marquée adjacente à une case non marquée, applique la fonction
            kill sur cette dernière.
            """
            nonlocal marked_count

            for y in range(self.h):
                for x in range(self.w):
                    unmarked_wall = get_random_unmarked_wall((x,y))
                    if (x,y) in marked and unmarked_wall is not None:
                        self.remove_wall(unmarked_wall)
                        marked.add(unmarked_wall[1]) # marquage de la nouvelle case
                        marked_count += 1

                        kill(unmarked_wall[1])

        while marked_count < self.h * self.w: # tant que toutes les cases ne sont pas marquées
            hunt()
            
    def generate_aldous_broder(self) -> None:
        """
        Implémente l'algorithme de génération ALDOUS-BRODER basé sur les marches aléatoires.
        """
        
        def select_random_neighbor(cell: tuple[int]) -> tuple[int]:
            """
            Renvoie une case prise aléatoirement parmi les voisines de la case passée en entrée.
            
            Args:
                cell (tuple[int]): la case
            """
            x,y = cell
            random.shuffle(DELTAS)
            
            for dx, dy in DELTAS:
                new_x, new_y = x + dx, y + dy
                if 0 <= new_x < self.w and 0 <= new_y < self.h:
                    return (new_x, new_y)
            
        self.init_walls()
        current = (random.randrange(self.w), random.randrange(self.h)) # case de départ choisie au hasard
        marked = set([current])
        marked_count = 1
        
        while marked_count < self.h * self.w: # tant qu'il y a des cases non marquées
            neighbor = select_random_neighbor(current) # choix d'une case voisine
            if neighbor not in marked:
                self.remove_wall((current, neighbor))
                marked.add(neighbor)
                marked_count += 1 # marquage de la case et retrait du mur si non marquée
            current = neighbor # la case voisine devient la case actuelle
            
    def generate_sidewinder(self) -> None:
        """
        Implémente l'algorithme de génération SIDEWINDER basé sur le choix équiprobable de détruire un mur.
        """
        self.init_walls()

        for y in range(self.h):
            run_start = 0 # coordonnée x de la première case du groupe de cases actuel dont tous les
            # murs verticaux (situé sur la même ligne) ont été retirés
            for x in range(self.w):
                if y > 0 and (x + 1 == self.w or random.random() < 0.5):
                    # choix d'une case au hasard parmi la dernière run de cases dont tous les murs verticaux
                    # ont été retirés 
                    random_cell = (run_start + random.randrange(x - run_start + 1), y) # retrait d'un mur Nord
                    self.remove_wall(((random_cell[0], random_cell[1] - 1), random_cell))
                    run_start = x + 1 # début d'une nouvelle run
                elif x < self.w - 1: # destruction du mur à l'est
                    self.remove_wall(((x, y), (x+1, y)))
                    
    def generate_growing_tree(self):
        """
        Implémente l'algorithme de génération GROWING TREE basé sur une liste de cases dont au moins
        un voisin n'est pas visité.
        """
        self.init_walls()

        start = (random.randrange(self.w), random.randrange(self.h)) # case de départ choisie aléatoirement
        marked = set([start])
        selected = HashXList([start])
        
        def get_unmarked_neighbour(coords: tuple[int]):
            """
            Renvoie une case prise aléatoirement parmi les voisines non marquées de la case passée en entrée.
            
            Args:
                coords (tuple[int]): la case
            """
            x,y = coords
            random.shuffle(DELTAS)
            
            for dx, dy in DELTAS:
                new_x, new_y = x + dx, y + dy
                if 0 <= new_x < self.w and 0 <= new_y < self.h and (new_x, new_y) not in marked:
                    return (new_x, new_y)
                
            return None
        
        while len(selected) > 0:
            current = selected[0] # choix d'une case au début de la liste
            unmarked_neighour = get_unmarked_neighbour(current)
            
            if unmarked_neighour is not None: # s'il y a un voisin non marqué
                self.remove_wall((current, unmarked_neighour))
                marked.add(unmarked_neighour)
                selected.add(unmarked_neighour) # retrait du mur et ajout de la nouvelle case à la liste
            else:
                selected.remove(current) # sinon retrait de la case de la liste
                
    def generate_division(self) -> None:
        """
        Implémente l'algorithme de génération RECURSIVE DIVISION basé sur la division du labyrinthe en régions
        et l'ajout progressif de murs.
        """
        self.walls = set() # au départ il n'y a aucun mur
        
        def divide(coords0: tuple[int], coords1: tuple[int]) -> None:
            """
            Divise la région définie par les deux coordonnées en deux, ajoute des murs partout à la frontière
            des deux sauf en un endroit, et s'applique récursivement sur les deux sous-régions.
            
            Args:
                coords0 (tuple[int]): la case en haut à gauche
                coords1 (tuple[int]): la case en bas à droite
            """
            x0,y0 = coords0
            x1,y1 = coords1
            
            if x0 == x1 or y0 == y1: # si la région ne contient plus aucune case (épaisseur ou largeur de 0)
                return
            
            if random.random() < 0.5: # la région est divisée verticalement
                middle_x = (x0+x1)//2
                walls = [((middle_x, y), (middle_x + 1, y)) for y in range(y0, y1+1)] # murs à ajouter
                areas = [((x0, y0), (middle_x, y1)),
                         ((middle_x + 1, y0), (x1, y1))] # deux nouvelles sous-régions
            else: # la région est divisée horizontalement
                middle_y = (y0+y1)//2
                walls = [((x, middle_y), (x, middle_y + 1)) for x in range(x0, x1+1)]
                areas = [((x0, y0), (x1, middle_y)),
                         ((x0, middle_y + 1), (x1,y1))]
                
            # ajout de tous les murs sauf un
            for wall in walls:
                self.add_wall(wall)
                
            random_connection = random.choice(walls)
            self.remove_wall(random_connection)
            
            for area in areas:
                divide(*area) # appel récursif sur les sous-régions
                
        divide((0,0), (self.w - 1, self.h - 1))
    
    def generate_binary_tree(self) -> None:
        """
        Implémente l'algorithme de génération BINARY TREE basé sur le choix équiprobable de détruire un mur.
        """
        self.init_walls()

        coords = [(x,y) for x in range(self.w) for y in range(self.h)] # toutes les cases
        for x, y in coords:
            if random.random() < 0.5:
                # destruction du mur Est si pas une bordure, sinon destruction du mur Sud si pas non plus une bordure
                if x < self.w - 1:
                    self.remove_wall(((x, y), (x+1, y)))
                elif y < self.h - 1:
                    self.remove_wall(((x, y), (x, y+1)))
            else:
                # destruction du mur Sud si pas une bordure, sinon destruction du mur Est si pas non plus une bordure
                if y < self.h - 1:
                    self.remove_wall(((x, y), (x, y+1)))
                elif x < self.w - 1:
                    self.remove_wall(((x, y), (x+1, y)))

    def generate_eller(self) -> None:
        """
        Implémente l'algorithme de génération ELLER basé sur le parcours par ligne avec
        DisjointSet.
        """
        self.init_walls()
        # associe à chaque set les cases qui y sont connectées
        flags = { (x, 0): [(x, 0)] for x in range(self.w) }
        # associe à chaque case son set
        cells_set = DisjointSet({ (x,0): (x,0) for x in range(self.w)})

        def get_border_walls(y: int) -> list[tuple[int]]:
            """
            Renvoie une liste des murs séparant deux cases connectées à des sets différents sur la ligne y.
            
            Args:
                y (int): ordonnée de la ligne
            """
            border = [((x,y), (x+1,y)) for x in range(self.w - 1) if not cells_set.connected((x,y), (x+1,y))]

            return border
        
        def connect(random_walls: list[tuple[tuple[int]]]) -> None:
            """
            Pour chaque mur spécifié, détruit le mur et fusionne les deux sets jusqu'alors séparés en un seul.
            
            Args:
                random_walls (list[tuple[tuple[int]]]): la liste des murs obtenus avec get_border_walls 
            """
            for wall in random_walls:
                (x1,y1), (x2, y2) = wall # coordonnées des deux cases du mur
                f1 = cells_set.find((x1,y1)) # ensemble des deux 
                f2 = cells_set.find((x2,y2))
                
                if f1 == f2: # il peut arriver que durant le parcours les cases finissent par être prématurément reliées par conséquence d'une connexion précédente
                    continue
                
                self.remove_wall(wall)
                cells_set.union((x1,y1), (x2, y2)) # union des sets
                for c in flags[f1]:
                    flags[f2].append(c) # mise à jour des cases connectées au set
                
                del flags[f1] # set à supprimer

        def vertical_connections() -> None:
            """
            Connecte au moins une case de chaque set à une case de la ligne suivante en détruisant
            les murs verticaux correspondants.
            """
            for _, cells in flags.items(): # pour chaque set
                k = max(1, random.randrange(len(cells))) # nombre de connexions verticales à faire (au moins 1)
                v_cells = random.sample(cells, k=k)
                for x, y in v_cells:
                    self.remove_wall(((x,y), (x,y+1)))
                    
        def update(y: int) -> tuple[DisjointSet, dict]:
            """
            Renvoie les nouvelles versions des variables cells_set et flags respectivement à la ligne y avant
            de passer à la ligne suivante.
            
            Args:
                y (int): l'ordonnée de la ligne qui vient d'être traitée
            """
            new_set = {}
            new_flags = {}
            for x in range(self.w): # pour chaque case de la ligne
                if self.has_wall(((x,y), (x,y+1))): # si mur formé avec la case du bas
                    new_set[(x, y+1)] = (x, y+1) # les sets sont différents
                    new_flags[(x, y+1)] = [(x, y+1)]
                else:
                    root = cells_set.find((x, y))
                    new_set[(x,y+1)] = root # sinon on attache la case du bas au set de la case du haut
                    new_flags[root] = new_flags.get(root, []) + [(x,y+1)]
                    
            return DisjointSet(new_set), new_flags
            
        for y in range(self.h - 1): # ligne par ligne
            border = get_border_walls(y) # les frontières entre sets
            k = max(1, random.randrange(len(border)))
            random_borders = random.sample(border, k) # choix d'un nombre aléatoire (supérieur à 1) de frontières à détruire
            connect(random_borders)
            vertical_connections() # retrait aléatoire des murs entre la ligne actuelle et la ligne du dessous
            cells_set, flags = update(y) # mise à jour des variables
            
        y = self.h - 1 # pour la dernière ligne on connecte toutes les cases siuées dans des sets différents
        
        for x in range(self.w - 1):
            if not cells_set.connected((x, y), (x+1, y)):
                connect([((x,y), (x+1, y))])
                                
    def __str__(self):
        wall_char = '█'
        _str = '\033[31m' # affichage en rouge
        max_width = min(self.w, os.get_terminal_size().columns//2 - 1) # largeur maximale que peut prendre le labyrinthe dans la console
        # si la taille du labyrinthe excède cette largeur, il sera tronqué pour éviter les problèmes d'affichage

        for y in range(self.h):
            # pour chaque ligne
            for x in range(max_width): # pour chaque case de la ligne
                # ajout du caractère bloc s'il y a un mur Nord, sinon espace
                north = self.has_wall(((x,y-1), (x,y)))
                _str += wall_char*2 if north else f"{wall_char} "

            _str += f"{wall_char}\n" # fin de la ligne et retour chariot

            for x in range(max_width): # pour chaque case de la ligne
                # ajout du caractère bloc s'il y a un mur Ouest, sinon espace
                west = self.has_wall(((x-1,y), (x,y)))
                _str += f"{wall_char} " if west else f"  "
                if x == max_width - 1:
                    # ajout de la bordure Est
                    _str += wall_char

            _str += f"\n"

        # ajout de la bordure Sud
        _str += wall_char*(2*max_width + 1)
        _str += "\033[39m"

        return _str

class MazeDrawer:
    """
    Classe gérant le dessin d'un labyrinthe parfait avec sa résolution sur un caneva grâce à la librairie pillow.
    """
    def __init__(self, maze: Maze):
        assert maze.h == maze.w, "Cannot draw a non-square maze."

        self.maze = maze
        self.cell_size = 0
        self.wall_width = 0

    def _draw_base_maze(self) -> Image:
        """"
        Dessine la représentation graphique de base du labyrinthe.
        
        Args:
            maze (Maze): le labyrinthe
            
        Returns:
            Image: le caneva pillow
        """        
        # calcul de la taille initiale du caneva
        picture_width = self.maze.w * self.cell_size + self.wall_width * (self.maze.w + 1) + 1
        picture_height = self.maze.h * self.cell_size + self.wall_width * (self.maze.h + 1) + 1
                
        img = Image.new('RGB', size=(picture_width, picture_height), color="white")
        drawer = ImageDraw.Draw(img)
            
        def draw_wall(coords0: tuple[int], coords1: tuple[int]) -> None:
            """
            Dessine sur le caneva le mur déterminé par les coordonnées passées en entrée.
            
            Args:
                coords0 (tuple[int]): première case
                coords1 (tuple[int]): deuxième case, voisine de la première
            """
            x0,y0 = coords0
            x1,y1 = coords1
                
            if abs(y0-y1) == 1: # dessin d'un mur horizontal
                start_rect = (self.wall_width*x0 + self.cell_size*x0, self.cell_size*(y0+1) + self.wall_width*(y0+1))
                end_rect = (start_rect[0] + self.wall_width*2 + self.cell_size, start_rect[1] + self.wall_width)
                drawer.rectangle((start_rect, end_rect), fill="black")
            else: # dessin d'un mur vertical
                start_rect = (self.wall_width*(x0+1) + self.cell_size*(x0+1), self.cell_size*y0 + self.wall_width*y0)
                end_rect = (start_rect[0] + self.wall_width, start_rect[1] + self.wall_width*2 + self.cell_size)
                drawer.rectangle((start_rect, end_rect), fill="black")
                
        # liste des murs de bordures
        borders: list[list[tuple[tuple[int]]]] = [
            [((-1, y), (0, y)) for y in range(self.maze.h)],
            [((x, -1), (x, 0)) for x in range(self.maze.w)],
            [((x, self.maze.h - 1), (x, self.maze.h)) for x in range(self.maze.w)],
            [((self.maze.w - 1, y), (self.maze.w, y)) for y in range(self.maze.h)]
        ]
        
        # dessin des bordures sur le caneva
        for group in borders:
            for border in group:
                draw_wall(*border)
        
        # dessin des murs du labyrinthe
        for wall in self.maze.walls:
            draw_wall(*wall)
                
        return img

    def _extend_img(self, img: Image, final_size: int) -> Image:
        """
        Redimensionne le caneva passé en entrée avec un algorithme adapté.
        
        Args:
            img (Image): le caneva
            final_size (int): la taille souhaitée en pixel
        
        Returns:
            Image: le caneva redimensionné
        """
        current_size, _ = img.size
        
        assert final_size >= current_size, "Cannot shrink a picture. Final image size must be greater than original image size."

        if current_size < final_size*0.54:
            ext = img.resize((final_size, final_size), resample=Image.Resampling.BOX)
        else:
            ext = img.resize((final_size, final_size), resample=Image.Resampling.BILINEAR)

        return ext

    def _solve_maze_img(self, img: Image, path_color: str = "red") -> None:
        """
        Résout le labyrinthe et dessine la solution sur le caneva passé en entrée.
        
        Args:
            img (Image): le caneva
            path_color (str): la couleur du chemin solution 
        """
        visited = set() # cases déjà visitée par l'algorithme de depth-first search
        start = (0,0)
        end = (self.maze.h - 1, self.maze.w - 1)
        
        drawer = ImageDraw.Draw(img)
        
        def draw_cell(cell: tuple[int]) -> None:
            """
            Dessine sur le caneva la case passée en entrée.
            
            Args:
                cell (tuple[int]): la case 
            """
            x,y = cell
            start_rect = (self.cell_size*x + self.wall_width*(x+1)+1, self.cell_size*y + self.wall_width*(y+1)+1)
            end_rect = (start_rect[0] + self.cell_size - 2, start_rect[1] + self.cell_size - 2)
            
            drawer.rectangle((start_rect, end_rect), fill=path_color)
            
        def draw_wall(coords0: tuple[int], coords1: tuple[int]) -> None:
            """
            Dessine sur le caneva le mur déterminé par les coordonnées passées en entrée.
            
            Args:
                coords0 (tuple[int]): première case
                coords1 (tuple[int]): deuxième case, voisine de la première
            """
            s = sorted([coords0, coords1])
            x0,y0 = s[0]
            x1,y1 = s[1]
                
            if abs(y0-y1) == 1: # mur horizontal
                start_rect = (self.wall_width*(x0+1) + self.cell_size*x0+1, self.cell_size*(y0+1) + self.wall_width*(y0+1))
                end_rect = (start_rect[0] + self.cell_size - 2, start_rect[1] + self.wall_width)
                drawer.rectangle((start_rect, end_rect), fill=path_color)
            else:
                start_rect = (self.wall_width*(x0+1) + self.cell_size*(x0+1), self.cell_size*y0 + self.wall_width*(y0+1)+1)
                end_rect = (start_rect[0] + self.wall_width, start_rect[1] + self.cell_size - 2)
                drawer.rectangle((start_rect, end_rect), fill=path_color)
                
        def get_neighbours(coords: tuple[int]) -> Generator[tuple[int], None, None]:
            """
            Renvoie un générateur avec les cases voisines accessibles de la case passée en entrée.
            
            Args:
                coords (tuple[int]): la case
            """
            x,y = coords
            
            for dx, dy in DELTAS:
                new_x, new_y = x + dx, y + dy
                if 0 <= new_x < self.maze.w and 0 <= new_y < self.maze.h and (new_x, new_y) not in visited and not self.maze.has_wall(((x, y), (new_x, new_y))):
                    yield (new_x, new_y)
        
        def walk(coords: tuple[int]) -> bool:
            """
            Résout le labyrinthe et dessine les cases du chemin solution sur le caneva en se déplaçant sur chaque
            case accessible de la case passée en entrée.
            
            Args:
                coords (tuple[int]): la case
                
            Returns:
                bool: True si la case passée en entrée fait partie du chemin solution, False sinon.
            """
            visited.add(coords)

            if coords == end: # si le point d'arrivée est atteint
                draw_cell(coords)
                return True
            
            for neighbour in get_neighbours(coords): # pour chaque voisin pas séparé de la case actuelle par un mur
                result = walk(neighbour) # appel récursif sur le voisin
                if result: # la case fait partie du chemin solution donc on la dessine
                    draw_cell(coords)
                    draw_wall(coords, neighbour)
                    return True
            
            return False
        
        walk(start)
        
    def _img_to_base64(self, img: Image) -> str:
        """
        Renvoie chaîne de caractères base64 représentant le caneva passé en entrée.
        
        Args:
            img (Image): le caneva
        """
        buffered = BytesIO()

        img.save(buffered, format="PNG")
        buffered.seek(0)
        img_byte = buffered.getvalue()
        img_str = "data:image/png;base64," + base64.b64encode(img_byte).decode()
        
        return img_str
        
    def draw(self, cell_size: int = 8, wall_width: int = 2, picture_size: int = 500, path_color: str = "red") -> tuple[str]:
        """
        Renvoie les canevas en base64 représentant graphiquement le labyrinthe, respectivement sans et avec la solution.
        
        Args:
            cell_size (int): côté des cases en pixel
            wall_width (int): largeur des murs en pixel
            picture_size (int): taille des canevas en pixel
            path_color (str): couleur du chemin solution
        """
        
        self.cell_size = cell_size
        self.wall_width = wall_width
        img = self._draw_base_maze() # caneva de base
        img_sol = img.copy() # caneva qui recueille le chemin solution
        self._solve_maze_img(img_sol, path_color)
        
        # extension des deux canevas
        img_ext = self._extend_img(img, picture_size)
        img_sol_ext = self._extend_img(img_sol, picture_size)
        
        # renvoi des base64
        return self._img_to_base64(img_ext), self._img_to_base64(img_sol_ext)

class GenerationMethods(Enum):
    """
    Classe associant à chaque algorithme un tuple contenant dans l'ordre :
    - le nom utilisé dans les formulaires ;
    - le nom utilisé pour le désigner sur le site ;
    - la méthode correspondante de la classe Maze.
    """
    ALDOUS_BRODER = ("aldous-broder", "Aldous-Broder", Maze.generate_aldous_broder)
    BINARY_TREE = ("binary-tree", "Binary tree", Maze.generate_binary_tree)
    ELLER = ("eller", "Eller", Maze.generate_eller)
    GROWING_TREE = ("growing-tree", "Growing tree", Maze.generate_growing_tree)
    HUNT_AND_KILL = ("hunt-kill", "Hunt & Kill", Maze.generate_hunt_and_kill)
    KRUSKAL = ("kruskal", "Kruskal", Maze.generate_kruskal)
    PRIM = ("prim", "Prim", Maze.generate_prim)
    BACKTRACKING = ("backtracking", "Recursive backtracking", Maze.generate_backtracking)
    DIVISION = ("division", "Recursive division", Maze.generate_division)
    SIDEWINDER = ("sidewinder", "Sidewinder", Maze.generate_sidewinder)
    WILSON = ("wilson", "Wilson", Maze.generate_wilson)