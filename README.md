# Python Maze Site
### Créé par un élève de Terminale NSI

Un site Web pour la génération et la visualisation de labyrinthes, basé sur **Python** et **FastHTML**.
Il est hébergé sur [Vercel](https://vercel.com) : https://python-maze.vercel.app.

## Lancement

Installez au préalable les dépendances requises :
```
pip install -r requirements.txt
```

Pour lancer le serveur il suffit d'exécuter le fichier `main.py`.

## Fonctionnalités

### Algorithmique

Le site comprend les algorithmes de génération suivants :
* *Aldous-Broder* (marches aléatoires)
* *Binary Tree* (parcours par ligne avec choix équiprobable de détruire un mur)
* *Eller* (parcours par ligne avec structure *union-find*)
* *Growing Tree* (liste de cases qui possèdent au moins un voisin non visité)
* *Hunt & Kill* (marches aléatoires)
* *Kruskal* (structure *union-find*)
* *Prim* (liste de murs entre une case marquée et une case non-marquée)
* *Recursive Backtracking* (creusage récursif d'un passage d'une case à une autre)
* *Recursive Division* (division récursive du labyrinthe en régions et ajout de murs)
* *Wilson* (marches aléatoires)

### Rendu sous forme d'image

Le *backend* dessine sur un caneva le labyrinthe généré et envoie au *frontend* sous forme de base64.

### Affichage de la solution

Le *backend* détermine la solution du labyrinthe et la trace sur une seconde image que le client peut afficher.

## À venir

L'implémentation d'une métrique pour évaluer la qualité (*i.e.* la difficulté de résolution) du labyrinthe généré.