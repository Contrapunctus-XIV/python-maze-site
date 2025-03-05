from typing import Iterable, Hashable
import random

class HashXList:
    """
    Classe utilisant les tables de hashage et les listes Python pour créer une structure de données
    O(1) en lecture, insertion, suppression, test de contenance et supportant l'aléatoire.
    
    Contrainte : les valeurs contenues doivent être hashables.
    """
    def __init__(self, iterable: Iterable[Hashable] = []):
        self.a = [] # liste des éléments
        self.m = {} # table de hashage associant chaque élément à son index
        
        i = 0
        for val in iterable:
            self.a.append(val)
            self.m[val] = i
            i += 1
        
    def __contains__(self, val):
        return val in self.m
    
    def __len__(self):
        return len(self.m)
    
    def __getitem__(self, i):
        return self.a[i]
        
    def add(self, val: Hashable) -> None:
        if val not in self:
            self.m[val] = len(self.a)
            self.a.append(val)
            
    def remove(self, val: Hashable) -> None:
        if val in self:
            i = self.m[val]
            self.a[i], self.a[-1] = self.a[-1], self.a[i]
            self.m[self.a[i]] = i
            self.a.pop()
            del self.m[val]
            
    def random(self) -> Hashable:
        return random.choice(self.a)