# PROJ601_CMI

## Warning

This project isn't runnable at the moment.  
If you are looking for or a working version, sorry but go on your way.

Also note the rest of this readme is a report for a course, and will be in french.

## Compte-rendu

L'objectif de ce projet était donc d'implémenter la méthode d'amplification de détails de surface 3D décrite dans  
Yohann Béarzi, Julie Digne, Raphaëlle Chaine. Wavejets: A Local Frequency Framework for Shape Details Amplification. Computer Graphics Forum, 2018, 37 (2), pp.13-24. 10.1111/cgf.13338. hal-01722993

Après une phase de lecture et compréhension du sujet, une première version a été implémentée.  
Celle-ci tentait de calculer les wavejets à partir des dérivés. Cependant, il devenait nécéssaire d'interpoler au préalable un grand nombre de points inexistants autour du point voulu. La première grande optimisation fût donc de remplacer cela par la résolution d'équation à l'aide d'une recherche de plan tangeant et le passage des points environnants dans l'espace formé par le-dit plan et un vecteur orthogonal.

Rapidement un second problème s'est posé, à savoir la recherche des plus proches voisins. En effet, la structure de données WaveFrontOBJ proposée ne permet pas de faire des recherches efficacement. Une structure de graphe a donc été ajoutée, optimisée pour un parcours basé sur le BFS (Breadth First Search, ou parcours en largeur).

Différents problèmes de plus petite taille ont pu être observés, dont les points n'appartenant pas à des triangles, ou seulement à des triangles avec leurs doublons, dont résulte le code suivant, présent dans `WavejetComputation.compute_heightmap`, dans `wavejets.py`

```py
if norm == 0: # Yes, it does happen
    return np.array([]), np.array([0., 0., 0.]), 0.
```

On été implémenté le calcul du plan tangeant à partir des vecteur normaux des faces voisines, le calcul des courbures moyenne et gaussienne, la recherche des k-voisins, la résolution d'équation afin d'obtenir les phis, ainsi que l'algorithme d'amélioration de détail.

Une fonction pour retrouver les phis définits par le plan tangeant en fonctions de ceux d'un autre plan à également été commencée, mais n'est pas utilisable en l'état. De nombreuses autres fonctions utilitaires ont été ajoutées, et beaucoup d'entre elles ne dépendant pas des attributs de notre objet elles ont également une forme `cls_` qui permet de les utiliser directement par la classe, sans créer d'instance.

Malheureusement il a été compliqué de vérifier la justesse des fonctions au fur et à mesure du projet dû au sujet, et il ne semble pas donner des résultats corrects à l'heure actuelle.

Il a été testé d'augmenter les échantillons (quitte à perdre en précision locale) mais cela n'a rien changé.

Divers améliorations auraient pu être ajoutées, comme la possibilité de donner les paramètres dans la ligne de commande plutôt que de les modifier en dur dans le code, un affichage de l'avancement plus espacé, ou encore certaines fonctions de calcul qui ne peuvent traiter qu'un seul point auraient pu être adaptées pour inclure une boucle.

Il est prévu que le fonctionnement final puisse être vu en lançant `examples.py`. Ce projet a été réalisé sous Python 3.8.10. Toutes les dépendances requises pour le projet sont dans `pip_requirements.txt` et peuvent être installées avec la commande `pip install -r pip_requirements.txt`.