# HelloWorld python dans un notebook

Ce répertoire permet de montrer deux façon de faire pour executer du code python ayant besoin de dépendances dans un jupyterlab.

La méthode 1 est d'embarquer les dépendancecs python dans le notebook via des `pip install` explicites. Cf le fichier `helloworld.ipynb` pour exemple.

La méthode 2 est d'ajouter un requirements.txt qui liste toutes les dépendances du notebook ou du script python et de faire un `pip install -r requirements.txt` avant d'exécuter son script python ou son notebook. Cf les fichiers `helloworld.py` et `requirements.txt` pour exemple.