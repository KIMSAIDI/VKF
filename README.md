# Projet dans le cadre du M2 ANDROIDE

Ce projet s'inscrit dans le cadre de l'UE **IAR** du Master 2 **ANDROIDE** à La Sorbonne. Son objectif principal est de reproduire les simulations du modèle **VKF** (Volatile Kalman Filter) et d'explorer les limites d'un algorithme d'apprentissage appliqué à la modélisation de l'expérience humaine.

## Contexte scientifique

Le travail repose sur l'article suivant, qui constitue la base de ce projet :  
[**A simple model for learning in volatile environments**](https://journals.plos.org/ploscompbiol/article/file?id=10.1371/journal.pcbi.1007963&type=printable)  

Nous avons également travaillé sur l'article écrit par George Velentzas, Costats Tzafestas et Mehdi Khamassi : 
[**Bridging Computational Neuroscience and Machine Learning on Non-Stationary Multi-Armed Bandits**](https://dumas.ccsd.cnrs.fr/ISIR_AMAC/hal-03775008v1)

## Encadrement

Ce projet est encadré par **Mehdi Khamassi**, chercheur spécialisé en robotique et sciences cognitives.

## Structure du projet

Dans le fichier **simulation_article** on retrouve les fichiers suivants :
- **vkf_lin.m** : Contient le script pour reproduire la simulation du VKF dans un environnement linéaire.
- **vkf_bin.m** : Contient le script pour reproduire la simulation du VKF dans un environnement binaire.
- **vol_higher_vkf_lin.m** :  Contient le script pour reproduire la simulation du VKF avec une volatilité élevé dans un environnement linéaire.
- **vol_higher_vkf_bin.m** :  Contient le script pour reproduire la simulation du VKF avec une volatilité élevé dans un environnement binaire.
- **benchmark_lin.m** :  Contient le script pour reproduire la simulation du VKF ainsi qu'un benchmark pour comparer les deux dans un environnement linéaire.
- **benchmark_bin.m** :  Contient le script pour reproduire la simulation du VKF ainsi qu'un benchmark pour comparer les deux dans un environnement binaire.
- **README.md** : Document expliquant le projet et son contexte.


Dans le fichier **simulation_multi_armed_bandits** on retrouve les fichiers suivants :
- **bandits.m** : Contient le code du multi armed bandit, traduction du code en .m de l'algorithme proposé par l'article **Bridging Computational Neuroscience and Machine Learning on Non-Stationary Multi-Armed Bandits**
- **simulate.m** : contient le code pour la simulation du code

Dans le fichier **VKF_with_bandits** on retrouve les fichiers suivants :
- **Main21** : Contient le code du multi armed bandit proposé par : https://github.com/GeoVelentzas/intellisys-2017-mab-mlbkf avec une adaptation de la volatilité pour que celle-ci soit adaptative
- **simulation** : Contient le code pour tester le code précédent
- **simulation_tmp** : Comme le fichier **simulation** mais avec le calcul des temps d’exécution, l’utilisation des CPU, ainsi que la mémoire requise
