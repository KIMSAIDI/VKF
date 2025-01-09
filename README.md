# Projet dans le cadre du M2 ANDROIDE

Ce projet s'inscrit dans le cadre de l'UE **IAR** du Master 2 **ANDROIDE** à La Sorbonne. Son objectif principal est de reproduire les simulations du modèle **VKF** (Volatile Kalman Filter) et d'explorer les limites d'un algorithme d'apprentissage appliqué à la modélisation de l'expérience humaine.

## Contexte scientifique

Le travail repose sur l'article suivant, qui constitue la base de ce projet :  
[**A simple model for learning in volatile environments**](https://journals.plos.org/ploscompbiol/article/file?id=10.1371/journal.pcbi.1007963&type=printable)  

## Encadrement

Ce projet est encadré par **Mehdi Khamassi**, chercheur spécialisé en robotique et sciences cognitives.

## Structure du projet

- **vkf_lin/** : Contient le script pour reproduire la simulation du VKF dans un environnement linéaire.
- **vkf_bin/** : Contient le script pour reproduire la simulation du VKF dans un environnement binaire.
- **vol_higher_vkf_lin/** :  Contient le script pour reproduire la simulation du VKF avec une volatilité élevé dans un environnement linéaire.
- **vol_higher_vkf_bin/** :  Contient le script pour reproduire la simulation du VKF avec une volatilité élevé dans un environnement binaire.
- **benchmark_lin/** :  Contient le script pour reproduire la simulation du VKF ainsi qu'un benchmark pour comparer les deux dans un environnement linéaire.
- **benchmark_bin/** :  Contient le script pour reproduire la simulation du VKF ainsi qu'un benchmark pour comparer les deux dans un environnement binaire.
- **README.md** : Document expliquant le projet et son contexte.
