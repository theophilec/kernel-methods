## Methodology
1) Calcul des noyaux pour chacun des trois datasets 
2) Cross-validation avec $K=5$ splits pour trouver les meilleurs paramètres
3) Indiquer les résultats obtenus dans le Google Sheets
4) Calculer les noyaux de test  
5) Soumettre 

Utiliser la _magic_ random seed
## Kernels
### Features
* Linear kernel
* Gaussian kernel : Influence de $\sigma$

### Kernels for strings 
* Weighted Degree (rapide), 1 paramètre
* Substring kernel, 2 paramètres
Fitter le premier paramètre sur le Weighted Degree kernel puis le réutiliser pour le substring kernel

## Algos
* KRR
* C-SVM : (TODO : Reparer l'erreur dataset 2)
* (TODO): Logistic regression

## Data processing
TODO: Center data $(I-\frac{1}{n}\mathbb{1}\mathbb{1}^T)$