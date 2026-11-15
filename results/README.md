# Résultats

Tous les chiffres de ce dossier sont lus dans des fichiers produits pendant l'entraînement ; aucun n'est recalculé à la main.

## Métriques

| Fichier | Source |
|---|---|
| [metrics_global.csv](metrics_global.csv) | [run/results.csv](run/results.csv) : meilleure époque en mAP50-95 (110) et dernière époque enregistrée (124) |
| [metrics_par_classe.csv](metrics_par_classe.csv) | sortie de `model.val()` sur `best.pt`, notebook [06](../notebooks/06_split_entrainement.ipynb) (dernière cellule) |

`run/results.csv` ne couvre que les époques **60 à 124** : l'entraînement a été repris depuis une sauvegarde (`resume=True`) et les époques 1 à 59 ne figurent pas dans ce fichier. Il s'est arrêté à l'époque 124 sur 150 prévues, avant que l'arrêt anticipé (`patience=25`) ne se déclenche.

## Figures

| Figure | Contenu | Source |
|---|---|---|
| `courbes_entrainement.png` | mAP, précision, rappel et pertes par époque | `run/results.csv` |
| `ap_par_classe.png` | mAP50-95 par classe | `metrics_par_classe.csv` |
| `labels.jpg` | distribution des classes et des boîtes dans le jeu d'entraînement | tracé par Ultralytics au lancement |
| `predictions_xview122_*.jpg` | prédictions de `best.pt` (SAHI, tuiles 640, seuil 0,25), une classe par image | notebook 07 ; image xView 122, absente de l'entraînement (voir ci-dessous) |
| `kakhovka_*.jpg` | barrage de Kakhovka avant / après sa destruction (6 juin 2023) : inventaire, alerte par comptage (v1, v2), analyse des transformations (v2) | notebook 08 |
| `demo_intrusion_xview1578.jpg`, `demo_chronologie_dota_P2760.jpg` | scénarios d'intrusion et de chronologie T0 / T1 / T2 | notebook 08 ; séquences **construites pour la démonstration** à partir d'une seule image |
| `exploration_xview_*.jpg` | exemples d'images xView, image la plus annotée (1831.tif, 7 618 boîtes) | notebook 01 |
| `v1_colab_detection_changement.jpg` | première version de la détection de changement (modèle Colab à 29 classes) | `archive/colab_v1/` |
| `run/train_batch*.jpg` | mosaïques d'entraînement avec les vérités terrain | Ultralytics |

**Images xView 122 et 1578.** Elles ne figurent ni dans `train` ni dans `val` de [configs/split_sources.json](../configs/split_sources.json) : ce sont des images du lot `val_images` de xView, dont les annotations ne sont pas publiques. Ce sont donc des images jamais vues pendant l'entraînement, mais aucune métrique ne peut y être calculée.

**Images de Kakhovka.** Captures Google Earth (filigrane © Google conservé), utilisées uniquement à titre de démonstration.

**Prédictions sur le jeu de validation annoté.** Les mosaïques `val_batch*_pred.jpg` d'Ultralytics n'ont pas été récupérées depuis Kaggle ; elles ne sont pas dans ce dépôt.
