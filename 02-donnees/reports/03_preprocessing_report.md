# Rapport de pretraitement - modele ST4000DM000

- Lignes traitees : 60805623
- Plafond de RUL : 125 jours
- Lignes au plafond (RUL=125, disque loin d'une panne ou jamais observe en panne = donnee censuree) : 60295996 (99.2%)

## Split train / validation / test (par disque, hash stable du serial_number)

| Split | Disques | Lignes |
|---|---|---|
| train | 25223 | 42562865 |
| val | 5421 | 9105568 |
| test | 5419 | 9137190 |

Split fait par disque entier (jamais ligne par ligne) pour eviter toute fuite de donnees : aucun disque n'apparait dans deux splits differents.

## Normalisation
Min-max par colonne SMART, parametres (min/max) calcules **uniquement sur le split train**, puis appliques tels quels a val et test (clip a [0,1] si une valeur de val/test depasse la plage vue en train). Parametres sauvegardes dans `normalization_params_ST4000DM000.json`.

## Limite : RUL censuree
Pour un disque jamais observe en panne dans la periode disponible (2016-2022), la vraie RUL est inconnue : on ne sait pas s'il tombera en panne demain ou dans 10 ans apres la fin de la periode observee. Le choix ici est de lui assigner RUL = plafond partout (traite comme 'loin d'une panne'), une simplification a mentionner explicitement dans les limites du modele (L4).

## Fenetrage : non materialise ici
Generer a l'avance toutes les fenetres glissantes possibles (30 jours, stride 1) sur ce volume produirait un fichier d'une taille ingerable (dizaines de milliards de valeurs). La fonction `make_windows()` du squelette LSTM (cahier des charges, C.3) doit etre appelee au moment de l'entrainement, a partir de ce fichier, pour construire les fenetres en memoire (par lot ou par disque).
