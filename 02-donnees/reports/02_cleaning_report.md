# Rapport de nettoyage - modele ST4000DM000

- Lignes en entree : 60806887
- Lignes en sortie : 60805623 (1264 retirees, 0.0%)
- Disques en entree : 36160
- Disques en sortie : 36063
- Pannes conservees : 3970

## Colonnes SMART constantes retirees (43)
Critere : ecart-type ~ 0 (algorithme de Welford, en ligne).

- smart_13_raw
- smart_15_raw
- smart_160_raw
- smart_161_raw
- smart_163_raw
- smart_164_raw
- smart_165_raw
- smart_166_raw
- smart_167_raw
- smart_168_raw
- smart_169_raw
- smart_171_raw
- smart_172_raw
- smart_175_raw
- smart_176_raw
- smart_178_raw
- smart_179_raw
- smart_180_raw
- smart_181_raw
- smart_182_raw
- smart_18_raw
- smart_201_raw
- smart_202_raw
- smart_206_raw
- smart_210_raw
- smart_218_raw
- smart_224_raw
- smart_22_raw
- smart_230_raw
- smart_231_raw
- smart_234_raw
- smart_23_raw
- smart_244_raw
- smart_245_raw
- smart_246_raw
- smart_247_raw
- smart_248_raw
- smart_24_raw
- smart_250_raw
- smart_251_raw
- smart_252_raw
- smart_254_raw
- smart_255_raw

## Colonnes SMART trop peu remplies retirees (20)
Critere : plus de 50% de valeurs manquantes sur l'ensemble du dataset isole (attributs propres a certaines generations de disques / firmware).

- smart_11_raw
- smart_16_raw
- smart_170_raw
- smart_173_raw
- smart_174_raw
- smart_177_raw
- smart_17_raw
- smart_195_raw
- smart_196_raw
- smart_200_raw
- smart_220_raw
- smart_222_raw
- smart_223_raw
- smart_225_raw
- smart_226_raw
- smart_232_raw
- smart_233_raw
- smart_235_raw
- smart_2_raw
- smart_8_raw

## Colonnes SMART conservees (24)

- smart_10_raw
- smart_12_raw
- smart_183_raw
- smart_184_raw
- smart_187_raw
- smart_188_raw
- smart_189_raw
- smart_190_raw
- smart_191_raw
- smart_192_raw
- smart_193_raw
- smart_194_raw
- smart_197_raw
- smart_198_raw
- smart_199_raw
- smart_1_raw
- smart_240_raw
- smart_241_raw
- smart_242_raw
- smart_3_raw
- smart_4_raw
- smart_5_raw
- smart_7_raw
- smart_9_raw

## Doublons
- 0 ligne(s) dupliquee(s) (meme disque + meme date) supprimee(s)

## Valeurs manquantes restantes
Trous ponctuels combles par disque (ffill/bfill, jamais entre deux disques). Une ligne n'est retiree que si TOUTES les colonnes SMART gardees sont manquantes : 0 ligne(s) supprimee(s) dans ce cas.
- Valeurs encore manquantes (NaN partiel, legitime) dans le fichier final : 0 -> a gerer explicitement lors du pretraitement si necessaire pour le modele.

## Piege rencontre : format de date non-standard
Certaines lignes du dataset Backblaze source utilisent un format de date different (ex. '2/25/18' au lieu de '2018-02-25'). Parse avec pandas format='mixed' pour gerer les deux formats sans perte de lignes.

## Disques a historique trop court
- Seuil : moins de 30 jours d'historique
- 97 disque(s) ecarte(s), soit 1264 ligne(s)

## Coherence de la colonne failure
- 44 disque(s) avec une anomalie -> a inspecter manuellement
- Disques concernes : ['Z300XGNY', 'Z302AYKZ', 'Z302BDTD', 'Z302BMWF', 'Z302G5CC', 'Z302SZKW', 'Z304EZL5', 'S300Z5B5', 'Z304K6DT', 'Z305D6P8', 'Z303Y10M', 'Z302AYQQ', 'Z303PFVQ', 'Z304VZDD', 'W300STQ5', 'Z303VKE9', 'Z3040PZK', 'Z300GZ0W', 'W3015JSX', 'Z304JGL2'] ...
