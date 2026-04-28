# Frankenstein Validation Report

## Contexte

- Roman de test : `Frankenstein`
- Chapitres Markdown : `24`
- Ingestion : `SQLite`
- Indexation semantique : `ChromaDB`
- Nombre de chunks indexes : `245`
- Etat des tests : `11 passed`

## Resultats

| Test | Requete | Resultat attendu | Resultat observe | Statut |
| --- | --- | --- | --- | --- |
| 1 | `the creature learns language` | chapitres `12-13` | chapitres `12-13` | `reussi` |
| 2 | `the creation of the being` | chapitres `3-4` | chapitres `3-4` | `reussi` |
| 3 | `Victor creates the monster` | chapitre `4` | resultats moins pertinents | `partiel` |
| 4 | `Victor animates the creature` | chapitre `4` | resultats moins pertinents | `partiel` |
| 5 | `the creature sees himself in water` | chapitre `12` | a completer plus tard | `a tester` |

## Observation

Les premiers tests montrent que la memoire semantique retrouve correctement certains passages importants du roman.

Les resultats dependent toutefois clairement de la formulation des requetes et du vocabulaire reellement present dans le texte source.

## Validation du Continuiste enrichi

La requete `where does the creature learn language?` retourne :

- des preuves textuelles pertinentes dans les chapitres `12-13`
- l'evenement structure pertinent : `The creature learns language`

Le croisement `ChromaDB + SQLite` et l'amelioration du ranking reduisent les evenements parasites qui remontaient auparavant sur des actions de Victor.
