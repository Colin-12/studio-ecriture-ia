# Protocole de validation memoire - Phase 1

## Objet

Ce document decrit un protocole de test pour valider la memoire locale de la Phase 1 sur un roman existant deja ingere.

La validation porte sur trois couches :

- `SQLite` pour les donnees structurees
- `ChromaDB` pour la recherche semantique sur les chapitres chunkes
- `NetworkX` pour les relations explicites ajoutees manuellement

Ce protocole ne suppose ni agent de generation, ni extraction automatique, ni LLM. Les reponses attendues doivent etre produites a partir de la memoire disponible, puis verifiees manuellement contre le roman source.

## Preconditions

Avant d'executer ces tests :

1. Le roman source doit etre present dans `manuscript/source_novel/`.
2. Les chapitres doivent avoir ete ingeres dans `SQLite`.
3. Les chapitres doivent avoir ete indexes dans `ChromaDB`.
4. Le graphe `NetworkX` doit avoir ete alimente manuellement si le test depend de relations explicites.
5. Les noms de personnages, lieux, scenes et chapitres doivent etre definis a partir du roman reel teste.

## Regle generale d'evaluation

Pour chaque question :

- lancer la recuperation depuis la ou les memoires indiquees ;
- comparer le resultat au texte source du roman ;
- noter si la memoire retrouve les bons elements, dans le bon ordre, avec le bon perimetre temporel ;
- documenter les faux positifs, les absences et les ambiguities.

## Question 1

**Question**

`Decrivez l'apparence de [personnage] telle qu'etablie au chapitre [N].`

**Objectif du test**

Verifier que la memoire retrouve les passages descriptifs pertinents sur un personnage a un point donne du roman, sans melanger des informations plus tardives.

**Memoire utilisee**

- combinaison `ChromaDB + SQLite`

**Critere de reussite**

- la recherche renvoie un ou plusieurs chunks provenant du bon chapitre ou des chapitres anterieurs pertinents ;
- les passages recuperes contiennent bien des elements d'apparence observables dans le texte source ;
- aucune information clairement introduite apres le chapitre cible n'est utilisee comme si elle etait deja connue.

**Limites actuelles**

- `ChromaDB` retrouve des passages semantiquement proches, mais ne produit pas de synthese ;
- `SQLite` ne stocke pas encore des attributs de personnage extraits structurellement ;
- sans couche de raisonnement temporel, la verification du perimetre "a la fin du chapitre N" reste manuelle.

## Question 2

**Question**

`Quels personnages sont presents dans [scene] ?`

**Objectif du test**

Verifier si la memoire permet d'identifier les personnages lies a une scene donnee.

**Memoire utilisee**

- combinaison `ChromaDB + NetworkX`

**Critere de reussite**

- si la scene a ete representee manuellement dans le graphe, les relations personnage-evenement ou personnage-lieu permettent de lister les personnages presents ;
- a defaut, la recherche semantique retrouve les chunks les plus proches de la scene et permet une verification manuelle dans le texte source ;
- les personnages identifies correspondent bien a la scene cible et non a une scene voisine.

**Limites actuelles**

- aucune extraction automatique de personnages par scene n'existe encore ;
- la notion de "scene" n'est pas modelisee explicitement dans `SQLite` ;
- le resultat depend fortement de la qualite de l'alimentation manuelle du graphe.

## Question 3

**Question**

`Que sait [personnage A] sur [personnage B] a la fin du chapitre [N] ?`

**Objectif du test**

Evaluer la capacite de la memoire a soutenir une question de coherence epistemique avec contrainte temporelle.

**Memoire utilisee**

- combinaison `ChromaDB + SQLite + NetworkX`

**Critere de reussite**

- les passages recuperes concernent bien les interactions, revelations ou observations anterieures ou egales au chapitre cible ;
- le graphe peut aider a tracer des relations explicites si elles ont ete saisies manuellement ;
- l'evaluateur humain peut reconstituer, a partir des elements retrouves, un etat de connaissance coherent de `A` sur `B`.

**Limites actuelles**

- aucune memoire d'etat mental des personnages n'est stockee explicitement ;
- aucune logique n'infere automatiquement ce qu'un personnage sait ou ignore ;
- ce test sert surtout a mesurer si la memoire fournit les bonnes preuves textuelles, pas une reponse finale automatisable a ce stade.

## Question 4

**Question**

`Citez trois descriptions de [lieu] dans l'ordre chronologique.`

**Objectif du test**

Verifier que la memoire peut retrouver plusieurs occurrences d'un lieu et les remettre dans l'ordre narratif.

**Memoire utilisee**

- combinaison `SQLite + ChromaDB`

**Critere de reussite**

- la recherche retrouve plusieurs passages associes au lieu cible ;
- les passages peuvent etre rattaches a leurs chapitres via `SQLite` ;
- l'ordre chronologique des citations est reconstitue correctement a partir des numeros de chapitre.

**Limites actuelles**

- `Location` existe dans `SQLite`, mais n'est pas encore alimente automatiquement depuis les chapitres ;
- les occurrences d'un lieu dans `ChromaDB` reposent sur la qualite de la requete et des chunks ;
- la selection de "trois descriptions" reste un travail manuel tant qu'aucune extraction de citations n'est implementee.

## Question 5

**Question**

`Quels setups plantes avant le chapitre [N] n'ont pas encore paye ?`

**Objectif du test**

Verifier la capacite de la memoire a suivre une logique setup/payoff inachevee avant un point donne du roman.

**Memoire utilisee**

- combinaison `SQLite + NetworkX`

**Critere de reussite**

- les objets `SetupPayoff` saisis dans `SQLite` permettent d'identifier les setups connus ;
- les chapitres de setup et de payoff, s'ils existent, permettent de filtrer ce qui est encore ouvert avant `N` ;
- le graphe peut servir a visualiser ou expliciter les dependances si des relations ont ete ajoutees manuellement.

**Limites actuelles**

- aucun setup/payoff n'est extrait automatiquement depuis le texte ;
- la qualite du test depend entierement de la saisie manuelle des setups/payoffs ;
- sans outillage supplementaire, la detection des setups "encore ouverts" reste une verification semi-manuelle.

## Sortie attendue du protocole

Pour chaque roman teste, produire un compte rendu court avec :

- la question posee ;
- les elements recuperes par la memoire ;
- le verdict `reussi`, `partiel` ou `echec` ;
- la cause principale en cas d'echec ;
- les limites observees a corriger dans une phase ulterieure.

## Usage du protocole

Ce protocole sert a valider que la Phase 1 fournit une base memoire exploitable avant d'ajouter des agents. Il ne valide pas encore la generation de reponses finales automatiques, mais la capacite de retrouver les bonnes preuves textuelles et structurelles.
