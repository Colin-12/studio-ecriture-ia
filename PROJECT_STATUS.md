# Project Status

## Etat actuel

Le projet `Studio d'ecriture IA par agents` est en `Phase 1 MVP`.
L'objectif courant est la validation d'une memoire locale exploitable avant l'ajout d'agents de generation.

Le depot est deja pousse sur GitHub.

## Ce qui fonctionne deja

- stockage structure en `SQLite` via `SQLAlchemy`
- ingestion de chapitres Markdown, un fichier par chapitre
- indexation semantique des chapitres dans `ChromaDB`
- graphe de coherence simple avec `NetworkX`
- interface `CLI` minimale pour afficher le roman, lister les chapitres et lancer une recherche semantique
- protocole de validation memoire documente pour la Phase 1
- tests locaux passes : `9 passed`

## Limites actuelles

- pas d'agents de generation
- pas de `LangGraph`
- pas de `LLM`
- pas d'extraction automatique de personnages, lieux, evenements ou setups/payoffs depuis le texte
- le graphe `NetworkX` doit encore etre alimente manuellement
- certaines questions de coherence ne peuvent pas encore recevoir de reponse automatisee ; la memoire fournit surtout des preuves textuelles et structurelles

## Prochaine etape recommandee

La prochaine etape recommandee est de valider la memoire sur un roman reel avec le protocole de `docs/phase_1_validation.md`.

L'objectif est de verifier, sur des cas concrets, que :

- l'ingestion Markdown est suffisante pour couvrir le roman
- la recherche semantique retrouve les bons passages
- les donnees structurees et le graphe sont assez riches pour preparer la suite

Une fois cette validation terminee, la priorite suivante pourra etre l'enrichissement structure de la memoire avant l'introduction des agents.
