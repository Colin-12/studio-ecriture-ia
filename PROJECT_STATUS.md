# Project Status

## Etat actuel

Le projet `Studio d'ecriture IA par agents` est en `Phase 1 MVP`.
L'objectif courant reste la validation d'une memoire locale exploitable avant l'ajout d'agents de generation.

Le depot est deja pousse sur GitHub et a jour.

## Ce qui fonctionne deja

- `Frankenstein` est decoupe en `24` chapitres Markdown
- ingestion `SQLite` operationnelle via `SQLAlchemy`
- indexation semantique `ChromaDB` operationnelle avec `245` chunks
- donnees structurees SQLite disponibles : `12` personnages, `6` lieux, `10` evenements
- Continuiste enrichi disponible, sans `LLM`, pour croiser les preuves textuelles `ChromaDB` et les evenements structures `SQLite`
- filtrage et ranking des evenements structures ameliores pour reduire les resultats parasites
- interface `CLI` disponible pour :
  `ingest`, `index`, `search`, `continuity`, `list-chapters`, `list-characters`, `list-locations`, `list-events`
- graphe `NetworkX` genere dans `data/processed/frankenstein_graph.json`
- documentation de validation memoire disponible pour la Phase 1
- tests locaux passes : `19 passed`

## Limites actuelles

- pas d'agents de generation
- pas de `LLM`
- pas d'extraction automatique de personnages, lieux, evenements ou setups/payoffs depuis le texte
- les donnees structurees sont encore seedees manuellement
- pas encore de raisonnement `LLM`, seulement de la recuperation de preuves textuelles
- certaines questions de coherence ne peuvent pas encore recevoir de reponse automatisee ; la memoire fournit surtout des preuves textuelles et structurelles

## Prochaine etape recommandee

La prochaine etape recommandee est de poursuivre la validation memoire sur `Frankenstein` avec le protocole de `docs/phase_1_validation.md`, puis d'ameliorer l'enrichissement structure avant d'introduire des agents.

Points a verifier en priorite :

- l'ingestion Markdown est suffisante pour couvrir le roman
- la recherche semantique retrouve les bons passages
- les donnees structurees et le graphe sont assez riches pour preparer la suite
