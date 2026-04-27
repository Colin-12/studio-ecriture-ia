# Studio d'ecriture IA par agents

Projet personnel local pour construire un systeme d'ecriture assistee par IA oriente multi-agents, avec une progression par phases.

## Objectif

L'objectif final est de produire des romans longs avec un systeme multi-agents specialise. Cette premiere phase se limite volontairement au systeme de memoire, afin de poser une base simple, lisible et exploitable avant d'ajouter la generation.

## Scope de la Phase 1

La Phase 1 couvre uniquement :

- l'ingestion de contenus sources
- la structuration d'une memoire locale
- le stockage de connaissances et metadonnees
- la retrieval pour retrouver des informations utiles
- les premiers liens de type graphe entre elements memorises

Cette phase n'implemente pas :

- les agents de generation
- LangGraph
- Claude
- Groq
- Streamlit

## Stack memoire prevue

- `ChromaDB` pour le stockage vectoriel local
- `sentence-transformers` pour les embeddings
- `SQLite` via `SQLAlchemy` pour les donnees structurees
- `networkx` pour les relations de type graphe
- `pandas`, `pydantic`, `pyyaml` et `python-dotenv` pour l'ingestion, la configuration et la validation
- `pytest` pour les tests

## Structure du projet

```text
data/raw/
data/processed/
data/chroma/
manuscript/source_novel/
src/ingest/
src/memory/
src/retrieval/
src/graph/
src/app/
db/
tests/
configs/
```

## Installation prevue

Les commandes ci-dessous sont preparees pour la suite, mais ne sont pas executees automatiquement.

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Etat actuel

Le depot contient maintenant le squelette propre de la Phase 1 du systeme de memoire. Les prochaines etapes pourront ajouter les modules d'ingestion, de stockage, d'indexation et de retrieval sans sur-ingenierie.
