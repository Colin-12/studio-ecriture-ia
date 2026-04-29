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

## Demo locale Phase 1

Cette sequence permet de tester la memoire localement sur un roman source en Markdown.

### 1. Initialiser la base SQLite

```bash
python -m src.memory.init_db
```

### 2. Ingerer les chapitres Markdown

Les fichiers `.md` doivent etre places dans `manuscript/source_novel/`.

```bash
python -m src.app.cli ingest
```

### 3. Lister les chapitres avec la CLI

```bash
python -m src.app.cli list-chapters
```

### 4. Indexer les chapitres dans ChromaDB

```bash
python -m src.app.cli index
```

### 5. Lancer une recherche semantique

Exemple avec la requete `the creation of the being` :

```bash
python -m src.app.cli search "the creation of the being"
```

### 6. Interroger le Continuiste simple

Exemple avec la question `where does the creature learn language?` :

```bash
python -m src.app.cli continuity "where does the creature learn language?"
```

Cette commande affiche :

- les passages semantiques retrouves
- les evenements structures pertinents
- une conclusion courte et deterministe, sans `LLM`

### 7. Lancer le workflow run-scene

Exemple complet :

```bash
python -m src.app.cli run-scene "Un homme découvre que ses souvenirs ont été modifiés par une IA" --story-mode original_story --genre thriller --tone sombre --pov first_person --language fr --use-llm --llm-mode ollama --force-revision --max-revision-rounds 1
```

Ce workflow utilise la chaine suivante :

- `SceneArchitect`
- `DevilAdvocate`
- `Visionary`
- `EmotionGuardian`
- `Continuity`
- `Stylist`
- `Editor`
- `QualityEvaluator`
- boucle de revision

`EmotionGuardian` ajoute avant le draft :

- `emotional_core`
- `internal_conflict`
- `fear_or_desire`
- `emotional_risk`
- `suggested_emotional_beat`

Notes :

- `Ollama` est utilise localement, sans API payante
- `--story-mode original_story` desactive l'usage d'un canon existant
- `--max-revision-rounds` borne strictement la revision, donc sans boucle infinie

### 8. Lancer les tests

```bash
python -m pytest -q
```

## Etat actuel

Le depot contient maintenant le squelette propre de la Phase 1 du systeme de memoire. Les prochaines etapes pourront ajouter les modules d'ingestion, de stockage, d'indexation et de retrieval sans sur-ingenierie.
