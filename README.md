# Studio d'écriture IA

Studio d'écriture multi-agents pour romans longs. Une *writer's room* automatisée où des agents LLM débattent, se challengent et arbitrent avant de produire la prose — chapitre après chapitre.

## Principe

Chaque chapitre passe par un graphe de débat LangGraph :

1. **Contract Parser** extrait les contraintes rigides du brief auteur (personnages, événement central, interdictions).
2. **Continuiste** interroge la mémoire sémantique (ChromaDB) et structurée (SQLite).
3. **Architecte** formule le brief initial.
4. **Avocat du Diable** et **Visionnaire** challengent le brief en parallèle.
5. **Gardien de l'Émotion** vérifie la cohérence émotionnelle.
6. **Architecte** arbitre et produit le brief final.
7. **Styliste** (Gemini 2.5 Flash) rédige la prose.
8. **Éditeur** note et déclenche une révision si le score < 3/5.

Les `hard_constraints` extraites à l'étape 1 sont injectées dans tous les prompts : les agents créatifs peuvent enrichir la zone libre, jamais modifier le contrat narratif.

## Architecture

```
Roman
  Chapitres (SQLite + Markdown)
    Graphe de débat (LangGraph)
      contract_parser -> continuity -> architect_initial
      -> [devil_advocate | visionary | emotion_guardian]
      -> architect_arbitrate -> stylist -> editor
      -> (révision si score < 3, max 2 tours)

Mémoire composite
  ChromaDB  — recherche sémantique par roman (collection novel_{id})
  SQLite    — chapitres, personnages, événements, setup/payoffs
  NetworkX  — graphe de cohérence narratif
```

**Routing LLM validé par benchmark** (3 briefs, 2 lecteurs indépendants) :

| Modèle | Score | Usage |
|--------|-------|-------|
| Gemini 2.5 Flash | 4.6/5 | Prose (stylist, editor, architect) |
| Groq Llama 3.3 70B | 2.9/5 | Challenge (devil, visionary, continuity) |
| Ollama qwen2.5:3b | 1.2/5 | JSON uniquement (event_extractor) |

## Interface

```bash
PYTHONPATH=. streamlit run src/app/streamlit_app.py
```

4 pages disponibles :

- **Tableau de bord** — métriques du roman, chapitres écrits, prochaine action recommandée
- **Écriture** — wizard 4 étapes (blueprint → contrat → débat → prose → mémoire)
- **Mémoire / Canon** — timeline, personnages, événements, setup/payoffs, contradictions
- **Récit court** — génération one-shot avec export Markdown

L'onboarding guide les nouveaux utilisateurs vers la création du premier roman et le premier chapitre.

## Installation

**Prérequis :** Python 3.11+, Ollama (optionnel, pour modèles locaux)

```bash
git clone https://github.com/Colin-12/studio-ecriture-ia
cd studio-ecriture-ia
python -m venv .venv
source .venv/Scripts/activate   # Windows Git Bash
pip install -r requirements.txt
```

**Configuration :**

```bash
cp .env.example .env
# Renseigner GROQ_API_KEY et GOOGLE_API_KEY dans .env
```

Variables utilisées :

```
GROQ_API_KEY=
GOOGLE_API_KEY=
OLLAMA_HOST=http://localhost:11434
```

Chaque appel LLM est journalisé dans `logs/llm_usage.jsonl`.

## Stack technique

| Composant | Rôle |
|-----------|------|
| LangGraph >= 0.2 | Orchestration du graphe de débat avec état |
| Gemini 2.5 Flash | Prose et arbitrage (Google AI Studio) |
| Groq Llama 3.3 70B | Agents de challenge (vitesse, coût nul) |
| ChromaDB | Recherche sémantique par roman |
| SQLite + SQLAlchemy | Mémoire structurée (romans, chapitres, events) |
| NetworkX | Graphe de cohérence narrative |
| Streamlit | Interface utilisateur |
| YAML routing | Configuration par agent et par profil LLM |

Profils LLM disponibles : `mixed_budget`, `free_only`, `local_only`.  
Rate limiting token-bucket intégré (Groq : 5 500 TPM).

## Tests

```bash
pytest -q --ignore=tests/test_cli.py
```

151 tests — ruff + mypy clean sur 75 fichiers sources.

## Roadmap

**Fait**

- Mémoire narrative composite : ChromaDB + SQLite + NetworkX
- Continuiste enrichi et Styliste Gemini 2.5 Flash
- Graphe de débat LangGraph validé avec vrais appels API
- Benchmark Styliste documenté (3 briefs, blind review)
- Interface Streamlit wizard 4 étapes
- Contraintes rigides (contract parser) préservées entre agents
- Rate limiter token-bucket pour Groq
- Onboarding guidé et création de roman en mini-wizard
- Collection ChromaDB par roman (`novel_{id}`)

**En cours / à venir**

- Prompts agents plus concis
- Déploiement Streamlit Cloud
- Premier roman complet généré bout en bout
- Historique des versions de chapitres
- Export PDF
