# État du projet

## Fonctionnalités livrées

- Phases 1 à 3 terminées : mémoire narrative composite (ChromaDB, SQLite,
  NetworkX), extraction automatique d'événements, connaissances des personnages
  avec filtrage temporel, suivi setup/payoff, agents de continuité et de style,
  débat multi-agents LangGraph.
- Interface Streamlit : tableau de bord, écriture guidée, mémoire/canon et récit court.
- Contraintes rigides extraites du brief et transmises aux agents du pipeline.
- Pipeline multi-scènes avec planification, continuité, rédaction, évaluation
  et sauvegarde du récit ; mémoire sémantique isolée par roman.
- Abstraction LLM multi-provider, profils YAML, fallbacks et journalisation des usages.

## Migration des modèles

Branche de travail : `chore/model-migration` (sans fusion).

- Registre `configs/model_registry.yaml` : rôles, providers, modèles et quotas.
- Prose et planification : `gemini-3.6-flash` ; sorties structurées et critiques :
  `openai/gpt-oss-120b` ; JSON léger : `openai/gpt-oss-20b` ; extraction locale : `qwen2.5:3b`.
- Routage et profils convertis aux rôles ; configurations explicites toujours acceptées.
- Gemini 3.x : `thinking_level` configurable, paramètres d'échantillonnage exclus.
- GPT-OSS : contenu final séparé du raisonnement et mode JSON object conservé.
- Quotas par modèle et par minute/jour UTC, avertissement à 80 % ; compteurs en mémoire
  du processus (non persistés et non partagés entre processus).
- `scripts/check_models.py` contrôle les catalogues Groq, Gemini et Ollama.
  Une clé absente ou un catalogue inaccessible donne `NON VÉRIFIÉ`.
- Benchmark stylist mis à jour ; les notes historiques ne valident pas ces nouveaux modèles.

## Validation et suites

Validation de la migration le 3 novembre 2026 : ruff et mypy passent,
179 tests passent (hors `tests/test_cli.py`). Les quatre modèles du registre
sont confirmés disponibles par leurs catalogues Groq, Gemini et Ollama.

Exécuter ruff sur src/tests/scripts, mypy sur src, pytest hors test_cli,
puis `python scripts/check_models.py`. Les tests unitaires utilisent des réponses simulées ;
la disponibilité réelle dépend du catalogue et des accès du compte.

À poursuivre : benchmark réel des nouveaux modèles, roman complet de bout en bout,
déploiement Streamlit Cloud, historique des versions et export PDF.
