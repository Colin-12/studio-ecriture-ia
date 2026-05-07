# Project Status

## Etat actuel

Deux chantiers complétés sur la branche phase-1-completion :
- ruff + mypy configurés (pyproject.toml), 119 tests passent
- Abstraction LLM provider-agnostique opérationnelle

## Décision actuelle

- Prototype agentique taggeré v0.3-agents-prototype
- Expansion agentique gelée temporairement
- Branche phase-1-completion active
- Priorité : compléter les trois tests mémoire Phase 1

## Diagnostic Phase 1

- Retrieval sémantique Frankenstein : fonctionnel
- Mémoire structurée : existe mais seedée manuellement
- Trois capacités critiques non validées :
  a. extraction automatique d'événements depuis la prose
  b. état épistémique des personnages (CharacterKnowledge)
  c. setup/payoff seedé et testé

## Refonte LLM  complétée

- src/llm/ : interface LLMProvider abstraite + 5 providers + 2 stubs
- configs/llm_routing.yaml : routing par agent
- configs/llm_profiles/ : free_only, local_only, mixed_budget
- Fallback automatique sur RateLimitError
- logs/llm_usage.jsonl : observabilité par session
- Flag CLI --llm-profile opérationnel
- Compatibilité descendante --llm-mode préservée

## Tests Phase 1 à construire

### Test 1  Extraction automatique d'événements
Objectif : lire un chapitre Markdown et produire des événements structurés sans saisie manuelle.
Référence : frankenstein_events_reference.json (15-20 événements, établie par lecture humaine).
Critères : precision >= 0.60, recall >= 0.50, <= 2 hallucinations par chapitre.
Code attendu : src/memory/event_extractor.py, tests/test_event_extractor.py

### Test 2  État épistémique des personnages
Objectif : répondre à "que sait un personnage à un chapitre donné ?".
Table : CharacterKnowledge(id, character_id, fact, learned_at_chapter, source_event_id, belief_status, confidence).
belief_status : true / false / suspected / hidden (obligatoire  couvre mensonge, secret, ambiguïté).
Critères : 0% de fuite temporelle (faits appris après N exclus), 4 valeurs de belief_status couvertes.
Code attendu : src/memory/knowledge.py, tests/test_character_knowledge.py

### Test 3  Setup / Payoff fonctionnel
Objectif : suivre ce qui est planté, payé, partiellement payé.
Enrichissement : champ progress (planted / partially_paid / fully_paid) + payoff_chapters (JSON list).
Critères : 5 setups seedés, 3 requêtes testées à chapitres différents, au moins 1 partially_paid correct.
Code attendu : src/memory/setup_payoff.py, tests/test_setup_payoff.py

### Gate de sortie Phase 1
Les trois tests ci-dessus doivent passer leurs critères chiffrés avant toute nouvelle feature agentique.
