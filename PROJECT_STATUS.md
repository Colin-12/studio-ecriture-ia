# Project Status

## Etat actuel

Le projet `Studio d'ecriture IA par agents` est en `Phase 1 MVP`.
Le projet dispose maintenant d'une base memoire Phase 1 exploitable et d'un workflow agentique local pour preparer des scenes.

Le depot est deja pousse sur GitHub et a jour.

## Ce qui fonctionne deja

- `Frankenstein` est decoupe en `24` chapitres Markdown
- ingestion `SQLite` operationnelle via `SQLAlchemy`
- indexation semantique `ChromaDB` operationnelle avec `245` chunks
- donnees structurees SQLite disponibles : `12` personnages, `6` lieux, `10` evenements
- Continuiste enrichi disponible, sans `LLM`, pour croiser les preuves textuelles `ChromaDB` et les evenements structures `SQLite`
- conclusion deterministe disponible dans le Continuiste enrichi, sans synthese `LLM`
- filtrage et ranking des evenements structures ameliores pour reduire les resultats parasites
- workflow `writer's room` disponible :
  `SceneArchitect -> DevilAdvocate -> Visionary -> EmotionGuardian -> Continuity -> Stylist/Ollama -> Editor -> QualityEvaluator -> BetaReader -> CommercialEditor -> Revision loop -> Output Markdown`
- `EmotionGuardianAgent` integre au workflow, avec :
  `emotional_core`, `internal_conflict`, `fear_or_desire`, `emotional_risk`, `suggested_emotional_beat`
- `EmotionGuardianAgent` s'adapte a `genre`, `tone`, `pov` et `language`
- `BetaReaderAgent` integre au workflow, avec :
  `confusion_points`, `engagement_points`, `boredom_risks`, `would_continue_reading`, `reader_notes`, `revision_targets`
- `CommercialEditorAgent` integre au workflow, avec :
  `hook_score`, `market_angle`, `title_suggestions`, `format_suggestion`, `publication_risk`, `commercial_notes`
- `story_mode` disponible :
  `existing_novel` ou `original_story`
- `StylistAgent` peut fonctionner en mode deterministe, en mode `mock`, ou via `Ollama` local
- `Ollama` local supporte `qwen2.5:3b`, sans API payante
- parametres narratifs supportes dans `run-scene` :
  `--genre`, `--tone`, `--pov`, `--language`
- evaluation qualite disponible avec les criteres :
  `originality`, `narrative_tension`, `emotion`, `coherence`, `style`, `reader_potential`
- revision supportee avec :
  `--max-revision-rounds` et `--force-revision`
- la revision est bornee par `max_revision_rounds`, donc sans boucle infinie
- `--save-output` sauvegarde les scenes en Markdown dans `outputs/`
- interface `CLI` disponible pour :
  `ingest`, `index`, `search`, `continuity`, `run-scene`, `list-chapters`, `list-characters`, `list-locations`, `list-events`
- graphe `NetworkX` genere dans `data/processed/frankenstein_graph.json`
- documentation de validation memoire disponible pour la Phase 1
- tests locaux passes : `57 passed`

## Limites actuelles

- pas d'agents de generation
- pas de `LLM` distant ou payant
- pas d'extraction automatique de personnages, lieux, evenements ou setups/payoffs depuis le texte
- les donnees structurees sont encore seedees manuellement
- les agents restent majoritairement deterministes hors usage explicite de `Ollama`
- certaines questions de coherence ne peuvent pas encore recevoir de reponse automatisee robuste ; la memoire fournit surtout des preuves textuelles et structurelles

## Prochaine etape recommandee

La prochaine etape recommandee est de stabiliser le workflow agentique sur plusieurs cas de scene, puis d'ameliorer progressivement la qualite des drafts et de la revision.

Points a verifier en priorite :

- la robustesse du mode `original_story`
- la qualite des drafts en mode `Ollama`
- la pertinence de l'evaluation qualite, du retour lecteur et de la boucle de revision
