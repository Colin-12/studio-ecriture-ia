# Project Status

## Etat actuel

Le projet `Studio d'ecriture IA par agents` est un MVP local de `writers room` narrative.

Il supporte maintenant :

- `run-scene`
- `create-story`
- `continue-story`
- un exemple canonise : `examples/trisha_revenge_story/`
- `NarrativeDecisionAgent`
- `UserIntentAgent`
- `canon_so_far` inter-scenes
- `story_memory.json`
- export Markdown
- `--agent-depth {fast, balanced, deep}`
- `--llm-keep-alive` pour `Ollama`

## Ce qui fonctionne deja

- memoire locale Phase 1 exploitable
- `run-scene` operationnel avec workflow agentique complet
- `create-story` operationnel pour generer un recit court en `3` scenes
- `continue-story` operationnel pour prolonger un recit canonise sans modifier le canon source
- `StoryArchitectAgent` pour structurer un recit court
- `NarrativeDecisionAgent` pour arbitrer les ajouts et produire des `canon_updates`
- `UserIntentAgent` pour interpreter la direction utilisateur comme intention creative non canonique
- `Ollama` local supporte `qwen2.5:3b`
- configuration recommandee actuelle :
  `--llm-mode ollama --llm-model qwen2.5:3b --llm-timeout 180 --llm-num-predict 420`
- `--llm-keep-alive` disponible pour garder le modele charge
- `agent_depth` disponible avec :
  `fast`, `balanced`, `deep`
- sorties Markdown disponibles pour scene, story et continuation
- warning `ChromaDB` connu mais non bloquant
- tests locaux passes : `122 passed`

## Limites actuelles

- plusieurs agents restent deterministes
- la qualite des drafts depend encore fortement du modele local
- la memoire narrative reste simple
- pas d'interface graphique
- pas de `LLM` distant ou payant

## Prochaine etape recommandee

- faire varier davantage les agents selon `agent_depth`
- enrichir la deliberation entre agents
- renforcer `continue-story`
- ameliorer la qualite stylistique et la stabilite narrative
