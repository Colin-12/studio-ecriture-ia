# Studio d'ecriture IA par agents

## Resume

Studio d'ecriture IA local base sur une `writers room` multi-agents.

Le projet supporte maintenant deux modes principaux :

- `create-story` : creer un recit court en `3` scenes a partir d'une impulsion humaine
- `continue-story` : continuer un recit existant a partir d'un canon + une direction humaine

Le depot contient aussi `examples/trisha_revenge_story/`, un exemple canonise pour tester la continuation de recit sans imposer la suite.

## Fonctionnalites principales

- generation de scene avec `run-scene`
- generation de recit court en `3` scenes avec `create-story`
- continuation de recit avec `continue-story`
- workflow narratif multi-agents
- `NarrativeDecisionAgent` pour arbitrer les ajouts et preparer le canon
- `UserIntentAgent` pour interpreter la direction utilisateur dans `continue-story` sans modifier le canon
- memoire inter-scenes avec `canon_so_far`
- export Markdown + `story_memory.json`
- support local de `Ollama`
- `--agent-depth {fast, balanced, deep}`
- `--llm-keep-alive` pour garder le modele Ollama charge

## Architecture actuelle

```text
StoryArchitect
  -> Scene workflow
       -> SceneArchitect
       -> DevilAdvocate
       -> Visionary
       -> EmotionGuardian
       -> Stylist / Ollama
       -> Editor
       -> QualityEvaluator
       -> BetaReader
       -> CommercialEditor
       -> NarrativeDecision
  -> Documentalist
  -> story_memory.json
```

En mode `continue-story`, la direction humaine est d'abord interpretee par `UserIntentAgent`, puis injectee comme intention creative non canonique dans le contexte de scene.

## Profondeur agentique

- `fast` : agents mostly deterministic, LLM mainly for stylist
- `balanced` : default writer room with deterministic analysis agents and LLM stylist
- `deep` : reserved for deeper LLM-based agent deliberation

## Commande recommandee

Configuration de reference pour `create-story` en francais :

```bash
python -m src.app.cli create-story "Un homme découvre que ses souvenirs ont été modifiés par une IA" --story-mode original_story --genre thriller --tone sombre --pov first_person --language fr --use-llm --llm-mode ollama --llm-model qwen2.5:3b --llm-timeout 180 --llm-num-predict 420 --max-revision-rounds 0 --save-output
```

Notes :

- `qwen2.5:3b` est la configuration recommandee pour le `StylistAgent`
- `--llm-keep-alive` peut etre utilise pour garder le modele charge entre plusieurs appels

## Exemple continue-story

```bash
python -m src.app.cli continue-story examples/trisha_revenge_story --direction "Anaïs veut comprendre si Trisha la volontairement attirée sur le parking." --use-llm --llm-mode ollama --llm-model qwen2.5:3b --llm-timeout 180 --llm-num-predict 420 --llm-keep-alive 10m --agent-depth fast --save-output
```

## Sorties generees

Avec `--save-output`, le projet genere des sorties Markdown exploitables :

- `outputs/` pour `run-scene`
- `outputs/stories/` pour `create-story`
- `continuations/` dans le dossier canon source pour `continue-story`

`create-story` exporte notamment :

```text
outputs/stories/YYYYMMDD_HHMMSS_story/
  story_plan.md
  scene_01.md
  scene_02.md
  scene_03.md
  summary.md
  story_memory.json
```

## Installation rapide

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python -m pytest -q
ollama pull qwen2.5:3b
ollama serve
```

## Tests

- etat actuel : `122 passed`
- un warning `ChromaDB` peut apparaitre selon l'environnement, mais il est non bloquant

## Limites actuelles

- la qualite litteraire reste variable selon le modele local
- `qwen2.5:1.5b` est plus leger mais moins fiable pour la prose narrative francaise
- plusieurs agents restent deterministes
- l'interface graphique n'est pas encore faite
- la memoire narrative reste simple, mais elle est deja fonctionnelle

## Prochaines etapes

- activer plus de `LLM` par agent selon `agent_depth`
- mieux extraire personnages, lieux, objets et canon
- enrichir `continue-story`
- ajouter une interface `Streamlit` ou web
- ameliorer la qualite stylistique
