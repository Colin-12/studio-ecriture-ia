# Studio d'ecriture IA par agents

## Resume

Studio d'ecriture IA local base sur une `writers room` multi-agents.

Le projet genere soit une scene, soit un recit court en `3` scenes a partir d'une idee, avec un workflow narratif structure, `Ollama` en local, une memoire inter-scenes simple et un export Markdown exploitable.

## Fonctionnalites principales

- generation de scene avec `run-scene`
- generation de recit court en `3` scenes avec `create-story`
- workflow narratif multi-agents
- utilisation locale de `Ollama`
- decisions narratives avec `NarrativeDecisionAgent`
- memoire inter-scenes avec `canon_so_far`
- export Markdown + `story_memory.json`

## Architecture actuelle

Le workflow `create-story` suit actuellement cette structure :

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

`StoryArchitectAgent` construit un plan en `3` scenes : `trigger`, `confrontation`, `decision`.

Chaque scene peut recevoir :

- le plan de scene
- le `story_context`
- le `canon_so_far`, c'est-a-dire un resume simple des scenes deja generees

`NarrativeDecisionAgent` intervient apres chaque scene pour :

- accepter ou rejeter des ajouts narratifs simples
- produire des `canon_updates`
- ajouter des contraintes utiles pour la suite si necessaire

## Commande de demo recommandee

```bash
python -m src.app.cli create-story "Un homme découvre que ses souvenirs ont été modifiés par une IA" --story-mode original_story --genre thriller --tone sombre --pov first_person --language fr --use-llm --llm-mode ollama --llm-model qwen2.5:3b --llm-timeout 180 --llm-num-predict 420 --max-revision-rounds 0 --save-output
```

Configuration recommandee :

- `qwen2.5:3b` pour la generation de scenes en francais
- `--max-revision-rounds 0` pour une execution plus directe du MVP

## Sorties generees

Avec `--save-output`, le projet cree un dossier de sortie de ce type :

```text
outputs/stories/YYYYMMDD_HHMMSS_story/
  story_plan.md
  scene_01.md
  scene_02.md
  scene_03.md
  summary.md
  story_memory.json
```

Ces fichiers permettent de relire :

- le plan narratif
- les scenes generees
- les decisions narratives par scene
- un resume global
- une memoire narrative simple reutilisable

## Exemple de resultat attendu

Exemple court de structure :

- `Title: La memoire reecrite`
- `Stylist mode: llm`
- `Narrative decision: accepted additions / canon updates`
- `Saved story output: outputs/stories/...`

## Installation rapide

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python -m pytest -q
ollama pull qwen2.5:3b
ollama serve
```

Ensuite, lancer la commande de demo `create-story`.

## Tests

- etat actuel : `109 passed`
- un warning `ChromaDB` peut apparaitre selon l'environnement, mais il est non bloquant pour le MVP actuel

## Limites actuelles

- la qualite litteraire reste variable selon le prompt et le modele local
- `qwen2.5:1.5b` est plus leger et souvent plus rapide, mais moins fiable pour la prose narrative francaise
- plusieurs agents restent deterministes par conception
- l'interface graphique n'est pas encore implementee
- la memoire narrative reste simple, mais elle est deja fonctionnelle pour un recit court

## Prochaines etapes

- activer plus de `LLM` par agent quand c'est utile
- mieux extraire personnages, lieux, objets et elements de canon
- ajouter un workflow `continue-story`
- proposer une interface `Streamlit` ou web
- ameliorer la qualite stylistique et la stabilite narrative
