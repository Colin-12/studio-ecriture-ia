# Frankenstein Validation Notes

## Etat des premiers tests

- `Frankenstein` a ete decoupe en `24` chapitres Markdown.
- Le roman a ete ingere dans `SQLite`.
- Le roman a ete indexe dans `ChromaDB` en `245` chunks.

## Resultats de recherche semantique

- La recherche `the creature learns language` retrouve correctement les chapitres `12-13`.
- La recherche `the creation of the being` retrouve correctement les chapitres `3-4`.
- La recherche `Victor creates the monster` ou `Victor animates the creature` est moins pertinente.

## Conclusion

La memoire semantique fonctionne sur ce premier test reel.

La qualite des resultats depend toutefois fortement :

- de la formulation de la requete ;
- du vocabulaire effectivement present dans le texte source.
