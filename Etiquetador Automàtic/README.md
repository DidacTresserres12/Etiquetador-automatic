**README - Etiquetador Automàtic (AutoLabeler)** 



**Descripció del Projecte**

L'Etiquetador Automàtic és una eina desenvolupada en Python per al processament, avaluació i anotació automàtica de conjunts de dades d'imatges. El sistema compara les prediccions generades per models de visió per ordinador o algorismes d'optimització (com Algorismes Genètics) contra un conjunt de dades de referència (Ground Truth).



**Característiques Principals**

* Processament d'Imatges: Avaluació eficient d'imatges emmagatzemades al directori de proves.
* Avaluació contra Ground Truth: Comparació quantitativa mitjançant mètriques definides enfront de gt.json i gt\_reduced.json.
* Algorismes d'Optimització / Selecció: Cerca avançada i selecció d'etiquetes òptimes .



**Funció dels Fitxers del Projecte**

* AG.py: Implementació de l'Algorisme Genètic encarregat de la cerca, optimització i selecció automàtica d'etiquetes i anotacions per a les imatges.
* gt.json: Fitxer en format JSON que conté el Ground Truth complet, és a dir, les etiquetes i anotacions reals de referència per avaluar la precisió del sistema.
* gt\_reduced.json: Versió reduïda del Ground Truth dissenyada per fer proves ràpides i verificacions d'execució en menor temps.
* Directori test/ (Practica2/images/test/): Carpeta que emmagatzema el conjunt d'imatges d'entrada en format JPG sobre les quals s'executa el procés d'etiquetatge.



**Ús**

Preparar l'entorn: Col·locar les imatges a avaluar dins del directori corresponent:.../images/test/

Executar el procés: Iniciar l'avaluació o l'etiquetatge automàtic executant algun dels scripts principals segons la necessitat:

Per a optimització per Algorisme Genètic:

python Practica2/AG.py





