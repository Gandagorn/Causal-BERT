##############################################################
# Annotations of German causal verbs, nouns and prepositions #
##############################################################

The data is provided under the terms of the Creative Commons BY-SA 
4.0 license specified at this URL:
  
        https://creativecommons.org/licenses/by-sa/4.0/


Contents of this File
=====================

(A) Introduction
(B) List of Files
(C) Statistics on the Annotations
(D) Lexicon File
(E) Useful Software
(F) Acknowledgements
(G) References


(A) Introduction
=================

This data release includes annotations of causal German verbs, 
nouns and prepositions from two sources:

a) The EuroParl Corpus (Koehn, 2005)
	http://www.statmt.org/europarl/
b) The TIGER Treebank (Brants et al., 2002)
	https://www.ims.uni-stuttgart.de/forschung/ressourcen/korpora/tiger/

The annotation process, IAA and annotation scheme are described 
in Rehbein & Ruppenhofer (to appear).


(B) List of Files
=================

The annotations are stored in three subfolders, one for each 
part of speech. All verbal annotations are stored in the same 
file, as we have more than 100 different causal triggers (word 
forms). Annotations for nouns (21 different causal triggers) 
and prepositions (26 causal triggers) are stored in separate 
files, with one file for each causal noun/preposition and data 
source (EuroParl/TIGER).

.
├── lexicon
├──── CC_lexicon.ods
├── data
├──── verb
├─────── verbs-tiger.tsv
├──── noun
├─────── Anlass-tiger.tsv
├─────── Ausloeser-tiger.tsv
├─────── Auswirkung-tiger.tsv
├─────── Ergebnis-tiger.tsv
├─────── Folge-tiger.tsv
├─────── Grund-tiger.tsv
├─────── Hauptursache-tiger.tsv
├─────── Hintergrund-tiger.tsv
├─────── Keim-tiger.tsv
├─────── Konsequenz-tiger.tsv
├─────── Motivation-tiger.tsv
├─────── Motiv-tiger.tsv
├─────── Quelle-tiger.tsv
├─────── Reaktion-tiger.tsv
├─────── Resultat-tiger.tsv
├─────── Spaetfolge-tiger.tsv
├─────── Stimulus-tiger.tsv
├─────── Triebkraft-tiger.tsv
├─────── Ursache-tiger.tsv
├─────── Veranlassung-tiger.tsv
├─────── Zweck-tiger.tsv
├──── prep 
├─────── an-europarl.tsv, an-tiger.tsv
├─────── angesichts-europarl.tsv, angesichts-tiger.tsv
├─────── aufgrund-europarl.tsv, aufgrund-tiger.tsv
├─────── aus-europarl.tsv, aus-tiger.tsv
├─────── bei-europarl.tsv, bei-tiger.tsv
├─────── dank-europarl.tsv, dank-tiger.tsv
├─────── durch-europarl.tsv, durch-tiger.tsv
├─────── fuer-europarl.tsv, fuer-tiger.tsv
├─────── halber-europarl.tsv, halber-tiger.tsv
├─────── infolge-europarl.tsv, infolge-tiger.tsv
├─────── mangels-europarl.tsv, mangels-tiger.tsv
├─────── mit-europarl.tsv, mit-tiger.tsv
├─────── mittels-europarl.tsv, mittels-tiger.tsv
├─────── nach-europarl.tsv, nach-tiger.tsv
├─────── ob-europarl.tsv, ob-tiger.tsv
├─────── ohne-europarl.tsv, ohne-tiger.tsv
├─────── trotz-europarl.tsv, trotz-tiger.tsv
├─────── ueber-europarl.tsv, ueber-tiger.tsv
├─────── ungeachtet-europarl.tsv, ungeachtet-tiger.tsv
├─────── unter-europarl.tsv, unter-tiger.tsv
├─────── von-europarl.tsv, von-tiger.tsv
├─────── vor-europarl.tsv, vor-tiger.tsv
├─────── wegen-europarl.tsv, wegen-tiger.tsv
├─────── zugunsten-europarl.tsv, zugunsten-tiger.tsv
├─────── zuliebe-europarl.tsv, zuliebe-tiger.tsv
├─────── zwecks-europarl.tsv, zwecks-tiger.tsv
└── README.md  (this file)


(C) Statistics on the Annotations
=================================

The annotations are summarised by the following set of numbers. 
For more details, please refer to Rehbein & Ruppenhofer (to appear).

---------------------------------------------------------------------------------------
|Source  | POS  |# forms|# instances|% causal| Consequence  | Motivation | Purpose    |
---------------------------------------------------------------------------------------
|Europarl| VERB |   112 |       932 |   78.9 | 76.3%   (561)| 22.0% (162)|  1.6%  (12)|
|TiGer 	 | NOUN |    21 |     1,178 |   69.3 | 43.9%   (359)| 50.2% (410)|  5.9%  (48)|
|TiGer 	 | PREP |    26 |       983 |   40.9 | 43.3%   (174)| 42.3% (170)| 14.4%  (58)|
|EuroParl| PREP |    26 |     1,297 |   54.7 | 39.3%   (279)| 36.1% (256)| 24.5% (174)|
---------------------------------------------------------------------------------------
|Total 	 |	|   159 |     4,390 |   60.7 | 51.5% (1,373)| 37.5% (998)| 11.0% (292)| 
---------------------------------------------------------------------------------------



(D) Lexicon File
================

The lexicon file `CC_lexicon.ods` presents annotation statistics for each of the lemmas covered.
In the spreadsheet file, adposition, noun and verb lemmas are presented on three separate tabs.

For each lemma, the file shows how many, if any, instances of the lemma have been annotated as 
cases of one of three subtypes of causation (Consequence, Motivation or Purpose) with a given 
combination of participant roles. 

For example, consider the entry below for the noun lemma  Auslöser `trigger'. 
3 instances were annotated as exhibiting the causal subtype Consequence, each realized with a 
Cause and an Effect participant role. A further 7 instances of Auslöser were labeled as 
exhibiting the causal subtype Motivation. In 5 of these 7 cases, the two  participant roles 
Cause and Effect were realized. In 2 of the 7 cases, only the Cause role was expressed. 
Auslöser was never annotated as exhibiting the causal subtype Purpose.
The final column, Degree, shows that Auslöser concerns positive causation (facilitate) rather
than negative causation (inhibit).


Lemma			Consequence		Motivation		Purpose		Degree
Ausloeser	x	Cause:Effect:3	x	Cause:Effect:5, Cause:2			facilitate

For further details on the annotation categories, please consult the paper.


(E) Useful Software
===================

The data was annotated, using WebAnno (Yimam & Gurevych, 2013), 
and the annotations can be uploaded and displayed, using this tool:

    https://webanno.github.io/webanno/



(F) Acknowledgements
====================

This research has been partially supported by the Leibniz Science Campus 
"Empirical Linguistics and Computational Modeling", funded by the 
Leibniz Association under grant no. SAS-2015-IDS-LWC and by the Ministry 
of Science, Research, and Art (MWK) of the state of Baden-Württemberg.

We would also like to acknowledge the work of our student annotator, 
Lasse Becker-Czarnetzki.



(G) References
==============

P. Koehn (2005): Europarl: A Parallel Corpus for Statistical Machine Translation, MT Summit.

S. Brants, S. Dipper, S. Hansen, W. Lezius, and G. Smith (2002): The TIGER treebank. Proceedings of the workshop on treebanks and linguistic theories. 

I. Rehbein & J. Ruppenhofer (to appear): A New Resource for German Causal Language.

S.M. Yimam & I. Gurevych (2013): Webanno: A flexible, web-based and visually supported system for distributed annotations. In The 51th Annual Meeting of the Association for Computational Linguistics (ACL) -- System Demonstrations. 


