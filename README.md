# Analyse de sentiments multi-sujets pour avis clients

## Description

Ce projet implémente un pipeline complet d'analyse de sentiment multi-sujets pour des avis clients en français. Le système utilise des techniques avancées d'apprentissage automatique pour :

- Segmenter automatiquement les avis en chunks sémantiques
- Détecter plusieurs sujets dans chaque segment
- Analyser le sentiment pour chaque sujet individuellement
- Fournir des analyses détaillées au niveau des sujets et des avis

## Fonctionnalités principales

✅ **Chunking sémantique** : Segmentation intelligente du texte basée sur la similitude sémantique  
✅ **Détection multi-sujets** : Identification de plusieurs thématiques par segment  
✅ **Analyse de sentiment par sujet** : Sentiment individuel pour chaque sujet détecté  
✅ **Configuration flexible** : Presets pour différents cas d'usage (rapide, précis, CPU)  
✅ **Export Excel** : Résultats détaillés exportés en format Excel  

## Installation

### Prérequis
- Python 3.8+
- pip
- un modèle d'embedding disponible en local avec `ollama pull [nomdumodele]`
  

### Installation des dépendances

```bash
pip install -r requirements.txt
```

### Modèle spaCy (optionnel)
```bash
python -m spacy download fr_core_news_sm
```

## Structure du projet

```
├── main_multi_topic.py              # Script principal d'exécution
├── config.py                        # Configuration et paramètres
├── semantic_chunker_simple.py       # Chunking sémantique
├── topic_detector_multi.py          # Détection multi-sujets
├── sentiment_analyzer_multi_topic.py # Analyse de sentiment
├── generate_fake_dataset.py         # Génération de données de test
├── requirements.txt                  # Dépendances Python
├── fake_reviews_dataset.xlsx        # Dataset d'exemple
└── README.md                        # Cette documentation
```

## Utilisation

### Utilisation basique

```bash
python main_multi_topic.py
```

Le script traite par défaut le fichier `fake_reviews_dataset.xlsx` et génère `analyzed_reviews_results_multi_topic.xlsx`.

### Configuration personnalisée

#### Utilisation des presets

```python
from config import load_config
from main_multi_topic import MultiTopicSentimentPipeline

# Preset rapide (moins précis, plus rapide)
config = load_config(preset='fast')

# Preset précis (plus précis, plus lent)
config = load_config(preset='accurate')

# Preset CPU (optimisé pour CPU uniquement)
config = load_config(preset='cpu')

pipeline = MultiTopicSentimentPipeline(config)
```

#### Variables d'environnement

```bash
export EMBEDDING_MODEL="granite-embedding"
export SENTIMENT_MODEL="cmarkea/distilcamembert-base-sentiment"
export SIMILARITY_THRESHOLD="0.7"
export CONFIDENCE_THRESHOLD="0.6"
export BATCH_SIZE="8"
export INPUT_DATA_PATH="mon_fichier.xlsx"
export OUTPUT_DATA_PATH="resultats.xlsx"
```

### Traitement d'un seul avis

```python
from main_multi_topic import MultiTopicSentimentPipeline

pipeline = MultiTopicSentimentPipeline()
result = pipeline.process_single_review("Cet hôtel est magnifique mais le service était décevant.")

print(result['overall_sentiment'])  # Sentiment général
print(result['topic_sentiments'])   # Sentiments par sujet
```

### Traitement d'un DataFrame

```python
import pandas as pd
from main_multi_topic import MultiTopicSentimentPipeline

df = pd.read_excel("mes_avis.xlsx")
pipeline = MultiTopicSentimentPipeline()
results = pipeline.process_dataframe(df)
```

## Format des données d'entrée

Le fichier Excel d'entrée doit contenir au minimum une colonne `text` avec les avis à analyser :

```
| text                                    |
|-----------------------------------------|
| "Excellent service, personnel aimable"  |
| "Chambre sale mais vue magnifique"      |
```

## Format des résultats

Le fichier de sortie contient les colonnes suivantes :

### Colonnes de base
- `text` : Texte original de l'avis
- `overall_sentiment` : Sentiment global (POSITIVE/NEGATIVE/NEUTRAL)

### Métriques de chunking
- `chunk_count` : Nombre de segments créés
- `chunks_with_topics` : Segments avec sujets détectés
- `chunks_without_topics` : Segments sans sujets

### Métriques de sujets
- `unique_topics_detected` : Liste des sujets uniques détectés
- `total_topic_instances` : Nombre total d'instances de sujets
- `topic_sentiments` : Mapping JSON sujet → sentiment

### Métriques de sentiment
- `positive_topic_sentiments` : Nombre de sentiments positifs par sujet
- `negative_topic_sentiments` : Nombre de sentiments négatifs par sujet
- `neutral_topic_sentiments` : Nombre de sentiments neutres par sujet

### Métriques de performance
- `processing_time` : Temps de traitement (secondes)
- `char_count` : Nombre de caractères dans l'avis
- `has_error` : Indicateur d'erreur

### Résultats détaillés
- `detailed_results` : JSON avec détails complets par chunk

## Configuration avancée

### Paramètres du chunker sémantique
```python
chunker_config = SemanticChunkerConfig(
    embedding_model="granite-embedding",
    similarity_threshold=0.7,  # Seuil de similitude (0.0-1.0)
    max_chunk_size=512,        # Taille max des chunks
    min_chunk_size=50,         # Taille min des chunks
    sentence_overlap=1         # Chevauchement entre chunks
)
```

### Paramètres du détecteur de sujets
```python
topic_config = TopicDetectorConfig(
    embedding_model="granite-embedding",
    confidence_threshold=0.6,   # Seuil de confiance (0.0-1.0)
    max_topics_per_chunk=3     # Nombre max de sujets par chunk
)
```

### Paramètres de l'analyseur de sentiment
```python
sentiment_config = SentimentAnalyzerConfig(
    model_name="cmarkea/distilcamembert-base-sentiment",
    device=None,              # "cuda", "cpu" ou None (auto)
    batch_size=8             # Taille des batches
)
```

## Performances

### Presets recommandés

| Preset | Vitesse | Précision | Usage recommandé |
|--------|---------|-----------|------------------|
| `fast` | ⚡⚡⚡ | ⭐⭐ | Tests, prototypage |
| `accurate` | ⚡ | ⭐⭐⭐ | Production, analyse fine |
| `cpu` | ⚡⚡ | ⭐⭐ | Serveurs sans GPU |

### Optimisation mémoire

Pour traiter de gros volumes :
- Réduire `batch_size`
- Limiter `max_topics_per_chunk`
- Utiliser `max_reviews_to_process` pour traiter par lots

## Dépannage

### Erreurs courantes

**Erreur de modèle non trouvé**
```bash
# Installer/mettre à jour transformers
pip install --upgrade transformers
```

**Erreur de mémoire GPU**
```python
# Forcer l'utilisation CPU
config.sentiment_analyzer.device = "cpu"
config.sentiment_analyzer.batch_size = 4
```

**Fichier Excel corrompu**
```python
# Vérifier l'encodage
df = pd.read_excel("fichier.xlsx", engine='openpyxl')
```

### Logs et debug

Activer le mode debug :
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Contribuer

1. Fork le projet
2. Créer une branche (`git checkout -b feature/nouvelle-fonctionnalite`)
3. Commit (`git commit -am 'Ajout nouvelle fonctionnalité'`)
4. Push (`git push origin feature/nouvelle-fonctionnalite`)
5. Créer une Pull Request

## Licence

Ce projet est distribué sous licence MIT. Voir le fichier `LICENSE` pour plus de détails.

## Support

Pour signaler un bug ou proposer une amélioration, créez une issue sur le repository GitHub.
