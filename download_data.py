"""
Script pour télécharger et préparer les données d'avis français pour l'analyse de sentiment.
"""

import pandas as pd
from datasets import load_dataset
import os

def download_french_book_reviews():
    """Télécharge le dataset d'avis de livres français de Hugging Face."""
    print("📥 Téléchargement du dataset d'avis de livres français...")
    
    try:
        # Charger le dataset
        dataset = load_dataset("Abirate/french_book_reviews")
        
        # Convertir en DataFrame
        df = pd.DataFrame(dataset['train'])
        
        print(f"✅ Dataset téléchargé avec succès : {len(df)} avis")
        print(f"📊 Colonnes disponibles : {list(df.columns)}")
        
        # Afficher quelques statistiques
        print("\n📈 Statistiques du dataset :")
        print(f"- Nombre total d'avis : {len(df)}")
        print(f"- Note moyenne : {df['rating'].mean():.2f}")
        print(f"- Distribution des sentiments :")
        print(f"  Positifs (label=1) : {(df['label'] == 1).sum()}")
        print(f"  Neutres (label=0) : {(df['label'] == 0).sum()}")
        print(f"  Négatifs (label=-1) : {(df['label'] == -1).sum()}")
        
        # Afficher quelques exemples
        print("\n📝 Exemples d'avis :")
        for i in range(min(3, len(df))):
            row = df.iloc[i]
            print(f"\n{i+1}. Livre : {row['book_title']}")
            print(f"   Auteur : {row['author']}")
            print(f"   Note : {row['rating']}/5")
            print(f"   Sentiment : {row['label']}")
            print(f"   Avis : {row['reader_review'][:150]}...")
        
        return df
        
    except Exception as e:
        print(f"❌ Erreur lors du téléchargement : {e}")
        return None

def prepare_data_for_analysis(df):
    """Prépare les données pour l'analyse de sentiment multi-sujets."""
    print("\n🔧 Préparation des données pour l'analyse...")
    
    # Renommer les colonnes pour correspondre au pipeline
    df_prepared = df.copy()
    df_prepared = df_prepared.rename(columns={'reader_review': 'text'})
    
    # Ajouter un ID unique
    df_prepared['id'] = range(len(df_prepared))
    
    # Filtrer les avis non vides
    df_prepared = df_prepared[df_prepared['text'].notna() & (df_prepared['text'].str.strip() != '')]
    
    # Limiter à 50 avis pour le test
    df_prepared = df_prepared.head(50)
    
    print(f"✅ Données préparées : {len(df_prepared)} avis prêts pour l'analyse")
    
    return df_prepared

def save_to_excel(df, filename="french_book_reviews_sample.xlsx"):
    """Sauvegarde les données en format Excel."""
    filepath = os.path.join(os.getcwd(), filename)
    df.to_excel(filepath, index=False)
    print(f"💾 Données sauvegardées : {filepath}")
    return filepath

def main():
    """Fonction principale."""
    print("🎯 Téléchargement et préparation des données d'avis français\n")
    
    # Télécharger les données
    df = download_french_book_reviews()
    
    if df is not None:
        # Préparer pour l'analyse
        df_prepared = prepare_data_for_analysis(df)
        
        # Sauvegarder
        filepath = save_to_excel(df_prepared)
        
        print(f"\n🎉 Prêt pour l'analyse !")
        print(f"📂 Fichier créé : {filepath}")
        print(f"▶️  Lancez maintenant : python main_multi_topic.py")
        
        return filepath
    
    return None

if __name__ == "__main__":
    main()