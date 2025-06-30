import pandas as pd
import random

# Fake French reviews - messy and covering multiple topics
fake_reviews = [
    "Bonjour, je vais souvent dans ce magasin de proximité et franchement les prix sont un peu élevés mais bon le personnel est sympa surtout la caissière du matin. Par contre les fruits et légumes ne sont pas toujours frais, hier j'ai acheté des pommes qui étaient un peu pourries. Sinon pour le pain c'est correct, la baguette est bonne le matin.",
    
    "Alors moi je trouve que c'est pas mal ce petit commerce, pratique pour dépanner. Mais alors les horaires c'est n'importe quoi ! Ils ferment à 19h30 alors que c'est marqué 20h sur la porte. Et le dimanche ils ouvrent pas... Le gérant pourrait faire un effort ! Ah et puis la station essence à côté c'est pratique aussi.",
    
    "Bof bof ce magasin... D'abord c'est pas propre, les rayons sont mal achalandés, y'a jamais ce qu'on cherche. Les produits sont souvent périmés aussi. Et puis le drive ça marche jamais, on attend 20 minutes pour rien. Même Uber Eats c'est mieux ! Au moins ça livre.",
    
    "Moi j'aime bien ce supérette, l'accueil est toujours sympa. Le boucher est super gentil et sa viande est de qualité. Par contre les prix... ohlala c'est cher ! Surtout les glaces au frigo, 6 euros la boîte c'est abusé. Mais bon pour le quartier c'est pratique.",
    
    "Catastrophe cette épicerie ! Déjà à la caisse ils font des erreurs sur les tickets, faut toujours vérifier. Ensuite le personnel est pas aimable du tout, surtout le patron qui est désagréable. Et puis quand on va déposer un colis au point relais, ils savent jamais où ça se trouve. C'est du n'importe quoi !",
    
    "Bon alors ce petit commerce, c'est pas le top mais ça dépanne. Les viennoiseries sont correctes le matin, la boulangerie fait l'affaire. Mais les horaires d'ouverture c'est pas clair, parfois c'est fermé alors que ça devrait être ouvert. Et puis le choix c'est limité dans les rayons.",
    
    "Super pratique ce magasin ! L'équipe est géniale, toujours souriante. Les produits froids sont bien conservés, le congélateur marche bien. Et puis ils ont installé une station essence récemment, c'est top pour faire le plein. Seul bémol : les prix sont un peu élevés mais bon c'est le commerce de proximité.",
    
    "Alors franchement c'est pas terrible... La propreté laisse à désirer, surtout près des légumes. Les pompes à essence sont toujours en panne. Et puis le drive c'est une catastrophe, on commande sur Uber Eats c'est plus simple. Au moins on a pas à supporter le personnel qui est pas sympa.",
    
    "Moi j'y vais régulièrement dans cette supérette. C'est vrai que c'est pas donné, les prix sont élevés mais l'accueil est chaleureux. La caissière me connaît maintenant. Et puis pour récupérer les colis Mondial Relais c'est pratique, ils sont ouverts tard. Juste dommage que le pain soit pas toujours frais.",
    
    "Bonne surprise ce petit magasin ! Le gérant est sympa, il connaît tous ses clients. Les fruits et légumes sont frais, pas comme dans les grosses surfaces. Et puis la boucherie est excellente, le boucher connaît son métier. Seul problème : ils ferment trop tôt, impossible de faire ses courses après le travail.",
    
    "Alors là c'est le pire magasin du quartier ! D'abord c'est pas propre, les sols sont sales. Ensuite les horaires sont fantaisistes, des fois c'est fermé pour rien. Et puis le personnel... alors là c'est le pompon ! Ils sont désagréables, surtout à la caisse. Et on parle même pas des prix qui sont abusés !",
    
    "Bon petit commerce de proximité. Pratique pour dépanner quand on oublie quelque chose. Les viennoiseries du matin sont bonnes, la baguette aussi. Et puis ils ont un point relais, c'est utile pour les colis. Juste dommage que les rayons soient un peu vides parfois et que ça manque de choix.",
    
    "Franchise de quartier correcte. L'accueil est sympa, le personnel sourit. Les produits frais sont bien conservés dans le frigo. Et puis la poissonnerie est pas mal, le poisson est frais. Seul truc qui m'énerve : les horaires ! Ils ferment à 19h le samedi, c'est trop tôt. Et le dimanche c'est fermé évidemment.",
    
    "Catastrophique ce magasin ! Déjà les prix sont hors de prix, c'est du vol ! Ensuite la propreté... n'en parlons pas, c'est dégueulasse. Et puis quand on veut déposer un colis, ils savent jamais comment faire. Même le drive marche pas, on attend des heures pour rien. Vraiment à éviter !",
    
    "Plutôt satisfait de cette épicerie. C'est vrai que c'est un peu cher mais c'est le prix de la proximité. L'équipe est sympa, surtout la patronne qui est toujours souriante. Les glaces au congélateur sont de bonne qualité. Et puis ils ont installé des pompes à essence, c'est pratique pour faire le plein.",
    
    "Alors moi je recommande ce petit commerce ! Déjà l'accueil est top, ils connaissent tous leurs clients. Ensuite les produits sont frais, surtout les fruits et légumes. Et puis le pain de la boulangerie est excellent, encore chaud le matin. Seul bémol : les horaires du dimanche, ils ouvrent que le matin.",
    
    "Bof ce magasin... Les rayons sont mal achalandés, y'a pas grand chose à manger. Les produits sont souvent périmés, faut faire attention aux dates. Et puis le personnel à la caisse fait des erreurs sur les tickets. Même pour récupérer un colis au point relais c'est compliqué. Vraiment pas terrible.",
    
    "Super ce petit commerce ! Pratique et utile pour le quartier. Les prix sont corrects, pas trop élevés. Et puis l'équipe est géniale, toujours de bonne humeur. La station essence marche bien, jamais en panne. Et puis ils font Uber Eats maintenant, c'est pratique pour se faire livrer.",
    
    "Alors franchement c'est moyen... La propreté c'est pas ça, surtout près de la boucherie. Les horaires sont pas respectés, des fois c'est fermé pour rien. Et puis le gérant est pas sympa du tout, il fait la gueule. Même les produits froids sont pas bien conservés, le frigo fait du bruit.",
    
    "Excellent ce magasin de proximité ! L'accueil est chaleureux, le personnel connaît tous les clients. Les viennoiseries sont délicieuses, la boulangerie fait du bon travail. Et puis le choix est correct dans les rayons, on trouve ce qu'on cherche. Juste dommage que les prix soient un peu élevés mais bon c'est normal."
]

# Create DataFrame
df = pd.DataFrame({
    'text': fake_reviews,
    'id': range(1, len(fake_reviews) + 1)
})

# Save to Excel
df.to_excel('fake_reviews_dataset.xlsx', index=False)
print(f"Dataset créé avec {len(fake_reviews)} avis clients en français")
print("Fichier sauvegardé : fake_reviews_dataset.xlsx")