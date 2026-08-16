# AI Travel Assistant

Assistant conversationnel en français pour une compagnie aérienne, réalisé pour le challenge technique. Il combine une base documentaire RAG et des outils métier pour fournir des réponses traçables.

## Démarrage rapide

Prérequis : Python 3.11+ et une clé API OpenRouter personnelle.

```powershell
pip install -r requirements.txt
Copy-Item .env.example .env
```

Dans `.env`, renseigner une clé créée depuis son propre compte OpenRouter. La
page de gestion et de création des clés est disponible sur
[OpenRouter Keys](https://openrouter.ai/keys) (connexion requise) :

```env
OPENROUTER_API_KEY=your_key_here
OPENROUTER_MODEL=openrouter/free
LLM_REQUEST_TIMEOUT_SECONDS=30
```

Lancer ensuite l'API :

```powershell
uvicorn main:app
```

Ouvrir `demo.html` dans le navigateur, puis poser une question. La documentation
interactive de l'API est également disponible sur <http://127.0.0.1:8000/docs>.

> Ne jamais partager ni versionner le fichier `.env`. Il est exclu par `.gitignore`.

## Fonctionnalités couvertes

- Questions générales avec RAG : bagages, remboursement, enregistrement, documents, modification de vol et assistance.
- Function calling : `search_flights`, `get_flight_status`, `get_airport_info` et `get_booking`.
- Cas combiné : une question sur l'annulation d'un vol et le remboursement consulte obligatoirement le statut et la politique documentaire.
- Mémoire de conversation par `conversation_id`, conservée 30 minutes en RAM.
- API FastAPI, interface de démonstration (`demo.html`) et source affichée pour chaque réponse : `rag`, le nom de l'outil, ou les deux.

## Architecture

```text
demo.html ou client HTTP
          │ POST /chat
          ▼
       main.py ── validation, CORS, sessions et erreurs HTTP
          ▼
       llm.py ── routage, garde-fous, LLM et function calling
        ├── rag.py ── ChromaDB + knowledge_base/*.txt
        └── tools.py ── vols, statuts, aéroports, réservations (mocks)
```

| Fichier | Responsabilité |
| --- | --- |
| `main.py` | API FastAPI, validation, CORS, logs et endpoint `/health`. |
| `llm.py` | Prompt, routage, appels au modèle génératif et exécution des outils. |
| `rag.py` | Indexation, découpage et recherche sémantique des règles de voyage. |
| `tools.py` | Simulations déterministes d'APIs métier. |
| `session_store.py` | Historique court par conversation, protégé par verrou. |
| `knowledge_base/*.txt` | Source de vérité des règles métier de démonstration. |

## Choix techniques

### RAG, documents et embeddings

Les règles de bagages ou de remboursement doivent rester ancrées dans des documents éditables, plutôt que dans la mémoire du modèle. Les 7 fichiers `.txt` de `knowledge_base/` sont découpés par paragraphe puis indexés dans ChromaDB.

Une fonction d'embedding locale de Chroma transforme chaque passage en vecteur numérique représentant son sens. La question utilisateur est transformée de la même manière afin de retrouver les passages les plus proches. Ces extraits sont ensuite donnés au modèle génératif pour rédiger la réponse. Le modèle d'embedding ne rédige pas de texte et aucun appel à une API d'embeddings externe n'est nécessaire.

L'index est persistant dans `chroma_db/`. Une empreinte des documents est enregistrée : si un fichier `.txt` change, l'index est reconstruit automatiquement au démarrage.

### Modèle génératif

Le modèle génératif est appelé via OpenRouter. Il lit la question, l'historique et les données issues du RAG ou des outils, rédige la réponse en français et peut demander l'appel d'une fonction via le function calling. Cette séparation permet de modifier une politique dans un fichier `.txt` sans réentraîner le modèle.

### Routage et fiabilité

Le routage combine des règles serveur et le function calling du LLM :

1. Les règles générales vont vers le RAG : bagages, remboursement, enregistrement, documents et assistance.
2. Les données dynamiques vont vers un outil : disponibilités, statut, réservation et aéroport.
3. Les cas critiques sont déterministes. Par exemple, « rembourser le vol AH1009 » impose `get_flight_status` et le RAG avant la génération.
4. Le modèle reçoit les résultats vérifiés et ne doit pas les contredire. Une seconde demande du même outil réutilise le résultat déjà vérifié.

Ainsi, les appels inutiles sont évités et les hallucinations sont limitées. Numéro de vol, référence de réservation, code aéroport ou date manquants ne sont jamais inventés. Une date incomplète comme `15 août` déclenche une demande de précision ; une date complète ou relative (`demain`) est acceptée.

Lorsqu'un numéro de vol est fourni sans date, le statut porte sur la date serveur du jour et cette convention est annoncée. Les résultats de statut et de recherche de vols sont déterministes pour une même requête : une même question ne change pas de réponse au hasard.

## Installation et configuration

```powershell
pip install -r requirements.txt
Copy-Item .env.example .env
```

Renseigner une configuration LLM dans `.env` :

```env
OPENROUTER_API_KEY=your_key_here
OPENROUTER_MODEL=openrouter/free
LLM_REQUEST_TIMEOUT_SECONDS=30
```

Le fichier `.env` reste local et ne doit jamais être commité.

## Lancement

Pour la démonstration, lancer sans rechargement automatique afin que la création de l'index Chroma ne rafraîchisse pas la page :

```powershell
uvicorn main:app
```

Pendant le développement :

```powershell
uvicorn main:app --reload
```

Ouvrir `demo.html` ou consulter la documentation interactive sur <http://127.0.0.1:8000/docs>. L'état de l'API est disponible sur <http://127.0.0.1:8000/health>.

## Contrat API

`POST /chat`

```json
{
  "message": "Quel est le statut du vol AH1235 aujourd'hui ?",
  "conversation_id": "optionnel"
}
```

```json
{
  "answer": "...",
  "source": "get_flight_status",
  "conversation_id": "..."
}
```

Erreurs gérées : `422` pour une question vide ou trop longue, `503` lorsque le fournisseur LLM est indisponible et `500` générique pour une erreur interne.

## Tests et validation

La suite contient **29 tests automatisés** :

| Couverture | Nombre | Vérifications |
| --- | ---: | --- |
| Outils métier | 14 | Validation, données inconnues, cohérence et déterminisme des vols. |
| RAG | 2 | Recherche de bagages et absence de résultat documentaire. |
| API et robustesse | 3 | Nouvelle conversation, fournisseur indisponible, endpoint `/health`. |
| Protocole et routage | 6 | Format standard des messages d'outil, RAG + outil, dates, réservation et aéroport. |
| Intégration LLM réelle | 4 | RAG, outil, cas combiné et mémoire de conversation. |

Tests locaux, sans clé API :

```powershell
pytest tests/ -v
```

Par défaut, cette commande exécute 25 tests locaux et marque les 4 tests
d'intégration LLM comme `skipped`. Ce comportement est volontaire : il évite
un appel réseau et la consommation éventuelle de crédits OpenRouter.

Tests utilisant le fournisseur LLM configuré :

```powershell
$env:RUN_LLM_INTEGRATION_TESTS="1"
pytest tests/test_app.py -v -k "chat_endpoint"
```

Dernière validation effectuée : **25 tests locaux réussis**. Avec la clé LLM configurée, la commande d'intégration a exécuté **6 tests réussis**, dont les **4 scénarios avec un vrai LLM**. Les deux autres contrôlent une nouvelle conversation et l'erreur `503`.

## Scénarios de démonstration

| Question à tester | Source attendue |
| --- | --- |
| `Quels sont les bagages autorisés en cabine ?` | `rag` |
| `Trouve-moi un vol Paris-Alger demain` | `search_flights` |
| `Quel est le statut du vol AH1235 aujourd'hui ?` | `get_flight_status` |
| `Donne-moi les informations de la réservation ABC123` | `get_booking` |
| `Mon vol AH1235 est annulé. Puis-je demander un remboursement ?` | `rag + get_flight_status` |
| `Donne-moi des informations sur l'aéroport CDG` | `get_airport_info` |
| `Quel est le statut du vol AH1009 le 16 août ?` | demande l'année |
| `Donne-moi les informations de la réservation XYZ789` | réservation introuvable |

Données mock disponibles : `ABC123` est la seule réservation de démonstration ; `CDG` et `ALG` sont les seuls aéroports documentés par l'outil mock.

## Limites et passage en production

- Les vols, aéroports et réservations sont des données mockées. En production, les outils appelleraient les APIs métier avec authentification et gestion des délais/erreurs.
- La mémoire est en RAM : elle n'est ni persistante ni partagée entre plusieurs instances. Redis avec expiration conviendrait à une version distribuée.
- L'application ne contient pas encore d'authentification ni de limitation de débit ; elles seraient indispensables avec des données clients réelles.
- En production, définir `APP_ENV=production` et `ALLOWED_ORIGINS` avec les domaines exacts du frontend.
- Les réponses restent générées par un LLM : les garde-fous et les sources affichées réduisent le risque d'erreur, mais une validation métier reste nécessaire pour une décision financière réelle.
