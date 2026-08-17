# Assistant de résumé multilingue

Application Streamlit qui résume des articles web (via URL) en français, anglais ou espagnol,
avec choix de la méthode (baseline TextRank ou modèle avancé mT5), de la longueur et de la
langue de sortie.

## Installation locale

\`\`\`bash
python -m venv venv
source venv/bin/activate  # Windows : venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
\`\`\`

## Structure du projet

- `app.py` — interface Streamlit
- `back.py` — logique NLP (extraction, résumé, traduction)
- `notebook.ipynb` — exploration, baseline, modèle avancé, évaluation ROUGE
- `requirements.txt` — dépendances

## Modèles utilisés

- Résumé abstractif : `csebuetnlp/mT5_multilingual_XLSum`
- Résumé extractif (baseline) : TF-IDF + TextRank (scikit-learn + networkx)
- Traduction : `Helsinki-NLP/opus-mt-*`

## Évaluation

Voir `notebook.ipynb` pour l'analyse ROUGE-1/2/L comparant la baseline et le modèle avancé
sur un corpus de 5 articles (français, anglais, espagnol), ainsi que l'analyse des erreurs
(hallucinations observées avec mT5 sur du contenu encyclopédique).