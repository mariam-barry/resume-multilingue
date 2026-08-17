"""Interface Streamlit de l'assistant de résumé multilingue."""

import streamlit as st

from back import (
    charger_modele_resume,
    charger_modele_bart,
    extraire_texte,
    detecter_langue,
    resumer_avance,
    resumer_baseline_textrank,
    traduire,
)

st.set_page_config(page_title="Résumé multilingue", page_icon="📝")
st.title("📝 Assistant de résumé multilingue")

onglet_app, onglet_comparaison = st.tabs(["Résumer un article", "📊 Comparaison des performances"])


@st.cache_resource
def get_modeles():
    tokenizer_mt5, modele_mt5 = charger_modele_resume()
    summarizer_bart = charger_modele_bart()
    return tokenizer_mt5, modele_mt5, summarizer_bart


tokenizer_mt5, modele_mt5, summarizer_bart = get_modeles()

CODES_LANGUE = {"Français": "fr", "Anglais": "en", "Espagnol": "es"}
longueurs_params_avance = {"Court": (40, 15), "Moyen": (120, 60), "Long": (250, 130)}
longueurs_params_baseline = {"Court": 3, "Moyen": 5, "Long": 8}

with onglet_app:
    st.write("Colle l'URL d'un article, choisis la méthode, la langue et la longueur du résumé.")

    url = st.text_input("URL de l'article")

    comparer = st.checkbox("Comparer baseline vs modèle avancé côte à côte", value=True)

    col1, col2 = st.columns(2)
    with col1:
        longueur = st.select_slider("Longueur du résumé", options=["Court", "Moyen", "Long"], value="Moyen")
    with col2:
        langue_sortie_label = st.selectbox(
            "Langue de sortie", ["Automatique (langue source)"] + list(CODES_LANGUE.keys())
        )

    if st.button("Résumer", type="primary"):
        if not url:
            st.warning("Merci de coller une URL.")
            st.stop()

        with st.spinner("Extraction du texte..."):
            try:
                texte = extraire_texte(url)
            except Exception as e:
                st.error(f"Erreur d'extraction : {e}")
                st.stop()

        with st.spinner("Détection de la langue..."):
            langue_source = detecter_langue(texte)
            modele_utilise = "BART (anglais)" if langue_source == "en" else "mT5 (multilingue)"
            st.info(f"Langue source détectée : **{langue_source}** → modèle avancé utilisé : **{modele_utilise}**")

        max_len, min_len = longueurs_params_avance[longueur]
        n_phrases = longueurs_params_baseline[longueur]

        with st.spinner("Génération du résumé avancé..."):
            resume_avance = resumer_avance(
                texte, langue_source, tokenizer_mt5, modele_mt5, summarizer_bart,
                max_length=max_len, min_length=min_len,
            )

        if comparer:
            with st.spinner("Génération de la baseline..."):
                resume_baseline = resumer_baseline_textrank(texte, n_phrases=n_phrases)

            col_a, col_b = st.columns(2)
            with col_a:
                st.subheader("🔹 Baseline (TextRank)")
                st.write(resume_baseline)
                st.caption(f"{len(resume_baseline.split())} mots")
            with col_b:
                st.subheader(f"🔸 Modèle avancé ({modele_utilise})")
                st.write(resume_avance)
                st.caption(f"{len(resume_avance.split())} mots")

            resume_final = resume_avance
        else:
            st.subheader("Résumé")
            st.write(resume_avance)
            st.caption(f"{len(resume_avance.split())} mots — modèle : {modele_utilise}")
            resume_final = resume_avance

        if langue_sortie_label != "Automatique (langue source)":
            langue_cible = CODES_LANGUE[langue_sortie_label]
            if langue_cible != langue_source:
                with st.spinner(f"Traduction vers {langue_sortie_label}..."):
                    try:
                        traduction = traduire(resume_final, langue_source, langue_cible)
                        st.subheader(f"Traduction ({langue_sortie_label})")
                        st.write(traduction)
                    except Exception as e:
                        st.warning(f"Traduction indisponible ({langue_source}→{langue_cible}) : {e}")

        with st.expander("Voir le texte source extrait"):
            st.write(texte[:3000] + "...")

with onglet_comparaison:
    st.write(
        "Résultats de l'évaluation ROUGE réalisée dans le notebook, sur un corpus de "
        "5 articles Wikipedia (français, anglais, espagnol), comparant la baseline "
        "TextRank et le modèle avancé (mT5)."
    )

    import pandas as pd

    # Valeurs reprises telles quelles depuis l'évaluation faite dans le notebook (étape 8bis, v1)
    donnees_comparaison = pd.DataFrame({
        "Métrique": ["ROUGE-1 (F1)", "ROUGE-L (F1)"],
        "Baseline (TextRank)": [0.127, 0.077],
        "Modèle avancé (mT5)": [0.371, 0.340],
    })
    st.dataframe(donnees_comparaison, use_container_width=True)

    st.bar_chart(donnees_comparaison.set_index("Métrique"))

    st.caption(
        "Le modèle avancé obtient un score ROUGE-1 environ 3x supérieur à la baseline, "
        "confirmant la meilleure qualité des résumés abstractifs par rapport à l'extraction "
        "statistique simple."
    )