"""Logique NLP : extraction de texte, détection de langue, résumé, traduction."""

import re

import torch
import trafilatura
import networkx as nx
from langdetect import detect, DetectorFactory, LangDetectException
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    MarianMTModel,
    MarianTokenizer,
    pipeline,
)

DetectorFactory.seed = 0

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

_cache_traduction = {}

STOPWORDS_FR = [
    "le", "la", "les", "un", "une", "des", "de", "du", "et", "à", "en",
    "est", "sont", "que", "qui", "dans", "pour", "sur", "avec", "par",
    "ce", "cette", "ces", "il", "elle", "ils", "elles", "au", "aux",
]


# --- Chargement des modèles ---

def charger_modele_resume():
    """Charge le modèle mT5 multilingue (résumé, langues non-anglaises)."""
    nom_modele = "csebuetnlp/mT5_multilingual_XLSum"
    tokenizer = AutoTokenizer.from_pretrained(nom_modele)
    modele = AutoModelForSeq2SeqLM.from_pretrained(nom_modele).to(DEVICE)
    return tokenizer, modele


def charger_modele_bart():
    """Charge BART, spécialisé résumé en anglais."""
    device_id = 0 if DEVICE == "cuda" else -1
    return pipeline("summarization", model="facebook/bart-large-cnn", device=device_id)


def charger_modele_traduction(langue_source, langue_cible):
    """Charge (avec cache) un modèle de traduction Helsinki-NLP."""
    cle = f"{langue_source}-{langue_cible}"
    if cle not in _cache_traduction:
        nom_modele = f"Helsinki-NLP/opus-mt-{langue_source}-{langue_cible}"
        tokenizer = MarianTokenizer.from_pretrained(nom_modele)
        modele = MarianMTModel.from_pretrained(nom_modele).to(DEVICE)
        _cache_traduction[cle] = (tokenizer, modele)
    return _cache_traduction[cle]


# --- Extraction & détection ---

def extraire_texte(url):
    """Télécharge et extrait le texte principal d'une page web."""
    downloaded = trafilatura.fetch_url(url)
    if downloaded is None:
        raise ValueError("Impossible de télécharger la page.")
    texte = trafilatura.extract(
        downloaded, include_tables=False, include_comments=False, favor_precision=True
    )
    if texte is None:
        raise ValueError("Impossible d'extraire le texte de la page.")
    return texte.strip()


def detecter_langue(texte):
    """Détecte la langue d'un texte. Retourne 'fr' par défaut si la détection échoue
    (texte trop court, caractères spéciaux, pas assez de signal textuel)."""
    try:
        return detect(texte[:2000])
    except LangDetectException:
        return "fr"


# --- Résumé : modèle avancé, choix automatique BART (en) / mT5 (autres langues) ---

def resumer_bart(texte, summarizer_bart, max_length=150, min_length=30):
    """Résumé abstractif en anglais avec BART."""
    texte_tronque = texte[:3500]
    resultat = summarizer_bart(
        texte_tronque, max_length=max_length, min_length=min_length, do_sample=False
    )
    return resultat[0]["summary_text"]


def resumer_mt5(texte, tokenizer, modele, langue_source, max_length=150, min_length=30):
    """Résumé abstractif multilingue avec mT5 (langues non-anglaises)."""
    texte_tronque = texte[:4000]
    # test SANS préfixe pour comparer
    entrees = tokenizer(
        texte_tronque, return_tensors="pt", max_length=512, truncation=True
    ).to(DEVICE)
    sortie = modele.generate(
        **entrees,
        max_length=max_length,
        min_length=min_length,
        num_beams=4,
        no_repeat_ngram_size=3,
        length_penalty=1.0,   # remis à neutre (1.0) le temps du test
    )
    return tokenizer.decode(sortie[0], skip_special_tokens=True)


def resumer_avance(texte, langue_source, tokenizer_mt5, modele_mt5, summarizer_bart, max_length=150, min_length=30):
    """Choisit automatiquement BART (anglais) ou mT5 (autres langues)."""
    if langue_source == "en":
        return resumer_bart(texte, summarizer_bart, max_length=max_length, min_length=min_length)
    return resumer_mt5(texte, tokenizer_mt5, modele_mt5, langue_source, max_length=max_length, min_length=min_length)


# --- Résumé : baseline extractive (TextRank / TF-IDF) ---

def decouper_phrases(texte):
    """Découpage en phrases sur ponctuation forte et points-virgules."""
    phrases = re.split(r"(?<=[.!?;])\s+", texte)
    phrases = [p.strip() for p in phrases if len(p.strip()) > 20]
    return phrases


def resumer_baseline_textrank(texte, n_phrases=5, max_caracteres=8000):
    """Résumé extractif TextRank (TF-IDF + graphe de similarité cosinus).
    Le texte source est limité en longueur pour éviter les temps de calcul excessifs
    et les effets de bord sur des documents très longs."""
    texte_limite = texte[:max_caracteres]
    phrases = decouper_phrases(texte_limite)
    if len(phrases) <= n_phrases:
        return " ".join(phrases)

    vectorizer = TfidfVectorizer(stop_words=STOPWORDS_FR)
    matrice = vectorizer.fit_transform(phrases)
    matrice_similarite = cosine_similarity(matrice)

    graphe = nx.from_numpy_array(matrice_similarite)
    scores = nx.pagerank(graphe)

    indices_tries = sorted(scores, key=scores.get, reverse=True)[:n_phrases]
    indices_top_ordonnes = sorted(indices_tries)

    phrases_selectionnees = [phrases[i] for i in indices_top_ordonnes]
    return " ".join(phrases_selectionnees)


# --- Traduction ---

def traduire(texte, langue_source, langue_cible):
    """Traduit un texte (typiquement un résumé) d'une langue vers une autre."""
    if langue_source == langue_cible:
        return texte
    tokenizer, modele = charger_modele_traduction(langue_source, langue_cible)
    entrees = tokenizer(texte, return_tensors="pt", truncation=True, padding=True).to(DEVICE)
    sortie = modele.generate(**entrees, max_length=200)
    return tokenizer.decode(sortie[0], skip_special_tokens=True)