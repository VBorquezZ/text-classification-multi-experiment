import os
import re
import time
import random
import pandas as pd
from threading import local
from tqdm.contrib.concurrent import thread_map

import nltk
from nltk.corpus import wordnet

import nlpaug.augmenter.word as naw

from deep_translator import GoogleTranslator
from translate_df_col import translate_df_column


def _clean_text_for_aug(x):
    if pd.isna(x):
        return None
    s = str(x).strip()
    return s if s else None

def _validate_ratio(ratio_to_replace: float):
    if not (0 <= ratio_to_replace <= 1):
        raise ValueError("ratio_to_replace debe estar entre 0 y 1.")


def _validate_n_generate(N_to_generate: int):
    if not isinstance(N_to_generate, int) or N_to_generate < 1:
        raise ValueError("N_to_generate debe ser un entero >= 1.")
    

############################################
############## Back Translate ##############
############################################


def back_translate_df_sequence(
    df: pd.DataFrame,
    col: str,
    original_lang: str,
    language_sequence: list,
    MAX_WORKERS=min(8, (os.cpu_count() or 4) * 2),
    THROTTLE_PER_REQUEST_SEC=0.05,
    final_suffix="backtrans",
    keep_intermediate=False,
):
    """
    Aplica traducción secuencial:
    original_lang -> language_sequence[0] -> language_sequence[1] -> ... -> original_lang

    Ejemplo:
    original_lang="en", language_sequence=["fr", "de"]
    hace: en -> fr -> de -> en
    """
    if not isinstance(language_sequence, list) or len(language_sequence) == 0:
        raise ValueError("language_sequence debe ser una lista no vacía.")

    df_work = df.copy()
    current_col = col
    current_lang = original_lang

    # ir traduciendo por la secuencia intermedia
    for step_idx, next_lang in enumerate(language_sequence, start=1):
        step_suffix = f"step{step_idx}_{next_lang}"

        df_work, errors = translate_df_column(
            df=df_work,
            col=current_col,
            source_lang=current_lang,
            target_lang=next_lang,
            MAX_WORKERS=MAX_WORKERS,
            THROTTLE_PER_REQUEST_SEC=THROTTLE_PER_REQUEST_SEC,
            suffix=step_suffix,
        )

        current_col = f"{current_col}_{step_suffix}"
        current_lang = next_lang

    # volver al idioma original
    df_work, errors = translate_df_column(
        df=df_work,
        col=current_col,
        source_lang=current_lang,
        target_lang=original_lang,
        MAX_WORKERS=MAX_WORKERS,
        THROTTLE_PER_REQUEST_SEC=THROTTLE_PER_REQUEST_SEC,
        suffix=final_suffix,
    )

    final_col = f"{current_col}_{final_suffix}"

    if keep_intermediate:
        return df_work, final_col, errors

    # devolver solo original + columna final traducida
    cols_to_keep = list(df.columns) + [final_col]
    out = df_work[cols_to_keep].copy()
    out = out.rename(columns={final_col: f"{col}_{final_suffix}"})
    return out, errors

def augment_df_with_backtranslation(
    df: pd.DataFrame,
    text_col: str,
    original_lang: str,
    language_paths: list,
    keep_original: bool = True,
    MAX_WORKERS=min(12, (os.cpu_count() or 4) * 2),
    THROTTLE_PER_REQUEST_SEC=0.05,
):
    """
    language_paths: lista de rutas intermedias.
    Ejemplo:
    [
        ["fr"],
        ["de"],
        ["fr", "de"],
        ["it", "pt"]
    ]

    Genera una nueva variante por cada path.
    """
    df_base = df.copy()
    df_base[text_col] = df_base[text_col].map(_clean_text_for_aug)
    df_base = df_base.dropna(subset=[text_col]).reset_index(drop=True)

    augmented_parts = []
    total_errors = []

    if keep_original:
        original_df = df_base.copy()
        original_df["aug_strategy"] = "original"
        original_df["aug_source_index"] = original_df.index
        original_df["aug_iteration"] = 0
        augmented_parts.append(original_df)

    for i, lang_seq in enumerate(language_paths, start=1):
        bt_df, errors = back_translate_df_sequence(
            df=df_base,
            col=text_col,
            original_lang=original_lang,
            language_sequence=lang_seq,
            MAX_WORKERS=MAX_WORKERS,
            THROTTLE_PER_REQUEST_SEC=THROTTLE_PER_REQUEST_SEC,
            final_suffix=f"bt_{i}",
            keep_intermediate=False,
        )
        total_errors.append(errors)
        new_text_col = f"{text_col}_bt_{i}"

        part = df_base.copy()
        part[text_col] = bt_df[new_text_col]
        part["aug_strategy"] = "back_translation"
        part["aug_source_index"] = part.index
        part["aug_iteration"] = i
        part["aug_lang_path"] = "->".join([original_lang] + lang_seq + [original_lang])
        augmented_parts.append(part)

    df_aug = pd.concat(augmented_parts, ignore_index=True)
    df_aug = df_aug.drop_duplicates(subset=[text_col] + [c for c in df.columns if c != text_col]).reset_index(drop=True)
    return df_aug, total_errors


############################################
########### Synonym Replacement ############
############################################


_WORD_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)

def tokenize_preserve_punct(text: str):
    return _WORD_RE.findall(text)

def detokenize_preserve_punct(tokens):
    text = " ".join(tokens)
    # arreglos simples de espacios antes de puntuación
    text = re.sub(r"\s+([,.!?;:])", r"\1", text)
    text = re.sub(r"\(\s+", "(", text)
    text = re.sub(r"\s+\)", ")", text)
    return text.strip()

def get_synonyms_wordnet(word: str):
    """
    Devuelve sinónimos candidatos en inglés usando WordNet.
    Excluye el mismo término y normaliza underscores -> spaces.
    """
    synonyms = set()

    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            candidate = lemma.name().replace("_", " ").strip()
            if candidate.lower() != word.lower():
                synonyms.add(candidate)

    return list(synonyms)

def synonym_replace_text(text: str, ratio_to_replace: float, random_state=None):
    """
    Reemplaza aleatoriamente una fracción de palabras por sinónimos.
    """
    _validate_ratio(ratio_to_replace)

    rng = random.Random(random_state)
    tokens = tokenize_preserve_punct(text)

    # elegimos solo tokens alfabéticos simples para reemplazo
    candidate_positions = [
        i for i, tok in enumerate(tokens)
        if tok.isalpha()
    ]

    if not candidate_positions:
        return text

    n_to_replace = max(1, int(len(candidate_positions) * ratio_to_replace)) if ratio_to_replace > 0 else 0
    if n_to_replace == 0:
        return text

    positions = rng.sample(candidate_positions, min(n_to_replace, len(candidate_positions)))
    new_tokens = tokens.copy()

    for pos in positions:
        original = tokens[pos]
        synonyms = get_synonyms_wordnet(original)

        # filtramos candidatos razonables
        synonyms = [
            s for s in synonyms
            if s.lower() != original.lower()
            and len(s.strip()) > 0
        ]

        if synonyms:
            replacement = rng.choice(synonyms)

            # preservar mayúscula inicial si aplica
            if original[0].isupper():
                replacement = replacement.capitalize()

            new_tokens[pos] = replacement

    return detokenize_preserve_punct(new_tokens)

def augment_df_with_synonym_replacement(
    df: pd.DataFrame,
    text_col: str,
    ratio_to_replace: float,
    N_to_generate: int,
    keep_original: bool = True,
    random_state: int = 42,
):
    """
    Genera N nuevas variantes por fila usando synonym replacement.
    Devuelve un dataframe aumentado.
    """
    _validate_ratio(ratio_to_replace)
    _validate_n_generate(N_to_generate)

    df_base = df.copy()
    df_base[text_col] = df_base[text_col].map(_clean_text_for_aug)
    df_base = df_base.dropna(subset=[text_col]).reset_index(drop=True)

    augmented_rows = []

    if keep_original:
        original_df = df_base.copy()
        original_df["aug_strategy"] = "original"
        original_df["aug_source_index"] = original_df.index
        original_df["aug_iteration"] = 0
        augmented_rows.append(original_df)

    for row_idx, row in df_base.iterrows():
        original_text = row[text_col]

        for n in range(1, N_to_generate + 1):
            seed = random_state + row_idx * 1000 + n
            new_text = synonym_replace_text(
                original_text,
                ratio_to_replace=ratio_to_replace,
                random_state=seed,
            )

            new_row = row.copy()
            new_row[text_col] = new_text
            new_row["aug_strategy"] = "synonym_replacement"
            new_row["aug_source_index"] = row_idx
            new_row["aug_iteration"] = n
            augmented_rows.append(pd.DataFrame([new_row]))

    df_aug = pd.concat(augmented_rows, ignore_index=True)

    df_aug = df_aug.drop_duplicates(subset=[text_col] + [c for c in df.columns if c != text_col]).reset_index(drop=True)
    return df_aug


############################################
###### Contextual Replaces & Inserts #######
############################################


def build_contextual_augmenter(
    model_path: str = "distilbert-base-uncased",
    action: str = "substitute",
    aug_p: float = 0.1,
    device: str = "cuda" if __import__("torch").cuda.is_available() else "cpu",
):
    """
    action: 'substitute' o 'insert'
    aug_p: proporción aproximada de tokens a modificar
    """
    if action not in {"substitute", "insert"}:
        raise ValueError("action debe ser 'substitute' o 'insert'.")

    aug = naw.ContextualWordEmbsAug(
        model_path=model_path,
        action=action,
        aug_p=aug_p,
        device=device,
    )
    return aug


def augment_df_with_contextual_augmentation(
    df: pd.DataFrame,
    text_col: str,
    ratio_to_replace: float,
    N_to_generate: int,
    action: str = "substitute",   # "substitute" o "insert"
    model_path: str = "distilbert-base-uncased",
    keep_original: bool = True,
):
    """
    Genera variantes usando contextual augmentation.
    """
    _validate_ratio(ratio_to_replace)
    _validate_n_generate(N_to_generate)

    df_base = df.copy()
    df_base[text_col] = df_base[text_col].map(_clean_text_for_aug)
    df_base = df_base.dropna(subset=[text_col]).reset_index(drop=True)

    aug = build_contextual_augmenter(
        model_path=model_path,
        action=action,
        aug_p=ratio_to_replace,
    )

    augmented_rows = []

    if keep_original:
        original_df = df_base.copy()
        original_df["aug_strategy"] = "original"
        original_df["aug_source_index"] = original_df.index
        original_df["aug_iteration"] = 0
        augmented_rows.append(original_df)

    for row_idx, row in df_base.iterrows():
        original_text = row[text_col]

        for n in range(1, N_to_generate + 1):
            try:
                new_text = aug.augment(original_text)

                # nlpaug a veces devuelve lista
                if isinstance(new_text, list):
                    new_text = new_text[0] if len(new_text) > 0 else original_text

            except Exception:
                new_text = original_text

            new_row = row.copy()
            new_row[text_col] = new_text
            new_row["aug_strategy"] = f"contextual_{action}"
            new_row["aug_source_index"] = row_idx
            new_row["aug_iteration"] = n
            augmented_rows.append(pd.DataFrame([new_row]))

    df_aug = pd.concat(augmented_rows, ignore_index=True)
    df_aug = df_aug.drop_duplicates(subset=[text_col] + [c for c in df.columns if c != text_col]).reset_index(drop=True)
    return df_aug