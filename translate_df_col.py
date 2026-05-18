import pandas as pd
import time
import random
from threading import local
from deep_translator import GoogleTranslator
from tqdm.contrib.concurrent import thread_map
import os 

def translate_df_column(
    df,
    col="contenido",
    source_lang="auto",
    target_lang="en",
    MAX_WORKERS=min(12, (os.cpu_count() or 4) * 2),
    THROTTLE_PER_REQUEST_SEC=0.05,
    suffix=None,
    return_original_on_error=False,
):
    if suffix is None:
        suffix = target_lang

    translation_map = {}

    unique_text = (
        df[col]
        .dropna()
        .astype(str)
        .map(str.strip)
        .replace("", pd.NA)
        .dropna()
        .unique()
    )

    to_translate = [t for t in unique_text if t not in translation_map]

    _thread = local()

    def get_translator():
        if not hasattr(_thread, "translator"):
            _thread.translator = GoogleTranslator(
                source=source_lang,
                target=target_lang,
            )
        return _thread.translator

    def translate_one(txt, retries=3, base_delay=0.7, throttle=0.0):
        if throttle > 0:
            time.sleep(throttle)

        last_error = None
        for attempt in range(retries + 1):
            try:
                tr = get_translator().translate(txt)
                return txt, tr, None
            except Exception as e:
                last_error = e
                if attempt == retries:
                    if return_original_on_error:
                        return txt, txt, repr(last_error)
                    return txt, pd.NA, repr(last_error)

                time.sleep(base_delay * (2 ** attempt) + random.random() * 0.2)

    pairs = thread_map(
        lambda t: translate_one(
            t,
            retries=3,
            base_delay=0.7,
            throttle=THROTTLE_PER_REQUEST_SEC,
        ),
        to_translate,
        max_workers=MAX_WORKERS,
        desc=f"Translating {source_lang}->{target_lang}",
    )

    errors = {}
    for txt, tr, err in pairs:
        translation_map[txt] = tr
        if err is not None:
            errors[txt] = err

    def translate_or_none(x):
        if pd.isna(x):
            return pd.NA
        s = str(x).strip()
        if not s:
            return pd.NA
        return translation_map.get(s, pd.NA)

    df_result = df.copy()
    df_result[f"{col}_{suffix}"] = df_result[col].map(translate_or_none)

    return df_result, errors