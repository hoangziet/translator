from app.model.translator import (
    loaded_translator, 
    translate as translate_fn
)

def translate_text(text: str):
    """
    Translate the text using the loaded translator model.

    Args:
        text (str): The text to be translated.

    Returns:
        str: The translated text.
    """
    translation, logit, tokens = translate_fn(loaded_translator, text)
    return translation