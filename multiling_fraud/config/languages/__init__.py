"""Language packs. Each `config/languages/<code>.py` bundles everything language-specific
(display name, scene labels, fraud-type labels, task templates + record builders, TTS voices)
so the core pipeline stays language-agnostic. Add a new language by adding a pack here."""
import importlib


def get_language(code):
    try:
        return importlib.import_module(f"config.languages.{code}")
    except ModuleNotFoundError as e:
        raise SystemExit(
            f"No language pack for '{code}'. Create config/languages/{code}.py "
            f"(copy en.py and translate the labels/templates)."
        ) from e
