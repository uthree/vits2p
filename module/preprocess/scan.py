import json
from pathlib import Path
from module.g2p import G2PModule


# update metadata
def scan_cache(config):
    cache_dir = Path(config.preprocess.cache_dir) / "vits"
    models_dir = Path(config.save.models_dir)
    metadata_path = models_dir / "metadata.json"
    if not models_dir.exists():
        models_dir.mkdir()
    speakers = []
    for subdir in cache_dir.glob("*"):
        if subdir.is_dir():
            speakers.append(subdir.name)
    speakers = sorted(speakers)
    g2p = G2PModule()
    phonemes = g2p.phonemes
    languages = g2p.languages()
    metadata = {
        "speakers": speakers,
        "phonemes": phonemes,
        "languages": languages,
        "n_fft": config.preprocess.n_fft,
        "frame_size": config.preprocess.frame_size
    }
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f)