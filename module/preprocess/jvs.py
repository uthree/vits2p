from pathlib import Path
from tqdm import tqdm
from .processor import Preprocessor


def process_category(path: Path, category: str, processor: Preprocessor, speaker_name, counter):
    print(f"Ppocessing {str(path)}")
    audio_dir = path / "wav24kHz16bit"
    transcription_path = path / "transcripts_utf8.txt"
    with open(transcription_path, encoding='utf-8') as f:
        transcription_text = f.read()

    for metadata in tqdm(transcription_text.split("\n")):
        s = metadata.split(":")
        if len(s) >= 2:
            audio_file_name, transcription = s[0], s[1]
            audio_file_path = audio_dir / (audio_file_name + ".wav")
            if not audio_file_path.exists():
                continue
            processor.write_cache(
                audio_file_path,
                transcription,
                'ja',
                speaker_name,
                counter
            )
            counter += 1
    return counter

def preprocess_jvs(source_root_path: Path, config):
    processor = Preprocessor(config)
    for subdir in source_root_path.glob("*/"):
        if subdir.is_dir():
            print(f"Processing {subdir}")
            speaker_name = subdir.name
            counter = 0
            counter = process_category(subdir / "nonpara30", "nonpara30", processor, speaker_name, counter)
            counter = process_category(subdir / "parallel100", "paralell100", processor, speaker_name, counter)