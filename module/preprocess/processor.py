import torch
import torch.nn.functional as F
from module.g2p import G2PModule
from pathlib import Path
import torchaudio
from torchaudio.functional import resample
from module.utils.f0_estimation import estimate_f0


class Preprocessor:
    def __init__(self, config):
        self.g2p = G2PModule()
        self.sample_rate = config.sample_rate
        self.frame_size = config.frame_size
        self.n_fft = config.n_fft
        self.spec_max_length = config.spec_max_length
        self.text_max_length = config.text_max_length
        self.cache_dir = Path(config.cache_dir)

    def write_cache(self, audio_path: Path, text: str, language: str, speaker: str, data_id: int):
        wf, sr = torchaudio.load(audio_path)
        if sr != self.sample_rate:
            wf = resample(wf, sr, self.sample_rate)

        wf = wf.sum(dim=0, keepdim=True)

        # adjust length
        length = self.frame_size * self.spec_max_length
        spec_length = torch.LongTensor([wf.shape[1] // self.frame_size])
        if wf.shape[1] < length:
            pad_length = length - wf.shape[1]
            pad = torch.zeros(1, pad_length)
            wf = torch.cat([wf, pad], dim=1)
        if wf.shape[1] > length:
            wf = wf[:, :length]

        # f0 estimation
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        f0 = estimate_f0(wf.to(device), self.sample_rate, self.frame_size).cpu()

        text, text_length, language_id = self.g2p.encode(text, language, self.text_max_length)
        data = {
            "spec_length": spec_length,
            "text": text,
            "text_length": text_length,
            "language_id": language_id,
            "f0": f0
        }
        if not self.cache_dir.exists():
            self.cache_dir.mkdir()
        if not (self.cache_dir / "vits").exists():
            (self.cache_dir / "vits").mkdir()
        subdir = (self.cache_dir / "vits" / speaker)
        if not subdir.exists():
            subdir.mkdir()
        features_path = subdir / f"{data_id}.pt"
        wf_path = subdir / f"{data_id}.wav"

        torch.save(data, features_path)
        torchaudio.save(wf_path, src=wf, sample_rate=self.sample_rate)