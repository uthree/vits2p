import torch
import torchaudio
import json
from pathlib import Path
import lightning as L
from torch.utils.data import DataLoader
import os


class VITSDataset(torch.utils.data.Dataset):
    def __init__(self, cache_dir='dataset_cache', metadata='metadata.json'):
        super().__init__()
        self.root = Path(cache_dir) / "vits"
        self.audio_paths = []
        self.feature_paths = []
        self.metadata = json.load(open(metadata))

        for path in self.root.rglob("*.wav"):
            self.audio_paths.append(path)
            self.feature_paths.append(path.with_suffix(".pt"))

    def __getitem__(self, idx):
        features = torch.load(self.feature_paths[idx])
        wf, _sr = torchaudio.load(self.audio_paths[idx])
        wf = wf.sum(0)
        spk = self.feature_paths[idx].parent.name
        text = features['text'].squeeze(0)
        sid = self.metadata['speakers'].index(spk)
        spec = features['spec'].squeeze(0)
        spec_length = features['spec_length'].squeeze(0)
        f0 = features['f0'].squeeze(0)
        text_length = features['text_length'].squeeze(0)
        lang_id = features['language_id'].squeeze(0)
        return wf, spec, spec_length, f0, text, text_length, sid, lang_id
        
    def __len__(self):
        return len(self.feature_paths)


class VITSDataModule(L.LightningDataModule):
    def __init__(
            self,
            cache_dir='dataset_cache',
            metadata='metadata.json',
            batch_size=1,
            num_workers=1,
            ):
        super().__init__()
        self.cache_dir = cache_dir
        self.metadata = metadata
        self.batch_size = batch_size
        self.num_workers = num_workers

    def train_dataloader(self):
        dataset = VITSDataset(
                self.cache_dir,
                self.metadata)
        dataloader = DataLoader(
                dataset,
                self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                persistent_workers=(os.name=='nt'))
        return dataloader