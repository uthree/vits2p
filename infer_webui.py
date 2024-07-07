import argparse
from pathlib import Path
import json

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchaudio.functional import resample
import matplotlib.pyplot as plt

from module.vits import VITS
from module.g2p import G2PModule
from module.utils.config import load_json_file

import gradio as gr



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-md", "--metadata", type=str, default="weights/metadata.json")
    parser.add_argument("-ckpt", "--ckpt-path", type=str, default="weights/vits.ckpt")
    parser.add_argument("-d", "--device", default="cpu")
    parser.add_argument("-p", "--pitch-shift", default=0.0, type=float)
    args = parser.parse_args()

    outputs_path = Path("outputs")
    if not outputs_path.exists():
        outputs_path.mkdir()

    device = torch.device(args.device)

    print(f"loading model from {args.ckpt_path}")
    vits = VITS.load_from_checkpoint(args.ckpt_path, map_location=device)
    net_g = vits.net_g
    g2p = G2PModule()
    metadata = json.load(open(args.metadata))
    speakers = metadata['speakers']
    languages = metadata['languages']
    sample_rate = metadata['sample_rate']

    @torch.inference_mode()
    def synthesize(text, speaker, language, length_scale, pitch_shift):
        sid = torch.LongTensor([speakers.index(speaker)]).to(device)
        lang_id = torch.LongTensor([languages.index(language)]).to(device)
        text_encoded, text_length, _ = g2p.encode(text, language)
        text_encoded = text_encoded.to(device)
        text_length = text_length.to(device)
        
        output_wf, attn, y_mask, (z_p_dur, m_p_dur, logs_p_dur), (z_p_audio, m_p_audio, logs_p_audio) = net_g.infer(
            text_encoded, text_length, sid,
            length_scale=length_scale,
            pitch_shift=pitch_shift
        )
        output_wf = output_wf.squeeze(1)

        output_wf = output_wf.clamp(-1.0, 1.0)
        output_wf = output_wf * 32768.0
        output_wf = output_wf.to(torch.int16).squeeze(0).cpu().numpy()
        return (sample_rate, output_wf)
    
    demo = gr.Interface(
        synthesize,
        inputs=[
            gr.Text(label="Text"),
            gr.Dropdown(label="Speaker", choices=speakers, value=speakers[0]),
            gr.Dropdown(label="Language", choices=languages, value=languages[0]),
            gr.Slider(label="Length Scale", value=1.0, minimum=0.1, maximum=3.0),
            gr.Slider(label="Pitch Shift", value=0, minimum=-12, maximum=12)
        ],
        outputs=[
            gr.Audio(label="Output")
        ]
    )
    
    demo.launch()