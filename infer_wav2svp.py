import librosa
import os
import yaml
import json
import inference
import importlib
from tqdm import tqdm

from modules.rmvpe.inference import RMVPE
from utils.slicer2 import Slicer
from utils.infer_utils import build_midi_file
from build_svp import build_svp


def load_config(config_path: str) -> dict:
    if config_path.endswith('.yaml'):
        with open(config_path, 'r', encoding='utf8') as f:
            config = yaml.safe_load(f)
    elif config_path.endswith('.json'):
        with open(config_path, 'r', encoding='utf8') as f:
            config = json.load(f)
    else:
        raise ValueError(f'Unsupported config file format: {config_path}')
    return config

config = load_config('weights/config.yaml')
sr = config['audio_sample_rate']

def audio_slicer(audio_path: str) -> list:
    """
    Returns:
        list of dict: [{
            "offset": np.float64,
            "waveform": array of float, dtype=float32,
        }, ...]
    """
    waveform, _ = librosa.load(audio_path, sr=sr, mono=True)
    slicer = Slicer(sr=sr, max_sil_kept=1000)
    chunks = slicer.slice(waveform)
    return chunks

def get_midi(chunks: list, model_path: str) -> list:
    """
    Args:
        chunks (list): results from audio_slicer

    Returns:
        list of dict: [{
            "note_midi": array of float, dtype=float32,
            "note_dur": array of float,
            "note_rest": array of bool,
        }, ...]
    """
    infer_cls = inference.task_inference_mapping[config['task_cls']]
    pkg = ".".join(infer_cls.split(".")[:-1])
    cls_name = infer_cls.split(".")[-1]
    infer_cls = getattr(importlib.import_module(pkg), cls_name)
    assert issubclass(infer_cls, inference.BaseInference), \
        f'Inference class {infer_cls} is not a subclass of {inference.BaseInference}.'
    infer_ins = infer_cls(config=config, model_path=model_path)
    midis = infer_ins.infer([c['waveform'] for c in chunks])
    return midis

def save_midi(midis: list, tempo: int, chunks: list, midi_path: str) -> None:
    midi_file = build_midi_file([c['offset'] for c in chunks], midis, tempo=tempo)
    midi_file.save(midi_path)

def get_f0(chunks: list) -> list:
    """
    Args:
        chunks (list): results from audio_slicer

    Returns:
        list of dict: [{
            "offset": np.float64,
            "f0": array of float, dtype=float32,
        }, ...]
    """
    f0 = []
    rmvpe = RMVPE(model_path='weights/rmvpe.pt') # hop_size=160
    print("loading RMVPE model")

    for chunk in tqdm(chunks, desc='Extracting F0'):
        f0_data = {
            "offset": chunk['offset'],
            "f0": rmvpe.infer_from_audio(chunk['waveform'], sample_rate=sr) # sample_rate会重采样至16000
        }
        f0.append(f0_data)
    return f0

def wav2svp(audio_path, model_path, tempo=120):
    os.makedirs('results', exist_ok=True)

    chunks = audio_slicer(audio_path)
    midis = get_midi(chunks, model_path)
    f0 = get_f0(chunks)

    basename = os.path.basename(audio_path).split('.')[0]
    template = load_config('template.json')

    print("building svp file")
    svp_path = build_svp(template, midis, f0, tempo, basename)

    print("building midi file")
    midi_path = os.path.join('results', f'{basename}.mid')
    save_midi(midis, tempo, chunks, midi_path)
    print("Success")

    return svp_path, midi_path

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Inference for wav2svp')
    parser.add_argument('audio_path', type=str, help='Path to the input audio file')
    parser.add_argument('--model_path', type=str, default="weights/model_steps_64000_simplified.ckpt", help='Path to the model file, default: weights/model_steps_64000_simplified.ckpt')
    parser.add_argument('--tempo', type=int, default=120, help='Tempo value for the midi file, default: 120')

    args = parser.parse_args()
    assert os.path.isfile("weights/rmvpe.pt"), "RMVPE model not found"
    assert os.path.isfile(args.model_path), "SOME Model not found"
    wav2svp(args.audio_path, args.model_path, args.tempo)