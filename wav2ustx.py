import librosa
import yaml
import json
import inference
import importlib
import argparse
import warnings
import logging
import subprocess
import os
from time import time
from inference.msst_infer import MSSeparator
from utils.logger import get_logger
from modules.rmvpe.inference import RMVPE
from utils.slicer2 import Slicer
from build_svp import build_svp
from pathlib import Path
from tqdm import tqdm
from hashlib import md5
from shutil import copy, move, rmtree

#==============公用函数==============
#创建随机文件名
def random_filename():
    return md5(str(time()).encode()).hexdigest()

#==============人声分离==============
#分离人声伴奏
def vocal_remover_step1(input, output):
    logger = get_logger(console_level=logging.INFO)
    warnings.filterwarnings("ignore", category=UserWarning)
    start_time = time()
    separator = MSSeparator(
        model_type='bs_roformer',
        config_path=f'configs_backup/vocal_models/model_bs_roformer_ep_368_sdr_12.9628.yaml',
        model_path=f'pretrain/vocal_models/model_bs_roformer_ep_368_sdr_12.9628.ckpt',
        device='auto',
        device_ids=[0],
        output_format='wav',
        use_tta=False,
        store_dirs=output,
        logger=logger,
        debug=False
    )
    success_files = separator.process_folder(input)
    separator.del_cache()
    logger.info(f"Successfully separated files: {success_files}, total time: {time() - start_time:.2f} seconds.")
    
#去除和声
def vocal_remover_step2(input, output):
    logger = get_logger(console_level=logging.INFO)
    warnings.filterwarnings("ignore", category=UserWarning)
    start_time = time()
    separator = MSSeparator(
        model_type='mel_band_roformer',
        config_path='configs_backup/vocal_models/config_mel_band_roformer_karaoke.yaml',
        model_path='pretrain/vocal_models/model_mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956.ckpt',
        device='auto',
        device_ids=[0],
        output_format='wav',
        use_tta=False,
        store_dirs=output,
        logger=logger,
        debug=False
    )
    success_files = separator.process_folder(input)
    separator.del_cache()
    logger.info(f"Successfully separated files: {success_files}, total time: {time() - start_time:.2f} seconds.")
    
#去除混响
def vocal_remover_step3(input, output):
    logger = get_logger(console_level=logging.INFO)
    warnings.filterwarnings("ignore", category=UserWarning)
    start_time = time()
    separator = MSSeparator(
        model_type='mel_band_roformer',
        config_path='configs_backup/single_stem_models/deverb_bs_roformer_8_256dim_8depth.yaml',
        model_path='mpretrain/single_stem_models/deverb_bs_roformer_8_256dim_8depth.ckpt',
        device='auto',
        device_ids=[0],
        output_format='wav',
        use_tta=False,
        store_dirs=output,
        logger=logger,
        debug=False
    )
    success_files = separator.process_folder(input)
    separator.del_cache()
    logger.info(f"Successfully separated files: {success_files}, total time: {time() - start_time:.2f} seconds.")
    
#去除噪声
def vocal_remover_step4(input, output):
    logger = get_logger(console_level=logging.INFO)
    warnings.filterwarnings("ignore", category=UserWarning)
    start_time = time()
    separator = MSSeparator(
        model_type='single_stem_models',
        config_path='configs_backup/single_stem_models/model_mel_band_roformer_denoise.yaml',
        model_path='pretrain/single_stem_models/denoise_mel_band_roformer_aufr33_aggr_sdr_27.9768.ckpt',
        device='auto',
        device_ids=[0],
        output_format='wav',
        use_tta=False,
        store_dirs=output,
        logger=logger,
        debug=False
    )
    success_files = separator.process_folder(input)
    separator.del_cache()
    logger.info(f"Successfully separated files: {success_files}, total time: {time() - start_time:.2f} seconds.")
    
#==============工程生成==============
#加载配置文件
def load_config(config_path: str) -> dict:
    if config_path.endswith('.json'):
        config = json.loads(Path(config_path).read_text(encoding='utf8'))
    elif config_path.endswith('.yaml'):
        config = yaml.safe_load(Path(config_path).read_text(encoding='utf8'))
    else:
        raise ValueError(f'Unsupported config file format: {config_path}')
    return config
config = load_config('weights/config.yaml')
sr = config['audio_sample_rate']

#音频切片
def audio_slicer(audio_path: str) -> list:
    waveform, _ = librosa.load(audio_path, sr=sr, mono=True)
    slicer = Slicer(sr=sr, max_sil_kept=1000)
    chunks = slicer.slice(waveform)
    return chunks

#获取midi
def get_midi(chunks: list, model_path: str) -> list:
    infer_cls = inference.task_inference_mapping[config['task_cls']]
    pkg = ".".join(infer_cls.split(".")[:-1])
    cls_name = infer_cls.split(".")[-1]
    infer_cls = getattr(importlib.import_module(pkg), cls_name)
    assert issubclass(infer_cls, inference.BaseInference), \
        f'Inference class {infer_cls} is not a subclass of {inference.BaseInference}.'
    infer_ins = infer_cls(config=config, model_path=model_path)
    midis = infer_ins.infer([c['waveform'] for c in chunks])
    return midis
    
#获取f0
def get_f0(chunks: list) -> list:
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

#音频转svp
def wav2svp(audio_path, tempo):
    Path('results').mkdir(parents=True, exist_ok=True)

    chunks = audio_slicer(audio_path)
    midis = get_midi(chunks, 'weights/model_steps_64000_simplified.ckpt')
    f0 = get_f0(chunks)

    basename = Path(audio_path).name.split('.')[0]
    template = load_config('template.json')

    print("building svp file")
    svp_path = build_svp(template, midis, f0, tempo, basename)

    return svp_path

#==============工程转换==============
#svp转ustx
def svp2ustx(svp_path):
    ustx_path = svp_path.replace('.svp', '.ustx')
    command = f'libresvip-cli proj convert "{svp_path}" "{ustx_path}"'
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    input_values = ['n\n', 'n\n', 'y\n', 'n\n', 'n\n', 'n\n', 'n\n', 'plain\n', 'convert\n', 'split\n', 'n\n', 'n\n', 'n\n', 'n\n', 'n\n','non-arpa\n']
    for value in input_values:
        process.stdin.write(value)
        process.stdin.flush()
        
    process.communicate()
    
#打包结果
def pack_result():
    Path('project_pack').mkdir(parents=True, exist_ok=True)
    file_name = random_filename()
    command = f'7z a -tzip "project_pack/{file_name}.zip" "final"'
    subprocess.run(command, shell=True)
    print(f'工程已生成并打包至 project_pack/{file_name}.zip')
    
    
#==============主函数==============
def wav2ustx(audio, tempo, enabled_steps):
    #初始化
    Path('results').mkdir(parents=True, exist_ok=True)
    Path('project_pack').mkdir(parents=True, exist_ok=True)
    Path('tempfiles').mkdir(parents=True, exist_ok=True)
    Path('final').mkdir(parents=True, exist_ok=True)
    
    base_name = str(Path(audio).name).replace('.wav', '')
    
    copy(audio, f'tempfiles/{base_name}.wav')
    src = audio
    
    
    #执行步骤
    if 'voice_remove' in enabled_steps:
        Path('results/step1').mkdir(parents=True, exist_ok=True)
        vocal_remover_step1('tempfiles', 'results/step1')
        move(f'results/step1/{base_name}_instrumental.wav', f'final/{base_name}_instrumental.wav')
        src = f'results/step1/{base_name}_vocals.wav'
        
    if 'harmony_remove' in enabled_steps:
        Path('results/step2').mkdir(parents=True, exist_ok=True)
        vocal_remover_step2('results/step1', 'results/step2')
        Path(f'results/step2/{base_name}_other.wav').unlink()
        src = f'results/step2/{base_name}_karaoke.wav'
        
    if 'deverb' in enabled_steps:
        Path('results/step3').mkdir(parents=True, exist_ok=True)
        vocal_remover_step3('results/step2', 'results/step3')
        Path(f'results/step3/{base_name}_reverb.wav').unlink()
        src = f'results/step3/{base_name}_noreverb.wav'
        
    if 'denoise' in enabled_steps:
        Path('results/step4').mkdir(parents=True, exist_ok=True)
        vocal_remover_step4('results/step3', 'results/step4')
        Path(f'results/step2/{base_name}_other.wav').unlink()
        src = f'results/step4/{base_name}_dry.wav'
    
    move(src, f'final/{base_name}.wav')
    svp_path = wav2svp(f'final/{base_name}.wav', tempo)
    svp2ustx(svp_path)
    pack_result()
    rmtree('tempfiles')
    rmtree('results')
    rmtree('final')
    
#==============命令行==============
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='wav2ustx')
    parser.add_argument('-a','--audio', type=str, help='音频文件', required=True)
    parser.add_argument('-t','--tempo', type=int, help='曲速', required=True)
    parser.add_argument('-s','--enabled_steps', type=str, help='启用的步骤，用逗号分隔，可选值：voice_remove, harmony_remove, deverb, denoise', required=True)
    args = parser.parse_args()
    wav2ustx(args.audio, args.tempo, args.enabled_steps.split(','))