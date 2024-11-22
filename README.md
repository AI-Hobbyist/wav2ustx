## 歌声转工程
基于 MSST 和 Wav2Svp 修改，可一键分离伴奏及人声，并根据人声生成工程，工程支持 **Synth V (svp)** 和 **OpenUtau (ustx)**


## 使用方法
1. 安装PyTorch
```shell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```
2. 安装依赖
```shell
pip install -r requirements.txt
```
3. 命令行推理
```shell
python wav2ustx.py 歌声路径 输出路径 -t 曲速 -s 启用步骤
```

4. webui推理
```shell
python webui.py
```
### 关于启用步骤
**可选值：** vocal_separation, harmony_removal, deverb, denoise

**注意事项：** 每步要用英文逗号 , 分隔。

### MSST用到的模型（可以直接打开MSST Webui下载安装）
**人声分离：** model_bs_roformer_ep_368_sdr_12.9628

**去除和声：** model_mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956

**去除混响：** deverb_bs_roformer_8_256dim_8depth

**去除噪声：** denoise_mel_band_roformer_aufr33_aggr_sdr_27.9768

### Wav2Svp用到的模型（下载后复制到 weights 文件夹）
**midi提取：** model_steps_64000_simplified （[点击下载](https://github.com/openvpi/SOME/releases/tag/v0.0.1)）

**f0提取：** rmvpe （[点击下载](https://github.com/yxlllc/RMVPE/releases)）

## 参考项目
**MSST-WebUI：** [https://github.com/SUC-DriverOld/MSST-WebUI](https://github.com/SUC-DriverOld/MSST-WebUI)

**Wav2Svp：**[https://github.com/SUC-DriverOld/wav2svp](https://github.com/SUC-DriverOld/wav2svp)

**LibreSVIP：**[https://github.com/SoulMelody/LibreSVIP](https://github.com/SoulMelody/LibreSVIP)
