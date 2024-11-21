import os
import gradio as gr
from infer_wav2svp import infer

def inference(model, input, bpm):
    model_path = os.path.join('weights', model)
    if not input.lower().endswith('.wav'):
        raise gr.Error('Input file must be a wav file.')
    return infer(input, model_path, bpm)

def webui():
    choices = []
    for file in os.listdir('weights'):
        if file.endswith('.ckpt'):
            choices.append(file)
    
    with gr.Blocks() as webui:
        gr.Markdown('''<div align="center"><font size=6><b>wav2svp - Waveform to Synthesizer V Project</b></font></div>''')
        gr.Markdown("Upload an audio file and download the svp file with midi and pitch data.")
        with gr.Row():
            with gr.Column():
                model = gr.Dropdown(
                    label="SOME Model", choices=choices, value=choices[0],
                    multiselect=False, allow_custom_value=False
                )
                input = gr.File(label="Input Audio File", type="filepath")
                bpm = gr.Number(label='BPM Value', minimum=20, maximum=200, value=120)
                run = gr.Button(value="Generate svp File", variant="primary")
            with gr.Column():
                output_svp = gr.File(label="Output svp File", type="filepath", interactive=False)
                output_midi = gr.File(label="Output midi File", type="filepath", interactive=False)
        run.click(inference, [model, input, bpm], [output_svp, output_midi])
    webui.launch(inbrowser=True)

if __name__ == '__main__':
    webui()