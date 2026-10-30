# mc: hidden one-call endpoint, after the UI is built
_MC.update(dit=dit_handler, llm=llm_handler)
import gradio as gr
with demo:
    with gr.Row(visible=False):
        mc_caption = gr.Textbox()
        mc_seconds = gr.Number(value=30)
        mc_bpm = gr.Number(value=0, precision=0)
        mc_key = gr.Textbox()
        mc_seed = gr.Number(value=-1, precision=0)
        mc_thinking = gr.Checkbox(value=True)
        mc_audio = gr.Audio(type="filepath")
        mc_meta = gr.Textbox()
        mc_button = gr.Button()
    mc_button.click(fn=mc_music, inputs=[mc_caption, mc_seconds, mc_bpm, mc_key, mc_seed,
                                         mc_thinking], outputs=[mc_audio, mc_meta],
                    api_name="mc_music")
