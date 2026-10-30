# mc: hidden one-call endpoint (gr.File: the clip reaches the model byte for byte)
with gr.Row(visible=False):
    mc_inputs = [gr.File(), gr.Number(value=2048, precision=0), gr.Number(value=1152, precision=0),
                 gr.Number(value=4, precision=0), gr.Checkbox(value=True),
                 gr.Number(value=97, precision=0)]
    mc_out = gr.File()
    mc_button = gr.Button()
mc_button.click(mc_api.mc_enhance, inputs=mc_inputs, outputs=[mc_out], api_name="mc_enhance")
