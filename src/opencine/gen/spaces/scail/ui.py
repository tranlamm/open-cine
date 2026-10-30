# mc: hidden one-call endpoint (gr.File: inputs reach the model byte for byte)
with gr.Row(visible=False):
    mc_inputs = [gr.File(), gr.File(), gr.File(), gr.File(), gr.Textbox(),
                 gr.Checkbox(value=True), gr.Number(value=8, precision=0),
                 gr.Number(value=1.0), gr.Number(value=1.0),
                 gr.Number(value=42, precision=0), gr.Number(value=512, precision=0),
                 gr.Number(value=896, precision=0), gr.Number(value=81, precision=0)]
    mc_out = gr.File()
    mc_button = gr.Button()
mc_button.click(mc_scail, inputs=mc_inputs, outputs=[mc_out], api_name="mc_scail")
