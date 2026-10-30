# mc: hidden one-call endpoint (gr.File: the clip and the picture reach the model byte for byte)
with gr.Row(visible=False):
    mc_inputs = [gr.File(), gr.File(), gr.Textbox(), gr.Number(value=124, precision=0),
                 gr.Number(value=576, precision=0), gr.Number(value=1024, precision=0),
                 gr.Number(value=8, precision=0), gr.Number(value=46, precision=0),
                 gr.Number(value=0, precision=0), gr.Number(value=0, precision=0), gr.Textbox()]
    mc_outputs = [gr.File(), gr.Textbox()]
    mc_button = gr.Button()
mc_button.click(mc_api.mc_charswap, inputs=mc_inputs, outputs=mc_outputs, api_name="mc_charswap")
