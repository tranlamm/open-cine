# mc: hidden one-call endpoint
with gr.Row(visible=False):
    mc_inputs = [gr.File(), gr.File(), gr.Textbox(), gr.Number(value=121, precision=0),
                 gr.Number(value=1024, precision=0), gr.Number(value=576, precision=0),
                 gr.Number(value=0.9), gr.Number(value=42, precision=0)]
    mc_out = gr.File()
    mc_button = gr.Button()
mc_button.click(mc_restyle, inputs=mc_inputs, outputs=[mc_out], api_name="mc_restyle")
