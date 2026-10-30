# mc: the same call + the second pass's LMS LoRA strength and caption (0/1)
with gr.Row(visible=False):
    mc_inputs_hq = [gr.File(), gr.File(), gr.Textbox(), gr.Number(value=124, precision=0),
                    gr.Number(value=416, precision=0), gr.Number(value=736, precision=0),
                    gr.Number(value=8, precision=0), gr.Number(value=46, precision=0),
                    gr.Number(value=768, precision=0), gr.Number(value=1344, precision=0),
                    gr.Textbox(), gr.Number(value=0.7), gr.Number(value=1, precision=0)]
    mc_outputs_hq = [gr.File(), gr.Textbox()]
    mc_button_hq = gr.Button()
mc_button_hq.click(mc_api.mc_charswap, inputs=mc_inputs_hq, outputs=mc_outputs_hq,
                   api_name="mc_charswap_hq")
