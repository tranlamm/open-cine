# mc: hidden one-call Ref2VA endpoint (gr.File: the pictures reach the model byte for byte)
with gr.Row(visible=False):
    mc_inputs = [gr.File(file_count="multiple"), gr.File(), gr.Textbox(),
                 gr.Number(value=124, precision=0), gr.Number(value=1344, precision=0),
                 gr.Number(value=768, precision=0), gr.Number(value=8, precision=0),
                 gr.Number(value=42, precision=0), gr.Textbox()]
    mc_outputs = [gr.File(), gr.Textbox()]
    mc_button = gr.Button()
mc_button.click(mc_api.mc_ref2va, inputs=mc_inputs, outputs=mc_outputs, api_name="mc_ref2va")
