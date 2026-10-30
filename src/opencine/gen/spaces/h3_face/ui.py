# mc: hidden face-pass endpoint (crop clip, pictures, soundtrack WAV, prompt, frames, size,
# steps, seed, one strength per frame)
with gr.Row(visible=False):
    mc_face_inputs = [gr.File(), gr.File(file_count="multiple"), gr.File(), gr.Textbox(),
                      gr.Number(value=124, precision=0), gr.Number(value=768, precision=0),
                      gr.Number(value=8, precision=0), gr.Number(value=42, precision=0),
                      gr.Textbox()]
    mc_face_outputs = [gr.File(), gr.Textbox()]
    mc_face_button = gr.Button()
mc_face_button.click(mc_api.mc_facefix, inputs=mc_face_inputs, outputs=mc_face_outputs,
                     api_name="mc_facefix")
