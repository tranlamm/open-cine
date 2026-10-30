# mc: one-call API: images, prompt, seed, width, height -> PNG (no prompt rewrite)
with gr.Row(visible=False):
    mc_images = gr.Gallery(type="filepath")
    mc_prompt = gr.Textbox()
    mc_seed = gr.Number(precision=0)
    mc_width = gr.Number(precision=0)
    mc_height = gr.Number(precision=0)
    mc_result = gr.Image(type="pil", image_mode="RGB", format="png")
    mc_button = gr.Button()

def mc_edit(images, prompt, seed, width, height):
    if not prompt or not prompt.strip():
        raise gr.Error("Please enter a prompt.")
    paths = gallery_paths(images)
    if paths:
        check_prompt_guard(prompt)
    return generate_image_stage(paths, prompt, prompt, True, None, int(seed), int(height),
                                int(width), None, None, negative_prompt="",
                                prompt_extend=False)

mc_button.click(fn=mc_edit, inputs=[mc_images, mc_prompt, mc_seed, mc_width, mc_height],
                outputs=[mc_result], api_name="mc_edit")
