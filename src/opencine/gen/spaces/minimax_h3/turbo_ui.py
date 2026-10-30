# mc: a plain Blocks API in place of the gr.Workflow canvas (same call as multimodalart/minimax-h3)
def mc_generate(prompt, image_path=None, last_image_path=None, canvas="768x1344 · 9:16 full",
                duration=5, steps=7, seed=42, upsample=False):
    video, report, refined = generate_video(prompt, image_path, last_image_path, canvas,
                                            duration, steps, seed, upsample)
    return video["path"], report, refined


with gr.Blocks(title="MiniMax-H3 Turbo") as demo:
    with gr.Row():
        with gr.Column():
            mc_prompt = gr.Textbox(label="Prompt", lines=3)
            with gr.Row():
                mc_image = gr.Image(label="First frame (optional)", type="filepath")
                mc_last = gr.Image(label="Last frame (optional)", type="filepath")
            mc_canvas = gr.Dropdown(label="Canvas", choices=list(CANVASES),
                                    value="768x1344 · 9:16 full")
            mc_duration = gr.Number(label="Duration (s)", value=5)
            mc_steps = gr.Number(label="Steps", value=7, precision=0)
            mc_seed = gr.Number(label="Seed", value=42, precision=0)
            mc_upsample = gr.Checkbox(label="Upsample prompt", value=False)
            mc_run = gr.Button("Generate", variant="primary")
        with gr.Column():
            mc_video = gr.Video(label="Video + soundtrack")
            mc_report = gr.Textbox(label="Report")
            mc_refined = gr.Textbox(label="Upsampled prompt")
    mc_run.click(mc_generate, [mc_prompt, mc_image, mc_last, mc_canvas, mc_duration, mc_steps,
                               mc_seed, mc_upsample], [mc_video, mc_report, mc_refined],
                 api_name="generate")
