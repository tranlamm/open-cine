# mc: build each model on the CPU, then .to("cuda") (ZeroGPU packs it); built straight onto
# CUDA, safe_open(device="cuda") reaches the driver at start-up ("No CUDA GPUs are available")
for _mc_ledger in (pipeline.stage_1_model_ledger, pipeline.stage_2_model_ledger):
    _mc_ledger._target_device = lambda: torch.device("cpu")
