if lora_path is None and os.getenv("SCAIL_LORA_REPO"):  # mc: a LoRA from the Hub
    lora_path = hf_hub_download(os.environ["SCAIL_LORA_REPO"], os.environ["SCAIL_LORA_FILE"])
