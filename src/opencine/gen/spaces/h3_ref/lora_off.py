if filename.strip().lower() in ("", "none"):  # mc: no task LoRA (plain Ref2VA)
    LOAD_NOTES.append(f"{label}: off")
    continue
