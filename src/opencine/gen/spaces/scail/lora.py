"""LoRA fusion for SCAIL-2 (replaced by Open Cine): Kijai-format LoRAs folded into
the model in place, like ComfyUI applies them."""
import torch


@torch.no_grad()
def fuse_lora_with_diff_b(model, lora_state_dict, alpha=1.0):
    state = model.state_dict()  # shares the parameters' storage: copy_ changes the model
    done = {"pair": 0, "diff": 0, "diff_b": 0, "skip": 0}

    def target(key, suffix, field):
        name = key[:-len(suffix)]
        if name.startswith("diffusion_model."):
            name = name[len("diffusion_model."):]
        return state.get(f"{name}.{field}")

    def add(t, delta):
        t.copy_(t.float().add_(delta.reshape(t.shape)).to(t.dtype))

    for key, value in lora_state_dict.items():
        if key.endswith(".lora_down.weight"):
            base = key[:-len(".lora_down.weight")]
            up, w = lora_state_dict.get(base + ".lora_up.weight"), target(key, ".lora_down.weight", "weight")
            if up is None or w is None:
                print(f"[Skip] {key}")
                done["skip"] += 1
                continue
            scale = alpha
            if (a := lora_state_dict.get(base + ".alpha")) is not None:
                scale *= float(a) / value.shape[0]
            delta = up.to(w.device, torch.float32).flatten(1) @ value.to(w.device, torch.float32).flatten(1)
            add(w, delta.mul_(scale))
            done["pair"] += 1
        elif key.endswith((".diff_b", ".diff")):
            kind = "diff_b" if key.endswith(".diff_b") else "diff"
            t = target(key, "." + kind, "bias" if kind == "diff_b" else "weight")
            if t is None:
                print(f"[Skip] {key}")
                done["skip"] += 1
                continue
            add(t, value.to(t.device, torch.float32).mul(alpha))
            done[kind] += 1
    print(f"[LoRA] folded {done}")
