class _FlowFirstOrder:
    """mc: Euler (deterministic) or LCM (fresh noise each step) on the shifted flow grid."""

    def __init__(self, steps, shift, stochastic, device):
        s = torch.linspace(1.0, 0.0, steps + 1, dtype=torch.float64)[:-1]
        s = shift * s / (1 + (shift - 1) * s)
        self.sigmas = torch.cat([s, s.new_zeros(1)]).tolist()
        self.timesteps = (s * 1000).float().to(device)
        self.stochastic, self.i = stochastic, 0

    def step(self, model_output, timestep, sample, return_dict=False, generator=None):
        s, n = self.sigmas[self.i], self.sigmas[self.i + 1]
        self.i += 1
        v, x = model_output.float(), sample.float()
        if not self.stochastic or n == 0:
            return (x + (n - s) * v,)
        x0 = x - s * v
        dev = generator.device if generator is not None else x0.device
        eps = torch.randn(x0.shape, generator=generator, device=dev).to(x0.device)
        return ((1 - n) * x0 + n * eps,)
