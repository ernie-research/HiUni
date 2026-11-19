import torch
import math


DIFFUSION_LOSS_TYPES = ("v-loss", "x-loss")
DIFFUSION_LOSS_REDUCTIONS = ("auto", "per_image_token", "per_token")


def normalize_diffusion_loss_type(loss_type):
    """Normalize the YAML loss selector to the canonical ``v-loss``/``x-loss`` names."""
    if loss_type is None:
        return None
    normalized = str(loss_type).strip().lower().replace("_", "-")
    aliases = {
        "v": "v-loss",
        "velocity": "v-loss",
        "velocity-loss": "v-loss",
        "v-loss": "v-loss",
        "x": "x-loss",
        "x0": "x-loss",
        "x0-loss": "x-loss",
        "x-loss": "x-loss",
    }
    try:
        return aliases[normalized]
    except KeyError as exc:
        choices = ", ".join(DIFFUSION_LOSS_TYPES)
        raise ValueError(
            f"Unknown diffusion_loss_type={loss_type!r}; choose from: {choices}."
        ) from exc


def normalize_diffusion_loss_reduction(reduction):
    """Normalize the diffusion loss reduction selector.

    ``auto`` uses per-image-token averaging, which keeps images with different
    resolutions equally weighted in the dense Pixel-DiT objective.
    """
    if reduction is None:
        return None
    normalized = str(reduction).strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "auto": "auto",
        "default": "auto",
        "per_image_token": "per_image_token",
        "per_image": "per_image_token",
        "image": "per_image_token",
        "image_token": "per_image_token",
        "per_token": "per_token",
        "token": "per_token",
    }
    try:
        return aliases[normalized]
    except KeyError as exc:
        choices = ", ".join(DIFFUSION_LOSS_REDUCTIONS)
        raise ValueError(
            f"Unknown diffusion_loss_reduction={reduction!r}; choose from: {choices}."
        ) from exc


def diffusion_parameterization_weight(
    sigmas, predict_x0, loss_type, predict_x0_sigma_min=0.05
):
    """Map an x0/velocity head error to the selected x-loss/v-loss objective."""
    normalized = normalize_diffusion_loss_type(loss_type) or "v-loss"
    if normalized == "x-loss":
        return torch.ones_like(sigmas) if predict_x0 else sigmas ** 2
    if predict_x0:
        return 1.0 / sigmas.clamp(min=predict_x0_sigma_min) ** 2
    return torch.ones_like(sigmas)


def compute_dynamic_noise_scale(
    image_seq_lens: torch.Tensor,
    ref_token_count: int = 256,
    base_scale: float = 1.0,
) -> torch.Tensor:
    """返回 [num_images] 的 σ_R 缩放因子."""
    return base_scale * (image_seq_lens.float() / ref_token_count).sqrt()


class SchedulerConfig:
    """
    scheduler configuration class
    """

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def get(self, key, default=None):
        """
        get method
        """
        return self.__dict__.get(key, default)

    def __getattr__(self, name):
        return self.__dict__.get(name)


class FlowMatchScheduler:
    """
    FlowMatchScheduler for flow-matching / rectified-flow training and inference.

    Timestep sampling modes (controlled by `timestep_sampling`):
        - "uniform"           : t ~ Uniform(sigma_min, sigma_max)  (vanilla)
        - "shift_uniform"     : shifted-uniform with alpha=shift
                                density  p(t) ∝ alpha / (alpha + (1-alpha)*t)^2
                                inverse-CDF: t = alpha*u / (1 + (alpha-1)*(1-u))
        - "shift_logit_norm"  : logit-normal with mu and sigma=logit_norm_sigma
                                sample: t = sigmoid(mu + sigma*eps),  eps~N(0,1)
                                mu = logit_norm_mu if set, else log(alpha).
                                Set logit_norm_mu to decouple mu from `shift`.

    `shift` (alpha) is set directly in the yaml config, typically chosen to match
    the target training resolution.

    Loss reweighting (BSMNTW bell-curve weights) is controlled by `use_loss_reweight`.
    Set to False to use constant weight=1.
    """

    def __init__(
        self,
        num_inference_steps=100,
        num_train_timesteps=1000,
        shift=3.0,
        sigma_max=1.0,
        sigma_min=0.003 / 1.002,
        inverse_timesteps=False,
        extra_one_step=False,
        reverse_sigmas=False,
        # --- legacy exponential-shift (Flux) ---
        exponential_shift=False,
        exponential_shift_mu=None,
        shift_terminal=None,
        # --- timestep sampling mode ---
        timestep_sampling="uniform",   # "uniform" | "shift_uniform" | "shift_logit_norm"
        logit_norm_sigma=1.0,          # sigma for shift_logit_norm mode
        logit_norm_mu=None,            # mu for shift_logit_norm; None => log(shift)
        # --- loss reweight switch ---
        use_loss_reweight=True,
        # --- dynamic noise scale ---
        use_dynamic_noise_scale=False,
        dynamic_noise_ref_tokens=256,
        dynamic_noise_base_scale=1.0,
    ):
        self.num_train_timesteps = num_train_timesteps
        self.shift = shift
        self.sigma_max = sigma_max
        self.sigma_min = sigma_min
        self.inverse_timesteps = inverse_timesteps
        self.extra_one_step = extra_one_step
        self.reverse_sigmas = reverse_sigmas
        self.exponential_shift = exponential_shift
        self.exponential_shift_mu = exponential_shift_mu
        self.shift_terminal = shift_terminal
        self.timestep_sampling = timestep_sampling
        self.logit_norm_sigma = logit_norm_sigma
        self.logit_norm_mu = logit_norm_mu
        self.use_loss_reweight = use_loss_reweight
        self.use_dynamic_noise_scale = use_dynamic_noise_scale
        self.dynamic_noise_ref_tokens = dynamic_noise_ref_tokens
        self.dynamic_noise_base_scale = dynamic_noise_base_scale

        self.set_timesteps(num_inference_steps, training=True)
        self.config = SchedulerConfig(
            num_train_timesteps=num_train_timesteps,
            shift=shift,
        )

    # ------------------------------------------------------------------
    # set_timesteps  (inference / init)
    # ------------------------------------------------------------------

    def set_timesteps(
        self,
        num_inference_steps=100,
        denoising_strength=1.0,
        training=False,
        shift=None,
        dynamic_shift_len=None,
        exponential_shift_mu=None,
    ):
        """
        Build the sigma/timestep grid used for inference (and BSMNTW training weights).

        dynamic_shift_len: image token count; used only by the legacy
                           exponential_shift (Flux-style) branch.
        """
        if shift is not None:
            self.shift = shift

        sigma_start = (
            self.sigma_min + (self.sigma_max - self.sigma_min) * denoising_strength
        )
        if self.extra_one_step:
            self.sigmas = torch.linspace(
                sigma_start, self.sigma_min, num_inference_steps + 1
            )[:-1]
        else:
            self.sigmas = torch.linspace(
                sigma_start, self.sigma_min, num_inference_steps
            )
        if self.inverse_timesteps:
            self.sigmas = torch.flip(self.sigmas, dims=[0])

        if self.exponential_shift:
            if exponential_shift_mu is not None:
                mu = exponential_shift_mu
            elif dynamic_shift_len is not None:
                mu = self.calculate_shift(dynamic_shift_len)
            else:
                mu = self.exponential_shift_mu
            self.sigmas = math.exp(mu) / (math.exp(mu) + (1 / self.sigmas - 1))
        else:
            self.sigmas = (
                self.shift * self.sigmas / (1 + (self.shift - 1) * self.sigmas)
            )

        if self.shift_terminal is not None:
            one_minus_z = 1 - self.sigmas
            scale_factor = one_minus_z[-1] / (1 - self.shift_terminal)
            self.sigmas = 1 - (one_minus_z / scale_factor)
        if self.reverse_sigmas:
            self.sigmas = 1 - self.sigmas

        self.timesteps = self.sigmas * self.num_train_timesteps

        if training and self.use_loss_reweight:
            x = self.timesteps
            y = torch.exp(
                -2 * ((x - num_inference_steps / 2) / num_inference_steps) ** 2
            )
            y_shifted = y - y.min()
            bsmntw_weighing = y_shifted * (num_inference_steps / y_shifted.sum())
            self.linear_timesteps_weights = bsmntw_weighing
        else:
            # constant weight = 1; kept as tensor so callers need not branch
            self.linear_timesteps_weights = torch.ones(len(self.timesteps))

        self.training = training

    # ------------------------------------------------------------------
    # training-time timestep sampling
    # ------------------------------------------------------------------

    def sample_timesteps(self, num_images, generator=None, device=None):
        """
        Sample one sigma per image for training.

        Modes (self.timestep_sampling):
          "uniform"
              t ~ Uniform on the pre-built discrete sigma grid (legacy).

          "shift_uniform"
              Shifted-uniform distribution with alpha = self.shift.
              Density:    p(t) ∝ alpha / (alpha + (1-alpha)*t)^2
              Inverse-CDF: t = alpha*u / (1 + (alpha-1)*(1-u)),  u~U(0,1)

          "shift_logit_norm"
              Logit-normal with mu = log(alpha), sigma = self.logit_norm_sigma.
              Sample:  t = sigmoid(log(alpha) + sigma*eps),  eps~N(0,1)

        `alpha` = self.shift, set directly in yaml per training resolution.

        Returns:
            sigmas:    Tensor [num_images]
            timesteps: Tensor [num_images]  (= sigmas * num_train_timesteps)
        """
        dev = device if device is not None else "cpu"

        if self.timestep_sampling == "uniform":
            ids = torch.randint(
                0, len(self.timesteps),
                (num_images,),
                generator=generator,
                device=dev,
            )
            sigmas = self.sigmas.to(dev)[ids]
            return sigmas, sigmas * self.num_train_timesteps

        alpha = self.shift
        u = torch.rand(num_images, generator=generator, device=dev)  # U(0,1)

        if self.timestep_sampling == "shift_uniform":
            # inverse-CDF of  p(t) ∝ alpha / (alpha + (1-alpha)*t)^2
            sigmas = alpha * u / (1.0 + (alpha - 1.0) * (1.0 - u))

        elif self.timestep_sampling == "shift_logit_norm":
            # t = sigmoid(mu + sigma * eps),  eps ~ N(0,1)
            # mu defaults to log(alpha) for backward-compat, but can be set
            # independently via logit_norm_mu to decouple it from `shift`
            # (which also warps the inference sigma grid).
            mu = self.logit_norm_mu if self.logit_norm_mu is not None else math.log(alpha)
            eps = torch.randn(num_images, generator=generator, device=dev)
            sigmas = torch.sigmoid(mu + self.logit_norm_sigma * eps)

        else:
            raise ValueError(
                f"Unknown timestep_sampling='{self.timestep_sampling}'. "
                "Choose from: 'uniform', 'shift_uniform', 'shift_logit_norm'."
            )

        sigmas = sigmas.clamp(self.sigma_min, self.sigma_max)
        timesteps = sigmas * self.num_train_timesteps
        return sigmas, timesteps

    # ------------------------------------------------------------------
    # inference step
    # ------------------------------------------------------------------

    def step(
        self,
        model_output,
        timestep,
        sample,
        to_final=False,
        predict_x0=False,
        predict_x0_sigma_min=0.05,
        **kwargs,
    ):
        """
        step function for sampling
        """
        if isinstance(timestep, torch.Tensor):
            timestep = timestep.to(self.timesteps.device)
            compare_sigmas = bool(timestep.max().item() <= 1.0)
        else:
            compare_sigmas = isinstance(timestep, (int, float)) and timestep <= 1.0

        if compare_sigmas:
            timestep_id = torch.argmin((self.sigmas - timestep).abs())
        else:
            timestep_id = torch.argmin((self.timesteps - timestep).abs())
        sigma = self.sigmas[timestep_id]
        if to_final or timestep_id + 1 >= len(self.timesteps):
            sigma_ = 1 if (self.inverse_timesteps or self.reverse_sigmas) else 0
        else:
            sigma_ = self.sigmas[timestep_id + 1]
        if predict_x0:
            if predict_x0_sigma_min is None:
                model_output = (sample - model_output) / sigma
            else:
                model_output = (sample - model_output) / sigma.clamp(min=predict_x0_sigma_min)
        prev_sample = sample + model_output * (sigma_ - sigma)
        return prev_sample

    def return_to_timestep(self, timestep, sample, sample_stablized):
        """
        return to timestep
        """
        if isinstance(timestep, torch.Tensor):
            timestep = timestep.cpu()
        timestep_id = torch.argmin((self.timesteps - timestep).abs())
        sigma = self.sigmas[timestep_id]
        model_output = (sample - sample_stablized) / sigma
        return model_output

    # ------------------------------------------------------------------
    # noise addition
    # ------------------------------------------------------------------

    def add_noise(self, original_samples, noise, timestep):
        """
        add noise for single element
        """
        if isinstance(timestep, torch.Tensor):
            timestep = timestep.cpu()
        timestep_id = torch.argmin((self.timesteps - timestep).abs())
        sigma = self.sigmas[timestep_id]
        sample = (1 - sigma) * original_samples + sigma * noise
        return sample

    def add_noise_batch(self, original_samples, noise, timesteps):
        """
        original_samples: [B, C, ...]
        noise:            [B, C, ...]
        timesteps:        [B]
        """
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor(timesteps, device=original_samples.device)

        timesteps = timesteps.to(self.timesteps.device)

        dist = (self.timesteps[None, :] - timesteps[:, None]).abs()
        timestep_ids = dist.argmin(dim=1)  # [B]

        sigmas = self.sigmas[timestep_ids].to(original_samples.device)  # [B]

        while sigmas.ndim < original_samples.ndim:
            sigmas = sigmas.unsqueeze(-1)

        sample = (1 - sigmas) * original_samples + sigmas * noise
        return sample

    def add_noise_batch_packing(self, original_samples, noise, timesteps):
        """
        Per token add noise for flattened packed sequences.

        Args:
            original_samples: [Total_Tokens, C]
            noise:            [Total_Tokens, C]
            timesteps:        [Total_Tokens] (aligned 1:1 with samples)
        """
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor(timesteps, device=original_samples.device)

        timesteps = timesteps.to(self.timesteps.device)
        dist = (timesteps.unsqueeze(1) - self.timesteps.unsqueeze(0)).abs()
        idx = dist.argmin(dim=1)
        sigmas = self.sigmas[idx].to(original_samples.device)
        sigmas = sigmas.unsqueeze(-1)
        sample = (1 - sigmas) * original_samples + sigmas * noise
        return sample

    def add_noise_batch_packing_direct(self, original_samples, noise, sigmas):
        """
        Add noise using pre-computed sigma values (no grid lookup).
        Used when timesteps are sampled continuously (shift_uniform / shift_logit_norm).

        Args:
            original_samples: [Total_Tokens, C]
            noise:            [Total_Tokens, C]
            sigmas:           [Total_Tokens] or [Total_Tokens, 1]
        """
        if sigmas.ndim == 1:
            sigmas = sigmas.unsqueeze(-1)
        sigmas = sigmas.to(original_samples.device)
        return (1 - sigmas) * original_samples + sigmas * noise

    # ------------------------------------------------------------------
    # training targets
    # ------------------------------------------------------------------

    def training_target(self, sample, noise):
        """
        get training target
        """
        target = noise - sample
        return target

    def training_target_x0(self, sample, noise):
        """
        x0-prediction target: return clean sample x0 directly (JiT style)
        """
        return sample

    # ------------------------------------------------------------------
    # sigma lookup helpers
    # ------------------------------------------------------------------

    def get_sigmas_batch_packing(self, timesteps, device):
        """
        Get sigma values for packing mode timesteps.
        timesteps: [Total_Tokens] - per-token timesteps
        Returns: [Total_Tokens, 1] sigmas for broadcasting with [Total_Tokens, C] tensors
        """
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor(timesteps, device=self.timesteps.device)
        else:
            timesteps = timesteps.to(self.timesteps.device)

        dist = (timesteps.unsqueeze(1) - self.timesteps.unsqueeze(0)).abs()
        idx = dist.argmin(dim=1)
        sigmas = self.sigmas[idx].to(device)
        return sigmas.unsqueeze(-1)

    # ------------------------------------------------------------------
    # loss reweighting
    # ------------------------------------------------------------------

    def training_weight(self, timestep):
        """
        Training reweight function (batch mode, [B] timesteps).
        Returns constant 1 when use_loss_reweight=False.
        """
        timestep = timestep.to(self.timesteps.device)
        dist = (self.timesteps[None, :] - timestep[:, None]).abs()
        timestep_id = dist.argmin(dim=1)  # [B]
        weights = self.linear_timesteps_weights[timestep_id]  # [B]
        return weights

    def training_weight_packing(self, timestep):
        """
        Training reweight function for flattened packed sequences.

        Args:
            timestep: Tensor of shape [Total_Tokens].

        Returns:
            weights: Tensor of shape [Total_Tokens].
            Returns constant 1 when use_loss_reweight=False.
        """
        if not torch.is_tensor(timestep):
            timestep = torch.tensor(timestep, device=self.timesteps.device)
        else:
            timestep = timestep.to(self.timesteps.device)
        dist = (timestep.unsqueeze(1) - self.timesteps.unsqueeze(0)).abs()
        timestep_id = dist.argmin(dim=1)
        weights = self.linear_timesteps_weights[timestep_id]  # [Total_Tokens]
        return weights

    # ------------------------------------------------------------------
    # legacy helpers
    # ------------------------------------------------------------------

    def calculate_shift(
        self,
        image_seq_len,
        base_seq_len=256,
        max_seq_len=8192,
        base_shift=0.5,
        max_shift=0.9,
    ):
        """
        Compute mu for exponential-shift (Flux-style) from sequence length.
        """
        m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
        b = base_shift - m * base_seq_len
        mu = image_seq_len * m + b
        return mu
