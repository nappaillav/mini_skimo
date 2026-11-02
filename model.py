"""Student implementation: continuous skills conditioned on image observations."""
from copy import deepcopy
import torch
from torch import nn


def make_mlp(input_dim, output_dim, hidden_dim=128):
    """Shared architecture: two 128-wide ReLU layers and a linear output."""
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, output_dim),
    )


def sample_skill(mu, logvar, deterministic=False):
    """[B,Z] Gaussian parameters -> tanh-transformed sample [B,Z].

    Reparameterize u=mu+std*noise, then z=tanh(u). In deterministic mode use
    tanh(mu). logvar is log variance, not log standard deviation.
    Do not detach: reconstruction and dynamics must train the skill encoder.
    """
    # raise NotImplementedError("Exercise 2: Gaussian reparameterization")
    if deterministic:
        return torch.tanh(mu)

    noise = torch.randn_like(mu)
    u = mu + torch.exp(0.5 * logvar) * noise
    return torch.tanh(u)

class ImageEncoder(nn.Module):
    def __init__(self, state_dim=32):
        super().__init__()
        # Two stride-2 convolutions: 28 -> 14 -> 7; flatten and project to state_dim.
        # This h is continuous; do not reuse the VQ quantizer.
        # raise NotImplementedError("Exercise 1: image -> continuous state")
        self.conv1 = nn.Conv2d(1, 32, 2,2)
        self.conv2 = nn.Conv2d(32, 16, 2,2)
        # self.conv3 = nn.Conv2d(16, 8, 2,2)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(16*7*7, state_dim)

    def forward(self, images):
        """[B,1,28,28] -> h [B,state_dim]."""
        B, _,_,_ = images.shape
        x = self.relu(self.conv2(self.relu(self.conv1(images)))).view(B, -1)
        return self.fc(x)

class ImageDecoder(nn.Module):
    def __init__(self, state_dim=32):
        super().__init__()
        # Linear projection to a feature grid; upsample 7 -> 14 -> 28; sigmoid.
        # raise NotImplementedError("Exercise 1: continuous state -> image")
        self.upconv1 = nn.ConvTranspose2d(16, 8, 2,2)
        self.upconv2 = nn.ConvTranspose2d(8, 1, 2,2)
        self.fc = nn.Linear(state_dim, 16*7*7)
        self.relu = nn.ReLU()

    def forward(self, h):
        """[B,state_dim] -> [B,1,28,28] in [0,1]."""
        x = self.relu(self.fc(h)).view(-1, 16 , 7, 7)
        return torch.sigmoid(self.upconv2(self.relu(self.upconv1(x))))


class SkillEncoder(nn.Module):
    def __init__(self, state_dim=32,  skill_dim=4, horizon=10, action_dim = 2,):
        super().__init__()
        # Minimal choice: flatten H concatenated (h_t,a_t) vectors, then an MLP.
        # Output separate mu and logvar heads. Clamp logvar, e.g. [-10,5].
        # raise NotImplementedError("Exercise 2: trajectory -> Gaussian parameters")
        self.network = make_mlp(horizon * (state_dim + action_dim), 2 * skill_dim)

    def forward(self, states, actions):
        """[B,H,state_dim], [B,H,2] -> mu,logvar each [B,skill_dim].

        Exclude the final observation s_H: inputs correspond to a_0,...,a_(H-1).
        """
        x = torch.cat([states, actions], -1).flatten(start_dim=1)
        mu, logvar = self.network(x).chunk(2, dim=-1)
        logvar = logvar.clamp(-10, 5)
        return mu, logvar

class SkillPolicy(nn.Module):
    def __init__(self, state_dim=32, skill_dim=4, action_dim=2):
        super().__init__()
        # raise NotImplementedError("Exercise 2: MLP on concatenated state and skill")
        self.policy = make_mlp(state_dim + skill_dim, action_dim)

    def forward(self, h, z):
        """[B,state_dim], [B,skill_dim] -> action [B,2], bounded by tanh.

        To reconstruct H actions, repeat the SAME skill across H observed states.
        This is a state-conditioned policy, not an open-loop action decoder.
        """
        return torch.tanh(self.policy(torch.cat([h, z], dim=-1)))

class SkillDynamics(nn.Module):
    def __init__(self, state_dim=32, skill_dim=4):
        super().__init__()
        self.skill_dynamics = make_mlp(state_dim + skill_dim, state_dim)

    def forward(self, h, z):
        """[B,state_dim], [B,skill_dim] -> predicted h_after_H [B,state_dim]."""
        return self.skill_dynamics(torch.cat([h, z], dim=-1))


class SkillPrior(nn.Module):
    def __init__(self, state_dim=32, skill_dim=4):
        super().__init__()
        self.network = make_mlp(state_dim, 2 * skill_dim)

    def forward(self, h):
        """[B,state_dim] -> two [B,skill_dim] base-Gaussian parameters."""
        mu, logvar = self.network(h).chunk(2, dim=-1)
        logvar = logvar.clamp(-10, 5)
        return mu, logvar


class SkiMo(nn.Module):
    def __init__(self, state_dim=32, skill_dim=4, horizon=10):
        super().__init__()
        self.horizon = horizon
        self.encoder = ImageEncoder(state_dim)
        self.decoder = ImageDecoder(state_dim)
        self.skill_encoder = SkillEncoder(state_dim, skill_dim, horizon)
        self.policy = SkillPolicy(state_dim, skill_dim)
        self.dynamics = SkillDynamics(state_dim, skill_dim)
        self.prior = SkillPrior(state_dim, skill_dim)
        self.target_encoder = deepcopy(self.encoder).requires_grad_(False)

    def forward(self, images, actions, deterministic=False):
        """images [B,N*H+1,1,28,28], actions [B,N*H,2] -> dict of tensors.

        Infer one posterior skill per H-step block from ONLINE observed states.
        Predict actions using observed states (teacher forcing).
        Roll dynamics from h_0 recursively, using one inferred skill per block.
        Never reset imagined states to observations inside this rollout.
        Compute target boundary states with target_encoder under no_grad.

        Required dict keys and shapes:
          actions_hat: [B,N*H,2]
          images_hat: [B,N,1,28,28] (decode ONLINE states at 0,H,...,(N-1)H)
          predicted_states, target_states: [B,N,state_dim] (at H,...,N*H)
          posterior_mu, posterior_logvar, prior_mu, prior_logvar: [B,N,skill_dim]
        Prior input is detached online state at each block's start (teaching choice).
        Reconstruction uses observed images, not the decoded imagined endpoints.

        # important point this optimizes for mutltiskill

        SkiMo trains across multiple consecutive skills because it plans across multiple skills.

        Suppose reaching a goal requires:

        move toward drawer → open drawer → reach inside

        During planning, only the starting observation is available. Every later state must come from the dynamics model:

        ĥ₁=D(h₀,z₀),  ĥ₂=D(ĥ₁,z₁),  ĥ₃=D(ĥ₂,z₂)

        A model trained only on single-skill transitions always receives an encoded real observation as input. During planning, it instead receives its own imperfect predictions. Small errors can then accumulate.

        SkiMo’s multi-step training exposes the dynamics model to that situation:

                        N
        L_{dynamics} =  ∑  ‖ ĥᵢ-E_{target}(s_{iH}) ‖²
                    i=1

        Each predicted state is compared with the corresponding observed outcome, and gradients pass through the recursive predictions. This encourages dynamics that remain useful across a planning horizon.

        It still learns one skill per H-step block. “Multiple skills” here means training on a sequence of skill transitions, rather than encoding the whole sequence into one skill.

        prior detached 

        """
        # raise NotImplementedError("Exercise 5: compose joint pretraining")
        B, T, C, height, width = images.shape
        N, H = T//self.horizon, self.horizon
        
        h = self.encoder(images.reshape(B*T, C, height, width)).reshape(B, T, -1) # [B, N*H+1, 32]
        D = h.shape[-1]
        A = actions.shape[-1]
        block_state = h[:, :-1].reshape(B*N, H, D)
        start_states = h[:, :-1:H]
        
        posterior_mu, posterior_logvar = self.skill_encoder(block_state, actions.reshape(B*N, H, A))
        prior_mu, prior_logvar = self.prior(start_states.detach().reshape(B*N, D)) # detach 

        z = sample_skill(posterior_mu, posterior_logvar, deterministic) # [B*N, skill_dim]
        skill_dim = z.shape[-1]
        repeated_z = z[:, None, :].expand(-1, H, -1)

        # policy 
        actions_hat = self.policy(block_state.reshape(-1, D), repeated_z.reshape(-1, skill_dim)).reshape(B, N*H, A) # do i need to use detach here
        
        # target_encoder 
        with torch.no_grad():
            target_states = self.target_encoder(images[:, H::H].reshape(B*N, C, height, width)).reshape(B, N, D)

        # skill_dynamics 
        block_skills = z.reshape(B, N, skill_dim) # N skills per batch
        imagined_h = h[:, 0]
        predicted = []

        for i in range(N):
            imagined_h = self.dynamics(imagined_h, block_skills[:, i])
            predicted.append(imagined_h)

        predicted_states = torch.stack(predicted, dim=1)  # [B,N,D]

        # decoder 
        images_hat = self.decoder(start_states.reshape(B*N, D)).reshape(B, N, C, height, width)

        return {
            "actions_hat": actions_hat,
            "images_hat": images_hat,
            "predicted_states": predicted_states,
            "target_states": target_states,
            "posterior_mu": posterior_mu.reshape(B, N, skill_dim),
            "posterior_logvar": posterior_logvar.reshape(B, N, skill_dim),
            "prior_mu": prior_mu.reshape(B, N, skill_dim),
            "prior_logvar": prior_logvar.reshape(B, N, skill_dim),
        }
    @torch.no_grad()
    def update_target(self, tau=0.01):
        """target <- (1-tau)*target + tau*online, after each optimizer step.

        The suggested encoder has no batchnorm buffers; only parameters need EMA.
        """
        # raise NotImplementedError("Exercise 3: exponential moving average")
        for target, online in zip(
            self.target_encoder.parameters(),
            self.encoder.parameters()
        ):
            target.mul_(1-tau)
            target.add_(online, alpha=tau)



if __name__ == "__main__":

    inp = torch.randn(10, 1, 28, 28)
    enc = ImageEncoder()
    dec = ImageDecoder()
    out = enc(inp)
    inp_hat = dec(out)
    print(out.shape)
    print(inp_hat.shape)