"""Student objectives for the simplified image-based SkiMo."""
import torch
import torch.nn.functional as F


def diagonal_kl(mu_q, logvar_q, mu_p, logvar_p):
    """KL(q||p), diagonal Gaussians. Broadcast-compatible inputs [...,Z].

    Return [...] by summing over Z, WITHOUT averaging batch/block dimensions.
    Use log variance throughout. This is analytic KL, not a sampled estimate.
    Under a shared invertible tanh transform the KL is unchanged, so compute it
    using the base Gaussian parameters (never apply tanh to mu/logvar here).

    Why is computing KL before tanh valid? Because KL is unchanged when we apply the same invertible transformation to both distributions:

        \[
        \mathrm{KL}\bigl(q(u)\,\|\,p(u)\bigr)
        =
        \mathrm{KL}\bigl(q(z)\,\|\,p(z)\bigr),
        \qquad z=\tanh(u)
        \]

    """
    # raise NotImplementedError("Exercise 2: diagonal Gaussian KL")
    per_coordinate = 0.5 * (
        logvar_p - logvar_q
        + (
            logvar_q.exp()
            + (mu_q - mu_p).square()
        ) / logvar_p.exp()
        - 1
    )
    return per_coordinate.sum(dim=-1)


def pretrain_loss(outputs, images, actions, horizon=10, beta=0.001,
                  image_weight=1.0, dynamics_weight=1.0, prior_weight=1.0):
    """Return total tensor, dict of UNWEIGHTED scalar component tensors.

    Keys: action, kl, image, dynamics, prior.
    action: MSE(actions_hat, actions).
    kl: mean KL(posterior || standard normal).
    image: MSE(images_hat, images[:, :-1:horizon]).
    dynamics: MSE(predicted_states, detached target_states).
    prior: mean KL(detached posterior || learned prior).
    total = action + beta*kl + image_weight*image
            + dynamics_weight*dynamics + prior_weight*prior.

    Both posterior parameter tensors must be detached in prior loss. Do not
    detach posterior for action, KL regularization, or dynamics training.
    """
    mu_q = outputs["posterior_mu"]
    logvar_q = outputs["posterior_logvar"]

    # action_loss
    # print(outputs["actions_hat"].shape)
    # print(actions.shape)
    action_loss =  F.mse_loss(outputs["actions_hat"], actions)

    # Posterior skills
    kl_loss = diagonal_kl(
        mu_q,
        logvar_q,
        torch.zeros_like(mu_q),
        torch.zeros_like(logvar_q),
    ).mean()

    # image_reconstruct_loss 
    target_images = images[:, :-1:horizon]
    pixel_weights = 1.0 + 100.0 * target_images
    # image_loss = F.mse_loss( pixel_weights * outputs["images_hat"], images[:, :-1:horizon])
    image_loss = (
        pixel_weights
        * (outputs["images_hat"] - target_images).square()
    ).mean()


    # dynamics_loss 
    dynamics_loss = F.mse_loss(
        outputs["predicted_states"],
        outputs["target_states"].detach(),
    )

    prior_loss = diagonal_kl(
        mu_q.detach(),
        logvar_q.detach(),
        outputs["prior_mu"],
        outputs["prior_logvar"],
    ).mean()

    total = (
        action_loss
        + beta * kl_loss
        + image_weight * image_loss
        + dynamics_weight * dynamics_loss
        + prior_weight * prior_loss
    )

    return total, {
        "action": action_loss,
        "kl": kl_loss,
        "image": image_loss,
        "dynamics": dynamics_loss,
        "prior": prior_loss,
    }
    
