import torch
from tqdm import tqdm
import os
import numpy as np

def egreedy(
    scheduler,
    x,
    vae_function,
    score_function,
    prompt,
    params,
    steps=20,
    t_start=None,
    t_end=None,
    order=2,
    skip_type="time_uniform",
    lower_order_final=True,
    solver_type="dpmsolver",
    flow_shift=1.0,
):
    t_0 = 1.0 / scheduler.noise_schedule.total_N if t_end is None else t_end
    t_T = scheduler.noise_schedule.T if t_start is None else t_start

    device = x.device

    with torch.no_grad():
        assert steps >= order
        timesteps = scheduler.get_time_steps(
            skip_type=skip_type, t_T=t_T, t_0=t_0, N=steps, device=device, shift=flow_shift
        )
        assert timesteps.shape[0] - 1 == steps

        # Init the initial values.
        step = 0
        t = timesteps[step]
        t_prev_list = [None, t]
        model_prev_list = [None, scheduler.model_fn(x, t)]
        if scheduler.correcting_xt_fn is not None:
            x = scheduler.correcting_xt_fn(x, t, step)
            
        # Compute the remaining values by `order`-th order multistep DPM-Solver.
        for step in tqdm(range(1, steps + 1), disable=os.getenv("DPM_TQDM", "False") == "True"):
            t = timesteps[step]
            pivot = torch.randn_like(x)

            if step == 0 or (lower_order_final and step == steps):
                step_order = 1
            else:
                step_order = order

            for _ in range(params['K']):
                noise_candidates = []
                noise2score = {}

                for _ in range(params['N']):
                    if torch.rand(1).item() < params['eps']:
                        noise_candidates.append(torch.randn_like(x))

                    else:
                        to_add = torch.randn_like(x)
                        to_add = to_add / torch.norm(to_add)
                        noise_candidates.append(pivot + to_add * torch.rand(1).item() * params['lambda'] * np.sqrt(x.shape[-1] * x.shape[-2] * x.shape[-3]))

                for noise_candidate in noise_candidates:
                    tminusone = scheduler.multistep_dpm_solver_update(
                        x, model_prev_list, t_prev_list, t, step_order, zs=noise_candidate, solver_type=solver_type
                    )
                    if scheduler.correcting_xt_fn is not None:
                        tminusone = scheduler.correcting_xt_fn(tminusone, t, step)

                    pred_timusone = scheduler.data_prediction_fn(tminusone, t)

                    image = vae_function(pred_timusone)

                    score = score_function(
                        (image * 127.5 + 128).clip(0, 255).to(torch.uint8),
                        prompt
                    )

                    noise2score[image] = score

                pivot = max(noise2score, key=noise2score.get)
            
            x = scheduler.multistep_dpm_solver_update(
                x, model_prev_list, t_prev_list, t, step_order, zs=pivot, solver_type=solver_type
            )
            if scheduler.correcting_xt_fn is not None:
                x = scheduler.correcting_xt_fn(x, t, step)

            for i in range(order - 1):
                t_prev_list[i] = t_prev_list[i + 1]
                model_prev_list[i] = model_prev_list[i + 1]

            t_prev_list[-1] = t
            # We do not need to evaluate the final model value.
            if step < steps:
                model_prev_list[-1] = scheduler.model_fn(x, t)
            # update progress bar
            scheduler.update_progress(step + 1, len(timesteps))
    
    return x