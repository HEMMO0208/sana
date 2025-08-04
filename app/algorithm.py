from tqdm import tqdm
import torch

# def beam_search():
#     max_score = None

#     num_warmup_steps = len(timesteps) - num_inference_steps * scheduler.scheduler.order
#     scheduler._num_timesteps = len(timesteps)

#     best_latents = [copy.deepcopy(latents) for _ in range(params['B'])]

#     for i, t in tqdm(enumerate(timesteps)):
#         all_scored_latents = []
#         all_scores = []
        
#         for beam_latents in best_latents:
#             if scheduler.interrupt:
#                 continue

#             # expand the latents if we are doing classifier free guidance
#             latent_model_input = torch.cat([beam_latents] * 2) if scheduler.do_classifier_free_guidance else beam_latents
#             latent_model_input = scheduler.scheduler.scale_model_input(latent_model_input, t)

#             # predict the noise residual
#             noise_pred = scheduler.unet(
#                 latent_model_input,
#                 t,
#                 encoder_hidden_states=prompt_embeds,
#                 timestep_cond=timestep_cond,
#                 cross_attention_kwargs=scheduler.cross_attention_kwargs,
#                 added_cond_kwargs=added_cond_kwargs,
#                 return_dict=False,
#             )[0]

#             # perform guidance
#             if scheduler.do_classifier_free_guidance:
#                 noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
#                 noise_pred = noise_pred_uncond + scheduler.guidance_scale * (noise_pred_text - noise_pred_uncond)

#             if scheduler.do_classifier_free_guidance and scheduler.guidance_rescale > 0.0:
#                 # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
#                 noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=scheduler.guidance_rescale)

#             noise_candidates = [torch.randn_like(beam_latents) for _ in range(params['N'])]

#             for noise_candidate in noise_candidates:
#                 latents_cand, pred_x0 = scheduler.scheduler.step(noise_pred, t, beam_latents, variance_noise=noise_candidate, **extra_step_kwargs, return_dict=False)

#                 # expand the latents if we are doing classifier free guidance
#                 latent_model_input_tminusone = torch.cat([latents_cand] * 2) if scheduler.do_classifier_free_guidance else latents_cand
#                 latent_model_input_tminusone = scheduler.scheduler.scale_model_input(latent_model_input_tminusone, t)

#                 # predict the noise residual
#                 noise_pred_tminusone = scheduler.unet(
#                     latent_model_input_tminusone,
#                     t,
#                     encoder_hidden_states=prompt_embeds,
#                     timestep_cond=timestep_cond,
#                     cross_attention_kwargs=scheduler.cross_attention_kwargs,
#                     added_cond_kwargs=added_cond_kwargs,
#                     return_dict=False,
#                 )[0]

#                 # perform guidance
#                 if scheduler.do_classifier_free_guidance:
#                     noise_pred_uncond_tminusone, noise_pred_text_tminusone = noise_pred_tminusone.chunk(2)
#                     noise_pred_tminusone = noise_pred_uncond_tminusone + scheduler.guidance_scale * (noise_pred_text_tminusone - noise_pred_uncond_tminusone)

#                 if scheduler.do_classifier_free_guidance and scheduler.guidance_rescale > 0.0:
#                     # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
#                     noise_pred_tminusone = rescale_noise_cfg(noise_pred_tminusone, noise_pred_text, guidance_rescale=scheduler.guidance_rescale)

#                 next_latents, pred_next_tminusone = scheduler.scheduler.step(noise_pred_tminusone, t, latents_cand, **extra_step_kwargs, return_dict=False)

#                 with torch.no_grad():
#                     image = scheduler.vae.decode(pred_next_tminusone / scheduler.vae.config.scaling_factor, return_dict=False, generator=generator)[0]
#                 # score image
#                 score = score_function(
#                     images = [(image * 127.5 + 128).clip(0, 255).to(torch.uint8)],
#                     prompts = [prompt],
#                     timesteps= None,
#                 )

#                 # Convert score to a Python float for reliable comparison
#                 if torch.is_tensor(score):
#                     # score might be a tensor of shape (1,) or scalar tensor
#                     score_value = score.item()
#                 else:
#                     score_value = float(score)

#                 # Store the actual next latents that were scored
#                 all_scored_latents.append(latents_cand)
#                 all_scores.append(score_value)
            
#         # Sort by scores and select top B candidates
#         sorted_indices = sorted(range(len(all_scores)), key=lambda k: all_scores[k], reverse=True)
#         best_indices = sorted_indices[:params['B']]
#         best_latents = [all_scored_latents[idx] for idx in best_indices]
        
#         # Update callbacks only once per timestep
#         if callback_on_step_end is not None:
#             callback_kwargs = {}
#             for k in callback_on_step_end_tensor_inputs:
#                 callback_kwargs[k] = locals()[k]
#             callback_kwargs["latents"] = best_latents[0]  # Use the best latent for callbacks
#             callback_outputs = callback_on_step_end(scheduler, i, t, callback_kwargs)

#             prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
#             negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)

#         # call the callback, if provided
#         if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % scheduler.scheduler.order == 0):
#             if callback is not None and i % callback_steps == 0:
#                 step_idx = i // getattr(scheduler.scheduler, "order", 1)
#                 callback(step_idx, t, best_latents[0])

#         if XLA_AVAILABLE:
#             xm.mark_step()

#     # Select the best latent at the end
#     max_score = float('-inf')
#     latents = best_latents[0]
#     for latents_cand in best_latents:
#         with torch.no_grad():
#             image = scheduler.vae.decode(latents_cand / scheduler.vae.config.scaling_factor, return_dict=False, generator=generator)[0]
#         score = score_function(
#             images = [(image * 127.5 + 128).clip(0, 255).to(torch.uint8)],
#             prompts = [prompt],
#             timesteps= None,
#         )
#         score_value = score.item() if torch.is_tensor(score) else float(score)
#         if score_value > max_score:
#             max_score = score_value
#             latents = latents_cand

#             if not output_type == "latent":
#             image = scheduler.vae.decode(latents / scheduler.vae.config.scaling_factor, return_dict=False, generator=generator)[0]
#             has_nsfw_concept = None
#         else:
#             image = latents
#             has_nsfw_concept = None
        
#         if max_score is None:
#             max_score = score_function(
#                 images = [(image * 127.5 + 128).clip(0, 255).to(torch.uint8)],
#                 prompts = [prompt],
#                 timesteps= None,
#             )
        
#         # Postprocess the image
#         if has_nsfw_concept is None:
#             do_denormalize = [True] * image.shape[0]
#         else:
#             do_denormalize = [not has_nsfw for has_nsfw in has_nsfw_concept]
        
#         image = scheduler.image_processor.postprocess(image, output_type=output_type, do_denormalize=do_denormalize)
        
#         # Offload all models
#         scheduler.maybe_free_model_hooks()
        
#         return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept), max_score


# class MCTSNode:
#     latents: torch.Tensor
#     children: List["MCTSNode"] = field(default_factory=list)
#     parent: Optional["MCTSNode"] = None
#     visits: int = 0
#     total_reward: float = 0.0
    
#     def add_child(scheduler, child_latents):
#         child = MCTSNode(child_latents, parent=scheduler)
#         scheduler.children.append(child)
#         return child
    
#     def ucb_score(scheduler, exploration_constant=1.0):
#         # If node hasn't been visited, give it infinite score for exploration
#         if scheduler.visits == 0:
#             return float('inf')
        
#         # Exploitation term
#         exploitation = scheduler.total_reward / scheduler.visits
        
#         # Exploration term
#         parent_visits = scheduler.parent.visits if scheduler.parent else 1
#         exploration = exploration_constant * math.sqrt(math.log(parent_visits) / scheduler.visits)
        
#         return exploitation + exploration


# def mcts():
#     max_score = None

#     num_warmup_steps = len(timesteps) - num_inference_steps * scheduler.scheduler.order
#     scheduler._num_timesteps = len(timesteps)

#     # Process each timestep using MCTS
#     for i, t in tqdm(enumerate(timesteps)):
#         if scheduler.interrupt:
#             continue
        
#         # Create root node with current latents
#         root = MCTSNode(latents)
#         root.visits = 1  # Initialize root node with one visit
        
#         # Run multiple MCTS iterations for this timestep
#         for _ in range(params['S']):
#             # Selection phase: traverse tree to find promising leaf node
#             node = root
#             while node.children and all(child.visits > 0 for child in node.children):
#                 # Choose child with highest UCB score
#                 node = max(node.children, key=lambda child: child.ucb_score(params.get('c', 1.414)))
            
#             # Expansion phase: if node is not fully expanded, add a child
#             if len(node.children) < params['N']:
#                 # Generate latent model input
#                 latent_model_input = torch.cat([node.latents] * 2) if scheduler.do_classifier_free_guidance else node.latents
#                 latent_model_input = scheduler.scheduler.scale_model_input(latent_model_input, t)
                
#                 # Predict noise residual
#                 noise_pred = scheduler.unet(
#                     latent_model_input,
#                     t,
#                     encoder_hidden_states=prompt_embeds,
#                     timestep_cond=timestep_cond,
#                     cross_attention_kwargs=scheduler.cross_attention_kwargs,
#                     added_cond_kwargs=added_cond_kwargs,
#                     return_dict=False,
#                 )[0]
                
#                 # Perform guidance
#                 if scheduler.do_classifier_free_guidance:
#                     noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
#                     noise_pred = noise_pred_uncond + scheduler.guidance_scale * (noise_pred_text - noise_pred_uncond)
                
#                 if scheduler.do_classifier_free_guidance and scheduler.guidance_rescale > 0.0:
#                     noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=scheduler.guidance_rescale)
                
#                 # Generate random noise and step
#                 noise = torch.randn_like(node.latents)
#                 child_latents, pred_x0 = scheduler.scheduler.step(
#                     noise_pred, t, node.latents, variance_noise=noise, **extra_step_kwargs, return_dict=False
#                 )
                
#                 # Add child node
#                 child = node.add_child(child_latents)
#                 node = child
            
#             # Simulation phase: evaluate the node by decoding and scoring
#             # First get the predicted clean image
#             latent_model_input = torch.cat([node.latents] * 2) if scheduler.do_classifier_free_guidance else node.latents
#             latent_model_input = scheduler.scheduler.scale_model_input(latent_model_input, t)
            
#             noise_pred = scheduler.unet(
#                 latent_model_input,
#                 t,
#                 encoder_hidden_states=prompt_embeds,
#                 timestep_cond=timestep_cond,
#                 cross_attention_kwargs=scheduler.cross_attention_kwargs,
#                 added_cond_kwargs=added_cond_kwargs,
#                 return_dict=False,
#             )[0]
            
#             if scheduler.do_classifier_free_guidance:
#                 noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
#                 noise_pred = noise_pred_uncond + scheduler.guidance_scale * (noise_pred_text - noise_pred_uncond)
            
#             if scheduler.do_classifier_free_guidance and scheduler.guidance_rescale > 0.0:
#                 noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=scheduler.guidance_rescale)
            
#             # Do full naive sampling from current timestep to end
#             temp_latents = node.latents.clone()
#             for j in range(i, len(timesteps)):
#                 t_inner = timesteps[j]
                
#                 # expand the latents if we are doing classifier free guidance
#                 latent_model_input_inner = torch.cat([temp_latents] * 2) if scheduler.do_classifier_free_guidance else temp_latents
#                 latent_model_input_inner = scheduler.scheduler.scale_model_input(latent_model_input_inner, t_inner)
                
#                 # predict the noise residual
#                 noise_pred_inner = scheduler.unet(
#                     latent_model_input_inner,
#                     t_inner,
#                     encoder_hidden_states=prompt_embeds,
#                     timestep_cond=timestep_cond,
#                     cross_attention_kwargs=scheduler.cross_attention_kwargs,
#                     added_cond_kwargs=added_cond_kwargs,
#                     return_dict=False,
#                 )[0]
                
#                 # perform guidance
#                 if scheduler.do_classifier_free_guidance:
#                     noise_pred_uncond_inner, noise_pred_text_inner = noise_pred_inner.chunk(2)
#                     noise_pred_inner = noise_pred_uncond_inner + scheduler.guidance_scale * (noise_pred_text_inner - noise_pred_uncond_inner)
                
#                 if scheduler.do_classifier_free_guidance and scheduler.guidance_rescale > 0.0:
#                     noise_pred_inner = rescale_noise_cfg(noise_pred_inner, noise_pred_text_inner, guidance_rescale=scheduler.guidance_rescale)
                
#                 # Step with deterministic noise (zero noise for deterministic sampling)
#                 temp_latents, _ = scheduler.scheduler.step(
#                     noise_pred_inner, t_inner, temp_latents, **extra_step_kwargs, return_dict=False
#                 )
            
#             # The final temp_latents is our pred_x0
#             pred_x0 = temp_latents
        
#         # Select best child after all iterations
#         if root.children:
#             best_child = max(root.children, key=lambda child: child.total_reward / child.visits if child.visits > 0 else -float('inf'))
#             latents = best_child.latents
        
#         # Callbacks
#         if callback_on_step_end is not None:
#             callback_kwargs = {}
#             for k in callback_on_step_end_tensor_inputs:
#                 callback_kwargs[k] = locals()[k]
#             callback_outputs = callback_on_step_end(scheduler, i, t, callback_kwargs)

#             latents = callback_outputs.pop("latents", latents)
#             prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
#             negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)

#         # call the callback, if provided
#         if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % scheduler.scheduler.order == 0):
#             if callback is not None and i % callback_steps == 0:
#                 step_idx = i // getattr(scheduler.scheduler, "order", 1)
#                 callback(step_idx, t, latents)

#         if XLA_AVAILABLE:
#             xm.mark_step()

#         if not output_type == "latent":
#             image = scheduler.vae.decode(latents / scheduler.vae.config.scaling_factor, return_dict=False, generator=generator)[0]
#             has_nsfw_concept = None
#         else:
#             image = latents
#             has_nsfw_concept = None
        
#         if max_score is None:
#             max_score = score_function(
#                 images = [(image * 127.5 + 128).clip(0, 255).to(torch.uint8)],
#                 prompts = [prompt],
#                 timesteps= None,
#             )
        
#         # Postprocess the image
#         if has_nsfw_concept is None:
#             do_denormalize = [True] * image.shape[0]
#         else:
#             do_denormalize = [not has_nsfw for has_nsfw in has_nsfw_concept]
        
#         image = scheduler.image_processor.postprocess(image, output_type=output_type, do_denormalize=do_denormalize)
        
#         # Offload all models
#         scheduler.maybe_free_model_hooks()
        
#         return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept), max_score

@torch.no_grad()
def eps_greedy(method, scheduler, steps, t_start=None, t_end=None, score_function, params, device):
    t_0 = 1.0 / scheduler.noise_schedule.total_N if t_end is None else t_end
    t_T = scheduler.noise_schedule.T if t_start is None else t_start

    max_score = None

    for i, t in tqdm(enumerate(steps)):
        if scheduler.interrupt:
            continue

        # expand the latents if we are doing classifier free guidance
        latent_model_input = torch.cat([latents] * 2) if scheduler.do_classifier_free_guidance else latents
        latent_model_input = scheduler.scheduler.scale_model_input(latent_model_input, t)

        # predict the noise residual
        noise_pred = scheduler.unet(
            latent_model_input,
            t,
            encoder_hidden_states=prompt_embeds,
            timestep_cond=timestep_cond,
            cross_attention_kwargs=scheduler.cross_attention_kwargs,
            added_cond_kwargs=added_cond_kwargs,
            return_dict=False,
        )[0]

        # perform guidance
        if scheduler.do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + scheduler.guidance_scale * (noise_pred_text - noise_pred_uncond)

        if scheduler.do_classifier_free_guidance and scheduler.guidance_rescale > 0.0:
            # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
            noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=scheduler.guidance_rescale)

        # compute the previous noisy sample x_t -> x_t-1

        pivot = torch.randn_like(latents)

        if method == "eps_greedy" or method == "zero_order":
            for _ in range(params['K']):
                noise_candidates = []
                for _ in range(params['N']):
                    # choose random float between 0 and 1
                    r = torch.rand(1).item()
                    if r < params['eps'] if method == "eps_greedy" else 0.0:
                        noise_candidates.append(torch.randn_like(latents))
                    else:
                        to_add = torch.randn_like(latents)
                        to_add = to_add / torch.norm(to_add)
                        noise_candidates.append(pivot + to_add * torch.rand(1).item() * params['lambda'] * np.sqrt(latents.shape[-1] * latents.shape[-2] * latents.shape[-3]))

                noise2score = {}

                for noise_candidate in noise_candidates:
                    latents_cand, pred_x0 = scheduler.scheduler.step(noise_pred, t, latents, variance_noise=noise_candidate, **extra_step_kwargs, return_dict=False)
                    

                    # expand the latents if we are doing classifier free guidance
                    latent_model_input_tminusone = torch.cat([latents_cand] * 2) if scheduler.do_classifier_free_guidance else latents_cand
                    latent_model_input_tminusone = scheduler.scheduler.scale_model_input(latent_model_input_tminusone, t)

                    # predict the noise residual
                    noise_pred_tminusone = scheduler.unet(
                        latent_model_input_tminusone,
                        t,
                        encoder_hidden_states=prompt_embeds,
                        timestep_cond=timestep_cond,
                        cross_attention_kwargs=scheduler.cross_attention_kwargs,
                        added_cond_kwargs=added_cond_kwargs,
                        return_dict=False,
                    )[0]

                    # perform guidance
                    if scheduler.do_classifier_free_guidance:
                        noise_pred_uncond_tminusone, noise_pred_text_tminusone = noise_pred_tminusone.chunk(2)
                        noise_pred_tminusone = noise_pred_uncond_tminusone + scheduler.guidance_scale * (noise_pred_text_tminusone - noise_pred_uncond_tminusone)

                    if scheduler.do_classifier_free_guidance and scheduler.guidance_rescale > 0.0:
                        # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                        noise_pred_tminusone = rescale_noise_cfg(noise_pred_tminusone, noise_pred_text, guidance_rescale=scheduler.guidance_rescale)

                    next_tminusone, pred_next_tminusone = scheduler.scheduler.step(noise_pred_tminusone, t, latents_cand, **extra_step_kwargs, return_dict=False)

                    with torch.no_grad():
                        image = scheduler.vae.decode(pred_next_tminusone / scheduler.vae.config.scaling_factor, return_dict=False, generator=generator)[0]
                        
                    # score image
                    score = score_function(
                        images = [(image * 127.5 + 128).clip(0, 255).to(torch.uint8)],
                        prompts = [prompt],
                        timesteps= None,
                    )

                    # Convert score to a Python float for reliable comparison
                    if torch.is_tensor(score):
                        # score might be a tensor of shape (1,) or scalar tensor
                        score_value = score.item()
                    else:
                        score_value = float(score)

                    noise2score[noise_candidate] = score_value
                
                # Select the noise candidate with the highest score
                max_score = max(noise2score.values())
                pivot = max(noise2score, key=noise2score.get)
        
        latents, _ = scheduler.scheduler.step(noise_pred, t, latents, variance_noise=pivot, **extra_step_kwargs, return_dict=False)

        if callback_on_step_end is not None:
            callback_kwargs = {}
            for k in callback_on_step_end_tensor_inputs:
                callback_kwargs[k] = locals()[k]
            callback_outputs = callback_on_step_end(scheduler, i, t, callback_kwargs)

            latents = callback_outputs.pop("latents", latents)
            prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
            negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)

        # call the callback, if provided
        if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % scheduler.scheduler.order == 0):
            # progress_bar.update()
            if callback is not None and i % callback_steps == 0:
                step_idx = i // getattr(scheduler.scheduler, "order", 1)
                callback(step_idx, t, latents)

        if XLA_AVAILABLE:
            xm.mark_step()

            if not output_type == "latent":
            image = scheduler.vae.decode(latents / scheduler.vae.config.scaling_factor, return_dict=False, generator=generator)[0]
            has_nsfw_concept = None
        else:
            image = latents
            has_nsfw_concept = None
        
        if max_score is None:
            max_score = score_function(
                images = [(image * 127.5 + 128).clip(0, 255).to(torch.uint8)],
                prompts = [prompt],
                timesteps= None,
            )
        
        # Postprocess the image
        if has_nsfw_concept is None:
            do_denormalize = [True] * image.shape[0]
        else:
            do_denormalize = [not has_nsfw for has_nsfw in has_nsfw_concept]
        
        image = scheduler.image_processor.postprocess(image, output_type=output_type, do_denormalize=do_denormalize)
        
        # Offload all models
        scheduler.maybe_free_model_hooks()
        
        return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept), max_score
     