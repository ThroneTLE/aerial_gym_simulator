"""
Custom A2C Agent with Auxiliary Loss and BC (Behavior Cloning) Loss.

This agent extends the standard A2CAgent to add:
1. Auxiliary loss that forces the latent vector to predict payload physical properties
2. BC loss that encourages policy to imitate teacher actions
"""

import torch
import torch.nn.functional as F
from rl_games.algos_torch import a2c_continuous
from rl_games.algos_torch import torch_ext
from rl_games.common import common_losses
from rl_games.common.experience import ExperienceBuffer
from rl_games.common.a2c_common import swap_and_flatten01
from collections import defaultdict


class A2CAgentWithAuxLoss(a2c_continuous.A2CAgent):
    """
    A2CAgent with auxiliary loss and BC (behavior cloning) loss.
    
    The auxiliary loss forces the 8-dim latent to predict payload properties:
    [payload_mass, com_x, com_y, com_z, released_mass]
    
    The BC loss encourages the policy to match teacher actions:
    BC_loss = MSE(policy_action, teacher_action)
    
    BC coefficient decays: bc_coef * (bc_alpha ^ n_updates) clamped to bc_min_coef
    """
    
    def __init__(self, base_name, params):
        super().__init__(base_name, params)
        config = params.get('config', {})
        
        # Get aux loss weight from config
        self.aux_loss_coef = config.get('aux_loss_coef', 0.1)
        
        # BC loss parameters with decay
        self.bc_coef = config.get('bc_coef', 0.1)
        self.bc_alpha = config.get('bc_alpha', 0.999)
        self.bc_min_coef = config.get('bc_min_coef', 0.01)
        self._bc_update_count = 0
        
        # Storage for teacher actions during rollout
        self._teacher_actions_buffer = None
        
        # NaN Prevention: gradient NaN detection and recovery
        self.skip_nan_gradients = config.get('skip_nan_gradients', True)
        self.reset_optimizer_on_persistent_nan = config.get('reset_optimizer_on_persistent_nan', True)
        self.nan_tolerance = config.get('nan_tolerance', 5)
        self.nan_skip_count = 0
        self.total_nan_skips = 0
        
        print(f"[A2CAgentWithAuxLoss] Auxiliary loss coefficient: {self.aux_loss_coef}")
        print(f"[A2CAgentWithAuxLoss] BC loss: coef={self.bc_coef}, alpha={self.bc_alpha}, min={self.bc_min_coef}")
        print(f"[A2CAgentWithAuxLoss] NaN防护: skip={self.skip_nan_gradients}, reset={self.reset_optimizer_on_persistent_nan}, tolerance={self.nan_tolerance}")
        
    def init_tensors(self):
        """Override to register teacher_actions and privileged_obs in experience_buffer."""
        super().init_tensors()
        
        # Register teacher_actions in experience_buffer for proper minibatch generation
        action_dim = self.actions_num
        self.experience_buffer.tensor_dict['teacher_actions'] = torch.zeros(
            (self.horizon_length, self.num_actors, action_dim),
            dtype=torch.float32,
            device=self.ppo_device
        )
        # Add to tensor_list so it's included in get_transformed_list
        if 'teacher_actions' not in self.tensor_list:
            self.tensor_list.append('teacher_actions')
        print(f"[A2CAgentWithAuxLoss] Registered teacher_actions in experience_buffer (shape: [horizon={self.horizon_length}, actors={self.num_actors}, action_dim={action_dim}])")
        
        # Register privileged_obs in experience_buffer for aux loss computation
        # Get privileged_obs dim from config or infer from first obs
        priv_dim = self.network.priv_dim if hasattr(self.network, 'priv_dim') else 7
        self.experience_buffer.tensor_dict['privileged_obs'] = torch.zeros(
            (self.horizon_length, self.num_actors, priv_dim),
            dtype=torch.float32,
            device=self.ppo_device
        )
        if 'privileged_obs' not in self.tensor_list:
            self.tensor_list.append('privileged_obs')
        print(f"[A2CAgentWithAuxLoss] Registered privileged_obs in experience_buffer (shape: [horizon={self.horizon_length}, actors={self.num_actors}, priv_dim={priv_dim}])")
    
    def _get_current_bc_coef(self):
        """Get current BC coefficient with exponential decay."""
        decayed = self.bc_coef * (self.bc_alpha ** self._bc_update_count)
        return max(self.bc_min_coef, decayed)
    
    def play_steps(self):
        """Override to collect teacher_actions during rollout."""
        # Initialize teacher actions buffer for this rollout
        teacher_actions_list = []
        
        # Store original update_list and dataset creation
        update_list = self.update_list
        
        step_time = 0.0

        for n in range(self.horizon_length):
            if self.use_action_masks:
                masks = self.vec_env.get_action_masks()
                res_dict = self.get_masked_action_values(self.obs, masks)
            else:
                res_dict = self.get_action_values(self.obs)
            self.experience_buffer.update_data('obses', n, self.obs['obs'])
            self.experience_buffer.update_data('dones', n, self.dones)
            
            # Store privileged_obs for aux loss computation during training
            priv_obs = self.obs.get('privileged_obs', None)
            if priv_obs is not None:
                self.experience_buffer.update_data('privileged_obs', n, priv_obs)
            else:
                # Fill with zeros if not available
                priv_shape = self.experience_buffer.tensor_dict['privileged_obs'].shape[-1]
                zeros = torch.zeros((self.num_actors, priv_shape), device=self.ppo_device)
                self.experience_buffer.update_data('privileged_obs', n, zeros)

            for k in update_list:
                self.experience_buffer.update_data(k, n, res_dict[k]) 
            if self.has_central_value:
                self.experience_buffer.update_data('states', n, self.obs['states'])

            step_time_start = torch.cuda.Event(enable_timing=True)
            step_time_end = torch.cuda.Event(enable_timing=True)
            step_time_start.record()

            self.obs, rewards, self.dones, infos = self.env_step(res_dict['actions'])
            step_time_end.record()
            torch.cuda.synchronize()
            step_time += step_time_start.elapsed_time(step_time_end)

            shaped_rewards = self.rewards_shaper(rewards)
            if self.value_bootstrap and 'time_outs' in infos:
                shaped_rewards += self.gamma * res_dict['values'] * self.cast_obs(infos['time_outs']).unsqueeze(1).float()

            self.experience_buffer.update_data('rewards', n, shaped_rewards)
            
            # Store teacher_actions in experience_buffer
            teacher_actions_for_buffer = None
            if isinstance(self.obs, dict) and 'teacher_actions' in self.obs and self.obs['teacher_actions'] is not None:
                teacher_actions_for_buffer = self.obs['teacher_actions']
            elif 'teacher_actions' in infos:
                teacher_actions_for_buffer = infos['teacher_actions']
            
            if teacher_actions_for_buffer is not None:
                self.experience_buffer.update_data('teacher_actions', n, teacher_actions_for_buffer)
                teacher_actions_list.append(teacher_actions_for_buffer.clone())
            else:
                # Fill with zeros if no teacher actions (shouldn't happen in teacher mode)
                zeros = torch.zeros((self.num_actors, self.actions_num), device=self.ppo_device)
                self.experience_buffer.update_data('teacher_actions', n, zeros)

            self.current_rewards += rewards
            self.current_shaped_rewards += shaped_rewards
            self.current_lengths += 1
            all_done_indices = self.dones.nonzero(as_tuple=False)
            done_indices = all_done_indices[::self.num_agents]
  
            self.game_rewards.update(self.current_rewards[done_indices])
            self.game_shaped_rewards.update(self.current_shaped_rewards[done_indices])
            self.game_lengths.update(self.current_lengths[done_indices])
            self.algo_observer.process_infos(infos, done_indices)
            
            not_dones = 1.0 - self.dones.float()

            self.current_rewards = self.current_rewards * not_dones.unsqueeze(1)
            self.current_shaped_rewards = self.current_shaped_rewards * not_dones.unsqueeze(1)
            self.current_lengths = self.current_lengths * not_dones

        self.last_lr, self.entropy_coef = self.scheduler.update(self.last_lr, self.entropy_coef, self.epoch_num, 0, 0)
        
        # Get last values for GAE calculation
        with torch.no_grad():
            if self.has_central_value:
                self.last_values = self.get_central_value(self.obs)
            else:
                res_dict = self.get_action_values(self.obs)
                self.last_values = res_dict['values']
        
            # Store teacher actions if available
        if teacher_actions_list:
            # Stack along horizon dimension: (horizon, num_envs, action_dim)
            self._teacher_actions_buffer = torch.stack(teacher_actions_list, dim=0)
        else:
            self._teacher_actions_buffer = None
        self._has_teacher_actions = bool(teacher_actions_list)

        # GAE calculation
        fdones = self.dones.float()
        mb_fdones = self.experience_buffer.tensor_dict['dones'].float()
        mb_values = self.experience_buffer.tensor_dict['values']
        mb_rewards = self.experience_buffer.tensor_dict['rewards']
        mb_advs = self.discount_values(fdones, self.last_values, mb_fdones, mb_values, mb_rewards)
        mb_returns = mb_advs + mb_values

        batch_dict = self.experience_buffer.get_transformed_list(swap_and_flatten01, self.tensor_list)
        batch_dict['returns'] = swap_and_flatten01(mb_returns)
        batch_dict['played_frames'] = self.batch_size
        batch_dict['step_time'] = step_time
        
        # Add teacher_actions to batch_dict so they are included in dataset
        if self._teacher_actions_buffer is not None and 'teacher_actions' not in batch_dict:
            batch_dict['teacher_actions'] = swap_and_flatten01(self._teacher_actions_buffer)

        return batch_dict

    def prepare_dataset(self, batch_dict):
        """Override to include teacher_actions and privileged_obs in dataset."""
        # super().prepare_dataset(batch_dict) returns None, it sets self.dataset internally
        super().prepare_dataset(batch_dict)
        
        # Fallback: ensure teacher_actions are in dataset if not added by super()
        # This handles cases where super() creates a new dataset object ignoring extra keys in batch_dict
        if self._teacher_actions_buffer is not None:
             self._teacher_actions_flat = swap_and_flatten01(self._teacher_actions_buffer)
             
             # Try to add to dataset if missing
             key_missing = False
             if hasattr(self, 'dataset'):
                 if isinstance(self.dataset, dict):
                     if 'teacher_actions' not in self.dataset:
                         key_missing = True
                 elif hasattr(self.dataset, 'keys') and 'teacher_actions' not in self.dataset.keys():
                      key_missing = True
                 
                 if key_missing:
                    if isinstance(self.dataset, dict):
                        self.dataset['teacher_actions'] = self._teacher_actions_flat
                    else:
                        try:
                            self.dataset['teacher_actions'] = self._teacher_actions_flat
                        except TypeError:
                            if hasattr(self.dataset, 'update'):
                                self.dataset.update({'teacher_actions': self._teacher_actions_flat})
        else:
            self._teacher_actions_flat = None
        
        # Store privileged_obs flat for minibatch access in train_actor_critic
        if 'privileged_obs' in batch_dict:
            self._privileged_obs_flat = batch_dict['privileged_obs']
        else:
            self._privileged_obs_flat = None

    def train_actor_critic(self, input_dict):
        """Override to pass teacher_actions and privileged_obs to calc_gradients via input_dict."""
        # Add teacher_actions to input_dict if available
        teacher_actions = input_dict.get('teacher_actions', None)
        if not getattr(self, '_has_teacher_actions', False):
            teacher_actions = None
        elif teacher_actions is None and hasattr(self, '_teacher_actions_flat') and self._teacher_actions_flat is not None:
            curr_idx = input_dict.get('idx', None)
            if curr_idx is not None:
                teacher_actions = self._teacher_actions_flat[curr_idx]
        input_dict['teacher_actions'] = teacher_actions
        
        # Add privileged_obs to input_dict if available (same pattern as teacher_actions)
        privileged_obs = input_dict.get('privileged_obs', None)
        if privileged_obs is None and hasattr(self, '_privileged_obs_flat') and self._privileged_obs_flat is not None:
            curr_idx = input_dict.get('idx', None)
            if curr_idx is not None:
                privileged_obs = self._privileged_obs_flat[curr_idx]
        input_dict['privileged_obs'] = privileged_obs
        
        self.calc_gradients(input_dict)
        return self.train_result
    
    def calc_gradients(self, input_dict):
        """Override to add auxiliary loss and BC loss."""
        value_preds_batch = input_dict['old_values']
        old_action_log_probs_batch = input_dict['old_logp_actions']
        advantage = input_dict['advantages']
        old_mu_batch = input_dict['mu']
        old_sigma_batch = input_dict['sigma']
        return_batch = input_dict['returns']
        actions_batch = input_dict['actions']
        obs_batch = input_dict['obs']
        obs_batch = self._preproc_obs(obs_batch)

        lr_mul = 1.0
        curr_e_clip = self.e_clip

        batch_dict = {
            'is_train': True,
            'prev_actions': actions_batch, 
            'obs' : obs_batch,
        }
        
        # Add privileged_obs if available (for aux loss computation)
        if 'privileged_obs' in input_dict:
            batch_dict['privileged_obs'] = input_dict['privileged_obs']

        rnn_masks = None
        if self.is_rnn:
            rnn_masks = input_dict['rnn_masks']
            batch_dict['rnn_states'] = input_dict['rnn_states']
            batch_dict['seq_length'] = self.seq_length

            if self.zero_rnn_on_done:
                batch_dict['dones'] = input_dict['dones']            

        with torch.cuda.amp.autocast(enabled=self.mixed_precision):
            res_dict = self.model(batch_dict)
            action_log_probs = res_dict['prev_neglogp']
            values = res_dict['values']
            entropy = res_dict['entropy']
            mu = res_dict['mus']
            sigma = res_dict['sigmas']

            a_loss = self.actor_loss_func(old_action_log_probs_batch, action_log_probs, advantage, self.ppo, curr_e_clip)

            if self.has_value_loss:
                c_loss = common_losses.critic_loss(self.model, value_preds_batch, values, curr_e_clip, return_batch, self.clip_value)
            else:
                c_loss = torch.zeros(1, device=self.ppo_device)
            if self.bound_loss_type == 'regularisation':
                b_loss = self.reg_loss(mu)
            elif self.bound_loss_type == 'bound':
                b_loss = self.bound_loss(mu)
            else:
                b_loss = torch.zeros(1, device=self.ppo_device)
            
            # Ensure b_loss is a tensor (reg_loss/bound_loss may return int 0)
            if not torch.is_tensor(b_loss):
                b_loss = torch.zeros(1, device=self.ppo_device)
            
            # === AUXILIARY LOSS ===
            aux_loss = torch.zeros(1, device=self.ppo_device)
            if hasattr(self.model, 'a2c_network') and hasattr(self.model.a2c_network, 'get_aux_loss'):
                aux_loss = self.model.a2c_network.get_aux_loss()
                if not torch.is_tensor(aux_loss):
                    aux_loss = torch.tensor(aux_loss, device=self.ppo_device)
                aux_loss = aux_loss.mean()  # Ensure scalar
            # ======================
            
            # === BC LOSS ===
            bc_loss = torch.zeros(1, device=self.ppo_device)
            teacher_actions = input_dict.get('teacher_actions', None)

            if teacher_actions is not None:
                if not torch.is_tensor(teacher_actions):
                    teacher_actions = torch.as_tensor(
                        teacher_actions, device=mu.device, dtype=mu.dtype
                    )
                else:
                    teacher_actions = teacher_actions.to(device=mu.device, dtype=mu.dtype)
                if teacher_actions.shape != mu.shape and teacher_actions.numel() == mu.numel():
                    teacher_actions = teacher_actions.view_as(mu)
                if teacher_actions.shape == mu.shape:
                    # MSE between clamped policy mean and teacher actions
                    # 关键修复：对 mu 做 clamp，使 BC loss 与 imitation error 度量一致
                    mu_clamped = torch.clamp(mu, -1.0, 1.0)
                    bc_loss = F.mse_loss(mu_clamped, teacher_actions)
            # ==============
            
            losses, sum_mask = torch_ext.apply_masks([a_loss.unsqueeze(1), c_loss, entropy.unsqueeze(1), b_loss.unsqueeze(1)], rnn_masks)
            a_loss, c_loss, entropy, b_loss = losses[0], losses[1], losses[2], losses[3]

            # Add auxiliary loss and BC loss to total loss
            # Handle None bounds_loss_coef
            bounds_coef = self.bounds_loss_coef if self.bounds_loss_coef is not None else 0.0
            current_bc_coef = self._get_current_bc_coef()
            
            loss = a_loss + 0.5 * c_loss * self.critic_coef - entropy * self.entropy_coef + b_loss * bounds_coef
            loss = loss + aux_loss * self.aux_loss_coef
            loss = loss + bc_loss * current_bc_coef
            
            if self.multi_gpu:
                self.optimizer.zero_grad()
            else:
                for param in self.model.parameters():
                    param.grad = None

        self.scaler.scale(loss).backward()
        
        # NaN Prevention: Check gradients before optimizer step
        grad_finite = self._check_and_handle_nan_gradients()
        
        if grad_finite:
            self.trancate_gradients_and_step()
        elif self.skip_nan_gradients:
            # Skip this update, clear gradients
            self.optimizer.zero_grad()
            print(f"[NaN警告] Epoch {self.epoch_num}: 梯度包含NaN/Inf，跳过本次更新 (累计跳过: {self.total_nan_skips})")
        else:
            # Original behavior: proceed anyway (may cause crash)
            self.trancate_gradients_and_step()

        with torch.no_grad():
            reduce_kl = rnn_masks is None
            kl_dist = torch_ext.policy_kl(mu.detach(), sigma.detach(), old_mu_batch, old_sigma_batch, reduce_kl)
            if rnn_masks is not None:
                kl_dist = (kl_dist * rnn_masks).sum() / rnn_masks.numel()

        self.diagnostics.mini_batch(self,
        {
            'values' : value_preds_batch,
            'returns' : return_batch,
            'new_neglogp' : action_log_probs,
            'old_neglogp' : old_action_log_probs_batch,
            'masks' : rnn_masks
        }, curr_e_clip, 0)      

        # Store losses for logging
        self._aux_loss = aux_loss.item() if torch.is_tensor(aux_loss) else aux_loss
        self._bc_loss = bc_loss.item() if torch.is_tensor(bc_loss) else bc_loss
        self._current_bc_coef = current_bc_coef

        self.train_result = (a_loss, c_loss, entropy, \
            kl_dist, self.last_lr, lr_mul, \
            mu.detach(), sigma.detach(), b_loss)
    
    def train_epoch(self):
        """Override to log auxiliary loss and BC loss, and update BC coefficient."""
        result = super().train_epoch()
        
        # Increment BC update counter for decay
        self._bc_update_count += 1
        
        # Log losses to tensorboard if available
        if self.writer is not None:
            if hasattr(self, '_aux_loss'):
                self.writer.add_scalar('losses/aux_loss', self._aux_loss, self.epoch_num)
            if hasattr(self, '_bc_loss'):
                self.writer.add_scalar('losses/bc_loss', self._bc_loss, self.epoch_num)
            if hasattr(self, '_current_bc_coef'):
                self.writer.add_scalar('losses/bc_coef', self._current_bc_coef, self.epoch_num)
        
        return result
    
    def _check_and_handle_nan_gradients(self):
        """
        Check if gradients contain NaN/Inf and handle accordingly.
        
        Returns:
            True if gradients are finite, False if NaN/Inf detected
        """
        if not self.skip_nan_gradients:
            return True  # Skip check if disabled
        
        # Check all parameter gradients
        has_nan = False
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                if not torch.isfinite(param.grad).all():
                    has_nan = True
                    nan_count = torch.isnan(param.grad).sum().item()
                    inf_count = torch.isinf(param.grad).sum().item()
                    print(f"  - NaN/Inf in gradient '{name}': NaN={nan_count}, Inf={inf_count}")
                    break  # Stop checking after first NaN found
        
        if has_nan:
            self.nan_skip_count += 1
            self.total_nan_skips += 1
            
            # Log to tensorboard if available
            if self.writer is not None:
                self.writer.add_scalar('debug/nan_skip_count', self.total_nan_skips, self.epoch_num)
            
            # Reset optimizer state if persistent NaN
            if self.reset_optimizer_on_persistent_nan and self.nan_skip_count >= self.nan_tolerance:
                print(f"[NaN Recovery] 连续{self.nan_skip_count}次NaN，重置优化器状态")
                # Reset Adam state (clear momentum and variance)
                self.optimizer.state = defaultdict(dict)
                self.nan_skip_count = 0
                
                if self.writer is not None:
                    self.writer.add_scalar('debug/optimizer_reset_count', 1, self.epoch_num)
            
            return False
        else:
            # Reset consecutive NaN counter on successful update
            if self.nan_skip_count > 0:
                self.nan_skip_count = 0
            return True
