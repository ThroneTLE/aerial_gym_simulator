"""
Custom A2C Agent with Auxiliary Loss for Privileged Encoder.

This agent extends the standard A2CAgent to add an auxiliary loss
that forces the latent vector to predict payload physical properties.
"""

import torch
from rl_games.algos_torch import a2c_continuous
from rl_games.algos_torch import torch_ext
from rl_games.common import common_losses


class A2CAgentWithAuxLoss(a2c_continuous.A2CAgent):
    """
    A2CAgent with auxiliary loss for privileged encoder.
    
    The auxiliary loss forces the 8-dim latent to predict payload properties:
    [payload_mass, com_x, com_y, com_z, released_mass]
    
    This prevents the latent from being ignored by the policy network.
    """
    
    def __init__(self, base_name, params):
        super().__init__(base_name, params)
        # Get aux loss weight from config
        self.aux_loss_coef = params.get('config', {}).get('aux_loss_coef', 0.1)
        print(f"[A2CAgentWithAuxLoss] Auxiliary loss coefficient: {self.aux_loss_coef}")
    
    def calc_gradients(self, input_dict):
        """Override to add auxiliary loss."""
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
            
            losses, sum_mask = torch_ext.apply_masks([a_loss.unsqueeze(1), c_loss, entropy.unsqueeze(1), b_loss.unsqueeze(1)], rnn_masks)
            a_loss, c_loss, entropy, b_loss = losses[0], losses[1], losses[2], losses[3]

            # Add auxiliary loss to total loss
            # Handle None bounds_loss_coef
            bounds_coef = self.bounds_loss_coef if self.bounds_loss_coef is not None else 0.0
            loss = a_loss + 0.5 * c_loss * self.critic_coef - entropy * self.entropy_coef + b_loss * bounds_coef
            loss = loss + aux_loss * self.aux_loss_coef
            
            if self.multi_gpu:
                self.optimizer.zero_grad()
            else:
                for param in self.model.parameters():
                    param.grad = None

        self.scaler.scale(loss).backward()
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

        # Store aux_loss for logging
        self._aux_loss = aux_loss.item() if torch.is_tensor(aux_loss) else aux_loss

        self.train_result = (a_loss, c_loss, entropy, \
            kl_dist, self.last_lr, lr_mul, \
            mu.detach(), sigma.detach(), b_loss)
    
    def train_epoch(self):
        """Override to log auxiliary loss."""
        result = super().train_epoch()
        
        # Log aux loss to tensorboard if available
        if hasattr(self, '_aux_loss') and self.writer is not None:
            self.writer.add_scalar('losses/aux_loss', self._aux_loss, self.epoch_num)
        
        return result
