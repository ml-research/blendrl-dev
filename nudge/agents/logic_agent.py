import pickle
import random
from pathlib import Path

import torch
import torch.nn as nn
from torch.distributions import Categorical
from nsfr.common import get_nsfr_model
from nsfr.utils.common import load_module
from nudge.env import NudgeBaseEnv
from nudge.torch_utils import softor


class NsfrActorCritic(nn.Module):
    def __init__(self, env: NudgeBaseEnv, rules: str, device, rng=None):
        super(NsfrActorCritic, self).__init__()
        self.device = device
        self.rng = random.Random() if rng is None else rng
        self.env = env
        self.actor = get_nsfr_model(env.name, rules, device=device, train=True)
        self.prednames = self.get_prednames()

        mlp_module_path = f"in/envs/{self.env.name}/mlp.py"
        module = load_module(mlp_module_path)
        self.critic = module.MLP(device=device, out_size=1, logic=True)

        self.num_actions = len(self.prednames)
        self.uniform = Categorical(
            torch.tensor([1.0 / self.num_actions for _ in range(self.num_actions)], device=device))
        self.upprior = Categorical(
            torch.tensor([0.9] + [0.1 / (self.num_actions - 1) for _ in range(self.num_actions - 1)], device=device))

        self.env_action_id_to_action_pred_indices = self._build_action_id_dict()

    def forward(self):
        raise NotImplementedError

    def _print(self):
        print("==== Logic Policy ====")
        self.actor.print_program()

    def act(self, logic_state, epsilon=0.0):
        action_probs = self.actor(logic_state)

        # e-greedy
        if self.rng.random() < epsilon:
            # random action with epsilon probability
            dist = self.uniform
            action = dist.sample()
        else:
            dist = Categorical(action_probs)
            action = (action_probs[0] == max(action_probs[0])).nonzero(as_tuple=True)[0].squeeze(0).to(self.device)
            if torch.numel(action) > 1:
                action = action[0]
        # action = dist.sample()
        action_logprob = dist.log_prob(action)
        return action.detach(), action  # action_logprob.detach()

    def evaluate(self, neural_state, logic_state, action):
        action_probs = self.actor(logic_state)
        dist = Categorical(action_probs)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(neural_state)

        return action_logprobs, state_values, dist_entropy

    def get_prednames(self):
        return self.actor.get_prednames()

    def get_action_and_value(self, neural_state, logic_state, action=None):
        # compute action
        # n_envs * n_actions

        action_probs = self.to_action_distribution(self.actor(logic_state))
        # action_probs  = self.actor(logic_state)
        dist = Categorical(action_probs)
        if action is None:
            action = dist.sample()
        logprob = dist.log_prob(action)

        # compute value
        # batch_size * 1
        value = self.critic(logic_state)

        return action, logprob, dist.entropy(), value

    def get_value(self, neural_state, logic_state):
        _, _, _, value = self.get_action_and_value(neural_state, logic_state)
        return value

    def save(self, checkpoint_path, directory: Path, step_list, reward_list, weight_list):
        torch.save(self.state_dict(), checkpoint_path)
        with open(directory / "data.pkl", "wb") as f:
            pickle.dump(step_list, f)
            pickle.dump(reward_list, f)
            pickle.dump(weight_list, f)

    def _build_action_id_dict(self):
        env_action_names = list(self.env.pred2action.keys())
        # action_probs = torch.zeros(len(env_action_names))
        env_action_id_to_action_pred_indices = {}
        for i, env_action_name in enumerate(env_action_names):
            env_action_id_to_action_pred_indices[i] = []

        for i, env_action_name in enumerate(env_action_names):
            for j, action_pred_name in enumerate(self.actor.get_prednames()):
                if env_action_name in action_pred_name:
                    env_action_id_to_action_pred_indices[i].append(j)
            if not env_action_id_to_action_pred_indices[i]:
                dummy_index = len(self.actor.get_prednames())
                env_action_id_to_action_pred_indices[i].append(dummy_index)

        return env_action_id_to_action_pred_indices

    def to_action_distribution(self, raw_action_probs):
        """Converts raw action probabilities to a distribution."""
        batch_size = raw_action_probs.size(0)
        env_action_names = list(self.env.pred2action.keys())

        raw_action_probs = torch.cat([raw_action_probs, torch.zeros(batch_size, 1, device=self.device)], dim=1)
        # save raw_action_probs for explanations (attributions)
        self.raw_action_probs = raw_action_probs
        raw_action_logits = torch.logit(raw_action_probs, eps=0.01)
        dist_values = []
        for i in range(len(env_action_names)):
            if i in self.env_action_id_to_action_pred_indices:
                indices = torch.tensor(self.env_action_id_to_action_pred_indices[i], device=self.device) \
                    .expand(batch_size, -1).to(self.device)
                gathered = torch.gather(raw_action_logits, 1, indices)
                # merged value for i-th action for samples in the batch
                merged = softor(gathered, dim=1)  # (batch_size, 1)
                dist_values.append(merged)

        action_values = torch.stack(dist_values, dim=1)  # (batch_size, n_actions)
        action_dist = torch.softmax(action_values, dim=1)
        action_dist = self.reshape_action_distribution(action_dist)
        return action_dist

    def reshape_action_distribution(self, action_dist):
        batch_size = action_dist.size(0)
        if action_dist.size(1) < self.env.n_raw_actions:
            zeros = torch.zeros(batch_size, self.env.n_raw_actions - action_dist.size(1), device=self.device,
                                requires_grad=True)
            action_dist = torch.cat([action_dist, zeros], dim=1)
        return action_dist
