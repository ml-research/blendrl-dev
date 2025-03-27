import random
import pickle
from pathlib import Path
import os


import torch
import torch.nn as nn
import torch.nn.functional as F
from nudge.agents.logic_agent import NsfrActorCritic
from nudge.agents.neural_agent import NeuralPPO, ActorCritic
from nudge.torch_utils import softor

# from nudge.env import NudgeBaseEnv
from torch.distributions.categorical import Categorical


from torch.distributions import Categorical
from nsfr.utils.common import load_module
from nsfr.common import get_nsfr_model

from utils import get_blender, load_cleanrl_agent
from nudge.utils import print_program

from captum.attr import (
    GradientShap,
    DeepLift,
    DeepLiftShap,
    IntegratedGradients,
    LayerConductance,
    NeuronConductance,
    NoiseTunnel,
)


class BlenderActor(nn.Module):
    """
    BlendeRL actor that combines neural and logic policies.

    Args:
        env: environment
        neural_actor: neural policy
        logic_actor: logic policy
        blender: blending policy
        actor_mode: actor mode, one of ["hybrid", "logic", "neural"]
        blender_mode: blender mode, one of ["logic", "neural"]
        blend_function: blending function, one of ["softmax", "gumbel_softmax"]
        device: device
    """

    def __init__(
        self,
        env,
        neural_actor,
        logic_actor,
        blender,
        actor_mode,
        blender_mode,
        blend_function,
        device=None,
        explain=False,
    ):
        """
        Initialize a BlendeRL agent.
        Args:
            env: environment
            neural_actor: neural policy
            logic_actor: logic policy
            blender: blending policy
            actor_mode: actor mode, one of ["hybrid", "logic", "neural"]
            blender_mode: blender mode, one of ["logic", "neural"]
            blend_function: blending function, one of ["softmax", "gumbel_softmax"]
            device: device
        """
        super(BlenderActor, self).__init__()
        self.env = env
        self.neural_actor = neural_actor
        self.logic_actor = logic_actor
        self.blender = blender
        self.actor_mode = actor_mode
        self.blender_mode = blender_mode
        self.blend_function = blend_function
        self.device = device
        self.explain = explain
        self.env_action_id_to_action_pred_indices = self._build_action_id_dict()

    def _build_action_id_dict(self):
        """
        Initialize a dictionary that maps environment action id to predicate indices.
        Returns:
            env_action_id_to_action_pred_indices: dictionary that maps environment action id to predicate indices
        """
        env_action_names = list(self.env.pred2action.keys())
        # action_probs = torch.zeros(len(env_action_names))
        env_action_id_to_action_pred_indices = {}
        # init dic
        for i, env_action_name in enumerate(env_action_names):
            env_action_id_to_action_pred_indices[i] = []

        for i, env_action_name in enumerate(env_action_names):
            exist_flag = False
            for j, action_pred_name in enumerate(self.logic_actor.get_prednames()):
                if env_action_name in action_pred_name:
                    # if i not in env_action_id_to_action_pred_indices:
                    #    env_action_id_to_action_pred_indices[i] = []
                    env_action_id_to_action_pred_indices[i].append(j)
                    exist_flag = True
            if not exist_flag:
                # i-th env action is not defined by any rules thus will be always 0.0
                # refer to dummy predicte index
                # pred1, pred2, ..., predn, dummy_pred
                dummy_index = len(self.logic_actor.get_prednames())
                env_action_id_to_action_pred_indices[i].append(dummy_index)
        return env_action_id_to_action_pred_indices

    def get_explanation(self, neural_state, logic_state, action):
        """
        Get the explanation of the blending weights.

        Args:
            neural_state: neural state
            logic_state: logic state
        """
        neural_explanation = self.get_neural_explanation(neural_state, action)
        logic_explanation = self.get_logic_explanation(logic_state, action)
        # blend??
        weights = self.to_blender_policy_distribution(neural_state, logic_state)[0]
        # blended_explanation = weights[0, 0] * neural_explanation + weights[0, 1] * logic_explanation
        return neural_explanation, logic_explanation, weights.detach().cpu().numpy()

    def get_neural_explanation(self, neural_state, action):
        self.neural_actor.eval()
        baseline = torch.zeros_like(neural_state).to(self.device)
        ig = IntegratedGradients(self.neural_actor)
        attributions, delta = ig.attribute(
            neural_state, baseline, target=action, return_convergence_delta=True
        )
        minimum = attributions.min()
        maximum = attributions.max()
        attributions = (attributions - minimum) / (maximum - minimum)
        return attributions

    def get_logic_explanation(self, logic_state, action):
        # self.logic_action_probs.max().backward()
        # self.logic_actor.V_T.max().backward()
        self.raw_action_probs.max().backward()
        # print(self.logic_action_probs, self.logic_action_probs.max())
        atom_attributes = self.logic_actor.dummy_zeros.grad
        # normalize to [0, 1]
        minimum = atom_attributes.min()
        maximum = atom_attributes.max()
        atom_attributes = (atom_attributes - minimum) / (maximum - minimum)
        new_minimum = atom_attributes.min()
        new_maximum = atom_attributes.max()
        if atom_attributes.max() > 1.0:
            pass
        # atom_attributes = atom_attributes / atom_attributes.max()
        # print(atom_attributes)
        # self.logic_actor.print_valuations(min_value=0.5)
        # self.logic_actor.print_valuations_input(atom_attributes, min_value=0.5)
        self.logic_actor.dummy_zeros.grad.zero_()
        return atom_attributes

    def get_logic_explanation_IG(self, logic_state, action):
        self.logic_actor.eval()
        baseline = torch.zeros_like(logic_state).to(self.device)
        # env action to predicate indices
        # indices = self.env_action_id_to_action_pred_indices[action.item()]
        # target_pred = self.logic_actor(logic_state)[indices].max().item()
        logic_pred_probs = self.logic_actor(logic_state)
        target_pred = torch.argmax(logic_pred_probs).item()

        ig = IntegratedGradients(self.logic_actor)
        attributions, delta = ig.attribute(
            logic_state, baseline, target=target_pred, return_convergence_delta=True
        )
        return attributions

    def compute_action_probs_hybrid(self, neural_state, logic_state):
        """
        Compute action probabilities for hybrid actor.
        Args:
            neural_state: neural state
            logic_state: logic state
        Returns:
            action_probs: action probabilities
            weights: blending weights
        """
        # state size: B * N
        batch_size = neural_state.size(0)
        logic_action_probs = self.to_action_distribution(self.logic_actor(logic_state))
        neural_action_probs = self.to_neural_action_distribution(neural_state)
        self.logic_action_probs = logic_action_probs
        self.neural_action_probs = neural_action_probs
        # action_probs size : B * N_actions
        batch_size = neural_state.size(0)
        # weights size: B * 2
        weights = self.to_blender_policy_distribution(neural_state, logic_state)
        # save weights: w1 and w2
        self.w_policy = weights[0]
        n_actions = neural_action_probs.size(1)
        # expanded weights size: B * N_actions * 2
        weights_expanded = weights.unsqueeze(1).repeat(1, n_actions, 1)
        # p_action = w1 * p_neural + w2 * p_logic
        action_probs = (
            weights_expanded[:, :, 0] * neural_action_probs
            + weights_expanded[:, :, 1] * logic_action_probs
        )
        return action_probs, weights

    def compute_action_probs_logic(self, logic_state):
        """
        Compute action probabilities using only logic actor.

        Args:
            logic_state: logic state
        Returns:
            action_probs: action probabilities
            weights: blending weights [0.0, 1.0]
        """
        self.w_policy = torch.tensor([0.0, 1.0], device=self.device)
        logic_action_probs = self.to_action_distribution(self.logic_actor(logic_state))
        neural_action_probs = torch.zeros_like(logic_action_probs).to(
            device=self.device
        )
        self.logic_action_probs = logic_action_probs
        self.neural_action_probs = neural_action_probs
        return logic_action_probs, torch.tensor(
            [0.0, 1.0], device=self.device
        ).unsqueeze(0).expand(logic_state.size(0), -1)

    def compute_action_probs_neural(self, neural_state):
        """
        Compute action probabilities using only neural actor.

        Args:
            neural_state: neural state
        Returns:
            action_probs: action probabilities
            weights: blending weights [1.0, 0.0]
        """
        self.w_policy = torch.tensor([1.0, 0.0], device=self.device)
        neural_action_probs = self.to_neural_action_distribution(neural_state)
        logic_action_probs = torch.zeros_like(neural_action_probs).to(
            device=self.device
        )
        self.neural_action_probs = neural_action_probs
        self.logic_action_probs = logic_action_probs
        return neural_action_probs, torch.tensor(
            [1.0, 0.0], device=self.device
        ).unsqueeze(0).expand(neural_state.size(0), -1)

    def to_blender_policy_distribution(self, neural_state, logic_state):
        """
        Merge neural and logic policies using the blender funciton.

        Args:
            neural_state: neural state
            logic_state: logic state
        Returns:
            action_probs: action probabilities
        """
        # get prob for neural and logic policy
        # probs = extract_policy_probs(self.blender, V_T, self.device)
        # to logit
        assert self.blender_mode in [
            "logic",
            "neural",
        ], "Invalid blender mode {}".format(self.blender_mode)
        assert self.blend_function in [
            "softmax",
            "gumbel_softmax",
        ], "Invalid blend function {}".format(self.blend_function)

        if self.blender_mode == "logic":
            policy_probs = self.blender(logic_state)
        else:
            policy_probs = self.blender(neural_state)

        logits = torch.logit(policy_probs, eps=0.01)
        if self.blend_function == "softmax":
            return torch.softmax(logits, dim=1)
        else:
            return F.gumbel_softmax(logits, dim=1)

    def to_action_distribution(self, raw_action_probs):
        """
        Converts raw action probabilities to a distribution.

        Args:
            raw_action_probs: raw action scores

        Returns:
            action_dist: action distribution
        """
        batch_size = raw_action_probs.size(0)
        env_action_names = list(self.env.pred2action.keys())

        raw_action_probs = torch.cat(
            [raw_action_probs, torch.zeros(batch_size, 1, device=self.device)], dim=1
        )
        # save raw_action_probs for explanations (attributions)
        self.raw_action_probs = raw_action_probs
        raw_action_logits = torch.logit(raw_action_probs, eps=0.01)
        dist_values = []
        for i in range(len(env_action_names)):
            if i in self.env_action_id_to_action_pred_indices:
                indices = torch.tensor(self.env_action_id_to_action_pred_indices[i])
                indices = indices.expand(batch_size, -1)
                indices = indices.to(self.device)
                gathered = torch.gather(raw_action_logits, 1, indices)
                # merged value for i-th action for samples in the batch
                merged = softor(gathered, dim=1)  # (batch_size, 1)
                dist_values.append(merged)

        action_values = torch.stack(dist_values, dim=1)  # (batch_size, n_actions)
        action_dist = torch.softmax(action_values, dim=1)

        action_dist = self.reshape_action_distribution(action_dist)
        return action_dist

    def to_neural_action_distribution(self, neural_state):
        """
        Obtain action distribution from neural policy.

        Args:
            neural_state: neural state
        Returns:
            action_dist: action distribution
        """
        hidden = self.neural_actor.network(neural_state)
        logits = self.neural_actor.actor(hidden)
        probs = Categorical(logits=logits)
        action_dist = probs.probs
        return action_dist

    def reshape_action_distribution(self, action_dist):
        """
        Reshape action distribution to match the number of actions in the environment.

        Args:
            action_dist: action distribution
        Returns:
            action_dist: reshaped action distribution
        """
        batch_size = action_dist.size(0)
        if action_dist.size(1) < self.env.n_raw_actions:
            zeros = torch.zeros(
                batch_size,
                self.env.n_raw_actions - action_dist.size(1),
                device=self.device,
                requires_grad=True,
            )
            action_dist = torch.cat([action_dist, zeros], dim=1)
        return action_dist

    def forward(self, neural_state, logic_state):
        """
        Forward pass of the actor.

        Args:
            neural_state: neural state
            logic_state: logic state
        Returns:
            action_probs: action probabilities
            weights: blending weights
        """
        assert self.actor_mode in [
            "hybrid",
            "logic",
            "neural",
        ], "Invalid actor mode {}".format(self.actor_mode)
        if self.actor_mode == "hybrid":
            return self.compute_action_probs_hybrid(neural_state, logic_state)
        elif self.actor_mode == "logic":
            return self.compute_action_probs_logic(logic_state)
        else:
            return self.compute_action_probs_neural(neural_state)


class BlenderActorCritic(nn.Module):
    """
    BlendeRL actor-critic that combines neural and logic policies.

    Args:
        env: environment
        rules: rules
        actor_mode: actor mode, one of ["hybrid", "logic", "neural"]
        blender_mode: blender mode, one of ["logic", "neural"]
        blend_function: blending function, one of ["softmax", "gumbel_softmax"]
        device: device
        rng: random number generator
    """

    def __init__(
        self,
        env,
        rules,
        actor_mode,
        blender_mode,
        blend_function,
        reasoner,
        device,
        rng=None,
        explain=False,
    ):
        super(BlenderActorCritic, self).__init__()
        self.device = device
        self.rng = random.Random() if rng is None else rng
        self.actor_mode = actor_mode
        self.blender_mode = blender_mode
        self.blend_function = blend_function
        self.env = env
        self.rules = rules
        self.explain = explain
        mlp_module_path = f"in/envs/{self.env.name}/mlp.py"
        module = load_module(mlp_module_path)
        self.visual_neural_actor = load_cleanrl_agent(pretrained=False, device=device)
        if reasoner == "neumann":
            from neumann.common import get_neumann_model
            self.logic_actor = get_neumann_model(
                env.name, rules, device=device, train=True, explain=explain
            )
            self.blender = get_blender(
                env,
                rules,
                device,
                blender_mode=blender_mode,
                train=True,
                explain=explain,
            )
        elif reasoner == "nsfr":
            self.logic_actor = get_nsfr_model(
                env.name, rules, device=device, train=True, explain=explain
            )
            self.blender = get_blender(
                env,
                rules,
                device,
                blender_mode=blender_mode,
                train=True,
                explain=explain,
            )
        # self.logic_actor = get_nsfr_model(env.name, rules, device=device, train=True)
        self.logic_critic = module.MLP(device=device, out_size=1, logic=True)
        self.actor = BlenderActor(
            env,
            self.visual_neural_actor,
            self.logic_actor,
            self.blender,
            actor_mode,
            blender_mode,
            blend_function,
            device=device,
        )

        # the number of actual actions on the environment
        self.num_actions = len(self.env.pred2action.keys())

        self.uniform = Categorical(
            torch.tensor(
                [1.0 / self.num_actions for _ in range(self.num_actions)], device=device
            )
        )
        self.upprior = Categorical(
            torch.tensor(
                [0.9]
                + [0.1 / (self.num_actions - 1) for _ in range(self.num_actions - 1)],
                device=device,
            )
        )

    def _print(self):
        """
        Print the weighted logic rules for actor and blender.
        """
        if self.blender_mode == "logic":
            print("==== Blender ====")
            print_program(self.blender)
        print("==== Logic Policy ====")
        print_program(self.logic_actor)

    def get_policy_weights(self):
        """
        Get the blending policy weights stored in the latest forward computation.

        Returns:
            weights: blending policy weights
        """
        return self.actor.w_policy

    def forward(self):
        raise NotImplementedError

    def get_explanation(self, neural_state, logic_state):
        """
        Get the explanation of the blending weights.

        Args:
            neural_state: neural state
            logic_state: logic state
        Returns:
            explanation: explanation
        """
        self.actor.get_explanation(neural_state, logic_state)

    def act(self, neural_state, logic_state, epsilon=0.0):
        """
        Compute an action using the actor. Used only by the play script (render.py).

        Args:
            neural_state: neural state
            logic_state: logic state
            epsilon: epsilon for e-greedy
        Returns:
            action: action
            action_logprob: action log probability
        """
        action_probs, blending_weights = self.actor(neural_state, logic_state)

        # e-greedy
        if self.rng.random() < epsilon:
            # random action with epsilon probability
            dist = self.uniform
            action = dist.sample()
        else:
            dist = Categorical(action_probs)
            # action = (action_probs[0] == max(action_probs[0])).nonzero(as_tuple=True)[0].squeeze(0).to(self.device)
            action = dist.sample()
            # print(action)
            if torch.numel(action) > 1:
                action = action[0]
        # action = dist.sample()
        action_logprob = dist.log_prob(action)
        action_prob = torch.exp(action_logprob)
        return action.detach(), action_prob  # action_logprob.detach()

    def get_prednames(self):
        """
        Get the predicate names representing actions.
        Returns:
            prednames: predicate names
        """
        return self.actor.get_prednames()

    def get_action_and_value(self, neural_state, logic_state, action=None):
        """
        Compute an action and value.
        Args:
            neural_state: neural state
            logic_state: logic state
            action: action
        Returns:
            action: action
            logprob: log probability
            entropy: entropy
            value: value
        """
        # Compute action probabilities using blenderl actor
        # size: n_envs * n_actions
        action_probs, blending_weights = self.actor(neural_state, logic_state)
        dist = Categorical(action_probs)
        blend_dist = Categorical(blending_weights)
        if action is None:
            action = dist.sample()
        logprob = dist.log_prob(action)

        # Compute state values using each neural and logic value function
        # size: n_envs * 1
        neural_value = self.get_neural_value(neural_state)
        logic_value = self.get_logic_value(logic_state)
        # blend the two values using blending weights and compute the final value
        blended_value = (
            blending_weights[:, 0] * neural_value.squeeze(1)
            + blending_weights[:, 1] * logic_value.squeeze(1)
        ).unsqueeze(1)

        return action, logprob, dist.entropy(), blend_dist.entropy(), blended_value

    def get_neural_value(self, neural_state):
        """
        Compute the value using the neural value function from a RGB state.
        Args:
            neural_state: neural state
        Returns:
            value: value
        """
        value = self.visual_neural_actor.get_value(neural_state)
        return value

    def get_logic_value(self, logic_state):
        """
        Compute the value using the logic value function from a OCAtari state.
        Args:
            logic_state: logic state
        Returns:
            value: value
        """
        value = self.logic_critic(logic_state)
        return value

    def get_value(self, neural_state, logic_state):
        """
        Compute the value using the blending value function.
        Args:
            neural_state: neural state
            logic_state: logic state
        Returns:
            value: value
        """
        _, _, _, _, value = self.get_action_and_value(neural_state, logic_state)
        return value

    def save(
        self, checkpoint_path, directory: Path, step_list, reward_list, weight_list
    ):
        """
        Save the model.

        Args:
            checkpoint_path: checkpoint path
            directory: directory
            step_list: step list
            reward_list: reward list
            weight_list: weight list
        """
        torch.save(self.state_dict(), checkpoint_path)
        with open(directory / "data.pkl", "wb") as f:
            pickle.dump(step_list, f)
            pickle.dump(reward_list, f)
            pickle.dump(weight_list, f)
