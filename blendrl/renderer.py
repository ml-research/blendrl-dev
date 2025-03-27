from datetime import datetime
from typing import Union

from networkx import minimum_cut_value

import numpy as np
import torch as th
import pygame
import vidmaker

from nudge.agents.logic_agent import NsfrActorCritic
from nudge.agents.neural_agent import ActorCritic
from nudge.utils import load_model, yellow
from nudge.env import NudgeBaseEnv

SCREENSHOTS_BASE_PATH = "out/screenshots/"
PREDICATE_PROBS_COL_WIDTH = 500 * 2
FACT_PROBS_COL_WIDTH = 1000
CELL_BACKGROUND_DEFAULT = np.array([40, 40, 40])
CELL_BACKGROUND_HIGHLIGHT = np.array([40, 150, 255])
CELL_BACKGROUND_HIGHLIGHT_POLICY = np.array([234, 145, 152])
CELL_BACKGROUND_SELECTED = np.array([80, 80, 80])


class Renderer:
    model: Union[NsfrActorCritic, ActorCritic]
    window: pygame.Surface
    clock: pygame.time.Clock

    def __init__(
        self,
        agent_path: str = None,
        env_name: str = "seaquest",
        device: str = "cpu",
        fps: int = None,
        deterministic=True,
        env_kwargs: dict = None,
        render_predicate_probs=True,
        seed=0,
    ):

        self.fps = fps
        self.deterministic = deterministic
        self.render_predicate_probs = render_predicate_probs

        # Load model and environment
        self.model = load_model(
            agent_path, env_kwargs_override=env_kwargs, device=device
        )
        self.env = NudgeBaseEnv.from_name(
            env_name, mode="deictic", seed=seed, **env_kwargs
        )
        # self.env = self.model.env
        self.env.reset()

        print(self.model._print())

        print(
            f"Playing '{self.model.env.name}' with {'' if deterministic else 'non-'}deterministic policy."
        )

        if fps is None:
            fps = 15
        self.fps = fps

        try:
            self.action_meanings = self.env.env.get_action_meanings()
            self.keys2actions = self.env.env.unwrapped.get_keys_to_action()
        except Exception:
            print(
                yellow(
                    "Info: No key-to-action mapping found for this env. No manual user control possible."
                )
            )
            self.action_meanings = None
            self.keys2actions = {}
        self.current_keys_down = set()

        self.predicates = self.model.logic_actor.prednames

        self._init_pygame()

        self.running = True
        self.paused = False
        self.fast_forward = False
        self.reset = False
        self.takeover = False

    def _init_pygame(self):
        pygame.init()
        pygame.display.set_caption("Environment")
        frame = self.env.env.render()
        self.env_render_shape = frame.shape[:2]
        window_shape = list(self.env_render_shape)
        if self.render_predicate_probs:
            window_shape[0] += PREDICATE_PROBS_COL_WIDTH
        self.window = pygame.display.set_mode(window_shape, pygame.SCALED)
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Calibri", 24)

    def run(self):
        length = 0
        ret = 0

        obs, obs_nn = self.env.reset()
        obs_nn = th.tensor(obs_nn, device=self.model.device)
        # print(obs_nn.shape)

        while self.running:
            self.reset = False
            self._handle_user_input()
            if not self.paused:
                if not self.running:
                    break  # outer game loop

                if self.takeover:  # human plays game manually
                    # assert False, "Unimplemented."
                    action = self._get_action()
                else:  # AI plays the game
                    # print("obs_nn: ", obs_nn.shape)
                    action, logprob = self.model.act(
                        obs_nn, obs
                    )  # update the model's internals
                    value = self.model.get_value(obs_nn, obs)

                (new_obs, new_obs_nn), reward, done, terminations, infos = (
                    self.env.step(action, is_mapped=self.takeover)
                )
                new_obs_nn = th.tensor(new_obs_nn, device=self.model.device)

                self._render()

                if self.takeover and float(reward) != 0:
                    print(f"Reward {reward:.2f}")

                if self.reset:
                    done = True
                    new_obs = self.env.reset()
                    self._render()

                obs = new_obs
                obs_nn = new_obs_nn
                length += 1

                if done:
                    print(f"Return: {ret} - Length {length}")
                    ret = 0
                    length = 0
                    self.env.reset()

        pygame.quit()

    def _get_action(self):
        if self.keys2actions is None:
            return 0  # NOOP
        pressed_keys = list(self.current_keys_down)
        pressed_keys.sort()
        pressed_keys = tuple(pressed_keys)
        if pressed_keys in self.keys2actions.keys():
            return self.keys2actions[pressed_keys]
        else:
            return 0  # NOOP

    def _handle_user_input(self):
        events = pygame.event.get()
        for event in events:
            if event.type == pygame.QUIT:  # window close button clicked
                self.running = False

            elif event.type == pygame.KEYDOWN:  # keyboard key pressed
                if event.key == pygame.K_p:  # 'P': pause/resume
                    self.paused = not self.paused

                elif event.key == pygame.K_r:  # 'R': reset
                    self.reset = True

                elif event.key == pygame.K_f:  # 'F': fast forward
                    self.fast_forward = not (self.fast_forward)

                elif event.key == pygame.K_t:  # 'T': trigger takeover
                    if self.takeover:
                        print("AI takeover")
                    else:
                        print("Human takeover")
                    self.takeover = not self.takeover

                elif event.key == pygame.K_o:  # 'O': toggle overlay
                    self.env.env.render_oc_overlay = not (
                        self.env.env.render_oc_overlay
                    )

                elif event.key == pygame.K_c:  # 'C': capture screenshot
                    file_name = (
                        f"{datetime.strftime(datetime.now(), '%Y-%m-%d-%H-%M-%S')}.png"
                    )
                    pygame.image.save(self.window, SCREENSHOTS_BASE_PATH + file_name)

                elif (event.key,) in self.keys2actions.keys():  # env action
                    self.current_keys_down.add(event.key)

            elif event.type == pygame.KEYUP:  # keyboard key released
                if (event.key,) in self.keys2actions.keys():
                    self.current_keys_down.remove(event.key)

                # elif event.key == pygame.K_f:  # 'F': fast forward
                #     self.fast_forward = False

    def _render(self):
        self.window.fill((20, 20, 20))  # clear the entire window
        self._render_policy_probs()
        self._render_predicate_probs()
        self._render_neural_probs()
        self._render_env()

        pygame.display.flip()
        pygame.event.pump()
        if not self.fast_forward:
            self.clock.tick(self.fps)

    def _render_env(self):
        frame = self.env.env.render()
        frame_surface = pygame.Surface(self.env_render_shape)
        pygame.pixelcopy.array_to_surface(frame_surface, frame)
        self.window.blit(frame_surface, (0, 0))

    def _render_policy_probs_rows(self):
        anchor = (self.env_render_shape[0] + 10, 25)

        model = self.model
        policy_names = ["neural", "logic"]
        weights = model.get_policy_weights()
        for i, w_i in enumerate(weights):
            w_i = w_i.item()
            name = policy_names[i]
            # Render cell background
            color = (
                w_i * CELL_BACKGROUND_HIGHLIGHT_POLICY
                + (1 - w_i) * CELL_BACKGROUND_DEFAULT
            )
            pygame.draw.rect(
                self.window,
                color,
                [
                    anchor[0] - 2,
                    anchor[1] - 2 + i * 35,
                    PREDICATE_PROBS_COL_WIDTH - 12,
                    28,
                ],
            )
            # print(w_i, name)

            text = self.font.render(str(f"{w_i:.3f} - {name}"), True, "white", None)
            text_rect = text.get_rect()
            text_rect.topleft = (self.env_render_shape[0] + 10, 25 + i * 35)
            self.window.blit(text, text_rect)

    def _render_policy_probs(self):
        anchor = (self.env_render_shape[0] + 10, 25)

        model = self.model
        policy_names = ["neural", "logic"]
        weights = model.get_policy_weights()
        for i, w_i in enumerate(weights):
            w_i = w_i.item()
            name = policy_names[i]
            # Render cell background
            color = (
                w_i * CELL_BACKGROUND_HIGHLIGHT_POLICY
                + (1 - w_i) * CELL_BACKGROUND_DEFAULT
            )
            pygame.draw.rect(
                self.window,
                color,
                [
                    anchor[0] - 2 + i * 500,
                    anchor[1] - 2,
                    (PREDICATE_PROBS_COL_WIDTH / 2 - 12) * w_i,
                    28,
                ],
            )

            text = self.font.render(str(f"{w_i:.3f} - {name}"), True, "white", None)
            text_rect = text.get_rect()
            if i == 0:
                text_rect.topleft = (self.env_render_shape[0] + 10, 25)
            else:
                text_rect.topleft = (self.env_render_shape[0] + 10 + i * 500, 25)
            self.window.blit(text, text_rect)

    def _render_predicate_probs(self):
        anchor = (self.env_render_shape[0] + 10, 25)
        nsfr = self.model.actor.logic_actor
        # nsfr.print_valuations(min_value=0.1)
        pred_vals = {
            pred: nsfr.get_predicate_valuation(pred, initial_valuation=False)
            for pred in nsfr.prednames
        }
        for i, (pred, val) in enumerate(pred_vals.items()):
            i += 2
            # Render cell background
            color = (
                val * CELL_BACKGROUND_HIGHLIGHT + (1 - val) * CELL_BACKGROUND_DEFAULT
            )
            pygame.draw.rect(
                self.window,
                color,
                [
                    anchor[0] - 2 + PREDICATE_PROBS_COL_WIDTH / 2,
                    anchor[1] - 2 + i * 35,
                    (PREDICATE_PROBS_COL_WIDTH / 2 - 12) * val,
                    28,
                ],
            )

            text = self.font.render(str(f"{val:.3f} - {pred}"), True, "white", None)
            text_rect = text.get_rect()
            text_rect.topleft = (
                self.env_render_shape[0] + 10 + PREDICATE_PROBS_COL_WIDTH / 2,
                25 + i * 35,
            )
            self.window.blit(text, text_rect)

    def _render_neural_probs(self):
        anchor = (self.env_render_shape[0] + 10, 25)
        blender_actor = self.model.actor
        action_vals = blender_actor.neural_action_probs[0].detach().cpu().numpy()
        action_names = [
            "noop",
            "fire",
            "up",
            "right",
            "left",
            "down",
            "upright",
            "upleft",
            "downright",
            "downleft",
            "upfire",
            "rightfire",
            "leftfire",
            "downfire",
            "uprightfire",
            "upleftfire",
            "downrightfire",
            "downleftfire",
        ]
        for i, (pred, val) in enumerate(zip(action_names, action_vals)):
            i += 2
            # Render cell background
            color = (
                val * CELL_BACKGROUND_HIGHLIGHT + (1 - val) * CELL_BACKGROUND_DEFAULT
            )
            pygame.draw.rect(
                self.window,
                color,
                [
                    anchor[0] - 2,
                    anchor[1] - 2 + i * 35,
                    (PREDICATE_PROBS_COL_WIDTH / 2 - 12) * val,
                    28,
                ],
            )

            text = self.font.render(str(f"{val:.3f} - {pred}"), True, "white", None)
            text_rect = text.get_rect()
            text_rect.topleft = (self.env_render_shape[0] + 10, 25 + i * 35)
            self.window.blit(text, text_rect)

    def _render_facts(self, th=0.1):
        anchor = (self.env_render_shape[0] + 10, 25)

        # nsfr = self.nsfr_reasoner
        nsfr = self.model.actor.logic_actor

        fact_vals = {}
        v_T = nsfr.V_T[0]
        preds_to_skip = [
            ".",
            "true_predicate",
            "test_predicate_global",
            "test_predicate_object",
        ]
        for i, atom in enumerate(nsfr.atoms):
            if v_T[i] > th:
                if atom.pred.name not in preds_to_skip:
                    fact_vals[atom] = v_T[i].item()

        for i, (fact, val) in enumerate(fact_vals.items()):
            i += 2
            # Render cell background
            color = (
                val * CELL_BACKGROUND_HIGHLIGHT + (1 - val) * CELL_BACKGROUND_DEFAULT
            )
            pygame.draw.rect(
                self.window,
                color,
                [anchor[0] - 2, anchor[1] - 2 + i * 35, FACT_PROBS_COL_WIDTH - 12, 28],
            )

            text = self.font.render(str(f"{val:.3f} - {fact}"), True, "white", None)
            text_rect = text.get_rect()
            text_rect.topleft = (self.env_render_shape[0] + 10, 25 + i * 35)
            self.window.blit(text, text_rect)
