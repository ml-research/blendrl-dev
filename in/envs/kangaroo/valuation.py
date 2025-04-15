import torch as th

""" in ocatari/ram/kangaroo.py :
        MAX_ESSENTIAL_OBJECTS = {
            'Player': 1, (0)
            'Child': 1, (1)
            'Fruit': 3, (2)
            'Bell': 1, (5)
            'Platform': 20,
            'Ladder': 6,
            'Monkey': 4,
            'FallingCoconut': 1,
            'ThrownCoconut': 3,
            'Life': 8,
            'Time': 1,}       
"""


class PredicateMLP(th.nn.Module):
    def __init__(self, input_size, hidden_size=64, output_size=1):
        super().__init__()
        self.model = th.nn.Sequential(
            th.nn.Linear(input_size, hidden_size),
            th.nn.ReLU(),
            th.nn.Linear(hidden_size, hidden_size // 2),
            th.nn.ReLU(),
            th.nn.Linear(hidden_size // 2, output_size),
            th.nn.Sigmoid()
        )

    def forward(self, x):
        y = self.model(x.float()).squeeze(-1)
        return y
        # return self.model(x.float())


class PredicateModel(th.nn.Module):
    def __init__(self, device):
        super().__init__()
        self.on_ladder_mlp = PredicateMLP(input_size=8).to(device)
        self.right_of_ladder_mlp = PredicateMLP(input_size=8).to(device)
        self.left_of_ladder_mlp = PredicateMLP(input_size=8).to(device)
        self.false_predicate_mlp = PredicateMLP(input_size=1).to(device)
        self.on_pl_ladder_mlp = PredicateMLP(input_size=8).to(device)
        self.on_pl_player_mlp = PredicateMLP(input_size=8).to(device)
        self.close_by_fruit_mlp = PredicateMLP(input_size=8).to(device)
        self.close_by_bell_mlp = PredicateMLP(input_size=8).to(device)
        self.close_by_monkey_mlp = PredicateMLP(input_size=8).to(device)
        self.close_by_throwncoconut_mlp = PredicateMLP(input_size=8).to(device)
        self.close_by_fallingcoconut_mlp = PredicateMLP(input_size=8).to(device)
        self.nothing_around_mlp = PredicateMLP(input_size=49 * 4).to(device)
        self.same_level_ladder_mlp = PredicateMLP(input_size=8).to(device)

    def forward(self, predicate_name, *inputs):
        mlp = getattr(self, f"{predicate_name}_mlp")
        input_tensor = th.cat(inputs, dim=-1)
        return mlp(input_tensor)


# Instantiate the unified model
predicate_model = PredicateModel(device=th.device("cuda" if th.cuda.is_available() else "cpu"))


# Replace functions with MLP-based implementations
def nothing_around(objs: th.Tensor) -> th.Tensor:
    n_envs = objs.shape[0]
    return predicate_model("nothing_around", objs.view(n_envs, -1))


def on_pl_player(player: th.Tensor, obj: th.Tensor) -> th.Tensor:
    return predicate_model("on_pl_player", player, obj)


def on_pl_ladder(ladder: th.Tensor, obj: th.Tensor) -> th.Tensor:
    return predicate_model("on_pl_ladder", ladder, obj)


def close_by_fruit(player: th.Tensor, obj: th.Tensor) -> th.Tensor:
    return predicate_model("close_by_fruit", player, obj)


def close_by_bell(player: th.Tensor, obj: th.Tensor) -> th.Tensor:
    return predicate_model("close_by_bell", player, obj)


def close_by_monkey(player: th.Tensor, obj: th.Tensor) -> th.Tensor:
    return predicate_model("close_by_monkey", player, obj)


def close_by_throwncoconut(player: th.Tensor, throwncoconut: th.Tensor) -> th.Tensor:
    return predicate_model("close_by_throwncoconut", player, throwncoconut)


def close_by_fallingcoconut(player: th.Tensor, fallingcoconut: th.Tensor) -> th.Tensor:
    return predicate_model("close_by_fallingcoconut", player, fallingcoconut)


def on_ladder(player: th.Tensor, obj: th.Tensor) -> th.Tensor:
    return predicate_model("on_ladder", player, obj)


def left_of_ladder(player: th.Tensor, obj: th.Tensor) -> th.Tensor:
    return predicate_model("left_of_ladder", player, obj)


def right_of_ladder(player: th.Tensor, obj: th.Tensor) -> th.Tensor:
    return predicate_model("right_of_ladder", player, obj)


def same_level_ladder(player: th.Tensor, obj: th.Tensor) -> th.Tensor:
    return predicate_model("same_level_ladder", player, obj)
