import torch


class MLP(torch.nn.Module):
    def __init__(self, device, has_softmax=False, has_sigmoid=False, out_size=6, as_dict=False, logic=False, only_player_pos: bool = False):
        super().__init__()
        self.logic = logic
        self.as_dict = as_dict
        self.device = device
        self.only_player_pos = only_player_pos
        encoding_base_features = 2
        encoding_entity_features = 2
        encoding_max_entities = 43
        self.num_in_raw_features = (encoding_base_features + encoding_entity_features) * encoding_max_entities
        self.num_in_features = self.num_in_raw_features if not self.only_player_pos else 2

        modules = [
            torch.nn.Linear(self.num_in_features, 120).to(device),
            torch.nn.ReLU(inplace=True).to(device),
            torch.nn.Linear(120, 60).to(device),
            torch.nn.ReLU(inplace=True).to(device),
            torch.nn.Linear(60, out_size).to(device)
            # torch.nn.Linear(120, out_size).to(device)
        ]
        # modules = [
        #     torch.nn.Linear(self.num_in_features, 40),
        #     torch.nn.ReLU(inplace=True),
        #     torch.nn.Linear(40, out_size)
        # ] 

        if has_softmax:
            modules.append(torch.nn.Softmax(dim=-1))
            
        if has_sigmoid:
            modules.append(torch.nn.Sigmoid())

        self.mlp = torch.nn.Sequential(*modules)
        self.mlp.to(device)

    def forward(self, state):
        features = state.float().view(-1, self.num_in_raw_features)
        if self.only_player_pos:
            features = features[:, 1:3]
        y = self.mlp(features)
        return y
