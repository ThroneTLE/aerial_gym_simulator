import isaacgym  # 先导，避免 torch 冲突
from rl_games.algos_torch.model_builder import ModelBuilder
from aerial_gym.rl_training.rl_games.nn import privileged_actor_critic  # 注册

cfg = {
    'name': 'privileged_actor_critic',
    'model': {'name': 'continuous_a2c_logstd'},
    'network': {
        'name': 'privileged_actor_critic',
        'mlp': {'units':[256,256,128], 'activation':'elu', 'initializer':{'name':'default','scale':2}},
        'privileged_dim': 41,
        'privileged_embed': 8,
        'privileged_hidden': 128,
        'space': {'continuous': {'mu_activation': None, 'sigma_activation': None,
                                 'mu_init': {'name':'default'}, 'sigma_init': {'name':'default'},
                                 'fixed_sigma': False, 'min_logstd': -4.5, 'max_logstd': 1.0}},
    },
    'input_shape': (29,),
    'actions_num': 4,
    'value_size': 1,
    'action_space': 'continuous',
    'num_actors': 1,
}

net = ModelBuilder().load(cfg)
sd = net.model.state_dict()  # 用内部 model
priv_keys = [k for k in sd if 'priv_encoder' in k]
print('has priv_encoder:', len(priv_keys) > 0)
if priv_keys:
    print('sample priv keys:', priv_keys[:3])
first_w = [k for k in sd if k.endswith('actor_mlp.0.weight')][0]
print('actor_mlp.0.weight shape:', sd[first_w].shape)
