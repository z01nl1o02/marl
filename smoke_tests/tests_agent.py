import os,sys
import numpy as np
from pathlib import Path
from collections import Counter
import torch
sys.path.insert(0, str(Path(__file__).absolute().parent / ".."))

def tests_interface():
    from marl.src.main import Agent
    actions_all = [[0,0],[0,1],[1,0],[0,-1],[-1,0]]
    H, W = 512, 512
    rng = np.random.default_rng(42)
    agent = Agent(agent_id=0, input_shape=(1,3,H,W), 
                  n_actions = len(actions_all),
                  agent_action_selection_mode="bfs", 
                  replay_buffer_size=1000, rng=rng,
                  use_global_view=True, flag_train=True)
    for eps in [0.5, 0.05]:
        agent.epsilon = eps
        for n in range(20):
            state = rng.uniform(low=-1, high=1,size=(3,H,W)) 
            state = torch.from_numpy(state).to(torch.float32)
            pos = rng.choice([i for i in range(W * H)])
            xy = (pos%W, pos//W)
            iaction,info = agent.select_action(state, xy, actions_all) 
            print(
                f"{n:0>3d} "
                f"epsilon {agent.epsilon:3.2f} {xy}->"
                f"{iaction}, {info}"
                )
        
tests_interface() 
    
