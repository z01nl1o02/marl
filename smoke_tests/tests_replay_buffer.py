import os,sys
import numpy as np
from pathlib import Path
from collections import Counter
import torch
sys.path.insert(0, str(Path(__file__).absolute().parent / ".."))

def test_states_identity():
    from marl.src.main import MARLEnv,ReplayBuffer,set_seed
    ACTIONS = [(-1,0), (1,0), (0,1),(0,-1)]
    HISTORY_LEN = 2
    MAX_STEPS = 512
    MAX_TEST_STEPS = 4 * MAX_STEPS
    BATCH_SIZE = 64
    rng = set_seed(42)
    env = MARLEnv(dataset_root="D:\\dataset\\WFLW",split="train",rf_size=64, face_size=512,actions=ACTIONS,
                  target_range_steps=(10,5), use_abs_dist_as_reward=False, 
                  history_len=HISTORY_LEN,
                  max_steps = MAX_STEPS,
                  dataset_size_max=64,
                  rng = rng,
                  use_global_view=True)
    rng = np.random.default_rng(seed=42)
    replay_buffers = [
        ReplayBuffer(capacity=max((MAX_TEST_STEPS,10000)),rng=rng,use_global_view=True) for _ in range(env.num_agents)]
    
    states_gt = []
    states, states_xy = env.reset()
    for step in range(MAX_TEST_STEPS):
        actions = rng.choice(np.arange(len(ACTIONS)),size=env.num_agents).flatten()
        next_states, rewards, dones, truncated, info =  env.step(actions)
        next_states_xy = info['states_xy']
        rf_size = info['rf_size']
        image_id = info['image_id']
        image = info['image']
        states_gt.append(
            {
                "states":states.copy(), # (num_agent, history_len*3, rf_size, rf_size)
                "next_states":next_states.copy(),
                "states_xy":states_xy.copy(), 
                "next_states_xy":next_states_xy.copy(),
             })
        
        for iagent in range(env.num_agents):
            replay_buffers[iagent].push(
                states[iagent], states_xy[iagent],
                actions[iagent], rewards[iagent].item(),
                next_states[iagent],next_states_xy[iagent],
                dones[iagent], image_id, image, rf_size 
            )
        states = next_states.copy()
        states_xy = next_states_xy.copy() 
        if np.all(dones) or truncated:
            states, states_xy = env.reset()
            print(f"reset env.step={step:0>5d}")
            print("="*80,"replay buffer")
            for i in range(env.num_agents):
                print(f"agent {i}, buffer {len(replay_buffers[i].buffer)}, images: {len(replay_buffers[i].image_pool.keys())}")
                
    for itry in range(10):
        for iagent in range(env.num_agents):
            states, actions, rewards, next_states, dones, info = replay_buffers[iagent].sample(BATCH_SIZE) 
            indices = info['indices']
            
            states_ = np.concat([states_gt[i]['states'][iagent][None] for i in indices], axis=0)
            next_states_ = np.concat([states_gt[i]['next_states'][iagent][None] for i in indices], axis=0)
            
            states_ = torch.from_numpy(states_)
            next_states_ = torch.from_numpy(next_states_)
            if torch.allclose(states, states_) and torch.allclose(next_states, next_states_):
                print(f"try#{itry} agent#{iagent} OK")
            else:
                print(f"try#{itry} agent#{iagent} ERROR!!!!!")


if __name__ == "__main__":
    test_states_identity()
 