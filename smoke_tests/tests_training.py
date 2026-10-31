import os,sys
import numpy as np
from pathlib import Path
from collections import Counter
import torch
sys.path.insert(0, str(Path(__file__).absolute().parent / ".."))

def test_training_loop():
    from marl.src.main import MARLEnv,set_seed, Agent, PatchEncoder
    ACTIONS = [(-1,0), (1,0), (0,1),(0,-1)]
    HISTORY_LEN = 2
    MAX_STEPS = 512
    MAX_TEST_STEPS = 4 * MAX_STEPS
    BATCH_SIZE = 64
    use_global_view = True
    device = torch.device("cuda:0") 
    rng = set_seed(42)
    env = MARLEnv(dataset_root="D:\\dataset\\WFLW",split="train",rf_size=64, face_size=512,actions=ACTIONS,
                  target_range_steps=(10,5), use_abs_dist_as_reward=False, 
                  history_len=HISTORY_LEN,
                  max_steps = MAX_STEPS,
                  dataset_size_max=64,
                  rng = rng,
                  use_global_view=use_global_view)
    input_channels = 3 *HISTORY_LEN
    if use_global_view:
        input_channels += 3 
    shared_patch_encoder = PatchEncoder((1, input_channels, env.rf_size, env.rf_size))
    optim_shared = torch.optim.Adam(
        shared_patch_encoder.parameters(), lr = 1e-3
    )
    agents_all : list[Agent] = []
    for i in range(env.num_agents):
        agent = Agent(
            agent_id = i, input_shape=(1,input_channels,env.rf_size, env.rf_size),
            n_actions=len(ACTIONS), replay_buffer_size=max(10000, MAX_TEST_STEPS),device=device,
            use_global_view=use_global_view, rng=rng, epsilon_decay=0.9, shared_patch_encoder=shared_patch_encoder,
            flag_train=True, agent_action_selection_mode="bfs"
        )
        agents_all.append(agent)
   
    states, states_xy = env.reset()
    for step in range(MAX_TEST_STEPS):
        actions = rng.choice(np.arange(len(ACTIONS)),size=env.num_agents).flatten()
        next_states, rewards, dones, truncated, info =  env.step(actions)
        next_states_xy = info['states_xy']
        rf_size = info['rf_size']
        image_id = info['image_id']
        image = info['image']
       
        for iagent in range(env.num_agents):
            agents_all[iagent].replay_buffer.push(
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
                print(f"agent {i}, buffer {len(agents_all[i].replay_buffer.buffer)}, images: {len(agents_all[i].replay_buffer.image_pool.keys())}")
  
    loss_func = torch.nn.HuberLoss(reduction="none")
    updating_params = [
        (agent.policy_net, agent.target_net, agent.optimizer, agent.replay_buffer) for agent in agents_all
    ]
    for iloop in range(10):
        loss_agents = []
        optims_enabled = [optim_shared]
        for iagent, (policy_net, target_net, optim, replay_buffer) in enumerate(updating_params):
            states, actions, rewards, next_states, dones, batch_info = replay_buffer.sample(64)
            if states is None:
                loss_agents.append(torch.zeros((1,),device=device))
                continue
            states_gpu, actions_gpu, rewards_gpu, next_states_gpu, dones_gpu, weights_gpu = states.to(device), actions.to(device), rewards.to(device), next_states.to(device), dones.to(device), torch.from_numpy(batch_info['weights']).to(device)
            currentQ = policy_net(states_gpu).gather(1, actions_gpu)
            with torch.no_grad():
                next_action_pred = policy_net(next_states_gpu).max(-1)[1].unsqueeze(-1)
                nextV = target_net(next_states_gpu).gather(1, next_action_pred)
                targetQ = rewards_gpu + 0.99 * nextV * (1-dones_gpu) # bellman equation
            tderror_agent = torch.abs(currentQ.detach() - targetQ.detach()).cpu()
            replay_buffer.update_priorities(
                batch_info['indices'], tderror_agent.numpy().flatten()
            )
            loss_agent = loss_func(currentQ, targetQ) * weights_gpu.reshape(-1,1)
            optims_enabled.append(optim)
            loss_agents.append(loss_agent.sum()[None])
        if optims_enabled != []:
            print(f"loop#{iloop:0>3d}|agent#{iagent:0>2d}|{len(optims_enabled):0>2d} updated")
            for opt in optims_enabled:
                opt.zero_grad()
            torch.mean(torch.concat(loss_agents)).backward() 
            for opt in optims_enabled:
                opt.step()

if __name__ == "__main__":
    test_training_loop()
 