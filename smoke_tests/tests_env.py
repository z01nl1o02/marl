import os,sys
import numpy as np
from pathlib import Path
from collections import Counter
sys.path.insert(0, str(Path(__file__).absolute().parent / ".."))


def test_targets():
    import cv2
    from marl.src.main import MARLEnv
    actions = [(-1,0), (1,0), (0,1),(0,-1)]
    env = MARLEnv(dataset_root="D:\\dataset\\WFLW",split="train",rf_size=64, face_size=512,actions=actions,
                  target_range_steps=(10,5), use_abs_dist_as_reward=False, history_len=5, 
                  dataset_size_max=64)
    env.reset()
    vis = env.get_visual_data(size=640)
    cv2.imshow("test_targets",vis)
    cv2.waitKey(5000) 
    
    
def test_reward_range():
    from marl.src.main import MARLEnv, _calc_percentile, _format_container
    face_size = 512
    actions = [(-2,0), (2,0), (0,2),(0,-2)]
    rng = np.random.default_rng(42)
    env = MARLEnv(dataset_root="D:\\dataset\\WFLW",split="train",rf_size=64, face_size=face_size,actions=actions,
                  target_range_steps=(10,5), use_abs_dist_as_reward=False, history_len=5, 
                  dataset_size_max=64,rng=rng)
    env.reset()
    
    rewards_all = []
    for iaction in range(len(actions)):
        rewards = env.step([iaction] * env.num_agents)[1]
        assert rewards.shape == (env.num_agents,)
        assert len(rewards.flatten().tolist()) == env.num_agents
        rewards_all.extend(rewards.flatten().tolist())
    reward_stats = _calc_percentile(rewards_all)
    print(f"reward distribution: {_format_container(reward_stats)}") 
    

def test_exploit_explore():
    from marl.src.main import Agent
    actions = [(-1,0), (1,0), (0,1),(0,-1)]
    input_shape = (3,3,512,512)
    W, H = input_shape[-2:][::-1]
    
    for mode in ["dfs","bfs","random"]:
        for max_step in [600, 6000, 60000]:  
            print( "="*80  + f"mode = {mode} \t max_step = {max_step}")
            agent = Agent(agent_id=0, 
                        input_shape=input_shape,
                        n_actions=len(actions),
                        agent_action_selection_mode=mode, 
                        flag_train=True, 
                        epsilon=2.0)    
            xy_agent = np.array([W*0.5, H*0.5],dtype=np.int32).reshape((2,))
            
            visited = np.zeros((H,W), dtype=np.float32)
            visited[xy_agent[1], xy_agent[0]] += 1
            actions_selected = []
            dist2center = []
            for _ in range(max_step):
                i_action = agent.select_action(None, xy_agent, actions)
                actions_selected.append(i_action)
                xy_agent[0] += actions[i_action][0]
                xy_agent[1] += actions[i_action][1]
                
                if np.all(xy_agent >= (0,0)) and np.all(xy_agent < (W,H)):
                    visited[xy_agent[1], xy_agent[0]] += 1
                    
                    dist = np.linalg.norm((xy_agent[0] - W*0.5, xy_agent[1] - H*0.5),ord=2)
                    dist2center.append(dist)
                
            visited = visited.flatten()
            print(f"action selected: \t {dict(Counter(actions_selected))}")
            print(
                f"visited map:\t min = {np.min(visited)} "
                f"\t P5 = {np.percentile(visited,5)} "
                f"\t P10 = {np.percentile(visited,10)} "
                f"\t P20 = {np.percentile(visited,20)} "
                f"\t P50 = {np.percentile(visited,50)} "
                f"\t P80 = {np.percentile(visited,80)} "
                f"\t P90 = {np.percentile(visited,90)} "
                f"\t max = {np.max(visited)} "
                )
            print(
                f"dist2center:\t min = {np.min(dist2center):5.1f} "
                f"\t P5 = {np.percentile(dist2center,5):5.1f} "
                f"\t P10 = {np.percentile(dist2center,10):5.1f} "
                f"\t P20 = {np.percentile(dist2center,20):5.1f} "
                f"\t P50 = {np.percentile(dist2center,50):5.1f} "
                f"\t P80 = {np.percentile(dist2center,80):5.1f} "
                f"\t P90 = {np.percentile(dist2center,90):5.1f} "
                f"\t max = {np.max(dist2center):5.1f} "
                )
if __name__ == "__main__":
    #test_exploit_explore() 
    #test_targets()
    test_reward_range()
    