"""
测试配置 c1.toml

- [x] 支持广度优先和深度优先agent_action_selection_mode
- [x] ACTION支持[0,0], 允许 agent 保持原地静止
- [x] freezing_success_agent 允许已经到达目标位置的 agent 停止更新replay和位置
- [ ] 把参数分层
- [x] 支持"课程", 先放宽"成功"的定义,当成功率足够高以后再逐渐提高"成功"质量
- [ ] 退火 epsilon 替代动作选择
- [x] 分析 select_action 是否能覆盖解空间   
- [x] 支持seed, 确保结果可复现
- [x] Double DQN
- [x] 统计 target_net 预测的取值范围,观察double DQN的效果
- [ ] target network软跟新: Polyak
- [x] 增加 td-err-mean, td-err-std 监控 target net 更新频率
- [ ] 基于 td-err 设置 replay buffer 中样本权重
- [ ] 验证 history_len 对训练的影响
- [ ] 验证 epsilon_decay 对训练的影响
- [ ] 把全图缩放后，融入 agent 的输入信号，增加全局信息
- [x] 使用 mean face 优化初始位置
- [x] 共享 embeddding 层,只用一个Optimizer做更新共享层(使用多个 agent 对应 loss 的均值做backward)
- [x] replay buffer中不要直接存储obs, 而是obs的索引，避免重复的帧被多次存储
- [x] 每步施加惩罚，避免太多零激励样本
- [ ] 把连续 20 episode  success 统计值大于80% s作为课程切换的依据
- [x] 对DQN参数初始化
- [ ] 基于 net 输出 prob 统计 熵
- [x] 支持 heatmap 做初始化
- [x] 支持 heatmap 初始化的扰动
- [x] 每个 agent 单独统计成功率
- [x] 支持 huber loss
- [x] 统计 target_q 的取值范围, 缩放到合适的范围
-----------------DEPERATED-----------------------------------------
- [ ] replay buffer中存储image path + 坐标,动态生成obs 
- [ ] replay buffer中obs采用uint8存储. 
- [ ] 按照更新次数对epsilon做退火,不要基于采样轮次做退火
-------------------------------------------
1. online network每次更新抽取~0.25的新增样本即可
2. BFS/DFS/random对于selection action差异不大
"""


import os,sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque,Counter
import cv2
from loguru import logger
import cloudpickle as pickle
from fire import Fire
from tqdm import tqdm
from copy import deepcopy
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from collections import deque
from torch.utils.tensorboard.writer import SummaryWriter
from datetime import datetime
import tomllib
import shutil
import gc
from typing import Iterable
SAVE_REPLAY_BUFFER = False # 保存repaly buffer需要较多内存


def set_seed(seed:int) -> np.random.Generator:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    np.random.seed(seed)
    return np.random.default_rng(seed)


def _format_container(obj, width=6, precision=3):
    if isinstance(obj, float):
        return f"{obj:{width}.{precision}f}"
    elif isinstance(obj, np.ndarray):
        if obj.size > 1:
            return _format_container(obj.flatten().tolist(),width,precision)
        else:
            return f"{obj.item():{width}.{precision}f}"
    elif isinstance(obj, int):
        return f"{obj:{width}d}"
    elif isinstance(obj, dict):
        items = ", ".join(f"{_format_container(k, width,precision)}:{_format_container(v, width,precision)}" for k, v in obj.items())
        return f"{{{items}}}"
    elif isinstance(obj, (list, tuple)):
        items = ", ".join(_format_container(item, width,precision) for item in obj)
        brackets = "[]" if isinstance(obj, list) else "()"
        return f"{brackets[0]}{items}{brackets[1]}"
    else:
        return str(obj)

def _calc_percentile(data:Iterable, P = (25,50,75,90)):
    if isinstance(data, np.ndarray):
        data = data.flatten() 
    results = {"min":np.min(data).item()}
    for p in P:
        results[f"P{p}"] = np.percentile(data, p).item()
    results['max'] = np.max(data).item()
    results['avg'] = np.mean(data).item()
    return results


class ReplayBuffer:
    def __init__(self, capacity=10000, rng:None|np.random.Generator = None):
        self.buffer = deque(maxlen=capacity)
        self.image_pool:dict[int, dict] = {}
        self.rng = rng if rng else np.random.default_rng(42)

    def reset(self):
        self.buffer.clear()
        self.image_pool.clear()
        
    def push(self, 
             state,state_xy, 
             action, reward, 
             next_state,  next_state_xy,
             done, 
             image_id:int, 
             image:np.ndarray, 
             rf_size:int):
        
        if image_id not in self.image_pool:
            self.image_pool[image_id] = {"image":image.copy(), "rf_size":rf_size}
        
        
        #state = torch.FloatTensor(np.array(state)).unsqueeze(0)
        state = state_xy.copy()
        
        action = torch.LongTensor(np.array([action])).unsqueeze(0)
        
        reward = torch.FloatTensor(np.array([reward])).unsqueeze(0)
        
        #next_state = torch.FloatTensor(np.array(next_state)).unsqueeze(0)
        next_state = next_state_xy.copy()
        
        done = torch.FloatTensor(np.array([1 if done else 0])).unsqueeze(0)
        
        self.buffer.append((image_id,state, action, reward, next_state, done))

    def stat(self) -> dict[str,dict]:
        rewards_all, dones_all, actions_all = [], [], []
        for i,s,a,r,ns,d in self.buffer:
            rewards_all.append(r.item())
            dones_all.append(int(d.item()))
            actions_all.append(a.item())
        return {
            "total": len(self.buffer),
            "reward": _calc_percentile(rewards_all),
            "done": dict(Counter(dones_all)),
            "image_pool": len(self.image_pool.keys())
        } 
    
    def _recover_states(self, image_ids:list[int],states_xy:list[np.ndarray]) -> list[np.ndarray]:
        states_all =  []
        for image_id, state_xy in zip(image_ids, states_xy):
            rf_size = self.image_pool[image_id]["rf_size"]
            env = MARLEnvSim(self.image_pool[image_id]['image'], rf_size)
            # HINT:
            # state_xy: (history_len, 2)
            # state: (history_len*3, rf_size, rf_size)
            history_len, _ = state_xy.shape
            states = np.zeros((history_len*3, rf_size, rf_size),dtype=np.float32)
            state_xy = state_xy.reshape(-1,2)
            for ixy in range(state_xy.shape[0]):
                x,y = state_xy[ixy]
                data = env._crop_roi((x, y))
                data = np.transpose(data, (2, 0, 1))
                states[ixy*3:(ixy+1)*3] = data
            states_all.append(states[None]) 
        return states_all
         
        
    def sample(self, batch_size)->tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,torch.Tensor, dict]:
        if batch_size > len(self.buffer):
            return (None,) * 6
        indices = self.rng.choice(len(self.buffer), size=batch_size, replace=True)
        batch = [self.buffer[i] for i in indices]
        
        image_ids, states_xy, actions, rewards, next_states_xy, dones = zip(*batch)
        
        states = self._recover_states(image_ids, states_xy)
        states = np.concat(states,axis=0)
        states = torch.from_numpy(states)
        
        actions = torch.concat(actions,dim=0)
        
        rewards = torch.concat(rewards,dim=0)
        
        next_states = self._recover_states(image_ids, next_states_xy)
        next_states = np.concat(next_states, axis=0)
        next_states = torch.from_numpy(next_states)
        
        dones = torch.concat(dones,dim=0)
        return states, actions, rewards, next_states, dones, {"indices":indices.copy()}

    def __len__(self):
        return len(self.buffer)

class DatasetWFLW:
    # https://wywu.github.io/dataset/
    def __init__(self, dataset_root: str, split: str, dataset_size_max: int | None = None, points_indices: tuple | None = None, 
                 face_size: int | None = None, min_face_size: int = 512) -> None:
        self.face_size = face_size
        image_root = os.path.join(dataset_root, "WFLW_images")
        if dataset_size_max is not None:  # a subset of full datset
            logger.warning(f"use dataset with limited size {dataset_size_max}")
            annotation_file = os.path.join(dataset_root, "WFLW_annotations", "list_98pt_rect_attr_train_test", "list_98pt_rect_attr_train.txt")
        else:
            assert split in {"train", "test"}
            annotation_file = os.path.join(dataset_root, "WFLW_annotations", "list_98pt_rect_attr_train_test", f"list_98pt_rect_attr_{split}.txt")
        if points_indices is None:
            #points_indices = (0, 5, 16, 27, 32, 57, 76, 82, 60, 64, 68, 72)
            points_indices = (76, 82, 57, 96, 97)
        self.points_num = len(points_indices)
        self.dataset = []
        if dataset_size_max is not None:
            cache_file = os.path.join(dataset_root, f"wflw_{split}_cache_with_size_limit_{dataset_size_max}.pkl")
        else:
            cache_file = os.path.join(dataset_root, f"wflw_{split}_cache.pkl")
        if os.path.exists(cache_file):
            print("load dataset cache file: {}".format(cache_file))
            with open(cache_file, 'rb') as f:
                images_info = pickle.load(f)
        else:
            images_info = {}
        self.meanface = np.zeros((len(points_indices),2),dtype=np.float32)
        with open(annotation_file, 'r') as f:
            for items in tqdm(f, desc="load WFLW..."):
                items = items.strip()
                if items == "":
                    continue
                items = items.split(' ')
                XYall = np.array([float(d) for d in items[0:196]], dtype=np.float32).reshape(-1, 2)
                bbox_def = np.array([int(d) for d in items[196:196+4]], dtype=np.int32).reshape(-1)
                pose, expression, illumination, makeup, occlusion, blur = np.array([int(d) for d in items[196+4:196+4+6]])
                if pose != 0 or expression != 0 or illumination != 0 or makeup != 0 or occlusion != 0 or blur != 0:
                    continue
                
                image_path = items[-1].strip()
                image_path = os.path.join(image_root, image_path)
                assert os.path.exists(image_path)
               
                if image_path in images_info:
                    info = images_info[image_path]
                    H, W = info['height'],  info['width']
                else:
                    H, W = cv2.imread(image_path, 0).shape
                    images_info[image_path] = {"width": W,  "height": H}
                     
                # FIXME: some landmark is out of bbox
                def _normalize_face_bbox(face_kpts:np.ndarray, 
                                         image_size:tuple[int,int],
                                         face_size_minmax: tuple[int,int],
                                         bbox_def: tuple[int,int,int,int]
                                         ) -> np.ndarray | None:
                    imgw, imgh = image_size
                    fs_min, fs_max = face_size_minmax
                    px0, px1 = face_kpts[:, 0].min(), face_kpts[:, 0].max()
                    py0, py1 = face_kpts[:, 1].min(), face_kpts[:, 1].max()
                    x0, y0, x1, y1 = bbox_def
                    x0, x1 = min([px0, x0]),  max((px1, x1))
                    y0, y1 = min([py0, y0]),  max((py1, y1))
                    cx, cy = (x1+x0)/2,  (y1+y0)/2
                    facew, faceh = (x1-x0),  (y1-y0)
                    x0, x1 = (cx - facew/2 * 1.5), (cx + facew/2 * 1.5)
                    y0, y1 = (cy - faceh/2 * 1.5), (cy + faceh/2 * 1.5)
                    x0, y0 = max((x0, 0)), max((y0, 0))
                    x1, y1 = min((x1, imgw)), min((y1, imgh))
                    if y1 - y0 < fs_min or x1 - x0 < fs_min:
                        return None
                    if y1 - y0 > fs_max or x1 - x0 > fs_max:
                        return None
                    bbox = np.array([x0, y0, x1, y1], dtype=np.int32).reshape(-1)
                    return bbox
                face_bbox = _normalize_face_bbox(XYall, (W,H), (min_face_size, np.iinfo('int32').max), bbox_def)     
                if face_bbox is None:
                    continue
                XY = [XYall[i].reshape(1, 2) for i in points_indices]
                XY = np.concatenate(XY, axis=0).astype(np.int32)
                self.meanface += (XY - np.array([face_bbox[0], face_bbox[1]]).reshape(1,2)) / np.array([face_bbox[2]-face_bbox[0], face_bbox[3]-face_bbox[1]],dtype=np.float32).reshape(1,2)
                self.dataset.append(
                    {"image": image_path, "face_bbox": face_bbox, "points": XY}
                )
                
                if dataset_size_max is not None and len(self.dataset) >= dataset_size_max:
                    break
        self.meanface /= len(self.dataset)       
        with open(cache_file, 'wb') as f:
            pickle.dump(images_info, f)
        return
    
    def __len__(self):
        return len(self.dataset)
    
    def get_point_num(self):
        return self.points_num
    
    def get(self, index) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        assert index >= 0 and index < len(self.dataset)
        item = self.dataset[index]
        image = cv2.imread(item['image'], 1)
        assert image is not None
        # crop face region
        x0, y0, x1,y1 = item['face_bbox']
        face_image = image[y0:y1, x0:x1]
        face_kpts = item['points'] - np.array([x0, y0]).reshape(-1, 2)  
        if self.face_size is not None: # TODO: make square before resize
            imgh, imgw = face_image.shape[0:2]
            sx,sy = self.face_size / imgw, self.face_size/imgh
            face_image = cv2.resize(face_image, [self.face_size, self.face_size], interpolation=cv2.INTER_LINEAR)
            face_kpts = (face_kpts * np.array([sx,sy]).reshape(1,2)).astype(np.int32)
        h,w = face_image.shape[0:2]
        return face_image, face_kpts, (self.meanface * np.array([w,h]).reshape(1,2)).astype(np.int32)

class MARLEnvSim:
    """
    为了节省replay buffer中存储空间而设计的一个类，允许从坐标值生成obs
    所有代码逻辑必须和 MARLEnv 保持一致
    """
    
    def __init__(self, image_data:np.ndarray, rf_size:int):
        self.current_image = image_data.copy()
        self.rf_size = rf_size
        self.current_image_patched = self._prepare_current_image()
    def _prepare_current_image(self) -> None:
        pad_size = self.rf_size
        current_image_patched = np.pad(self.current_image, ((pad_size, pad_size), (pad_size, pad_size), (0, 0)),
                                            constant_values=128, mode="constant")
        return current_image_patched
    def _crop_roi(self, xy: tuple[int, int]):
        
        assert self.current_image is not None and self.current_image_patched is not None
        H, W = self.current_image.shape[0:2]
        nx, ny = xy
        size = int(self.rf_size)
        #assert not (nx < 0 or nx > W or ny < 0 or ny > H) # REVIEW: this is different from MARLEnv::_crop_roi()
        if  nx < 0 or ny < 0:
            return np.zeros((self.rf_size, self.rf_size, 3),dtype=np.float32)
  
        x0 = int(nx - size/2)
        x1 = x0 + size
        y0 = int(ny - size/2)
        y1 = y0 + size

        offset = self.rf_size 
        roi_data = (self.current_image_patched[y0 + offset:y1 + offset, x0 + offset:x1 + offset] - 128.0)/128.0
        return roi_data


       

class MARLEnv:
    def __init__(self, dataset_root: str, split: str, rf_size: int, face_size: int,
                 actions: list[tuple[float, float]], 
                 target_range_steps: tuple[int],
                 use_abs_dist_as_reward:bool,
                 rng: np.random.Generator,
                 reward_scale: float = 10.0,
                 history_len: int = 1, dataset_size_max: int | None = None, max_steps: int = 1000,
                 agent_xy_init_method: str = "random"):
        self.actions = actions
        
        self.target_range_steps = target_range_steps
        self.current_target_range_step_idx = 0

        self.use_abs_dist_as_reward = use_abs_dist_as_reward
        self.flag_train = True if split.lower() == "train" else False
        
       
        self.rng = rng
        self.history_len = history_len
        self.dataset = DatasetWFLW(dataset_root, split=split, dataset_size_max=dataset_size_max, face_size=face_size)
        self.num_agents = self.dataset.get_point_num()
        self.xy_agents = np.zeros((self.num_agents, 2), dtype=np.int32)
        self.history_obs_agents = [deque(maxlen=history_len) for _ in range(self.num_agents)]
        self.history_xy_agents = [deque(maxlen=history_len) for _ in range(self.num_agents)]
        self.resize_method = cv2.INTER_AREA

        self.rf_size = rf_size

        self.current_image, self.current_image_patched = None, None
        self.xy_targets = None
        
        self.current_file_id = 0

        self.max_steps = max_steps 
        self.num_steps = 0
        self.reward_on_timecost = -1.0 / max_steps

        self.reward_scale = reward_scale #all xy is in [0,1]

        self.xy_init_count = np.zeros((face_size, face_size), dtype=np.float32)
        self.xy_init_count[face_size//2, face_size//2] = 1.0 #avoid zero-division
        self.face_size:int = face_size
        
        self.agent_xy_init_method = agent_xy_init_method
        if agent_xy_init_method.lower() == "heatmap":
            from marl.src.heatmap import load_heatmap_provider
            weights_path = "runs/heatmap_prior/heatmap_net_best.pth"
            self.agent_xy_init_func = load_heatmap_provider(weights_file=weights_path, num_points=self.num_agents)
            logger.warning(f"load heatmap provider: {weights_path}")
            self.init_jitter = 0.2
            self.decay_jitter = 0.9
        else:
            self.agent_xy_init_func = None 

        return
    
    def __len__(self):
        return len(self.dataset)
   
    
    def next_target_range_step(self) -> float|None:
        if self.current_target_range_step_idx + 1 < len(self.target_range_steps):
            self.current_target_range_step_idx += 1
            return self.target_range_steps[self.current_target_range_step_idx]
        return None
    
    def get_point_num(self):
        return self.dataset.get_point_num()

    def _random_init_xy(self):
        logit = np.exp(-self.xy_init_count)
        logit -= logit.min()
        p = (logit / logit.sum()).flatten()
        p = np.nan_to_num(p, nan=1.0 / (self.face_size * self.face_size))
        xy = np.random.choice([k for k in range(self.face_size * self.face_size)], size=self.num_agents, p=p, replace=False)
        x = xy % self.face_size
        y = xy // self.face_size
        self.xy_init_count[y, x] += 1
        if self.xy_init_count.min() > 100:
            self.xy_init_count -= self.xy_init_count.min()
        return np.vstack([x, y]).transpose()

    def reset(self, file_id: int = -1) -> tuple[np.ndarray, np.ndarray]:
        self._switch_image(file_id_required=file_id)
        for i_agent in range(self.num_agents):
            self.history_obs_agents[i_agent].clear()
            self.history_xy_agents[i_agent].clear()
            for _ in range(self.history_len):
                self.history_obs_agents[i_agent].append(np.zeros((1, 3, self.rf_size, self.rf_size), dtype=np.float32))
                self.history_xy_agents[i_agent].append(np.zeros((1,1,2)) + np.iinfo("int32").min)
        return self.get_observation()
    
    def get_visual_data(self, size: int = 64):
        assert self.current_image is not None and self.current_image_patched is not None
        assert self.xy_targets is not None
        H, W = self.current_image.shape[0:2]
        visual_data = deepcopy(self.current_image)
        cmap = plt.get_cmap("jet")
        for i_agent in range(self.num_agents):
            color = [int(c * 255) for c in cmap((i_agent + 1)/self.num_agents)[0:3]]
            nx, ny = self.xy_agents[i_agent, 0], self.xy_agents[i_agent, 1]
            cv2.circle(visual_data, (nx, ny), self.rf_size//2, color, 3)

            tx, ty = self.xy_targets[i_agent, 0], self.xy_targets[i_agent, 1]
            cv2.line(visual_data, (nx, ny), (tx, ty), color, 3)
        return cv2.resize(visual_data, (size, size), interpolation=cv2.INTER_AREA)

    def _prepare_current_image(self) -> None:
        assert self.current_file_id >= 0 and self.current_file_id < len(self.dataset)
        image_data, label_data, label_data_init = self.dataset.get(self.current_file_id)
        self.current_image = image_data
        self.xy_targets = label_data
        pad_size = self.rf_size
        if self.agent_xy_init_method.lower() == "random":
            self.xy_agents = self._random_init_xy()
        elif self.agent_xy_init_method.lower() == "fixed":
            self.xy_agents = label_data_init 
        elif self.agent_xy_init_func is not None:
            H,W, _ = image_data.shape
            xy_init = np.asarray(self.agent_xy_init_func(image_data), dtype=np.float32)
            assert xy_init.shape == (self.num_agents, 2)
            # TODO: stat error and add init xy jitter
            #self.last_init_err = float(np.mean(np.linalg.norm(xy_init - self.xy_targets, axis=-1)))
            if self.flag_train and self.init_jitter > 0:  # jitter curriculum: RL learns to correct residual error
                xy_init = xy_init + self.rng.normal(0.0, self.init_jitter, xy_init.shape).astype(np.float32)
                xy_init = np.clip(xy_init, 0.0, 1.0)
            xy_init = (xy_init*np.array([W,H]).reshape(1,2)).astype(np.int32)
            self.xy_agents = xy_init                     
        else:
            assert 0, f"invalid init-xy-method = {self.agent_xy_init_method}"  
        # current_image_patched 仅仅用来简化obs生成过程
        self.current_image_patched = np.pad(self.current_image, ((pad_size, pad_size), (pad_size, pad_size), (0, 0)),
                                            constant_values=128, mode="constant")
        return
    
    def _switch_image(self, file_id_required=-1) -> None:
        if self.flag_train:
            file_id = file_id_required % len(self.dataset) if file_id_required >= 0 else random.randint(0, len(self.dataset)-1)
        else:
            file_id = file_id_required % len(self.dataset) if file_id_required >= 0 else (self.current_file_id + 1) % len(self.dataset)
        self.current_file_id = file_id
        self._prepare_current_image()
        self.num_steps = 0
        return
   
    def _crop_roi(self, xy: tuple[int, int]):
        assert self.current_image is not None and self.current_image_patched is not None
        H, W = self.current_image.shape[0:2]
        nx, ny = xy
        size = int(self.rf_size)
        assert not (nx < 0 or nx > W or ny < 0 or ny > H) # reference to _take_actions()
  
        x0 = int(nx - size/2)
        x1 = x0 + size
        y0 = int(ny - size/2)
        y1 = y0 + size

        offset = self.rf_size 
        roi_data = (self.current_image_patched[y0 + offset:y1 + offset, x0 + offset:x1 + offset] - 128.0)/128.0
        return roi_data

    def get_observation(self) -> tuple[np.ndarray, np.ndarray]:
        assert self.current_file_id >= 0 and self.current_file_id < len(self.dataset)
        for i_agent, (x, y) in enumerate(self.xy_agents):
            data = self._crop_roi((x, y))
            data = np.transpose(data, (2, 0, 1))[np.newaxis, :, :, :]
            self.history_obs_agents[i_agent].append(data)
            xy = np.array([x,y]).reshape(1,1,2)
            self.history_xy_agents[i_agent].append(xy)
        obs_all = []
        xy_all = []
        for i_agent in range(self.num_agents):
            obs = np.concatenate([d for d in self.history_obs_agents[i_agent]], axis=1)
            obs_all.append(obs)
            xy = np.concatenate([d for d in self.history_xy_agents[i_agent]], axis=1)
            xy_all.append(xy)
        obs_data = np.concatenate(obs_all, axis=0)
        xy_data = np.concatenate(xy_all,axis=0)
        return obs_data.copy(), xy_data

    def _take_actions(self, actions: list[int|None]) -> None:
        assert len(actions) == self.num_agents
        assert self.current_image is not None
        for i_agent, i_action in enumerate(actions):
            if i_action < 0:
                continue # 输入 -1,表示不再移动 agent
            # 必须和 select_action 中计算方法保持一致
            x = self.xy_agents[i_agent, 0] + self.actions[i_action][0]
            y = self.xy_agents[i_agent, 1] + self.actions[i_action][1]
            self.xy_agents[i_agent, :] = (x, y)
        self.xy_agents = np.clip(self.xy_agents, 0, self.face_size-1)
        return

    def step(self, actions: list[int]) -> tuple[np.ndarray, np.ndarray, list[bool], dict]:
        """
        actions: [(dx0,dy0), ... (dxn, dyn)]
        return: (observation, reward, play_end, aux_info)
        """
        self.num_steps += 1
        xy_agents_old = np.copy(self.xy_agents)
        self._take_actions(actions)
        xy_agents_new = self.xy_agents

        obs_new, obs_xy_new = self.get_observation()
        assert self.xy_targets is not None
        dist_new = np.linalg.norm(self.xy_targets - xy_agents_new,  axis=-1, ord=2)
        dist_old = np.linalg.norm(self.xy_targets - xy_agents_old, axis=-1, ord=2)
        if self.use_abs_dist_as_reward:
            reward = 1 - (dist_new/self.face_size) * self.reward_scale / 1.0 # 使用绝对量做reward, 需要仔细设计避免被 agent 发现 "捷径"
        else:
            """
            reward 的取值范围 [-step * reward_scale/face_size, -step * reward_scale/face_size] 
            step = 2, reward_scale = 10, face_size = 512 -> reward is in [-0.04, 0.04]
            """
            reward = (dist_old - dist_new) * self.reward_scale / self.face_size  #reward shaping    
            
        if hasattr(self, "reward_on_timecost"):
            reward += self.reward_on_timecost
            
        target_range = self.target_range_steps[self.current_target_range_step_idx]
        success_mask = [dist_new[i] < target_range for i in range(self.num_agents)]
       
        info = {"success": np.all(success_mask),
                "num_steps": self.num_steps,
                "success_agents": np.mean(dist_new < target_range),
                "success_agent_mask":success_mask,
                "states_xy" : obs_xy_new.copy(),
                "image_id": self.current_file_id,
                "image": self.current_image,
                "rf_size": self.rf_size
                } 
        return obs_new, reward, success_mask, self.num_steps >= self.max_steps, info
    
class PatchEncoder(nn.Module):
    def __init__(self, shape_input: tuple[int, int, int, int]) -> None:
        super().__init__()
        B, C, H, W = shape_input
        self.conv1 = nn.Conv2d(C, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        for conv in [self.conv1, self.conv2, self.conv3]:
            torch.nn.init.xavier_normal_(conv.weight)
            torch.nn.init.zeros_(conv.bias)
         
        with torch.no_grad():
            dummy_input = torch.zeros(shape_input)
            conv_final = self.conv3(self.conv2(self.conv1(dummy_input)))
            self.out_dim = conv_final.view(conv_final.size(0), -1).shape[-1]

    def forward(self, x): 
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        return x 

class CollabDQN(nn.Module):
    def __init__(self, shape_input: tuple[int, int, int, int], n_actions: int, patch_encoder: PatchEncoder | None = None):
        super().__init__()
        self.shared_patch_encoder = False
        if patch_encoder is None:
            self.patch_encoder = PatchEncoder(shape_input)
        else: 
            self.shared_patch_encoder = True
            self.patch_encoder = patch_encoder
        self.fc1 = nn.Linear(self.patch_encoder.out_dim, 512)
        torch.nn.init.normal_(self.fc1.weight, 0, 0.1)
        torch.nn.init.zeros_(self.fc1.bias)
        self.head = nn.Linear(512, n_actions)
        torch.nn.init.normal_(self.head.weight, 0, 0.1)
        torch.nn.init.zeros_(self.head.bias)
    def unshared_parameters(self) -> list[torch.nn.Parameter]:
        params = list(self.fc1.parameters()) + list(self.head.parameters())
        if not self.patch_encoder:
            params += list(self.patch_encoder.parameters())
        return params 
    def forward(self, x):
        x = self.patch_encoder(x)
        x = torch.relu(self.fc1(x))
        return self.head(x)
    


# =======================
# 3. agents
# =======================

class Explorer:
    """
    探索解空间
    """
    def __init__(self, xrange: tuple[int, int], yrange: tuple[int, int]):
        self.xrange = xrange
        self.yrange = yrange
        self.xy2visit: dict[tuple[int, int], float] = {}
    def set_visit(self, xy: tuple[int, int], num: float =1):
        x, y = xy
        if x < self.xrange[0] or y < self.yrange[0] or x > self.xrange[1] or y > self.yrange[1]:
            return
        if xy not in self.xy2visit:
            self.xy2visit[xy] = num
        else:
            self.xy2visit[xy] += num
        return
    def get_visit(self, xy) -> float:
        x, y = xy
        if x < self.xrange[0] or y < self.yrange[0] or x > self.xrange[1] or y > self.yrange[1]:
            return 0.0
        if xy in self.xy2visit:
            return self.xy2visit[xy]
        return 0.0
        
        

class Agent:
    def __init__(self, agent_id, input_shape, n_actions, 
                 agent_action_selection_mode:str,
                 replay_buffer_size:int,
                 rng: np.random.Generator,
                 lr=1e-3, # adam
                 device: str = "cuda:0",
                 epsilon=0.6, epsilon_decay=0.8, epsilon_min=0.05, 
                 shared_patch_encoder: PatchEncoder | None = None, 
                 flag_train: bool = True):
        """
        Args:
            shared_policy_net: 如果共享权重，传入共享的网络对象
        """
        self.flag_train = flag_train
        self.device = torch.device(device)
        self.agent_id = agent_id
        self.n_actions = n_actions
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay # 0.8^(20) = 0.011
        self.epsilon_min = epsilon_min
        self.agent_action_selection_mode = agent_action_selection_mode
        self.rng = rng
        self.replay_buffer = ReplayBuffer(capacity=replay_buffer_size, rng=rng)  # eat up memory!!! but have to be larger than max_step

        self.policy_net = CollabDQN(input_shape, n_actions, patch_encoder=shared_patch_encoder)
        self.policy_net.to(self.device)
        self.optimizer = optim.Adam(self.policy_net.unshared_parameters(), lr=lr)
        # 目标网络通常不共享，或者以较慢速度同步，这里保持独立以稳定训练
        self.target_net = CollabDQN(input_shape, n_actions)
        self.target_net.to(self.device)
        self.update_target_network()

        _, _, H, W = input_shape
        self.explorer = Explorer((0, W-1), (0, H-1))
        self.W, self.H = W, H


    def select_action(self, state, current_xy: np.ndarray, actions:list[tuple[int,int]]):
        if self.flag_train and self.rng.uniform(0,1) < self.epsilon:
            x, y = current_xy
            visits = []
            xys = []
            for i_action in range(self.n_actions):
                nx, ny = int(actions[i_action][0] + x), int(actions[i_action][1] + y)
                visits.append(self.explorer.get_visit((nx, ny)))
                xys.append((nx, ny))
            # TODO: use exp-energy
            total = sum(visits)
            if total < 1.0:
                return self.rng.choice([k for k in range(self.n_actions)],size=1).item()
            if self.agent_action_selection_mode.lower() == "dfs":
                p = [v / total for v in visits] # 深度优先
            elif self.agent_action_selection_mode.lower() == "bfs":
                p = [(total -  v) / total for v in visits] # 广度优先
                t = np.sum(p)
                p = [v / t for v in p] 
            else: # 随机游走
                p = [1/len(visits) for _ in range(len(visits))]
            i_action = self.rng.choice(np.arange(self.n_actions), size=1, p=p, replace=False).item()
            self.explorer.set_visit(xys[i_action])
            return i_action
        else:
            with torch.no_grad():
                # state shape: (4, H, W) -> add batch dim: (1, 4, H, W)
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.policy_net(state_tensor).cpu()
                i_action = q_values.max(1).indices.item()

                x, y = current_xy
                nx, ny = int(actions[i_action][0] + x), int(actions[i_action][1] + y)
                self.explorer.set_visit((nx, ny))
                
                return i_action

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())


# =======================
# Marl 核心协作逻辑
# =======================
class MARLBase:
    def __init__(self, num_agents, input_shape, n_actions, 
                 agent_action_selection_mode:str,
                 replay_buffer_size:int,
                 rng: np.random.Generator,
                 epsilon_decay: float = 0.99,
                 device: str = "cuda:0", 
                 use_weight_sharing=False, 
                 flag_train=True):
        self.rng = rng
        self.episode_count = 0
        self.device = torch.device(device)

        self.num_agents = num_agents
        self.use_weight_sharing = use_weight_sharing
        
            
        # 如果使用权重共享，创建一个共享的策略网络
        if use_weight_sharing:
            self.shared_patch_encoder = PatchEncoder(input_shape)
            self.optimizer = torch.optim.Adam(self.shared_patch_encoder.parameters(),
                                              lr = 1e-3)
        else:
            self.shared_patch_encoder = None
            self.optimizer = None

        # 2. 初始化智能体
        self.agents = []
        for i in range(num_agents):
            agent = Agent(
                agent_id=i,
                input_shape=input_shape,
                n_actions=n_actions,
                replay_buffer_size=replay_buffer_size,
                device=device,
                rng=rng,
                epsilon_decay=epsilon_decay,
                shared_patch_encoder=self.shared_patch_encoder,
                flag_train=flag_train,
                agent_action_selection_mode=agent_action_selection_mode
            )
            self.agents.append(agent)
            

    def save(self, path):
        states = {}
        for agent in self.agents:
            states.update(
                {f"agent{agent.agent_id}_target_net": agent.target_net.state_dict(),
                 f"agent{agent.agent_id}_policy_net": agent.policy_net.state_dict(),
                 }
            )
            if SAVE_REPLAY_BUFFER:
                states.update( 
                 {f"agent{agent.agent_id}_replay": agent.replay_buffer}
                )
                print(f"save agent {agent.agent_id}. replay buffer: {len(agent.replay_buffer)}")
        torch.save(states, path)
        return

    def load(self, path):
        with open(path, 'rb') as f:
            states = torch.load(f, map_location=self.device)
        for agent in self.agents:
            agent.target_net.load_state_dict(states[f"agent{agent.agent_id}_target_net"])
            agent.policy_net.load_state_dict(states[f"agent{agent.agent_id}_policy_net"])
            if SAVE_REPLAY_BUFFER:
                agent.replay_buffer = states[f"agent{agent.agent_id}_replay"]

        if self.shared_patch_encoder is not None:
            self.shared_patch_encoder.load_state_dict(self.agents[0].policy_net.patch_encoder.state_dict())
            for k in range(1, len(self.agents)):
                self.agents[k].policy_net.patch_encoder = self.shared_patch_encoder  # set shared encoder!!!!!
        return

class MARLTrain(MARLBase):
    def __init__(self, num_agents, input_shape, n_actions, 
                 agent_action_selection_mode:str,
                 replay_buffer_size:int,
                 epsilon_decay:float,
                 rng : np.random.Generator,
                 gamma: float = 0.99,
                 device: str = "cuda:0", 
                 use_weight_sharing=False, 
                 freezing_success_agent=False,
                 training_loss = "mse",
                 use_double_dqn = True):
        super().__init__(num_agents=num_agents, 
                         input_shape=input_shape, 
                         n_actions=n_actions,
                         device=device,
                         rng = rng,
                         epsilon_decay=epsilon_decay,
                         replay_buffer_size=replay_buffer_size,
                         use_weight_sharing=use_weight_sharing, 
                         flag_train=True,
                         agent_action_selection_mode=agent_action_selection_mode)
        self.freezing_success_agent = freezing_success_agent
        self.rng = rng
        self.agents_tderr_smoothed = [0.0 for _ in range(self.num_agents)]
        self.tderr_ema_beta = 0.99
        if training_loss.lower() == "huber":
            self.loss_func = torch.nn.SmoothL1Loss(beta=1.0)
        elif training_loss.lower() == "mse":
            self.loss_func = torch.nn.MSELoss()
        self.gamma = gamma
        self.use_double_dqn = use_double_dqn
    def reset_replay_buffer(self):
        for agent in self.agents:
            agent.replay_buffer.reset()
        self.agents_tderr_smoothed = [0.0 for _ in range(self.num_agents)]
            
    def train_step(self, batch_size: int = 64, update_steps: int = 100)->dict:
        """
        执行一次梯度更新。
        """
        agents_tderr_mean = [[] for _ in range(self.num_agents)]
        agents_tderr_std = [[] for _ in range(self.num_agents)]
        nets_to_update = [(agent.policy_net, agent.target_net, agent.optimizer, agent.replay_buffer) for agent in self.agents]
        loss_all = []
        target_q_all = []
        target_net_out_all = []
        for _ in range(update_steps):
            loss_agents = []
            optimizers_to_update = [] 
            if self.optimizer is not None:
                optimizers_to_update += [self.optimizer]
            for iagent, (policy_net, target_net, optimizer, replay_buffer) in enumerate(nets_to_update):
                # 从共享缓冲区采样
                states, actions, rewards, next_states, dones,_ = replay_buffer.sample(batch_size)
                if states is None:
                    loss_agents.append(torch.zeros((1,),device=self.device))
                    continue # replay buffer中样本太少，直接跳过训练
                    

                states_gpu = states.to(self.device)
                actions_gpu = actions.to(self.device)
                rewards_gpu = rewards.to(self.device)
                next_states_gpu = next_states.to(self.device)
                dones_gpu = dones.to(self.device).to(torch.long)


                # 计算当前Q值
                current_action_value = policy_net(states_gpu)
                current_q = current_action_value.gather(1, actions_gpu)
                # 计算目标Q值
                with torch.no_grad():
                    if self.use_double_dqn:
                        next_a = policy_net(next_states_gpu).max(-1)[1].unsqueeze(-1)
                        next_q = target_net(next_states_gpu).gather(1, next_a)
                    else:
                        next_q = target_net(next_states_gpu).max(-1)[0].unsqueeze(1)
                    target_q = rewards_gpu + (self.gamma * next_q * (1 - dones_gpu))
                    target_q_all.extend([qq.item() for qq in target_q.flatten().cpu()])
                    target_net_out_all.extend([qq.item() for qq in next_q.flatten().cpu()])
                    
                tderr_agent = torch.abs(current_q.detach() - target_q).cpu()
                
                agents_tderr_mean[iagent].append(torch.mean(tderr_agent).item())
                agents_tderr_std[iagent].append(torch.std(tderr_agent).item())
                self.agents_tderr_smoothed[iagent] = self.tderr_ema_beta * self.agents_tderr_smoothed[iagent] + (1-self.tderr_ema_beta) * agents_tderr_mean[iagent][-1]
                
                
                loss_agent = self.loss_func(current_q, target_q)
                
                optimizers_to_update.append(optimizer)  
                loss_agents.append(loss_agent[None])
            
            if optimizers_to_update != []:
                for opt in optimizers_to_update:
                    opt.zero_grad()
                torch.mean(torch.concat(loss_agents)).backward() 
                params_all = []
                for agent in self.agents:
                    params_all += list(agent.policy_net.unshared_parameters())
                if self.shared_patch_encoder is not None:
                    params_all += list(self.shared_patch_encoder.parameters())
                # TODO: put max_norm into config file
                torch.nn.utils.clip_grad_norm_(params_all, max_norm=10.0, norm_type=2.0 )
                for opt in optimizers_to_update: 
                    opt.step()
                loss_agents = [l.cpu().item() for l in loss_agents]
                loss_all.append(np.array(loss_agents).reshape(1, -1))
            else:
                loss_all.append(np.zeros(1,self.num_agents))
        loss_all = np.mean(np.concatenate(loss_all, axis=0), axis=0)
        return {"loss": loss_all, 
                "tderr_mean_agents":[np.mean(v_batch).item() if v_batch is not [] else 0 for v_batch in agents_tderr_mean], 
                "tderr_std_agents": [np.mean(v_batch).item() if v_batch is not [] else 0 for v_batch in agents_tderr_std],
                "tderr_mean_smoothed_agents": deepcopy(self.agents_tderr_smoothed),
                "target_q": target_q_all,
                "target_net_out": target_net_out_all
                }

    def run_episode(self, env, max_steps_collected: int = 100, batch_size: int = 64, update_steps=100, target_update_freq: int = 10):
        """
        运行一个完整的Episode，收集所有智能体的经验到共享缓冲区
        """
        # 重置环境，获取初始状态列表 [state_0, state_1, ..., state_k]
        states,states_xy = env.reset()
        actions_all = []
        visual_data_all = [env.get_visual_data()]
        rewards_all = []
        success_agent_mask = [False for _ in range(self.num_agents)]
        success_agents = []
        while len(rewards_all) <= max_steps_collected:
            while True:
                actions = []
                # 1. 并行选择动作
                for i, agent in enumerate(self.agents):
                    if success_agent_mask[i]:
                        action = -1 # disable move
                    else:
                        action = agent.select_action(states[i], env.xy_agents[i], env.actions)
                    actions.append(action)
                actions_all.append(np.array(actions).astype(np.int32).reshape(1, -1))
                
                # 2. 执行动作，与环境交互
                next_states, rewards, dones, truncated, info = env.step(actions)
                rf_size = info['rf_size'] 
                next_states_xy = info['states_xy']  
                image_id = info['image_id']  
                image = info['image']   
                   

                visual_data_all += [env.get_visual_data()]
                rewards_all.append(np.array(rewards).astype(np.float32).reshape(1, -1))
                
                # 3. 存储经验到缓冲区
                if self.freezing_success_agent:
                    for i, agent in enumerate(self.agents):
                        if success_agent_mask[i]:
                            continue # 已经达到目标位置的agent不再更新对应的replay buffer, 保持当前样本集
                        agent.replay_buffer.push(states[i], states_xy[i], actions[i], rewards[i].item(), next_states[i], next_states_xy[i], dones[i], image_id, image, rf_size)
                else:
                    for i, agent in enumerate(self.agents):
                        agent.replay_buffer.push(states[i], states_xy[i], actions[i], rewards[i].item(), next_states[i], next_states_xy[i], dones[i], image_id, image, rf_size)

                if self.freezing_success_agent: # 更新 success_agent_mask
                    success_agent_mask = [success_agent_mask[i] or info['success_agent_mask'][i] for i in range(self.num_agents)]
                    if np.all(success_agent_mask):
                        assert np.all(dones)
                 
                states = next_states
                states_xy = next_states_xy
                # 终止条件检查
                if all(dones) or truncated:
                    success_agents += [info['success_agents']] # episode结束时 agent达到目标位置的比例
                    states,states_xy = env.reset() # reset for a new episode
                    success_agent_mask = [False for _ in range(self.num_agents)]
                    break

        actions_all = np.concatenate(actions_all, axis=0)
        rewards_all = np.concatenate(rewards_all, axis=0)
        ent_agents = []
        for i_agent in range(actions_all.shape[1]):
            actions = actions_all[:, i_agent]
            _, counts = np.unique(actions, return_counts=True)
            p = counts/counts.sum()
            ent_agents.append((-p * np.log2(p)).sum().item())
        # 4. 训练更新
        train_info = self.train_step(batch_size=batch_size, update_steps=update_steps)
        
        # 5. 衰减探索率
        for agent in self.agents:
            if agent.epsilon > agent.epsilon_min:
                agent.epsilon *= agent.epsilon_decay
            if agent.epsilon < agent.epsilon_min:
                agent.epsilon = agent.epsilon_min
                
        # 6. 定期更新目标网络
        if hasattr(self, 'episode_count'):
            if self.episode_count % target_update_freq == 0:
                logger.info("update target network")
                for agent in self.agents:
                    agent.update_target_network()
        max_num_visual_data = min([env.max_steps, len(visual_data_all)])
        visual_data_all = visual_data_all[-max_num_visual_data:]
        vis_step = max([max_num_visual_data // 100,1])
        visual_data_all = visual_data_all[::vis_step]
        visual_data = torch.from_numpy(np.concatenate([np.transpose(d, (2, 0, 1))[None] for d in visual_data_all], axis=0))
        visual_data_image = make_grid(visual_data, nrow=10, padding=5, normalize=False).permute(1, 2, 0).numpy()
        del visual_data_all
        gc.collect()
        train_info.update(
            {"entropy": ent_agents, 
             "visual_data": visual_data_image,
             "reward_mean": np.mean(rewards_all, axis=0),
             "reward_std": np.std(rewards_all, axis=0),
             "success_agents": tuple(success_agents),
             "epsilon_agents": tuple([a.epsilon for a in self.agents])
             }
        )
        return train_info
    
def train(config_file: str, dataset_root: str, desc: str = "marl", device: str = "cuda:0", weights_file: str | None = None, eval_freq: int = 10):
    with open(config_file, 'rb') as f:
        cfg = tomllib.load(f)

    seed = cfg.get("seed",42)
    rng = set_seed(seed)
    face_size = cfg.get("face_size", 512)
    rf_size = cfg.get('rf_size', 64)
    num_episode = cfg.get("num_episode", 1000)
    history_len = cfg.get("history_len", 5) # 占用内存
    max_steps = cfg.get("max_steps", 1000)  # max steps each eposide
    actions = cfg.get("actions", [[1, 0], [-1, 0], [0, 1], [0, -1]])
    reward_scale = cfg.get("reward_scale", 10.0)
    batch_size = cfg.get("batch_size", 64)
    dataset_size_max = cfg.get("dataset_size_max", None)
    agent_xy_init_method = cfg.get("agent",{}).get("xy_init_method","random")
    agent_use_double_dqn = cfg.get("agent",{}).get("use_double_dqn",False)
    update_steps = cfg.get("update_steps", 100)
    shared_patch_encoder = cfg.get("shared_patch_encoder", True)
    target_update_freq = cfg.get("target_update_freq", 10)
    agent_action_selection_mode = cfg.get("agent_action_selection_mode", "dfs")
    use_abs_dist_as_reward = cfg.get("env",{}).get("use_abs_dist_as_reward",True) 
    # freezing_success_agent: 已经达到目标的agent不再更新replay buffer, 不再移动, 但依然参与训练(避免震荡)
    freezing_success_agent = cfg.get("training",{}).get("freezing_success_agent",False)
    target_range_steps = cfg.get("training",{}).get("target_range_steps",(1,))
    training_loss = cfg.get("training",{}).get("loss", "mse")
    
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    exp_name = f"{os.path.splitext(os.path.basename(config_file))[0]}_{desc}"
    outdir = os.path.join(os.path.dirname(os.path.basename(__file__)), "runs", f"{exp_name}_{timestamp}")
    summary_writer = SummaryWriter(outdir, "marl")
    shutil.copyfile(config_file, os.path.join(outdir, "cfg.toml"))
    logger.add(os.path.join(outdir, "marl.log"))

    logger.info(f"{cfg}")
    logger.info(f"device: {device}")
    logger.info(f"saved folder {outdir}")
    logger.info(f"actions: {actions}")
    logger.info(f"agent action selection model: {agent_action_selection_mode}")
    logger.info(f"use_abs_dist_as_reward: {use_abs_dist_as_reward}")
    logger.info(f"use_double_dqn: {agent_use_double_dqn}")

    input_channels = 3 * history_len
    input_shape = (1, input_channels, rf_size, rf_size)
    n_actions = len(actions)
    logger.info(f"face size: {face_size}") 
    env_tr = MARLEnv(dataset_root, split="train", rng=rng, face_size=face_size, rf_size=rf_size, history_len=history_len,
                     actions=actions, reward_scale=reward_scale, dataset_size_max=dataset_size_max,
                     max_steps=max_steps, agent_xy_init_method=agent_xy_init_method,use_abs_dist_as_reward=use_abs_dist_as_reward,
                     target_range_steps=target_range_steps)
    env_te = MARLEnv(dataset_root, split="test", rng=rng, face_size=face_size, rf_size=rf_size, history_len=history_len,
                     actions=actions, reward_scale=reward_scale, dataset_size_max=dataset_size_max,
                     max_steps=max_steps, agent_xy_init_method=agent_xy_init_method,use_abs_dist_as_reward=use_abs_dist_as_reward,
                     target_range_steps=(target_range_steps[-1],))
    trainer = MARLTrain(
        num_agents=env_tr.num_agents,
        input_shape=input_shape,
        n_actions=n_actions,
        device=device,
        rng = rng,
        use_double_dqn=agent_use_double_dqn,
        training_loss=training_loss,
        replay_buffer_size=max_steps*100, # make replay buffer larger enough
        epsilon_decay =0.95, 
        agent_action_selection_mode=agent_action_selection_mode,
        use_weight_sharing=shared_patch_encoder,  # Collab-DQN 
        freezing_success_agent=freezing_success_agent
    )
    evaluater = MARLEval(
        num_agents=env_tr.num_agents,
        input_shape=input_shape,
        n_actions=n_actions,
        device=device,
        rng = rng,
        use_weight_sharing=shared_patch_encoder,  # Collab-DQN 
        agent_action_selection_mode=agent_action_selection_mode
    )
    logger.info(f"dataset size: train {len(env_tr)} | test {len(env_te)}")

    if weights_file is not None:
        trainer.load(weights_file)
        logger.warning(f"loading weights from {weights_file}")
        
       

    top_eval_metric = np.finfo("float").min
    for episode in range(num_episode):
        trainer.episode_count = episode
        episode_info = trainer.run_episode(env_tr,
                                           #max_steps_collected: should be much smaller than replay buffer capacity. and resample 0.25 to training each episode
                                           max_steps_collected=max([max_steps, 4*batch_size*update_steps//update_steps]), 
                                           batch_size=batch_size, 
                                           update_steps=update_steps,
                                           target_update_freq=target_update_freq)
        
        success_agents_percentile = _calc_percentile(episode_info['success_agents'])
        logger.info(f"{episode:0>5d} success_agents: {_format_container(success_agents_percentile)}")
        
        for k,v in success_agents_percentile.items():
            summary_writer.add_scalar(f"train/success_agentsAll/{k}", v, episode)
        logger.info(f"{episode:0>5d} epsilon_agents: {_format_container(_calc_percentile(episode_info['epsilon_agents']))}")
        
        for i_agent, agent in enumerate(trainer.agents):
            agent_stat = agent.replay_buffer.stat()
            logger.info(
                f"agent {i_agent} replay:"
                f"{_format_container(agent_stat)}"
            )
            for k, v in agent_stat['reward'].items():
                summary_writer.add_scalar(f"train/success/agent_{i_agent:0>3d}/{k}", v, episode)
        
        target_q_stats = _calc_percentile(episode_info['target_q'])
        logger.info(f"{episode:0>5d} target_q: {_format_container(target_q_stats)}") 
        
        target_net_out_stats = _calc_percentile(episode_info['target_net_out'])
        logger.info(f"{episode:0>5d} target_net_out: {_format_container(target_net_out_stats)}") 
          
        if np.percentile(episode_info['success_agents'],80) >= 0.98:
            target_range_step_last =  env_tr.next_target_range_step()
            if target_range_step_last is not None:
                trainer.reset_replay_buffer()  # clear buffer to remove sample from last curriculum 
                logger.warning("reset replay buffer")       
                p50 = np.percentile(episode_info['success_agents'],50)
                p80 = np.percentile(episode_info['success_agents'],80)
                p95 = np.percentile(episode_info['success_agents'],95)
                logger.warning(f"target range step {target_range_step_last} | success agents (P50,P80,P95)=({p50:5.2f},{p80:5.2f},{p95:5.2f})")
            else:
                logger.warning("stop update target range step")
                
        for i_agent in range(env_tr.num_agents):
            for name in ['loss', 'entropy', 'reward_mean', 'reward_std','tderr_mean_smoothed_agents','tderr_std_agents','tderr_mean_agents']:
                summary_writer.add_scalar(f"train/{name}/agent{i_agent:0>3d}", episode_info[name][i_agent], episode)
            summary_writer.add_scalar(f"train/epsilon/agent_{i_agent:0>3d}", trainer.agents[i_agent].epsilon, episode)
       
        # TODO: move this into env class
        env_tr.init_jitter = env_tr.init_jitter * env_tr.decay_jitter 
        if env_tr.init_jitter < 0.01:
            env_tr.init_jitter = 0.01
        logger.info(
            f"episode {episode:0>4d}|"
            f"init jitter: {env_tr.init_jitter:3.2f}|"
            f"loss: {_format_container(episode_info['loss'])}|"
            f"ent: {_format_container(episode_info['entropy'])}")

        
        if episode % eval_freq == 0:
            cv2.imwrite(os.path.join(outdir, f"{episode:0>8d}.jpg"), episode_info['visual_data'])
            trainer.save(os.path.join(outdir, "last.pth"))
            evaluater.load(os.path.join(outdir, "last.pth"))
            success_stat = evaluater.stat_success(env_te,sampling=1) # sampling > 1 to make eval faster
            summary_writer.add_scalar("eval/success/all", success_stat['all'], episode)
            summary_writer.add_scalar("eval/success/agents", success_stat['agents'], episode)
            logger.info(
                f"episode {episode:0>4d}|"
                f"eval sucess: {_format_container(success_stat['agents'])}"
                )
            for i in range(len(success_stat['agent_mask'])):
                v = success_stat['agent_mask'][i]
                logger.info(f"agent_{i:0>2d} succcess {v:3.2f}")
            if success_stat['agents'] > top_eval_metric:
                top_eval_metric = success_stat['agents']
                logger.info(
                    f"NEW BEST MODEL! Episode {episode}|"
                    f"eval success: {success_stat['agents']:4.3f}")
                evaluater.save(os.path.join(outdir, "best.pth"))
    logger.info(f"training stop with best metric {top_eval_metric}")
    trainer.save(os.path.join(outdir, "final.pth"))

    summary_writer.close()

class MARLEval(MARLBase):
    def __init__(self, num_agents, input_shape, n_actions, 
                 agent_action_selection_mode:str,
                 rng: np.random.Generator,
                 device: str = "cuda:0", use_weight_sharing=False):
        super().__init__(num_agents=num_agents, 
                         input_shape=input_shape, 
                         n_actions=n_actions,
                         device=device,
                         rng = rng,
                         replay_buffer_size=1,
                         use_weight_sharing=use_weight_sharing, flag_train=False,
                         agent_action_selection_mode=agent_action_selection_mode)
    def stat_success(self, env, sampling=1) -> dict:
        success_stat = {"all":[], "agents":[], "agent_mask":[]}
        for i_episode in tqdm(range(0,len(env),sampling),desc=f"eval with sampling={sampling}"):
            info = self.run_episode(env)
            success_stat['all'].append(info['success'])
            success_stat['agents'].append(info['success_agents'])
            success_stat['agent_mask'].append(np.array(info['success_agent_mask']).reshape(1,-1)) 
        return {"all": np.mean(success_stat['all']), 
                "agents": np.mean(success_stat['agents']), 
                "agent_mask":np.mean(success_stat['agent_mask'],axis=0).flatten().tolist()}

    def run_episode(self, env):
        # 重置环境，获取初始状态列表 [state_0, state_1, ..., state_k]
        states,_ = env.reset()
        actions_all = []
        visual_data_all = [env.get_visual_data()]
        rewards_all = []
        dxdy_all = np.zeros((self.num_agents, 2), dtype=np.float32)
        success: bool = False
        while True:
            actions = []
            # 1. 并行选择动作
            for i, agent in enumerate(self.agents):
                action = agent.select_action(states[i], env.xy_agents[i], env.actions)
                actions.append(action)
                dxdy_all[i, :] += np.array(env.actions[action])
            actions_all.append(np.array(actions).astype(np.int32).reshape(1, -1))
            
            # 2. 执行动作，与环境交互
            next_states, rewards, dones, truncated, info = env.step(actions)

            visual_data_all += [env.get_visual_data()]
            rewards_all.append(np.array(rewards).astype(np.float32).reshape(1, -1))
            states = next_states
            
            # 终止条件检查
            if all(dones) or truncated:
                #logger.warning(f"get end of one episode\n \t {info}")
                break

        success = info.get("success", False)
        success_agents = info.get("success_agents",0.0)
        success_agent_mask = info.get("success_agent_mask",[False] * self.num_agents)

        actions_all = np.concatenate(actions_all, axis=0)
        rewards_all = np.concatenate(rewards_all, axis=0)
        ent_agents = []
        for i_agent in range(actions_all.shape[1]):
            actions = actions_all[:, i_agent]
            _, counts = np.unique(actions, return_counts=True)
            p = counts/counts.sum()
            ent_agents.append((-p * np.log2(p)).sum())
                
        visual_data = torch.from_numpy(np.concatenate([np.transpose(d, (2, 0, 1))[None] for d in visual_data_all], axis=0))
        visual_data_image = make_grid(visual_data, nrow=10, padding=5, normalize=False).permute(1, 2, 0).numpy()
        eval_info = {
            "entropy": ent_agents,
            "visual_data": visual_data_image.copy(),
            "reward_mean": np.mean(rewards_all, axis=0),
            "reward_std": np.std(rewards_all, axis=0),
            "dxdy": dxdy_all,
            "success": success,
            "success_agents": success_agents,
            "success_agent_mask": success_agent_mask
            } 
        return eval_info

def test(exp_dir, dataset_root, device: str = "cuda:0"):
    config_file = os.path.join(exp_dir, "cfg.toml")
    weights_file = os.path.join(exp_dir, 'last.pth')
    assert os.path.exists(config_file) and os.path.exists(weights_file)
    with open(config_file, 'rb') as f:
        cfg = tomllib.load(f)
    seed = cfg.get("seed",42)
    rng = set_seed(seed)
    face_size = cfg.get("face_size", 512)
    rf_size = cfg.get('rf_size', 64)
    history_len = cfg.get("history_len", 5)
    max_steps = cfg.get("max_steps", 1000)
    actions = cfg.get("actions", [[1, 0], [-1, 0], [0, 1], [0, -1]])
    reward_scale = cfg.get("reward_scale", 10.0)
    dataset_size_max = cfg.get("dataset_size_max", None)
    agent_init_xy = cfg.get("agent",{}).get("init_xy_mode","random")
    shared_patch_encoder = cfg.get("shared_patch_encoder", True)
    use_abs_dist_as_reward = cfg.get("env",{}).get("use_abs_dist_as_reward",True) 
    target_range_steps = cfg.get("training",{}).get("target_range_steps",(1,))




    outdir = os.path.join(exp_dir, "test")
    logger.add(os.path.join(outdir, "marl_test.log"))

    logger.info(f"{cfg}")
    logger.info(f"saved folder {outdir}")
    logger.info(f"actions: {actions}")
    logger.info(f"dataset: {dataset_root}")
    logger.info(f"use_abs_dist_as_reward: {use_abs_dist_as_reward}")
    logger.info(f"target_range_steps: {target_range_steps}")
    
    input_channels = 3 * history_len
    input_shape = (1, input_channels, rf_size, rf_size)
    n_actions = len(actions)
    logger.info(f"face size: {face_size}") 
    env = MARLEnv(dataset_root, split="test", face_size=face_size, rng=rng,
                  rf_size=rf_size, history_len=history_len,
                  actions=actions, reward_scale=reward_scale, dataset_size_max=dataset_size_max,
                  max_steps=max_steps, random_init_xy=random_init_xy, 
                  use_abs_dist_as_reward=use_abs_dist_as_reward,
                  target_range_steps=(target_range_steps[-1],))
    eval = MARLEval(
        num_agents=env.num_agents,
        input_shape=input_shape,
        n_actions=n_actions,
        device=device,
        rng = rng,
        use_weight_sharing=shared_patch_encoder
        )
    logger.info(f"dataset size {len(env)}")

    eval.load(weights_file)
    logger.warning(f"loading weights from {weights_file}")
    
    for episode in range(len(env)):
        eval.episode_count = episode
        episode_info = eval.run_episode(env)
        logger.info(f"""Episode {episode}|success: {episode_info['success']}|success_agents: {episode_info['success_agents']}""")
        cv2.imwrite(os.path.join(outdir, f"{episode:0>8d}.jpg"), episode_info['visual_data'])
    
if __name__ == "__main__":
    Fire(
        {"train": train, 
         "test": test
         })