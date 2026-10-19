import os
import warnings
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
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
#UNIT_STEP = 0.01
#
#ACTIONS: dict[int, tuple[float, float]] = {
#    0: (UNIT_STEP, 0.0),
#    1: (-UNIT_STEP, 0.0),
#    2: (0.0, UNIT_STEP),
#    3: (0.0, -UNIT_STEP),
#    }

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
        with open(annotation_file, 'r') as f:
            for items in tqdm(f, desc="load WFLW..."):
                items = items.strip()
                if items == "":
                    continue
                items = items.split(' ')
                XYall = np.array([float(d) for d in items[0:196]], dtype=np.float32).reshape(-1, 2)
                bbox = np.array([int(d) for d in items[196:196+4]], dtype=np.int32).reshape(-1)
                pose, expression, illumination, makeup, occlusion, blur = np.array([int(d) for d in items[196+4:196+4+6]])
                if pose != 0 or expression != 0 or illumination != 0 or makeup != 0 or occlusion != 0 or blur != 0:
                    continue
                
                image_path = items[-1].strip()
                image_path = os.path.join(image_root, image_path)
                assert os.path.exists(image_path)
                
                
                # fix: some landmark is out of bbox
                px0, px1 = XYall[:, 0].min(), XYall[:, 0].max()
                py0, py1 = XYall[:, 1].min(), XYall[:, 1].max()
                x0, y0, x1, y1 = bbox
                x0, x1 = min([px0, x0]),  max((px1, x1))
                y0, y1 = min([py0, y0]),  max((py1, y1))
                cx, cy = (x1+x0)/2,  (y1+y0)/2
                facew, faceh = (x1-x0),  (y1-y0)
                x0, x1 = (cx - facew/2 * 1.5), (cx + facew/2 * 1.5)
                y0, y1 = (cy - faceh/2 * 1.5), (cy + faceh/2 * 1.5)
                if image_path in images_info:
                    info = images_info[image_path]
                    H, W = info['height'],  info['width']
                else:
                    H, W = cv2.imread(image_path, 0).shape
                    images_info[image_path] = {"width": W,  "height": H}
                x0, y0 = max((x0, 0)), max((y0, 0))
                x1, y1 = min((x1, W)), min((y1, H))
                if y1 - y0 < min_face_size or x1 - x0 < min_face_size:
                    continue
                bbox = np.array([x0, y0, x1, y1], dtype=np.float32).reshape(-1)
                
                XY = [XYall[i].reshape(1, 2) for i in points_indices]
                XY = np.concatenate(XY, axis=0).astype(np.float32)

                XY = XY / np.array([W, H]).reshape(1, 2)
                bbox = bbox / np.array([W, H, W, H]).reshape(-1)
                self.dataset.append(
                    {"image": image_path, "bbox": bbox, "points": XY}
                )
                
                if dataset_size_max is not None and len(self.dataset) >= dataset_size_max:
                    break
                
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
        H, W = image.shape[0:2]
        left, top, right, bottom = item['bbox']
        nx0, ny0, nx1, ny1 = (item['bbox'] * np.array([W, H, W, H])).astype(np.int32)
        face_image = image[ny0:ny1, nx0:nx1]
        XY = item['points'] - np.array([left, top]).reshape(-1, 2)  
        XY = XY / np.array([right - left, bottom - top]).reshape(1,2) 
        if self.face_size is not None: # TODO: make square before resize
            face_image = cv2.resize(face_image, [self.face_size, self.face_size], interpolation=cv2.INTER_LINEAR)
        return face_image, XY, np.ones_like(XY, dtype=np.float32).reshape(-1, 2) * 0.5

class MARLEnv:
    def __init__(self, dataset_root: str, split: str, rf_size: int, face_size: int,
                 unit_step: float, actions: list[tuple[float, float]], reward_scale: float = 10.0,
                 history_len: int = 1, dataset_size_max: int | None = None, max_steps: int = 1000,
                 random_init_xy: bool = True, init_provider=None, init_jitter: float = 0.0,
                 per_agent_done: bool = True):
        self.UNIT_STEP = unit_step
        self.ACTIONS = actions
        self.per_agent_done = per_agent_done  # P1: agents that reach the target freeze for the rest of the episode

        self.flag_train = True if split.lower() == "train" else False
        self.random_init_xy = random_init_xy
        # hybrid scheme: heatmap prior init. init_provider(image_bgr) -> (n, 2) normalized xy
        self.init_provider = init_provider
        self.init_jitter = init_jitter
        self.last_init_err = float("nan")  # mean init-to-target distance (before jitter)

        self.history_len = history_len
        self.dataset = DatasetWFLW(dataset_root, split=split, dataset_size_max=dataset_size_max, face_size=face_size)
        self.num_agents = self.dataset.get_point_num()
        self.agent_dones = np.zeros(self.num_agents, dtype=bool)  # properly initialized in reset()
        self.steps_to_success = np.full(self.num_agents, -1.0, dtype=np.float32)  # per-agent arrival step, -1 = not arrived
        self.last_frames = np.zeros((self.num_agents, 3, rf_size, rf_size), dtype=np.uint8)  # P2: newest raw frame per agent
        self.xy_agents = np.zeros((self.num_agents, 2), dtype=np.float32)
        self.history_obs_agents = [deque(maxlen=history_len) for _ in range(self.num_agents)]
        self.resize_method = cv2.INTER_AREA

        self.rf_size = rf_size

        self.current_image, self.current_image_patched = None, None
        self.xy_targets = None
        
        self.current_file_id = 0

        self.max_steps = max_steps 
        self.num_steps = 0

        self.reward_scale = reward_scale #all xy is in [0,1]

        self.xy_init_count = np.zeros((face_size, face_size), dtype=np.float32)
        self.xy_init_count[face_size//2, face_size//2] = 1.0 #avoid zero-division
        self.face_size = face_size

        return
    
    def __len__(self):
        return len(self.dataset)
    
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
        return np.vstack([x, y]).transpose() / self.face_size

    def reset(self, file_id: int = -1):
        self._switch_image(file_id_required=file_id)
        for i_agent in range(self.num_agents):
            self.history_obs_agents[i_agent].clear()
            for _ in range(self.history_len):
                self.history_obs_agents[i_agent].append(np.zeros((1, 3, self.rf_size, self.rf_size), dtype=np.float32))
        self._reset_agent_dones()
        return self.get_observation()

    def _reset_agent_dones(self):
        """P1: per-agent termination state. Agents already within threshold at init are done at step 0."""
        self.steps_to_success = np.full(self.num_agents, -1.0, dtype=np.float32)
        if self.per_agent_done and self.xy_targets is not None:
            dist = np.linalg.norm(self.xy_targets - self.xy_agents, axis=-1)
            self.agent_dones = (dist < self.UNIT_STEP).astype(bool)
            self.steps_to_success[np.where(self.agent_dones)[0]] = 0.0
        else:
            self.agent_dones = np.zeros(self.num_agents, dtype=bool)
    
    def get_visual_data(self, size: int = 64):
        assert self.current_image is not None and self.current_image_patched is not None
        assert self.xy_targets is not None
        H, W = self.current_image.shape[0:2]
        visual_data = deepcopy(self.current_image)
        cmap = plt.get_cmap("jet")
        for i_agent in range(self.num_agents):
            color = [int(c * 255) for c in cmap((i_agent + 1)/self.num_agents)[0:3]]
            nx, ny = int(W * self.xy_agents[i_agent, 0]), int(H * self.xy_agents[i_agent, 1])
            cv2.circle(visual_data, (nx, ny), self.rf_size//2, color, 3)

            tx, ty = int(W * self.xy_targets[i_agent, 0]), int(H * self.xy_targets[i_agent, 1])
            cv2.line(visual_data, (nx, ny), (tx, ty), color, 3)
        return cv2.resize(visual_data, (size, size), interpolation=cv2.INTER_AREA)

    def _prepare_current_image(self) -> None:
        assert self.current_file_id >= 0 and self.current_file_id < len(self.dataset)
        image_data, label_data, label_data_init = self.dataset.get(self.current_file_id)
        self.current_image = image_data
        self.xy_targets = label_data
        pad_size = self.rf_size
        if self.init_provider is not None:  # heatmap prior init: start near predicted peaks
            xy_init = np.asarray(self.init_provider(image_data), dtype=np.float32)
            assert xy_init.shape == (self.num_agents, 2)
            self.last_init_err = float(np.mean(np.linalg.norm(xy_init - self.xy_targets, axis=-1)))
            if self.flag_train and self.init_jitter > 0:  # jitter curriculum: RL learns to correct residual error
                xy_init = xy_init + np.random.normal(0.0, self.init_jitter, xy_init.shape).astype(np.float32)
                xy_init = np.clip(xy_init, 0.0, 1.0)
            self.xy_agents = xy_init
        elif self.random_init_xy:
            self.xy_agents = self._random_init_xy()
        else:
            self.xy_agents = label_data_init # avoid constant init location
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
   
    def _crop_roi(self, xy: tuple[float, float]):
        assert self.current_image is not None and self.current_image_patched is not None
        H, W = self.current_image.shape[0:2]
        nx, ny = int(W * xy[0]), int(H * xy[1])
        size = int(self.rf_size)
        assert not (nx < 0 or nx > W or ny < 0 or ny > H) # reference to _take_actions()
  
        x0 = int(nx - size/2)
        x1 = x0 + size
        y0 = int(ny - size/2)
        y1 = y0 + size

        offset = self.rf_size 
        # P2: return the RAW uint8 patch; normalization happens when the stacked obs is built
        roi_data = self.current_image_patched[y0 + offset:y1 + offset, x0 + offset:x1 + offset]
        return roi_data

    def get_observation(self) -> np.ndarray:
        assert self.current_file_id >= 0 and self.current_file_id < len(self.dataset)
        for i_agent, (x, y) in enumerate(self.xy_agents):
            raw = self._crop_roi((x, y))  # (H, W, 3) uint8
            frame = np.ascontiguousarray(np.transpose(raw, (2, 0, 1)))  # (3, H, W) uint8
            if not (self.per_agent_done and self.agent_dones[i_agent]):
                self.last_frames[i_agent] = frame  # P1: frozen agents keep their last frame
            data = ((frame.astype(np.float32) - 128.0) / 128.0)[np.newaxis, :, :, :]  # (1, 3, H, W)
            self.history_obs_agents[i_agent].append(data)
        obs_all = []
        for i_agent in range(self.num_agents):
            obs = np.concatenate([d for d in self.history_obs_agents[i_agent]], axis=1)
            obs_all.append(obs)
        obs_data = np.concatenate(obs_all, axis=0)
        return obs_data.copy() 

    def _take_actions(self, actions: list[int]) -> None:
        assert len(actions) == self.num_agents
        assert self.current_image is not None
        for i_agent, i_action in enumerate(actions):
            if self.per_agent_done and self.agent_dones[i_agent]:
                continue  # P1: done agents are frozen at their target (docs/improvement-suggestions.md #5.2)
            x = self.xy_agents[i_agent, 0] + self.ACTIONS[i_action][0]
            y = self.xy_agents[i_agent, 1] + self.ACTIONS[i_action][1]
            x = min(max(x, 0.0), 1.0)
            y = min(max(y, 0.0), 1.0)
            self.xy_agents[i_agent, :] = (x, y)
        return

    def step(self, actions: list[int]) -> tuple[np.ndarray, float, list[bool], dict]:
        """
        actions: [(dx0,dy0), ... (dxn, dyn)]
        return: (observation, reward, play_end, aux_info)
        P1: with per_agent_done, each agent terminates individually when dist < UNIT_STEP:
        its coordinates freeze, it stops earning rewards, and success is recorded at arrival time.
        """
        self.num_steps += 1
        #xy_agents_old = np.copy(self.xy_agents)
        dones_before = self.agent_dones.copy()
        self._take_actions(actions)
        xy_agents_new = self.xy_agents

        obs_new = self.get_observation()
        assert self.xy_targets is not None
        dist_new = np.linalg.norm(self.xy_targets - xy_agents_new,  axis=-1, ord=2)
        #dist_old = np.linalg.norm(self.xy_targets - xy_agents_old, axis=-1, ord=2)

        #reward = (dist_old - dist_new) * self.reward_scale  #reward shaping    
        reward = 1 - dist_new * self.reward_scale / 1.0

        if self.per_agent_done:
            newly_arrived = (~dones_before) & (dist_new < self.UNIT_STEP)
            for i_agent in np.where(newly_arrived)[0]:
                self.steps_to_success[i_agent] = float(self.num_steps)  # success recorded at done time
            self.agent_dones = dones_before | (dist_new < self.UNIT_STEP)
            timeout = self.num_steps >= self.max_steps
            # already-done agents earn nothing this step; the arriving agent still gets its real final reward
            rewards = np.where(dones_before, 0.0, reward).astype(np.float32)
            dones = self.agent_dones.copy()
            if timeout:  # D2: keep current semantics, timeout ends the episode with done=True (bootstrap masked)
                dones[:] = True
            return obs_new, rewards, [bool(d) for d in dones], {
                "success": bool(np.all(self.agent_dones)), "num_steps": self.num_steps,
                "success_agents": float(np.mean(self.agent_dones)), "timeout": bool(timeout),
                "steps_to_success": self.steps_to_success.copy(), "active_before": ~dones_before}

        flag_success = np.all(dist_new < self.UNIT_STEP)
        if flag_success:
            return obs_new, reward, [True]*self.num_agents, {"success": True, "num_steps": self.num_steps, 
                                                             "success_agents": np.mean(dist_new < self.UNIT_STEP)} 
        
        if self.num_steps >= self.max_steps:
            return obs_new, reward, [True]*self.num_agents, {"success": False, "num_steps": self.num_steps, 
                                                             "success_agents": np.mean(dist_new < self.UNIT_STEP)} 
        
        return obs_new, reward, [False]*self.num_agents, {"success": False, "num_steps": self.num_steps,
                                                          "success_agents": np.mean(dist_new < self.UNIT_STEP)} 

class PatchEncoder(nn.Module):
    def __init__(self, shape_input: tuple[int, int, int, int]) -> None:
        super().__init__()
        B, C, H, W = shape_input
        self.conv1 = nn.Conv2d(C, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

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
        self.head = nn.Linear(512, n_actions)
    def forward(self, x):
        x = self.patch_encoder(x)
        x = torch.relu(self.fc1(x))
        return self.head(x)
    

class ReplayBuffer:
    """Frame-level replay buffer (P2): ~40x memory reduction.

    Each push stores ONE uint8 frame -- the newest frame of the transition's current state.
    Full stacked observations (history_len frames, oldest->newest) are reconstructed at
    sample time by walking back until an episode-start entry or the eviction boundary
    (zero-padded on the left, matching env.reset() semantics). The next_state of terminal
    transitions is zero-filled (masked by (1-done) in the TD target). Old design stored two
    full float32 stacks per transition (~480KB each); now one uint8 frame (~12KB).
    """
    def __init__(self, capacity: int = 4096, history_len: int = 1, frame_shape: tuple = (3, 64, 64)):
        self.capacity = capacity
        self.history_len = history_len
        C, H, W = frame_shape
        self.frame_shape = (C, H, W)
        self.frames = np.zeros((capacity, C, H, W), dtype=np.uint8)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.ep_starts = np.zeros(capacity, dtype=bool)  # this entry is the first transition of an episode
        self.pos = 0
        self.size = 0

    def push(self, frame: np.ndarray, action: int, reward: float, done: bool, ep_start: bool = False):
        k = self.pos
        self.frames[k] = frame
        self.actions[k] = action
        self.rewards[k] = reward
        self.dones[k] = 1.0 if done else 0.0
        self.ep_starts[k] = bool(ep_start)
        self.pos = (k + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def _slots(self, ord_idx: np.ndarray) -> np.ndarray:
        """insertion-order indices (0 = oldest valid entry) -> ring slots"""
        return (self.pos - self.size + ord_idx) % self.capacity

    def _stack(self, end_idx: np.ndarray) -> np.ndarray:
        """Build (B, history_len, C, H, W) uint8 stacks whose NEWEST frame sits at
        insertion-order index end_idx[i]. Walking back stops after an episode-start
        entry or the eviction boundary; older channels stay zero."""
        B = len(end_idx)
        h = self.history_len
        C, H, W = self.frame_shape
        # missing channels are filled with pixel 128 (= normalized 0.0), matching env.reset()'s
        # zero-filled float history -- uint8 0 would wrongly normalize to -1.0
        out = np.full((B, h, C, H, W), 128, dtype=np.uint8)
        alive = np.ones(B, dtype=bool)
        for j in range(h):  # j = steps back from the newest frame
            o = end_idx - j
            valid = alive & (o >= 0) & (o < self.size)
            sel = np.where(valid)[0]
            if len(sel) == 0:
                break
            slots = self._slots(o[sel])
            out[sel, h - 1 - j] = self.frames[slots]
            alive[sel] = ~self.ep_starts[slots]  # stop before the previous episode
        return out

    def _gather(self, ord_idx: np.ndarray):
        slots = self._slots(ord_idx)
        dones_np = self.dones[slots]
        cur_u8 = self._stack(ord_idx)
        nxt_u8 = np.full_like(cur_u8, 128)  # terminal next_states stay neutral (masked by (1-done) in the target)
        nonterm = ~dones_np.astype(bool)
        if nonterm.any():  # terminal next_states are zero (masked by (1-done) in the target)
            nxt_u8[nonterm] = self._stack(ord_idx[nonterm] + 1)
        h, (C, H, W) = self.history_len, self.frame_shape
        # normalization identical to env: (pixel - 128) / 128, bit-exact in float32
        states = ((cur_u8.astype(np.float32) - 128.0) / 128.0).reshape(len(ord_idx), h * C, H, W)
        next_states = ((nxt_u8.astype(np.float32) - 128.0) / 128.0).reshape(len(ord_idx), h * C, H, W)
        return (torch.from_numpy(states), torch.from_numpy(self.actions[slots]).long().unsqueeze(1),
                torch.from_numpy(self.rewards[slots]).unsqueeze(1), torch.from_numpy(next_states),
                torch.from_numpy(dones_np).unsqueeze(1))

    def sample(self, batch_size: int):
        idx = np.asarray(random.sample(range(self.size), batch_size))  # requires size >= batch_size
        return self._gather(idx)

    def sample_recent(self, n: int):
        """the n most recently pushed entries (freshest data, used for validation TD)"""
        n = min(n, self.size)
        idx = np.arange(self.size - n, self.size)
        return self._gather(idx)

    def __len__(self):
        return self.size

# =======================
# 3. agents
# =======================

class TDMonitor:
    """P4: TD-error based training health monitor. See docs/td-error-guide.md.

    Scale-free health rules (all heuristics to draw attention -- cross-check with eval metrics):
      level 3 (fatal):   NaN/Inf in TD or Q
      level 2 (explode): fast EMA of |TD| (or |Q|) grew more than rate_ratio x within the last
                         rate_window episodes (RATE of growth, so a healthy bootstrap ramp that
                         plateaus does not alarm, while sustained divergence does), or |Q|
                         crossed q_hard_bound (theoretical |Q| <= max|r|/(1-gamma) backstop)
      level 1 (stagnant): rel-TD EMA changed less than stagnation_delta over stagnation_window
      level 0 (ok)
    """
    def __init__(self, num_agents: int, alpha_fast: float = 0.05, min_history: int = 50,
                 rate_window: int = 50, rate_ratio: float = 2.0, q_hard_bound: float = 1000.0,
                 stagnation_alpha: float = 0.01, stagnation_window: int = 200, stagnation_delta: float = 0.01):
        self.num_agents = num_agents
        self.alpha_fast = alpha_fast
        self.min_history = min_history
        self.rate_window = rate_window
        self.rate_ratio = rate_ratio
        self.q_hard_bound = q_hard_bound
        self.stagnation_alpha = stagnation_alpha
        self.stagnation_window = stagnation_window
        self.stagnation_delta = stagnation_delta
        self.ema_fast = {"td_abs": None, "q_abs": None}   # per-agent arrays
        self.ema_rel = None                               # per-agent rel-TD EMA
        self.n_updates = 0
        self.fast_hist = {"td_abs": deque(maxlen=rate_window + 1),
                          "q_abs": deque(maxlen=rate_window + 1)}
        self.hist_td_rel = deque(maxlen=stagnation_window + 1)
        self.health = 0
        self.status = "OK (warming up)"

    @staticmethod
    def _ema_update(cur, values, alpha):
        if cur is None:
            return values.copy()  # NaN entries mark never-active agents (excluded from aggregates)
        # keep the previous EMA where the current value is NaN (agent skipped this round)
        return np.where(np.isfinite(values), alpha * values + (1.0 - alpha) * cur, cur)

    @staticmethod
    def _agg(arr: np.ndarray) -> float:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            return float(np.nanmean(arr)) if np.isfinite(arr).any() else float("nan")

    def update(self, metrics: dict) -> tuple[int, str]:
        """metrics: per-agent arrays with keys td_abs, td_rel, q_mean (NaN for inactive agents)."""
        td_abs, td_rel, q_mean = metrics.get("td_abs"), metrics.get("td_rel"), metrics.get("q_mean")
        if td_abs is None or not np.isfinite(td_abs).any():
            return self.health, self.status  # nothing collected yet
        if not (np.isfinite(td_abs).all() and np.isfinite(q_mean[np.isfinite(q_mean)]).all() \
                and np.isfinite(td_rel[np.isfinite(td_rel)]).all()):
            self.health, self.status = 3, "FATAL: NaN/Inf in TD/Q (diverged)"
            logger.error(f"td_health: {self.status}")
            return self.health, self.status
        td_abs = np.asarray(td_abs, dtype=np.float64)
        td_rel = np.asarray(td_rel, dtype=np.float64)
        q_abs = np.abs(np.asarray(q_mean, dtype=np.float64))
        self.ema_fast["td_abs"] = self._ema_update(self.ema_fast["td_abs"], td_abs, self.alpha_fast)
        self.ema_fast["q_abs"] = self._ema_update(self.ema_fast["q_abs"], q_abs, self.alpha_fast)
        self.ema_rel = self._ema_update(self.ema_rel, td_rel, self.stagnation_alpha)
        td_fast, q_fast = self._agg(self.ema_fast["td_abs"]), self._agg(self.ema_fast["q_abs"])
        self.n_updates += 1
        self.fast_hist["td_abs"].append(td_fast)
        self.fast_hist["q_abs"].append(q_fast)
        self.hist_td_rel.append(self._agg(self.ema_rel))
        self.health, self.status = 0, "OK"
        if self.n_updates >= self.min_history:  # warm up a full rate window before rate checks
            td_ref = self.fast_hist["td_abs"][0]   # value ~rate_window episodes ago
            q_ref = self.fast_hist["q_abs"][0]
            if np.isfinite(td_ref) and td_fast > self.rate_ratio * max(td_ref, 1e-8):
                self.health, self.status = 2, f"WARN: |TD| exploding ({td_fast:.3g} > {self.rate_ratio}x {td_ref:.3g} within {self.rate_window} eps)"
                logger.warning(f"td_health: {self.status} -- consider lower lr / loss_type=huber / target_update_freq")
            elif np.isfinite(q_ref) and q_fast > self.rate_ratio * max(q_ref, 1e-8):
                self.health, self.status = 2, f"WARN: |Q| exploding ({q_fast:.3g} > {self.rate_ratio}x {q_ref:.3g} within {self.rate_window} eps)"
                logger.warning(f"td_health: {self.status} -- Q overestimation/divergence")
            elif np.isfinite(q_fast) and q_fast > self.q_hard_bound:
                # backstop: theoretical |Q| <= max|r| / (1-gamma); healthy Q here is single/double digits
                self.health, self.status = 2, f"WARN: |Q| beyond hard bound ({q_fast:.3g} > {self.q_hard_bound})"
                logger.warning(f"td_health: {self.status} -- unbounded Q growth (slow divergence)")
        if self.health == 0 and len(self.hist_td_rel) > self.stagnation_window:
            ref = self.hist_td_rel[0]
            if np.isfinite(ref) and abs(self.hist_td_rel[-1] - ref) <= self.stagnation_delta * max(abs(ref), 1e-8):
                self.health, self.status = 1, f"WARN: rel-TD stagnant ({self.hist_td_rel[-1]:.4g} vs {ref:.4g} > {self.stagnation_window} eps)"
                logger.warning(f"td_health: {self.status} -- check exploration / task reachability")
        return self.health, self.status

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
        if x < self.xrange[0] or y < self.yrange[0] or x > self.xrange[1] or y > self.xrange[1]:
            return
        if xy not in self.xy2visit:
            self.xy2visit[xy] = num
        else:
            self.xy2visit[xy] += num
        return
    def get_visit(self, xy) -> float:
        x, y = xy
        if x < self.xrange[0] or y < self.yrange[0] or x > self.xrange[1] or y > self.xrange[1]:
            return np.nan
        if xy in self.xy2visit:
            return self.xy2visit[xy]
        return 0.1 #init to 1.0
        
        

class Agent:
    def __init__(self, agent_id, input_shape, n_actions, lr=1e-4, gamma=0.99, 
                 device: str = "cuda:0",
                 epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01, 
                 shared_patch_encoder: PatchEncoder | None = None, flag_train: bool = True):
        """
        Args:
            shared_policy_net: 如果共享权重，传入共享的网络对象
        """
        self.flag_train = flag_train
        self.device = torch.device(device)
        self.agent_id = agent_id
        self.n_actions = n_actions
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        _, C_total, H, W = input_shape
        # P2: frame-level buffer; one uint8 frame per transition, stacks rebuilt at sample time
        self.replay_buffer = ReplayBuffer(capacity=4096, history_len=C_total // 3, frame_shape=(3, H, W))

        self.policy_net = CollabDQN(input_shape, n_actions, patch_encoder=shared_patch_encoder)
        self.policy_net.to(self.device)
        # P3: optimizer is owned by MARLBase (single Adam with param groups) so that the shared
        # encoder receives ONE coherent joint update per step instead of N conflicting ones
        # 目标网络通常不共享，或者以较慢速度同步，这里保持独立以稳定训练
        self.target_net = CollabDQN(input_shape, n_actions)
        self.target_net.to(self.device)
        self.update_target_network()

        _, _, H, W = input_shape
        self.explorer = Explorer((0, W-1), (0, H-1))
        self.W, self.H = W, H


    def select_action(self, state, current_xy: np.ndarray, ACTIONS):
        if self.flag_train and random.random() < self.epsilon:
            x, y = current_xy
            visits = []
            xys = []
            for i_action in range(self.n_actions):
                nx, ny = int(self.W * (ACTIONS[i_action][0] + x)), int(self.H * (ACTIONS[i_action][1] + y))
                visits.append(self.explorer.get_visit((nx, ny)))
                xys.append((nx, ny))
            visits = np.nan_to_num(np.asarray(visits, dtype=np.float64), nan=0.0)
            total = visits.sum()
            if total <= 0.0:  # all candidates out of range (e.g. agent at image corner): uniform random
                i_action = random.randrange(self.n_actions)
            else:
                p = (visits / total).tolist()
                i_action = np.random.choice(np.arange(self.n_actions), size=1, p=p, replace=False).item()
            self.explorer.set_visit(xys[i_action])
            return i_action
        else:
            with torch.no_grad():
                # state shape: (4, H, W) -> add batch dim: (1, 4, H, W)
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.policy_net(state_tensor).cpu()
                i_action = q_values.max(1).indices.item()

                x, y = current_xy
                nx, ny = int(self.W *(ACTIONS[i_action][0] + x)), int(self.H * (ACTIONS[i_action][1] + y))
                self.explorer.set_visit((nx, ny))
                return i_action

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())


# =======================
# Marl 核心协作逻辑
# =======================
class MARLBase:
    def __init__(self, num_agents, input_shape, n_actions, 
                 device: str = "cuda:0", use_weight_sharing=False, flag_train=True,
                 lr: float = 1e-4, encoder_lr: float | None = None):

        self.episode_count = 0
        self.device = torch.device(device)
        self.lr = lr
        self.encoder_lr = lr if encoder_lr is None else encoder_lr  # P3: independent lr for the shared encoder

        self.num_agents = num_agents
        self.use_weight_sharing = use_weight_sharing
        
            
        # 如果使用权重共享，创建一个共享的策略网络
        if use_weight_sharing:
            self.shared_patch_encoder = PatchEncoder(input_shape)
        else:
            self.shared_patch_encoder = None

        # 2. 初始化智能体
        self.agents = []
        for i in range(num_agents):
            agent = Agent(
                agent_id=i,
                input_shape=input_shape,
                n_actions=n_actions,
                device=device,
                shared_patch_encoder=self.shared_patch_encoder,
                flag_train=flag_train
            )
            self.agents.append(agent)

        if flag_train:
            self._build_optimizers()

    def _build_optimizers(self):
        """P3: ONE Adam over disjoint param groups.
        - shared encoder params form a single group (one consistent Adam state) with encoder_lr;
        - each agent's head (fc1+head) forms its own group with lr (updates identical to before);
        - without sharing, groups cover each agent's full net (mathematically identical to
          the previous per-agent Adams since Adam state is per-parameter).
        """
        groups = []
        if self.use_weight_sharing:
            groups.append({"params": self.shared_patch_encoder.parameters(), "lr": self.encoder_lr})
        for agent in self.agents:
            if self.use_weight_sharing:
                head_params = [p for name, p in agent.policy_net.named_parameters()
                               if not name.startswith("patch_encoder.")]
            else:
                head_params = list(agent.policy_net.parameters())
            groups.append({"params": head_params, "lr": self.lr})
        self.optimizer = optim.Adam(groups)
            

    def save(self, path, save_replay: bool = False):
        states = {}
        for agent in self.agents:
            entry = {f"agent{agent.agent_id}_target_net": agent.target_net.state_dict(),
                     f"agent{agent.agent_id}_policy_net": agent.policy_net.state_dict()}
            if save_replay:  # P2: replay buffers optional, default off (~50MB/agent vs ~15MB total without)
                entry[f"agent{agent.agent_id}_replay"] = agent.replay_buffer
            states.update(entry)
        torch.save(states, path)
        return

    def load(self, path):
        with open(path, 'rb') as f:
            states = torch.load(f, map_location=self.device, weights_only=False)  # local trusted ckpts
        for agent in self.agents:
            agent.target_net.load_state_dict(states[f"agent{agent.agent_id}_target_net"])
            agent.policy_net.load_state_dict(states[f"agent{agent.agent_id}_policy_net"])
            rb = states.get(f"agent{agent.agent_id}_replay", None)
            if isinstance(rb, ReplayBuffer) and rb.frame_shape == agent.replay_buffer.frame_shape \
                    and rb.history_len == agent.replay_buffer.history_len:
                agent.replay_buffer = rb  # resume new-format buffer
            else:
                # A2: legacy (deque) or missing/shape-mismatched buffer -> discard, re-collect experience;
                # network weights load normally, training resumes with fresh data collection
                logger.warning(f"agent{agent.agent_id}: replay buffer not restored (legacy/missing), "
                               f"experience will be re-collected")

        if self.shared_patch_encoder is not None:
            self.shared_patch_encoder.load_state_dict(self.agents[0].policy_net.patch_encoder.state_dict())
            for k in range(1, len(self.agents)):
                self.agents[k].policy_net.patch_encoder = self.shared_patch_encoder  # set shared encoder!!!!!
        return

class MARLTrain(MARLBase):
    def __init__(self, num_agents, input_shape, n_actions, 
                 device: str = "cuda:0", use_weight_sharing=False,
                 lr: float = 1e-4, encoder_lr: float | None = None, loss_type: str = "mse"):
        super().__init__(num_agents=num_agents, 
                         input_shape=input_shape, 
                         n_actions=n_actions,
                         device=device,
                         use_weight_sharing=use_weight_sharing, flag_train=True,
                         lr=lr, encoder_lr=encoder_lr)
        # P3: configurable loss (cfg loss_type); huber is more robust to reward-scale outliers
        if loss_type == "huber":
            self.loss_fn = nn.SmoothL1Loss()  # delta = 1.0
        else:
            assert loss_type == "mse", f"unknown loss_type: {loss_type}"
            self.loss_fn = nn.MSELoss()
        self.td_monitor = TDMonitor(num_agents)  # P4: TD-error health tracking (docs/td-error-guide.md)
    def train_step(self, batch_size: int = 64, update_steps: int = 100)->dict:
        """
        P3: joint gradient update.
        Per update step: every active agent backpropagates its own loss (head gradients are
        unchanged vs the old per-agent optimizers); the shared encoder's accumulated gradient
        is then divided by the number of active agents (B2: mean gradient) and a SINGLE Adam
        step is taken -- one coherent update with one optimizer state, replacing the old
        N conflicting sequential optimizer steps on the same parameters.
        Agents whose replay buffer holds fewer than batch_size samples are skipped (P1:
        buffers grow at different speeds once agents terminate individually).
        """
        gamma = self.agents[0].gamma
        active = [agent for agent in self.agents if len(agent.replay_buffer) >= batch_size]
        loss_all = []
        # P4: per-agent TD diagnostics, averaged over update steps (NaN for skipped agents)
        td_keys = ("td_abs", "td_signed", "td_rel", "q_mean", "y_mean")
        td_all = {k: [] for k in td_keys}
        if len(active) == 0:  # nothing to learn from yet
            out = {"loss": np.full(self.num_agents, np.nan)}
            out.update({k: np.full(self.num_agents, np.nan) for k in td_keys})
            return out
        for _ in range(update_steps):
            self.optimizer.zero_grad()
            loss_agents = np.full(self.num_agents, np.nan)
            td_agents = {k: np.full(self.num_agents, np.nan) for k in td_keys}
            for agent in active:
                states, actions, rewards, next_states, dones = agent.replay_buffer.sample(batch_size)

                states_gpu = states.to(self.device)
                actions_gpu = actions.to(self.device)
                rewards_gpu = rewards.to(self.device)
                next_states_gpu = next_states.to(self.device)
                dones_gpu = dones.to(self.device)

                # 计算当前Q值
                current_action_value = agent.policy_net(states_gpu)
                current_q = current_action_value.gather(1, actions_gpu)
                # 计算目标Q值
                with torch.no_grad():
                    next_q = agent.target_net(next_states_gpu).max(-1)[0].unsqueeze(1)
                    target_q = rewards_gpu + (gamma * next_q * (1 - dones_gpu))
                    
                loss = self.loss_fn(current_q, target_q)
                loss.backward()  # accumulates into shared encoder grads across agents
                loss_agents[agent.agent_id] = loss.detach().cpu().item()
                # P4: TD-error diagnostics (zero extra forward passes)
                with torch.no_grad():
                    td = current_q - target_q
                    td_agents["td_abs"][agent.agent_id] = td.abs().mean().item()
                    td_agents["td_signed"][agent.agent_id] = td.mean().item()
                    td_agents["td_rel"][agent.agent_id] = (td.abs() / (target_q.abs() + 1e-6)).mean().item()
                    td_agents["q_mean"][agent.agent_id] = current_q.mean().item()
                    td_agents["y_mean"][agent.agent_id] = target_q.mean().item()
            
            if self.use_weight_sharing:  # B2: encoder gets the MEAN of the agents' gradients
                for p in self.shared_patch_encoder.parameters():
                    if p.grad is not None:
                        p.grad /= len(active)
            self.optimizer.step()
            loss_all.append(loss_agents.reshape(1, -1))
            for k in td_keys:
                td_all[k].append(td_agents[k].reshape(1, -1))
        loss_all = np.mean(np.concatenate(loss_all, axis=0), axis=0)
        out = {"loss": loss_all}
        for k in td_keys:
            out[k] = np.mean(np.concatenate(td_all[k], axis=0), axis=0)
        return out

    def eval_td_recent(self, batch_size: int = 256, min_samples: int = 32) -> np.ndarray:
        """P4: mean |TD| on the freshest collected samples (no-grad), measured BEFORE this
        episode's gradient updates. If replay-TD falls but this does not, the net is just
        memorizing the buffer (see docs/td-error-guide.md #4)."""
        val = np.full(self.num_agents, np.nan)
        gamma = self.agents[0].gamma
        for agent in self.agents:
            if len(agent.replay_buffer) < min_samples:
                continue  # P1: agents that terminate early may have little fresh data
            states, actions, rewards, next_states, dones = \
                agent.replay_buffer.sample_recent(min(batch_size, len(agent.replay_buffer)))
            with torch.no_grad():
                q = agent.policy_net(states.to(self.device)).gather(1, actions.to(self.device))
                y = (rewards.to(self.device)
                     + gamma * agent.target_net(next_states.to(self.device)).max(-1)[0].unsqueeze(1)
                     * (1 - dones.to(self.device)))
                val[agent.agent_id] = (q - y).abs().mean().item()
        return val

    def run_episode(self, env, max_steps_collected: int = 100, batch_size: int = 64, update_steps=100, target_update_freq: int = 10):
        """
        运行一个完整的Episode，收集所有智能体的经验到共享缓冲区
        """
        # 重置环境，获取初始状态列表 [state_0, state_1, ..., state_k]
        states = env.reset()
        actions_all = []
        visual_data_all = [env.get_visual_data()]
        rewards_all = []   # (T, N): reward per agent per step (0 for frozen agents)
        activity_all = []  # (T, N): 1 if the agent acted this step (was not done before the step)
        actions_taken = [[] for _ in self.agents]  # per-agent actions actually executed
        steps_to_success_eps, success_agents_eps = [], []
        pending_ep = np.ones(self.num_agents, dtype=bool)  # P2: next push of each agent starts a new episode
        while len(rewards_all) <= max_steps_collected:
            while True:
                pre_dones = env.agent_dones.copy()
                frames_cur = env.last_frames.copy()  # P2: newest raw frames of the current states (before the step)
                actions = [0] * self.num_agents  # placeholder for done agents (ignored by env)
                # 1. 并行选择动作（P1: done agent 跳过，不再推理/探索/移动）
                for i, agent in enumerate(self.agents):
                    if pre_dones[i]:
                        continue
                    action = agent.select_action(states[i], env.xy_agents[i], env.ACTIONS)
                    actions[i] = action
                actions_all.append(np.array(actions).astype(np.int32).reshape(1, -1))
                
                # 2. 执行动作，与环境交互
                # env.step 应返回: next_states, rewards, dones
                next_states, rewards, dones, info = env.step(actions)

                if False:
                    for i_agent in range(next_states.shape[0]):
                        obs = np.clip(np.transpose(next_states[i_agent],(1, 2, 0)) * 128 + 128, 0, 255).astype(np.uint8)
                        frames = [obs[:, :, i:i+3] for i in range(0, obs.shape[-1], 3)]
                        frames_image = np.vstack(frames)
                        cv2.imwrite(f"./obs_agent_{i_agent:0>3d}.png", frames_image)


                visual_data_all += [env.get_visual_data()]
                rewards_all.append(np.asarray(rewards, dtype=np.float32).reshape(1, -1))
                activity_all.append((~pre_dones).astype(np.float32).reshape(1, -1))
                # 3. 存储经验到缓冲区（P1: done agent 停止收集；P2: 只存当前帧，采样时重构堆叠）
                for i, agent in enumerate(self.agents):
                    if pre_dones[i]:
                        continue
                    actions_taken[i].append(actions[i])
                    agent.replay_buffer.push(frames_cur[i], actions[i], rewards[i], dones[i],
                                             ep_start=bool(pending_ep[i]))
                    pending_ep[i] = False

                states = next_states
                
                # 终止条件检查
                if all(dones):
                    # logger.warning(f"get end of one episode\n \t {info}")
                    steps_to_success_eps.append(np.array(info.get("steps_to_success", 
                                                  np.full(self.num_agents, np.nan)), dtype=np.float32))
                    success_agents_eps.append(float(info.get("success_agents", 0.0)))
                    states = env.reset() # reset for a new episode
                    pending_ep[:] = True  # P2: next push of every agent is an episode start
                    break

        actions_all = np.concatenate(actions_all, axis=0)
        rewards_all = np.concatenate(rewards_all, axis=0)
        activity_all = np.concatenate(activity_all, axis=0)
        # per-agent stats over ACTIVE steps only (frozen agents would drag means toward 0)
        reward_mean = np.zeros(self.num_agents, dtype=np.float32)
        reward_std = np.zeros(self.num_agents, dtype=np.float32)
        for i_agent in range(self.num_agents):
            vals = rewards_all[:, i_agent][activity_all[:, i_agent] > 0]
            if len(vals) > 0:
                reward_mean[i_agent] = float(np.mean(vals))
                reward_std[i_agent] = float(np.std(vals))
        ent_agents = []
        for i_agent in range(self.num_agents):
            actions = np.asarray(actions_taken[i_agent])
            if len(actions) == 0:
                ent_agents.append(0.0)
                continue
            _, counts = np.unique(actions, return_counts=True)
            p = counts/counts.sum()
            ent_agents.append(float((-p * np.log2(p)).sum()))
        # P1: progress metrics over the episodes collected in this call
        if len(steps_to_success_eps) > 0:
            sts = np.stack(steps_to_success_eps, axis=0)
            sts[sts < 0] = np.nan  # -1 = never arrived
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)  # all-NaN columns are normal (agent never arrived)
                steps_to_success = np.nanmean(sts, axis=0) if not np.all(np.isnan(sts)) else np.full(self.num_agents, np.nan)
            episode_success_agents = float(np.mean(success_agents_eps))
        else:
            steps_to_success = np.full(self.num_agents, np.nan)
            episode_success_agents = float("nan")
        # 4. 训练更新（P4: 先测新鲜样本 TD，再更新，最后喂给健康监控）
        val_td_abs = self.eval_td_recent()
        train_info = self.train_step(batch_size=batch_size, update_steps=update_steps)
        train_info["val_td_abs"] = val_td_abs
        if getattr(self, "td_monitor", None) is not None:
            level, status = self.td_monitor.update(train_info)
            train_info["td_health"] = level
            train_info["td_health_status"] = status
        
        # 5. 衰减探索率
        for agent in self.agents:
            if agent.epsilon > agent.epsilon_min:
                agent.epsilon *= agent.epsilon_decay
                
        # 6. 定期更新目标网络
        if hasattr(self, 'episode_count'):
            if self.episode_count % target_update_freq == 0:
                for agent in self.agents:
                    agent.update_target_network()
        max_num_visual_data = min([env.max_steps, len(visual_data_all)])
        visual_data_all = visual_data_all[0:max_num_visual_data]
        visual_data = torch.from_numpy(np.concatenate([np.transpose(d, (2, 0, 1))[None] for d in visual_data_all], axis=0))
        visual_data_image = make_grid(visual_data, nrow=10, padding=5, normalize=False).permute(1, 2, 0).numpy()
        train_info.update(
            {"entropy": ent_agents, "visual_data": visual_data_image,
             "reward_mean": reward_mean,
             "reward_std": reward_std,
             "steps_to_success": steps_to_success,
             "episode_success_agents": episode_success_agents}
        )
        return train_info
    
def train(config_file: str, dataset_root: str, desc: str = "marl", device: str = "cuda:0", weights_file: str | None = None, eval_freq: int = 10):

    with open(config_file, 'rb') as f:
        cfg = tomllib.load(f)

    face_size = cfg.get("face_size", 512)
    rf_size = cfg.get('rf_size', 512)
    num_episode = cfg.get("num_episode", 1000)
    history_len = cfg.get("history_len", 5)
    max_steps = cfg.get("max_steps", 1000)  # max steps each eposide
    UNIT_STEP = cfg.get("UNIT_STEP", 0.01)
    ACTIONS_rel = cfg.get("ACTIONS", [[1, 0], [-1, 0], [0, 1], [0, -1]])
    reward_scale = cfg.get("reward_scale", 10.0)
    batch_size = cfg.get("batch_size", 64)
    dataset_size_max = cfg.get("dataset_size_max", None)
    random_init_xy = cfg.get("random_init_xy", True)
    update_steps = cfg.get("update_steps", 100)
    shared_patch_encoder = cfg.get("shared_patch_encoder", True)
    target_update_freq = cfg.get("target_update_freq", 10)
    # hybrid scheme: heatmap prior init + RL refinement
    init_mode = cfg.get("init_mode", "center")  # center | heatmap
    heatmap_weights = cfg.get("heatmap_weights", None)
    heatmap_init_jitter = cfg.get("heatmap_init_jitter", 0.0)  # initial jitter std (normalized)
    heatmap_init_jitter_min = cfg.get("heatmap_init_jitter_min", 0.0)
    heatmap_init_jitter_decay = cfg.get("heatmap_init_jitter_decay", 1.0)  # per-episode multiplicative decay
    per_agent_done = cfg.get("per_agent_done", True)  # P1: per-agent termination (agents freeze at target)
    save_replay = cfg.get("save_replay", False)  # P2: pickle replay buffers into checkpoints (default off)
    lr = cfg.get("lr", 1e-4)  # P3: per-agent head learning rate (was hardcoded default before)
    encoder_lr = cfg.get("encoder_lr", None)  # P3: shared encoder lr, defaults to lr
    loss_type = cfg.get("loss_type", "mse")  # P3: "mse" | "huber"
    
    ACTIONS = []
    for (x, y) in ACTIONS_rel:
        ACTIONS.append((x*UNIT_STEP, y*UNIT_STEP))

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    exp_name = f"{os.path.splitext(os.path.basename(config_file))[0]}_{desc}"
    outdir = os.path.join(os.path.dirname(os.path.basename(__file__)), "runs", f"{exp_name}_{timestamp}")
    summary_writer = SummaryWriter(outdir, "marl")
    shutil.copyfile(config_file, os.path.join(outdir, "cfg.toml"))
    logger.add(os.path.join(outdir, "marl.log"))

    logger.info(f"{cfg}")
    logger.info(f"device: {device}")
    logger.info(f"saved folder {outdir}")
    logger.info(f"ACTIONS: {ACTIONS}")
    logger.info(f"UNIT in pixel: {int(UNIT_STEP * face_size)}")

    assert face_size * UNIT_STEP > 1.0
    input_channels = 3 * history_len
    input_shape = (1, input_channels, rf_size, rf_size)
    n_actions = len(ACTIONS)
    logger.info(f"face size: {face_size}") 
    env_tr = MARLEnv(dataset_root, split="train", face_size=face_size, rf_size=rf_size, history_len=history_len, unit_step=UNIT_STEP,
                     actions=ACTIONS, reward_scale=reward_scale, dataset_size_max=dataset_size_max,
                     max_steps=max_steps, random_init_xy=random_init_xy, per_agent_done=per_agent_done)
    env_te = MARLEnv(dataset_root, split="test", face_size=face_size, rf_size=rf_size, history_len=history_len, unit_step=UNIT_STEP,
                     actions=ACTIONS, reward_scale=reward_scale, dataset_size_max=dataset_size_max,
                     max_steps=max_steps, random_init_xy=random_init_xy, per_agent_done=per_agent_done)
    trainer = MARLTrain(
        num_agents=env_tr.num_agents,
        input_shape=input_shape,
        n_actions=n_actions,
        device=device,
        use_weight_sharing=shared_patch_encoder,  # Collab-DQN 
        lr=lr, encoder_lr=encoder_lr, loss_type=loss_type
    )
    evaluater = MARLEval(
        num_agents=env_tr.num_agents,
        input_shape=input_shape,
        n_actions=n_actions,
        device=device,
        use_weight_sharing=shared_patch_encoder  # Collab-DQN 
    )
    logger.info(f"dataset size: train {len(env_tr)} | test {len(env_te)}")

    if init_mode == "heatmap":
        assert heatmap_weights is not None and os.path.exists(heatmap_weights), \
            f"init_mode=heatmap requires heatmap_weights, got: {heatmap_weights}"
        from marl.src.heatmap import load_heatmap_provider
        init_provider = load_heatmap_provider(heatmap_weights, num_points=env_tr.get_point_num(),
                                              face_size=face_size, device=device)
        env_tr.init_provider = init_provider
        env_te.init_provider = init_provider
        env_tr.init_jitter = heatmap_init_jitter  # eval env: no jitter, measure raw prior quality
        logger.info(f"init_mode=heatmap | weights: {heatmap_weights} | "
                    f"init_jitter(tr): {heatmap_init_jitter} (min {heatmap_init_jitter_min}, decay {heatmap_init_jitter_decay})")

    if weights_file is not None:
        trainer.load(weights_file)
        logger.warning(f"loading weights from {weights_file}")

    top_eval_metric = np.finfo("float").min
    for episode in range(num_episode):
        trainer.episode_count = episode
        if init_mode == "heatmap":  # anneal init jitter: coarse correction -> fine correction
            env_tr.init_jitter = max(heatmap_init_jitter_min, heatmap_init_jitter * (heatmap_init_jitter_decay ** episode))
        episode_info = trainer.run_episode(env_tr, max_steps_collected=max_steps*10, 
                                           batch_size=batch_size, update_steps=update_steps,
                                           target_update_freq=target_update_freq)
        for i_agent in range(env_tr.num_agents):
            for name in ['loss', 'entropy', 'reward_mean', 'reward_std']:
                summary_writer.add_scalar(f"{name}/agent{i_agent:0>3d}", episode_info[name][i_agent], episode)
            if np.isfinite(episode_info["steps_to_success"][i_agent]):
                summary_writer.add_scalar(f"train/steps_to_success/agent{i_agent:0>3d}", episode_info["steps_to_success"][i_agent], episode)
            summary_writer.add_scalar(f"epsilon/agent_{i_agent:0>3d}", trainer.agents[i_agent].epsilon, episode)
        if np.isfinite(episode_info["episode_success_agents"]):
            summary_writer.add_scalar("train/episode_success_agents", episode_info["episode_success_agents"], episode)
        # P4: TD-error diagnostics (docs/td-error-guide.md)
        def _agg(arr):
            if arr is None or not np.isfinite(arr).any():
                return float("nan")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                return float(np.nanmean(arr))
        tag_map = {'td_abs': 'td/abs_mean', 'td_signed': 'td/signed_mean', 'td_rel': 'td/rel',
                   'q_mean': 'td/q_mean', 'y_mean': 'td/target_q_mean', 'val_td_abs': 'td/val_abs'}
        for key, name in tag_map.items():
            for i_agent in range(env_tr.num_agents):
                v = episode_info[key][i_agent]
                if np.isfinite(v):
                    summary_writer.add_scalar(f"{name}/agent{i_agent:0>3d}", v, episode)
            if np.isfinite(episode_info[key]).any():
                summary_writer.add_scalar(f"{name}_all", _agg(episode_info[key]), episode)
        summary_writer.add_scalar("td/health", episode_info.get("td_health", 0), episode)
        if episode % eval_freq == 0:
            cv2.imwrite(os.path.join(outdir, f"{episode:0>8d}.jpg"), episode_info['visual_data'])
            trainer.save(os.path.join(outdir, "last.pth"), save_replay=save_replay)
            evaluater.load(os.path.join(outdir, "last.pth"))
            success_stat = evaluater.stat_success(env_te)
            summary_writer.add_scalar("eval/success/all", success_stat['all'], episode)
            summary_writer.add_scalar("eval/success/agents", success_stat['agents'], episode)
            summary_writer.add_scalar("eval/init_err", success_stat['init_err'], episode)
            logger.info(f"Episode {episode}|loss: {episode_info['loss']}|ent: {episode_info['entropy']}|"
                        f"eval sucess: {success_stat['agents']}|init_err: {success_stat['init_err']:.4f}|"
                        f"td: |δ|={_agg(episode_info.get('td_abs')):.3g} rel={_agg(episode_info.get('td_rel')):.3g} "
                        f"val={_agg(episode_info.get('val_td_abs')):.3g}|health: {episode_info.get('td_health_status', '-')}")
            if success_stat['agents'] > top_eval_metric:
                top_eval_metric = success_stat['agents']
                logger.info(f"NEW BEST MODEL! Episode {episode}|eval success: {success_stat['agents']}")
                evaluater.save(os.path.join(outdir, "best.pth"))
    logger.info(f"training stop with best metric {top_eval_metric}")
    trainer.save(os.path.join(outdir, "final.pth"), save_replay=save_replay)

    summary_writer.close()

class MARLEval(MARLBase):
    def __init__(self, num_agents, input_shape, n_actions, 
                 device: str = "cuda:0", use_weight_sharing=False):
        super().__init__(num_agents=num_agents, 
                         input_shape=input_shape, 
                         n_actions=n_actions,
                         device=device,
                         use_weight_sharing=use_weight_sharing, flag_train=False)
    def stat_success(self, env) -> dict:
        success_stat = {"all":[], "agents":[], "init_err":[]}
        for i_episode in range(len(env)):
            info = self.run_episode(env)
            success_stat['all'].append(info['success'])
            success_stat['agents'].append(info['success_agents'])
            success_stat['init_err'].append(info['init_err'])
        return {"all": np.mean(success_stat['all']), "agents": np.mean(success_stat['agents']),
                "init_err": float(np.mean(success_stat['init_err']))}

    def run_episode(self, env):
        # 重置环境，获取初始状态列表 [state_0, state_1, ..., state_k]
        states = env.reset()
        actions_all = []
        visual_data_all = [env.get_visual_data()]
        rewards_all = []
        dxdy_all = np.zeros((self.num_agents, 2), dtype=np.float32)
        success: bool = False
        success_agents: float = 0.0
        while True:
            pre_dones = env.agent_dones.copy()
            actions = [0] * self.num_agents  # placeholder for done agents (ignored by env)
            # 1. 并行选择动作（P1: done agent 跳过）
            for i, agent in enumerate(self.agents):
                if pre_dones[i]:
                    continue
                action = agent.select_action(states[i], env.xy_agents[i], env.ACTIONS)
                actions[i] = action
                dxdy_all[i, :] += np.array(env.ACTIONS[action])
            actions_all.append(np.array(actions).astype(np.int32).reshape(1, -1))
            
            # 2. 执行动作，与环境交互
            next_states, rewards, dones, info = env.step(actions)

            if False:
                for i_agent in range(next_states.shape[0]):
                    obs = np.clip(np.transpose(next_states[i_agent],(1, 2, 0)) * 128 + 128, 0, 255).astype(np.uint8)
                    frames = [obs[:, :, i:i+3] for i in range(0, obs.shape[-1], 3)]
                    frames_image = np.vstack(frames)
                    cv2.imwrite(f"./obs_agent_{i_agent:0>3d}.png", frames_image)

            visual_data_all += [env.get_visual_data()]
            rewards_all.append(np.array(rewards).astype(np.float32).reshape(1, -1))
            states = next_states
            
            # 终止条件检查
            if all(dones):
                #logger.warning(f"get end of one episode\n \t {info}")
                break

        success = info.get("success", False)
        success_agents = info.get("success_agents",0.0)

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
            "steps_to_success": info.get("steps_to_success", np.full(self.num_agents, np.nan)),
            "init_err": getattr(env, "last_init_err", float("nan"))
            } 
        return eval_info

def test(exp_dir, dataset_root, device: str = "cuda:0"):
    config_file = os.path.join(exp_dir, "cfg.toml")
    weights_file = os.path.join(exp_dir, 'last.pth')
    assert os.path.exists(config_file) and os.path.exists(weights_file)
    with open(config_file, 'rb') as f:
        cfg = tomllib.load(f)

    face_size = cfg.get("face_size", 512)
    rf_size = cfg.get('rf_size', 512)
    history_len = cfg.get("history_len", 5)
    max_steps = cfg.get("max_steps", 1000)
    UNIT_STEP = cfg.get("UNIT_STEP", 0.01)
    ACTIONS_rel = cfg.get("ACTIONS", [[1, 0], [-1, 0], [0, 1], [0, -1]])
    reward_scale = cfg.get("reward_scale", 10.0)
    dataset_size_max = cfg.get("dataset_size_max", None)
    random_init_xy = cfg.get("random_init_xy", True)
    shared_patch_encoder = cfg.get("shared_patch_encoder", True)
    per_agent_done = cfg.get("per_agent_done", True)
    init_mode = cfg.get("init_mode", "center")
    heatmap_weights = cfg.get("heatmap_weights", None)
    
    ACTIONS = []
    for (x, y) in ACTIONS_rel:
        ACTIONS.append((x*UNIT_STEP, y*UNIT_STEP))

    outdir = os.path.join(exp_dir, "test")
    logger.add(os.path.join(outdir, "marl_test.log"))

    logger.info(f"{cfg}")
    logger.info(f"saved folder {outdir}")
    logger.info(f"ACTIONS: {ACTIONS}")
    logger.info(f"UNIT in pixel: {int(UNIT_STEP * face_size)}")
    logger.info(f"dataset: {dataset_root}")

    assert face_size * UNIT_STEP > 1.0
    input_channels = 3 * history_len
    input_shape = (1, input_channels, rf_size, rf_size)
    n_actions = len(ACTIONS)
    logger.info(f"face size: {face_size}") 
    env = MARLEnv(dataset_root, split="test", face_size=face_size,
                  rf_size=rf_size, history_len=history_len, unit_step=UNIT_STEP,
                  actions=ACTIONS, reward_scale=reward_scale, dataset_size_max=dataset_size_max,
                  max_steps=max_steps, random_init_xy=random_init_xy, per_agent_done=per_agent_done)
    eval = MARLEval(
        num_agents=env.num_agents,
        input_shape=input_shape,
        n_actions=n_actions,
        device=device,
        use_weight_sharing=shared_patch_encoder
        )
    logger.info(f"dataset size {len(env)}")

    if init_mode == "heatmap":
        assert heatmap_weights is not None and os.path.exists(heatmap_weights)
        from marl.src.heatmap import load_heatmap_provider
        env.init_provider = load_heatmap_provider(heatmap_weights, num_points=env.get_point_num(),
                                                  face_size=face_size, device=device)
        logger.info(f"init_mode=heatmap | weights: {heatmap_weights}")

    eval.load(weights_file)
    logger.warning(f"loading weights from {weights_file}")
    
    for episode in range(len(env)):
        eval.episode_count = episode
        episode_info = eval.run_episode(env)
        logger.info(f"""Episode {episode}|success: {episode_info['success']}|success_agents: {episode_info['success_agents']}""")
        cv2.imwrite(os.path.join(outdir, f"{episode:0>8d}.jpg"), episode_info['visual_data'])
    
if __name__ == "__main__":
    def train_heatmap_entry(*args, **kwargs):
        from marl.src.heatmap import train_heatmap
        return train_heatmap(*args, **kwargs)

    Fire(
        {"train": train,
         "test": test,
         "train_heatmap": train_heatmap_entry,
         })