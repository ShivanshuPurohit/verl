from collections import deque
import threading
import random
import ray
import time
from verl.single_controller.base import Worker
from verl import DataProto
from tensordict import TensorDict
import torch

# Local replay buffer to store (query, response, old_logprobs, ground_truth, dataset)
class ReplayBuffer:
    def __init__(self, capacity: int):
        print("Constructing replay buffer with capacity", capacity)
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.latest_batch = []  # Track the most recent batch added
        self.lock = threading.Lock()
        
    def add(self, item, add_to_latest_batch=False):
        with self.lock:
            self.buffer.append(item)
            if add_to_latest_batch:
                self.latest_batch.append(item)
    
    def sample(self, batch_size: int, latest_batch_prob=0.0):
        """
        Sample items with a probability of drawing from the latest batch.
        
        Args:
            batch_size: Number of items to sample
            latest_batch_prob: Probability (0.0-1.0) of sampling from the latest batch
                               rather than the entire buffer
        
        Returns:
            List of sampled items
        """
        with self.lock:
            if not self.buffer:
                return []
                
            if not self.latest_batch or latest_batch_prob <= 0:
                # If no latest batch or probability is 0, sample from the entire buffer
                return random.sample(list(self.buffer), min(batch_size, len(self.buffer)))
            
            result = []
            remaining_to_sample = batch_size
            
            # Determine how many samples to take from each source
            latest_batch_samples = []
            main_buffer_samples = []
            
            for _ in range(remaining_to_sample):
                # For each item, decide whether to sample from latest batch or main buffer
                if random.random() < latest_batch_prob and self.latest_batch:
                    source = self.latest_batch
                    target_list = latest_batch_samples
                else:
                    source = list(self.buffer)
                    target_list = main_buffer_samples
                
                if source:
                    target_list.append(random.choice(source))
            
            result = latest_batch_samples + main_buffer_samples
            
            # Ensure we don't have duplicates
            seen = set()
            unique_results = []
            for item in result:
                item_id = id(item)  # Use object id as a unique identifier
                if item_id not in seen:
                    seen.add(item_id)
                    unique_results.append(item)
            
            # If we lost some items due to deduplication, sample more
            if len(unique_results) < batch_size and len(self.buffer) >= batch_size:
                additional_needed = batch_size - len(unique_results)
                potential_additions = [item for item in self.buffer if id(item) not in seen]
                if potential_additions:
                    unique_results.extend(random.sample(potential_additions, 
                                          min(additional_needed, len(potential_additions))))
            
            return unique_results
    
    def size(self) -> int:
        with self.lock:
            return len(self.buffer)
    
    def __getitem__(self, idx):
        with self.lock:
            return self.buffer[idx]
            
    def clear(self):
        with self.lock:
            self.buffer.clear()
            self.latest_batch = []
    
    def add_batch(self, items: list):
        """
        Add a batch of items and mark them as the latest batch.
        
        Args:
            items: List of items to add to the buffer
        """
        # Clear the latest batch tracking
        self.latest_batch = []
        
        # Add each item to both the main buffer and latest batch tracking
        for item in items:
            self.add(item, add_to_latest_batch=True)


@ray.remote
class SharedReplayBufferWorker:
    """
    A Ray-based worker that serves as a shared replay buffer between inference worker and learner.
    This allows both processes to coordinate with proper synchronization.
    """
    def __init__(self, capacity: int):
        print(f"Initializing SharedReplayBufferWorker with capacity {capacity}")
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.latest_batch = []
    
    def add(self, item):
        """Add a single item to the buffer"""
        # Check if we're currently syncing weights
        # if self.syncing_weights:
        #     print("Buffer is currently locked for weight syncing, waiting...")
        #     # Simple polling approach for sync status
        #     while self.syncing_weights:
        #         time.sleep(0.1)
        #     print("Weight sync complete, proceeding with add")
                
        self.buffer.append(item)
        return True
    
    def add_batch(self, items):
        """Add a batch of items to the buffer and mark as latest batch"""
        # Clear the latest batch tracking
        self.latest_batch = []
        
        # Add all items to both the buffer and latest batch
        for item in items:
            self.buffer.append(item)
            self.latest_batch.append(item)
            
        return len(items)
    
    def sample(self, batch_size: int, latest_batch_prob=0.0):
        """Sample items from the buffer with option to prioritize latest batch"""
        if not self.buffer:
            return []
            
        if not self.latest_batch or latest_batch_prob <= 0:
            # Sample from entire buffer
            sampled = random.sample(list(self.buffer), min(batch_size, len(self.buffer)))
        else:
            # Sample with probability of drawing from latest batch
            latest_batch_samples = []
            main_buffer_samples = []
            
            for _ in range(batch_size):
                # For each item, decide whether to sample from latest batch or main buffer
                if random.random() < latest_batch_prob and self.latest_batch:
                    source = self.latest_batch
                    target_list = latest_batch_samples
                else:
                    source = list(self.buffer)
                    target_list = main_buffer_samples
                
                if source:
                    target_list.append(random.choice(source))
            
            sampled = latest_batch_samples + main_buffer_samples
            
            # Ensure we don't have duplicates
            seen = set()
            unique_results = []
            for item in sampled:
                item_id = id(item)  # Use object id as a unique identifier
                if item_id not in seen:
                    seen.add(item_id)
                    unique_results.append(item)
            
            # If we lost some items due to deduplication, sample more
            if len(unique_results) < batch_size and len(self.buffer) >= batch_size:
                additional_needed = batch_size - len(unique_results)
                potential_additions = [item for item in self.buffer if id(item) not in seen]
                if potential_additions:
                    unique_results.extend(random.sample(potential_additions, 
                                          min(additional_needed, len(potential_additions))))
            
            sampled = unique_results
        
        return sampled
    
    def size(self):
        """Return the current size of the buffer"""
        return len(self.buffer)
    
    def get_stats(self):
        """Return statistics about the buffer"""
        return {
            "size": len(self.buffer),
            "capacity": self.capacity,
            "fill_ratio": len(self.buffer) / self.capacity if self.capacity > 0 else 0,
            "latest_batch_size": len(self.latest_batch),
        }
    
    def clear(self):
        """Clear the buffer"""
        self.buffer.clear()
        self.latest_batch = []
        return True

@ray.remote
class RolloutWorker:
    """
    A Ray actor that runs inference for the RL system.
    Generates new experiences and adds them to the shared replay buffer.
    """
    def __init__(self, config, replay_buffer_worker_ref=None):
        self.config = config
        self.replay_buffer_worker_ref = replay_buffer_worker_ref
        
        # Initialize the rollout worker group
        self.rollout_wg = None  # Will be initialized in setup
        self.rm_wg = None       # Will be initialized in setup
        self.reward_fn = None   # Will be initialized in setup
        self.kl_ctrl = None     # Will be initialized in setup
        
        # Flags for tracking
        self.data_generation_count = 0
        self.is_running = False
        self.should_stop = False
        self.is_paused = False
        self.pause_condition = threading.Condition()
        
    def setup(self, data_queue_ref, metrics_queue_ref, rollout_wg, rm_wg, reward_fn, kl_ctrl):
        """Setup the worker with necessary references and components"""
        self.data_queue = data_queue_ref
        self.metrics_queue_ref = metrics_queue_ref
        self.rollout_wg = rollout_wg
        self.rm_wg = rm_wg
        self.reward_fn = reward_fn
        self.kl_ctrl = kl_ctrl
        return True
        
    def generate_data(self, batch_dict):
        """Generate a single batch of data and add to replay buffer"""
        import uuid
        import numpy as np
        from verl.trainer.ppo.ray_trainer import apply_kl_penalty, compute_advantage
        
        # Check if we're paused or if the replay buffer is syncing weights
        with self.pause_condition:
            while self.is_paused:
                print("RolloutWorker is paused, waiting for resume signal...")
                self.pause_condition.wait()
                print("RolloutWorker resumed")
        
        # Additionally check buffer sync status
        # if self.replay_buffer_worker_ref:
        #     is_syncing = ray.get(self.replay_buffer_worker_ref.is_syncing_weights.remote())
        #     if is_syncing:
        #         print("Waiting for weight sync to complete before generating...")
        #         while ray.get(self.replay_buffer_worker_ref.is_syncing_weights.remote()):
        #             time.sleep(0.5)
        #         print("Weight sync complete, continuing with generation")
        
        # Prepare the generation batch
        batch = DataProto.from_single_dict(batch_dict)
        gen_batch = batch.pop(batch_keys=['input_ids', 'attention_mask', 'position_ids'])
        
        # Generate sequences
        start_time = time.time()
        gen_batch_output = self.rollout_wg.generate_sequences(gen_batch).get()
        generation_time = time.time() - start_time
        print(f"Generated sequences in {generation_time:.3f}s")
        
        # Increment generation counter
        self.data_generation_count += 1
        
        # Add unique IDs
        batch.non_tensor_batch['uid'] = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))],
                                                 dtype=object)
        
        # Repeat to align with responses
        batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
        batch = batch.union(gen_batch_output)
        
        # Compute rewards
        use_rm = hasattr(self, 'rm_wg') and self.rm_wg is not None
        if use_rm:
            reward_tensor = self.rm_wg.compute_rm_score(batch)
            batch = batch.union(reward_tensor)
            
        # Combine with rule-based reward
        reward_tensor = self.reward_fn(batch)
        batch.batch['token_level_scores'] = reward_tensor

        # Apply KL penalty if needed
        if not self.config.actor_rollout_ref.actor.get('use_kl_loss', False):
            batch, kl_metrics = apply_kl_penalty(batch,
                                                kl_ctrl=self.kl_ctrl,
                                                kl_penalty=self.config.algorithm.kl_penalty)
        else:
            batch.batch['token_level_rewards'] = batch.batch['token_level_scores']
            
        # Create response mask
        response_mask = batch.batch['attention_mask'][:, -batch.batch['responses'].size(-1):]
        batch.batch['response_mask'] = response_mask
        
        # Compute advantage
        batch = compute_advantage(batch,
                                 adv_estimator=self.config.algorithm.adv_estimator,
                                 gamma=self.config.algorithm.get("gamma", 1.0),
                                 lam=self.config.algorithm.get("lam", 1.0),
                                 tau=self.config.algorithm.get("tau", 1.0))
        
        # Add to replay buffer
        num_items = batch.batch['advantages'].shape[0]
        
        # Create individual items for the buffer
        items = []
        for i in range(num_items):
            item = DataProto(batch=TensorDict({
                key: value[i].reshape(1, -1) if isinstance(value, torch.Tensor) else value
                for key, value in batch.batch.items()
            }, batch_size=[1]))
            
            # Add non_tensor_batch data if it exists
            if hasattr(batch, 'non_tensor_batch'):
                for key, value in batch.non_tensor_batch.items():
                    if isinstance(value, (list, np.ndarray)) and len(value) > i:
                        item.non_tensor_batch[key] = value[i:i+1]
            
            # Add meta_info if it exists
            if hasattr(batch, 'meta_info'):
                for key, value in batch.meta_info.items():
                    item.meta_info[key] = value
                    
            items.append(item)
        
        # Add batch to replay buffer
        if self.replay_buffer_worker_ref:
            added = ray.get(self.replay_buffer_worker_ref.add_batch.remote(items))
            print(f"Added {added} items to replay buffer from rollout worker")

        self.metrics_queue_ref.put({
            "data_generation_count": self.data_generation_count,
            "items_generated": num_items,
            "generation_time": generation_time
        })
        
    
    def run_generation_loop(self, max_iters=None):
        """Run the generation loop to continuously produce data"""
        print("Starting RolloutWorker generation loop")
        self.is_running = True
        self.should_stop = False
        self.is_paused = False
        
        try:
            counter = 0
            while True:
                if self.should_stop:
                    print("Stopping RolloutWorker due to stop signal (before queue get)")
                    break

                print(f"Worker waiting to get batch {counter} from queue...")
                # Use timeout to allow checking should_stop periodically
                try:
                    batch_dict = self.data_queue.get(block=True, timeout=1.0)
                except ray.queue.Empty: # from queue import Empty
                    print("Queue empty, checking stop flag...")
                    continue # Go back to check self.should_stop

                print(f"Worker got batch {counter} from queue.")

                if batch_dict is None: # Check for sentinel
                    print("Worker received sentinel value, exiting loop.")
                    break

                if max_iters is not None and counter >= max_iters:
                    print("Reached max iters, breaking")
                    # Note: Need to decide if you drain the queue or just stop processing
                    break

                if self.should_stop:
                    print("Stopping RolloutWorker due to stop signal (after queue get)")
                    break

                # Generate data
                self.generate_data(batch_dict)
                counter += 1
        except Exception as e:
            import traceback
            traceback.print_stack()
            print("Got exception:", e)
            raise e
        finally:
            self.is_running = False
            print("RolloutWorker generation loop finished")
            
        return {"data_generation_count": self.data_generation_count}
    
    def pause(self):
        """Pause the worker temporarily"""
        with self.pause_condition:
            self.is_paused = True
            print("RolloutWorker paused")
        return True
    
    def resume(self):
        """Resume the paused worker"""
        with self.pause_condition:
            self.is_paused = False
            self.pause_condition.notify_all()
            print("RolloutWorker resumed")
        return True
    
    def stop(self):
        """Signal the worker to stop permanently"""
        self.should_stop = True
        # If worker is paused, unpause it so it can see the stop signal
        with self.pause_condition:
            self.is_paused = False
            self.pause_condition.notify_all()
        return True
    
    def is_active(self):
        """Check if the worker is still running"""
        return self.is_running
    
    def is_paused(self):
        """Check if the worker is currently paused"""
        return self.is_paused
    
    def get_stats(self):
        """Get statistics about the worker"""
        return {
            "data_generation_count": self.data_generation_count,
            "is_running": self.is_running,
            "is_paused": self.is_paused
        }
