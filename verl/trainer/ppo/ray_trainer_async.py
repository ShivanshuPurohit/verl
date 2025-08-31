from verl.trainer.ppo.ray_trainer import (
    RayPPOTrainer,
    compute_timing_metrics,
    _timer,
    compute_advantage,
    compute_data_metrics,
    reduce_metrics,
    AdvantageEstimator,
    apply_kl_penalty,
)
import torch
import uuid
from pprint import pprint
import ray
from ray.util.queue import Queue
import threading
from copy import deepcopy

import numpy as np
from omegaconf import OmegaConf
from verl import DataProto


def load_data_onto_queue(loader, queue_ref):
    try:
        print("Data loading thread started.")
        for i, batch_dict in enumerate(loader):
            queue_ref.put(batch_dict) # Blocks if queue is full
        print("Loading thread: Dataloader exhausted, putting sentinel.")
        queue_ref.put(None) # Sentinel value
    except Exception as e:
        print(f"!!! Error in data loading thread: {e}")
        queue_ref.put(None) # Ensure sentinel is put even on error
    finally:
        print("Data loading thread finished.")

class RayTrainerAsync(RayPPOTrainer):
    def fit(self):
        """
        The training loop of PPO with asynchronous data generation using Ray.
        Data generation and model training run in separate Ray workers,
        coordinated through a shared replay buffer.
        """
        from verl.utils.tracking import Tracking
        from verl.utils.replay_buffer import SharedReplayBufferWorker, RolloutWorker
        from tensordict import TensorDict
        import time
        from omegaconf import OmegaConf

        logger = Tracking(project_name=self.config.trainer.project_name,
                          experiment_name=self.config.trainer.experiment_name,
                          default_backend=self.config.trainer.logger,
                          config=OmegaConf.to_container(self.config, resolve=True))

        self.global_steps = 0
        
        # load checkpoint before doing anything
        self._load_checkpoint()

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get('val_before_train', True):
            val_metrics = self._validate()
            pprint(f'Initial validation metrics: {val_metrics}')
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get('val_only', False):
                return

        # Create shared replay buffer as a Ray actor
        replay_buffer_size = self.config.actor_rollout_ref.rollout.replay_buffer_size
        replay_buffer_ref = SharedReplayBufferWorker.remote(capacity=replay_buffer_size)

        metrics_queue = Queue(maxsize=1)
        data_queue = Queue(maxsize=1)

        loader_thread = threading.Thread(
            target=load_data_onto_queue,
            args=(self.train_dataloader, data_queue),
            daemon=True # Set daemon=True so it doesn't block exit if main thread finishes
        )
        loader_thread.start()
        
        # Create rollout worker as a Ray actor
        rollout_worker_ref = RolloutWorker.remote(
            config=self.config,
            replay_buffer_worker_ref=replay_buffer_ref
        )
        
        # Initialize rollout worker with necessary components
        ray.get(rollout_worker_ref.setup.remote(
            data_queue_ref=data_queue,
            metrics_queue_ref=metrics_queue,
            rollout_wg=self.rollout_wg,
            rm_wg=self.rm_wg if self.use_rm else None,
            reward_fn=self.reward_fn,
            kl_ctrl=self.kl_ctrl
        ))

        # Start the inference worker in a separate process
        # Run in background to start generating data immediately
        rollout_worker_generation_task = rollout_worker_ref.run_generation_loop.remote()
        
        # We start from step 1
        self.global_steps += 1
        
        # Wait for buffer to have enough data
        buffer_size = ray.get(replay_buffer_ref.size.remote())
        while buffer_size < self.config.data.train_batch_size:
            print(f"Waiting for replay buffer to be filled... size {buffer_size}")
            time.sleep(5)
            buffer_size = ray.get(replay_buffer_ref.size.remote())

        while True:
            metrics = {}
            timing_raw = {}

            with _timer('step', timing_raw):
                # Sample a batch from the replay buffer
                batch_data = ray.get(replay_buffer_ref.sample.remote(
                    batch_size=self.config.data.train_batch_size, 
                    latest_batch_prob=self.config.actor_rollout_ref.actor.get("online_prob", 0.0)
                ))
                
                # Convert list of DataProto objects to a single DataProto
                tds = torch.cat([data.batch for data in batch_data], dim=0)
                batch = DataProto.from_single_dict(tds.to_dict())

                # Balance the number of valid tokens on each dp rank
                # Note that this breaks the order of data inside the batch
                self._balance_batch(batch, metrics=metrics)

                # Compute global_valid tokens
                batch.meta_info['global_token_num'] = torch.sum(batch.batch['attention_mask'], dim=-1).tolist()

                if self.use_reference_policy:
                    # Compute reference log_prob
                    with _timer('ref', timing_raw):
                        ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                        batch = batch.union(ref_log_prob)

                batch.meta_info['temperature'] = self.config.actor_rollout_ref.rollout.temperature
                batch.batch['old_log_probs'] = batch.batch['logprobs']

                # Update critic
                if self.use_critic:
                    critic_output = self.critic_wg.update_critic(batch)

                # Update actor
                if self.use_critic and self.config.trainer.critic_warmup <= self.global_steps:
                    actor_output = self.actor_wg.update_actor(batch)
                elif not self.use_critic:
                    actor_output = self.actor_wg.update_actor(batch)

                # Wait for results
                with _timer('update_actor', timing_raw):
                    actor_output = actor_output.get()
                    actor_output_metrics = reduce_metrics(actor_output.meta_info['metrics'])
                    metrics.update(actor_output_metrics)

                if self.use_critic:
                    with _timer('update_critic', timing_raw):
                        critic_output = critic_output.get()
                        critic_output_metrics = reduce_metrics(critic_output.meta_info['metrics'])
                        metrics.update(critic_output_metrics)

                # Save checkpoint if needed
                if self.config.trainer.save_freq > 0 and \
                        self.global_steps % self.config.trainer.save_freq == 0:
                    with _timer('save_checkpoint', timing_raw):
                        self._save_checkpoint()

                # Periodically synchronize parameters from actor to rollout workers
                if self.global_steps % self.config.actor_rollout_ref.rollout.update_interval == 0:
                    with _timer("weight_sync", timing_raw):
                        # Pause the rollout worker during weight sync
                        # print("Print pausing rollout worker (this will wait until it's done first)")
                        # rollout_worker_ref.pause.remote()
                        # print("Finished pausing, starting weight sync")
                        
                        # # Signal the replay buffer that weight syncing is starting
                        # ray.get(replay_buffer_ref.start_weight_sync.remote())
                        
                        # Perform weight synchronization
                        actor_ref = self.actor_wg.weight_sync()
                        rollout_ref = self.rollout_wg.weight_sync()
                        ray.get(actor_ref + rollout_ref)
                        # print("finished weight syncing, resuming rollout worker")
                        
                        # Signal the replay buffer that weight syncing is complete
                        # ray.get(replay_buffer_ref.end_weight_sync.remote())
                        
                        # Resume the rollout worker after weight sync
                        # ray.get(rollout_worker_ref.resume.remote())
                        # print("Finished weight sync and resumed rollout worker")

            # Run validation if needed
            if self.val_reward_fn is not None and self.config.trainer.test_freq > 0 and \
                self.global_steps % self.config.trainer.test_freq == 0:
                with _timer('val', timing_raw):
                    val_metrics: dict = self._validate()
                metrics.update(val_metrics)

            # Add replay buffer metrics
            buffer_stats = ray.get(replay_buffer_ref.get_stats.remote())
            metrics['replay_buffer/size'] = buffer_stats['size']
            metrics['replay_buffer/capacity'] = buffer_stats['capacity']
            metrics['replay_buffer/fill_ratio'] = buffer_stats['fill_ratio']
            
            # Add rollout worker metrics if available
            try:
                worker_stats = metrics_queue.get(timeout=1.0)
                metrics['batches_generated'] = worker_stats['data_generation_count']
                timing_raw['generate_rollout'] = worker_stats['generation_time']
            except:
                pass

            # Collect metrics
            metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
            metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))

            # Log all metrics
            logger.log(data=metrics, step=self.global_steps)

            self.global_steps += 1

            # Check if we've reached the total training steps
            if self.global_steps >= self.total_training_steps:
                # Pause the rollout worker if it's running
                is_active = ray.get(rollout_worker_ref.is_active.remote())
                if is_active:
                    # Make sure worker is not paused when stopping
                    ray.get(rollout_worker_ref.resume.remote())
                    # Then stop the worker
                    ray.get(rollout_worker_ref.stop.remote())
                
                # Perform final validation if needed
                if self.val_reward_fn is not None:
                    val_metrics = self._validate()
                    pprint(f'Final validation metrics: {val_metrics}')
                    logger.log(data=val_metrics, step=self.global_steps)
                
                print("Training completed successfully")
                return
