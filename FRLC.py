import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from collections import deque
import random
import argparse
import os
import csv
from datetime import datetime
from FLIM_env import FLIMEnv, NUM_DEVICES
import warnings
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
from coalition_game import CoalitionGame, DynamicCoalitionFormation
from differential_privacy import DifferentialPrivacy, LocalDifferentialPrivacy

warnings.filterwarnings('ignore')

if not os.path.exists('result'):
    os.makedirs('result')

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
result_filename = f'result/training_results_{timestamp}.csv'

with open(result_filename, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow([
        'Episode', 'Total Devices', 'Server Reward', 'ND Reward', 'RD Reward',
        'FD Reward', 'ED Reward', 'CD Reward', 'Avg Leak Prob', 'Recent Avg Leak',
        'Privacy Improvement', 'Recent Privacy Improvement', 'Defense Success Rate',
        'Recent Defense Rate', 'Coalition Stability', 'Recent Stability',
        'Action Quality', 'Server Utility', 'ND Utility', 'RD Utility',
        'FD Utility', 'ED Utility', 'CD Utility', 'Avg Payment',
        'Communication Cost', 'Computation Cost', 'Total Cost', 'Avg Time',
        'DP Net Budget Used', 'DP Epsilon', 'DP Usage Percentage',
        'DP Released Budget', 'DP Release Rate',
        'Num Coalitions', 'Avg Coalition Size', 'Stability Trend', 'Convergence Progress',
        'Coalition Quality', 'Coalition Efficiency', 'Fairness Gini'
    ])

DEFAULT_NUM_DEVICES = NUM_DEVICES


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


set_seed(42)


class OptimizedActorCritic(nn.Module):

    def __init__(self, input_dim, action_dims, coalition_dim=5, hidden_size=256):
        super(OptimizedActorCritic, self).__init__()
        self.action_dims = action_dims

        self.coalition_embedding = nn.Sequential(
            nn.Linear(coalition_dim, 16),
            nn.ReLU()
        )

        self.shared = nn.Sequential(
            nn.Linear(input_dim + 16, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )

        self.actor_heads = nn.ModuleList()
        for action_dim in action_dims:
            self.actor_heads.append(
                nn.Sequential(
                    nn.Linear(hidden_size, action_dim),
                    nn.Softmax(dim=-1)
                )
            )

        self.critic = nn.Linear(hidden_size, 1)
        self.coalition_quality = nn.Linear(hidden_size, 1)

    def forward(self, state, coalition_info=None):
        if isinstance(state, list):
            state = state[0] if len(state) > 0 else torch.zeros(1, 4)

        if coalition_info is None:
            coalition_info = torch.zeros(state.shape[0], 5)

        coalition_features = self.coalition_embedding(coalition_info)

        combined = torch.cat([state, coalition_features], dim=-1)
        shared_features = self.shared(combined)

        action_probs = []
        for actor_head in self.actor_heads:
            action_probs.append(actor_head(shared_features))

        value = self.critic(shared_features)
        quality = torch.sigmoid(self.coalition_quality(shared_features))

        return action_probs, value, quality

    def act(self, state, coalition_info=None, deterministic=False, temperature=1.0):
        action_probs, value, quality = self.forward(state, coalition_info)

        actions = []
        log_probs = []

        for probs in action_probs:

            adjusted_temp = temperature * (1.0 + 0.1 * quality.item())

            if adjusted_temp != 1.0:
                probs = torch.pow(probs, 1.0 / adjusted_temp)
                probs = probs / probs.sum()

            if deterministic:
                action = torch.argmax(probs, dim=-1)
                log_prob = torch.log(probs.gather(-1, action.unsqueeze(-1)).squeeze(-1))
            else:
                dist = Categorical(probs)
                action = dist.sample()
                log_prob = dist.log_prob(action)

            actions.append(action.item())
            log_probs.append(log_prob)

        return actions, log_probs, value, quality


class OptimizedPPOAgent:

    def __init__(self, input_dim, action_dims, coalition_dim=5, lr=3e-4, gamma=0.99,
                 eps_clip=0.2, k_epochs=4, name="agent"):
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs
        self.name = name
        self.action_dims = action_dims

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.policy = OptimizedActorCritic(input_dim, action_dims, coalition_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)

        self.policy_old = OptimizedActorCritic(input_dim, action_dims, coalition_dim).to(self.device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss()

        self.coalition_weight = 0.02

    def get_coalition_info(self, device_id, coalition_game):
        coalition_info = np.zeros(5)

        if coalition_game.coalitions:
            for idx, coalition in enumerate(coalition_game.coalitions):
                if device_id in coalition:
                    coalition_info[0] = len(coalition) / coalition_game.num_devices
                    coalition_info[1] = idx / max(len(coalition_game.coalitions), 1)

                    type_counts = set()
                    for member in coalition:
                        device_type = coalition_game.device_type_mapping.get(member, 'ND')
                        type_counts.add(device_type)
                    coalition_info[2] = len(type_counts) / 5

                    if device_id in coalition_game.shapley_values:
                        shapley_val = coalition_game.shapley_values[device_id]
                        max_shapley = max(abs(v) for v in coalition_game.shapley_values.values())
                        coalition_info[3] = shapley_val / max(max_shapley, 1)

                    break

        if coalition_game.stability_history:
            coalition_info[4] = coalition_game.stability_history[-1]

        return coalition_info

    def select_action(self, state, coalition_info=None, episode=0, max_episodes=500):
        with torch.no_grad():
            if isinstance(state, list):
                state = state[0] if len(state) > 0 else np.zeros(4)

            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)

            if coalition_info is not None:
                coalition_info = torch.FloatTensor(coalition_info).unsqueeze(0).to(self.device)

            if episode < max_episodes * 0.3:
                deterministic = False
                temperature = 1.0
            else:
                progress = (episode - max_episodes * 0.3) / (max_episodes * 0.7)
                deterministic = random.random() < progress * 0.8
                temperature = max(0.5, 1.0 - progress * 0.5)

            actions, log_probs, state_value, quality = self.policy_old.act(
                state, coalition_info, deterministic, temperature
            )

        return actions, log_probs, state_value, quality

    def update(self, memory, coalition_memory, episode=0, max_episodes=500):
        if len(memory.states) == 0:
            return

        current_eps_clip = self.eps_clip * max(0.5, 1.0 - episode / max_episodes * 0.3)

        coalition_weight = self.coalition_weight * min(0.5, episode / (max_episodes * 0.8))

        old_states = torch.FloatTensor(np.array(memory.states)).to(self.device)
        old_actions = torch.LongTensor(np.array(memory.actions)).to(self.device)
        old_coalition_info = torch.FloatTensor(np.array(coalition_memory.coalition_info)).to(self.device)
        old_quality = torch.FloatTensor(np.array(coalition_memory.quality)).to(self.device)

        old_logprobs_list = []
        for log_prob_set in memory.logprobs:
            if isinstance(log_prob_set, list):
                old_logprobs_list.append(torch.stack(log_prob_set).sum())
            else:
                old_logprobs_list.append(log_prob_set)
        old_logprobs = torch.stack(old_logprobs_list).to(self.device)

        rewards = []
        discounted_reward = 0
        for reward in reversed(memory.rewards):
            discounted_reward = reward + self.gamma * discounted_reward
            rewards.insert(0, discounted_reward)

        rewards = torch.FloatTensor(rewards).to(self.device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

        for _ in range(self.k_epochs):
            action_probs, state_values, predicted_quality = self.policy(old_states, old_coalition_info)

            logprobs = []
            for i in range(len(old_states)):
                log_prob_sum = 0
                for j, probs in enumerate(action_probs):
                    if j < old_actions.shape[1]:
                        dist = Categorical(probs[i])
                        action = old_actions[i, j]
                        log_prob_sum += dist.log_prob(action)
                logprobs.append(log_prob_sum)

            if len(logprobs) > 0:
                logprobs = torch.stack(logprobs)

                ratios = torch.exp(logprobs - old_logprobs.detach())
                advantages = rewards - state_values.squeeze().detach()

                surr1 = ratios * advantages
                surr2 = torch.clamp(ratios, 1 - current_eps_clip, 1 + current_eps_clip) * advantages

                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = 0.5 * self.MseLoss(state_values.squeeze(), rewards)
                quality_loss = self.MseLoss(predicted_quality.squeeze(), old_quality)

                loss = policy_loss + value_loss + coalition_weight * quality_loss

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
                self.optimizer.step()

        self.policy_old.load_state_dict(self.policy.state_dict())
        memory.clear()
        coalition_memory.clear()


class Memory:
    def __init__(self):
        self.states = []
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []

    def clear(self):
        self.states = []
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []


class CoalitionMemory:

    def __init__(self):
        self.coalition_info = []
        self.quality = []
        self.coalition_rewards = []

    def clear(self):
        self.coalition_info = []
        self.quality = []
        self.coalition_rewards = []


def calculate_coalition_reward(device_id, coalition_game, base_reward, leak_improvement, stability_rate):

    coalition_bonus = 0

    if coalition_game.coalitions and coalition_game.episode_count > 30:
        device_coalition = None
        for coalition in coalition_game.coalitions:
            if device_id in coalition:
                device_coalition = coalition
                break

        if device_coalition:

            coalition_size = len(device_coalition)
            optimal_size = max(4, min(6, coalition_game.num_devices // 5))

            if abs(coalition_size - optimal_size) <= 1:
                size_bonus = 3
            else:
                size_bonus = 0

            stability_bonus = stability_rate * 5

            fairness_bonus = 0
            if device_id in coalition_game.shapley_values:
                shapley_value = coalition_game.shapley_values[device_id]
                if shapley_value > 0:
                    fairness_bonus = min(2, shapley_value / 5)

            coalition_bonus = size_bonus + stability_bonus + fairness_bonus

    if coalition_game.episode_count > 50:
        progress_weight = min(0.15, (coalition_game.episode_count - 50) / 300 * 0.15)
    else:
        progress_weight = 0

    return base_reward + coalition_bonus * progress_weight


def train_multi_agent(num_devices=DEFAULT_NUM_DEVICES, max_episodes=600):

    env = FLIMEnv(num_devices=num_devices)

    dp_module = DifferentialPrivacy(
        epsilon=3.0,
        delta=1e-5,
        clip_norm=1.0,
        noise_multiplier=0.1
    )

    local_dp = LocalDifferentialPrivacy(epsilon=1.0, mechanism='laplace')

    coalition_game = CoalitionGame(
        num_devices=num_devices,
        device_types=env.device_types,
        device_counts=env.device_counts
    )

    dynamic_coalition = DynamicCoalitionFormation(coalition_game)

    dp_statistics = []
    coalition_statistics = []

    print(f"=" * 50)
    print(f"Starting Optimized Multi-Agent Federated Learning")
    print(f"Total devices: {num_devices}")
    print(f"Device distribution: {env.device_counts}")
    print(f"With Differential Privacy: ε={dp_module.epsilon}, δ={dp_module.delta}")
    print(f"Privacy Release Mechanism: Enabled (Release Rate={dp_module.release_rate})")
    print(f"Coalition Strategy: Optimized for convergence with stability")
    print(f"=" * 50)

    agents = {}
    memories = {}
    coalition_memories = {}

    agents['server'] = OptimizedPPOAgent(
        input_dim=3, action_dims=[10, 10, 10], coalition_dim=5, lr=3e-4, name="Server"
    )
    memories['server'] = Memory()
    coalition_memories['server'] = CoalitionMemory()

    device_types = ['ND', 'RD', 'FD', 'ED', 'CD']
    for device_type in device_types:
        if env.device_counts.get(device_type, 0) > 0:
            if device_type == 'RD':
                agents[device_type] = OptimizedPPOAgent(
                    input_dim=4, action_dims=[10, 10], coalition_dim=5, lr=3e-4, name=device_type
                )
            else:
                agents[device_type] = OptimizedPPOAgent(
                    input_dim=4, action_dims=[10, 10, 10], coalition_dim=5, lr=3e-4, name=device_type
                )
            memories[device_type] = Memory()
            coalition_memories[device_type] = CoalitionMemory()

    max_timesteps = 16
    update_timestep = 32

    rewards_by_type = {agent_name: [] for agent_name in agents.keys()}
    utilities_by_type = {agent_name: [] for agent_name in agents.keys()}

    leak_probs = []
    payments = []
    times = []
    total_costs = []
    privacy_improvements = []
    communication_costs = []
    computation_costs = []
    defense_success_rates = []
    coalition_stability_rates = []
    action_quality_scores = []

    time_step = 0

    for episode in range(max_episodes):
        if episode == 0:
            should_update_coalition = True
        elif episode < 50:
            should_update_coalition = (episode % 8 == 0)
        elif episode < 150:
            should_update_coalition = (episode % 15 == 0)
        else:
            should_update_coalition = (episode % 25 == 0)

        if should_update_coalition:
            privacy_context = {
                'leak_prob': env.leak_prob if episode > 0 else 0.8,
                'episode': episode
            }

            device_contributions = {}
            for i in range(num_devices):
                device_type = coalition_game.device_type_mapping.get(i, 'ND')
                base_contrib = {'ND': 8, 'RD': 6, 'FD': 5, 'ED': 7, 'CD': 9}.get(device_type, 5)

                device_contributions[i] = base_contrib + np.random.uniform(-0.2, 0.2)

            shapley_values = coalition_game.compute_shapley_values(device_contributions, privacy_context)

            if episode < 100 and dynamic_coalition.should_reorganize(privacy_context):
                print(f"  Reorganizing coalitions at episode {episode}")

            coalitions = coalition_game.form_coalitions(
                device_contributions,
                privacy_context=privacy_context,
                min_coalition_size=2,
                max_coalition_size=None
            )

            coalition_stats = coalition_game.get_coalition_statistics()
            coalition_statistics.append(coalition_stats)

        server_obs, nd_obs, rd_obs, fd_obs = env.reset()

        observations = {
            'server': server_obs,
            'ND': nd_obs,
            'RD': rd_obs,
            'FD': fd_obs,
            'ED': nd_obs,
            'CD': nd_obs
        }

        episode_rewards_dict = {agent_name: 0 for agent_name in agents.keys()}
        episode_leak = []
        episode_payment = []
        episode_time = []
        episode_comm_time = []
        episode_comp_time = []
        episode_defense_rates = []
        initial_leak = None
        episode_action_quality = []

        device_coalition_info = {}
        device_id_counter = 0
        for device_type in device_types:
            count = env.device_counts.get(device_type, 0)
            for i in range(count):
                coalition_info = agents[device_type].get_coalition_info(device_id_counter, coalition_game)
                device_coalition_info[device_id_counter] = coalition_info
                device_id_counter += 1
        current_stability = coalition_game.stability_history[-1] if coalition_game.stability_history else 0.5

        for t in range(max_timesteps):
            time_step += 1

            if t == 0:
                initial_leak = env.leak_prob

            actions = {}
            log_probs = {}
            values = {}
            qualities = {}

            for agent_name, agent in agents.items():
                if agent_name == 'server':
                    obs = observations['server']

                    global_coalition_info = np.zeros(5)
                    if coalition_game.coalitions:
                        global_coalition_info[0] = len(coalition_game.coalitions) / num_devices
                        global_coalition_info[1] = np.mean([len(c) for c in coalition_game.coalitions]) / num_devices
                        global_coalition_info[2] = current_stability
                        global_coalition_info[3] = coalition_game.episode_count / max_episodes
                        global_coalition_info[4] = env.defense_success_rate if hasattr(env,
                                                                                       'defense_success_rate') else 0.5

                    global_coalition_info *= 0.5
                    coalition_info = global_coalition_info
                else:
                    obs = observations.get(agent_name, nd_obs)

                    coalition_info = device_coalition_info.get(0, np.zeros(5)) * 0.5

                action, log_prob, value, quality = agent.select_action(
                    obs, coalition_info, episode, max_episodes
                )
                actions[agent_name] = action
                log_probs[agent_name] = log_prob
                values[agent_name] = value
                qualities[agent_name] = quality
                episode_action_quality.append(quality.item())

            combined_action = {
                'server': actions['server'],
                'ND': actions.get('ND', [0, 0, 0]),
                'RD': actions.get('RD', [0, 0]),
                'FD': actions.get('FD', [0, 0, 0]),
                'ED': actions.get('ED', [0, 0, 0]),
                'CD': actions.get('CD', [0, 0, 0])
            }

            result = env.step(combined_action)

            if len(result) == 8:
                server_obs_next, nd_obs_next, rd_obs_next, fd_obs_next, server_rew, client_rew, done, info = result
            else:
                print(f"Warning: Unexpected step return format")
                break

            observations_next = {
                'server': server_obs_next,
                'ND': nd_obs_next,
                'RD': rd_obs_next,
                'FD': fd_obs_next,
                'ED': nd_obs_next,
                'CD': nd_obs_next
            }

            server_reward = server_rew[0] if isinstance(server_rew, list) else server_rew
            nd_reward = client_rew[0]
            rd_reward = client_rew[1]
            fd_reward = client_rew[2]

            all_device_rewards = info[0].get('all_device_rewards', {})

            leak_improvement = initial_leak - env.leak_prob if initial_leak else 0
            rewards_dict = {}
            device_id = 0
            for device_type in device_types:
                count = env.device_counts.get(device_type, 0)
                if count > 0:
                    if device_type == 'ND':
                        base_reward = nd_reward
                    elif device_type == 'RD':
                        base_reward = rd_reward
                    elif device_type == 'FD':
                        base_reward = fd_reward
                    else:
                        base_reward = all_device_rewards.get(device_type, 0)

                    coalition_reward = calculate_coalition_reward(
                        device_id, coalition_game, base_reward, leak_improvement, current_stability
                    )
                    rewards_dict[device_type] = coalition_reward
                    device_id += count

            stability_bonus = current_stability * 5
            global_coalition_bonus = 0
            if coalition_game.coalitions and coalition_game.episode_count > 50:

                avg_coalition_size = np.mean([len(c) for c in coalition_game.coalitions])
                optimal_size = max(4, min(6, num_devices // 5))
                if abs(avg_coalition_size - optimal_size) <= 1:
                    global_coalition_bonus = 3

            rewards_dict['server'] = server_reward + stability_bonus + global_coalition_bonus

            is_done = done[0] if isinstance(done, list) else done

            for agent_name in agents.keys():
                if agent_name in memories:
                    memory = memories[agent_name]
                    coalition_memory = coalition_memories[agent_name]

                    obs = observations[agent_name]
                    obs = obs[0] if isinstance(obs, list) else obs

                    memory.states.append(obs)
                    memory.actions.append(actions[agent_name])
                    memory.logprobs.append(log_probs[agent_name])
                    memory.rewards.append(rewards_dict.get(agent_name, 0))
                    memory.is_terminals.append(is_done)

                    if agent_name == 'server':
                        coalition_info = global_coalition_info
                    else:
                        coalition_info = device_coalition_info.get(0, np.zeros(5))

                    coalition_memory.coalition_info.append(coalition_info)
                    coalition_memory.quality.append(qualities[agent_name].item())
                    coalition_memory.coalition_rewards.append(rewards_dict.get(agent_name, 0))

            for agent_name in agents.keys():
                episode_rewards_dict[agent_name] += rewards_dict.get(agent_name, 0)

            episode_leak.append(info[0]["leak"])
            episode_payment.append(info[0].get("payment", 0))
            episode_time.append(info[0].get("global_time", 0))
            episode_comm_time.append(info[0].get("communication_time", 0))
            episode_comp_time.append(info[0].get("computation_time", 0))
            episode_defense_rates.append(info[0].get("defense_success_rate", 0))

            observations = observations_next

            if time_step % update_timestep == 0:
                for agent_name, agent in agents.items():
                    if agent_name in memories:
                        agent.update(memories[agent_name], coalition_memories[agent_name],
                                     episode, max_episodes)

            if is_done:
                break

        if episode >= 20 and episode % 10 == 0:
            try:
                for agent_name, agent in agents.items():
                    if hasattr(agent, 'policy'):
                        model_params = {}
                        for name, param in agent.policy.named_parameters():
                            if param.requires_grad:
                                model_params[name] = param

                        if model_params:
                            dp_params = dp_module.apply_differential_privacy(model_params)

                            for name, param in agent.policy.named_parameters():
                                if name in dp_params:
                                    param.data = dp_params[name].data

                dp_stats = dp_module.get_statistics()
                dp_statistics.append(dp_stats)

            except Exception as e:
                print(f"Warning: DP application failed at episode {episode}: {str(e)}")

        if coalition_game.stability_history:
            coalition_stability_rates.append(coalition_game.stability_history[-1])

        final_leak = np.mean(episode_leak) if episode_leak else 0
        privacy_improvement = initial_leak - final_leak if initial_leak else 0
        avg_payment = np.mean(episode_payment) if episode_payment else 0
        avg_time = np.mean(episode_time) if episode_time else 0
        avg_comm_time = np.mean(episode_comm_time) if episode_comm_time else 0
        avg_comp_time = np.mean(episode_comp_time) if episode_comp_time else 0
        avg_defense_rate = np.mean(episode_defense_rates) if episode_defense_rates else 0
        total_cost = avg_payment + avg_time
        avg_action_quality = np.mean(episode_action_quality) if episode_action_quality else 0
        action_quality_scores.append(avg_action_quality)

        for agent_name in agents.keys():
            rewards_by_type[agent_name].append(episode_rewards_dict[agent_name])

            if agent_name == 'server':
                utility = episode_rewards_dict[agent_name] - total_cost * 0.3
            else:
                utility = episode_rewards_dict[agent_name] - avg_payment / max(num_devices, 1)
            utilities_by_type[agent_name].append(utility)

        leak_probs.append(final_leak)
        payments.append(avg_payment)
        times.append(avg_time)
        total_costs.append(total_cost)
        privacy_improvements.append(privacy_improvement)
        communication_costs.append(avg_comm_time)
        computation_costs.append(avg_comp_time)
        defense_success_rates.append(avg_defense_rate)

        recent_leak = np.mean(leak_probs[-10:]) if len(leak_probs) >= 10 else np.mean(leak_probs)
        recent_improvement = np.mean(privacy_improvements[-10:]) if len(
            privacy_improvements) >= 10 else privacy_improvement
        recent_defense_rate = np.mean(defense_success_rates[-10:]) if len(
            defense_success_rates) >= 10 else avg_defense_rate
        recent_stability = np.mean(coalition_stability_rates[-10:]) if len(
            coalition_stability_rates) >= 10 else current_stability
        recent_action_quality = np.mean(action_quality_scores[-10:]) if len(
            action_quality_scores) >= 10 else avg_action_quality

        print(f"\nEpisode {episode}/{max_episodes}")
        print(f"  Total devices: {num_devices}")

        for agent_name in ['server', 'ND', 'RD', 'FD', 'ED', 'CD']:
            if agent_name in agents:
                print(f"  {agent_name} Reward: {episode_rewards_dict[agent_name]:.2f}")

        print(f"  Avg Leak Prob: {final_leak:.4f} (Recent avg: {recent_leak:.4f})")
        print(f"  Privacy Improvement: {privacy_improvement:.4f} (Recent avg: {recent_improvement:.4f})")
        print(f"  Defense Success Rate: {avg_defense_rate:.3f} (Recent avg: {recent_defense_rate:.3f})")
        print(f"  Action Quality: {avg_action_quality:.3f}")
        print(f"  Server Utility: {utilities_by_type['server'][-1]:.2f}")

        device_utils = []
        for device_type in ['ND', 'RD', 'FD', 'ED', 'CD']:
            if device_type in utilities_by_type:
                device_utils.append(f"{device_type}={utilities_by_type[device_type][-1]:.2f}")
        print(f"  Device Utilities: {', '.join(device_utils)}")

        print(f"  Avg Payment: {avg_payment:.2f}")
        print(f"  Communication Cost: {avg_comm_time:.2f}")
        print(f"  Computation Cost: {avg_comp_time:.2f}")
        print(f"  Total Cost: {total_cost:.2f}")
        print(f"  Avg Time: {avg_time:.2f}")

        dp_net_budget = 0.0
        dp_released_budget = 0.0
        dp_epsilon = dp_module.epsilon
        dp_usage = 0.0
        dp_release_rate = dp_module.release_rate
        if dp_statistics:
            latest_dp = dp_statistics[-1]
            dp_net_budget = latest_dp['net_privacy_spent']
            dp_released_budget = latest_dp['total_privacy_released']
            dp_usage = latest_dp['privacy_budget_usage'] * 100
            print(f"  DP Net Budget Used: {dp_net_budget:.3f}/{dp_epsilon:.1f} "
                  f"({dp_usage:.1f}%)")
            print(f"  DP Released Budget: {dp_released_budget:.3f} "
                  f"(Release Rate: {dp_release_rate:.2%})")
            print(f"  DP Remaining Budget: {latest_dp['remaining_budget']:.3f}")

        num_coalitions = 0
        avg_coalition_size = 0.0
        stability_trend = "N/A"
        convergence_progress = 0.0
        coalition_quality = 0.0
        coalition_efficiency = 0.0
        fairness_gini = 0.0

        if coalition_statistics:
            latest_coalition = coalition_statistics[-1]
            num_coalitions = latest_coalition['num_coalitions']
            avg_coalition_size = latest_coalition['average_coalition_size']
            stability_trend = latest_coalition.get('stability_trend', 'stable')
            convergence_progress = latest_coalition.get('convergence_progress', 0)

            coalition_quality = latest_coalition.get('coalition_quality_score', 0.0)
            coalition_efficiency = latest_coalition.get('coalition_efficiency', 0.0)

            shapley_fairness = latest_coalition.get('shapley_fairness', {})
            fairness_gini = shapley_fairness.get('gini_coefficient', 0.0) if shapley_fairness else 0.0

            print(f"  Coalitions: {num_coalitions}, "
                  f"Avg Size: {avg_coalition_size:.1f}")
            print(f"  Stability Trend: {stability_trend}, "
                  f"Convergence: {convergence_progress * 100:.1f}%")
            print(f"  Coalition Quality: {coalition_quality:.3f}, "
                  f"Efficiency: {coalition_efficiency:.3f}")
            print(f"  Fairness (Gini): {fairness_gini:.3f}")

        print("-" * 50)

        with open(result_filename, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                episode, num_devices,
                episode_rewards_dict.get('server', 0),
                episode_rewards_dict.get('ND', 0),
                episode_rewards_dict.get('RD', 0),
                episode_rewards_dict.get('FD', 0),
                episode_rewards_dict.get('ED', 0),
                episode_rewards_dict.get('CD', 0),
                final_leak, recent_leak,
                privacy_improvement, recent_improvement,
                avg_defense_rate, recent_defense_rate,
                current_stability, recent_stability,
                avg_action_quality,
                utilities_by_type.get('server', [0])[-1],
                utilities_by_type.get('ND', [0])[-1],
                utilities_by_type.get('RD', [0])[-1],
                utilities_by_type.get('FD', [0])[-1],
                utilities_by_type.get('ED', [0])[-1],
                utilities_by_type.get('CD', [0])[-1],
                avg_payment, avg_comm_time, avg_comp_time,
                total_cost, avg_time,
                dp_net_budget, dp_epsilon, dp_usage,
                dp_released_budget, dp_release_rate,
                num_coalitions, avg_coalition_size,
                stability_trend, convergence_progress,
                coalition_quality, coalition_efficiency, fairness_gini
            ])

    print("\n" + "=" * 50)
    print("TRAINING COMPLETED - OPTIMIZED RESULTS WITH DIFFERENTIAL PRIVACY")
    print("=" * 50)

    final_episodes = 50

    if len(leak_probs) >= final_episodes:
        final_leak = np.mean(leak_probs[-final_episodes:])
        final_improvement = np.mean(privacy_improvements[-final_episodes:])
        final_defense = np.mean(defense_success_rates[-final_episodes:])
        final_payment = np.mean(payments[-final_episodes:])
        final_stability = np.mean(coalition_stability_rates[-final_episodes:]) if coalition_stability_rates else 0

        print(f"Final {final_episodes} Episodes Average:")
        print(f"  Privacy Leak: {final_leak:.4f}")
        print(f"  Privacy Improvement: {final_improvement:.4f}")
        print(f"  Defense Success Rate: {final_defense:.3f}")
        print(f"  Average Payment: {final_payment:.2f}")

        if dp_statistics:
            final_dp = dp_statistics[-1]
            print(f"\nFinal Differential Privacy Status:")
            print(f"  Total Privacy Budget: {dp_module.epsilon:.1f}")
            print(f"  Net Privacy Spent: {final_dp['net_privacy_spent']:.3f}")
            print(f"  Total Released: {final_dp['total_privacy_released']:.3f}")
            print(f"  Remaining Budget: {final_dp['remaining_budget']:.3f}")
            print(f"  Release Rate: {final_dp['release_rate']:.2%}")
            print(f"  Average Noise Level: {final_dp['average_noise_level']:.4f}")

        success_01 = sum(1 for x in leak_probs[-final_episodes:] if x < 0.1) / final_episodes
        success_02 = sum(1 for x in leak_probs[-final_episodes:] if x < 0.2) / final_episodes
        success_03 = sum(1 for x in leak_probs[-final_episodes:] if x < 0.3) / final_episodes

        print(f"\nSuccess Rates (last {final_episodes} episodes):")
        print(f"  Leak < 0.1: {success_01:.1%}")
        print(f"  Leak < 0.2: {success_02:.1%}")
        print(f"  Leak < 0.3: {success_03:.1%}")

        print(f"\nConvergence Check:")
        if final_leak < 0.1 and final_defense > 0.8 and final_stability > 0.9:
            print(f"  ✓ CONVERGED: Privacy leak < 0.1, Defense > 0.8, Stability > 0.9")
        elif final_leak < 0.2 and final_defense > 0.7 and final_stability > 0.8:
            print(f"  ◉ PARTIALLY CONVERGED: Good progress on all metrics")
        else:
            print(f"  ✗ Not fully converged yet")

    if coalition_stability_rates:
        early_stability = np.mean(coalition_stability_rates[:50]) if len(coalition_stability_rates) >= 50 else np.mean(
            coalition_stability_rates[:len(coalition_stability_rates) // 3])
        late_stability = np.mean(coalition_stability_rates[-50:]) if len(coalition_stability_rates) >= 50 else np.mean(
            coalition_stability_rates[-len(coalition_stability_rates) // 3:])

        print(f"\nCoalition Stability Analysis:")
        print(f"  Early Training Stability: {early_stability:.3f}")
        print(f"  Late Training Stability: {late_stability:.3f}")
        print(
            f"  Improvement: {(late_stability - early_stability):.3f} ({(late_stability / early_stability - 1) * 100:.1f}%)")

    print("=" * 50)

    return agents, {
        'rewards_by_type': rewards_by_type,
        'utilities_by_type': utilities_by_type,
        'leak_probs': leak_probs,
        'payments': payments,
        'times': times,
        'total_costs': total_costs,
        'privacy_improvements': privacy_improvements,
        'communication_costs': communication_costs,
        'computation_costs': computation_costs,
        'defense_success_rates': defense_success_rates,
        'coalition_stability_rates': coalition_stability_rates,
        'action_quality_scores': action_quality_scores,
        'dp_statistics': dp_statistics,
        'coalition_statistics': coalition_statistics
    }


def test_agents(agents, num_devices=DEFAULT_NUM_DEVICES, num_episodes=10):
    env = FLIMEnv(num_devices=num_devices)

    coalition_game = CoalitionGame(
        num_devices=num_devices,
        device_types=env.device_types,
        device_counts=env.device_counts
    )

    test_results = {
        'rewards_by_type': {agent_name: [] for agent_name in agents.keys()},
        'utilities_by_type': {agent_name: [] for agent_name in agents.keys()},
        'leak_probs': [],
        'payments': [],
        'times': [],
        'privacy_improvements': [],
        'communication_costs': [],
        'computation_costs': [],
        'defense_success_rates': [],
        'coalition_stability_rates': [],
        'action_quality_scores': []
    }

    test_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    test_filename = f'result/test_results_{test_timestamp}.csv'

    with open(test_filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'Episode', 'Total Devices', 'Server Reward', 'ND Reward', 'RD Reward',
            'FD Reward', 'ED Reward', 'CD Reward', 'Leak Probability',
            'Privacy Improvement', 'Defense Success Rate', 'Coalition Stability',
            'Action Quality', 'Server Utility', 'ND Utility', 'RD Utility',
            'FD Utility', 'ED Utility', 'CD Utility', 'Avg Payment',
            'Communication Cost', 'Computation Cost', 'Total Cost', 'Avg Time',
            'Coalition Quality', 'Coalition Efficiency', 'Fairness Gini'
        ])

    for episode in range(num_episodes):
        device_contributions = {}
        for i in range(num_devices):
            device_type = coalition_game.device_type_mapping.get(i, 'ND')
            base_contrib = {'ND': 8, 'RD': 6, 'FD': 5, 'ED': 7, 'CD': 9}.get(device_type, 5)
            device_contributions[i] = base_contrib + np.random.uniform(-1, 1)

        coalition_game.compute_shapley_values(device_contributions)
        coalition_game.form_coalitions(device_contributions)

        server_obs, nd_obs, rd_obs, fd_obs = env.reset()

        observations = {
            'server': server_obs,
            'ND': nd_obs,
            'RD': rd_obs,
            'FD': fd_obs,
            'ED': nd_obs,
            'CD': nd_obs
        }

        episode_rewards_dict = {agent_name: 0 for agent_name in agents.keys()}
        episode_leak = []
        episode_payment = []
        episode_time = []
        episode_comm_time = []
        episode_comp_time = []
        episode_defense_rates = []
        initial_leak = env.leak_prob
        episode_action_quality = []

        done = False
        while not done:
            actions = {}
            with torch.no_grad():
                for agent_name, agent in agents.items():
                    if agent_name == 'server':
                        obs = observations['server']
                        coalition_info = np.zeros(5)
                        if coalition_game.coalitions:
                            coalition_info[0] = len(coalition_game.coalitions) / num_devices
                            coalition_info[1] = np.mean([len(c) for c in coalition_game.coalitions]) / num_devices
                            if coalition_game.stability_history:
                                coalition_info[2] = coalition_game.stability_history[-1]
                    else:
                        obs = observations.get(agent_name, nd_obs)
                        coalition_info = agent.get_coalition_info(0, coalition_game)

                    action, _, _, quality = agent.select_action(obs, coalition_info, num_episodes, num_episodes)
                    actions[agent_name] = action
                    episode_action_quality.append(quality.item())

            combined_action = {
                'server': actions['server'],
                'ND': actions.get('ND', [0, 0, 0]),
                'RD': actions.get('RD', [0, 0]),
                'FD': actions.get('FD', [0, 0, 0]),
                'ED': actions.get('ED', [0, 0, 0]),
                'CD': actions.get('CD', [0, 0, 0])
            }

            result = env.step(combined_action)

            if len(result) == 8:
                server_obs_next, nd_obs_next, rd_obs_next, fd_obs_next, server_rew, client_rew, done_list, info = result
            else:
                break

            server_reward = server_rew[0] if isinstance(server_rew, list) else server_rew
            nd_reward = client_rew[0]
            rd_reward = client_rew[1]
            fd_reward = client_rew[2]

            all_device_rewards = info[0].get('all_device_rewards', {})

            rewards_dict = {
                'server': server_reward,
                'ND': nd_reward,
                'RD': rd_reward,
                'FD': fd_reward,
                'ED': all_device_rewards.get('ED', 0),
                'CD': all_device_rewards.get('CD', 0)
            }

            done = done_list[0] if isinstance(done_list, list) else done_list

            for agent_name in agents.keys():
                episode_rewards_dict[agent_name] += rewards_dict.get(agent_name, 0)

            episode_leak.append(info[0]["leak"])
            episode_payment.append(info[0].get("payment", 0))
            episode_time.append(info[0].get("global_time", 0))
            episode_comm_time.append(info[0].get("communication_time", 0))
            episode_comp_time.append(info[0].get("computation_time", 0))
            episode_defense_rates.append(info[0].get("defense_success_rate", 0))

            observations = {
                'server': server_obs_next,
                'ND': nd_obs_next,
                'RD': rd_obs_next,
                'FD': fd_obs_next,
                'ED': nd_obs_next,
                'CD': nd_obs_next
            }

        final_leak = np.mean(episode_leak) if episode_leak else 0
        privacy_improvement = initial_leak - final_leak
        avg_payment = np.mean(episode_payment) if episode_payment else 0
        avg_time = np.mean(episode_time) if episode_time else 0
        avg_comm_time = np.mean(episode_comm_time) if episode_comm_time else 0
        avg_comp_time = np.mean(episode_comp_time) if episode_comp_time else 0
        avg_defense_rate = np.mean(episode_defense_rates) if episode_defense_rates else 0
        total_cost = avg_payment + avg_time
        avg_action_quality = np.mean(episode_action_quality) if episode_action_quality else 0

        for agent_name in agents.keys():
            test_results['rewards_by_type'][agent_name].append(episode_rewards_dict[agent_name])

            if agent_name == 'server':
                utility = episode_rewards_dict[agent_name] - total_cost * 0.3
            else:
                utility = episode_rewards_dict[agent_name] - avg_payment / max(num_devices, 1)
            test_results['utilities_by_type'][agent_name].append(utility)

        test_results['leak_probs'].append(final_leak)
        test_results['payments'].append(avg_payment)
        test_results['times'].append(avg_time)
        test_results['privacy_improvements'].append(privacy_improvement)
        test_results['communication_costs'].append(avg_comm_time)
        test_results['computation_costs'].append(avg_comp_time)
        test_results['defense_success_rates'].append(avg_defense_rate)
        test_results['action_quality_scores'].append(avg_action_quality)

        if coalition_game.stability_history:
            test_results['coalition_stability_rates'].append(coalition_game.stability_history[-1])
        else:
            test_results['coalition_stability_rates'].append(0)

        with open(test_filename, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                episode, num_devices,
                episode_rewards_dict.get('server', 0),
                episode_rewards_dict.get('ND', 0),
                episode_rewards_dict.get('RD', 0),
                episode_rewards_dict.get('FD', 0),
                episode_rewards_dict.get('ED', 0),
                episode_rewards_dict.get('CD', 0),
                final_leak, privacy_improvement, avg_defense_rate,
                test_results['coalition_stability_rates'][-1], avg_action_quality,
                test_results['utilities_by_type'].get('server', [0])[-1],
                test_results['utilities_by_type'].get('ND', [0])[-1],
                test_results['utilities_by_type'].get('RD', [0])[-1],
                test_results['utilities_by_type'].get('FD', [0])[-1],
                test_results['utilities_by_type'].get('ED', [0])[-1],
                test_results['utilities_by_type'].get('CD', [0])[-1],
                avg_payment, avg_comm_time, avg_comp_time, total_cost, avg_time
            ])

    print("\n" + "=" * 50)
    print(f"TEST RESULTS (Average over {num_episodes} episodes)")
    print(f"Total devices: {num_devices}")
    print("=" * 50)

    print("\nREWARDS:")
    for agent_name in agents.keys():
        rewards = test_results['rewards_by_type'][agent_name]
        if rewards:
            print(f"  {agent_name}: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")

    print("\nPRIVACY & COSTS:")
    print(f"  Leak Probability: {np.mean(test_results['leak_probs']):.4f} ± {np.std(test_results['leak_probs']):.4f}")
    print(
        f"  Privacy Improvement: {np.mean(test_results['privacy_improvements']):.4f} ± {np.std(test_results['privacy_improvements']):.4f}")
    print(
        f"  Defense Success Rate: {np.mean(test_results['defense_success_rates']):.3f} ± {np.std(test_results['defense_success_rates']):.3f}")
    print(
        f"  Action Quality: {np.mean(test_results['action_quality_scores']):.3f} ± {np.std(test_results['action_quality_scores']):.3f}")

    if test_results['coalition_stability_rates']:
        print(
            f"  Coalition Stability: {np.mean(test_results['coalition_stability_rates']):.3f} ± {np.std(test_results['coalition_stability_rates']):.3f}")

    success_rate_01 = sum(1 for x in test_results['leak_probs'] if x < 0.1) / len(test_results['leak_probs'])
    success_rate_02 = sum(1 for x in test_results['leak_probs'] if x < 0.2) / len(test_results['leak_probs'])

    print("\nSUCCESS RATES:")
    print(f"  Leak < 0.1: {success_rate_01:.1%}")
    print(f"  Leak < 0.2: {success_rate_02:.1%}")

    print("\nCOSTS:")
    print(f"  Average Payment: {np.mean(test_results['payments']):.2f} ± {np.std(test_results['payments']):.2f}")
    print(
        f"  Communication Cost: {np.mean(test_results['communication_costs']):.2f} ± {np.std(test_results['communication_costs']):.2f}")
    print(
        f"  Computation Cost: {np.mean(test_results['computation_costs']):.2f} ± {np.std(test_results['computation_costs']):.2f}")
    print(
        f"  Total Cost: {np.mean([test_results['payments'][i] + test_results['times'][i] for i in range(len(test_results['payments']))]):.2f}")

    print("=" * 50)

    return test_results


def get_args():
    parser = argparse.ArgumentParser(description='Multi-Agent Federated Learning with Differential Privacy')
    parser.add_argument('--num-devices', type=int, default=DEFAULT_NUM_DEVICES,
                        help='Total number of devices in the federated learning system')
    parser.add_argument('--max-episodes', type=int, default=500,
                        help='Maximum number of training episodes')
    parser.add_argument('--test-episodes', type=int, default=20,
                        help='Number of test episodes')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    set_seed(args.seed)

    print("Starting Optimized Multi-Agent Federated Learning...")
    print("With Enhanced Differential Privacy (ε=3.0) and Privacy Release Mechanism")
    print("=" * 50)

    agents, training_metrics = train_multi_agent(num_devices=args.num_devices, max_episodes=args.max_episodes)

    print("\nTesting trained agents...")
    test_results = test_agents(agents, num_devices=args.num_devices, num_episodes=args.test_episodes)

    print("\nTraining and testing completed successfully!")
    print(f"Configuration: {args.num_devices} devices, {args.max_episodes} training episodes")
    print(f"Results saved to: {result_filename}")
    print("Differential Privacy with Release Mechanism successfully integrated!")
