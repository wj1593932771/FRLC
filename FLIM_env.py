import numpy as np
import gym
from gym import spaces

NUM_DEVICES = 300
DEVICE_TYPE_RATIOS = {
    'ND': 0.50,
    'RD': 0.14,
    'FD': 0.10,
    'ED': 0.22,
    'CD': 0.04
}


class FLIMEnv(gym.Env):

    def __init__(self, num_devices=NUM_DEVICES):
        super(FLIMEnv, self).__init__()

        self.num_devices = num_devices
        self.device_types = ['ND', 'RD', 'FD', 'ED', 'CD']

        self.device_counts = self._calculate_device_counts()

        self.max_steps = 16
        self.current_step = 0

        self.leak_prob = 0.8
        self.payment = 0
        self.communication_time = 0
        self.computation_time = 0
        self.global_time = 0
        self.K_theta = 1.0

        self.defense_success_rate = 0.2
        self.current_attack_intensity = 0.1
        self.device_cooperation_history = []
        self.server_security_level = 0.0

        self.device_states = {}
        self.device_rewards = {}
        self._initialize_device_states()

        print(f"Environment initialized: total devices={self.num_devices}")
        print(f"Device distribution: {self.device_counts}")

    def _calculate_device_counts(self):
        counts = {}
        remaining = self.num_devices

        for i, device_type in enumerate(self.device_types[:-1]):
            count = int(self.num_devices * DEVICE_TYPE_RATIOS[device_type])
            counts[device_type] = count
            remaining -= count

        counts[self.device_types[-1]] = remaining

        return counts

    def _initialize_device_states(self):
        for device_type in self.device_types:
            count = self.device_counts[device_type]
            if count > 0:
                self.device_states[device_type] = np.zeros((count, 4))
                self.device_rewards[device_type] = np.zeros(count)

    def reset(self):
        self.current_step = 0

        self.leak_prob = 0.8 + np.random.uniform(-0.1, 0.1)
        self.payment = 0
        self.communication_time = 0
        self.computation_time = 0
        self.global_time = 0

        self.defense_success_rate = 0.2
        self.current_attack_intensity = np.random.uniform(0.35, 0.65)

        self.device_cooperation_history = []
        self.server_security_level = 0.0

        server_obs = [np.array([self.leak_prob, self.payment / 10.0, self.global_time / 100.0])]

        device_observations = {}
        for device_type in self.device_types:
            count = self.device_counts[device_type]
            if count > 0:
                obs = np.array([self.leak_prob, 0.0, 0.0, 0.0])
                device_observations[device_type] = [obs] * count
            else:
                device_observations[device_type] = []

        nd_obs = device_observations.get('ND', [np.array([self.leak_prob, 0.0, 0.0, 0.0])])
        rd_obs = device_observations.get('RD', [np.array([self.leak_prob, 0.0, 0.0, 0.0])])
        fd_obs = device_observations.get('FD', [np.array([self.leak_prob, 0.0, 0.0, 0.0])])

        self.all_device_observations = device_observations

        return server_obs, nd_obs[0:1], rd_obs[0:1], fd_obs[0:1]

    def step(self, actions):
        self.current_step += 1

        prev_leak_prob = self.leak_prob

        server_actions = actions['server']
        nd_actions = actions.get('ND', [0, 0, 0])
        rd_actions = actions.get('RD', [0, 0])
        fd_actions = actions.get('FD', [0, 0, 0])
        ed_actions = actions.get('ED', [0, 0, 0])
        cd_actions = actions.get('CD', [0, 0, 0])

        all_device_actions = {
            'ND': nd_actions,
            'RD': rd_actions,
            'FD': fd_actions,
            'ED': ed_actions,
            'CD': cd_actions
        }

        self._update_privacy_leakage_multi(server_actions, all_device_actions)
        self._update_payment_multi(server_actions, all_device_actions)
        self._update_time_costs_multi(server_actions, all_device_actions)

        self._update_defense_success_rate(server_actions, all_device_actions)

        server_obs_next = [np.array([self.leak_prob, self.payment / 10.0, self.global_time / 100.0])]

        device_observations_next = {}
        for device_type in self.device_types:
            count = self.device_counts[device_type]
            if count > 0:
                obs = np.array([self.leak_prob, self.payment / 10.0, self.global_time / 100.0, 0.0])
                device_observations_next[device_type] = [obs] * count
            else:
                device_observations_next[device_type] = []

        nd_obs_next = device_observations_next.get('ND', [
            np.array([self.leak_prob, self.payment / 10.0, self.global_time / 100.0, 0.0])])
        rd_obs_next = device_observations_next.get('RD', [
            np.array([self.leak_prob, self.payment / 10.0, self.global_time / 100.0, 0.0])])
        fd_obs_next = device_observations_next.get('FD', [
            np.array([self.leak_prob, self.payment / 10.0, self.global_time / 100.0, 0.0])])

        server_reward = self._calculate_server_reward(prev_leak_prob)

        all_device_rewards = {}
        for device_type in self.device_types:
            if self.device_counts[device_type] > 0:
                device_actions = all_device_actions.get(device_type, [0, 0, 0])
                reward = self._calculate_device_reward(device_type, device_actions, prev_leak_prob)
                all_device_rewards[device_type] = reward
            else:
                all_device_rewards[device_type] = 0

        nd_reward = all_device_rewards.get('ND', 0)
        rd_reward = all_device_rewards.get('RD', 0)
        fd_reward = all_device_rewards.get('FD', 0)

        server_rewards = [server_reward]
        client_rewards = [nd_reward, rd_reward, fd_reward]

        done = self.current_step >= self.max_steps
        done_list = [done]

        info = [{
            'leak': self.leak_prob,
            'communication_time': self.communication_time,
            'computation_time': self.computation_time,
            'K_theta': self.K_theta,
            'payment': self.payment,
            'expected_time': self.communication_time + self.computation_time,
            'global_time': self.global_time,
            'privacy_improvement': prev_leak_prob - self.leak_prob,
            'num_devices': self.num_devices,
            'device_counts': self.device_counts,
            'all_device_rewards': all_device_rewards,
            'defense_success_rate': self.defense_success_rate
        }]

        return (server_obs_next, nd_obs_next[0:1], rd_obs_next[0:1], fd_obs_next[0:1],
                server_rewards, client_rewards, done_list, info)

    def _update_defense_success_rate(self, server_actions, all_device_actions):
        base_defense = max(0.0, 1.0 - self.leak_prob)

        device_cooperation = self._calculate_device_cooperation(all_device_actions)

        server_security = self._calculate_server_security(server_actions)

        self._update_attack_intensity()

        cooperation_bonus = device_cooperation * 0.25
        security_bonus = server_security * 0.20
        attack_penalty = self.current_attack_intensity * 0.45

        scale_bonus = min(0.15, np.log(self.num_devices + 1) / 20)

        self.defense_success_rate = base_defense * (1 + cooperation_bonus + security_bonus + scale_bonus) * (
                    1 - attack_penalty)

        self.defense_success_rate = np.clip(self.defense_success_rate, 0.0, 0.99)

    def _calculate_device_cooperation(self, all_device_actions):
        total_cooperation = 0
        total_devices = 0

        for device_type in self.device_types:
            count = self.device_counts.get(device_type, 0)
            if count > 0:
                device_actions = all_device_actions.get(device_type, [0, 0, 0])

                if len(device_actions) >= 2:
                    privacy_action = device_actions[0] / 9.0
                    participation = device_actions[1] / 9.0

                    cooperation = (privacy_action * 0.6 + participation * 0.4)
                    total_cooperation += cooperation * count
                    total_devices += count

        if total_devices > 0:
            avg_cooperation = total_cooperation / total_devices
        else:
            avg_cooperation = 0.5

        self.device_cooperation_history.append(avg_cooperation)
        if len(self.device_cooperation_history) > 10:
            self.device_cooperation_history.pop(0)

        if len(self.device_cooperation_history) > 1:
            cooperation_stability = 1.0 - min(1.0, np.var(self.device_cooperation_history))
            return avg_cooperation * cooperation_stability

        return avg_cooperation

    def _calculate_server_security(self, server_actions):
        encryption_level = server_actions[0] / 9.0
        aggregation_strategy = server_actions[1] / 9.0
        security_protocol = server_actions[2] / 9.0

        self.server_security_level = (encryption_level * 0.4 +
                                      aggregation_strategy * 0.3 +
                                      security_protocol * 0.3)

        synergy_effect = min(encryption_level, aggregation_strategy, security_protocol)
        security_with_synergy = self.server_security_level + synergy_effect * 0.2

        return np.clip(security_with_synergy, 0.0, 1.0)

    def _update_attack_intensity(self):
        base_change = np.random.uniform(-0.05, 0.05)

        if self.defense_success_rate > 0.8:
            attack_escalation = np.random.uniform(0.0, 0.1)
        else:
            attack_escalation = 0.0

        if self.leak_prob < 0.1:
            targeted_attack = np.random.uniform(0.0, 0.15)
        else:
            targeted_attack = 0.0

        self.current_attack_intensity += base_change + attack_escalation + targeted_attack
        self.current_attack_intensity = np.clip(self.current_attack_intensity, 0.1, 0.8)

    def _update_privacy_leakage_multi(self, server_actions, all_device_actions):
        encryption_level = server_actions[0] / 9.0
        aggregation_strategy = server_actions[1] / 9.0
        server_privacy_action = server_actions[2] / 9.0

        total_privacy_contribution = 0
        total_weight = 0

        for device_type in self.device_types:
            count = self.device_counts.get(device_type, 0)
            if count > 0:
                device_actions = all_device_actions.get(device_type, [0])
                device_privacy = device_actions[0] / 9.0

                privacy_weight = self._get_device_privacy_weight(device_type)

                count_weight = np.log(count + 1) / np.log(self.num_devices + 1)

                total_privacy_contribution += device_privacy * privacy_weight * count_weight
                total_weight += count_weight

        if total_weight > 0:
            avg_device_privacy = total_privacy_contribution / total_weight
        else:
            avg_device_privacy = 0.5

        server_encryption_contrib = 0.4 * (encryption_level ** 2.0)
        server_aggregation_contrib = 0.3 * (aggregation_strategy ** 2.0)
        server_privacy_contrib = 0.2 * (server_privacy_action ** 2.0)

        device_contribution = 0.6 * (avg_device_privacy ** 1.8)

        scale_bonus = 0.1 * (1 - np.exp(-self.num_devices / 20))

        synergy_bonus = 0.3 * min(
            (encryption_level + aggregation_strategy + server_privacy_action) / 3,
            avg_device_privacy
        ) ** 2.5

        privacy_preservation = (server_encryption_contrib + server_aggregation_contrib +
                                server_privacy_contrib + device_contribution +
                                synergy_bonus + scale_bonus)

        privacy_preservation = np.clip(privacy_preservation, 0.0, 1.2)

        self.leak_prob = max(0.001, 1.0 - privacy_preservation)

    def _get_device_privacy_weight(self, device_type):
        weights = {
            'ND': 1.2,
            'RD': 0.9,
            'FD': 0.7,
            'ED': 1.1,
            'CD': 1.0
        }
        return weights.get(device_type, 1.0)

    def _update_payment_multi(self, server_actions, all_device_actions):
        incentive_level = server_actions[2] / 9.0

        total_participation = 0
        for device_type in self.device_types:
            count = self.device_counts.get(device_type, 0)
            if count > 0:
                device_actions = all_device_actions.get(device_type, [0, 0])
                if len(device_actions) > 1:
                    participation = device_actions[1] / 9.0
                else:
                    participation = 0.5
                total_participation += participation * count

        avg_participation = total_participation / max(self.num_devices, 1)

        base_payment = 5.0
        scale_factor = 1.0 + np.log(self.num_devices) / 10
        self.payment = (base_payment + incentive_level * 10) * avg_participation * scale_factor

    def _update_time_costs_multi(self, server_actions, all_device_actions):
        compression = server_actions[1] / 9.0
        self.communication_time = (1.0 - compression * 0.5) * 5 * np.log(self.num_devices + 1)

        model_complexity = server_actions[0] / 9.0

        total_compute = 0
        total_device_count = 0

        for device_type in self.device_types:
            count = self.device_counts.get(device_type, 0)
            if count > 0:
                device_actions = all_device_actions.get(device_type, [0, 0, 0])
                if len(device_actions) > 2:
                    compute = 1.0 - device_actions[2] / 9.0
                else:
                    compute = self._get_default_compute(device_type)

                total_compute += compute * count
                total_device_count += count

        if total_device_count > 0:
            avg_compute = total_compute / total_device_count
        else:
            avg_compute = 0.5

        self.computation_time = model_complexity * avg_compute * 10

        self.global_time += self.communication_time + self.computation_time

    def _get_default_compute(self, device_type):
        compute_power = {
            'ND': 0.3,
            'RD': 0.5,
            'FD': 0.7,
            'ED': 0.4,
            'CD': 0.2
        }
        return compute_power.get(device_type, 0.5)

    def _calculate_server_reward(self, prev_leak_prob):
        privacy_improvement = prev_leak_prob - self.leak_prob
        scale_factor = 1.0 + np.log(self.num_devices / 10 + 1)
        improvement_reward = privacy_improvement * 300 * scale_factor

        if self.leak_prob < 0.01:
            target_bonus = 50
        elif self.leak_prob < 0.1:
            target_bonus = 20
        elif self.leak_prob < 0.2:
            target_bonus = 10
        elif self.leak_prob < 0.3:
            target_bonus = 5
        else:
            target_bonus = 0

        device_factor = 1.0 + np.log(self.num_devices / 10 + 1)
        privacy_penalty = self.leak_prob * 50 * device_factor

        payment_cost = self.payment * 0.2 / device_factor
        time_cost = (self.communication_time + self.computation_time) * 0.15 / device_factor

        participation_bonus = 0
        active_device_ratio = sum(self.device_counts.values()) / self.num_devices
        if active_device_ratio > 0.7:
            participation_bonus = 10 * (1 + np.log(self.num_devices / 10 + 1))

        if self.leak_prob < 0.01:
            excellence_bonus = 100 * scale_factor
        elif self.leak_prob < 0.05:
            excellence_bonus = 30 * scale_factor
        else:
            excellence_bonus = 0

        cost_efficiency_bonus = 0
        if self.leak_prob < 0.2:
            total_cost = self.payment + self.communication_time + self.computation_time
            if total_cost < 50:
                cost_efficiency_bonus = (50 - total_cost) * 0.05

        defense_bonus = 0
        if hasattr(self, 'defense_success_rate'):
            if self.defense_success_rate > 0.6:
                defense_bonus = (self.defense_success_rate - 0.6) * 25 * scale_factor * 0.1

            if self.leak_prob < 0.1 and self.defense_success_rate > 0.8:
                defense_bonus += 10 * scale_factor * 0.05

        reward = (improvement_reward - privacy_penalty + target_bonus +
                  excellence_bonus - payment_cost - time_cost +
                  participation_bonus + cost_efficiency_bonus + defense_bonus)

        return reward

    def _calculate_device_reward(self, device_type, actions, prev_leak_prob):
        device_count = self.device_counts[device_type]
        if device_count > 0:
            type_payment = self.payment * (device_count / self.num_devices)
            payment_received = type_payment / device_count
        else:
            payment_received = 0

        if len(actions) > 1:
            participation_level = actions[1] / 9.0
        else:
            participation_level = 0.5

        participation_cost = participation_level * 2

        privacy_improvement = prev_leak_prob - self.leak_prob
        scale_factor = 1.0 + np.log(self.num_devices / 10 + 1)
        privacy_reward = privacy_improvement * 120 * scale_factor

        privacy_action = actions[0] / 9.0
        privacy_action_bonus = privacy_action * 12 * scale_factor

        device_factor = 1.0 + np.log(self.num_devices / 10 + 1)
        privacy_penalty = self.leak_prob * 8 / device_factor

        type_bonus = self._get_device_type_bonus(device_type)

        collaboration_bonus = 0
        if self.num_devices > 5:
            collaboration_bonus = 2.0 * np.log(self.num_devices) / np.log(10)

        if self.leak_prob < 0.01:
            excellence_bonus = 20 * scale_factor
        elif self.leak_prob < 0.05:
            excellence_bonus = 5 * scale_factor
        else:
            excellence_bonus = 0

        compute_efficiency_bonus = 0
        if len(actions) > 2 and privacy_action > 0.5:
            compute_action = actions[2] / 9.0
            if compute_action < 0.5:
                compute_efficiency_bonus = (0.5 - compute_action) * privacy_action * 2 * 0.05

        device_defense_bonus = 0
        if hasattr(self, 'defense_success_rate') and self.defense_success_rate > 0.6:
            defense_contribution = privacy_action * 0.6 + participation_level * 0.4
            device_defense_bonus = defense_contribution * self.defense_success_rate * 10 * 0.03

        utility = (payment_received + privacy_reward + privacy_action_bonus +
                   excellence_bonus - participation_cost - privacy_penalty +
                   type_bonus + collaboration_bonus +
                   compute_efficiency_bonus + device_defense_bonus)

        return utility

    def _get_device_type_bonus(self, device_type):
        bonuses = {
            'ND': 1.5,
            'RD': 1.2,
            'FD': 1.0,
            'ED': 1.3,
            'CD': 1.6
        }
        return bonuses.get(device_type, 1.0)

    def render(self, mode='human'):
        if mode == 'human':
            print(f"Step: {self.current_step}")
            print(f"Total Devices: {self.num_devices}")
            print(f"Device Distribution: {self.device_counts}")
            print(f"Privacy Leakage: {self.leak_prob:.6f}")
            print(f"Payment: {self.payment:.2f}")
            print(f"Communication Time: {self.communication_time:.2f}")
            print(f"Computation Time: {self.computation_time:.2f}")
            print(f"Defense Success Rate: {self.defense_success_rate:.3f}")
            print("-" * 40)

    def set_device_number(self, num_devices):
        self.num_devices = num_devices
        self.device_counts = self._calculate_device_counts()
        self._initialize_device_states()
        print(f"Device count updated to: {self.num_devices}")
        print(f"New device distribution: {self.device_counts}")
