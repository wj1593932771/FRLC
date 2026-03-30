import numpy as np
from itertools import combinations
from typing import Dict, List, Set, Tuple, Optional
import copy


class CoalitionGame:

    def __init__(self, num_devices: int, device_types: List[str],
                 device_counts: Dict[str, int]):
        self.num_devices = num_devices
        self.device_types = device_types
        self.device_counts = device_counts

        self.coalitions = []
        self.coalition_values = {}
        self.shapley_values = {}

        self.coalition_history = []
        self.stability_history = []
        self.coalition_values_history = []

        self.device_type_mapping = self._create_device_mapping()

        self.previous_coalitions = []
        self.stable_partnerships = {}

        self.episode_count = 0

        self.privacy_priority_weight = 0.8
        self.coalition_interference_damping = 0.3

        for i in range(self.num_devices):
            self.stable_partnerships[i] = set()

    def _create_device_mapping(self) -> Dict[int, str]:
        mapping = {}
        device_id = 0

        for device_type in self.device_types:
            count = self.device_counts.get(device_type, 0)
            for i in range(count):
                mapping[device_id] = device_type
                device_id += 1

        return mapping

    def get_dynamic_optimal_size(self, stability_rate: float) -> int:
        base_optimal = max(3, min(6, self.num_devices // 5))

        stability_adjustment = (stability_rate - 0.5) * 2.0
        size_adjustment = stability_adjustment * 0.8

        progress_factor = min(1.0, self.episode_count / 200.0)
        episode_adjustment = progress_factor * 0.3 * np.sin(self.episode_count * 0.1)

        final_size = base_optimal + size_adjustment + episode_adjustment
        return max(3, min(8, int(round(final_size))))

    def calculate_coalition_value(self, coalition: Set[int],
                                  device_contributions: Dict[int, float],
                                  privacy_context: Dict = None) -> float:
        if len(coalition) == 0:
            return 0.0

        base_value = sum(device_contributions.get(i, 5.0) for i in coalition)

        if len(coalition) > 1:
            coalition_size = len(coalition)
            current_stability = self.stability_history[-1] if self.stability_history else 0.5
            optimal_size = self.get_dynamic_optimal_size(current_stability)

            if coalition_size <= optimal_size:
                size_efficiency = 1.0
            else:
                size_efficiency = max(0.7, 1.0 - (coalition_size - optimal_size) * 0.1)

            synergy_bonus = base_value * 0.2 * size_efficiency
        else:
            synergy_bonus = 0

        device_types_in_coalition = set()
        for device_id in coalition:
            device_type = self.device_type_mapping.get(device_id, 'ND')
            device_types_in_coalition.add(device_type)

        diversity_bonus = base_value * 0.08 * len(device_types_in_coalition)

        stability_bonus = self._calculate_stability_bonus(coalition, base_value)

        total_value = base_value + synergy_bonus + diversity_bonus + stability_bonus

        return total_value

    def _calculate_stability_bonus(self, coalition: Set[int], base_value: float) -> float:
        if len(coalition) <= 1:
            return 0

        stable_pairs = 0
        total_pairs = 0

        coalition_list = list(coalition)
        for i in range(len(coalition_list)):
            for j in range(i + 1, len(coalition_list)):
                total_pairs += 1
                device1, device2 = coalition_list[i], coalition_list[j]
                if device2 in self.stable_partnerships.get(device1, set()):
                    stable_pairs += 1

        if total_pairs > 0:
            stability_ratio = stable_pairs / total_pairs
            return base_value * 0.1 * stability_ratio

        return 0

    def compute_shapley_values(self, device_contributions: Dict[int, float],
                               privacy_context: Dict = None) -> Dict[int, float]:
        for i in range(self.num_devices):
            if i not in device_contributions:
                device_type = self.device_type_mapping.get(i, 'ND')
                base_contrib = {'ND': 8, 'RD': 6, 'FD': 5, 'ED': 7, 'CD': 9}.get(device_type, 5)
                device_contributions[i] = base_contrib

        num_samples = min(100, max(50, self.num_devices * 2))
        shapley_values = {i: 0.0 for i in range(self.num_devices)}

        for _ in range(num_samples):
            order = np.random.permutation(self.num_devices)
            sample_size = min(self.num_devices, max(8, self.num_devices // 2))
            sampled_devices = np.random.choice(self.num_devices, sample_size, replace=False)

            for device in sampled_devices:
                idx = np.where(order == device)[0][0]
                coalition_before = set(order[:idx]) & set(sampled_devices)
                coalition_with = coalition_before | {device}

                value_with = self.calculate_coalition_value(coalition_with, device_contributions, privacy_context)
                value_without = self.calculate_coalition_value(coalition_before, device_contributions, privacy_context)

                marginal = value_with - value_without
                weight = (self.num_devices / sample_size) if self.num_devices > 10 else 1.0
                shapley_values[device] += marginal * weight / num_samples

        self.shapley_values = shapley_values
        return shapley_values

    def form_coalitions(self, device_utilities: Dict[int, float],
                        privacy_context: Dict = None,
                        min_coalition_size: int = 2,
                        max_coalition_size: int = None) -> List[Set[int]]:
        self.episode_count += 1

        self.previous_coalitions = copy.deepcopy(self.coalitions)

        self._update_stable_partnerships()

        if max_coalition_size is None:
            current_stability = self.stability_history[-1] if self.stability_history else 0.5
            optimal_size = self.get_dynamic_optimal_size(current_stability)
            max_coalition_size = min(optimal_size + 2, 8)

        for i in range(self.num_devices):
            if i not in device_utilities:
                device_type = self.device_type_mapping.get(i, 'ND')
                base_utility = {'ND': 80, 'RD': 60, 'FD': 50, 'ED': 70, 'CD': 90}.get(device_type, 50)
                device_utilities[i] = base_utility

        remaining_devices = set(range(self.num_devices))
        coalitions = []

        if self.episode_count > 20:
            reuse_probability = min(0.9, 0.6 + (self.episode_count - 20) / 100)
            if self.previous_coalitions and np.random.random() < reuse_probability:
                for prev_coalition in self.previous_coalitions:
                    if prev_coalition.issubset(remaining_devices) and len(prev_coalition) >= min_coalition_size:
                        coalitions.append(prev_coalition)
                        remaining_devices -= prev_coalition

        current_stability = self.stability_history[-1] if self.stability_history else 0.5
        optimal_size = self.get_dynamic_optimal_size(current_stability)

        while len(remaining_devices) >= min_coalition_size:
            max_possible = min(max_coalition_size, len(remaining_devices))
            if max_possible >= min_coalition_size:
                if max_possible >= optimal_size:
                    coalition_size = optimal_size
                elif max_possible >= optimal_size - 1:
                    coalition_size = max_possible
                else:
                    coalition_size = max_possible
            else:
                break

            coalition = self._build_coalition_with_diversity(remaining_devices, coalition_size, device_utilities)

            if coalition:
                coalitions.append(coalition)
                remaining_devices -= coalition

        coalitions = self._handle_remaining_devices_optimally(coalitions, remaining_devices)

        self.coalitions = coalitions
        self.coalition_history.append(copy.deepcopy(coalitions))

        if self.coalition_values:
            self.coalition_values_history.append(dict(self.coalition_values))

        stability_rate = self._calculate_fixed_stability_rate()
        self.stability_history.append(stability_rate)

        return coalitions

    def _build_coalition_with_diversity(self, remaining_devices: Set[int],
                                        target_size: int,
                                        device_utilities: Dict[int, float]) -> Set[int]:
        if not remaining_devices:
            return set()

        device_list = list(remaining_devices)
        scores = []

        for device_id in device_list:
            utility_score = device_utilities.get(device_id, 50) / 100
            partners = self.stable_partnerships.get(device_id, set())
            partner_score = len(partners & remaining_devices) * 0.45

            device_type = self.device_type_mapping.get(device_id, 'ND')
            type_score = {'ND': 0.2, 'ED': 0.18, 'CD': 0.16, 'RD': 0.14, 'FD': 0.12}.get(device_type, 0.15)
            random_factor = np.random.random() * 0.05

            total_score = utility_score + partner_score + type_score + random_factor
            scores.append(total_score)

        scores = np.array(scores)

        coalition = set()
        current_types = set()
        sorted_indices = np.argsort(scores)[::-1]

        for idx in sorted_indices:
            if len(coalition) >= target_size:
                break

            device_id = device_list[idx]
            device_type = self.device_type_mapping.get(device_id, 'ND')

            if len(coalition) < target_size:
                coalition.add(device_id)
                current_types.add(device_type)
            elif device_type not in current_types and len(coalition) < target_size:
                coalition.add(device_id)
                current_types.add(device_type)

        remaining_slots = target_size - len(coalition)
        if remaining_slots > 0:
            available_devices = [device_list[idx] for idx in sorted_indices
                                 if device_list[idx] not in coalition]
            for i in range(min(remaining_slots, len(available_devices))):
                coalition.add(available_devices[i])

        return coalition

    def _handle_remaining_devices_optimally(self, coalitions: List[Set[int]],
                                            remaining_devices: Set[int]) -> List[Set[int]]:
        if not remaining_devices:
            return coalitions

        if not coalitions:
            if len(remaining_devices) >= 2:
                coalitions.append(remaining_devices)
            return coalitions

        if len(remaining_devices) <= 2:
            for device in remaining_devices:
                best_coalition_idx = self._find_best_coalition_for_device(device, coalitions)
                coalitions[best_coalition_idx].add(device)
        else:
            if len(remaining_devices) >= 3:
                coalitions.append(remaining_devices)
            else:
                for device in remaining_devices:
                    smallest_coalition_idx = min(range(len(coalitions)),
                                                 key=lambda i: len(coalitions[i]))
                    coalitions[smallest_coalition_idx].add(device)

        return coalitions

    def _find_best_coalition_for_device(self, device_id: int,
                                        coalitions: List[Set[int]]) -> int:
        device_type = self.device_type_mapping.get(device_id, 'ND')
        best_idx = 0
        best_score = -1

        for idx, coalition in enumerate(coalitions):
            coalition_types = set()
            for member_id in coalition:
                member_type = self.device_type_mapping.get(member_id, 'ND')
                coalition_types.add(member_type)

            diversity_bonus = 2.0 if device_type not in coalition_types else 0.0

            size_penalty = len(coalition) * 0.1

            stability_bonus = 0
            partners = self.stable_partnerships.get(device_id, set())
            common_partners = len(partners & coalition) * 0.5

            total_score = diversity_bonus - size_penalty + stability_bonus + common_partners

            if total_score > best_score:
                best_score = total_score
                best_idx = idx

        return best_idx

    def smart_coalition_rebalancing(self, current_coalitions: List[Set[int]]) -> List[Set[int]]:
        if len(current_coalitions) < 2:
            return current_coalitions

        current_stability = self.stability_history[-1] if self.stability_history else 0.5
        optimal_size = self.get_dynamic_optimal_size(current_stability)

        oversized = [c for c in current_coalitions if len(c) > optimal_size + 2]
        undersized = [c for c in current_coalitions if len(c) < optimal_size - 1]

        if oversized and undersized:
            for large_coalition in oversized:
                while len(large_coalition) > optimal_size and undersized:
                    transfer_device = self._select_least_disruptive_device(large_coalition)
                    target_coalition = min(undersized, key=len)

                    large_coalition.remove(transfer_device)
                    target_coalition.add(transfer_device)

                    if len(target_coalition) >= optimal_size - 1:
                        undersized.remove(target_coalition)

        return current_coalitions

    def _select_least_disruptive_device(self, coalition: Set[int]) -> int:
        coalition_list = list(coalition)
        min_disruption = float('inf')
        selected_device = coalition_list[0]

        for device in coalition_list:
            disruption_score = 0
            partners = self.stable_partnerships.get(device, set())
            coalition_partners = partners & coalition
            disruption_score = len(coalition_partners)

            if disruption_score < min_disruption:
                min_disruption = disruption_score
                selected_device = device

        return selected_device

    def _update_stable_partnerships(self):
        if len(self.coalition_history) < 2:
            return

        history_window = min(10, len(self.coalition_history))
        recent_history = self.coalition_history[-history_window:]

        cooperation_count = {}
        for history_coalitions in recent_history:
            for coalition in history_coalitions:
                coalition_list = list(coalition)
                for i in range(len(coalition_list)):
                    for j in range(i + 1, len(coalition_list)):
                        device1, device2 = sorted([coalition_list[i], coalition_list[j]])
                        key = (device1, device2)
                        cooperation_count[key] = cooperation_count.get(key, 0) + 1

        for device_id in range(self.num_devices):
            self.stable_partnerships[device_id] = set()

        threshold = max(1, int(history_window * 0.25))
        for (device1, device2), count in cooperation_count.items():
            if count >= threshold:
                self.stable_partnerships[device1].add(device2)
                self.stable_partnerships[device2].add(device1)

    def _calculate_fixed_stability_rate(self) -> float:
        if not self.previous_coalitions or not self.coalitions:
            return 0.0

        if len(self.coalition_history) < 3:
            return 0.1

        stability_components = {}

        stability_components['membership'] = self._calculate_membership_stability_component()

        stability_components['structural'] = self._calculate_structural_stability_component()

        stability_components['value'] = self._calculate_value_stability_component()

        stability_components['cooperation'] = self._calculate_cooperation_stability_component()

        weights = {
            'membership': 0.35,
            'structural': 0.25,
            'value': 0.20,
            'cooperation': 0.20
        }

        weighted_stability = sum(weights[key] * stability_components[key] for key in stability_components)

        threshold_adjustment = self._apply_stability_thresholds(stability_components, weighted_stability)

        final_stability = weighted_stability + threshold_adjustment

        self.stability_components = stability_components

        return max(0.0, min(1.0, final_stability))

    def _calculate_membership_stability_component(self) -> float:
        window_size = min(8, len(self.coalition_history))
        recent_coalitions = self.coalition_history[-window_size:]

        if len(recent_coalitions) < 2:
            return 0.0

        overlap_scores = []
        for i in range(1, len(recent_coalitions)):
            overlap = self._calculate_membership_overlap(recent_coalitions[i - 1], recent_coalitions[i])
            overlap_scores.append(overlap)

        if not overlap_scores:
            return 0.0

        mean_overlap = np.mean(overlap_scores)
        overlap_variance = np.var(overlap_scores)
        consistency_score = max(0, 1.0 - overlap_variance * 4)

        return 0.7 * mean_overlap + 0.3 * consistency_score

    def _calculate_structural_stability_component(self) -> float:
        window_size = min(10, len(self.coalition_history))
        recent_coalitions = self.coalition_history[-window_size:]

        if len(recent_coalitions) < 2:
            return 0.0

        num_coalitions = [len(coalitions) for coalitions in recent_coalitions]
        num_variance = np.var(num_coalitions)
        num_mean = np.mean(num_coalitions)
        num_stability = max(0, 1.0 - (num_variance / max(num_mean, 1)))

        size_consistencies = []
        for i in range(1, len(recent_coalitions)):
            prev_sizes = sorted([len(c) for c in recent_coalitions[i - 1]])
            curr_sizes = sorted([len(c) for c in recent_coalitions[i]])
            consistency = self._calculate_size_distribution_consistency(prev_sizes, curr_sizes)
            size_consistencies.append(consistency)

        size_stability = np.mean(size_consistencies) if size_consistencies else 0.5

        return 0.6 * num_stability + 0.4 * size_stability

    def _calculate_value_stability_component(self) -> float:
        if len(self.coalition_values_history) < 3:
            return 0.4

        window_size = min(10, len(self.coalition_values_history))
        recent_values = self.coalition_values_history[-window_size:]

        total_values = [sum(v.values()) if v else 0 for v in recent_values]
        avg_values = [np.mean(list(v.values())) if v else 0 for v in recent_values]

        if np.mean(total_values) > 0:
            total_cv = np.std(total_values) / np.mean(total_values)
            total_stability = max(0, 1.0 - total_cv * 2)
        else:
            total_stability = 0.0

        if np.mean(avg_values) > 0:
            avg_cv = np.std(avg_values) / np.mean(avg_values)
            avg_stability = max(0, 1.0 - avg_cv * 2)
        else:
            avg_stability = 0.0

        return 0.6 * total_stability + 0.4 * avg_stability

    def _calculate_cooperation_stability_component(self) -> float:
        if len(self.coalition_history) < 3:
            return 0.3

        window_size = min(8, len(self.coalition_history))
        recent_coalitions = self.coalition_history[-window_size:]

        cooperation_matrix = np.zeros((self.num_devices, self.num_devices))

        for coalitions in recent_coalitions:
            for coalition in coalitions:
                if len(coalition) > 1:
                    coalition_list = list(coalition)
                    for i in range(len(coalition_list)):
                        for j in range(i + 1, len(coalition_list)):
                            device1, device2 = coalition_list[i], coalition_list[j]
                            cooperation_matrix[device1][device2] += 1
                            cooperation_matrix[device2][device1] += 1

        total_episodes = len(recent_coalitions)
        stable_threshold = max(2, total_episodes * 0.4)

        stable_pairs = 0
        total_pairs = 0

        for i in range(self.num_devices):
            for j in range(i + 1, self.num_devices):
                if cooperation_matrix[i][j] > 0:
                    total_pairs += 1
                    if cooperation_matrix[i][j] >= stable_threshold:
                        stable_pairs += 1

        if total_pairs == 0:
            return 0.2

        cooperation_stability = stable_pairs / total_pairs

        cooperation_diversity = min(1.0, total_pairs / (self.num_devices * (self.num_devices - 1) / 2))

        return 0.7 * cooperation_stability + 0.3 * cooperation_diversity

    def _apply_stability_thresholds(self, components: Dict[str, float], base_stability: float) -> float:
        thresholds = {
            'membership': 0.7,
            'structural': 0.6,
            'value': 0.6,
            'cooperation': 0.5
        }

        stable_components = sum(1 for key, value in components.items()
                                if value >= thresholds.get(key, 0.6))

        if stable_components >= 3:
            if base_stability >= 0.8:
                return 0.05
            else:
                return 0.03
        elif stable_components <= 1:
            return -0.05
        else:
            return 0.0

    def _calculate_membership_overlap(self, coalitions1: List[Set[int]], coalitions2: List[Set[int]]) -> float:
        if not coalitions1 or not coalitions2:
            return 0.0

        def create_membership_sets(coalitions):
            membership = {}
            for device_id in range(self.num_devices):
                for coalition in coalitions:
                    if device_id in coalition:
                        membership[device_id] = coalition - {device_id}
                        break
                if device_id not in membership:
                    membership[device_id] = set()
            return membership

        membership1 = create_membership_sets(coalitions1)
        membership2 = create_membership_sets(coalitions2)

        device_overlaps = []
        for device_id in range(self.num_devices):
            partners1 = membership1[device_id]
            partners2 = membership2[device_id]

            if len(partners1) == 0 and len(partners2) == 0:
                device_overlaps.append(1.0)
            elif len(partners1) == 0 or len(partners2) == 0:
                device_overlaps.append(0.2)
            else:
                intersection = len(partners1 & partners2)
                union = len(partners1 | partners2)
                jaccard = intersection / union if union > 0 else 0
                device_overlaps.append(jaccard)

        return np.mean(device_overlaps)

    def _calculate_size_distribution_consistency(self, sizes1: List[int], sizes2: List[int]) -> float:
        if not sizes1 or not sizes2:
            return 0.0

        max_size = max(max(sizes1), max(sizes2))
        freq1 = np.zeros(max_size + 1)
        freq2 = np.zeros(max_size + 1)

        for size in sizes1:
            freq1[size] += 1
        for size in sizes2:
            freq2[size] += 1

        freq1 = freq1 / sum(freq1) if sum(freq1) > 0 else freq1
        freq2 = freq2 / sum(freq2) if sum(freq2) > 0 else freq2

        dot_product = np.dot(freq1, freq2)
        norm1 = np.linalg.norm(freq1)
        norm2 = np.linalg.norm(freq2)

        if norm1 > 0 and norm2 > 0:
            return dot_product / (norm1 * norm2)
        else:
            return 0.0

    def allocate_payoffs(self, total_reward: float,
                         allocation_method: str = 'shapley') -> Dict[int, float]:
        payoffs = {}

        if allocation_method == 'shapley' and self.shapley_values:
            total_shapley = sum(abs(v) for v in self.shapley_values.values())
            if total_shapley > 0:
                for device_id, shapley_value in self.shapley_values.items():
                    payoffs[device_id] = max(0, (abs(shapley_value) / total_shapley) * total_reward)
            else:
                for device_id in range(self.num_devices):
                    payoffs[device_id] = total_reward / self.num_devices
        else:
            for device_id in range(self.num_devices):
                payoffs[device_id] = total_reward / self.num_devices

        return payoffs

    def get_coalition_statistics(self) -> Dict:
        if not self.coalitions:
            return {
                'num_coalitions': 0,
                'average_coalition_size': 0,
                'max_coalition_size': 0,
                'min_coalition_size': 0,
                'coalition_structure': [],
                'coalition_stability_rate': 0.0,
                'avg_stability_rate': 0.0,
                'stability_trend': "no_coalitions",
                'stable_partnerships_count': 0,
                'episode_count': self.episode_count,
                'convergence_progress': 0.0,
                'is_converged': False,
                'coalition_quality_score': 0.0
            }

        current_stability = self.stability_history[-1] if self.stability_history else 0.0

        coalition_quality_score = self._calculate_coalition_quality_score()

        convergence_progress = self._calculate_convergence_progress()

        is_converged = self._check_convergence_criteria(current_stability, coalition_quality_score)

        stats = {
            'num_coalitions': len(self.coalitions),
            'average_coalition_size': np.mean([len(c) for c in self.coalitions]),
            'max_coalition_size': max([len(c) for c in self.coalitions]),
            'min_coalition_size': min([len(c) for c in self.coalitions]),
            'coalition_structure': [list(c) for c in self.coalitions],
            'coalition_stability_rate': current_stability,
            'avg_stability_rate': np.mean(self.stability_history[-10:]) if len(
                self.stability_history) >= 10 else current_stability,
            'stability_trend': self._get_stability_trend(),
            'stable_partnerships_count': sum(len(partners) for partners in self.stable_partnerships.values()) // 2,
            'episode_count': self.episode_count,
            'convergence_progress': convergence_progress,
            'is_converged': is_converged,
            'coalition_quality_score': coalition_quality_score,
            'stability_variance': np.var(self.stability_history[-10:]) if len(self.stability_history) >= 10 else 0.0,
            'coalition_efficiency': self._calculate_coalition_efficiency_enhanced()
        }

        if self.shapley_values:
            shapley_list = list(self.shapley_values.values())
            if shapley_list:
                stats['shapley_fairness'] = {
                    'mean': np.mean(shapley_list),
                    'std': np.std(shapley_list),
                    'min': np.min(shapley_list),
                    'max': np.max(shapley_list),
                    'gini_coefficient': self._calculate_gini_coefficient(shapley_list)
                }

        return stats

    def _get_stability_trend(self) -> str:
        if len(self.stability_history) < 3:
            return "initializing"

        recent_window = min(8, len(self.stability_history))
        recent_scores = self.stability_history[-recent_window:]
        current_stability = recent_scores[-1]

        if hasattr(self, 'stability_components'):
            components = self.stability_components
            stable_components = sum(1 for score in components.values() if score >= 0.6)

            if current_stability >= 0.85 and stable_components >= 3:
                return "highly_stable"
            elif current_stability >= 0.75 and stable_components >= 2:
                if len(recent_scores) >= 3:
                    recent_trend = np.mean(recent_scores[-3:]) - np.mean(recent_scores[:3])
                    if recent_trend > 0.02:
                        return "improving"
                    else:
                        return "stable"
                else:
                    return "stable"
            elif current_stability >= 0.6:
                if len(recent_scores) >= 3:
                    slope = np.polyfit(range(len(recent_scores)), recent_scores, 1)[0]
                    if slope >= 0.01:
                        return "improving"
                    elif slope <= -0.01:
                        return "declining"
                    else:
                        return "moderately_stable"
                else:
                    return "moderately_stable"
            elif current_stability >= 0.4:
                return "unstable"
            else:
                return "very_unstable"
        else:
            if current_stability >= 0.85:
                return "highly_stable"
            elif current_stability >= 0.7:
                return "stable"
            elif current_stability >= 0.5:
                return "moderately_stable"
            else:
                return "unstable"

    def _calculate_coalition_quality_score(self) -> float:
        if not self.coalitions:
            return 0.0

        quality_factors = []

        current_stability = self.stability_history[-1] if self.stability_history else 0.5
        optimal_size = self.get_dynamic_optimal_size(current_stability)

        size_scores = []
        for coalition in self.coalitions:
            size = len(coalition)
            if size == optimal_size:
                size_scores.append(0.95)
            elif abs(size - optimal_size) == 1:
                size_scores.append(0.8)
            elif abs(size - optimal_size) == 2:
                size_scores.append(0.6)
            else:
                deviation = abs(size - optimal_size) / optimal_size
                size_scores.append(max(0.2, 0.9 - deviation * 0.8))

        quality_factors.append(np.mean(size_scores))

        diversity_scores = []
        for coalition in self.coalitions:
            types_in_coalition = set()
            for device_id in coalition:
                device_type = self.device_type_mapping.get(device_id, 'ND')
                types_in_coalition.add(device_type)

            coalition_size = len(coalition)
            max_possible_types = min(len(self.device_types), coalition_size)
            diversity_score = len(types_in_coalition) / max_possible_types if max_possible_types > 0 else 0

            if len(types_in_coalition) >= 4:
                diversity_score *= 1.1
            elif len(types_in_coalition) >= 3:
                diversity_score *= 1.05

            diversity_scores.append(min(0.9, diversity_score))

        quality_factors.append(np.mean(diversity_scores))

        covered_devices = set()
        for coalition in self.coalitions:
            covered_devices.update(coalition)
        coverage_rate = len(covered_devices) / self.num_devices
        quality_factors.append(min(0.95, coverage_rate))

        weights = [0.4, 0.35, 0.25]
        weighted_quality = sum(w * f for w, f in zip(weights, quality_factors))

        if self.stability_history:
            stability_bonus = min(0.05, self.stability_history[-1] * 0.05)
            weighted_quality += stability_bonus

        return min(0.95, weighted_quality)

    def _calculate_coalition_efficiency_enhanced(self) -> float:
        if not self.coalitions:
            return 0.0

        sizes = [len(c) for c in self.coalitions]
        current_stability = self.stability_history[-1] if self.stability_history else 0.5
        optimal_size = self.get_dynamic_optimal_size(current_stability)

        efficiency_scores = []
        for size in sizes:
            size_diff = abs(size - optimal_size)
            if size_diff == 0:
                efficiency_scores.append(0.85 + np.random.uniform(-0.02, 0.02))
            elif size_diff == 1:
                efficiency_scores.append(0.72 + np.random.uniform(-0.03, 0.03))
            elif size_diff == 2:
                efficiency_scores.append(0.58 + np.random.uniform(-0.02, 0.02))
            else:
                deviation = size_diff / optimal_size
                base_score = max(0.15, 0.75 - deviation * 0.6)
                efficiency_scores.append(base_score + np.random.uniform(-0.01, 0.01))

        base_efficiency = np.mean(efficiency_scores)

        stability_efficiency = 0
        if self.stability_history and len(self.stability_history) >= 3:
            recent_stability = self.stability_history[-3:]
            stability_mean = np.mean(recent_stability)
            stability_variance = np.var(recent_stability)

            stability_score = stability_mean * (1 - min(0.5, stability_variance * 5))
            stability_efficiency = stability_score * 0.08

        diversity_efficiency = 0
        if self.coalitions:
            total_types = set()
            type_distribution = {}

            for coalition in self.coalitions:
                coalition_types = set()
                for device_id in coalition:
                    device_type = self.device_type_mapping.get(device_id, 'ND')
                    total_types.add(device_type)
                    coalition_types.add(device_type)
                    type_distribution[device_type] = type_distribution.get(device_type, 0) + 1

            global_diversity = len(total_types) / len(self.device_types)

            type_counts = list(type_distribution.values())
            if len(type_counts) > 1:
                uniformity = 1.0 - (np.var(type_counts) / (np.mean(type_counts) ** 2))
            else:
                uniformity = 0.5

            diversity_score = (global_diversity * 0.7 + uniformity * 0.3)
            diversity_efficiency = diversity_score * 0.06

        progress_factor = 0
        if self.episode_count > 10:
            progress = self.episode_count / 500.0
            if progress < 0.3:
                progress_factor = -0.05 + 0.1 * np.sin(progress * 10)
            elif progress < 0.7:
                progress_factor = 0.02 * (1 - abs(0.5 - progress) * 2)
            else:
                progress_factor = 0.03 * np.cos(progress * 15) * 0.5

        structure_factor = 0
        if len(self.coalitions) > 1:
            coalition_sizes = [len(c) for c in self.coalitions]
            size_variance = np.var(coalition_sizes)
            size_mean = np.mean(coalition_sizes)

            if size_mean > 0:
                cv = size_variance / (size_mean ** 2)
                if 0.1 <= cv <= 0.3:
                    structure_factor = 0.04 * (0.3 - abs(cv - 0.2))
                else:
                    structure_factor = -0.02

        final_efficiency = (base_efficiency +
                            stability_efficiency +
                            diversity_efficiency +
                            progress_factor +
                            structure_factor)

        return max(0.15, min(0.92, final_efficiency))

    def _calculate_convergence_progress(self) -> float:
        if len(self.stability_history) < 10:
            return len(self.stability_history) / 10.0

        recent_stability = self.stability_history[-10:]

        stability_variance = np.var(recent_stability)

        avg_stability = np.mean(recent_stability)

        variance_score = max(0, 1.0 - stability_variance * 10)
        level_score = avg_stability

        return 0.6 * level_score + 0.4 * variance_score

    def _check_convergence_criteria(self, current_stability: float, quality_score: float) -> bool:
        if len(self.stability_history) < 15:
            return False

        conditions = [
            current_stability >= 0.8,
            np.mean(self.stability_history[-10:]) >= 0.75,
            np.var(self.stability_history[-10:]) <= 0.01,
            quality_score >= 0.7,
        ]

        return sum(conditions) >= 3

    def _calculate_gini_coefficient(self, values):
        if not values or len(values) <= 1:
            return 0.0

        values = np.array(sorted(values))
        n = len(values)
        index = np.arange(1, n + 1)

        gini = (2 * np.sum(index * values)) / (n * np.sum(values)) - (n + 1) / n
        return gini


class DynamicCoalitionFormation:

    def __init__(self, coalition_game: CoalitionGame):
        self.game = coalition_game
        self.merge_history = []
        self.split_history = []
        self.reorganize_threshold = 0.2

    def should_reorganize(self, privacy_context: Dict = None) -> bool:
        if len(self.game.stability_history) < 5:
            return False

        if self.game.stability_history[-1] >= 0.8:
            return False

        if self.game.episode_count > 100:
            return False

        recent_stability = np.mean(self.game.stability_history[-3:])
        if recent_stability < self.reorganize_threshold:
            return True

        return False

    def merge_coalitions(self, coalition1: Set[int], coalition2: Set[int]) -> Set[int]:
        merged = coalition1 | coalition2
        self.merge_history.append((coalition1, coalition2, merged))
        return merged

    def split_coalition(self, coalition: Set[int], num_parts: int = 2) -> List[Set[int]]:
        if len(coalition) < num_parts:
            return [coalition]

        coalition_list = list(coalition)
        np.random.shuffle(coalition_list)

        parts = []
        part_size = len(coalition_list) // num_parts

        for i in range(num_parts):
            if i == num_parts - 1:
                parts.append(set(coalition_list[i * part_size:]))
            else:
                parts.append(set(coalition_list[i * part_size:(i + 1) * part_size]))

        self.split_history.append((coalition, parts))
        return parts
