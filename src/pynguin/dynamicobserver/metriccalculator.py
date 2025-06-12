from abc import ABC, abstractmethod
from scipy.spatial.distance import hamming
from scipy.stats import binned_statistic
import numpy as np
import math
import statistics
from pynguin.ga.testsuitechromosome import TestSuiteChromosome
from pynguin.dynamicobserver.metricutils import MetricHelper, Metric, FitnessObservationMethod
import pynguin.configuration as config
import logging


class MetricCalculator(ABC):
    _logger = logging.getLogger(__name__)

    def __init__(self):
        self.helper = MetricHelper()
        self.SLIDING_WINDOW_SIZE = config.configuration.metric_configuration.sliding_window_size


    @abstractmethod
    def calculate_metric(self, actual_search_results: list[TestSuiteChromosome], search_iteration : int, method: FitnessObservationMethod) -> tuple[Metric, float]:
        """Called to calculate specific metric.

        Args:
            actual_search_results: Actual results of current search.
            search_iteration: Number of iterations done.
        """

class PopulationInformationContentCalculator(MetricCalculator):

    def calculate_metric(self, actual_search_results: list[TestSuiteChromosome], search_iteration : int, method: FitnessObservationMethod) -> tuple[Metric, float]:
        if search_iteration % self.SLIDING_WINDOW_SIZE == 0 and search_iteration > 0:
            calculation_iteration = search_iteration // self.SLIDING_WINDOW_SIZE

            match method:
                case FitnessObservationMethod.MAX:
                    binary_fitness_evolution = self.helper.get_fitness_evoluation_binary_string(self.helper.get_max_fitness_per_generation(actual_search_results, calculation_iteration))
                    return self._calculate_pic(binary_fitness_evolution)
                case FitnessObservationMethod.MIN:
                    binary_fitness_evolution = self.helper.get_fitness_evoluation_binary_string(self.helper.get_min_fitness_per_generation(actual_search_results, calculation_iteration))
                    return self._calculate_pic(binary_fitness_evolution)
                case FitnessObservationMethod.MEAN:
                    binary_fitness_evolution = self.helper.get_fitness_evoluation_binary_string(self.helper.get_mean_fitness_per_generation(actual_search_results, calculation_iteration))
                    return self._calculate_pic(binary_fitness_evolution)
                case FitnessObservationMethod.MEDIAN:
                    binary_fitness_evolution = self.helper.get_fitness_evoluation_binary_string(self.helper.get_median_fitness_per_generation(actual_search_results, calculation_iteration))
                    return self._calculate_pic(binary_fitness_evolution)
                case _:
                    return Metric.EMPTY, 0
        else:
            return Metric.EMPTY, 0

    def _calculate_pic(self, binary_fitness_evolution: str):
        substring_probabilities = self.helper.calculate_substring_probability(binary_fitness_evolution)

        pic = 0
        for i in range(2, len(binary_fitness_evolution)):
            substring = binary_fitness_evolution[i-2 : i]
            pic += substring_probabilities[substring] * math.log2(substring_probabilities[substring])

        return Metric.PIC, pic * -1

class FitnessVarianceCalculator(MetricCalculator):
        def calculate_metric(self, actual_search_results: list[TestSuiteChromosome], search_iteration : int, method: FitnessObservationMethod) -> tuple[Metric, float]:
            match method:
                case FitnessObservationMethod.MEAN:
                    generation_fitnesses = self.helper.get_fitness_of_generation(actual_search_results[search_iteration])
                    mean_fitness = statistics.mean(generation_fitnesses)
                    self._logger.info(f"meanfitness: {mean_fitness}")
                    return self._calculate_variance(mean_fitness, generation_fitnesses)
                case FitnessObservationMethod.MEDIAN:
                    generation_fitnesses = self.helper.get_fitness_of_generation(actual_search_results[search_iteration])
                    median_fitness = statistics.median(generation_fitnesses)
                    self._logger.info(f"medianfitness: {median_fitness}")
                    return self._calculate_variance(median_fitness, generation_fitnesses)
                case _:
                    return Metric.EMPTY, 0

        def _calculate_variance(self, subtrahend: float, generation_fitnesses: list[float]):
            fitness_deviation = 0
            for fitness in generation_fitnesses:
                fitness_deviation += (fitness - subtrahend)**2

            if len(generation_fitnesses) > 1:
                variance = (1 / (len(generation_fitnesses) - 1)) * fitness_deviation
                return Metric.FV, variance

            return Metric.EMPTY, 0



class ChangeRateCalculator(MetricCalculator):

        def calculate_metric(self, actual_search_results: list[TestSuiteChromosome], search_iteration : int, method: FitnessObservationMethod) -> tuple[Metric, float]:
            if search_iteration % self.SLIDING_WINDOW_SIZE == 0 and search_iteration > 0:
                calculation_iteration = search_iteration // self.SLIDING_WINDOW_SIZE

                match method:
                    case FitnessObservationMethod.MAX:
                        binary_fitness_evolution = self.helper.get_fitness_evoluation_binary_string(self.helper.get_max_fitness_per_generation(actual_search_results, calculation_iteration))
                        return self._calculate_change_rate(binary_fitness_evolution)
                    case FitnessObservationMethod.MIN:
                        binary_fitness_evolution = self.helper.get_fitness_evoluation_binary_string(self.helper.get_min_fitness_per_generation(actual_search_results, calculation_iteration))
                        return self._calculate_change_rate(binary_fitness_evolution)
                    case FitnessObservationMethod.MEAN:
                        binary_fitness_evolution = self.helper.get_fitness_evoluation_binary_string(self.helper.get_mean_fitness_per_generation(actual_search_results, calculation_iteration))
                        return self._calculate_change_rate(binary_fitness_evolution)
                    case FitnessObservationMethod.MEDIAN:
                        binary_fitness_evolution = self.helper.get_fitness_evoluation_binary_string(self.helper.get_median_fitness_per_generation(actual_search_results, calculation_iteration))
                        return self._calculate_change_rate(binary_fitness_evolution)
                    case _:
                        return (Metric.EMPTY, 0)
            else:
                return (Metric.EMPTY, 0)

        def _calculate_change_rate(self, binary_fitness_evolution: str):
            sum = 0
            for fitness_evolution in binary_fitness_evolution:
                sum += int(fitness_evolution)

            return (Metric.CR, sum / self.SLIDING_WINDOW_SIZE)

class AutocorrelationCalculator(MetricCalculator):

    STEP_SIZE = 1

    def calculate_metric(self, actual_search_results: list[TestSuiteChromosome], search_iteration : int, method: FitnessObservationMethod) -> tuple[Metric, float]:
        if search_iteration % self.SLIDING_WINDOW_SIZE == 0 and search_iteration > 0:
            calculation_iteration = search_iteration // self.SLIDING_WINDOW_SIZE

            fitnesses_of_generations: list[float] = []
            for i in range(0 + int(self.SLIDING_WINDOW_SIZE * (calculation_iteration - 1 ) ), self.SLIDING_WINDOW_SIZE + self.SLIDING_WINDOW_SIZE * (calculation_iteration - 1 )):
                fitness_of_generation: list[float] = self.helper.get_fitness_of_generation(actual_search_results[i])
                for fitness in fitness_of_generation:
                    fitnesses_of_generations.append(fitness)

            match method:
                case FitnessObservationMethod.MAX:
                    best_fitnesses = self.helper.get_max_fitness_per_generation(actual_search_results, calculation_iteration)
                    return self._calculate_autocorrelation(best_fitnesses, statistics.mean(fitnesses_of_generations))
                case FitnessObservationMethod.MIN:
                    min_fitnesses = self.helper.get_min_fitness_per_generation(actual_search_results, calculation_iteration)
                    return self._calculate_autocorrelation(min_fitnesses, statistics.mean(fitnesses_of_generations))
                case FitnessObservationMethod.MEAN:
                    mean_fitnesses = self.helper.get_mean_fitness_per_generation(actual_search_results, calculation_iteration)
                    return self._calculate_autocorrelation(mean_fitnesses, statistics.mean(fitnesses_of_generations))
                case FitnessObservationMethod.MEDIAN:
                    median_fitnesses = self.helper.get_median_fitness_per_generation(actual_search_results, calculation_iteration)
                    return self._calculate_autocorrelation(median_fitnesses, statistics.mean(fitnesses_of_generations))
                case _:
                    return (Metric.EMPTY, 0)
        else:
            return (Metric.EMPTY, 0)

    def _calculate_autocorrelation(self, fitnesses_to_observe, subtrahend):
        autocorellation_numerator: float = 0
        autocorellation_denominator: float = 0
        for i in range(0, self.SLIDING_WINDOW_SIZE ):
            autocorellation_denominator += (fitnesses_to_observe[i] - subtrahend) ** 2

        if autocorellation_denominator == 0:
            return (Metric.EMPTY, 0)

        for i in range(0, self.SLIDING_WINDOW_SIZE - self.STEP_SIZE):
            autocorellation_numerator += (fitnesses_to_observe[i] - subtrahend) * (fitnesses_to_observe[i + self.STEP_SIZE] - subtrahend)

        return Metric.AC, autocorellation_numerator / autocorellation_denominator

class NeutralityVolumeCalculator(MetricCalculator):

    def calculate_metric(self, actual_search_results: list[TestSuiteChromosome], search_iteration : int, method: FitnessObservationMethod) -> tuple[Metric, float]:
        if search_iteration % self.SLIDING_WINDOW_SIZE == 0 and search_iteration > 0:
            calculation_iteration = search_iteration // self.SLIDING_WINDOW_SIZE

            match method:
                case FitnessObservationMethod.MAX:
                    max_fitnesses = self.helper.get_max_fitness_per_generation(actual_search_results, calculation_iteration)
                    return self._calculate_neutrality_volume(max_fitnesses)
                case FitnessObservationMethod.MIN:
                    min_fitnesses = self.helper.get_min_fitness_per_generation(actual_search_results, calculation_iteration)
                    return self._calculate_neutrality_volume(min_fitnesses)
                case FitnessObservationMethod.MEAN:
                    mean_fitnesses = self.helper.get_mean_fitness_per_generation(actual_search_results, calculation_iteration)
                    return self._calculate_neutrality_volume(mean_fitnesses)
                case FitnessObservationMethod.MEDIAN:
                    median_fitnesses = self.helper.get_median_fitness_per_generation(actual_search_results, calculation_iteration)
                    return self._calculate_neutrality_volume(median_fitnesses)
                case _:
                    return (Metric.EMPTY, 0)
        else:
            return Metric.EMPTY, 0

    def _calculate_neutrality_volume(self, fitnesses_to_observe: list[float]):
        self._logger.info(f"Fitnesses NV: {fitnesses_to_observe} with NV: {len(set(fitnesses_to_observe))}.")
        return Metric.NV, len(set(fitnesses_to_observe))

class DiversityCalculator(MetricCalculator):
    def calculate_metric(self, actual_search_results: list[TestSuiteChromosome], search_iteration : int, method: FitnessObservationMethod) -> tuple[Metric, float]:

        match method:
            case FitnessObservationMethod.MAX:
                return Metric.DIV, max(self._calculate_test_case_distance(actual_search_results[search_iteration]))
            case FitnessObservationMethod.MEAN:
                return Metric.DIV, statistics.mean(self._calculate_test_case_distance(actual_search_results[search_iteration]))
            case FitnessObservationMethod.MEDIAN:
                return Metric.DIV, statistics.median(self._calculate_test_case_distance(actual_search_results[search_iteration]))
            case FitnessObservationMethod.MIN:
                return Metric.DIV, min(self._calculate_test_case_distance(actual_search_results[search_iteration]))
            case _:
                return Metric.EMPTY, 0

    def _calculate_test_case_distance(self,  generation: TestSuiteChromosome) -> list[float] :
        distances: list[float] = []
        for i in range(len(generation.test_case_chromosomes) - 1):
            first_testcase = self.helper.test_case_to_string(generation.test_case_chromosomes[i].test_case)
            for j in range(i + 1, len(generation.test_case_chromosomes)):
                second_testcase = self.helper.test_case_to_string(generation.test_case_chromosomes[j].test_case)
                a, b = self._align_length(first_testcase, second_testcase)
                distances.append(hamming(a, b))

        return distances


    def _align_length(self, first_testcase: str, second_testcase: str):
        first = [char for char in first_testcase]
        second = [char for char in second_testcase]

        max_len = max(len(first), len(second))
        first += [''] * (max_len - len(first))
        second += [''] * (max_len - len(second))

        return first, second


class FunctionDispersionCalculator(MetricCalculator):
    # Evtl. existieren zu wenige Individuen
    def calculate_metric(self, actual_search_results: list[TestSuiteChromosome], search_iteration : int, method: FitnessObservationMethod) -> tuple[Metric, float]:

        match method:
            case FitnessObservationMethod.MAX:
                return Metric.FD, max(self._calculate_distances_between_normalized_fitnesses(self._normalize_fitnesses(self.helper.get_fitness_of_generation(actual_search_results[search_iteration]))))
            case FitnessObservationMethod.MEAN:
                return Metric.FD, statistics.mean(self._calculate_distances_between_normalized_fitnesses(self._normalize_fitnesses(self.helper.get_fitness_of_generation(actual_search_results[search_iteration]))))
            case FitnessObservationMethod.MEDIAN:
                return Metric.DIV, statistics.median(self._calculate_distances_between_normalized_fitnesses(self._normalize_fitnesses(self.helper.get_fitness_of_generation(actual_search_results[search_iteration]))))
            case FitnessObservationMethod.MIN:
                return Metric.DIV, min(self._calculate_distances_between_normalized_fitnesses(self._normalize_fitnesses(self.helper.get_fitness_of_generation(actual_search_results[search_iteration]))))
            case _:
                return Metric.EMPTY, 0

    def _calculate_distances_between_normalized_fitnesses(self, normalized_fitnesses: list[float]) -> list[float]:
        distances: list[float] = []
        for i in range(len(normalized_fitnesses) - 1):
            for j in range(i + 1, len(normalized_fitnesses)):
                distances.append(normalized_fitnesses[i] - normalized_fitnesses[j])

        return distances

    def _normalize_fitnesses(self, fitnesses : list[float]) -> list[float]:
        normalized_fitnesses: list[float] = []

        for fitness in fitnesses:
            normalized_fitnesses.append((fitness - min(fitnesses)) / (max(fitnesses) - min(fitnesses)))

        return normalized_fitnesses

class StateVarianceCalculator(MetricCalculator):
    def calculate_metric(self, actual_search_results: list[TestSuiteChromosome], search_iteration : int, method: FitnessObservationMethod) -> tuple[Metric, float]:
        generation_fitness = self.helper.get_fitness_of_generation(actual_search_results[search_iteration])
        match method:
            case FitnessObservationMethod.MAX:
                bins = binned_statistic(generation_fitness, generation_fitness, bins=5, statistic=FitnessObservationMethod.MAX.value)
                return self._calculate_binned_variance(bins.statistic, statistics.mean(generation_fitness))
            case FitnessObservationMethod.MEAN:
                bins = binned_statistic(generation_fitness, generation_fitness, bins=5, statistic=FitnessObservationMethod.MEAN.value)
                return self._calculate_binned_variance(bins.statistic, statistics.mean(generation_fitness))
            case FitnessObservationMethod.MEDIAN:
                bins = binned_statistic(generation_fitness, generation_fitness, bins=5, statistic=FitnessObservationMethod.MEDIAN.value)
                return self._calculate_binned_variance(bins.statistic, statistics.mean(generation_fitness))
            case FitnessObservationMethod.MIN:
                bins = binned_statistic(generation_fitness, generation_fitness, bins=5, statistic=FitnessObservationMethod.MIN.value)
                return self._calculate_binned_variance(bins.statistic, statistics.mean(generation_fitness))
            case _:
                return Metric.EMPTY, 0

    def _calculate_binned_variance(self, binned_fitnesses: np.ndarray, subtrahend: float):
        self._logger.info(f"Anzahl NaNs: {np.sum(np.isnan(binned_fitnesses))}")

        fitness_values = np.nan_to_num(binned_fitnesses, nan=0.0, posinf=0.0, neginf=0.0)

        fitness_deviation = np.sum((fitness_values - subtrahend) ** 2)

        if len(fitness_values) > 1:
            variance = (1 / (len(fitness_values) - 1)) * fitness_deviation
            return Metric.SV, variance

        return Metric.EMPTY, 0
