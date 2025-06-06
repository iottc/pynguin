from abc import ABC, abstractmethod
import math
import statistics
from pynguin.ga.testsuitechromosome import TestSuiteChromosome
from pynguin.dynamicobserver.metricutils import MetricHelper, Metric, FitnessObservationMethod
import logging

class MetricCalculator(ABC):
    SLIDING_WINDOW_SIZE : int = 10
    _logger = logging.getLogger(__name__)

    def __init__(self):
        self.helper = MetricHelper()

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
                case FitnessObservationMethod.BEST:
                    binary_fitness_evolution = self.helper.get_fitness_evoluation_binary_string(self.helper.get_best_fitness_per_generation(actual_search_results, calculation_iteration, self.SLIDING_WINDOW_SIZE))
                    return self._calculate_pic(binary_fitness_evolution)
                case FitnessObservationMethod.MEAN:
                    binary_fitness_evolution = self.helper.get_fitness_evoluation_binary_string(self.helper.get_mean_fitness_per_generation(actual_search_results, calculation_iteration, self.SLIDING_WINDOW_SIZE))
                    return self._calculate_pic(binary_fitness_evolution)
                case FitnessObservationMethod.MEDIAN:
                    binary_fitness_evolution = self.helper.get_fitness_evoluation_binary_string(self.helper.get_median_fitness_per_generation(actual_search_results, calculation_iteration, self.SLIDING_WINDOW_SIZE))
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
                    case FitnessObservationMethod.BEST:
                        binary_fitness_evolution = self.helper.get_fitness_evoluation_binary_string(self.helper.get_best_fitness_per_generation(actual_search_results, calculation_iteration, self.SLIDING_WINDOW_SIZE))
                        return self._calculate_change_rate(binary_fitness_evolution)
                    case FitnessObservationMethod.MEAN:
                        binary_fitness_evolution = self.helper.get_fitness_evoluation_binary_string(self.helper.get_mean_fitness_per_generation(actual_search_results, calculation_iteration, self.SLIDING_WINDOW_SIZE))
                        return self._calculate_change_rate(binary_fitness_evolution)
                    case FitnessObservationMethod.MEDIAN:
                        binary_fitness_evolution = self.helper.get_fitness_evoluation_binary_string(self.helper.get_median_fitness_per_generation(actual_search_results, calculation_iteration, self.SLIDING_WINDOW_SIZE))
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

    STEP_SIZE: int = 1

    def calculate_metric(self, actual_search_results: list[TestSuiteChromosome], search_iteration : int, method: FitnessObservationMethod) -> tuple[Metric, float]:
        if search_iteration % self.SLIDING_WINDOW_SIZE == 0 and search_iteration > 0:
            calculation_iteration = search_iteration // self.SLIDING_WINDOW_SIZE

            fitnesses_of_generations: list[float] = []
            for i in range(0 + int(self.SLIDING_WINDOW_SIZE * (calculation_iteration - 1 ) ), self.SLIDING_WINDOW_SIZE + self.SLIDING_WINDOW_SIZE * (calculation_iteration - 1 )):
                fitness_of_generation: list[float] = self.helper.get_fitness_of_generation(actual_search_results[i])
                for fitness in fitness_of_generation:
                    fitnesses_of_generations.append(fitness)

            match method:
                case FitnessObservationMethod.BEST:
                    best_fitnesses = self.helper.get_best_fitness_per_generation(actual_search_results, calculation_iteration, self.SLIDING_WINDOW_SIZE)
                    return self._calculate_autocorrelation(best_fitnesses, statistics.mean(fitnesses_of_generations))
                case FitnessObservationMethod.MEAN:
                    mean_fitnesses = self.helper.get_mean_fitness_per_generation(actual_search_results, calculation_iteration, self.SLIDING_WINDOW_SIZE)
                    return self._calculate_autocorrelation(mean_fitnesses, statistics.mean(fitnesses_of_generations))
                case FitnessObservationMethod.MEDIAN:
                    median_fitnesses = self.helper.get_median_fitness_per_generation(actual_search_results, calculation_iteration, self.SLIDING_WINDOW_SIZE)
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
                case FitnessObservationMethod.BEST:
                    best_fitnesses = self.helper.get_best_fitness_per_generation(actual_search_results, calculation_iteration, self.SLIDING_WINDOW_SIZE)
                    return self._calculate_neutrality_volume(best_fitnesses)
                case FitnessObservationMethod.MEAN:
                    mean_fitnesses = self.helper.get_mean_fitness_per_generation(actual_search_results, calculation_iteration, self.SLIDING_WINDOW_SIZE)
                    return self._calculate_neutrality_volume(mean_fitnesses)
                case FitnessObservationMethod.MEDIAN:
                    median_fitnesses = self.helper.get_median_fitness_per_generation(actual_search_results, calculation_iteration, self.SLIDING_WINDOW_SIZE)
                    return self._calculate_neutrality_volume(median_fitnesses)
                case _:
                    return (Metric.EMPTY, 0)
        else:
            return Metric.EMPTY, 0

    def _calculate_neutrality_volume(self, fitnesses_to_observe: list[float]):
        self._logger.info(f"Fitnesses NV: {fitnesses_to_observe} with NV: {len(set(fitnesses_to_observe))}.")
        return Metric.NV, len(set(fitnesses_to_observe))
