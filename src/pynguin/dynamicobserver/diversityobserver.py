import pynguin.ga.testsuitechromosome as tsc
import pynguin.ga.searchobserver as so
import pynguin.dynamicobserver.metriccalculator as calc
from pynguin.dynamicobserver.metricutils import MetricWriter, MetricMeasure, DynamicMetricHelper, Metric, FitnessObservationMethod
import time

import logging

class DiversityObserver(so.SearchObserver):

    _logger = logging.getLogger(__name__)

    def __init__(self):
        self.start_time: int
        self.iterations : int
        self.iteration_results: list[tsc.TestSuiteChromosome] = []
        self.metric_results: list[MetricMeasure] = []

        self._writer = MetricWriter()
        self.helper = DynamicMetricHelper()
        self._calculators: list[calc.DynamicMetricCalculator] = []


    def before_search_start(self, start_time_ns: int) -> None:
        self.iterations = 0
        self.start_time = start_time_ns
        self._init_calculators()

    def before_first_search_iteration(self, initial: tsc.TestSuiteChromosome) -> None:
        self._logger.info(f"Initial Population with size: {len(initial.test_case_chromosomes)}.")

        self.iteration_results.append(initial)
        self._init_metric_calculation()


    def after_search_iteration(self, best: tsc.TestSuiteChromosome) -> None:
        self.iterations += 1
        self.iteration_results.append(best)
        self._logger.info(f"Added {self.iterations}'th evolution with size: {len(best.test_case_chromosomes)}.")
        self._init_metric_calculation()


    def after_search_finish(self) -> None:
        end_time = time.time_ns()
        self._logger.info(f"Search Duration: {end_time - self.start_time} ns")
        self._writer.write_metrics(self.metric_results)

    def _init_calculators(self) -> None:
        self._calculators.append(calc.PopulationInformationContentCalculator())
        self._calculators.append(calc.FitnessVarianceCalculator())
        self._calculators.append(calc.ChangeRateCalculator())

    def _init_metric_calculation(self):
        for calculator in self._calculators:
            for observation_method in FitnessObservationMethod:
                measure  = calculator.calculate_metric(self.iteration_results, self.iterations, observation_method)
                if Metric.EMPTY is not measure[0]:
                    self.metric_results.append(MetricMeasure(measure[0], observation_method, self.iterations, measure[1], self.helper.get_actual_population_size(self.iteration_results), self.helper.get_average_fitness_of_generation(self.iteration_results[len(self.iteration_results) -1]), time.time_ns() - self.start_time))
