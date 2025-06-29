import pynguin.configuration as config
from pynguin.testcase.testcase import TestCase
from pynguin.ga.testsuitechromosome import TestSuiteChromosome
from pathlib import Path
from dataclasses import dataclass
from enum import Enum
import logging
import statistics


class Metric(Enum):
    EMPTY = "Empty"
    PIC = "PIC"
    CR = "CR"
    FV = "FV"
    NSC = "NSC"
    AC = "AC"
    NV = "NV"
    DIV = "DIV"
    FD = "FD"
    SV = "SVC"

class FitnessObservationMethod(Enum):
    MEAN = "mean"
    MAX = "max"
    MEDIAN = "median"
    MIN = "min"

@dataclass
class MetricMeasure:
    name: Metric
    fitness_observation: FitnessObservationMethod
    iteration: int
    result: float
    population_size: int
    max_fitness: float
    average_fitness: float
    min_fitness: float
    duration: float

@dataclass
class TestCasesStrings:
    iteration: int
    test_cases: list[str]

@dataclass
class FitnessValues:
    iteration: int
    fitness_values: list[float]

class MetricWriter():
    FILE_HEADER = "module,iteration,populationSize,maxFitness,averageFitness,minFitness,metric,result,durationInMin\n"
    _logger = logging.getLogger(__name__)

    def write_metrics(self, metrics: list[MetricMeasure]) -> None:
        try:
            metrics.sort(key=lambda measure: (measure.name.value, measure.iteration))
            output_dir = Path(config.configuration.statistics_output.report_dir).resolve()
            output_file = output_dir / "metric_statistics.csv"
            header_necessary: bool

            try:
                output_file.resolve(strict=True)
            except FileNotFoundError:
                header_necessary = True
            else:
                header_necessary = False

            with open(output_file, "a") as file:
                if header_necessary:
                    file.write(self.FILE_HEADER)
                for measure in metrics:
                    file.write(f"{config.configuration.module_name},{measure.iteration},{measure.population_size},{measure.max_fitness},{measure.average_fitness},{measure.min_fitness},{measure.name.value + measure.fitness_observation.value},{measure.result}, {str(measure.duration / 60000000000 )}\n")
        except OSError as error:
            self._logger.exception("Error while writing statistics: %s", error)

class RawDataWriter:

    FILE_HEADER_FITNESS_VALUES = "module,iteration,maxFitness,minFitness,averageFitness,fitnesses\n"
    FILE_HEADER_TEST_CASES = "module,iteration,testcases\n"

    _logger = logging.getLogger(__name__)

    def write_raw_data(self, fitness_values: list[FitnessValues], test_cases: list[TestCasesStrings]) -> None:
        self._write_test_cases(test_cases)
        self._write_fitness_values(fitness_values)

    def _write_test_cases(self, test_cases: list[TestCasesStrings]):
        try:
            output_dir = Path(config.configuration.statistics_output.report_dir).resolve()
            output_file = output_dir / "test_cases.csv"
            header_necessary: bool = self._is_header_necessary(output_file)

            with open(output_file, "a") as file:
                if header_necessary:
                    file.write(self.FILE_HEADER_TEST_CASES)
                for test_case in test_cases:
                    test_cases_string = "|".join(test_case.test_cases)
                    file.write(f"{config.configuration.module_name},{test_case.iteration},{test_cases_string}\n")
        except OSError as error:
            self._logger.exception("Error while writing statistics: %s", error)

    def _write_fitness_values(self, fitness_values: list[FitnessValues]):
        try:
            output_dir = Path(config.configuration.statistics_output.report_dir).resolve()
            output_file = output_dir / "fitness_values.csv"
            header_necessary: bool = self._is_header_necessary(output_file)

            with open(output_file, "a") as file:
                if header_necessary:
                    file.write(self.FILE_HEADER_FITNESS_VALUES)
                for fitnesses in fitness_values:
                    fitness_values_string = "|".join(str(fitness) for fitness in fitness_values)
                    file.write(f"{config.configuration.module_name},{fitnesses.iteration},{max(fitnesses.fitness_values)},{min(fitnesses.fitness_values)},{statistics.mean(fitnesses.fitness_values)},{fitness_values_string}\n")
        except OSError as error:
            self._logger.exception("Error while writing statistics: %s", error)

    def _is_header_necessary(self, output_file) -> bool:
        try:
            output_file.resolve(strict=True)
        except FileNotFoundError:
            return True
        else:
            return False


class MetricHelper:
    _logger = logging.getLogger(__name__)

    def __init__(self):
        self.SLIDING_WINDOW_SIZE = config.configuration.metric_configuration.sliding_window_size

    def get_mean_fitness_per_generation(self, actual_search_results: list[TestSuiteChromosome], calculation_iteration: int) -> list[float]:
        result = []

        for i in range(0 + int(self.SLIDING_WINDOW_SIZE * (calculation_iteration - 1 ) ), self.SLIDING_WINDOW_SIZE + self.SLIDING_WINDOW_SIZE * (calculation_iteration - 1 )):
            generation = actual_search_results[i]
            mean_fitness : float = (statistics.mean(test.get_fitness() for test in generation.test_case_chromosomes))
            self._logger.info(f"Mean fitness of iteration {i} is {mean_fitness}")
            result.append(mean_fitness)

        return result

    def get_median_fitness_per_generation(self, actual_search_results: list[TestSuiteChromosome], calculation_iteration: int) -> list[float]:
        result = []

        for i in range(0 + int(self.SLIDING_WINDOW_SIZE * (calculation_iteration - 1 ) ), self.SLIDING_WINDOW_SIZE + self.SLIDING_WINDOW_SIZE * (calculation_iteration - 1 )):
            generation = actual_search_results[i]
            median_fitness : float = (statistics.median(test.get_fitness() for test in generation.test_case_chromosomes))
            self._logger.info(f"Median fitness of iteration {i} is {median_fitness}")
            result.append(median_fitness)

        return result

    def get_max_fitness_per_generation(self, actual_search_results: list[TestSuiteChromosome], calculation_iteration: int) -> list[float]:
        result = []

        for i in range(0 + int(self.SLIDING_WINDOW_SIZE * (calculation_iteration - 1 ) ), self.SLIDING_WINDOW_SIZE + self.SLIDING_WINDOW_SIZE * (calculation_iteration - 1 )):
            generation = actual_search_results[i]
            best_fitness : float = (max(test.get_fitness() for test in generation.test_case_chromosomes))
            self._logger.info(f"Max fitness of iteration {i} is {best_fitness}")
            result.append(best_fitness)

        return result

    def get_min_fitness_per_generation(self, actual_search_results: list[TestSuiteChromosome], calculation_iteration: int) -> list[float]:
        result = []

        for i in range(0 + int(self.SLIDING_WINDOW_SIZE * (calculation_iteration - 1 ) ), self.SLIDING_WINDOW_SIZE + self.SLIDING_WINDOW_SIZE * (calculation_iteration - 1 )):
            generation = actual_search_results[i]
            best_fitness : float = (min(test.get_fitness() for test in generation.test_case_chromosomes))
            self._logger.info(f"Min fitness of iteration {i} is {best_fitness}")
            result.append(best_fitness)

        return result

    def get_fitness_of_generation(self, generation: TestSuiteChromosome) -> list[float]:
        fitnesses: list[float] = []

        for test in generation.test_case_chromosomes:
            fitnesses.append(test.get_fitness())

        self._logger.info(f"Fitnesses are: {' '.join(map(str,fitnesses))}")

        return fitnesses

    def calculate_substring_probability(self, best_fitnesses_binary: str) -> dict[str, float]:
        result = {}
        result['00'] = best_fitnesses_binary.count('00') / (len(best_fitnesses_binary) - 1)
        result['01'] = best_fitnesses_binary.count('01') / (len(best_fitnesses_binary) - 1)
        result['10'] = best_fitnesses_binary.count('10') / (len(best_fitnesses_binary) - 1)
        result['11'] = best_fitnesses_binary.count('11') / (len(best_fitnesses_binary) - 1)

        self._logger.info(f"Substring Propabilities for given Substring {best_fitnesses_binary} is: {' '.join(map(str, result.values()))}")
        return result

    def get_fitness_evoluation_binary_string(self, best_fitnesses: list[float]) -> str:
        result = []

        for i in range(1, len(best_fitnesses)):
            if best_fitnesses[i] - best_fitnesses[i - 1] == 0:
                result.append(0)
            else:
                result.append(1)

        result_string =  ''.join(str(x) for x in result)
        self._logger.info(f"Result Fitness Binary String: {result_string}")
        return result_string

    def get_average_fitness_of_generation(self, generation: TestSuiteChromosome) -> float:
        return statistics.mean(test.get_fitness() for test in generation.test_case_chromosomes)

    def get_actual_population_size(self, actual_search_results: list[TestSuiteChromosome]) -> int:
        return len(actual_search_results[len(actual_search_results) - 1].test_case_chromosomes)

    def test_case_to_string(self, test_case: TestCase) -> str:
        statements_count = len(test_case.statements)
        self._logger.info(f"Anzahl TestCaseStatements: {statements_count}")

        statements = [str(statement) for statement in test_case.statements]

        statements_string = ", ".join(statements)

        return f"TestCase mit {statements_count} Statements: [{statements_string}]"
