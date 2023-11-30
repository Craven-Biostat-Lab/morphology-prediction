from typing import Dict, Iterable, Set, Any
from collections import defaultdict

from ax import RangeParameter, ParameterType, Objective, Experiment, SearchSpace, OptimizationConfig
from ax.service.ax_client import AxClient, ObjectiveProperties
from ax.service.scheduler import Scheduler, SchedulerOptions
from ax.modelbridge.dispatch_utils import choose_generation_strategy
from ax.core.runner import Runner
from ax.core.base_trial import BaseTrial, TrialStatus
from ax.core.metric import Metric, MetricFetchResult, MetricFetchE
from ax.utils.common.result import Ok, Err


def create_parser():
    """Command line argument parser."""
    from argparse import ArgumentParser
    parser = ArgumentParser('Net training and tuning with Condor and Ax')

    return parser


class CondorJobRunner(Runner):
    """Ax Trial Runner for HTCondor, based on the tutorials available at ax.dev"""
    def run(self, trial: BaseTrial):
        """Deploys a trial using HTCondor.
        
        Args:
            trial: The trial to deploy.
            
        Returns:
            Dict of run metadata from the deployment process.
        """

        # Schedule job here

        # Return identifying information

    def poll_trial_status(self, trials: Iterable) -> Dict[TrialStatus, Set[int]]:
        """Checks the status of any non-terminal trials and returns their
        indices as a mapping from TrialStatus to a list of indices. Required
        for runners used with Ax ``Scheduler``.

        NOTE: Does not need to handle waiting between polling calls while trials
        are running; this function should just perform a single poll.

        Args:
            trials: Trials to poll.

        Returns:
            A dictionary mapping TrialStatus to a list of trial indices that have
            the respective status at the time of the polling. This does not need to
            include trials that at the time of polling already have a terminal
            (ABANDONED, FAILED, COMPLETED) status (but it may).
        """
        status_dict = defaultdict(set)
        for trial in trials:
            # Get status here
            status = TrialStatus.FAILED
            status_dict[status].add(trial.index)
        
        return status_dict


class CondorJobMetric(Metric):
    """Pulls data for trial from external system"""
    def fetch_trial_data(self, trial: BaseTrial, **kwargs: Any) -> MetricFetchResult:
        try:
            # Get AUROC
            raise NotImplementedError()
            return Ok()
        except Exception as e:
            return Err(MetricFetchE(message=f'Failed to fetch {self.name}', exception=e))


def train_evaluate(parametrization):
    pass

def main(args):

    # TODO: Get from args
    n_parallel_jobs = 10
    job_timeout_hours = 2

    parameters = [
        RangeParameter(
            name = 'learning_rate',
            parameter_type = ParameterType.FLOAT,
            lower = 1e-6,
            upper = 0.4
        ),
        RangeParameter(
            name='channel_width',
            parameter_type=ParameterType.INT,
            lower=1,
            upper=1000,
            log_scale=True
        ),
        RangeParameter(
            name='encoder_depth',
            parameter_type=ParameterType.INT,
            lower=1,
            upper=10
        ),
        RangeParameter(
            name='gnn_depth',
            parameter_type=ParameterType.INT,
            lower=1,
            upper=10
        ),
        RangeParameter(
            name='decoder_depth',
            parameter_type=ParameterType.INT,
            lower=1,
            upper=10
        ),
        RangeParameter(
            name='dropout_rate',
            parameter_type=ParameterType.FLOAT,
            lower=0.0,
            upper=1.0,
        )
    ]

    objective = Objective(metric=CondorJobMetric(name='auroc'), minimize=False)

    experiment = Experiment(
        name='net_tuning',
        search_space=SearchSpace(parameters=parameters),
        optimization_config=OptimizationConfig(objective=objective),
        runner=CondorJobRunner()
    )

    # Automatically choose generation strategy
    # Don't set explicit parallelism limits
    generation_strategy = choose_generation_strategy(search_space=experiment.search_space)

    scheduler = Scheduler(
        experiment=experiment,
        generation_strategy=generation_strategy,
        options=SchedulerOptions()
    )

    # For future, we need to set up optimization criteria and run_all_trials
    scheduler.run_n_trials(max_trials=n_parallel_jobs, timeout_hours=job_timeout_hours)

if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    main(args)