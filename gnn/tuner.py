from typing import Dict, Iterable, Set, Any
from collections import defaultdict
from pathlib import Path
import hashlib
import json
import base64

from ax import RangeParameter, ParameterType, Objective, Experiment, SearchSpace, OptimizationConfig
from ax.service.ax_client import AxClient, ObjectiveProperties
from ax.service.scheduler import Scheduler, SchedulerOptions
from ax.modelbridge.dispatch_utils import choose_generation_strategy
from ax.core.runner import Runner
from ax.core.base_trial import BaseTrial, TrialStatus
from ax.core.metric import Metric, MetricFetchResult, MetricFetchE
from ax.utils.common.result import Ok, Err

import htcondor


def create_parser():
    """Command line argument parser."""
    from argparse import ArgumentParser
    parser = ArgumentParser('Net training and tuning with Condor and Ax')

    return parser


staus_mapping = {
    1: TrialStatus.RUNNING, # Idle
    2: TrialStatus.RUNNING, # Running
    #3 - Removed
    #4 - Completed
    #5 - Held
    #6 - Transferring output
    #7 - Suspended
}


def hash_params(in_dict, length = None):
    """Hash of a dictionary.
    Adapted from https://github.com/Reed-CompBio/spras/blob/master/spras/util.py
    """
    the_hash = hashlib.sha1()
    dict_encoded = json.dumps(in_dict, sort_keys=True).encode()
    the_hash.update(dict_encoded)
    result = base64.b32encode(the_hash.digest()).decode('ascii')
    if length is None or length < 1 or length > len(result):
        return result
    else:
        return result[:length]


class CondorJobRunner(Runner):
    """Ax Trial Runner for HTCondor, based on the tutorials available at ax.dev"""


    def __init__(
        self,
        container_image_path: str,
        training_script_path: Path,
        data_path: Path,
        logs_path: Path,
        inputs_path: Path,
        outputs_path: Path
    ):
        self.container_image_path = container_image_path
        self.training_script_path = training_script_path
        self.data_path = data_path
        self.logs_path = logs_path
        self.inputs_path = inputs_path
        self.outputs_path = outputs_path
        self.schedd = htcondor.Schedd()

        # Check that training script and data file exist
        assert self.training_script_path.exists()
        assert self.data_path.exists()

        # Ensure log and input dirs exist
        self.logs_path.parent.mkdir(parents=True, exist_ok=True)
        self.inputs_path.mkdir(parents=True, exist_ok=True)


    def run(self, trial: BaseTrial):
        """Deploys a trial using HTCondor.
        
        Args:
            trial: The trial to deploy.
            
        Returns:
            Dict of run metadata from the deployment process.
        """

        parameters = trial.arm.parameters

        param_hash = hash_params(parameters)

        input_path = self.inputs_path / f'{param_hash}.params'

        output_path = self.outputs_path / param_hash

        # Ensure output dir exists
        output_path.mkdir(parents=True, exist_ok=True)

        # Prep input file
        with input_path.open('wt') as out_handle:
            json.dump(parameters, out_handle, sort_keys=True)

        job_arguments = (
            f'--data {self.data_path.name} '
            '--model model.pt'
            '--loss-curves-csv loss.csv'
            f'--hyperparameters {self.inputs_path.name}'
        )

        # Schedule job here
        condor_job = htcondor.Submit({
            'universe': 'container',
            'container_image': self.container_image_path,
            'executable': 'python', # Or use container's runscript?
            'arguments': job_arguments,
            'log': self.logs_path.as_posix(),
            'out': (output_path / 'job.out').as_posix(),
            'err': (output_path / 'job.err').as_posix(),
            '+WantFlocking': 'true',
            '+WantGlideIn': 'true',
            '+WantGPULab': 'true',
            '+GPUJobLength': 'short',
            'transfer_input_files': ','.join(
                input_path.as_posix(),
                self.training_script_path.as_posix(),
                self.data_path.as_posix()
            ),
            # no transfer_output_files
            'request_cpus': '1',
            'request_gpus': '1',
            'request_memory': '24GB',
            'request_disk': '24GB',
            'require_gpus': '(GlobalMemoryMb >= 40000)' 
        })

        submitted = self.schedd.submit(condor_job)

        # Record the cluster ID.
        # This run metadata will be attached to trial as `trial.run_metadata`
        # by the base `Scheduler`.
        return {'cluster_id': submitted.cluster()}

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

        projection = ['JobStatus']
        
        status_dict = defaultdict(set)
        for trial in trials:
            constraint = f'ClusterID == {trial.x}'
            jobs = self.schedd.query(constraint=constraint, projection=projection)
            if not jobs:
                jobs = self.schedd.history(constraint=constraint, projection=projection)
            for job in jobs:
                status_dict[status_mapping[job['JobStatus']]].add(trial.index)
        
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