from typing import Dict, Iterable, Set, Any
from collections import defaultdict
from pathlib import Path
import hashlib
import json
import yaml
import base64
import logging

import pandas as pd

from ax import RangeParameter, ParameterType, Objective, Experiment, SearchSpace, OptimizationConfig
#from ax.service.ax_client import AxClient, ObjectiveProperties
from ax.service.scheduler import Scheduler, SchedulerOptions
from ax.service.utils.instantiation import InstantiationBase
from ax.modelbridge.dispatch_utils import choose_generation_strategy
from ax.core.runner import Runner
from ax.core.base_trial import BaseTrial, TrialStatus
from ax.core.data import Data
from ax.core.metric import Metric, MetricFetchResult, MetricFetchE
from ax.utils.common.result import Ok, Err

import htcondor


logger = logging.getLogger(__name__)


def create_parser():
    """Command line argument parser."""
    from argparse import ArgumentParser
    parser = ArgumentParser('Net training and tuning with Condor and Ax')

    parser.add_argument('--data', type=Path)
    parser.add_argument('--parameter-space', type=Path)
    parser.add_argument('quiet', action='store_true')

    return parser


status_mapping = {
    1: TrialStatus.STAGED, # Idle
    2: TrialStatus.RUNNING, # Running
    3: TrialStatus.ABANDONED, # Removed
    4: TrialStatus.CANDIDATE, # Completed
    5: TrialStatus.RUNNING, # Held -- treated as still running
    6: TrialStatus.RUNNING, # Transferring output
    7: TrialStatus.RUNNING # Suspended
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
        model_path: Path,
        data_path: Path,
        logs_path: Path,
        inputs_path: Path,
        outputs_path: Path
    ):
        logger.debug('Building CondorJobRunner')
        self.container_image_path = container_image_path
        self.training_script_path = training_script_path
        self.model_path = model_path
        self.data_path = data_path
        self.logs_path = logs_path
        self.inputs_path = inputs_path
        self.outputs_path = outputs_path
        self.schedd = htcondor.Schedd()

        logger.debug(f'Using HTCondor Scheduler: {self.schedd}')

        # Check that training script and data file exist
        assert self.training_script_path.exists()
        assert self.data_path.exists()

        # Ensure log and input dirs exist
        self.logs_path.parent.mkdir(parents=True, exist_ok=True)
        self.inputs_path.mkdir(parents=True, exist_ok=True)
        logger.debug('CondorJobRunner built.')


    def run(self, trial: BaseTrial):
        """Deploys a trial using HTCondor.
        
        Args:
            trial: The trial to deploy.
            
        Returns:
            Dict of run metadata from the deployment process.
        """

        logger.debug('Running Trial')

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
            f'{self.training_script_path.name} '
            f'--data {self.data_path.name} '
            '--model-path model.pt '
            '--loss-curve-csv loss.csv '
            '--best-metrics best.json '
            f'--hyperparameters {input_path.name}'
        )

        job_output_files = ['model.pt', 'loss.csv', 'best.json']

        logger.debug('Building submit object...')

        # Schedule job here
        condor_job = htcondor.Submit({
            'universe': 'container',
            'container_image': self.container_image_path,
            #'executable': 'python', # Use container's runscript?
            'arguments': job_arguments,
            'log': self.logs_path.as_posix(),
            'output': (output_path / 'job.out').as_posix(),
            'error': (output_path / 'job.err').as_posix(),
            '+WantFlocking': 'true',
            '+WantGlideIn': 'true',
            '+WantGPULab': 'true',
            '+GPUJobLength': '"short"',
            'transfer_input_files': ','.join((
                input_path.as_posix(),
                self.training_script_path.as_posix(),
                self.model_path.as_posix(),
                self.data_path.as_posix()
            )),
            'transfer_output_files': ','.join(job_output_files),
            'transfer_output_remaps': '"'+';'.join(
                f'{f}={(output_path / f).as_posix()}'
                for f in job_output_files
            )+'"',
            'request_cpus': '1',
            'request_gpus': '1',
            'request_memory': '24GB',
            'request_disk': '24GB',
            'require_gpus': '(Capability >= 3.7) && (GlobalMemoryMb >= 30000)' 
        })

        logger.debug(f'Submitting {condor_job}...')

        submitted = self.schedd.submit(condor_job)

        logger.debug('...submitted!')

        # Record the cluster ID.
        # This run metadata will be attached to trial as `trial.run_metadata`
        # by the base `Scheduler`.
        return {
            'cluster_id': submitted.cluster(),
            'output_metrics': output_path / 'best.json'
        }

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

        logger.debug('Polling jobs...')

        projection = ['JobStatus']
        
        status_dict = defaultdict(set)
        for trial in trials:
            constraint = f'ClusterID == {trial.run_metadata.get("cluster_id")}'
            jobs = self.schedd.query(constraint=constraint, projection=projection)
            if not jobs:
                jobs = self.schedd.history(constraint=constraint, projection=projection)
            for job in jobs:
                status_dict[status_mapping[job['JobStatus']]].add(trial.index)
        logger.debug(f'poll results: {status_dict}')
        return status_dict


class CondorJobMetric(Metric):
    """Pulls data for trial from external system"""
    def fetch_trial_data(self, trial: BaseTrial, **kwargs: Any) -> MetricFetchResult:
        logger.debug('Pulling trial result...')
        try:
            # Get AUROC
            # Read off from result file
            with trial.run_metadata.get('best_metrics') as in_handle:
                result = json.load(in_handle)
            auroc = result['best_auroc']
            df_dict = {
                "trial_index": trial.index,
                "metric_name": self.name,
                "arm_name": trial.arm.name,
                "mean": auroc,
                # Can be set to 0.0 if function is known to be noiseless
                # or to an actual value when SEM is known. Setting SEM to
                # `None` results in Ax assuming unknown noise and inferring
                # noise level from data.
                "sem": None,
            }
            return Ok(value=Data(df=pd.DataFrame.from_records([df_dict])))
        except Exception as e:
            return Err(MetricFetchE(message=f'Failed to fetch {self.name}', exception=e))


def get_parameters(parameters_json: Path):

    with parameters_json.open('rt') as in_handle:
        if parameters_json.suffix.lower() in {'.yml', '.yaml'}:
            parameters = yaml.safe_load(in_handle)
        else:
            parameters = json.load(in_handle)

    return [InstantiationBase.parameter_from_json(parameter) for parameter in parameters]


def main(args):

    logging.basicConfig(level=logging.DEBUG)

    logger.setLevel(logging.WARNING if args.quiet else logging.DEBUG)
    
    # TODO: Get from args
    n_jobs = 5
    job_timeout_hours = 2
    container_image_path = 'osdf:///chtc/staging/sverchkov/pyg1.sif'
    training_script_path = Path('train_nnc.py')
    model_path = Path('nets.py')
    data_path = args.data
    logs_path = Path('ax.log')
    inputs_path = Path('ax_in')
    outputs_path = Path('ax_out')

    parameters = get_parameters(args.parameter_space)

    objective = Objective(metric=CondorJobMetric(name='auroc'), minimize=False)

    experiment = Experiment(
        name='net_tuning',
        search_space=SearchSpace(parameters=parameters),
        optimization_config=OptimizationConfig(objective=objective),
        runner=CondorJobRunner(
            container_image_path=container_image_path,
            training_script_path=training_script_path,
            model_path=model_path,
            data_path=data_path,
            logs_path=logs_path,
            inputs_path=inputs_path,
            outputs_path=outputs_path
        )
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
    scheduler.run_n_trials(max_trials=n_jobs, timeout_hours=job_timeout_hours)

    # Save results
    experiment.fetch_data().df.to_csv(outputs_path/'experiment_data.csv')


if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    main(args)
