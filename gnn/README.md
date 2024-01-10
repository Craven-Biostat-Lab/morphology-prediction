## Apptainer container for jobs

The most current recommendation as of 2023-12-12 is to use containers for HTCondor
jobs that train NNs.

The container definition in `container.def` can be used to build a container with
the necessary packages to run our training script.

### Building the container

The command to build the container is
```
apptainer build container.sif container.def
```

### Troubleshooting

A "No space left on device" error can happen for various reasons when building an image.

#### Lack of space in the temporary build directory

The default `/tmp` device might not have the space required.
A workaround is to set the environment variable `APPTAINER_TMPDIR` to a device with sufficient space.

#### [Apptainer issue 1076](https://github.com/apptainer/singularity/issues/1076)

A workaround is to build in sandbox first:
```
apptainer build --sandbox sandbox/ container.def
apptainer build container.sif sandbox/
```