#!/bin/bash

set -eux

conda env create -f ci_dependencies.yml

conda activate mlopspython_ci