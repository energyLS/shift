import calendar
import io
import logging
import os
import shutil
import subprocess
import sys
import time
import zipfile
from datetime import datetime, timedelta
from pathlib import Path

import country_converter as coco
import geopandas as gpd
import numpy as np
import pandas as pd
import requests
import yaml
# from fake_useragent import UserAgent
# from pypsa.components import component_attrs, components
from shapely.geometry import Point
from tqdm import tqdm

logger = logging.getLogger(__name__)


def mock_snakemake(
    rulename, root_dir=None, submodule_dir=None, configfile=None, **wildcards
):
    """
    This function is expected to be executed from the "scripts"-directory of "
    the snakemake project. It returns a snakemake.script.Snakemake object,
    based on the Snakefile.

    If a rule has wildcards, you have to specify them in **wildcards**.

    Parameters
    ----------
    rulename: str
        name of the rule for which the snakemake object should be generated
    configfile: str
        path to config file to be used in mock_snakemake
    wildcards:
        keyword arguments fixing the wildcards. Only necessary if wildcards are
        needed.
    """
    import os

    import snakemake as sm

    try:
        from pypsa.descriptors import Dict
    except:
        from pypsa.definitions.structures import Dict  # from pypsa version v0.31
    from snakemake.script import Snakemake

    script_dir = Path(__file__).parent.resolve()
    if root_dir is None:
        root_dir = script_dir.parent
    else:
        root_dir = Path(root_dir).resolve()

    user_in_script_dir = Path.cwd().resolve() == script_dir
    if str(submodule_dir) in __file__:
        # the submodule_dir path is only need to locate the project dir
        os.chdir(Path(__file__[: __file__.find(str(submodule_dir))]))
    elif user_in_script_dir:
        os.chdir(root_dir)
    elif Path.cwd().resolve() != root_dir:
        raise RuntimeError(
            "mock_snakemake has to be run from the repository root"
            f" {root_dir} or scripts directory {script_dir}"
        )
    try:
        for p in sm.SNAKEFILE_CHOICES:
            if os.path.exists(p):
                snakefile = p
                break

        if isinstance(configfile, str):
            with open(configfile, "r") as file:
                configfile = yaml.safe_load(file)

        workflow = sm.Workflow(
            snakefile,
            overwrite_configfiles=[],
            rerun_triggers=[],
            overwrite_config=configfile,
        )
        workflow.include(snakefile)
        workflow.global_resources = {}
        try:
            rule = workflow.get_rule(rulename)
        except Exception as exception:
            print(
                exception,
                f"The {rulename} might be a conditional rule in the Snakefile.\n"
                f"Did you enable {rulename} in the config?",
            )
            raise
        dag = sm.dag.DAG(workflow, rules=[rule])
        wc = Dict(wildcards)
        job = sm.jobs.Job(rule, dag, wc)

        def make_accessable(*ios):
            for io in ios:
                for i in range(len(io)):
                    io[i] = os.path.abspath(io[i])

        make_accessable(job.input, job.output, job.log)
        snakemake = Snakemake(
            job.input,
            job.output,
            job.params,
            job.wildcards,
            job.threads,
            job.resources,
            job.log,
            job.dag.workflow.config,
            job.rule.name,
            None,
        )
        snakemake.benchmark = job.benchmark

        # create log and output dir if not existent
        for path in list(snakemake.log) + list(snakemake.output):
            Path(path).parent.mkdir(parents=True, exist_ok=True)

    finally:
        if user_in_script_dir:
            os.chdir(script_dir)
    return snakemake


def progress_retrieve(url, file, disable=False):
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    # Hotfix - Bug, tqdm not working with disable=False
    disable = True

    if disable:
        response = requests.get(url, headers=headers, stream=True)
        with open(file, "wb") as f:
            f.write(response.content)
    else:
        response = requests.get(url, headers=headers, stream=True)
        total_size = int(response.headers.get("content-length", 0))
        chunk_size = 1024

        with tqdm(
            total=total_size,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            desc=str(file),
        ) as t:
            with open(file, "wb") as f:
                for data in response.iter_content(chunk_size=chunk_size):
                    f.write(data)
                    t.update(len(data))