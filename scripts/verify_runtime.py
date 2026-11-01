"""Verify independent policy/simulator imports without loading checkpoints.

Args: None. Returns: JSON version/import report.
Raises: Import errors, version mismatches or external site-packages dependencies.
Design:
    Context: The task runtime is independent of prior experiments.
    Concern: An implicit .pth fallback can hide missing packages.
    Decision: Check core package locations and import both official policies.
    Alternatives: Reading package metadata alone.
    Quality Attributes: Verifiability. Trade-offs: Several seconds of imports.
Evidence:
    Internal: Frozen adapters use these same classes.
    External: Python sys.prefix and distribution metadata.
    Findings: Rollouts establish competence; this check establishes installation.
    Metrics: Key versions and successful imports. Confidence: Import contract.
Constraints: No network, checkpoint load, CUDA allocation or inference.
Reconsider When: The lock changes.
References: requirements-realtime.lock.
ADR: None.
"""
from importlib.metadata import version
import json
import os
from pathlib import Path
import sys
import yaml

# LIBERO otherwise prompts and writes its first configuration under a user home.
root=Path(__file__).resolve().parents[1]
package=Path(sys.prefix)/'lib/python3.12/site-packages/libero/libero'
configuration=root/'.cache/libero-import-config'
configuration.mkdir(parents=True,exist_ok=True)
(configuration/'config.yaml').write_text(yaml.safe_dump({
    'benchmark_root':str(package),'bddl_files':str(package/'bddl_files'),
    'init_states':str(package/'init_files'),'datasets':str(root/'.cache/libero-datasets'),
    'assets':str(root/'.cache/libero-assets')}))
os.environ['LIBERO_CONFIG_PATH']=str(configuration)
import torch
import torchvision
import numpy
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.pi0_fast.modeling_pi0_fast import PI0FastPolicy
from libero.libero import benchmark

expected={'torch':'2.7.1+cu128','torchvision':'0.22.1+cu128','numpy':'2.2.6',
          'transformers':'5.4.0','hf-libero':'0.1.4'}
actual={name:version(name) for name in expected}
if actual!=expected:raise RuntimeError(f'Runtime version mismatch: {actual}')
prefix=Path(sys.prefix).resolve()
for module in (torch,torchvision,numpy):
    if not Path(module.__file__).resolve().is_relative_to(prefix):
        raise RuntimeError('Core package imported from outside the active environment')
print(json.dumps({'status':'passed','python':'.'.join(map(str,sys.version_info[:3])),
                  'versions':actual,'policy_imports':['SmolVLAPolicy','PI0FastPolicy'],
                  'libero_import':bool(benchmark),'torch_cuda_build':torch.version.cuda,
                  'external_site_packages':False,'model_loaded':False}))
