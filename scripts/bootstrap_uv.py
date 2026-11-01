"""Install a fixed project-local installer from its public PyPI wheel.

Args: None; execute from the repository root.
Returns: Executable .tools/uv after checking its published SHA256.
Raises: Network, platform, checksum or filesystem errors.
Design:
    Context: A fresh Linux Python installation may not provide pip.
    Concern: Reproduction must not alter system Python or another project.
    Decision: Extract only the pinned uv binary into the project.
    Alternatives: A global pip installation or unpinned shell installer.
    Quality Attributes: Isolation, reproducibility. Trade-offs: One initial download.
Evidence:
    Internal: setup_realtime_vla.sh uses this binary.
    External: PyPI uv0.8.22 wheel metadata and digest.
    Findings: Linux x86_64 is the evaluated platform.
    Metrics: SHA256 equality. Confidence: Published artifact identity.
Constraints: No credentials, model weights or home-directory writes.
Reconsider When: Platform or installer version changes.
References: https://pypi.org/project/uv/0.8.22/.
ADR: None.
"""
import hashlib
import io
import json
from pathlib import Path
import platform
import urllib.request
import zipfile

if platform.system()!='Linux' or platform.machine() not in ('x86_64','AMD64'):
    raise RuntimeError('The evaluated target is Linux x86_64')
metadata=json.load(urllib.request.urlopen('https://pypi.org/pypi/uv/0.8.22/json',timeout=60))
wheel=next(item for item in metadata['urls'] if 'manylinux' in item['filename'] and 'x86_64' in item['filename'])
payload=urllib.request.urlopen(wheel['url'],timeout=180).read()
if hashlib.sha256(payload).hexdigest()!=wheel['digests']['sha256']:
    raise ValueError('Installer checksum mismatch')
with zipfile.ZipFile(io.BytesIO(payload)) as archive:
    name=next(name for name in archive.namelist() if name.endswith('/uv') and '/scripts/' in name)
    binary=Path('.tools/uv');binary.parent.mkdir(parents=True,exist_ok=True)
    binary.write_bytes(archive.read(name));binary.chmod(0o755)
