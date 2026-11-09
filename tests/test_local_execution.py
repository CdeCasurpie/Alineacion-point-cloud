"""Resource admission and publication behavior of the explicit local backend."""
import json
import os
from pathlib import Path

import pytest

from blockalign import artifacts


@pytest.fixture
def local_allocation(tmp_path, monkeypatch):
    monkeypatch.delenv('SLURM_JOB_ID', raising=False)
    context = tmp_path / 'context.json'
    context.write_text(json.dumps({'backend': 'local', 'pid': os.getpid()}))
    monkeypatch.setenv('BLOCKALIGN_LOCAL_CONTEXT', str(context))
    proc = tmp_path / 'proc-cgroup'
    proc.write_text('0::/test.scope\n')
    group = tmp_path / 'cgroups/test.scope'
    group.mkdir(parents=True)
    for name, value in {'memory.max': str(40 * 1024**3), 'memory.swap.max': '0', 'cpu.max': '800000 100000'}.items():
        (group / name).write_text(value)

    def path(value):
        if str(value) == '/proc/self/cgroup':
            return proc
        if str(value) == '/sys/fs/cgroup':
            return tmp_path / 'cgroups'
        return Path(value)

    monkeypatch.setattr(artifacts, 'Path', path)
    return context, group


def test_local_admission_checks_real_resource_limits(local_allocation):
    artifacts.require_job()


@pytest.mark.parametrize('name,value', [
    ('memory.max', 'max'),
    ('memory.max', str(41 * 1024**3)),
    ('memory.swap.max', '1'),
    ('cpu.max', 'max 100000'),
    ('cpu.max', '900000 100000'),
])
def test_local_admission_refuses_unbounded_or_excessive_resources(local_allocation, name, value):
    _, group = local_allocation
    (group / name).write_text(value)
    with pytest.raises(RuntimeError, match='resource-bounded'):
        artifacts.require_job()


def test_local_context_is_bound_to_its_process(local_allocation):
    context, _ = local_allocation
    context.write_text(json.dumps({'backend': 'local', 'pid': os.getpid() + 1}))
    with pytest.raises(RuntimeError, match='resource-bounded'):
        artifacts.require_job()


def test_missing_allocation_is_rejected_and_slurm_remains_supported(monkeypatch):
    monkeypatch.delenv('SLURM_JOB_ID', raising=False)
    monkeypatch.delenv('BLOCKALIGN_LOCAL_CONTEXT', raising=False)
    with pytest.raises(RuntimeError, match='Slurm'):
        artifacts.require_job()
    monkeypatch.setenv('SLURM_JOB_ID', '123')
    artifacts.require_job()


def test_local_publish_uses_rename_without_a_second_large_copy(tmp_path, monkeypatch):
    monkeypatch.setenv('BLOCKALIGN_LOCAL_CONTEXT', 'explicit-local-mode')
    source = tmp_path / 'scratch/result'
    source.mkdir(parents=True)
    payload = source / 'cloud.ply'
    payload.write_bytes(b'published immutable payload')
    inode = payload.stat().st_ino
    destination = tmp_path / 'exports/revision'

    def no_copy(*args, **kwargs):
        raise AssertionError('A same-filesystem local publication must not copy the payload')

    monkeypatch.setattr(artifacts.shutil, 'copytree', no_copy)
    artifacts.publish(source, destination)
    assert not source.exists()
    assert (destination / 'cloud.ply').read_bytes() == b'published immutable payload'
    assert (destination / 'cloud.ply').stat().st_ino == inode
    assert not list(destination.parent.glob('.pending-*'))


def test_nonlocal_publish_preserves_original_scratch(tmp_path, monkeypatch):
    monkeypatch.delenv('BLOCKALIGN_LOCAL_CONTEXT', raising=False)
    source = tmp_path / 'scratch'
    source.mkdir()
    (source / 'result.json').write_text('{}')
    artifacts.publish(source, tmp_path / 'published/result')
    assert (source / 'result.json').exists()
    assert (tmp_path / 'published/result/result.json').read_text() == '{}'
