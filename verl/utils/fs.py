#!/usr/bin/env python
# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# -*- coding: utf-8 -*-
"""File-system agnostic IO APIs"""
import os
import tempfile
import hashlib
import shutil

try:
    from hdfs_io import copy as hdfs_copy, makedirs as hdfs_makedirs, exists as hdfs_exists  # for internal use only
except ImportError:
    from .hdfs_io import copy as hdfs_copy, makedirs as hdfs_makedirs, exists as hdfs_exists

try:
    from tracto_io import copy as tracto_copy, makedirs as tracto_makedirs, exists as tracto_exists  # Import tracto IO functions
except ImportError:
    from .tracto_io import copy as tracto_copy, makedirs as tracto_makedirs, exists as tracto_exists

__all__ = ["copy", "exists", "makedirs"]

_HDFS_PREFIX = "hdfs://"
_TRACTO_PREFIX = "//"


def is_non_local(path):
    return path.startswith(_HDFS_PREFIX) or path.startswith(_TRACTO_PREFIX)

def is_tracto(path):
    return path.startswith(_TRACTO_PREFIX)

def is_hdfs(path):
    return path.startswith(_HDFS_PREFIX)

def md5_encode(path: str) -> str:
    return hashlib.md5(path.encode()).hexdigest()


def get_local_temp_path(remote_path: str, cache_dir: str) -> str:
    """Return a local temp path that joins cache_dir and basename of remote_path

    Args:
        remote_path: Path on remote storage (HDFS or YTsaurus)
        cache_dir: Local cache directory

    Returns:
        Local path for caching remote file
    """
    # make a base64 encoding of remote_path to avoid directory conflict
    encoded_remote_path = md5_encode(remote_path)
    temp_dir = os.path.join(cache_dir, encoded_remote_path)
    os.makedirs(temp_dir, exist_ok=True)
    dst = os.path.join(temp_dir, os.path.basename(remote_path))
    return dst


def copy_to_local(src: str, cache_dir=None, filelock='.file.lock', verbose=False) -> str:
    """Copy src from remote storage to local if src is on remote storage or directly return src.
    If cache_dir is None, we will use the default cache dir of the system.

    Args:
        src (str): a HDFS path, YTsaurus path, or local path
        cache_dir: Local directory to store downloaded files
        filelock: Lock file name
        verbose: Whether to print verbose logs

    Returns:
        a local path of the copied file
    """
    if is_hdfs(src):
        return copy_local_path_from_hdfs(src, cache_dir, filelock, verbose)
    elif is_tracto(src):
        return copy_local_path_from_tracto(src, cache_dir, filelock, verbose)
    return src

def copy_local_path_from_hdfs(src: str, cache_dir=None, filelock='.file.lock', verbose=False) -> str:
    """Copy a file from HDFS to local."""
    from filelock import FileLock

    assert src[-1] != '/', f'Make sure the last char in src is not / because it will cause error. Got {src}'

    if is_hdfs(src):
        # download from hdfs to local
        if cache_dir is None:
            # get a temp folder
            cache_dir = tempfile.gettempdir()
        os.makedirs(cache_dir, exist_ok=True)
        assert os.path.exists(cache_dir)
        local_path = get_local_temp_path(src, cache_dir)
        # get a specific lock
        filelock = md5_encode(src) + '.lock'
        lock_file = os.path.join(cache_dir, filelock)
        with FileLock(lock_file=lock_file):
            if not os.path.exists(local_path):
                if verbose:
                    print(f'Copy from {src} to {local_path}')
                hdfs_copy(src, local_path)
        return local_path
    else:
        return src


def copy_local_path_from_tracto(src: str, cache_dir=None, filelock=".file.lock", verbose=False):
    """Copy a file from YTsaurus to local using tracto_io functions."""
    from filelock import FileLock

    assert src[-1] != '/', f'Make sure the last char in src is not / because it will cause error. Got {src}'

    if is_tracto(src):
        # download from tracto to local
        if cache_dir is None:
            # get a temp folder
            cache_dir = tempfile.gettempdir()
        os.makedirs(cache_dir, exist_ok=True)
        assert os.path.exists(cache_dir)
        local_path = get_local_temp_path(src, cache_dir)
        
        # get a specific lock
        filelock = md5_encode(src) + '.lock'
        lock_file = os.path.join(cache_dir, filelock)
        with FileLock(lock_file=lock_file):
            if not os.path.exists(local_path) or os.path.getsize(local_path) == 0:
                if verbose:
                    print(f'Copy from {src} to {local_path}')
                # Use tracto_io.copy to properly handle YTsaurus paths
                tracto_copy(src, local_path)
        return local_path
    else:
        return src

def copy(src: str, dst: str, **kwargs) -> bool:
    """Copy files between local, HDFS and YTsaurus paths.
    
    Args:
        src: Source path (local, HDFS, or YTsaurus)
        dst: Destination path (local, HDFS, or YTsaurus)
        **kwargs: Additional arguments passed to the underlying copy functions
        
    Returns:
        True if successful, False otherwise
    """
    if is_hdfs(src) or is_hdfs(dst):
        return hdfs_copy(src, dst, **kwargs)
    elif is_tracto(src) or is_tracto(dst):
        return tracto_copy(src, dst, **kwargs)
    else:
        if os.path.isdir(src):
            shutil.copytree(src, dst, **kwargs)
            return True
        else:
            shutil.copy(src, dst, **kwargs)
            return True

def exists(path: str) -> bool:
    """Check if a path exists (works with local, HDFS and YTsaurus paths).
    
    Args:
        path: Path to check
        
    Returns:
        True if the path exists, False otherwise
    """
    if is_hdfs(path):
        return hdfs_exists(path)
    elif is_tracto(path):
        return tracto_exists(path)
    else:
        return os.path.exists(path)

def makedirs(path: str, mode=0o777, exist_ok=False, **kwargs) -> None:
    """Create directories recursively (works with local, HDFS and YTsaurus paths).
    
    Args:
        path: Directory path to create
        mode: Directory creation mode (for local paths)
        exist_ok: If True, don't raise error if directory exists
        **kwargs: Additional arguments for remote storage
    """
    if is_hdfs(path):
        hdfs_makedirs(path, **kwargs)
    elif is_tracto(path):
        tracto_makedirs(path, exist_ok=exist_ok, **kwargs)
    else:
        os.makedirs(path, mode=mode, exist_ok=exist_ok)
