import os
import logging
import shutil
import concurrent.futures
from functools import partial
try:
    import yt.wrapper as yt
except ImportError:
    # Allow the file to be imported even if yt is not installed,
    # errors will occur later if YT functions are called.
    yt = None

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv('VERL_SFT_LOGGING_LEVEL', 'WARN'))

try:
    # Allow configuration via environment variable
    log_level_str = os.environ.get("VERL_TRACTO_IO_LOG_LEVEL", "INFO").upper()
    log_level = getattr(logging, log_level_str, logging.INFO)
    logger.setLevel(log_level)
    # Optionally, add a handler if none is configured upstream
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.propagate = False # Prevent duplicate messages if root logger also has handlers
    logger.info(f"tracto_io logger initialized with level {log_level_str}")
except Exception as e:
    print(f"Error setting up tracto_io logger: {e}") # Use print as logger might not work

def get_yt_client():
    logger.debug("get_yt_client: Entered function.")
    if yt is None:
        logger.error("get_yt_client: yt-wrapper is None!")
        raise ModuleNotFoundError("yt-wrapper is not installed or failed to import")

    logger.debug("get_yt_client: Getting config from env...")
    yt_client_config = yt.default_config.get_config_from_env()
    yt_client_config["proxy"]["url"] = "alfa.yt.nebius.yt"
    yt_client_config["token"] = os.environ["YT_SECURE_VAULT_USER_YT_TOKEN"]
    logger.debug(f"get_yt_client: Got config: {yt_client_config.get('proxy')}") # Log proxy

    if "remote_temp_files_directory" in yt_client_config:
        logger.debug("get_yt_client: Removing remote_temp_files_directory from config.")
        del yt_client_config["remote_temp_files_directory"]
        # print("INFO: Removed 'remote_temp_files_directory' from YT client config to use default.") # Replaced with logger
    if "remote_temp_tables_directory" in yt_client_config:
        logger.debug("get_yt_client: Removing remote_temp_tables_directory from config.")
        del yt_client_config["remote_temp_tables_directory"]
        # print("INFO: Removed 'remote_temp_tables_directory' from YT client config to use default.") # Replaced with logger

    logger.debug("get_yt_client: Creating YtClient instance...")
    yt_client = yt.YtClient(config=yt_client_config)
    logger.debug("get_yt_client: YtClient instance created successfully.")
    return yt_client

def _get_yt_basename(yt_path: str) -> str:
    """Safely gets the basename of a YT path."""
    if yt and hasattr(yt, 'ypath_basename'):
        try:
            return yt.ypath_basename(yt_path)
        except Exception as e:
            logger.warning(f"Call to yt.ypath_basename('{yt_path}') failed: {e}. Falling back to manual basename.")
            # Fall through to manual implementation
    # Manual implementation or fallback
    # Remove trailing slashes, then split by the last slash
    cleaned_path = yt_path.rstrip('/')
    if '/' not in cleaned_path:
         # Handle root-level paths like '//' or '//node'
         return cleaned_path.split(':')[-1] # Handle potential cluster prefix like 'hahn:'
    return cleaned_path.split('/')[-1]

def _get_yt_dirname(yt_path: str) -> str:
    """Safely gets the dirname of a YT path."""
    if yt and hasattr(yt, 'ypath_dirname'):
        try:
            return yt.ypath_dirname(yt_path)
        except Exception as e:
            logger.warning(f"Call to yt.ypath_dirname('{yt_path}') failed: {e}. Falling back to manual dirname.")
            # Fall through to manual implementation
    # Manual implementation or fallback
    cleaned_path = yt_path.rstrip('/')
    if '/' not in cleaned_path or cleaned_path.count('/') <= 1 : # Handle '//' or '//path'
        return "//"
    return '/'.join(cleaned_path.split('/')[:-1])

def copy_to_remote(src: str, dest: str, dirs_exist_ok: bool = False):
    """
    Copies a local file or directory (src) to YTsaurus (dest).

    Args:
        src (str): The local source path.
        dest (str): The YTsaurus destination path (must start with //).
        dirs_exist_ok (bool): If False (default), an error is raised if the
                              destination directory structure cannot be created
                              (e.g., a component exists and is not a directory).
                              If True, existing directories will be treated like
                              makedirs(exist_ok=True).
    """
    if not dest.startswith("//"):
        raise ValueError(f"Destination path '{dest}' is not a YTsaurus path (must start with //)")
    if not os.path.exists(src):
        raise FileNotFoundError(f"Local source path '{src}' not found.")

    ytc = get_yt_client()
    dest_exists = exists(dest) # Use tracto_io.exists

    # Determine the final destination path
    final_dest = dest
    src_is_dir = os.path.isdir(src)

    if dest_exists and is_directory(dest): # Use tracto_io.is_directory
        # If dest exists and is a directory, copy src *into* it
        final_dest = yt.ypath_join(dest, _get_yt_basename(src))
        # Ensure the base directory exists on YT
        makedirs(dest, exist_ok=True) # Ensure the parent dest dir exists
    elif dest_exists and not is_directory(dest):
         raise FileExistsError(f"Destination YT path '{dest}' exists and is not a directory.")
    elif not dest_exists:
         # Ensure parent path exists if copying *as* dest
         parent_dest = _get_yt_dirname(final_dest)
         if parent_dest and parent_dest != "//":
             makedirs(parent_dest, exist_ok=True) # Use tracto_io.makedirs

    # If src is a directory, ensure the top-level destination is created
    if src_is_dir:
        makedirs(final_dest, exist_ok=dirs_exist_ok) # Use tracto_io.makedirs

    def _recursive_upload(local_path, yt_base_path):
        logger.info(f"_recursive_upload: Processing Local '{local_path}' -> YT '{yt_base_path}'")
        if os.path.isdir(local_path):
            logger.debug(f"_recursive_upload: '{local_path}' is a directory. Ensuring YT dir '{yt_base_path}' exists.")
            # Create corresponding YT directory (map_node) if needed
            # Note: makedirs handles parent creation and existence checks based on dirs_exist_ok
            try:
                makedirs(yt_base_path, exist_ok=dirs_exist_ok) # Use tracto_io.makedirs
            except Exception as e:
                 logger.error(f"_recursive_upload: Failed during makedirs for '{yt_base_path}'. Error: {e}", exc_info=True)
                 # Decide whether to continue or raise. Raising might be safer.
                 raise

            logger.debug(f"_recursive_upload: Listing items in local directory '{local_path}'")
            for item in os.listdir(local_path):
                item_local_path = os.path.join(local_path, item)
                # Ensure consistent yt import for path join
                try:
                    if yt is None: raise ImportError("yt is None")
                    item_yt_path = yt.ypath_join(yt_base_path, item)
                except (ImportError, AttributeError) as e:
                    logger.error(f"_recursive_upload: Failed yt.ypath_join('{yt_base_path}', '{item}'). Error: {e}")
                    raise
                _recursive_upload(item_local_path, item_yt_path)
        else: # It's a file
            logger.info(f"_recursive_upload: '{local_path}' is a file. Uploading to YT '{yt_base_path}'")
            try:
                # Ensure parent YT directory exists
                # Ensure consistent yt import for dirname
                try:
                     if yt is None: raise ImportError("yt is None")
                     # Use the safe dirname getter here
                     parent_yt_path = _get_yt_dirname(yt_base_path)
                except (ImportError, AttributeError) as e:
                     logger.error(f"_recursive_upload: Failed yt.ypath_dirname('{yt_base_path}'). Error: {e}")
                     raise

                if parent_yt_path and parent_yt_path != "//":
                    logger.debug(f"_recursive_upload: Ensuring parent YT dir '{parent_yt_path}' exists for file.")
                    try:
                        makedirs(parent_yt_path, exist_ok=True) # Use tracto_io.makedirs
                    except Exception as e:
                        logger.error(f"_recursive_upload: Failed during parent makedirs for '{parent_yt_path}'. Error: {e}", exc_info=True)
                        raise

                logger.debug(f"_recursive_upload: Opening local file '{local_path}' for reading.")
                with open(local_path, "rb") as f:
                    logger.debug(f"_recursive_upload: Calling ytc.write_file('{yt_base_path}', ...)")
                    ytc = get_yt_client() # Ensure we have a client instance
                    ytc.write_file(yt_base_path, f)
                    logger.info(f"_recursive_upload: Successfully wrote YT file '{yt_base_path}'")
            except IOError as e:
                logger.error(f"_recursive_upload: Error reading local file '{local_path}'. Error: {e}", exc_info=True)
                raise
            except yt.YtError as e:
                logger.error(f"_recursive_upload: Error writing YT file '{yt_base_path}'. Error: {e}", exc_info=True)
                raise
            except Exception as e: # Catch any other unexpected errors
                 logger.error(f"_recursive_upload: Unexpected error writing file '{local_path}' to '{yt_base_path}'. Error: {e}", exc_info=True)
                 raise

    _recursive_upload(src, final_dest)
    logger.info(f"Successfully finished copy_to_remote call for Local:{src} to YT:{final_dest}")

def exists(path: str) -> bool:
    """Checks if a path exists in YTsaurus."""
    logger.debug(f"exists: Checking path '{path}'")
    if not path.startswith("//"):
        logger.debug(f"exists: Path '{path}' is local, using os.path.exists.")
        return os.path.exists(path)
    try:
        logger.debug(f"exists: Path '{path}' is YT. Attempting to get YT client...")
        ytc = get_yt_client()
        logger.debug(f"exists: Successfully got YT client for path '{path}'.")
        logger.debug(f"exists: Calling ytc.exists('{path}')...")
        result = ytc.exists(path)
        logger.debug(f"exists: ytc.exists('{path}') returned: {result}")
        return result

    except Exception as e:
        logger.warning(f"exists: Error checking existence of YT path '{path}': {e}", exc_info=True)
        return False

def is_directory(path: str) -> bool:
    """Checks if a YTsaurus path is a directory (map_node)."""
    if not path.startswith("//"):
        # Or handle local paths
        return os.path.isdir(path)
    try:
        ytc = get_yt_client()
        # Check if it exists first to avoid errors on non-existent paths
        if not ytc.exists(path):
            return False
        # Use get to check the type attribute
        return ytc.get(path + "/@type") == "map_node"
    except Exception as e:
        # Handle cases like permission denied or other YT errors
        logger.warning(f"Error checking type of YT path {path}: {e}")
        return False

def makedirs(name: str, exist_ok: bool = False):
    """Creates a directory (map_node) in YTsaurus, including parents."""
    if not name.startswith("//"):
        logger.debug(f"makedirs: Path '{name}' is local, using os.makedirs.")
        os.makedirs(name, exist_ok=exist_ok)
        return

    logger.info(f"makedirs: Attempting to create YT directory '{name}' (exist_ok={exist_ok})")
    try:
        ytc = get_yt_client()
        # Check if it already exists and is a directory *before* creating if exist_ok is False
        if not exist_ok and ytc.exists(name):
             # Check type only if it exists and exist_ok is False
             node_type = ytc.get(name + "/@type")
             if node_type == "map_node":
                 # Already exists as a directory, raise error as exist_ok is False
                 raise FileExistsError(f"YTsaurus directory '{name}' already exists.")
             else:
                 # Exists but is not a directory
                 raise FileExistsError(f"YTsaurus path '{name}' exists but is not a directory (type: {node_type}).")

        # recursive=True creates parent directories
        # ignore_existing=True mimics exist_ok=True behavior within the API call itself
        logger.debug(f"makedirs: Calling ytc.create('map_node', '{name}', recursive=True, ignore_existing={exist_ok})")
        ytc.create("map_node", name, recursive=True, ignore_existing=exist_ok)
        logger.info(f"makedirs: Successfully ensured YT directory '{name}' exists.")
    except yt.YtError as e:
        # Catch specific YT errors if needed
        # ignore_existing=True should prevent "already exists" errors, but let's log just in case
        if "already exists" in str(e) and exist_ok:
            logger.warning(f"makedirs: YT directory '{name}' already existed (caught YtError despite ignore_existing=True). Error: {e}")
            # Consider verifying it's actually a map_node here if needed
            pass # Suppress error if exist_ok is True
        else:
            logger.error(f"makedirs: Failed to create YT directory '{name}'. Error: {e}", exc_info=True)
            raise # Re-raise the exception if it's unexpected or exist_ok is False

def _download_yt_file(yt_src_path: str, local_dest_path: str, client_config: dict):
    """Downloads a single file from YT to local, creating parent dirs."""
    try:
        local_parent_dir = os.path.dirname(local_dest_path)
        if local_parent_dir:
            os.makedirs(local_parent_dir, exist_ok=True)
        ytc = get_yt_client() # Assuming get_yt_client() is okay for threads
        logger.debug(f"Thread downloading YT='{yt_src_path}' -> Local='{local_dest_path}'")
        with open(local_dest_path, "wb") as local_file_handle:
            yt_file_stream = ytc.read_file(yt_src_path)
            shutil.copyfileobj(yt_file_stream, local_file_handle)
        logger.debug(f"Thread finished download YT='{yt_src_path}' -> Local='{local_dest_path}'")
        return None # Indicate success
    except Exception as e:
        logger.error(f"Thread FAILED download YT='{yt_src_path}' -> Local='{local_dest_path}': {e}", exc_info=True)
        return e

def copy_to_local(src: str, dest: str, dirs_exist_ok: bool = False, max_parallel_downloads: int = 32):
    """
    Copies a file or directory from YTsaurus (src) to the local filesystem (dest).
    Uses a ThreadPoolExecutor for parallel downloads if source is a directory.

    Args:
        src (str): The YTsaurus path (must start with //).
        dest (str): The local destination path.
        dirs_exist_ok (bool): Controls behavior if destination dirs exist.
        max_parallel_downloads (int): Max number of concurrent download threads.
    """
    if yt is None:
         raise ModuleNotFoundError("yt-wrapper is not installed or failed to import, cannot perform YTsaurus operations.")

    if not src.startswith("//"):
        raise ValueError(f"Source path '{src}' is not a YTsaurus path (must start with //)")

    # --- Configuration for max workers ---
    try:
        env_workers = os.environ.get("VERL_TRACTO_MAX_DOWNLOAD_WORKERS")
        if env_workers is not None:
            max_parallel_downloads = int(env_workers)
            # Log only once maybe?
            # if rank == 0: logger.info(f"Overriding max_parallel_downloads with environment variable VERL_TRACTO_MAX_DOWNLOAD_WORKERS={max_parallel_downloads}")
        elif max_parallel_downloads != 32: # Log if different from default
             logger.info(f"Using specified max_parallel_downloads={max_parallel_downloads}")
    except ValueError:
        logger.warning(f"Invalid value for VERL_TRACTO_MAX_DOWNLOAD_WORKERS: '{env_workers}'. Using default/passed value: {max_parallel_downloads}")
    # --- End Configuration ---

    logger.debug(f"copy_to_local: Starting YT copy. Source='{src}', Dest='{dest}', DirsExistOk={dirs_exist_ok}, Parallel={max_parallel_downloads}")

    # --- Destination path checks ---
    if os.path.exists(dest) and not os.path.isdir(dest):
         logger.error(f"copy_to_local: Destination path '{dest}' exists and is not a directory.")
         raise FileExistsError(f"Destination path '{dest}' exists and is not a directory.")
    if os.path.exists(dest) and os.path.isdir(dest) and os.listdir(dest) and not dirs_exist_ok:
        logger.error(f"copy_to_local: Destination directory '{dest}' exists and is not empty (dirs_exist_ok=False).")
        raise FileExistsError(f"Destination directory '{dest}' exists and is not empty.")
    # --- End Destination checks ---

    ytc = get_yt_client()

    # --- Source path checks ---
    logger.debug(f"copy_to_local: Checking existence of YT source path '{src}'")
    if not exists(src): # Use the exists from this module
        logger.error(f"copy_to_local: YTsaurus source path '{src}' not found.")
        raise FileNotFoundError(f"YTsaurus source path '{src}' not found.")

    logger.debug(f"copy_to_local: Checking if YT source path '{src}' is a directory.")
    src_is_dir = is_directory(src) # Use the is_directory from this module
    logger.debug(f"copy_to_local: YT source path '{src}' is_directory={src_is_dir}")
    # --- End Source path checks ---

    # --- Determine final destination path and create local base directories ---
    final_dest_base = dest
    try:
        if src_is_dir:
            if os.path.isdir(dest):
                 final_dest_base = os.path.join(dest, _get_yt_basename(src))
            else:
                 final_dest_base = dest
            logger.debug(f"copy_to_local [Dir]: Ensuring local base dir exists: '{final_dest_base}' (exist_ok={dirs_exist_ok})")
            os.makedirs(final_dest_base, exist_ok=dirs_exist_ok)
        else: # Source is a file
            if os.path.isdir(dest):
                 final_dest_base = os.path.join(dest, _get_yt_basename(src))
                 logger.debug(f"copy_to_local [File->Dir]: Final file destination: '{final_dest_base}'")
                 os.makedirs(dest, exist_ok=True)
            else:
                 final_dest_base = dest
                 logger.debug(f"copy_to_local [File->File]: Final file destination: '{final_dest_base}'")
                 local_parent_dir = os.path.dirname(final_dest_base)
                 if local_parent_dir:
                      logger.debug(f"copy_to_local [File->File]: Ensuring local parent dir exists: '{local_parent_dir}'")
                      os.makedirs(local_parent_dir, exist_ok=True)
    except Exception as e:
        logger.error(f"copy_to_local: Failed during local destination setup for '{final_dest_base}': {e}", exc_info=True)
        raise
    # --- End Destination setup ---

    if not src_is_dir:
        # --- Simple case: Download a single file ---
        logger.info(f"copy_to_local: Downloading single file YT='{src}' -> Local='{final_dest_base}'")
        result = _download_yt_file(src, final_dest_base, ytc.config)
        if isinstance(result, Exception):
             raise RuntimeError(f"Failed to download single file '{src}'") from result
        logger.info(f"copy_to_local: Successfully finished single file download.")
        # --- End Single file case ---
    else:
        # --- Directory case: List all files recursively first ---
        logger.info(f"copy_to_local: Source is directory. Listing files recursively from YT='{src}'...")
        files_to_download = [] # *** THIS IS WHERE IT'S DEFINED ***

        # *** HELPER FUNCTION FOR RECURSIVE LISTING ***
        def _list_recursive(yt_dir, local_base):
            # This function populates the outer 'files_to_download' list
            try:
                logger.debug(f"_list_recursive: Listing children in YT path: '{yt_dir}'")
                children = ytc.list(yt_dir)
                logger.debug(f"_list_recursive: Found children in '{yt_dir}': {children}")
                for child in children:
                    child_yt_path = yt.ypath_join(yt_dir, child)
                    child_local_path = os.path.join(local_base, child)
                    logger.debug(f"_list_recursive: Checking child '{child_yt_path}' type...")
                    if is_directory(child_yt_path): # Use is_directory from this module
                        logger.debug(f"_list_recursive: Child '{child}' is dir. Creating local dir '{child_local_path}' and recursing.")
                        # Ensure corresponding local subdir exists
                        os.makedirs(child_local_path, exist_ok=True) # Use exist_ok=True here
                        _list_recursive(child_yt_path, child_local_path) # Recurse
                    else:
                        logger.debug(f"_list_recursive: Child '{child}' is file. Adding to download list.")
                        files_to_download.append((child_yt_path, child_local_path)) # Add file to list
            except Exception as e:
                 logger.error(f"Failed during recursive list of YT path '{yt_dir}': {e}", exc_info=True)
                 raise # Propagate error during listing
        # *** END HELPER FUNCTION ***

        # *** CALL THE RECURSIVE LISTING FUNCTION ***
        _list_recursive(src, final_dest_base)
        # *** Now 'files_to_download' is populated ***

        logger.info(f"copy_to_local: Found {len(files_to_download)} files to download in parallel (max_workers={max_parallel_downloads}).")

        if not files_to_download:
             logger.warning(f"copy_to_local: No files found within YT directory '{src}'. Download complete (empty).")
             return # Nothing to download

        # --- Use ThreadPoolExecutor for parallel download ---
        exceptions = []
        logger.info(f"copy_to_local: Starting parallel download of {len(files_to_download)} files...")
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_parallel_downloads) as executor:
            download_func = partial(_download_yt_file, client_config=ytc.config)
            future_to_ytpath = {executor.submit(download_func, yt_path, local_path): yt_path for yt_path, local_path in files_to_download}

            processed_count = 0
            for future in concurrent.futures.as_completed(future_to_ytpath):
                processed_count += 1
                yt_path = future_to_ytpath[future]
                try:
                    result = future.result()
                    if result is not None:
                        exceptions.append(result)
                        logger.warning(f"Download failed for YT='{yt_path}' (error collected).")
                    # Optional: Add progress logging
                    if processed_count % 100 == 0 or processed_count == len(files_to_download):
                         logger.info(f"Downloaded {processed_count}/{len(files_to_download)} files...")
                except Exception as exc:
                    exceptions.append(exc)
                    logger.error(f"Unexpected error processing download future for YT='{yt_path}': {exc}", exc_info=True)

        # --- Check for errors after parallel download ---
        if exceptions:
            logger.error(f"copy_to_local: Encountered {len(exceptions)} errors during parallel download.")
            for i, ex in enumerate(exceptions[:5]): # Log first 5 errors
                logger.error(f"  Error {i+1}: {ex}")
            raise RuntimeError(f"Parallel download failed for one or more files from source '{src}'. See logs for details.")
        else:
            logger.info(f"copy_to_local: Successfully finished parallel download of {len(files_to_download)} files from YT='{src}' to Local='{final_dest_base}'.")
        # --- End Directory case ---

def copy(src: str, dst: str, dirs_exist_ok: bool = False, **kwargs):
    """
    Copies files or directories between local filesystem and YTsaurus, or within YTsaurus.

    Detects path types based on prefixes ('//' for YTsaurus, others are local).

    Args:
        src (str): Source path (local or YTsaurus).
        dst (str): Destination path (local or YTsaurus).
        dirs_exist_ok (bool): Passed to underlying copy functions. Controls behavior
                              when destination directories exist.
        **kwargs: Additional keyword arguments (currently unused, for future compatibility).
    """
    if yt is None and (src.startswith("//") or dst.startswith("//")):
         raise ModuleNotFoundError("yt-wrapper is not installed or failed to import, cannot perform YTsaurus operations.")

    src_is_tracto = src.startswith("//")
    dst_is_tracto = dst.startswith("//")
    max_parallel = kwargs.get('max_parallel_downloads', 32)

    if src_is_tracto and not dst_is_tracto:
        logger.info(f"copy [tracto->local]: Routing src='{src}' -> dst='{dst}' (Parallel={max_parallel})")
        copy_to_local(src, dst, dirs_exist_ok=dirs_exist_ok, max_parallel_downloads=max_parallel)
    elif not src_is_tracto and dst_is_tracto:
        logger.info(f"copy [local->tracto]: Routing src='{src}' -> dst='{dst}'")
        copy_to_remote(src, dst, dirs_exist_ok=dirs_exist_ok)
    elif src_is_tracto and dst_is_tracto:
        logger.info(f"copy [tracto->tracto]: Routing src='{src}' -> dst='{dst}'")
        ytc = get_yt_client()
        try:
            logger.debug(f"copy [tracto->tracto]: Calling ytc.copy(recursive=True, force=True)")
            ytc.copy(src, dst, recursive=True, force=True)
            logger.info(f"copy [tracto->tracto]: ytc.copy finished successfully.")
        except yt.YtError as e:
            logger.error(f"copy [tracto->tracto]: YTsaurus copy failed: src='{src}' -> dst='{dst}'. Error: {e}", exc_info=True)
            raise
    else: # Both are local paths
        logger.info(f"copy [local->local]: Routing src='{src}' -> dst='{dst}'")
        try:
            if os.path.isdir(src):
                logger.debug(f"copy [local->local]: Source is directory, using shutil.copytree")
                shutil.copytree(src, dst, dirs_exist_ok=dirs_exist_ok)
            else:
                logger.debug(f"copy [local->local]: Source is file, ensuring parent '{os.path.dirname(dst)}' exists and using shutil.copy2")
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                shutil.copy2(src, dst) # copy2 preserves more metadata
            logger.info(f"copy [local->local]: Local copy finished successfully.")
        except Exception as e:
            logger.error(f"copy [local->local]: Local copy failed: src='{src}' -> dst='{dst}'. Error: {e}", exc_info=True)
            raise

def listdir(path: str) -> list[str]:
    """Lists the contents of a directory in YTsaurus."""
    logger.debug(f"listdir: Listing YT path '{path}'")
    if not path.startswith("//"):
        logger.warning(f"listdir: Path '{path}' is not a YT path. Returning empty list.") # Or raise error?
        return [] # Or call os.listdir if we want unified behavior

    try:
        logger.debug(f"listdir: Getting YT client for '{path}'...")
        ytc = get_yt_client()
        logger.debug(f"listdir: Got YT client. Calling ytc.list('{path}')...")
        contents = ytc.list(path)
        logger.debug(f"listdir: ytc.list('{path}') returned {len(contents)} items.")
        return contents
    except yt.YtResponseError as e:
        # Handle directory not existing gracefully, similar to os.listdir raising FileNotFoundError
        if "resolve" in str(e).lower() or "not found" in str(e).lower():
             logger.warning(f"listdir: YT path '{path}' not found. Error: {e}")
             raise FileNotFoundError(f"YTsaurus path not found: {path}") from e
        else:
             logger.error(f"listdir: Error listing YT path '{path}': {e}", exc_info=True)
             raise # Re-raise other YT errors
    except Exception as e:
        logger.error(f"listdir: Unexpected error listing YT path '{path}': {e}", exc_info=True)
        raise

# Add listdir to __all__ if it exists at the top
# __all__ = [..., "listdir"]
