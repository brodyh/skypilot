"""Vast | Catalog

Runtime utilities for querying Vast.ai offerings.

Historically SkyPilot relied on a pre-generated CSV catalog. This module now
builds an in-memory dataframe directly from Vast's search API on each request
so that pricing and availability reflect the latest marketplace state.
"""

import collections
import contextlib
import contextvars
import fcntl
import json
import os
import re
import time
import typing
from typing import Any, cast, Dict, List, Optional, Tuple, Union

from sky import sky_logging
from sky import skypilot_config
from sky.adaptors import common as adaptors_common
from sky.adaptors import vast
from sky.catalog import common
from sky.utils import ux_utils

if typing.TYPE_CHECKING:  # pragma: no cover
    from sky.clouds import cloud

if typing.TYPE_CHECKING:  # pragma: no cover
    import pandas as pd  # type: ignore  # pylint: disable=unused-import
else:
    pd = adaptors_common.LazyImport('pandas')

logger = sky_logging.init_logger(__name__)

_SECURE_ONLY_OVERRIDE: contextvars.ContextVar[Optional[bool]] = (
    contextvars.ContextVar('sky_vast_secure_only_override', default=None))

_CATALOG_CACHE_TTL_SECONDS = 60 * 60  # 1 hour to match SkyPilot pattern


@contextlib.contextmanager
def secure_only_override(value: Optional[bool]) -> typing.Iterator[None]:
    if value is None:
        yield
        return
    token = _SECURE_ONLY_OVERRIDE.set(bool(value))
    try:
        yield
    finally:
        _SECURE_ONLY_OVERRIDE.reset(token)


# Map some GPU names returned by Vast to SkyPilot's canonical accelerator names.
_GPU_NAME_NORMALIZATION = {
    'TeslaV100': 'V100',
    'TeslaT4': 'T4',
    'TeslaP100': 'P100',
    'QRTX6000': 'RTX6000',
    'QRTX8000': 'RTX8000',
}

_CATALOG_COLUMNS = [
    'InstanceType',
    'AcceleratorName',
    'AcceleratorCount',
    'vCPUs',
    'MemoryGiB',
    'GpuInfo',
    'Price',
    'SpotPrice',
    'Region',
]

_MAX_OFFERS_PER_LOCATION = 5


def _resolve_secure_only(region: Optional[str] = None) -> bool:
    override = _SECURE_ONLY_OVERRIDE.get()
    if override is not None:
        return bool(override)
    return bool(
        skypilot_config.get_effective_region_config(cloud='vast',
                                                    region=region,
                                                    keys=('secure_only',),
                                                    default_value=True))


def _create_instance_type(offer: Dict) -> str:
    stubify = lambda x: re.sub(r'\s', '_', x)
    # Use effective CPU cores (accounts for gpu_frac for fractional instances)
    cpu_cores = offer.get('cpu_cores_effective', offer['cpu_cores'])
    # cpu_ram is already the effective amount you get, not bare metal
    cpu_ram = int(offer['cpu_ram'])
    return '{}x-{}-{}-{}'.format(offer['num_gpus'], stubify(offer['gpu_name']),
                                 int(cpu_cores), cpu_ram)


def _canonical_gpu_name(raw_name: str) -> str:
    gpu = re.sub('Ada', '-Ada', re.sub(r'\s', '', raw_name))
    gpu = re.sub(r'(Ti|PCIE|SXM4|SXM|NVL)$', '', gpu)
    gpu = re.sub(r'(RTX\d0\d0)(S|D)$', r'\1', gpu)
    return _GPU_NAME_NORMALIZATION.get(gpu, gpu)


def _gpu_info_string(name: str, num_gpus: float, total_ram: float) -> str:
    info = {
        'Gpus': [{
            'Name': name,
            'Count': num_gpus,
            'MemoryInfo': {
                'SizeInMiB': total_ram,
            }
        }],
        'TotalGpuMemoryInMiB': total_ram,
    }
    # common.list_accelerators_impl expects a single-quoted JSON string.
    double_quote = chr(34)
    single_quote_escaped = chr(92) + chr(39)
    return json.dumps(info).replace(double_quote, single_quote_escaped)


def _maybe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _build_dataframe(offers: List[Dict]) -> 'pd.DataFrame':
    if not offers:
        return pd.DataFrame(columns=_CATALOG_COLUMNS)

    price_map: Dict[str, List[Dict]] = collections.defaultdict(list)

    for offer in offers:
        try:
            instance_type = _create_instance_type(offer)
        except (KeyError, TypeError, ValueError):
            continue

        search_info = offer.get('search', {})
        price = _maybe_float(search_info.get('totalHour'))
        if price is None:
            continue

        region = offer.get('geolocation')
        if not isinstance(region, str):
            continue

        acc_count = _maybe_float(offer.get('num_gpus'))
        # Use effective CPU cores (accounts for gpu_frac)
        vcpus = _maybe_float(
            offer.get('cpu_cores_effective', offer.get('cpu_cores')))
        # cpu_ram is already the effective amount, not bare metal
        memory_raw = _maybe_float(offer.get('cpu_ram'))
        memory_gib = (memory_raw / 1024.0) if memory_raw is not None else None
        total_gpu_ram = _maybe_float(offer.get('gpu_total_ram'))

        entry: Dict[str, Union[str, float]] = {
            'InstanceType': instance_type,
            'AcceleratorCount': acc_count if acc_count is not None else 0.0,
            'vCPUs': vcpus if vcpus is not None else 0.0,
            'MemoryGiB': memory_gib if memory_gib is not None else 0.0,
            'Price': price,
            'Region': region,
        }

        gpu_name = _canonical_gpu_name(offer.get('gpu_name', ''))
        entry['AcceleratorName'] = gpu_name
        entry['GpuInfo'] = _gpu_info_string(
            gpu_name, acc_count if acc_count is not None else 0.0,
            total_gpu_ram if total_gpu_ram is not None else 0.0)

        min_bid = offer.get('min_bid')
        spot_price = _maybe_float(min_bid)
        if spot_price is None:
            spot_price = price
        entry['SpotPrice'] = spot_price

        price_map[instance_type].append(entry)

    records: List[Dict[str, Union[str, float]]] = []

    for entries in price_map.values():
        per_location: Dict[str, List[Dict[str, Union[
            str, float]]]] = collections.defaultdict(list)
        for entry in entries:
            region_value = typing.cast(str, entry['Region']).strip()
            stub = f'{entry["InstanceType"]}::{region_value}'
            per_location[stub].append(entry)

        for location_entries in per_location.values():
            valid_entries = [
                entry for entry in location_entries
                if _maybe_float(entry.get('Price')) is not None
            ]
            if not valid_entries:
                continue

            def _price_key(entry: Dict[str, Union[str, float]]) -> float:
                price = _maybe_float(entry.get('Price'))
                assert price is not None
                return cast(float, price)

            valid_entries.sort(key=_price_key)
            for entry in valid_entries[:_MAX_OFFERS_PER_LOCATION]:
                new_entry = entry.copy()
                # Ensure numeric fields are floats for downstream processing.
                price_val = _maybe_float(new_entry['Price'])
                if price_val is None:
                    continue
                new_entry['Price'] = price_val
                spot_val = _maybe_float(new_entry.get('SpotPrice'))
                if spot_val is None:
                    spot_val = price_val
                new_entry['SpotPrice'] = spot_val
                acc_count = _maybe_float(new_entry.get('AcceleratorCount'))
                if acc_count is not None:
                    new_entry['AcceleratorCount'] = acc_count
                vcpus_val = _maybe_float(new_entry.get('vCPUs'))
                if vcpus_val is not None:
                    new_entry['vCPUs'] = vcpus_val
                mem_val = _maybe_float(new_entry.get('MemoryGiB'))
                if mem_val is not None:
                    new_entry['MemoryGiB'] = mem_val
                records.append(new_entry)

    if not records:
        return pd.DataFrame(columns=_CATALOG_COLUMNS)

    df = pd.DataFrame.from_records(records, columns=_CATALOG_COLUMNS)
    # Ensure numeric columns have consistent types.
    for col in ('AcceleratorCount', 'vCPUs', 'MemoryGiB', 'Price', 'SpotPrice'):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df


def _query_vast_offers(secure_only: bool) -> List[Dict]:

    def _fetch(query_parts: List[str]) -> List[Dict]:
        query = ' '.join(query_parts)
        max_attempts = 5
        last_exception: Optional[Exception] = None
        for attempt in range(1, max_attempts + 1):
            try:
                offers = vast.vast().search_offers(query=query, limit=10000)
                if isinstance(offers, list):
                    return offers
                logger.info(
                    'Attempt %d/%d to fetch Vast offers returned '
                    'non-list payload: %r', attempt, max_attempts, offers)
            except ImportError:
                raise
            except Exception as exc:  # pylint: disable=broad-except
                last_exception = exc
                logger.info(
                    'Attempt %d/%d to fetch Vast offers query %s failed: %s',
                    attempt, max_attempts, query, exc)
                if attempt < max_attempts:
                    time.sleep(min(2**attempt, 5))
                continue
            time.sleep(min(2**attempt, 5))
        else:
            if last_exception is not None:
                raise RuntimeError(
                    f'Vast search_offers failed after {max_attempts} attempts'
                ) from last_exception
            raise RuntimeError('Vast search_offers failed after '
                               f'{max_attempts} attempts: non-list responses.')

    base_query = [
        'georegion=true',
        'inet_down>=100',
        'disk_space>=80',
    ]
    if secure_only:
        base_query.append('datacenter=true')

    offers = _fetch(base_query)

    return offers


def _get_vast_dataframe(secure_only: bool) -> 'pd.DataFrame':
    """Get Vast dataframe with time-based file caching (1 hour TTL)."""
    current_time = time.time()

    # Use file-based cache since API server spawns separate processes
    cache_key = f'secure_{secure_only}'
    cache_path = os.path.join('/tmp', f'vast_df_{cache_key}.pkl')
    cache_time_path = os.path.join('/tmp', f'vast_df_{cache_key}.time')
    lock_path = os.path.join('/tmp', f'vast_df_{cache_key}.lock')

    # Use file locking to prevent race conditions across processes
    with open(lock_path, 'w') as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            # Check if cached dataframe exists and is fresh
            if os.path.exists(cache_path) and os.path.exists(cache_time_path):
                try:
                    with open(cache_time_path, 'r') as f:
                        cached_time = float(f.read())
                    if current_time - cached_time < _CATALOG_CACHE_TTL_SECONDS:
                        df = pd.read_pickle(cache_path)
                        return df
                except Exception:  # pylint: disable=broad-except
                    # Cache corrupted, ignore and rebuild
                    pass

            # Cache miss or expired - fetch fresh data
            offers = _query_vast_offers(secure_only)
            df = _build_dataframe(offers)
            logger.info(
                'Vast catalog dataframe built with %d rows (secure_only=%s).',
                len(df), secure_only)
            if df.empty:
                logger.debug(
                    'Vast catalog query returned no offers (secure_only=%s).',
                    secure_only)

            # Update file cache atomically
            try:
                df.to_pickle(cache_path)
                with open(cache_time_path, 'w') as f:
                    f.write(str(current_time))
            except Exception:  # pylint: disable=broad-except
                # Cache write failed, continue without caching
                pass

            return df
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _df_for_region(region: Optional[str]) -> 'pd.DataFrame':
    secure_only = _resolve_secure_only(region)
    return _get_vast_dataframe(secure_only)


def instance_type_exists(instance_type: str) -> bool:
    df = _df_for_region(region=None)
    if common.instance_type_exists_impl(df, instance_type):
        return True
    # Fallback: instance may have been created with opposite secure_only setting
    secure_only = _resolve_secure_only(region=None)
    df_fallback = _get_vast_dataframe(not secure_only)
    return common.instance_type_exists_impl(df_fallback, instance_type)


def validate_region_zone(
        region: Optional[str],
        zone: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    if zone is not None:
        with ux_utils.print_exception_no_traceback():
            raise ValueError('Vast does not support zones.')
    df = _df_for_region(region)
    return common.validate_region_zone_impl('vast', df, region, zone)


def get_hourly_cost(instance_type: str,
                    use_spot: bool = False,
                    region: Optional[str] = None,
                    zone: Optional[str] = None) -> float:
    """Returns the cost, or the cheapest cost among all zones for spot."""
    if zone is not None:
        with ux_utils.print_exception_no_traceback():
            raise ValueError('Vast does not support zones.')
    df = _df_for_region(region)
    try:
        return common.get_hourly_cost_impl(df, instance_type, use_spot, region,
                                           zone)
    except ValueError:
        # Fallback: instance may have been created with opposite
        # secure_only setting
        secure_only = _resolve_secure_only(region)
        df_fallback = _get_vast_dataframe(not secure_only)
        return common.get_hourly_cost_impl(df_fallback, instance_type, use_spot,
                                           region, zone)


def get_vcpus_mem_from_instance_type(
        instance_type: str) -> Tuple[Optional[float], Optional[float]]:
    # Parse instance type directly: format is
    # {num_gpus}x-{gpu_name}-{cpu_cores}-{cpu_ram_mb}
    # Example: 1x-RTX_3060-64-64451 = 64 vCPUs, 64451 MB RAM
    try:
        parts = instance_type.split('-')
        if len(parts) >= 2:
            cpu_cores = float(parts[-2])
            cpu_ram_mb = float(parts[-1])
            cpu_ram_gib = cpu_ram_mb / 1024
            return cpu_cores, cpu_ram_gib
    except (ValueError, IndexError):
        pass

    # Fallback to catalog lookup if parsing fails
    df = _df_for_region(region=None)
    try:
        return common.get_vcpus_mem_from_instance_type_impl(df, instance_type)
    except ValueError:
        # Fallback: instance may have been created with opposite
        # secure_only setting
        secure_only = _resolve_secure_only(region=None)
        df_fallback = _get_vast_dataframe(not secure_only)
        return common.get_vcpus_mem_from_instance_type_impl(
            df_fallback, instance_type)


def get_default_instance_type(cpus: Optional[str] = None,
                              memory: Optional[str] = None,
                              disk_tier: Optional[str] = None,
                              region: Optional[str] = None,
                              zone: Optional[str] = None) -> Optional[str]:
    del disk_tier
    df = _df_for_region(region)
    return common.get_instance_type_for_cpus_mem_impl(df, cpus, memory, region,
                                                      zone)


def get_accelerators_from_instance_type(
        instance_type: str) -> Optional[Dict[str, Union[int, float]]]:
    df = _df_for_region(region=None)
    try:
        return common.get_accelerators_from_instance_type_impl(
            df, instance_type)
    except ValueError:
        # Fallback: instance may have been created with opposite
        # secure_only setting
        secure_only = _resolve_secure_only(region=None)
        df_fallback = _get_vast_dataframe(not secure_only)
        return common.get_accelerators_from_instance_type_impl(
            df_fallback, instance_type)


def get_instance_type_for_accelerator(
        acc_name: str,
        acc_count: int,
        cpus: Optional[str] = None,
        memory: Optional[str] = None,
        use_spot: bool = False,
        region: Optional[str] = None,
        zone: Optional[str] = None) -> Tuple[Optional[List[str]], List[str]]:
    """Returns a list of instance types that have the given accelerator."""
    if zone is not None:
        with ux_utils.print_exception_no_traceback():
            raise ValueError('Vast does not support zones.')
    df = _df_for_region(region)
    return common.get_instance_type_for_accelerator_impl(df=df,
                                                         acc_name=acc_name,
                                                         acc_count=acc_count,
                                                         cpus=cpus,
                                                         memory=memory,
                                                         use_spot=use_spot,
                                                         region=region,
                                                         zone=zone)


def get_region_zones_for_instance_type(instance_type: str,
                                       use_spot: bool) -> List['cloud.Region']:
    df = _df_for_region(region=None)
    df = df[df['InstanceType'] == instance_type]
    if df.empty:
        # Fallback: instance may have been created with opposite
        # secure_only setting
        secure_only = _resolve_secure_only(region=None)
        df_fallback = _get_vast_dataframe(not secure_only)
        df = df_fallback[df_fallback['InstanceType'] == instance_type]
    return common.get_region_zones(df, use_spot)


def list_accelerators(
        gpus_only: bool,
        name_filter: Optional[str],
        region_filter: Optional[str],
        quantity_filter: Optional[int],
        case_sensitive: bool = True,
        all_regions: bool = False,
        require_price: bool = True) -> Dict[str, List[common.InstanceTypeInfo]]:
    del require_price  # Unused.
    df = _df_for_region(region=None)
    return common.list_accelerators_impl('Vast', df, gpus_only, name_filter,
                                         region_filter, quantity_filter,
                                         case_sensitive, all_regions)
