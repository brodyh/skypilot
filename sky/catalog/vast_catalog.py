"""Vast | Catalog

Runtime utilities for querying Vast.ai offerings.

Historically SkyPilot relied on a pre-generated CSV catalog. This module now
builds an in-memory dataframe directly from Vast's search API on each request
so that pricing and availability reflect the latest marketplace state.
"""

import collections
import json
import math
import re
import typing
from typing import Dict, List, Optional, Tuple, Union

from sky import sky_logging
from sky import skypilot_config
from sky.adaptors import common as adaptors_common
from sky.adaptors import vast
from sky.catalog import common
from sky.utils import annotations
from sky.utils import ux_utils

if typing.TYPE_CHECKING:  # pragma: no cover
    from sky.clouds import cloud

if typing.TYPE_CHECKING:  # pragma: no cover
    import pandas as pd  # type: ignore  # pylint: disable=unused-import
else:
    pd = adaptors_common.LazyImport('pandas')

logger = sky_logging.init_logger(__name__)

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


def _resolve_secure_only(region: Optional[str] = None) -> bool:
    return bool(
        skypilot_config.get_effective_region_config(cloud='vast',
                                                    region=region,
                                                    keys=('secure_only',),
                                                    default_value=True))


def _create_instance_type(offer: Dict) -> str:
    stubify = lambda x: re.sub(r'\s', '_', x)
    return '{}x-{}-{}-{}'.format(offer['num_gpus'], stubify(offer['gpu_name']),
                                 offer['cpu_cores'], offer['cpu_ram'])


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


def _build_dataframe(offers: List[Dict]) -> 'pd.DataFrame':
    if not offers:
        return pd.DataFrame(columns=_CATALOG_COLUMNS)

    price_map: Dict[str, List[Dict]] = collections.defaultdict(list)

    for offer in offers:
        try:
            instance_type = _create_instance_type(offer)
            price = float(offer['search']['totalHour'])
        except (KeyError, TypeError, ValueError):
            continue

        region = offer.get('geolocation')
        if not isinstance(region, str):
            continue

        acc_count = float(offer.get('num_gpus', 0.0))
        vcpus = float(offer.get('cpu_cores', 0.0))
        memory_gib = float(offer.get('cpu_ram', 0.0)) / 1024.0
        total_gpu_ram = float(offer.get('gpu_total_ram', 0.0))

        entry: Dict[str, Union[str, float]] = {
            'InstanceType': instance_type,
            'AcceleratorCount': acc_count,
            'vCPUs': vcpus,
            'MemoryGiB': memory_gib,
            'Price': price,
            'Region': region,
        }

        gpu_name = _canonical_gpu_name(offer.get('gpu_name', ''))
        entry['AcceleratorName'] = gpu_name
        entry['GpuInfo'] = _gpu_info_string(gpu_name, acc_count, total_gpu_ram)

        min_bid = offer.get('min_bid')
        try:
            spot_price = float(min_bid) if min_bid is not None else price
        except (TypeError, ValueError):
            spot_price = price
        entry['SpotPrice'] = spot_price

        price_map[instance_type].append(entry)

    records: List[Dict[str, Union[str, float]]] = []

    for entries in price_map.values():
        valid_prices = sorted(float(e['Price']) for e in entries)
        if not valid_prices:
            continue
        target_index = max(math.ceil(0.5 * len(valid_prices)) - 1, 0)
        price_target = valid_prices[target_index]

        max_bid = max((float(e['SpotPrice']) for e in entries),
                      default=price_target)

        best_per_stub: Dict[str, Dict[str, Union[str, float]]] = {}
        for entry in entries:
            price = float(entry['Price'])
            region_value = typing.cast(str, entry['Region'])
            if price > price_target:
                continue

            stub = f'{entry["InstanceType"]} {region_value[-2:]}'
            existing = best_per_stub.get(stub)
            if existing is None or price < float(existing['Price']):
                new_entry = entry.copy()
                new_entry['Price'] = price_target
                new_entry['SpotPrice'] = max_bid
                best_per_stub[stub] = new_entry

        records.extend(best_per_stub.values())

    if not records:
        return pd.DataFrame(columns=_CATALOG_COLUMNS)

    df = pd.DataFrame.from_records(records, columns=_CATALOG_COLUMNS)
    # Ensure numeric columns have consistent types.
    for col in ('AcceleratorCount', 'vCPUs', 'MemoryGiB', 'Price', 'SpotPrice'):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df


def _query_vast_offers(secure_only: bool) -> List[Dict]:
    query_parts = [
        'chunked=true',
        'georegion=true',
        'inet_down>=100',
        'disk_space>=80',
    ]
    if secure_only:
        query_parts.append('datacenter=true')
    query = ' '.join(query_parts)

    try:
        offers = vast.vast().search_offers(query=query, limit=10000)
    except ImportError:
        raise
    except Exception as exc:  # pylint: disable=broad-except
        logger.warning('Failed to fetch Vast offers: %s', exc)
        return []

    if isinstance(offers, int):
        # Vast returns an integer error code on failure.
        logger.warning('Vast search_offers returned error code %s for query %s',
                       offers, query)
        return []
    if not isinstance(offers, list):
        logger.warning('Unexpected result from Vast search_offers: %r', offers)
        return []
    return offers


@annotations.lru_cache(scope='request')
def _get_vast_dataframe(secure_only: bool) -> 'pd.DataFrame':
    offers = _query_vast_offers(secure_only)
    df = _build_dataframe(offers)
    if df.empty:
        logger.debug('Vast catalog query returned no offers (secure_only=%s).',
                     secure_only)
    return df


def _df_for_region(region: Optional[str]) -> 'pd.DataFrame':
    secure_only = _resolve_secure_only(region)
    return _get_vast_dataframe(secure_only)


def instance_type_exists(instance_type: str) -> bool:
    df = _df_for_region(region=None)
    return common.instance_type_exists_impl(df, instance_type)


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
    return common.get_hourly_cost_impl(df, instance_type, use_spot, region,
                                       zone)


def get_vcpus_mem_from_instance_type(
        instance_type: str) -> Tuple[Optional[float], Optional[float]]:
    df = _df_for_region(region=None)
    return common.get_vcpus_mem_from_instance_type_impl(df, instance_type)


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
    return common.get_accelerators_from_instance_type_impl(df, instance_type)


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
