#!/usr/bin/env python3
"""Quick-and-dirty Vast offer explorer."""

import argparse
import json
from typing import Any, Dict, List

try:  # Optional dependency for tabular pretty-printing.
    import pandas as pd
except ImportError:  # pragma: no cover - pandas is bundled with SkyPilot.
    pd = None

from vastai_sdk import VastAI


def build_query(args: argparse.Namespace) -> str:
    clauses: List[str] = [
        "georegion=true",
        f"disk_space>={args.disk_space}",
        f"num_gpus={args.num_gpus}",
    ]

    if args.gpu_name:
        clauses.append(f'gpu_name="{args.gpu_name}"')

    if args.region:
        # Vast regions are appended to the end of the geolocation string; we just
        # match on the trailing country/region code (e.g., "US" or "EU").
        clauses.append(f'geolocation="{args.region}"')

    if args.inet_down is not None:
        clauses.append(f"inet_down>={args.inet_down}")

    if args.cpu_ram is not None:
        clauses.append(f'cpu_ram>="{args.cpu_ram}"')

    if args.secure_only:
        clauses.append("datacenter=true")

    return " ".join(clauses)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch live Vast.ai offers with a flexible query.")
    parser.add_argument(
        "--gpu-name",
        help="Exact GPU name to match. If omitted, no GPU filter.")
    parser.add_argument("--num-gpus",
                        type=int,
                        default=1,
                        help="Number of GPUs required (default: %(default)s).")
    parser.add_argument(
        "--cpu-ram",
        type=float,
        help="Minimum CPU RAM in GiB (e.g., 64). If omitted, no filter.")
    parser.add_argument(
        "--disk-space",
        type=int,
        default=80,
        help="Minimum disk space in GiB (default: %(default)s).",
    )
    parser.add_argument(
        "--inet-down",
        type=int,
        help=
        ("Minimum downstream bandwidth in Mbps. Matches Vast catalog filtering when provided."
        ),
    )
    parser.add_argument(
        "--region",
        help=
        ("Optional two-letter region/country suffix to match (e.g., 'US', 'DE')."
        ))
    parser.add_argument("--secure-only",
                        action="store_true",
                        help="Restrict to datacenter (verified) hosts.")
    parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Max number of offers to show (default: %(default)s).")
    parser.add_argument("--raw",
                        action="store_true",
                        help="Print the raw JSON response for each offer.")
    args = parser.parse_args()

    query = build_query(args)
    client = VastAI()
    print(f"Query: {query}")
    offers = client.search_offers(query=query, limit=args.limit)

    if isinstance(offers, int):
        raise RuntimeError(
            f"Vast returned error code {offers} for query: {query}")

    if not offers:
        print("No matching offers.")
        return

    rows: List[Dict[str, Any]] = []

    for offer in offers:
        storage = offer.get("storage") or {}
        disk_space = storage.get("ssd_space")

        cpu_ram = offer.get("cpu_ram")
        cpu_ram_gib = cpu_ram / 1024 if isinstance(cpu_ram,
                                                   (int, float)) else None

        summary: Dict[str, Any] = {
            "id": offer["id"],
            "gpu_name": offer["gpu_name"],
            "num_gpus": offer["num_gpus"],
            "cpu_cores": offer["cpu_cores"],
            "cpu_ram_gib": cpu_ram_gib,
            "disk_gib": disk_space,
            "price_hour": offer["search"]["totalHour"],
            "min_bid": offer.get("min_bid"),
            "geolocation": offer["geolocation"],
        }
        rows.append(summary)
        if args.raw:
            print(json.dumps(offer, indent=2))
            print("-" * 60)

    if not rows:
        print("No matching offers.")
        return

    if pd is None:
        # Fallback: print JSON summaries.
        for row in rows:
            print(json.dumps(row, indent=2))
            print("-" * 60)
    else:
        df = pd.DataFrame(rows)
        df = df.sort_values(by="price_hour")
        print(df.to_string(index=False))


if __name__ == "__main__":
    main()
