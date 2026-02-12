"""
Pack solution source files into solution.json.

Reads configuration from config.toml and packs the appropriate source files
(Triton or CUDA) into a Solution JSON file for submission.
"""

import sys
from pathlib import Path

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    import tomllib
except ImportError:
    import tomli as tomllib

from flashinfer_bench import BuildSpec
from flashinfer_bench.agents import pack_solution_from_files


def load_config() -> dict:
    """Load configuration from config.toml."""
    config_path = PROJECT_ROOT / "config.toml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "rb") as f:
        return tomllib.load(f)


def pack_solution(output_path: Path = None) -> Path:
    """Pack solution files into a Solution JSON."""
    config = load_config()

    solution_config = config["solution"]
    build_config = config["build"]

    language = build_config["language"]
    entry_point = build_config["entry_point"]

    # Determine source directory based on language
    if language == "triton":
        source_dir = PROJECT_ROOT / "solution" / "triton"
    elif language == "cuda":
        source_dir = PROJECT_ROOT / "solution" / "cuda"
    else:
        raise ValueError(f"Unsupported language: {language}")

    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")

    # Create build spec
    spec = BuildSpec(
        language=language,
        target_hardware=["cuda"],
        entry_point=entry_point,
    )
    short_definition = solution_config["definition"]
    match short_definition:
        case "moe":
            definition = "moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048"
        case "dsa_sparse_attention":
            definition = "dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64"
        case "dsa_topk_indexer":
            definition = "dsa_topk_indexer_fp8_h64_d128_topk2048_ps64"
        case "gdn_decode":
            definition = "gdn_decode_qk4_v8_d128_k_last"
        case "gdn_prefill":
            definition = "gdn_prefill_qk4_v8_d128_k_last"

    # Pack the solution
    solution = pack_solution_from_files(
        path=str(source_dir),
        spec=spec,
        name=solution_config["name"],
        definition=definition,
        author=solution_config["author"],
    )

    # Write to output file
    if output_path is None:
        output_path = PROJECT_ROOT / "solution.json"

    output_path.write_text(solution.model_dump_json(indent=2))
    print(f"Solution packed: {output_path}")
    print(f"  Name: {solution.name}")
    print(f"  Definition: {solution.definition}")
    print(f"  Author: {solution.author}")
    print(f"  Language: {language}")

    return output_path


def main():
    """Entry point for pack_solution script."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Pack solution files into solution.json"
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output path for solution.json (default: ./solution.json)",
    )
    args = parser.parse_args()

    try:
        pack_solution(args.output)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
