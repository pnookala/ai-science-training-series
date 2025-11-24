#!/usr/bin/env python3
# benchmark_metis_sophia_final.py
# Fully cleaned and ready-to-run script to benchmark GPT-OSS models
# on ALCF Metis (SambaNova) and Sophia endpoints.
# Automatically prints results summary for homework.

import argparse
import asyncio
import statistics
import subprocess
import os
from typing import Any, Dict, List, Tuple

try:
    import httpx
except Exception:
    raise ImportError("Missing 'httpx'. Install with: pip install httpx[http2]")

try:
    from datasets import load_dataset
except Exception:
    load_dataset = None

try:
    from transformers import AutoTokenizer
except Exception:
    AutoTokenizer = None

# -------------------------- Helpers --------------------------
def shell_get_token_from_script() -> str:
    # Try to get token from inference_auth_token.py if available
    script = "inference_auth_token.py"
    if not os.path.exists(script):
        return ""
    try:
        out = subprocess.check_output(["python", script, "get_access_token"], stderr=subprocess.DEVNULL)
        token = out.decode().strip()
        return token
    except Exception:
        return ""

async def fetch_list_endpoints(access_token: str) -> Dict[str, Any]:
    url = "https://inference-api.alcf.anl.gov/resource_server/list-endpoints"
    headers = {"Authorization": f"Bearer {access_token}"}
    async with httpx.AsyncClient(http2=True, timeout=30.0) as client:
        r = await client.get(url, headers=headers)
        r.raise_for_status()
        return r.json()

def choose_metis_sophia_from_json(j: Dict[str, Any]) -> Tuple[str, str]:
    # Return base URLs for Metis and Sophia (vLLM/DGX) endpoints
    metis = None
    sophia = None
    clusters = j.get("clusters", {})
    if "metis" in clusters:
        m = clusters["metis"]
        base = m.get("base_url") or None
        if base:
            metis = "https://inference-api.alcf.anl.gov" + base
    for k, v in clusters.items():
        if k == "metis":
            continue
        base = v.get("base_url") or None
        if base:
            sophia = "https://inference-api.alcf.anl.gov" + base
            break
    return metis, sophia

def simple_tokenize(text: str) -> int:
    return max(1, len(text.split()))

# -------------------------- Dataset --------------------------
def load_prompts(dataset_name: str = "wikitext", subset_size: int = 200) -> List[str]:
    prompts = []
    if load_dataset is not None:
        try:
            if dataset_name == "wikitext":
                ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
            else:
                ds = load_dataset(dataset_name, split="train")
            for i, ex in enumerate(ds):
                if i >= subset_size:
                    break
                text = ex.get("text") or str(ex)
                prompts.append(text.strip().replace("\n", " ")[:300])
            if prompts:
                return prompts
        except Exception:
            pass
    for i in range(subset_size):
        prompts.append(f"Write a concise summary about AI topic #{i}.")
    return prompts

# -------------------------- Payload Builders --------------------------
def build_metis_payload(prompt: str, max_tokens: int = 128) -> Dict[str, Any]:
    return {
        "model": "gpt-oss-120b-131072",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
    }

def build_openai_payload(prompt: str, max_tokens: int = 128) -> Dict[str, Any]:
    return {
        "model": "gpt-oss-120b-131072",
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }

# -------------------------- Async Requests --------------------------
async def run_request(client: httpx.AsyncClient, url: str, payload: Dict[str, Any], endpoint_type: str) -> Tuple[float, int]:
    import time
    start = time.time()
    if endpoint_type == "metis":
        r = await client.post(f"{url}/api/v1/chat/completions/", json=payload)
    else:
        r = await client.post(f"{url}/v1/completions", json=payload)
    r.raise_for_status()
    resp = r.json()
    # count tokens approximately
    if endpoint_type == "metis":
        content = resp.get("choices", [{}])[0].get("message", {}).get("content", "")
    else:
        content = resp.get("choices", [{}])[0].get("text", "")
    tokens = simple_tokenize(content)
    elapsed = time.time() - start
    return elapsed, tokens

async def benchmark_endpoint(url: str, endpoint_type: str, prompts: List[str], concurrency_list: List[int]) -> Dict[str, Any]:
    results = {}
    async with httpx.AsyncClient(http2=True, timeout=60.0) as client:
        for conc in concurrency_list:
            latencies = []
            tokens_list = []
            tasks = []
            for prompt in prompts:
                payload = build_metis_payload(prompt) if endpoint_type=="metis" else build_openai_payload(prompt)
                tasks.append(run_request(client, url, payload, endpoint_type))
            res = await asyncio.gather(*tasks)
            for lat, tok in res:
                latencies.append(lat)
                tokens_list.append(tok)
            results[f"{conc}-concurrency"] = {"latencies": latencies, "tokens": tokens_list}
    return results

# -------------------------- Main --------------------------
async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--access-token", type=str, default=os.environ.get("ACCESS_TOKEN"))
    parser.add_argument("--dataset", type=str, default="wikitext")
    parser.add_argument("--subset_size", type=int, default=20)
    parser.add_argument("--concurrency", type=str, default="1,4,8")
    args = parser.parse_args()

    token = args.access_token or shell_get_token_from_script()
    if not token:
        print("No access token found. Set ACCESS_TOKEN or provide --access-token")
        return

    endpoints_json = await fetch_list_endpoints(token)
    metis_url, sophia_url = choose_metis_sophia_from_json(endpoints_json)
    prompts = load_prompts(args.dataset, args.subset_size)
    concurrency_list = [int(x) for x in args.concurrency.split(",")]

    final_summary = {}

    if metis_url:
        print("Benchmarking Metis...")
        metis_results = await benchmark_endpoint(metis_url, "metis", prompts, concurrency_list)
        final_summary["Metis"] = metis_results

    if sophia_url:
        print("Benchmarking Sophia...")
        sophia_results = await benchmark_endpoint(sophia_url, "sophia", prompts, concurrency_list)
        final_summary["Sophia"] = sophia_results

    # -------------------------- Print Homework-ready Summary --------------------------
    print("\n\nHomework-ready summary:\n")
    for endpoint, conc_dict in final_summary.items():
        all_latencies = []
        all_tokens = []
        for conc, data in conc_dict.items():
            all_latencies.extend(data["latencies"])
            all_tokens.extend(data["tokens"])
        avg_lat = sum(all_latencies)/len(all_latencies)
        med_lat = statistics.median(all_latencies)
        throughput = len(all_latencies)/sum(all_latencies)
        tokens_per_sec = sum(all_tokens)/sum(all_latencies)
        print(f"{endpoint}:")
        print(f"  Avg latency: {avg_lat:.3f} s")
        print(f"  Median latency: {med_lat:.3f} s")
        print(f"  Approx. throughput: {throughput:.2f} requests/sec")
        print(f"  Approx. tokens/sec: {tokens_per_sec:.2f}\n")

if __name__ == "__main__":
    asyncio.run(main())

