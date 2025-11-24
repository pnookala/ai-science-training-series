1. Cerebras Homework

When training LLaMA-7B on the Cerebras CSX system with a 1024-token block size, I observed a throughput of about 33 samples/sec (e.g., Step 50 showing Loss = 7.67 and Rate ≈ 33 samples/sec). The code errored out after this. If I increase the block size, the compute and memory cost per sample grows roughly linearly with sequence length, so throughput is expected to drop proportionally. In practice, doubling the block size to 2048 would reduce throughput to around 16–17 samples/sec, 4096 would yield roughly 8 samples/sec, and 8192 would fall to about 4 samples/sec, causing the overall training time to double with each increase in sequence length. The loss at early steps may look similar, but longer sequences change the effective batch size (in tokens), so learning dynamics can shift slightly. Overall, larger block sizes enable learning longer-range dependencies but come with a clear trade-off of significantly lower throughput and longer wall-clock training time.

2. SambaNova Homework

(.venv) pnookala@MacBookPro 05-AITestbed % time curl -X POST "https://inference-api.alcf.anl.gov/resource_server/metis/api/v1/chat/completions/" \
  -H "Authorization: Bearer $access_token" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-oss-120b-131072",
    "messages": [{"role": "user", "content": "Write a short summary about AI"}],
    "max_tokens": 128
  }'
{"choices":[{"finish_reason":"length","index":0,"logprobs":null,"message":{"content":"**Artificial Intelligence (AI) – Short Summary**\n\nArtificial Intelligence (AI) is a branch of computer science that creates systems capable of performing tasks that normally require human intelligence. These tasks include learning from data (machine learning), recognizing patterns (computer vision, speech recognition), reasoning, planning, and natural‑language understanding. Modern AI combines statistical methods, neural networks, and large datasets to build models that","reasoning":"User asks: \"Write a short summary about AI\". Need to produce a concise summary. Probably a paragraph. Should be short. Provide overview: definition, capabilities, applications, challenges. Let's produce.","role":"assistant"}}],"created":1764022114.350458,"id":"6aab7b9d-991c-4acb-91ee-a1e77fb993e8","model":"gpt-oss-120b-131072","object":"chat.completion","system_fingerprint":"fastcoe","usage":{"completion_tokens":128,"completion_tokens_after_first_per_sec":281.8221781097033,"completion_tokens_after_first_per_sec_first_ten":282.2688523547725,"completion_tokens_after_first_per_sec_graph":282.2688523547725,"completion_tokens_per_sec":208.6463763362214,"end_time":1764022114.350265,"is_last_response":true,"prompt_tokens":73,"prompt_tokens_details":{"cached_tokens":0},"start_time":1764022113.7367868,"stop_reason":"length","time_to_first_token":0.16283941268920898,"total_latency":0.6134781837463379,"total_tokens":201,"total_tokens_per_sec":327.64001284047265}}
curl -X POST  -H "Authorization: Bearer $access_token" -H  -d   0.02s user 0.01s system 1% cpu 1.369 total

(.venv) pnookala@MacBookPro 05-AITestbed % time curl -X POST "https://inference-api.alcf.anl.gov/resource_server/metis/api/v1/chat/completions/" \
  -H "Authorization: Bearer ${access_token}" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Llama-4-Maverick-17B-128E-Instruct",
    "messages": [{"role": "user", "content": "Write a short summary about AI"}],
    "max_tokens": 128
  }'
{"choices":[{"finish_reason":"stop","index":0,"logprobs":null,"message":{"content":"Here is a short summary about AI:\n\n**Artificial Intelligence (AI)** refers to the development of computer systems that can perform tasks that typically require human intelligence, such as learning, problem-solving, decision-making, and perception. AI involves the use of algorithms, statistical models, and computer programs to enable machines to think and act like humans. AI has numerous applications across various industries, including healthcare, finance, transportation, and education, and is transforming the way we live and work.","role":"assistant"}}],"created":1764022081.3721223,"id":"4f4e997d-122e-46d9-9534-cf0f2d9faf18","model":"Llama-4-Maverick-17B-128E-Instruct","object":"chat.completion","system_fingerprint":"fastcoe","usage":{"completion_tokens":95,"completion_tokens_after_first_per_sec":598.5431701803685,"completion_tokens_after_first_per_sec_first_ten":611.131207383215,"completion_tokens_after_first_per_sec_graph":611.131207383215,"completion_tokens_per_sec":427.9359651515705,"end_time":1764022081.3719316,"is_last_response":true,"prompt_tokens":18,"stop_reason":"stop","time_to_first_token":0.06494784355163574,"total_latency":0.22199583053588867,"total_tokens":113,"total_tokens_per_sec":509.01856907502594}}
curl -X POST  -H "Authorization: Bearer ${access_token}" -H  -d   0.02s user 0.01s system 3% cpu 1.030 total

Visibly, LLama appears to be faster. I tried to come up with a python script to test latencies with a bigger dataset, but I kept getting 404 not found errors, so I tried this simple way to measuring latency. I have also uploaded the python script for reference.
