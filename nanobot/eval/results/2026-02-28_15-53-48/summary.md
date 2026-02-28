# Evaluation Run: 2026-02-28_15-53-48

Generated: 2026-02-28T15:53:48.699653

## Summary

| Model | Passed | Tokens | Cost | Total | Mean |
|-------|--------|--------|------|-------|------|
| openrouter/anthropic/claude-sonnet-4.5 | 19/19 | 182,754 | $0.3397 | 90.2s | 4.7s |
| openrouter/anthropic/claude-haiku-4.5 | 19/19 | 179,972 | $0.1921 | 52.2s | 2.7s |
| openrouter/openai/gpt-5.1-codex-mini | 16/19 | 154,478 | $0.0443 | 73.0s | 3.8s |
| openrouter/google/gemini-3-flash-preview | 19/19 | 124,171 | $0.0497 | 53.5s | 2.8s |
| openrouter/google/gemini-2.5-flash | 19/19 | 99,298 | $0.0131 | 34.2s | 1.8s |
| openrouter/x-ai/grok-4.1-fast | 19/19 | 162,835 | $0.0371 | 112.1s | 5.9s |
| openrouter/minimax/minimax-m2.5 | 19/19 | 175,913 | $0.0327 | 112.3s | 5.9s |
| openrouter/z-ai/glm-5 | 19/19 | 128,997 | $0.1278 | 170.1s | 9.0s |
| openrouter/deepseek/deepseek-v3.2 | 19/19 | 208,639 | $0.0541 | 136.1s | 7.2s |
| openrouter/mistralai/mistral-large-2512 | 18/19 | 158,453 | $0.0821 | 70.1s | 3.7s |

## Task Results

| Task | claude-sonnet-4.5 | claude-haiku-4.5 | gpt-5.1-codex-mini | gemini-3-flash-preview | gemini-2.5-flash | grok-4.1-fast | minimax-m2.5 | glm-5 | deepseek-v3.2 | mistral-large-2512 |
|------|------|------|------|------|------|------|------|------|------|------|
| greeting | 5305ms | 4503ms | 2029ms | 2771ms | 1584ms | 6129ms | 4625ms | 8823ms | 6654ms | 3417ms |
| simple_math | 2101ms | 831ms | 1727ms | 1010ms | 663ms | 2621ms | 2047ms | 13800ms | 1742ms | 732ms |
| factual | 2555ms | 912ms | 1538ms | 942ms | 771ms | 1434ms | 1407ms | 3771ms | 1638ms | 611ms |
| file_write | 4351ms | 2565ms | 2457ms | 2305ms | 1616ms | 6356ms | 5052ms | 18881ms | 5838ms | 2446ms |
| file_read | 4237ms | 2351ms | 3532ms | 2102ms | 1467ms | 3949ms | 3983ms | 4892ms | 3788ms | 1964ms |
| file_edit | 4539ms | 2763ms | 4244ms | 2300ms | 1894ms | 7822ms | 3656ms | 17714ms | 11062ms | 2801ms |
| heartbeat_write | 4229ms | 2458ms | 3388ms | 2417ms | 1398ms | 7374ms | 7204ms | 6450ms | 9829ms | 9272ms |
| heartbeat_read | 4374ms | 2056ms | 2201ms | 2729ms | 2039ms | 4506ms | 4814ms | 7885ms | 4056ms | FAIL |
| memory_store | 7968ms | 4112ms | 6195ms | 5663ms | 2955ms | 12393ms | 6987ms | 10876ms | 11370ms | 5299ms |
| memory_recall | 2344ms | 1100ms | 2458ms | 1134ms | 922ms | 1925ms | 2229ms | 4893ms | 1476ms | 972ms |
| code_exec_simple | 7700ms | 4615ms | FAIL | 4334ms | 3994ms | 13538ms | 5631ms | 12595ms | 11319ms | 5706ms |
| code_exec_file | 7225ms | 4071ms | FAIL | 3722ms | 2217ms | 12496ms | 5427ms | 10752ms | 8541ms | 4241ms |
| cron_create | 4546ms | 2518ms | 3224ms | 2385ms | 1539ms | 4403ms | 3244ms | 8192ms | 5018ms | 2271ms |
| cron_list | 4915ms | 2486ms | 3943ms | 2352ms | 2386ms | 3686ms | 13258ms | 5020ms | 3891ms | 2493ms |
| recall_math | 2180ms | 1256ms | 1394ms | 1275ms | 655ms | 3483ms | 1601ms | 4606ms | 1537ms | 1228ms |
| recall_file | 2298ms | 932ms | 1375ms | 4834ms | 886ms | 2457ms | 2309ms | 2765ms | 2747ms | 1337ms |
| skill_create | 8484ms | 6014ms | 9935ms | 5811ms | 3274ms | 5018ms | 16200ms | 10137ms | 30332ms | 11561ms |
| skill_verify_script | 4861ms | 2790ms | FAIL | 2273ms | 1718ms | 7514ms | 5390ms | 4916ms | 6143ms | 4614ms |
| summarize_session | 6034ms | 3820ms | 3215ms | 3168ms | 2229ms | 4963ms | 17276ms | 13106ms | 9073ms | 7270ms |