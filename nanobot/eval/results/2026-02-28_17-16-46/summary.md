# Evaluation Run: 2026-02-28_17-16-46

Generated: 2026-02-28T17:16:46.772838

## Summary

| Model | Passed | Tokens | Cost | Total | Mean |
|-------|--------|--------|------|-------|------|
| openrouter/x-ai/grok-4 | 19/19 | 189,020 | $0.2968 | 430.2s | 22.6s |
| openrouter/anthropic/claude-sonnet-4.6 | 19/19 | 191,729 | $0.6157 | 75.2s | 4.0s |
| openrouter/google/gemini-2.5-pro | 18/19 | 147,755 | $0.1632 | 146.6s | 7.7s |
| openrouter/openai/gpt-5.1 | 19/19 | 178,768 | $0.2675 | 105.2s | 5.5s |

## Task Results

| Task | grok-4 | claude-sonnet-4.6 | gemini-2.5-pro | gpt-5.1 |
|------|------|------|------|------|
| greeting | 13052ms | 5929ms | 4475ms | 3264ms |
| simple_math | 6441ms | 1441ms | 14213ms | 3386ms |
| factual | 5836ms | 1229ms | 11570ms | 965ms |
| file_write | 43314ms | 3434ms | 4688ms | 3683ms |
| file_read | 13004ms | 3017ms | 7883ms | 2622ms |
| file_edit | 6944ms | 3653ms | 6044ms | 6759ms |
| heartbeat_write | 27457ms | 5258ms | FAIL | 5979ms |
| heartbeat_read | 11772ms | 3145ms | 8669ms | 4899ms |
| memory_store | 39033ms | 5659ms | 9127ms | 21266ms |
| memory_recall | 8280ms | 1534ms | 5345ms | 1340ms |
| code_exec_simple | 22529ms | 6557ms | 5611ms | 5019ms |
| code_exec_file | 41810ms | 5719ms | 7303ms | 4198ms |
| cron_create | 14612ms | 4416ms | 5416ms | 4814ms |
| cron_list | 11904ms | 4097ms | 6742ms | 4505ms |
| recall_math | 27179ms | 1466ms | 16713ms | 2689ms |
| recall_file | 23379ms | 1605ms | 7577ms | 3152ms |
| skill_create | 18670ms | 7060ms | 7131ms | 10440ms |
| skill_verify_script | 67960ms | 3298ms | 4241ms | 4213ms |
| summarize_session | 27033ms | 6689ms | 9309ms | 11966ms |