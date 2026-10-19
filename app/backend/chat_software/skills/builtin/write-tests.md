---
name: Write tests
description: Write focused unit tests for given code, run them in the sandbox and report what passed.
tools: run_code
tags: testing, pytest, unit tests, code
---
1. List the behaviours worth testing: normal cases, boundaries, error paths.
2. Write pytest-style tests (or unittest if pytest is unavailable) that are independent and deterministic.
3. Run them with run_code together with the code under test; include the exact output.
4. If a test fails, decide whether the test or the code is wrong and say which.
