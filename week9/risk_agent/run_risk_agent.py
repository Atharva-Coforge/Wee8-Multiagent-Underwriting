"""Container entrypoint. Score one ``EnrichedCase`` JSON file and print the result.

Runs only the risk scoring agent, with the week 8 retry on timeout.
The same image runs on AWS and on-prem; only the environment differs.
"""
