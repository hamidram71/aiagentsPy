"""Prompts the code depends on, as opposed to prompts you tune.

src/prompts.py is yours: you edit it to change how the agent behaves, and
updates never overwrite it. The prompts here are part of a code contract --
the node that sends one relies on the exact question it asks -- so they live
with the code and change with it.
"""

WRAP_UP_SYSTEM = """An analyst worked on ONE step of a data analysis and ran out of turns
before writing its conclusion. You get the step it was working on and the
tool outputs it received, oldest first. Decide whether those outputs already
answer the step.

Set has_finding=true when the outputs contain numbers that answer the step.
A check that found nothing wrong IS a result: "0 rows differ by more than
$0.50" or "5 duplicate rows, all identical" answers a validation step.
Then:
- statement: one sentence answering the step, with the numbers copied exactly
  from the outputs. Never compute, round, or estimate.
- evidence: the figures it rests on, copied from the outputs.

Set has_finding=false only when the outputs are errors, or contain nothing
that bears on the step.
"""
