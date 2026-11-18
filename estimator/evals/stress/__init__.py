"""Stress test of the CAG (Session 6 exercise): measure where it breaks.

- ``scenarios`` — three scripted 20-turn conversations with fact-trackers.
- ``metrics``   — ``LatencyBudgetMetric``, ``CostBudgetMetric``, ``MemoryDriftMetric``.
- ``fixtures``  — ``build_pdfs.py`` generates the calibrated PDF attachments.
- ``run``       — CLI runner: scenarios × attachment sizes × repeats → ``results.csv``.
- ``analyze``   — reads ``results.csv`` and prints the Markdown tables of ``REPORT.md``.
"""
