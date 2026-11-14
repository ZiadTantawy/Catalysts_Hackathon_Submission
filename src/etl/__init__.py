"""The ETL: one deterministic build of frame-level tables from the raw tactical bundle.

    python -m src.etl.build            # writes cache/, prints cache/report.txt

Nothing in this package reads any ground truth.
"""

__version__ = "1.2.0"
