"""Reusable writer for candidate-generation result tables.

Every candidate table uses the same long format:

    impression_uid: Int64
    cutoff: Int16, 1..C
    news_code: Int32

Example:
    from cg_result_writer import CandidateWriter

    writer = CandidateWriter(path)
    try:
        writer.add(impression_uids, cutoff_values, news_codes)
    finally:
        writer.close()
"""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

CANDIDATE_SCHEMA = pa.schema(
    [
        ("impression_uid", pa.int64()),
        ("cutoff", pa.int16()),
        ("news_code", pa.int32()),
    ]
)
DEFAULT_BATCH_ROWS = 2_000_000


class CandidateWriter:
    """Buffered writer for the standard long-format candidate table."""

    def __init__(self, path, batch_rows=DEFAULT_BATCH_ROWS):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.batch_rows = batch_rows
        self.frames = []
        self.pending = 0
        self.rows = 0
        self.writer = pq.ParquetWriter(
            self.path, CANDIDATE_SCHEMA, compression="zstd"
        )

    def add(self, impression_uid, cutoff, news_code):
        frame = pa.table(
            {
                "impression_uid": pa.array(impression_uid, type=pa.int64()),
                "cutoff": pa.array(cutoff, type=pa.int16()),
                "news_code": pa.array(news_code, type=pa.int32()),
            },
            schema=CANDIDATE_SCHEMA,
        )
        self.frames.append(frame)
        self.pending += frame.num_rows
        self.rows += frame.num_rows
        if self.pending >= self.batch_rows:
            self.flush()

    def flush(self):
        if not self.frames:
            return
        table = pa.concat_tables(self.frames)
        self.writer.write_table(table)
        self.frames = []
        self.pending = 0

    def close(self):
        self.flush()
        self.writer.close()


__all__ = [
    "CANDIDATE_SCHEMA",
    "CandidateWriter",
    "DEFAULT_BATCH_ROWS",
]
