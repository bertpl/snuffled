from hypothesis import settings

# Deterministic, tightly-capped profile: property runs exercise whole-pipeline extract_all(), so we
# keep the example count low (CI time) and disable the per-example deadline (a run can exceed it).
# derandomize=True makes CI reproducible without relying on a committed example database.
settings.register_profile("snuffled", max_examples=50, deadline=None, derandomize=True)
settings.load_profile("snuffled")
