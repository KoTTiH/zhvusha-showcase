"""Internal Memory-module pipelines (phase 5C).

Currently houses :mod:`src.memory.pipelines.enrichment` — the async
episode enrichment pipeline that :class:`src.memory.sonnet_enricher.SonnetEnricher`
delegates to. The consolidation flow is NOT structured via a pipeline;
see the design note on :class:`src.memory.protocols.ConsolidationProtocol`
for why.

This subpackage is forbidden to external importers by the
``memory_isolation`` importlinter contract. Public access to enrichment
goes through :class:`src.memory.protocols.EnrichmentProtocol` and
:func:`src.memory.get_enricher`.
"""
