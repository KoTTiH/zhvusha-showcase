"""Tests for PersonalityEvolution."""

from __future__ import annotations

from unittest.mock import patch

from src.personality.evolution import PersonalityEvolution


def _patch_embed(val=None):
    vec = val or [0.5] * 384
    return patch(
        "src.personality.evolution.EmbeddingService.embed",
        return_value=vec,
    )


def _patch_cosine(val=0.5):
    return patch(
        "src.personality.evolution.EmbeddingService.cosine_similarity",
        return_value=val,
    )


def _setup_personality(tmp_path):
    p = tmp_path / "personality"
    p.mkdir()
    (p / "core.md").write_text("# Core\nI am Zhvusha.\nI learn from experience.\n")
    (p / "genes.md").write_text("# Genes\n| Gene | Value |\n| Curiosity | HIGH |\n")
    (p / "MEMORY.md").write_text("# Memory Index\n- [core.md](core.md)\n")
    return PersonalityEvolution(p)


# --- should_create_new_file ---


async def test_create_new_file_3_plus_mentions(tmp_path):
    evo = _setup_personality(tmp_path)
    with _patch_embed(), _patch_cosine(0.3):  # No matching file
        assert await evo.should_create_new_file("новая тема", 3, 0.5) is True


async def test_create_new_file_high_importance(tmp_path):
    evo = _setup_personality(tmp_path)
    with _patch_embed(), _patch_cosine(0.3):
        assert await evo.should_create_new_file("разовый инсайт", 1, 0.9) is True


async def test_create_new_file_existing_covers_topic(tmp_path):
    evo = _setup_personality(tmp_path)
    with _patch_embed(), _patch_cosine(0.9):  # High similarity → file exists
        assert await evo.should_create_new_file("Core topic", 5, 0.9) is False


# --- get_target_file ---


async def test_get_target_file_finds_match(tmp_path):
    evo = _setup_personality(tmp_path)
    with _patch_embed(), _patch_cosine(0.85):
        result = await evo.get_target_file("I am Zhvusha")
    assert result is not None


async def test_get_target_file_no_match(tmp_path):
    evo = _setup_personality(tmp_path)
    with _patch_embed(), _patch_cosine(0.3):
        result = await evo.get_target_file("completely unrelated")
    assert result is None


# --- evolve_core ---


async def test_evolve_core_adds_insight(tmp_path):
    evo = _setup_personality(tmp_path)
    with _patch_embed(), _patch_cosine(0.2):  # Low similarity → not duplicate
        result = await evo.evolve_core(["I learned about patience"])
    assert result is True
    content = (tmp_path / "personality" / "core.md").read_text()
    assert "patience" in content


async def test_evolve_core_skips_duplicate(tmp_path):
    evo = _setup_personality(tmp_path)
    with _patch_embed(), _patch_cosine(0.9):  # High similarity → duplicate
        result = await evo.evolve_core(["I am Zhvusha again"])
    assert result is False


async def test_evolve_core_snapshots_on_overflow(tmp_path):
    evo = _setup_personality(tmp_path)
    # Write many lines
    core = tmp_path / "personality" / "core.md"
    core.write_text("# Core\n" + "\n".join(f"Line {i}" for i in range(35)))

    with _patch_embed(), _patch_cosine(0.1):
        await evo.evolve_core(["New insight"], max_lines=30)

    # Should have created snapshot
    snapshots = list(
        (tmp_path / "personality" / "history" / "snapshots").glob("core_v*.md")
    )
    assert len(snapshots) == 1


# --- evolve_genes ---


async def test_evolve_genes_adds_annotation(tmp_path):
    evo = _setup_personality(tmp_path)
    await evo.evolve_genes(
        [
            {"gene": "caution", "learned": "means 'check with Nikita'"},
        ]
    )
    content = (tmp_path / "personality" / "genes.md").read_text()
    assert "caution" in content
    assert "check with Nikita" in content
    assert "updated" in content


# --- get_personality_tree_summary ---


async def test_create_new_file_below_threshold(tmp_path):
    """Few mentions AND low importance → no new file."""
    evo = _setup_personality(tmp_path)
    assert await evo.should_create_new_file("topic", 1, 0.5) is False


async def test_get_target_file_skips_pending(tmp_path):
    """Files in .pending/ are excluded."""
    evo = _setup_personality(tmp_path)
    pending = tmp_path / "personality" / ".pending"
    pending.mkdir()
    (pending / "draft.md").write_text("Draft content\n")
    with _patch_embed(), _patch_cosine(0.95):
        result = await evo.get_target_file("Draft content")
    # Should match core.md or genes.md but NOT pending/draft.md
    if result is not None:
        assert ".pending" not in result.parts


async def test_get_target_file_skips_empty(tmp_path):
    """Empty markdown files are skipped."""
    evo = _setup_personality(tmp_path)
    (tmp_path / "personality" / "empty.md").write_text("")
    with _patch_embed(), _patch_cosine(0.3):
        result = await evo.get_target_file("some topic")
    assert result is None


async def test_evolve_core_missing_file(tmp_path):
    """evolve_core returns False if core.md missing."""
    p = tmp_path / "personality"
    p.mkdir()
    evo = PersonalityEvolution(p)
    result = await evo.evolve_core(["insight"])
    assert result is False


# --- suggest_new_dimension ---


async def test_suggest_new_dimension_skills(tmp_path):
    evo = _setup_personality(tmp_path)
    path = await evo.suggest_new_dimension("новый навык программирования", [])
    assert "skills" in str(path)


async def test_suggest_new_dimension_relationships(tmp_path):
    evo = _setup_personality(tmp_path)
    path = await evo.suggest_new_dimension("мой друг Максим", [])
    assert "relationships" in str(path)


async def test_suggest_new_dimension_default(tmp_path):
    evo = _setup_personality(tmp_path)
    path = await evo.suggest_new_dimension("random topic xyz", [])
    assert "insights" in str(path)


# --- _classify_category ---


def test_classify_category_all_types():
    assert PersonalityEvolution._classify_category("мой друг") == "relationships"
    assert PersonalityEvolution._classify_category("новый навык") == "skills"
    assert PersonalityEvolution._classify_category("важный принцип") == "values"
    assert PersonalityEvolution._classify_category("я осознал") == "meta"
    assert PersonalityEvolution._classify_category("xyz abc") == "insights"


# --- get_personality_tree_summary ---


def test_personality_tree_summary(tmp_path):
    evo = _setup_personality(tmp_path)
    summary = evo.get_personality_tree_summary()
    assert "Memory Index" in summary
    assert "I am Zhvusha" in summary
    assert "Genes" in summary


def test_personality_tree_summary_truncation(tmp_path):
    evo = _setup_personality(tmp_path)
    core = tmp_path / "personality" / "core.md"
    core.write_text("x" * 10000, encoding="utf-8")
    summary = evo.get_personality_tree_summary()
    assert "обрезано" in summary


def test_personality_tree_summary_missing_files(tmp_path):
    p = tmp_path / "personality"
    p.mkdir()
    evo = PersonalityEvolution(p)
    summary = evo.get_personality_tree_summary()
    assert summary == ""
