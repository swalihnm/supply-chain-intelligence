from __future__ import annotations

from sci.manifest import Manifest, file_sha256, text_sha256


def test_file_sha256_is_content_addressed(tmp_path):
    a = tmp_path / "a.txt"
    b = tmp_path / "b.txt"
    a.write_text("hello")
    b.write_text("hello")
    assert file_sha256(a) == file_sha256(b)
    b.write_text("hello ")
    assert file_sha256(a) != file_sha256(b)


def test_text_sha256_is_order_and_boundary_sensitive():
    # The null separator matters: without it ("ab","c") and ("a","bc") collide.
    assert text_sha256("ab", "c") != text_sha256("a", "bc")
    assert text_sha256("a", "b") != text_sha256("b", "a")


def test_manifest_roundtrip_and_staleness(tmp_path):
    mpath = tmp_path / "_manifest.json"
    out = tmp_path / "out.parquet"
    out.write_text("x")

    mf = Manifest(mpath)
    assert not mf.is_current("t", "sha1", out)

    mf.record("t", "sha1", out, rows=10)
    assert mf.is_current("t", "sha1", out)

    # A changed input invalidates the entry.
    assert not mf.is_current("t", "sha2", out)

    # A deleted output invalidates it too, even with a matching input.
    out.unlink()
    assert not mf.is_current("t", "sha1", out)


def test_manifest_persists_across_instances(tmp_path):
    mpath = tmp_path / "_manifest.json"
    out = tmp_path / "out.parquet"
    out.write_text("x")

    Manifest(mpath).record("t", "sha1", out, rows=10, source="a.csv")

    reloaded = Manifest(mpath)
    assert reloaded.is_current("t", "sha1", out)
    entry = reloaded.get("t")
    assert entry["rows"] == 10
    assert entry["source"] == "a.csv"
    assert reloaded.keys() == ["t"]
