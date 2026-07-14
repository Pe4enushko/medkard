from storage import IngestRunsStorage


async def test_pending_then_done_records_hash():
    async with IngestRunsStorage() as s:
        await s.upsert_pending("IR1")
        assert (await s.get_all())["IR1"] == ("pending", None)
        await s.mark_done("IR1", "hashA")
        assert (await s.get_all())["IR1"] == ("done", "hashA")


async def test_failed_preserves_last_done_hash():
    async with IngestRunsStorage() as s:
        await s.mark_done("IR2", "hashB")
        await s.upsert_pending("IR2")          # re-run starts
        assert (await s.get_all())["IR2"] == ("pending", "hashB")   # hash preserved
        await s.mark_failed("IR2", "boom")
        status, h = (await s.get_all())["IR2"]
        assert status == "failed" and h == "hashB"                  # hash still preserved
