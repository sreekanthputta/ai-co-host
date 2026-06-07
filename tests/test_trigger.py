import asyncio

import pytest

from src.riff.trigger import ForceTrigger


async def test_fire_and_take():
    t = ForceTrigger()
    t.fire(hint="now")
    req = t.take()
    assert req is not None and req.hint == "now"
    assert t.take() is None


async def test_wait_returns_request():
    t = ForceTrigger()
    t.fire(hint="x")
    req = await asyncio.wait_for(t.wait(), timeout=0.5)
    assert req.hint == "x"


async def test_fire_replaces_pending():
    t = ForceTrigger()
    t.fire(hint="first")
    t.fire(hint="second")
    req = t.take()
    assert req is not None and req.hint == "second"


async def test_wait_blocks_until_fire():
    t = ForceTrigger()

    async def fire_later():
        await asyncio.sleep(0.02)
        t.fire(hint="late")

    asyncio.create_task(fire_later())
    req = await asyncio.wait_for(t.wait(), timeout=0.5)
    assert req.hint == "late"
