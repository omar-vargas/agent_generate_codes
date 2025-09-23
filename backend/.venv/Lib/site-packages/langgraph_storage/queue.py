import asyncio
import os

import structlog
from blockbuster import BlockBuster
from langsmith import env as ls_env

from langgraph_api.config import (
    BG_JOB_HEARTBEAT,
    N_JOBS_PER_WORKER,
    STATS_INTERVAL_SECS,
)
from langgraph_api.graph import is_js_graph
from langgraph_api.schema import Run
from langgraph_api.webhook import call_webhook
from langgraph_api.worker import WorkerResult, worker
from langgraph_storage.database import connect
from langgraph_storage.ops import Runs

logger = structlog.stdlib.get_logger(__name__)

WORKERS: set[asyncio.Task] = set()
SHUTDOWN_GRACE_PERIOD_SECS = 5


async def queue():
    concurrency = N_JOBS_PER_WORKER
    loop = asyncio.get_running_loop()
    last_stats_secs: int | None = None
    last_sweep_secs: int | None = None
    semaphore = asyncio.Semaphore(concurrency)
    WEBHOOKS: set[asyncio.Task] = set()
    enable_blocking = os.getenv("LANGGRAPH_ALLOW_BLOCKING", "false").lower() == "true"

    # raise exceptions when a blocking call is detected inside an async function
    if enable_blocking:
        bb = None
        await logger.awarning(
            "Heads up: You've set --allow-blocking, which allows synchronous blocking I/O operations."
            " Be aware that blocking code in one run may tie up the shared event loop"
            " and slow down ALL other server operations. For best performance, either convert blocking"
            " code to async patterns or set BG_JOB_ISOLATED_LOOPS=true in production"
            " to isolate each run in its own event loop."
        )
    else:
        bb = _enable_blockbuster()

    def cleanup(task: asyncio.Task):
        WORKERS.remove(task)
        semaphore.release()
        try:
            if task.cancelled():
                return
            exc = task.exception()
            if exc and not isinstance(exc, asyncio.CancelledError):
                logger.exception(
                    f"Background worker failed for task {task}", exc_info=exc
                )
                return
            result: WorkerResult | None = task.result()
            if result and result["webhook"]:
                hook_task = loop.create_task(
                    call_webhook(result),
                    name=f"webhook-{result['run']['run_id']}",
                )
                WEBHOOKS.add(hook_task)
                hook_task.add_done_callback(WEBHOOKS.remove)
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            logger.exception("Background worker cleanup failed", exc_info=exc)

    await logger.ainfo(f"Starting {concurrency} background workers")
    try:
        run: Run | None = None
        while True:
            try:
                # check if we need to sweep runs
                do_sweep = (
                    last_sweep_secs is None
                    or loop.time() - last_sweep_secs > BG_JOB_HEARTBEAT * 2
                )
                # check if we need to update stats
                if calc_stats := (
                    last_stats_secs is None
                    or loop.time() - last_stats_secs > STATS_INTERVAL_SECS
                ):
                    last_stats_secs = loop.time()
                    active = len(WORKERS)
                    await logger.ainfo(
                        "Worker stats",
                        max=concurrency,
                        available=concurrency - active,
                        active=active,
                    )
                # wait for semaphore to respect concurrency
                await semaphore.acquire()
                # skip the wait, if 1st time, or got a run last time
                wait = run is None and last_stats_secs is not None
                # try to get a run, handle it
                run = None
                async for run, attempt in Runs.next(wait=wait, limit=1):
                    graph_id = (
                        run["kwargs"]
                        .get("config", {})
                        .get("configurable", {})
                        .get("graph_id")
                    )

                    if graph_id and is_js_graph(graph_id):
                        task_name = f"js-run-{run['run_id']}-attempt-{attempt}"
                    else:
                        task_name = f"run-{run['run_id']}-attempt-{attempt}"
                    task = asyncio.create_task(
                        worker(run, attempt, loop),
                        name=task_name,
                    )
                    task.add_done_callback(cleanup)
                    WORKERS.add(task)
                else:
                    semaphore.release()
                # run stats and sweep if needed
                if calc_stats or do_sweep:
                    async with connect() as conn:
                        # update stats if needed
                        if calc_stats:
                            stats = await Runs.stats(conn)
                            await logger.ainfo("Queue stats", **stats)
                        # sweep runs if needed
                        if do_sweep:
                            last_sweep_secs = loop.time()
                            run_ids = await Runs.sweep(conn)
                            logger.info("Sweeped runs", run_ids=run_ids)
            except Exception as exc:
                # keep trying to run the scheduler indefinitely
                logger.exception("Background worker scheduler failed", exc_info=exc)
                semaphore.release()
                await exit.aclose()
    finally:
        if bb:
            bb.deactivate()
        logger.info("Shutting down background workers")
        for task in WORKERS:
            task.cancel("Shutting down background workers.")
        for task in WEBHOOKS:
            task.cancel("Shutting down webhooks for background workers.")
        await asyncio.wait_for(
            asyncio.gather(*WORKERS, *WEBHOOKS, return_exceptions=True),
            SHUTDOWN_GRACE_PERIOD_SECS,
        )


def _enable_blockbuster() -> BlockBuster:
    ls_env.get_runtime_environment()  # this gets cached
    bb = BlockBuster(excluded_modules=[])
    bb.activate()
    # Note, we've cached this call in langsmith==0.3.21 so it shouldn't raise anyway
    # but we don't want to raise teh minbound just for that.
    bb.functions["os.stat"].deactivate()
    for function in bb.functions:
        if function.startswith("os.path."):
            bb.functions[function].deactivate()
    for module, func in (
        # Note, we've cached this call in langsmith==0.3.21 so it shouldn't raise anyway
        # but we don't want to raise teh minbound just for that.
        ("langsmith/client.py", "_default_retry_config"),
        # Only triggers in python 3.11 for getting subgraphs
        # Will be unnecessary once we cache the assistant schemas
        ("langgraph/pregel/utils.py", "get_function_nonlocals"),
        ("importlib/metadata/__init__.py", "metadata"),
        ("importlib/metadata/__init__.py", "read_text"),
    ):
        bb.functions["io.TextIOWrapper.read"].can_block_in(module, func)

    bb.functions["os.path.abspath"].can_block_in("inspect.py", "getmodule")

    for module, func in (
        ("uvicorn/lifespan/on.py", "startup"),
        ("uvicorn/lifespan/on.py", "shutdown"),
        ("ansitowin32.py", "write_plain_text"),
        ("logging/__init__.py", "flush"),
        ("logging/__init__.py", "emit"),
    ):
        bb.functions["io.TextIOWrapper.write"].can_block_in(module, func)
        bb.functions["io.BufferedWriter.write"].can_block_in(module, func)
    return bb
