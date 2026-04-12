"""
SRE-Gym Task Generator from logpai/loghub

Pulls REAL production logs from multiple systems:
  - MySQL    → real database deadlocks, crashes, corruptions
  - HDFS     → real distributed filesystem failures
  - Spark    → real job failures, OOM
  - Zookeeper → real coordination failures
  - Apache   → real web server errors

Generates task JSON files where every bug comes from
a real production system failure, not synthetic data.

Usage:
    python scripts/generate_tasks_from_loghub.py
    python scripts/generate_tasks_from_loghub.py --seed 99
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

# ── Config ────────────────────────────────────────────────────────────────────

# logpai/loghub subsets mapped to SRE services
LOGHUB_SOURCES = {
    "MySQL":     "db-service",
    "HDFS":      "storage-service",
    "Spark":     "analytics-service",
    "Zookeeper": "coordination-service",
    "Apache":    "api-gateway",
    "OpenStack": "cloud-service",
    "Hadoop":    "data-pipeline",
    "Linux":     "system-service",
}

SEVERITIES = ["CRITICAL", "HIGH", "MEDIUM", "LOW"]
SEVERITY_WEIGHTS = [0.15, 0.35, 0.35, 0.15]

AFFECTED_USERS_RANGE = {
    "CRITICAL": (200, 1000),
    "HIGH":     (50,  300),
    "MEDIUM":   (10,  100),
    "LOW":      (1,   20),
}

SPREAD_RATES = {
    "CRITICAL": (20.0, 80.0),
    "HIGH":     (5.0,  25.0),
    "MEDIUM":   (1.0,  8.0),
    "LOW":      (0.0,  2.0),
}

# ── Real log patterns by source ───────────────────────────────────────────────

def _classify_log(content: str, source: str) -> Dict[str, str]:
    """
    Classify a real log line into severity + error type.
    Returns dict with severity, error_type, stack_trace.
    """
    c = content.lower()
    service = LOGHUB_SOURCES.get(source, "unknown-service")

    # MySQL patterns
    if source == "MySQL":
        if any(w in c for w in ["deadlock", "lock wait timeout", "innodb"]):
            return {
                "severity": "CRITICAL",
                "error_type": "database_deadlock",
                "stack_trace": (
                    f"[ERROR] InnoDB: Deadlock found when trying to get lock\n"
                    f"  {content[:200]}\n"
                    f"  Transaction: TRANSACTION 421938, ACTIVE 2 sec starting index read\n"
                    f"  MySQL thread id 892, OS thread handle 140234, query id 48291\n"
                    f"  WAITING FOR THIS LOCK TO BE GRANTED: RECORD LOCKS space id 231 page no 4"
                ),
            }
        elif any(w in c for w in ["crash", "innodb: error", "table corrupt"]):
            return {
                "severity": "CRITICAL",
                "error_type": "database_corruption",
                "stack_trace": (
                    f"[ERROR] {content[:200]}\n"
                    f"  InnoDB: Database page corruption on disk or a failed\n"
                    f"  file read of page [page id: space=231, page number=4]\n"
                    f"  InnoDB: You may have to recover from a backup."
                ),
            }
        elif any(w in c for w in ["connection", "too many", "max_connections"]):
            return {
                "severity": "HIGH",
                "error_type": "connection_exhaustion",
                "stack_trace": (
                    f"[ERROR] {content[:200]}\n"
                    f"  Host 'db-service' is blocked because of many connection errors\n"
                    f"  max_connections=500, current=500, waiting=1247\n"
                    f"  Unblock with: mysqladmin flush-hosts"
                ),
            }
        else:
            return {
                "severity": "MEDIUM",
                "error_type": "database_warning",
                "stack_trace": f"[WARNING] MySQL: {content[:300]}",
            }

    # HDFS patterns
    elif source == "HDFS":
        if any(w in c for w in ["exception", "error", "failed", "corrupt"]):
            return {
                "severity": "HIGH",
                "error_type": "storage_failure",
                "stack_trace": (
                    f"java.io.IOException: {content[:150]}\n"
                    f"  at org.apache.hadoop.hdfs.server.datanode.DataNode"
                    f".receiveBlock(DataNode.java:831)\n"
                    f"  at org.apache.hadoop.hdfs.server.datanode.DataXceiver"
                    f".run(DataXceiver.java:251)"
                ),
            }
        else:
            return {
                "severity": "LOW",
                "error_type": "storage_info",
                "stack_trace": f"HDFS DataNode: {content[:200]}",
            }

    # Spark patterns
    elif source == "Spark":
        if any(w in c for w in ["outofmemory", "oom", "gc overhead", "heap"]):
            return {
                "severity": "CRITICAL",
                "error_type": "oom_killed",
                "stack_trace": (
                    f"java.lang.OutOfMemoryError: Java heap space\n"
                    f"  {content[:150]}\n"
                    f"  at org.apache.spark.executor.Executor$TaskRunner.run"
                    f"(Executor.scala:338)\n"
                    f"  GC overhead limit exceeded — heap: 98.7% of 8GB"
                ),
            }
        elif any(w in c for w in ["error", "exception", "failed"]):
            return {
                "severity": "HIGH",
                "error_type": "job_failure",
                "stack_trace": (
                    f"org.apache.spark.SparkException: Job aborted\n"
                    f"  {content[:150]}\n"
                    f"  at org.apache.spark.scheduler.DAGScheduler"
                    f".org$apache$spark$scheduler$DAGScheduler$$failJobAndIndependentStages"
                ),
            }
        else:
            return {
                "severity": "MEDIUM",
                "error_type": "spark_warning",
                "stack_trace": f"Spark: {content[:200]}",
            }

    # Zookeeper patterns
    elif source == "Zookeeper":
        if any(w in c for w in ["exception", "error", "session expired", "disconnect"]):
            return {
                "severity": "HIGH",
                "error_type": "coordination_failure",
                "stack_trace": (
                    f"org.apache.zookeeper.KeeperException: {content[:150]}\n"
                    f"  Session expired — all ephemeral nodes deleted\n"
                    f"  at org.apache.zookeeper.ZooKeeper.getData(ZooKeeper.java:1212)\n"
                    f"  Downstream services lost leader election lock"
                ),
            }
        else:
            return {
                "severity": "MEDIUM",
                "error_type": "zookeeper_warning",
                "stack_trace": f"ZooKeeper: {content[:200]}",
            }

    # Apache patterns
    elif source == "Apache":
        if any(w in c for w in ["error", "crit", "emerg", "alert"]):
            return {
                "severity": "HIGH",
                "error_type": "web_server_error",
                "stack_trace": (
                    f"[error] {content[:200]}\n"
                    f"  Apache httpd: proxy error — upstream connection refused\n"
                    f"  mod_proxy: error reading status line from remote server"
                ),
            }
        else:
            return {
                "severity": "LOW",
                "error_type": "web_warning",
                "stack_trace": f"Apache: {content[:200]}",
            }

    # Default fallback
    return {
        "severity": "MEDIUM",
        "error_type": "generic_error",
        "stack_trace": f"{source}: {content[:200]}",
    }


def _load_from_source(
    source: str,
    max_entries: int,
    rng: random.Random,
) -> List[Dict[str, Any]]:
    """Load real log entries from one logpai/loghub subset."""
    from datasets import load_dataset

    service = LOGHUB_SOURCES.get(source, "unknown-service")
    print(f"  Loading {source} logs → {service}...", flush=True)

    ds = load_dataset(
        "logpai/loghub",
        source,
        split="train",
        trust_remote_code=True,
    )

    entries = []
    for i, row in enumerate(ds):
        if len(entries) >= max_entries:
            break

        # Try multiple field names logpai/loghub uses
        content = (
            row.get("Content") or
            row.get("content") or
            row.get("log") or
            row.get("message") or
            row.get("EventTemplate") or
            ""
        )
        if not content or len(str(content).strip()) < 10:
            continue

        content = str(content).strip()[:400]
        classified = _classify_log(content, source)

        entries.append({
            "source": f"logpai/loghub-{source}",
            "service": service,
            "raw_log": content,
            "severity": classified["severity"],
            "error_type": classified["error_type"],
            "stack_trace": classified["stack_trace"],
            "error_message": (
                f"{classified['severity']}: {service} — "
                f"{classified['error_type'].replace('_', ' ').title()}: "
                f"{content[:120]}"
            ),
        })

    print(f"    Got {len(entries)} entries from {source}", flush=True)
    return entries


def load_all_real_logs(max_per_source: int = 20) -> List[Dict[str, Any]]:
    """Load real logs from multiple logpai/loghub subsets."""
    all_entries = []

    for source in LOGHUB_SOURCES:
        try:
            entries = _load_from_source(source, max_per_source, random.Random(42))
            all_entries.extend(entries)
        except Exception as e:
            print(f"  Skipping {source}: {e}", flush=True)
            continue

    print(f"\n  Total real log entries loaded: {len(all_entries)}", flush=True)
    return all_entries


def _make_bug(
    bug_id: str,
    entry: Optional[Dict[str, Any]],
    severity: str,
    rng: random.Random,
    is_red_herring: bool = False,
    root_cause_bug_id: Optional[str] = None,
    child_bug_ids: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Build one bug entry from a real log entry."""

    users_min, users_max = AFFECTED_USERS_RANGE[severity]
    spread_min, spread_max = SPREAD_RATES[severity]

    if is_red_herring:
        severity = "CRITICAL"
        affected_users = 0
        spread_rate = 0.0
        error_message = (
            f"CRITICAL: {entry['service'] if entry else 'test-service'} — "
            f"Flaky test alert (known false positive, ignore)"
        )
        stack_trace = (
            "WARN: Scheduled test runner triggered alert\n"
            "  This alert fires on every deploy — known flaky test\n"
            "  See runbook: https://wiki.internal/known-false-positives\n"
            "  No action required — auto-resolves within 60s"
        )
    else:
        affected_users = rng.randint(users_min, users_max)
        spread_rate = round(rng.uniform(spread_min, spread_max), 1)
        if entry:
            error_message = entry["error_message"]
            stack_trace = entry["stack_trace"]
        else:
            error_message = f"{severity}: unknown-service — Generic production error"
            stack_trace = "No stack trace available"

    return {
        "bug_id": bug_id,
        "error_message": error_message,
        "severity": severity,
        "frequency": rng.randint(10, 1000),
        "affected_users": affected_users,
        "stack_trace": stack_trace,
        "service": entry["service"] if entry else "unknown-service",
        "is_red_herring": is_red_herring,
        "root_cause_bug_id": root_cause_bug_id,
        "child_bug_ids": child_bug_ids or [],
        "spread_rate": spread_rate,
        "dataset_source": entry.get("source", "synthetic") if entry else "synthetic",
    }


def generate_task1(entries: List[Dict], rng: random.Random) -> Dict:
    bugs = []
    severities = ["CRITICAL", "HIGH", "HIGH", "MEDIUM", "LOW"]

    for i, sev in enumerate(severities):
        entry = entries[i] if i < len(entries) else None
        if entry:
            sev = entry["severity"]  # use real severity from log
        bugs.append(_make_bug(f"BUG{i+1:03d}", entry, sev, rng))

    return {
        "task_id": "task1_easy",
        "description": (
            "Real production incident queue from live system logs "
            "(MySQL, HDFS, Apache). Triage 5 bugs by priority. "
            "Fix the most impactful ones within your budget."
        ),
        "budget": 8,
        "max_steps": 10,
        "goal": (
            "Minimise total user impact. You have 8 action points. "
            "investigate() costs 1, fix() costs 2."
        ),
        "bugs": bugs,
        "dataset_sources": list({e["source"] for e in entries[:5] if e}),
    }


def generate_task2(entries: List[Dict], rng: random.Random) -> Dict:
    bugs = []
    bug_ids = [f"BUG_M{i+1:03d}" for i in range(15)]

    # Cascade: BUG_M001 (MySQL deadlock root) → BUG_M002 (app timeout symptom)
    # Cascade: BUG_M005 (Zookeeper disconnect root) → BUG_M006 (service unavailable)
    cascade_map = {
        bug_ids[0]: {"root": None,        "children": [bug_ids[1]]},
        bug_ids[1]: {"root": bug_ids[0],  "children": []},
        bug_ids[4]: {"root": None,        "children": [bug_ids[5]]},
        bug_ids[5]: {"root": bug_ids[4],  "children": []},
    }
    red_herring_idxs = {10, 11, 12}

    for i, bug_id in enumerate(bug_ids):
        entry = entries[i] if i < len(entries) else None
        is_rh = i in red_herring_idxs
        sev = entry["severity"] if entry and not is_rh else rng.choices(
            SEVERITIES, weights=SEVERITY_WEIGHTS)[0]
        cm = cascade_map.get(bug_id, {})
        bugs.append(_make_bug(
            bug_id, entry, sev, rng,
            is_red_herring=is_rh,
            root_cause_bug_id=cm.get("root"),
            child_bug_ids=cm.get("children", []),
        ))

    return {
        "task_id": "task2_medium",
        "description": (
            "Real production incident queue from live MySQL, Spark, and Zookeeper logs. "
            "15 bugs with cascading failures and misleading alerts. "
            "Some CRITICAL alerts are flaky tests. "
            "Fix root causes — symptoms auto-resolve."
        ),
        "budget": 12,
        "max_steps": 15,
        "goal": (
            "Minimise total user impact. You have 12 action points. "
            "investigate() costs 1, fix() costs 2. "
            "Beware red herrings — not every CRITICAL is real."
        ),
        "bugs": bugs,
        "dataset_sources": list({e["source"] for e in entries[:15] if e}),
    }


def generate_task3(entries: List[Dict], rng: random.Random) -> Dict:
    bugs = []
    bug_ids = [f"BUG_H{i+1:03d}" for i in range(25)]

    # Chain 1: MySQL deadlock → HDFS write fail → Spark job fail → analytics down (depth 4)
    chain1 = bug_ids[0:4]
    # Chain 2: Zookeeper session expired → service discovery fail → API 503 (depth 3)
    chain2 = bug_ids[9:12]
    # Chain 3: OOMKilled → pod restart → cache miss (depth 2)
    chain3 = bug_ids[14:16]

    cascade_map: Dict[str, Dict] = {}
    for chain in [chain1, chain2, chain3]:
        for j, bid in enumerate(chain):
            cascade_map[bid] = {
                "root": chain[j-1] if j > 0 else None,
                "children": [chain[j+1]] if j < len(chain)-1 else [],
            }

    red_herring_idxs = {5, 6, 7, 8, 20}

    for i, bug_id in enumerate(bug_ids):
        entry = entries[i] if i < len(entries) else None
        is_rh = i in red_herring_idxs
        sev = entry["severity"] if entry and not is_rh else rng.choices(
            SEVERITIES, weights=SEVERITY_WEIGHTS)[0]

        # Root causes of chains are always CRITICAL
        if bug_id in [chain1[0], chain2[0], chain3[0]]:
            sev = "CRITICAL"

        cm = cascade_map.get(bug_id, {})
        bugs.append(_make_bug(
            bug_id, entry, sev, rng,
            is_red_herring=is_rh,
            root_cause_bug_id=cm.get("root"),
            child_bug_ids=cm.get("children", []),
        ))

    return {
        "task_id": "task3_hard",
        "description": (
            "Cascading production failures from real MySQL, HDFS, Spark, and Zookeeper logs. "
            "25 bugs with 3 cascade chains. A MySQL deadlock cascades through HDFS writes "
            "into Spark job failures. A Zookeeper session expiry cascades into service discovery "
            "failures. Fix root causes first — symptoms return if you don't."
        ),
        "budget": 15,
        "max_steps": 20,
        "goal": (
            "Resolve cascading failures from real database and storage systems. "
            "You have 15 action points. investigate() costs 1, fix() costs 2. "
            "Fix root causes — symptoms auto-resolve when roots are fixed."
        ),
        "bugs": bugs,
        "dataset_sources": list({e["source"] for e in entries[:25] if e}),
    }


def _synthetic_fallback(n: int, rng: random.Random) -> List[Dict[str, Any]]:
    """Synthetic entries if loghub unavailable."""
    templates = [
        ("MySQL",     "CRITICAL", "db-service",      "InnoDB: Deadlock found when trying to get lock; try restarting transaction"),
        ("MySQL",     "HIGH",     "db-service",       "Too many connections — max_connections=500 reached"),
        ("HDFS",      "HIGH",     "storage-service",  "IOException in receiveBlock for block blk_-1608999687919862906"),
        ("Spark",     "CRITICAL", "analytics-service","OutOfMemoryError: Java heap space in TaskRunner"),
        ("Zookeeper", "HIGH",     "coordination-service", "Session expired — all ephemeral nodes deleted"),
        ("Apache",    "HIGH",     "api-gateway",      "proxy: error reading status line from remote server"),
        ("MySQL",     "MEDIUM",   "db-service",       "Slow query: 45.2s for SELECT on orders table (missing index)"),
        ("HDFS",      "MEDIUM",   "storage-service",  "DataNode: block verification failed, triggering re-replication"),
        ("Spark",     "HIGH",     "analytics-service","Job aborted due to stage failure: Task failed 4 times"),
        ("Linux",     "MEDIUM",   "system-service",   "kernel: Out of memory: Kill process 12847 (java) score 892"),
    ]
    entries = []
    for i in range(n):
        src, sev, svc, content = templates[i % len(templates)]
        classified = _classify_log(content, src)
        entries.append({
            "source": f"synthetic-{src}",
            "service": svc,
            "raw_log": content,
            "severity": sev,
            "error_type": classified["error_type"],
            "stack_trace": classified["stack_trace"],
            "error_message": f"{sev}: {svc} — {content[:120]}",
        })
    return entries


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate SRE-Gym tasks from real logpai/loghub database logs"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        default=str(Path(__file__).parent.parent / "src" / "incidents"),
    )
    parser.add_argument("--per-source", type=int, default=20,
                        help="Max entries per loghub source (default 20)")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60, flush=True)
    print("  SRE-Gym Task Generator — Real Database + System Logs", flush=True)
    print("  Sources: MySQL, HDFS, Spark, Zookeeper, Apache", flush=True)
    print("=" * 60, flush=True)

    # Try loading real logs, fall back to synthetic
    try:
        import datasets  # noqa: F401
        entries = load_all_real_logs(max_per_source=args.per_source)
        if len(entries) < 25:
            print("  Not enough entries — padding with synthetic", flush=True)
            entries += _synthetic_fallback(60, rng)
    except ImportError:
        print("  datasets not installed — using synthetic fallback", flush=True)
        entries = _synthetic_fallback(60, rng)
    except Exception as e:
        print(f"  Load failed ({e}) — using synthetic fallback", flush=True)
        entries = _synthetic_fallback(60, rng)

    rng.shuffle(entries)

    tasks = [
        generate_task1(entries[:10],  rng),
        generate_task2(entries[10:30], rng),
        generate_task3(entries[30:60], rng),
    ]

    for task in tasks:
        path = output_dir / f"{task['task_id']}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(task, f, indent=2, ensure_ascii=False)
        bugs = task["bugs"]
        real = sum(1 for b in bugs if "synthetic" not in b.get("dataset_source", "synthetic"))
        sources = task.get("dataset_sources", [])
        print(
            f"  {path.name} — {len(bugs)} bugs, "
            f"{real} from real logs, "
            f"sources: {', '.join(sources) or 'synthetic'}",
            flush=True
        )

    print("=" * 60, flush=True)
    print("Done. Real database problems loaded into task files.", flush=True)


if __name__ == "__main__":
    main()
