"""
KnowledgeUpdater — crawls ArXiv + Semantic Scholar + GitHub for LLM eval papers.
Updates SECOND-KNOWLEDGE-BRAIN.md weekly.
"""

import asyncio
import hashlib
import logging
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import aiohttp

ROOT = Path(__file__).parent.parent
KNOWLEDGE_BRAIN = ROOT / "SECOND-KNOWLEDGE-BRAIN.md"
logger = logging.getLogger(__name__)

ARXIV_CATEGORIES = ["cs.AI", "cs.LG"]
ARXIV_QUERIES = [
    "LLM evaluation benchmark",
    "language model evaluation methodology",
    "AI benchmark design",
    "LLM judge evaluation framework",
    "model comparison metrics",
]

SEMANTIC_SCHOLAR_QUERIES = [
    "HELM holistic evaluation language models",
    "MT-Bench multi-turn LLM evaluation",
    "LMSys chatbot arena ELO rating",
    "BIG-Bench language model benchmark",
    "operational LLM metrics production evaluation",
]

GITHUB_REPOS = [
    "EleutherAI/lm-evaluation-harness",
    "stanford-crfm/helm",
    "lm-sys/FastChat",
    "openai/evals",
]


class PaperEntry:
    def __init__(
        self,
        title: str,
        authors: str,
        year: str,
        venue: str,
        url: str,
        abstract: str,
    ):
        self.title = title
        self.authors = authors
        self.year = year
        self.venue = venue
        self.url = url
        self.abstract = abstract
        self.relevance_score = 0.0
        self.recency_score = 0.0
        self.total_score = 0.0

    @property
    def hash(self) -> str:
        return hashlib.sha256((self.title + self.url).encode()).hexdigest()[:16]

    def to_table_row(self) -> str:
        abstract_short = self.abstract[:150].replace("|", " ").replace("\n", " ")
        if len(self.abstract) > 150:
            abstract_short += "…"
        return (
            f"| {self.title[:60]} | {self.authors[:30]} | {self.year} | "
            f"{self.venue[:20]} | [{self.url[:40]}]({self.url}) | {abstract_short} | Auto-crawled |"
        )


RELEVANCE_KEYWORDS = [
    "benchmark", "evaluation", "LLM", "language model", "assessment",
    "metric", "judge", "ELO", "HELM", "MT-Bench", "BIG-Bench",
    "hallucination", "quality", "latency", "cost", "performance",
]


class KnowledgeUpdater:
    def __init__(self, config: dict, memory):
        self.config = config
        self.memory = memory
        self.top_n = config.get("knowledge_updater", {}).get("top_n", 10)
        self.recency_window_days = config.get("knowledge_updater", {}).get("recency_window_days", 90)

    async def run_update(self) -> Dict[str, Any]:
        logger.info("Knowledge update started")
        all_papers: List[PaperEntry] = []

        tasks = [
            self._crawl_arxiv(),
            self._crawl_semantic_scholar(),
            self._crawl_github_releases(),
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for result in results:
            if isinstance(result, list):
                all_papers.extend(result)
            elif isinstance(result, Exception):
                logger.warning(f"Crawl task error: {result}")

        scored = self._score_papers(all_papers)
        deduped = self._deduplicate(scored)
        top_papers = sorted(deduped, key=lambda p: p.total_score, reverse=True)[: self.top_n]

        added = self._append_to_brain(top_papers)
        logger.info(f"Knowledge update complete: {added} new papers added")
        return {
            "papers_found": len(all_papers),
            "papers_after_dedup": len(deduped),
            "papers_added": added,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    async def _crawl_arxiv(self) -> List[PaperEntry]:
        papers = []
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as session:
            for query in ARXIV_QUERIES[:3]:
                for cat in ARXIV_CATEGORIES:
                    url = (
                        f"https://export.arxiv.org/api/query"
                        f"?search_query=cat:{cat}+AND+all:{query.replace(' ', '+')}"
                        f"&start=0&max_results=10&sortBy=submittedDate&sortOrder=descending"
                    )
                    try:
                        async with session.get(url) as resp:
                            if resp.status != 200:
                                continue
                            text = await resp.text()
                        root = ET.fromstring(text)
                        ns = {"atom": "http://www.w3.org/2005/Atom"}
                        for entry in root.findall("atom:entry", ns):
                            title = (entry.find("atom:title", ns).text or "").strip().replace("\n", " ")
                            abstract = (entry.find("atom:summary", ns).text or "").strip().replace("\n", " ")
                            link_elem = entry.find("atom:link[@rel='alternate']", ns)
                            url_val = link_elem.get("href", "") if link_elem is not None else ""
                            authors_list = [
                                a.find("atom:name", ns).text
                                for a in entry.findall("atom:author", ns)
                                if a.find("atom:name", ns) is not None
                            ]
                            authors_str = ", ".join(authors_list[:3])
                            if len(authors_list) > 3:
                                authors_str += " et al."
                            published = entry.find("atom:published", ns)
                            year = published.text[:4] if published is not None and published.text else "2024"
                            if title:
                                papers.append(PaperEntry(title, authors_str, year, f"ArXiv {cat}", url_val, abstract))
                    except Exception as exc:
                        logger.debug(f"ArXiv crawl error for {query}/{cat}: {exc}")
        return papers

    async def _crawl_semantic_scholar(self) -> List[PaperEntry]:
        papers = []
        base = "https://api.semanticscholar.org/graph/v1/paper/search"
        fields = "title,authors,year,venue,externalIds,abstract"
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as session:
            for query in SEMANTIC_SCHOLAR_QUERIES:
                params = {"query": query, "limit": 8, "fields": fields}
                try:
                    async with session.get(base, params=params) as resp:
                        if resp.status != 200:
                            continue
                        data = await resp.json()
                    for item in data.get("data", []):
                        title = item.get("title", "")
                        authors_list = [a.get("name", "") for a in item.get("authors", [])[:3]]
                        authors_str = ", ".join(authors_list)
                        year = str(item.get("year", "2024"))
                        venue = item.get("venue", "Unknown")
                        ext_ids = item.get("externalIds", {})
                        doi = ext_ids.get("DOI", "")
                        arxiv_id = ext_ids.get("ArXiv", "")
                        url_val = (
                            f"https://arxiv.org/abs/{arxiv_id}"
                            if arxiv_id
                            else (f"https://doi.org/{doi}" if doi else "")
                        )
                        abstract = (item.get("abstract") or "")[:300]
                        if title and url_val:
                            papers.append(PaperEntry(title, authors_str, year, venue, url_val, abstract))
                except Exception as exc:
                    logger.debug(f"Semantic Scholar error for '{query}': {exc}")
        return papers

    async def _crawl_github_releases(self) -> List[PaperEntry]:
        papers = []
        async with aiohttp.ClientSession(
            headers={"User-Agent": "ai-benchmark-agent/1.0"},
            timeout=aiohttp.ClientTimeout(total=15),
        ) as session:
            for repo in GITHUB_REPOS:
                url = f"https://api.github.com/repos/{repo}/releases?per_page=3"
                try:
                    async with session.get(url) as resp:
                        if resp.status != 200:
                            continue
                        releases = await resp.json()
                    for rel in releases:
                        tag = rel.get("tag_name", "")
                        name = rel.get("name", tag)
                        published = rel.get("published_at", "")[:4]
                        html_url = rel.get("html_url", "")
                        body = (rel.get("body") or "")[:200]
                        title = f"{repo} {name} release"
                        papers.append(
                            PaperEntry(title, repo, published, "GitHub Release", html_url, body)
                        )
                except Exception as exc:
                    logger.debug(f"GitHub release error for {repo}: {exc}")
        return papers

    def _score_papers(self, papers: List[PaperEntry]) -> List[PaperEntry]:
        now = datetime.now(timezone.utc)
        cutoff = now - timedelta(days=self.recency_window_days)
        for paper in papers:
            try:
                year = int(paper.year)
                pub_date = datetime(year, 1, 1, tzinfo=timezone.utc)
                days_old = max(0, (now - pub_date).days)
                paper.recency_score = max(0.0, 1.0 - days_old / (self.recency_window_days * 2))
            except ValueError:
                paper.recency_score = 0.5

            text = (paper.title + " " + paper.abstract).lower()
            keyword_hits = sum(1 for kw in RELEVANCE_KEYWORDS if kw.lower() in text)
            paper.relevance_score = min(1.0, keyword_hits / 5.0)
            paper.total_score = 0.6 * paper.recency_score + 0.4 * paper.relevance_score
        return papers

    def _deduplicate(self, papers: List[PaperEntry]) -> List[PaperEntry]:
        seen = set()
        deduped = []
        for paper in papers:
            h = paper.hash
            if self.memory.is_known_paper(h):
                continue
            if h not in seen:
                seen.add(h)
                deduped.append(paper)
        return deduped

    def _append_to_brain(self, papers: List[PaperEntry]) -> int:
        if not papers:
            return 0

        if not KNOWLEDGE_BRAIN.exists():
            KNOWLEDGE_BRAIN.write_text("# SECOND-KNOWLEDGE-BRAIN\n\n## Key Research Papers\n\n", encoding="utf-8")

        content = KNOWLEDGE_BRAIN.read_text(encoding="utf-8")
        log_marker = "## Knowledge Update Log"

        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        new_lines = [f"\n### {date_str} — Auto-Crawl ({len(papers)} new papers)\n"]
        table_header = "| Title | Authors | Year | Venue | Link | Key Finding | Relevance |\n|-------|---------|------|-------|------|-------------|-----------|"
        new_lines.append(table_header)
        for paper in papers:
            new_lines.append(paper.to_table_row())
            self.memory.mark_paper_known(paper.hash, paper.title)

        new_block = "\n".join(new_lines) + "\n"

        if log_marker in content:
            updated = content.replace(log_marker, new_block + "\n" + log_marker, 1)
        else:
            updated = content + "\n" + log_marker + "\n" + new_block

        KNOWLEDGE_BRAIN.write_text(updated, encoding="utf-8")
        return len(papers)

    def start_scheduled(self):
        try:
            from apscheduler.schedulers.asyncio import AsyncIOScheduler
            scheduler = AsyncIOScheduler()
            scheduler.add_job(
                self.run_update,
                "cron",
                day_of_week="sun",
                hour=2,
                minute=0,
                id="knowledge_update",
            )
            scheduler.start()
            logger.info("Knowledge updater scheduled: weekly Sunday 02:00")
            return scheduler
        except ImportError:
            logger.warning("APScheduler not installed; scheduled knowledge update disabled")
            return None


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(ROOT))
    from agent.memory.memory_manager import BenchmarkMemoryManager

    async def main():
        mem = BenchmarkMemoryManager(ROOT / "data" / "benchmark.db")
        mem.init_db()
        updater = KnowledgeUpdater({}, mem)
        result = await updater.run_update()
        print(result)

    asyncio.run(main())
