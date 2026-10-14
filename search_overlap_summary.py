"""Summarize how much the searches in a config's database overlap, as a small HTML page.

Usage: python search_overlap_summary.py [config_db]   (default: scrape_cache_sel_v2.db)

A "search" is one row of search_queries; its products are the rows of search_hits
with the same filters. Overlap = how many searches returned each unique product.
"""
import html
import sqlite3
import statistics
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

DEFAULT_DB = "scrape_cache_sel_v2.db"
SEARCH_RESULT_CAP = 1000


def load_stats(db_path: Path) -> dict:
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        n_searches, n_capped, tpt_total, returned = con.execute(
            "SELECT COUNT(*), SUM(capped), SUM(total_count), SUM(returned) FROM search_queries"
        ).fetchone()
        max_returned = con.execute("SELECT MAX(returned) FROM search_queries").fetchone()[0]
        per_product: List[int] = [r[0] for r in con.execute("SELECT COUNT(*) FROM search_hits GROUP BY product_id")]
    finally:
        con.close()

    n_products = len(per_product)
    total_hits = sum(per_product)
    return {
        "n_searches": n_searches,
        "n_capped": n_capped or 0,
        "tpt_total": tpt_total or 0,
        "returned": returned or 0,
        "max_returned": max_returned or 0,
        "n_products": n_products,
        "total_hits": total_hits,
        "mean_per_product": total_hits / n_products,
        "median_per_product": statistics.median(per_product),
        "single": sum(1 for n in per_product if n == 1),
    }


def pct(x: float, digits: int = 1) -> str:
    return f"{100 * x:.{digits}f}%"


def render(db_name: str, s: dict) -> str:
    repeats = 1 - s["n_products"] / s["total_hits"]
    cards: List[Tuple[str, str]] = [
        (f"{s['n_products']:,}", "unique products"),
        (f"{s['total_hits']:,}", "total search hits"),
        (f"{s['n_searches']:,}", "searches"),
        (f"{s['mean_per_product']:.1f}x", "avg. searches per product"),
        (pct(repeats), "of hits are repeats"),
        (pct(s["single"] / s["n_products"]), "products found by only 1 search"),
        (pct(s["n_capped"] / s["n_searches"]), "searches hit the result cap"),
    ]
    bullets = [
        f"<b>{s['n_products']:,}</b> unique products were found from <b>{s['total_hits']:,}</b> search hits across "
        f"<b>{s['n_searches']:,}</b> searches. Each product was returned by <b>{s['mean_per_product']:.1f}</b> searches on average "
        f"(median {s['median_per_product']:.0f}); <b>{pct(repeats)}</b> of hits were repeats.",
        f"<b>{s['single']:,}</b> products ({pct(s['single'] / s['n_products'])}) were found by exactly one search; "
        f"<b>{pct(1 - s['single'] / s['n_products'])}</b> were found by two or more.",
        f"{s['n_capped']:,} of {s['n_searches']:,} searches ({pct(s['n_capped'] / s['n_searches'])}) hit the result cap "
        f"(max {s['max_returned']:,} per search). Together they retrieved {pct(s['returned'] / s['tpt_total'], 2)} of the matches TPT reported "
        f"({s['returned']:,} of {s['tpt_total']:,}), so overlap describes the top-ranked slice of each search, not the full catalog.",
    ]
    card_html = "".join(f"<div class='card'><div class='v'>{v}</div><div class='l'>{html.escape(l)}</div></div>" for v, l in cards)
    bullet_html = "".join(f"<li>{b}</li>" for b in bullets)
    return f"""<!doctype html><html><head><meta charset="utf-8"><title>Search overlap summary - {html.escape(db_name)}</title><style>
body{{font:14px/1.5 -apple-system,Segoe UI,Roboto,sans-serif;max-width:1000px;margin:0 auto;padding:24px;background:#f6f7f9;color:#1f2933}}
h1{{font-size:22px;margin:0 0 4px}}.muted{{color:#6b7280;font-size:12px}}
.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:12px;margin:16px 0}}
.card{{background:#fff;border:1px solid #e5e7eb;border-radius:8px;padding:12px 14px}}
.card .v{{font-size:22px;font-weight:700;color:#1e3a8a}}.card .l{{font-size:12px;color:#52606d}}
li{{margin:6px 0}}</style></head><body>
<h1>Search overlap summary: {html.escape(db_name)}</h1>
<div class="muted">Generated {datetime.now():%Y-%m-%d %H:%M} from search_hits and search_queries</div>
<div class="cards">{card_html}</div><ul>{bullet_html}</ul></body></html>"""


def main() -> None:
    db_path = Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).parent / DEFAULT_DB)
    if not db_path.exists():
        sys.exit(f"Database not found: {db_path}")
    out_path = db_path.with_name(f"{db_path.stem}_overlap_summary.html")
    out_path.write_text(render(db_path.name, load_stats(db_path)), encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
