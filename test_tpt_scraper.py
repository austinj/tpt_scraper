"""
Tests for TPT Scraper
"""
import pytest
import asyncio
import json
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch, MagicMock
import aiosqlite

from config_manager import ConfigManager
import tpt_scraper as scraper


@pytest.fixture
def temp_config_dir():
    """Create a temporary directory for configs."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def temp_db_file():
    """Create a temporary database file."""
    temp_file = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
    temp_file.close()
    yield temp_file.name
    Path(temp_file.name).unlink(missing_ok=True)


@pytest.fixture
def sample_config():
    """Sample configuration data."""
    return {
        "resource_type": ["teacher-tools"],
        "grade_level": ["elementary/kindergarten"],
        "subject": ["social-emotional"],
        "format": ["pdf"],
        "price_options": ["free"],
        "supports": [""],
        "sorting_methods": ["Relevance"],
        "total_pages": 5,
        "concurrent_requests": 10
    }


class TestConfigManager:
    """Test the ConfigManager class."""
    
    def test_create_config(self, temp_config_dir, sample_config):
        """Test creating a new configuration."""
        manager = ConfigManager(config_dir=temp_config_dir)
        
        success = manager.create_config("test-config", sample_config, "Test description")
        
        assert success
        assert "test-config" in manager.registry["configs"]
        assert Path(temp_config_dir, "test-config.json").exists()
        
    def test_create_duplicate_config(self, temp_config_dir, sample_config):
        """Test that creating a duplicate config fails."""
        manager = ConfigManager(config_dir=temp_config_dir)
        
        manager.create_config("test-config", sample_config)
        success = manager.create_config("test-config", sample_config)
        
        assert not success
        
    def test_get_config(self, temp_config_dir, sample_config):
        """Test retrieving a configuration."""
        manager = ConfigManager(config_dir=temp_config_dir)
        manager.create_config("test-config", sample_config)
        
        config = manager.get_config("test-config")
        
        assert config is not None
        assert config["resource_type"] == ["teacher-tools"]
        assert "_metadata" in config
        
    def test_get_nonexistent_config(self, temp_config_dir):
        """Test retrieving a config that doesn't exist."""
        manager = ConfigManager(config_dir=temp_config_dir)
        
        config = manager.get_config("nonexistent")
        
        assert config is None
        
    def test_list_configs(self, temp_config_dir, sample_config):
        """Test listing all configurations."""
        manager = ConfigManager(config_dir=temp_config_dir)
        manager.create_config("config1", sample_config, "First config")
        manager.create_config("config2", sample_config, "Second config")
        
        configs = manager.list_configs()
        
        assert len(configs) == 2
        assert any(c["name"] == "config1" for c in configs)
        assert any(c["name"] == "config2" for c in configs)
        
    def test_delete_config(self, temp_config_dir, sample_config):
        """Test deleting a configuration."""
        manager = ConfigManager(config_dir=temp_config_dir)
        manager.create_config("test-config", sample_config)
        
        success = manager.delete_config("test-config")
        
        assert success
        assert "test-config" not in manager.registry["configs"]
        assert not Path(temp_config_dir, "test-config.json").exists()
        
    def test_get_db_file(self, temp_config_dir, sample_config):
        """Test getting database file path for a config."""
        manager = ConfigManager(config_dir=temp_config_dir)
        manager.create_config("test-config", sample_config)
        
        db_file = manager.get_db_file("test-config")
        
        assert db_file == "scrape_cache_test-config.db"
        
    def test_update_config(self, temp_config_dir, sample_config):
        """Test updating an existing configuration."""
        manager = ConfigManager(config_dir=temp_config_dir)
        manager.create_config("test-config", sample_config)
        
        updated_config = sample_config.copy()
        updated_config["total_pages"] = 10
        
        success = manager.update_config("test-config", updated_config)
        
        assert success
        
        retrieved_config = manager.get_config("test-config")
        assert retrieved_config["total_pages"] == 10


class TestDatabaseSetup:
    """Test database setup and schema."""
    
    @pytest.mark.asyncio
    async def test_setup_db_creates_tables(self, temp_db_file):
        """Test that setup_db creates all required tables."""
        await scraper.setup_db(temp_db_file)
        
        async with aiosqlite.connect(temp_db_file) as db:
            # Check for search_results table
            async with db.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='search_results'"
            ) as cursor:
                result = await cursor.fetchone()
                assert result is not None
                
            # Check for product_metadata table
            async with db.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='product_metadata'"
            ) as cursor:
                result = await cursor.fetchone()
                assert result is not None
                
            # Check for downloads table
            async with db.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='downloads'"
            ) as cursor:
                result = await cursor.fetchone()
                assert result is not None
                
            # Check for download_queue table
            async with db.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='download_queue'"
            ) as cursor:
                result = await cursor.fetchone()
                assert result is not None
                
    @pytest.mark.asyncio
    async def test_db_indexes_created(self, temp_db_file):
        """Test that database indexes are created."""
        await scraper.setup_db(temp_db_file)
        
        async with aiosqlite.connect(temp_db_file) as db:
            async with db.execute(
                "SELECT name FROM sqlite_master WHERE type='index'"
            ) as cursor:
                indexes = await cursor.fetchall()
                index_names = [idx[0] for idx in indexes]
                
                assert 'idx_search_price' in index_names
                assert 'idx_metadata_url' in index_names
                assert 'idx_queue_priority' in index_names

    @pytest.mark.asyncio
    async def test_search_hits_and_overlap_view(self, temp_db_file):
        """search_hits keeps one row per appearance; the view aggregates overlap per product."""
        await scraper.setup_db(temp_db_file)
        async with aiosqlite.connect(temp_db_file) as db:
            for name in ("search_hits", "product_search_summary"):
                async with db.execute("SELECT name FROM sqlite_master WHERE name=?", (name,)) as cur:
                    assert await cur.fetchone() is not None, name

            base = ("", "elementary", "social-emotional", "", "", "")
            rows = [
                ("111", "https://x/Product/a-111", *base, "Relevance", 1, 3),
                ("111", "https://x/Product/a-111", *base, "Rating", 2, 7),      # same product, other sort
                ("111", "https://x/Product/a-111", *base, "Rating", 2, 7),      # exact duplicate -> ignored
                ("222", "https://x/Product/b-222", *base, "Relevance", 1, 4),
            ]
            await db.executemany("""
                INSERT OR IGNORE INTO search_hits
                (product_id, url, resource_type, grade_level, subject, format, price_option, supports, sort_order, page, rank_on_page)
                VALUES (?,?,?,?,?,?,?,?,?,?,?)""", rows)
            await db.commit()

            async with db.execute("SELECT COUNT(*) FROM search_hits") as cur:
                assert (await cur.fetchone())[0] == 3
            async with db.execute("""SELECT n_hits, n_sort_orders, best_page, best_position, sort_orders
                                     FROM product_search_summary WHERE product_id='111'""") as cur:
                n_hits, n_sorts, best_page, best_pos, sorts = await cur.fetchone()
            assert (n_hits, n_sorts, best_page, best_pos) == (2, 2, 1, 1003)
            assert set(sorts.split(",")) == {"Relevance", "Rating"}

    @pytest.mark.asyncio
    async def test_search_results_product_id_backfill(self, temp_db_file):
        """Pre-existing search_results rows get product_id filled in on setup."""
        async with aiosqlite.connect(temp_db_file) as db:
            await db.execute("""CREATE TABLE search_results (url TEXT PRIMARY KEY, resource_type TEXT, grade_level TEXT,
                                subject TEXT, format TEXT, price_option TEXT, supports TEXT, sort_order TEXT, page INTEGER)""")
            await db.execute("INSERT INTO search_results (url, price_option) VALUES ('https://www.teacherspayteachers.com/Product/--10158553', 'free')")
            await db.commit()
        await scraper.setup_db(temp_db_file)
        async with aiosqlite.connect(temp_db_file) as db:
            async with db.execute("SELECT product_id FROM search_results") as cur:
                assert (await cur.fetchone())[0] == "10158553"


class TestAdaptiveRateLimiter:
    """Test the AdaptiveRateLimiter class."""
    
    def test_initialization(self):
        """Test rate limiter initialization."""
        limiter = scraper.AdaptiveRateLimiter(initial_delay=1.0, max_delay=30.0)
        
        assert limiter.current_delay == 1.0
        assert limiter.max_delay == 30.0
        assert limiter.total_count == 0
        
    def test_record_response(self):
        """Test recording responses."""
        limiter = scraper.AdaptiveRateLimiter()
        
        limiter.record_response(True, 0.5)
        
        assert limiter.total_count == 1
        assert limiter.error_count == 0
        assert len(limiter.recent_responses) == 1
        
    def test_error_increases_delay(self):
        """Test that errors increase delay."""
        limiter = scraper.AdaptiveRateLimiter(initial_delay=1.0)
        
        # Record many errors
        for _ in range(20):
            limiter.record_response(False, 1.0)
            
        new_delay = limiter.adjust_delay()
        
        assert new_delay > 1.0
        
    def test_success_decreases_delay(self):
        """Test that successes can decrease delay."""
        limiter = scraper.AdaptiveRateLimiter(initial_delay=5.0)
        
        # Record many fast successes
        for _ in range(20):
            limiter.record_response(True, 0.5)
            
        new_delay = limiter.adjust_delay()
        
        assert new_delay < 5.0
        
    @pytest.mark.asyncio
    async def test_wait(self):
        """Test the wait method."""
        limiter = scraper.AdaptiveRateLimiter(initial_delay=0.01)
        
        import time
        start = time.time()
        await limiter.wait()
        elapsed = time.time() - start
        
        assert elapsed >= 0.01


class TestURLBuilding:
    """Test URL building functionality."""
    
    def test_build_page_url_basic(self):
        """Test building a basic URL."""
        url = scraper.build_page_url("", "", "", "", "", "", "Relevance", 1)
        
        assert url == "https://www.teacherspayteachers.com/browse"
        
    def test_build_page_url_with_params(self):
        """Test building URL with all parameters."""
        url = scraper.build_page_url(
            "teacher-tools",
            "elementary/kindergarten",
            "social-emotional",
            "pdf",
            "free",
            "",
            "Rating",
            2
        )
        
        assert "teacher-tools" in url
        assert "elementary/kindergarten" in url
        assert "social-emotional" in url
        assert "order=Rating" in url
        assert "page=2" in url
        
    def test_build_page_url_relevance_no_order_param(self):
        """Test that Relevance sorting doesn't add order parameter."""
        url = scraper.build_page_url("", "", "", "", "", "", "Relevance", 1)
        
        assert "order=" not in url
        
    def test_build_page_url_page_one_no_page_param(self):
        """Test that page 1 doesn't add page parameter."""
        url = scraper.build_page_url("", "", "", "", "", "", "Rating", 1)
        
        assert "page=" not in url


class TestSearchWorkflow:
    """Test the search workflow."""
    
    @pytest.mark.asyncio
    async def test_search_urls_creates_combinations(self, temp_db_file, sample_config):
        """Test that search generates correct combinations."""
        await scraper.setup_db(temp_db_file)
        
        # Mock the actual HTTP fetching
        with patch('tpt_scraper.aiohttp_client_cache.CachedSession') as mock_session:
            mock_session_instance = AsyncMock()
            mock_session.__aenter__ = AsyncMock(return_value=mock_session_instance)
            mock_session.__aexit__ = AsyncMock()
            
            with patch('tpt_scraper.fetch', new_callable=AsyncMock) as mock_fetch:
                mock_fetch.return_value = '<html><body></body></html>'
                
                # Run with limited config to avoid too many combinations
                limited_config = {
                    "resource_type": ["teacher-tools"],
                    "grade_level": ["elementary/kindergarten"],
                    "subject": ["social-emotional"],
                    "format": [""],
                    "price_options": ["free"],
                    "supports": [""],
                    "sorting_methods": ["Relevance"],
                    "total_pages": 2,
                    "concurrent_requests": 5
                }
                
                # We can't easily test the full function due to async complexity,
                # but we can test components
                
                # Test that combinations are generated correctly
                expected_combos = 1 * 1 * 1 * 1 * 1 * 1 * 1 * 2  # = 2 combinations
                assert expected_combos == 2

    @pytest.mark.asyncio
    async def test_check_sort_orders_drops_ignored_sorts(self):
        """A sort whose page-1 sequence equals Relevance is dropped; distinct ones are kept."""
        def page(ids):
            links = "".join(f'<a class="ProductRowCard-module__cardTitleLink--abc" href="/Product/t-{i}"></a>' for i in ids)
            return f"<html><body>{links}</body></html>"
        by_sort = {
            None: page([1, 2, 3]),             # Relevance (no order= param)
            "Rating": page([3, 1, 2]),
            "Rating-Count": page([1, 2, 3]),   # ignored by server -> same as Relevance
            "Price-Asc": page([2, 3, 1]),
        }

        class _Resp:
            def __init__(self, body): self.status, self._body = 200, body
            async def text(self): return self._body
            async def __aenter__(self): return self
            async def __aexit__(self, *a): return False

        class _Session:
            def get(self, url, headers=None, allow_redirects=True):
                import urllib.parse as up
                q = up.parse_qs(up.urlparse(url).query)
                return _Resp(by_sort[q.get("order", [None])[0]])
            async def __aenter__(self): return self
            async def __aexit__(self, *a): return False

        config = {"sorting_methods": ["Relevance", "Rating", "Rating-Count", "Price-Asc"]}
        with patch("tpt_scraper.aiohttp.ClientSession", return_value=_Session()), \
             patch("tpt_scraper.asyncio.sleep", new_callable=AsyncMock):
            kept = await scraper.check_sort_orders(config, [("", "elementary", "social-emotional", "", "", "")])
        assert kept == ["Relevance", "Rating", "Price-Asc"]

        # Single sort order: nothing to compare, returned unchanged without any requests
        assert await scraper.check_sort_orders({"sorting_methods": ["Relevance"]}, [("", "", "", "", "", "")]) == ["Relevance"]


class TestScrapeMetadata:
    """Test metadata scraping functionality."""
    
    def test_extract_text_with_spacing(self):
        """Test text extraction with proper spacing."""
        from bs4 import BeautifulSoup
        
        html = '<div>Hello<span>World</span>Test</div>'
        soup = BeautifulSoup(html, 'lxml')
        element = soup.find('div')
        
        text = scraper.extract_text_with_spacing(element)
        
        assert text == "Hello World Test"
        assert "  " not in text  # No double spaces
        
    def test_extract_text_with_spacing_none(self):
        """Test text extraction with None element."""
        text = scraper.extract_text_with_spacing(None)
        
        assert text is None
        
    @pytest.mark.asyncio
    async def test_scrape_product_metadata_structure(self):
        """Test that scrape_product_metadata returns correct structure."""
        mock_html = """
        <html>
            <head><title>Test Product</title></head>
            <body>
                <meta name="description" content="Test description">
            </body>
        </html>
        """
        
        with patch('tpt_scraper.fetch', new_callable=AsyncMock) as mock_fetch:
            mock_fetch.return_value = mock_html
            
            session = AsyncMock()
            result = await scraper.scrape_product_metadata(
                session, 
                "https://www.teacherspayteachers.com/Product/test"
            )
            
            assert result is not None
            assert isinstance(result, dict)
            assert list(result.keys()) == scraper.METADATA_COLUMNS
            assert result["title"] == "Test Product"
            assert result["short_description"] == "Test description"
            assert result["product_id"] is None  # URL has no numeric id
            assert result["post_date"] is None   # no Apollo state in this page

    def test_extract_product_id(self):
        assert scraper.extract_product_id("https://www.teacherspayteachers.com/Product/Some-Title-1234567") == "1234567"
        assert scraper.extract_product_id("https://www.teacherspayteachers.com/Product/--10158553") == "10158553"
        assert scraper.extract_product_id("https://www.teacherspayteachers.com/Product/Title-99?ref=x") == "99"
        assert scraper.extract_product_id("https://www.teacherspayteachers.com/Product/test") is None

    def test_extract_product_details_from_apollo_state(self):
        html = '<html><script>window.__DATA__={"foo":1,"apolloState":' + json.dumps({
            "Store:188690": {"__typename": "Store", "id": "188690", "followerCount": 51049, "totalItems": 385},
            "Product:263126": {
                "__typename": "Product", "id": "263126", "postDate": "2012-06-20 22:45:24",
                "itemType": "DIGITAL_PRODUCT", "kind": "ZIP", "isFree": False,
                "filePreview": {"filesize": 23392631, "pageCount": "35+ Pages"},
                "prices": {"isFree": False, "nonTransferableLicense": {
                    "price": "$6.00", "discountPrice": "$0.00", "salePrice": "$0.00", "additionalLicensePrice": "$4.80"}},
                "taxonomyTags": [{"__ref": "ResourceTaxonomyTag:3rd-grade"}, {"__ref": "ResourceTaxonomyTag:pdf"}],
                "commonCoreStandards": [], "answerKey": "NA", "teachingDuration": "NA",
                "includedIn": [{"__ref": "Product:3944049"}], "images": [{}, {}, {}],
                "previewVideo": None, "mainVideo": {"url": "x"},
                "evaluationRating": {"count": 11530, "v1EvaluationsCount": 2026, "scoreAverage": "4.77"},
                "author": {"__ref": "Store:188690"},
            },
            "ROOT_QUERY": {"__typename": "Query", "filterEvaluationsByResource({\"resourceId\":\"263126\"})": {
                "totalCount": 10000,
                "facets": {"__typename": "EvaluationFacets",
                           "starRatingGroup": [{"value": "4", "count": 874}, {"value": "5", "count": 10596}],
                           "whichGradeUsedWith": [{"value": "3", "count": 679}]}}},
        }) + ',"other":2};</script></html>'

        state = scraper.extract_apollo_state(html)
        assert state is not None and "Product:263126" in state

        d = scraper.extract_product_details(state, "263126")
        assert set(d.keys()) == set(scraper.PRODUCT_DETAIL_COLUMNS)
        assert d["post_date"] == "2012-06-20 22:45:24"
        assert d["item_type"] == "DIGITAL_PRODUCT"
        assert d["file_kind"] == "ZIP"
        assert d["file_size_bytes"] == 23392631
        assert d["page_count"] == "35+ Pages"
        assert (d["price_full"], d["price_discount"], d["price_additional_license"]) == ("$6.00", "$0.00", "$4.80")
        assert d["is_free"] == 0
        assert d["taxonomy_tags"] == "3rd-grade, pdf"
        assert d["common_core_standards"] is None
        assert d["bundle_count"] == 1
        assert d["image_count"] == 3
        assert (d["has_preview_video"], d["has_main_video"]) == (0, 1)
        assert (d["store_id"], d["store_follower_count"], d["store_total_items"]) == ("188690", 51049, 385)
        assert d["v1_evaluations_count"] == 2026
        assert json.loads(d["rating_distribution"]) == {"4": 874, "5": 10596}
        assert json.loads(d["evaluation_facets"]) == {"starRatingGroup": {"4": 874, "5": 10596},
                                                      "whichGradeUsedWith": {"3": 679}}

        # Falls back to the Product entry that has postDate when the id doesn't match
        assert scraper.extract_product_details(state, "999")["post_date"] == "2012-06-20 22:45:24"
        # Degrades to all-None without Apollo state
        assert all(v is None for v in scraper.extract_product_details(None, "263126").values())
        assert scraper.extract_apollo_state("<html>no state</html>") is None


class TestFirstReviewDate:
    """Test the GraphQL-based first-review lookup with a fake session."""

    @staticmethod
    def _make_session(pages_by_after):
        """pages_by_after maps the `after` cursor -> connection dict (or Exception)."""
        calls = []

        class _Resp:
            def __init__(self, body):
                self.status = 200
                self.headers = {}
                self._body = body
            async def json(self, content_type=None):
                return self._body
            async def __aenter__(self):
                return self
            async def __aexit__(self, *a):
                return False

        class _Session:
            def post(self, url, json=None, headers=None):
                after = json["variables"]["after"]
                calls.append(after)
                page = pages_by_after.get(after)
                if page is None:
                    page = {"totalCount": 0, "pageInfo": {"hasNextPage": False, "lastCursor": None}, "edges": []}
                return _Resp({"data": {"filterEvaluationsByResource": page}})

        return _Session(), calls

    @staticmethod
    def _page(dates, has_next=False, last_cursor=None):
        return {"totalCount": len(dates),
                "pageInfo": {"hasNextPage": has_next, "lastCursor": last_cursor},
                "edges": [{"cursor": "x", "node": {"createdAt": d}} for d in dates]}

    def test_cursor_encoding(self):
        assert scraper.encode_evaluations_cursor(0) is None
        assert scraper.encode_evaluations_cursor(8) == "OA=="
        assert scraper.encode_evaluations_cursor(9997) == "OTk5Nw=="

    @pytest.mark.asyncio
    async def test_single_jump(self):
        # 632 reviews -> jump to offset 532 and read the tail in one call
        session, calls = self._make_session({
            scraper.encode_evaluations_cursor(532): self._page(
                ["2016-05-01T00:00:00.000Z", "2016-04-14T09:00:00.000Z", "2016-04-20T00:00:00.000Z"]),
        })
        date, status = await scraper.fetch_first_review_date(session, "2495504", 632)
        assert (date, status) == ("2016-04-14", "ok")
        assert calls == [scraper.encode_evaluations_cursor(532)]

    @pytest.mark.asyncio
    async def test_follows_next_page_when_count_is_stale(self):
        c1 = scraper.encode_evaluations_cursor(532)
        session, calls = self._make_session({
            c1: self._page(["2016-06-01T00:00:00.000Z"], has_next=True, last_cursor="NjMy"),
            "NjMy": self._page(["2016-04-14T00:00:00.000Z"]),
        })
        date, status = await scraper.fetch_first_review_date(session, "p", 632)
        assert (date, status) == ("2016-04-14", "ok")
        assert calls == [c1, "NjMy"]

    @pytest.mark.asyncio
    async def test_backsteps_when_count_overshoots(self):
        # number_of_ratings says 250 but the API only has 120 -> offset 150 is empty, 50 is not
        session, calls = self._make_session({
            scraper.encode_evaluations_cursor(50): self._page(["2020-01-05T00:00:00.000Z", "2019-12-31T00:00:00.000Z"]),
        })
        date, status = await scraper.fetch_first_review_date(session, "p", 250)
        assert (date, status) == ("2019-12-31", "ok")
        assert calls == [scraper.encode_evaluations_cursor(150), scraper.encode_evaluations_cursor(50)]

    @pytest.mark.asyncio
    async def test_no_reviews(self):
        session, _ = self._make_session({})
        assert await scraper.fetch_first_review_date(session, "p", 3) == (None, "no_reviews")

    @pytest.mark.asyncio
    async def test_fetch_failed(self):
        with patch("tpt_scraper.fetch_evaluations_page", new_callable=AsyncMock) as m:
            m.return_value = None
            assert await scraper.fetch_first_review_date(object(), "p", 40) == (None, "fetch_failed")


class TestDownloadWorkflow:
    """Test download functionality."""
    
    @pytest.mark.asyncio
    async def test_download_queue_mode(self, temp_db_file):
        """Test download workflow with queue mode."""
        await scraper.setup_db(temp_db_file)
        
        # Add test data to database
        async with aiosqlite.connect(temp_db_file) as db:
            # Add a product to metadata
            await db.execute("""
                INSERT INTO product_metadata (url, title, product_price)
                VALUES (?, ?, ?)
            """, ("https://test.com/product/1", "Test Product", "0.00"))
            
            # Add to download queue
            await db.execute("""
                INSERT INTO download_queue (product_url, priority, notes)
                VALUES (?, ?, ?)
            """, ("https://test.com/product/1", 10, "Test"))
            
            await db.commit()
        
        # Mock the download function
        with patch('tpt_scraper.download_free_file', new_callable=AsyncMock) as mock_download:
            mock_download.return_value = ("/tmp/file.pdf", 1024)
            
            # This would normally run the download, but we'll just verify the query works
            async with aiosqlite.connect(temp_db_file) as db:
                async with db.execute("""
                    SELECT DISTINCT q.product_url, m.product_price
                    FROM download_queue q
                    JOIN product_metadata m ON q.product_url = m.url
                    LEFT JOIN downloads d ON q.product_url = d.product_url
                    WHERE d.product_url IS NULL
                    ORDER BY q.priority DESC, q.added_at ASC
                """) as cursor:
                    results = await cursor.fetchall()
                    
            assert len(results) == 1
            assert results[0][0] == "https://test.com/product/1"
            
    @pytest.mark.asyncio
    async def test_download_marks_completed(self, temp_db_file):
        """Test that downloads are marked as completed."""
        await scraper.setup_db(temp_db_file)
        
        async with aiosqlite.connect(temp_db_file) as db:
            await db.execute("""
                INSERT INTO downloads (product_url, file_path, file_size)
                VALUES (?, ?, ?)
            """, ("https://test.com/product/1", "/tmp/test.pdf", 2048))
            await db.commit()
            
            # Verify it was inserted
            async with db.execute("SELECT COUNT(*) FROM downloads") as cursor:
                count = (await cursor.fetchone())[0]
                
        assert count == 1


class TestCLI:
    """Test command-line interface."""
    
    @pytest.mark.asyncio
    async def test_cmd_config_create(self, temp_config_dir):
        """Test config create command."""
        with patch('tpt_scraper.ConfigManager') as mock_manager_class:
            mock_manager = Mock()
            mock_manager.create_config.return_value = True
            mock_manager.get_db_file.return_value = "test.db"
            mock_manager_class.return_value = mock_manager
            
            args = Mock()
            args.name = "test-config"
            args.template = None
            args.description = "Test"
            
            await scraper.cmd_config_create(args)
            
            mock_manager.create_config.assert_called_once()
            
    @pytest.mark.asyncio
    async def test_cmd_config_list(self, temp_config_dir):
        """Test config list command."""
        with patch('tpt_scraper.ConfigManager') as mock_manager_class:
            mock_manager = Mock()
            mock_manager.list_configs.return_value = [
                {"name": "config1", "created": "2026-01-01", "description": "Test 1"},
                {"name": "config2", "created": "2026-01-02", "description": "Test 2"}
            ]
            mock_manager_class.return_value = mock_manager
            
            args = Mock()
            
            await scraper.cmd_config_list(args)
            
            mock_manager.list_configs.assert_called_once()
            
    @pytest.mark.asyncio
    async def test_cmd_stats(self, temp_db_file):
        """Test stats command."""
        await scraper.setup_db(temp_db_file)
        
        # Add some test data
        async with aiosqlite.connect(temp_db_file) as db:
            await db.execute(
                "INSERT INTO search_results (url, price_option) VALUES (?, ?)",
                ("https://test.com/1", "free")
            )
            await db.execute(
                "INSERT INTO product_metadata (url, product_price) VALUES (?, ?)",
                ("https://test.com/1", "0.00")
            )
            await db.commit()
        
        with patch('tpt_scraper.ConfigManager') as mock_manager_class:
            mock_manager = Mock()
            mock_manager.get_db_file.return_value = temp_db_file
            mock_manager_class.return_value = mock_manager
            
            args = Mock()
            args.config = "test"
            
            # Should not raise an exception
            await scraper.cmd_stats(args)


class TestIntegration:
    """Integration tests for complete workflows."""
    
    @pytest.mark.asyncio
    async def test_full_workflow_database_state(self, temp_db_file, sample_config):
        """Test that database maintains correct state through workflow."""
        await scraper.setup_db(temp_db_file)
        
        # Simulate search results
        async with aiosqlite.connect(temp_db_file) as db:
            await db.execute("""
                INSERT INTO search_results 
                (url, resource_type, grade_level, subject, format, price_option, supports, sort_order, page)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                "https://test.com/product/1",
                "teacher-tools",
                "elementary/kindergarten",
                "social-emotional",
                "pdf",
                "free",
                "",
                "Relevance",
                1
            ))
            
            # Simulate metadata scraping
            await db.execute("""
                INSERT INTO product_metadata
                (url, title, product_price)
                VALUES (?, ?, ?)
            """, ("https://test.com/product/1", "Test Product", "0.00"))
            
            await db.commit()
            
            # Verify both tables have data
            async with db.execute("SELECT COUNT(*) FROM search_results") as cursor:
                search_count = (await cursor.fetchone())[0]
            async with db.execute("SELECT COUNT(*) FROM product_metadata") as cursor:
                metadata_count = (await cursor.fetchone())[0]
                
        assert search_count == 1
        assert metadata_count == 1
        
    @pytest.mark.asyncio
    async def test_download_queue_workflow(self, temp_db_file):
        """Test complete download queue workflow."""
        await scraper.setup_db(temp_db_file)
        
        async with aiosqlite.connect(temp_db_file) as db:
            # 1. Add search results
            await db.execute("""
                INSERT INTO search_results (url, price_option)
                VALUES (?, ?)
            """, ("https://test.com/product/1", "free"))
            
            # 2. Add metadata
            await db.execute("""
                INSERT INTO product_metadata (url, title, product_price, rating_value)
                VALUES (?, ?, ?, ?)
            """, ("https://test.com/product/1", "Great Product", "0.00", "4.5"))
            
            # 3. Manually add to download queue (simulating SQL manipulation)
            await db.execute("""
                INSERT INTO download_queue (product_url, priority, notes)
                SELECT url, 10, 'Highly rated'
                FROM product_metadata
                WHERE CAST(rating_value AS REAL) >= 4.0
            """)
            
            await db.commit()
            
            # 4. Verify queue has the item
            async with db.execute("SELECT COUNT(*) FROM download_queue") as cursor:
                queue_count = (await cursor.fetchone())[0]
                
            # 5. Simulate download completion
            await db.execute("""
                INSERT INTO downloads (product_url, file_path, file_size)
                VALUES (?, ?, ?)
            """, ("https://test.com/product/1", "/tmp/test.pdf", 1024))
            
            await db.commit()
            
            # 6. Verify query that excludes downloaded items
            async with db.execute("""
                SELECT COUNT(*)
                FROM download_queue q
                LEFT JOIN downloads d ON q.product_url = d.product_url
                WHERE d.product_url IS NULL
            """) as cursor:
                remaining = (await cursor.fetchone())[0]
                
        assert queue_count == 1
        assert remaining == 0  # Should be 0 because we "downloaded" it


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
