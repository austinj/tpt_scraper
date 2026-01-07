# Testing Guide for TPT Scraper

This guide explains how to run the test suite for the TPT scraper.

## Setup

Install test dependencies:

```bash
pip install -r test_requirements.txt
```

## Running Tests

### Run all tests
```bash
pytest
```

### Run with verbose output
```bash
pytest -v
```

### Run specific test file
```bash
pytest test_tpt_scraper.py
```

### Run specific test class
```bash
pytest test_tpt_scraper.py::TestConfigManager
```

### Run specific test
```bash
pytest test_tpt_scraper.py::TestConfigManager::test_create_config
```

### Run with coverage
```bash
pytest --cov=tpt_scraper --cov=config_manager --cov-report=html
```

## Test Structure

### Test Classes

- **TestConfigManager**: Tests for configuration management
  - Creating, updating, deleting configurations
  - Listing and retrieving configs
  - Database file path management

- **TestDatabaseSetup**: Tests for database schema
  - Table creation
  - Index creation
  - Schema validation

- **TestAdaptiveRateLimiter**: Tests for rate limiting
  - Response recording
  - Delay adjustment
  - Error handling

- **TestURLBuilding**: Tests for URL construction
  - Basic URLs
  - URLs with parameters
  - Query string handling

- **TestSearchWorkflow**: Tests for search functionality
  - Combination generation
  - URL extraction
  - Database storage

- **TestScrapeMetadata**: Tests for metadata scraping
  - HTML parsing
  - Text extraction with spacing
  - Data structure validation

- **TestDownloadWorkflow**: Tests for download functionality
  - Queue mode
  - Download completion tracking
  - File storage

- **TestCLI**: Tests for command-line interface
  - Config commands
  - Stats command
  - Argument parsing

- **TestIntegration**: Integration tests
  - Complete workflow tests
  - Database state validation
  - Multi-step operations

## Test Coverage

The test suite covers:

✅ Configuration management (create, read, update, delete)  
✅ Database schema and migrations  
✅ Rate limiting and adaptive behavior  
✅ URL building with various parameters  
✅ HTML parsing and text extraction  
✅ Download queue workflow  
✅ Command-line interface  
✅ Integration scenarios  

## Mocking

Tests use mocking to avoid:
- Actual HTTP requests to TPT
- Creating real files
- Browser automation (Playwright)

This makes tests:
- Fast (no network delays)
- Reliable (no external dependencies)
- Safe (no actual web scraping during tests)

## Writing New Tests

When adding new functionality:

1. Add test class or method to `test_tpt_scraper.py`
2. Use fixtures for common setup (temp directories, databases)
3. Mock external dependencies (HTTP, file system, browser)
4. Test both success and failure cases
5. Run tests before committing: `pytest`

### Example Test

```python
@pytest.mark.asyncio
async def test_my_feature(temp_db_file):
    """Test my new feature."""
    await scraper.setup_db(temp_db_file)
    
    # Setup test data
    async with aiosqlite.connect(temp_db_file) as db:
        await db.execute("INSERT INTO test_table VALUES (?)", ("test",))
        await db.commit()
    
    # Test the feature
    result = await scraper.my_feature(temp_db_file)
    
    # Assert expectations
    assert result is not None
    assert result == expected_value
```

## Continuous Integration

Tests can be run in CI/CD pipelines:

```yaml
# Example GitHub Actions workflow
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
        with:
          python-version: '3.9'
      - run: pip install -r requirements.txt
      - run: pip install -r test_requirements.txt
      - run: pytest --cov
```

## Troubleshooting

### Tests fail with "ModuleNotFoundError"
```bash
# Make sure you're in the venv
.\venv\Scripts\Activate.ps1

# Install test dependencies
pip install -r test_requirements.txt
```

### Async tests fail
```bash
# Make sure pytest-asyncio is installed
pip install pytest-asyncio

# Check pytest.ini has asyncio_mode = auto
```

### Database locked errors
```bash
# Close any SQLite browser connections
# Tests use temporary databases that should auto-clean
```

## Quick Test Checklist

Before committing changes:

- [ ] All existing tests pass: `pytest`
- [ ] New features have tests
- [ ] Tests use mocking appropriately
- [ ] No actual network requests in tests
- [ ] Database tests use temp files
- [ ] Tests are documented with docstrings
