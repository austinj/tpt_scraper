# TPT Scraper - Advanced Web Scraping Tool

A highly optimized, scalable web scraper for Teachers Pay Teachers (TPT) that efficiently extracts product URLs and detailed product information. Features advanced optimization capabilities for handling large-scale scraping operations with millions of filter combinations.

## 🚀 Key Features

### Core Functionality
- **URL Extraction**: Efficiently discovers product URLs across all TPT categories and filters
- **Product Scraping**: Extracts comprehensive product data including titles, descriptions, ratings, and prices
- **Robust Database Storage**: SQLite database with optimized schema for fast queries and data integrity

### Advanced Optimizations
- **Empty Combination Detection**: Automatically identifies and skips filter combinations that yield no results
- **Pre-filtering Support**: Generates and uses pre-filtered lists of valid combinations to maximize efficiency
- **Adaptive Performance**: Dynamic rate limiting and batch size adjustment based on real-time performance
- **Statistical Sampling**: Quick estimation of empty combination percentages for large config spaces
- **Comprehensive Monitoring**: Performance tracking, efficiency analysis, and optimization recommendations

### Scalability Features
- **Asynchronous Processing**: High-performance async/await pattern with configurable concurrency
- **Smart Caching**: HTTP response caching to minimize redundant requests
- **Database Optimizations**: Efficient indexing, batch operations, and connection pooling
- **Memory Management**: Optimized memory usage for processing millions of combinations

## 📦 Installation

```bash
# Clone the repository
git clone <repository-url>
cd tpt_scraper

# Install dependencies
pip install -r requirements.txt
```

## 🛠️ Configuration

Edit `config.json` to customize scraping parameters:

```json
{
    "resource_types": ["", "activities", "assessments"],
    "grade_levels": ["", "kindergarten", "1st", "2nd"],
    "subjects": ["math", "science", "social-studies"],
    "formats": ["", "pdf", "powerpoint"],
    "price_options": ["", "free", "paid"],
    "supports": ["", "easel-activity"],
    "sorting_methods": ["Relevance", "Price: Low to High"],
    "total_pages": 42,
    "concurrent_requests": 25,
    "sleep_between_batches": [0.5, 2.0]
}
```

## 🔧 Usage

### Basic Usage

```bash
# Run extraction stage (discover URLs)
python tptscrape.py extract

# Run scraping stage (get product details)
python tptscrape.py scrape

# Run both stages
python tptscrape.py extract-and-scrape

# Test with limited results
python tptscrape.py extract-test
python tptscrape.py scrape-test
```

### Optimization Workflow

For large-scale operations with many filter combinations:

```bash
# 1. Analyze your configuration
python optimize.py analyze

# 2. Run statistical sampling to estimate efficiency gains
python optimize.py sample --samples 5000

# 3. Generate pre-filtered combinations (for very large configs)
python optimize.py prefilter

# 4. Run complete optimization workflow
python optimize.py all
```

### Performance Monitoring

```bash
# Generate performance report
python monitor.py report

# Create performance trend plots
python monitor.py plot

# Real-time monitoring
python monitor.py watch
```

## 📊 Optimization Tools

### Statistical Sampling (`sample_combinations.py`)
Quickly estimates the percentage of empty filter combinations without full processing:
- Tests a random sample of combinations
- Provides statistical estimates with confidence intervals
- Helps decide if pre-filtering is worthwhile

### Pre-filtering (`filter_combinations.py`)
Generates a comprehensive list of valid (non-empty) combinations:
- Tests all possible filter combinations
- Creates `valid_combinations.json` for the main scraper
- Generates `config_filtered.json` with only valid options

### Performance Monitor (`monitor.py`)
Tracks scraper performance and provides optimization insights:
- Real-time performance metrics
- Empty combination efficiency analysis
- Trend visualization and recommendations

### CLI Optimizer (`optimize.py`)
Unified interface for all optimization tools:
- Configuration analysis
- Automated optimization workflows
- Status checking and recommendations

## 🏗️ Database Schema

```sql
-- URL extraction tracking
CREATE TABLE extracted_pages (...)
CREATE TABLE extracted_urls (...)

-- Product data storage  
CREATE TABLE scraped_data (...)

-- Optimization tables
CREATE TABLE empty_combinations (...)
CREATE TABLE performance_stats (...)
```

## ⚡ Performance Features

### Adaptive Rate Limiting
- Automatically adjusts request delays based on success rates
- Prevents overwhelming the target server
- Maintains optimal throughput

### Smart Batching
- Dynamic batch size adjustment based on performance
- Optimizes memory usage and processing speed
- Adapts to network conditions

### Empty Combination Skipping
- Tracks combinations that yield no results
- Skips known empty combinations in future runs
- Can save 20-80% of requests for large configurations

## 📈 Monitoring and Analytics

The scraper provides comprehensive monitoring:

- **Success/Error Rates**: Track request reliability
- **Processing Speed**: Monitor items processed per second
- **Empty Combination Stats**: Efficiency metrics and savings
- **Database Growth**: Track data collection progress
- **Performance Trends**: Historical analysis and visualization

## 🔍 Example Workflows

### Small to Medium Scale (< 10K combinations)
```bash
python tptscrape.py extract-and-scrape
python monitor.py report
```

### Large Scale (10K-100K combinations)
```bash
python optimize.py analyze
python optimize.py sample
python tptscrape.py extract-and-scrape
python monitor.py report
```

### Very Large Scale (100K+ combinations)
```bash
python optimize.py analyze
python optimize.py sample --samples 10000
python optimize.py prefilter  # May take hours
python tptscrape.py extract-and-scrape
python monitor.py plot
```

## 🛡️ Error Handling

- **Exponential Backoff**: Intelligent retry strategies
- **Network Resilience**: Handles timeouts and connection errors
- **Data Integrity**: Database transactions and conflict resolution
- **Graceful Degradation**: Continues operation despite individual failures

## 📝 Output Data

Extracted data includes:
- Product URLs and metadata
- Titles and descriptions
- Ratings and review counts
- Pricing information
- Category and filter classifications
- Performance and efficiency metrics

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

[Add your license information here]

## 🆘 Troubleshooting

### Common Issues

**High error rates**: Reduce `concurrent_requests` in config.json
**Slow performance**: Increase batch size or check network connectivity
**Memory issues**: Process in smaller batches or increase system memory
**Empty results**: Verify filter combinations are valid for TPT

### Getting Help

- Check the performance monitor for optimization suggestions
- Review logs for detailed error information
- Use test modes to verify functionality
- Monitor database growth and performance metrics

---

Built with ❤️ for efficient web scraping and data collection.