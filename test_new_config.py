import asyncio
import sys
import os

# Add the directory containing tptscrape.py to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from tptscrape import (
    build_page_url, 
    setup_db, 
    extraction_test,
    RESOURCE_TYPES,
    GRADE_LEVELS, 
    SUBJECTS,
    FORMATS,
    PRICE_OPTIONS,
    SUPPORTS,
    SORTING_METHODS
)

async def test_configuration():
    """Test the new configuration and URL building."""
    print("Testing new configuration structure...")
    
    # Test configuration loading
    print(f"Resource Types: {len(RESOURCE_TYPES)} items")
    print(f"Grade Levels: {len(GRADE_LEVELS)} items") 
    print(f"Subjects: {len(SUBJECTS)} items")
    print(f"Formats: {len(FORMATS)} items")
    print(f"Price Options: {len(PRICE_OPTIONS)} items")
    print(f"Supports: {len(SUPPORTS)} items")
    print(f"Sorting Methods: {len(SORTING_METHODS)} items")
    
    # Calculate total combinations
    total_combinations = (
        len(RESOURCE_TYPES) * 
        len(GRADE_LEVELS) * 
        len(SUBJECTS) * 
        len(FORMATS) * 
        len(PRICE_OPTIONS) * 
        len(SUPPORTS) * 
        len(SORTING_METHODS)
    )
    print(f"\nTotal parameter combinations: {total_combinations:,}")
    
    # Test URL building with different combinations
    print("\nTesting URL building:")
    
    test_cases = [
        # Test case 1: All empty parameters
        ("", "", "social-emotional", "", "", "", "Relevance", 1),
        
        # Test case 2: Full parameters
        ("teacher-tools", "elementary", "social-emotional", "pdf", "free", "", "Rating", 1),
        
        # Test case 3: With supports
        ("hands-on-activities", "middle-school", "social-emotional/character-education", "digital", "under-5", "special-education", "Price-Asc", 2),
        
        # Test case 4: High school with specific grade
        ("instruction", "high-school/12th-grade", "social-emotional/social-emotional-learning", "video", "above-10", "", "Most-Recent", 3),
    ]
    
    for i, (resource_type, grade_level, subject, format_type, price_option, supports, sort_order, page) in enumerate(test_cases, 1):
        url = build_page_url(resource_type, grade_level, subject, format_type, price_option, supports, sort_order, page)
        print(f"Test {i}: {url}")
    
    print("\nTesting database setup...")
    await setup_db()
    print("Database setup completed successfully!")
    
    print("\nTesting extraction with new parameters...")
    await extraction_test(test_limit=3)
    print("Extraction test completed!")

if __name__ == "__main__":
    asyncio.run(test_configuration())
