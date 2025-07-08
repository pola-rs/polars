#!/usr/bin/env python3
"""Test script to reproduce and validate the fix for window_size=0 issue."""

import polars as pl

def test_rolling_sum_window_zero():
    """Test rolling_sum with window_size=0 to ensure it raises proper error."""
    print("Testing rolling_sum with window_size=0...")
    
    # Create a test series
    s = pl.Series([1, 2, 3, 4, 5])
    print(f"Original series: {s}")
    
    try:
        # This should raise an error instead of panicking
        result = s.rolling_sum(window_size=0)
        print(f"ERROR: rolling_sum with window_size=0 should have failed but returned: {result}")
        return False
    except Exception as e:
        print(f"✓ Correctly caught error: {e}")
        return True

def test_rolling_sum_normal():
    """Test rolling_sum with normal window_size to ensure it still works."""
    print("\nTesting rolling_sum with normal window_size...")
    
    # Create a test series
    s = pl.Series([1, 2, 3, 4, 5])
    print(f"Original series: {s}")
    
    try:
        # This should work normally
        result = s.rolling_sum(window_size=2)
        print(f"✓ rolling_sum with window_size=2 worked: {result}")
        return True
    except Exception as e:
        print(f"ERROR: rolling_sum with window_size=2 failed: {e}")
        return False

def test_other_rolling_functions():
    """Test other rolling functions with window_size=0."""
    print("\nTesting other rolling functions with window_size=0...")
    
    s = pl.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    
    rolling_functions = ['rolling_mean', 'rolling_std', 'rolling_min', 'rolling_max']
    
    all_passed = True
    for func_name in rolling_functions:
        try:
            func = getattr(s, func_name)
            result = func(window_size=0)
            print(f"ERROR: {func_name} with window_size=0 should have failed but returned: {result}")
            all_passed = False
        except Exception as e:
            print(f"✓ {func_name} correctly caught error: {e}")
    
    return all_passed

def test_expr_rolling_sum():
    """Test expression rolling_sum with window_size=0."""
    print("\nTesting expression rolling_sum with window_size=0...")
    
    df = pl.DataFrame({"a": [1, 2, 3, 4, 5]})
    
    try:
        result = df.select(pl.col("a").rolling_sum(window_size=0))
        print(f"ERROR: expr rolling_sum with window_size=0 should have failed but returned: {result}")
        return False
    except Exception as e:
        print(f"✓ expr rolling_sum correctly caught error: {e}")
        return True

def main():
    """Run all tests."""
    print("=" * 60)
    print("Testing rolling window functions with window_size=0")
    print("=" * 60)
    
    tests = [
        test_rolling_sum_window_zero,
        test_rolling_sum_normal,
        test_other_rolling_functions,
        test_expr_rolling_sum
    ]
    
    all_passed = True
    for test in tests:
        passed = test()
        all_passed = all_passed and passed
    
    print("\n" + "=" * 60)
    if all_passed:
        print("✓ All tests passed!")
    else:
        print("✗ Some tests failed!")
    print("=" * 60)
    
    return all_passed

if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)
