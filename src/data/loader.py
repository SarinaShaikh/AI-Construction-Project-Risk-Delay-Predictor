"""
src/data/loader.py

Data loading module for SCOPE v0.2 construction project dataset.
Loads, validates, and provides access to all 25 CSV files.

Usage:
    from src.data.loader import load_raw_dataset, validate_schema, get_projects
    
    data = load_raw_dataset()
    valid, errors = validate_schema(data)
    projects = get_projects(data)
"""

from pathlib import Path

import pandas as pd

# Expected files in data/raw/
EXPECTED_FILES = [
    'activities.csv',
    'activity_states.csv',
    'construction_memory.csv',
    'counterfactuals.csv',
    'DATASET_CATALOG.csv',
    'DATA_DICTIONARY.csv',
    'DATA_QUALITY_REPORT.csv',
    'decisions.csv',
    'dependencies.csv',
    'environment.csv',
    'events.csv',
    'friction_canonical.csv',
    'friction_summary_canonical.csv',
    'intervention_options.csv',
    'master_projects_canonical.csv',
    'outcomes.csv',
    'procurement.csv',
    'projects.csv',
    'resources.csv',
    'resource_allocation.csv',
    'rework.csv',
]

# Expected row counts (approximate for validation)
EXPECTED_ROW_COUNTS = {
    'projects': (80, 120),  # ~100 projects
    'activities': (9000, 10000),  # ~9,279 activities
    'dependencies': (17000, 19000),  # ~18,176 dependencies
    'activity_states': (3000000, 5000000),  # ~4M+ records
    'construction_memory': (1000, 100000),  # Varies
}

# Required columns for each table
REQUIRED_COLUMNS = {
    'projects': ['project_id', 'project_type', 'planned_duration_days', 'planned_cost'],
    'activities': ['project_id', 'activity_id', 'activity_name', 'planned_duration_days'],
    'dependencies': ['project_id', 'predecessor_id', 'successor_id', 'lag_days'],
    'activity_states': ['project_id', 'activity_id', 'date'],
    'construction_memory': ['project_id', 'activity_id', 'observed_delay_days'],
}


def load_raw_dataset(data_dir: str = 'data/raw') -> dict[str, pd.DataFrame]:
    """
    Load all CSV files from SCOPE v0.2 dataset.
    
    Args:
        data_dir: Path to data/raw folder (default: 'data/raw')
    
    Returns:
        Dictionary mapping filename (without .csv) to DataFrame
        Example: {'projects': DataFrame, 'activities': DataFrame, ...}
    
    Raises:
        FileNotFoundError: If data_dir doesn't exist
        Exception: If critical files are missing
    
    Example:
        >>> data = load_raw_dataset()
        >>> print(data['projects'].shape)
        (100, 12)
    """
    
    # Convert to Path object for easier handling
    data_path = Path(data_dir)
    
    # Validate directory exists
    if not data_path.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    
    if not data_path.is_dir():
        raise NotADirectoryError(f"Path is not a directory: {data_dir}")
    
    # Find all CSV files in directory
    csv_files = sorted(data_path.glob('*.csv'))
    
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")
    
    print(f"Found {len(csv_files)} CSV files in {data_dir}")
    
    # Load each CSV into a dictionary
    data = {}
    errors = []
    
    for csv_file in csv_files:
        filename = csv_file.stem  # Remove .csv extension
        
        try:
            print(f"  Loading {csv_file.name}...", end=" ")
            df = pd.read_csv(csv_file)
            data[filename] = df
            print(f"✓ ({len(df):,} rows, {len(df.columns)} columns)")
        except (pd.errors.EmptyDataError, pd.errors.ParserError, OSError) as e:
            error_msg = f"Failed to load {csv_file.name}: {e!s}"
            errors.append(error_msg)
            print(f"✗ Error: {e!s}")
    
    if errors:
        print(f"\n⚠️  {len(errors)} file(s) failed to load:")
        for error in errors:
            print(f"  - {error}")
    
    print(f"\n✅ Successfully loaded {len(data)} files")
    return data


def validate_schema(data: dict[str, pd.DataFrame]) -> tuple[bool, list[str]]:
    """
    Validate dataset structure, row counts, columns, data types, and integrity.
    
    Args:
        data: Dictionary from load_raw_dataset()
    
    Returns:
        Tuple of (is_valid: bool, errors: list of error messages)
        If is_valid is False, errors contains descriptions of problems found
    
    Example:
        >>> data = load_raw_dataset()
        >>> valid, errors = validate_schema(data)
        >>> if not valid:
        ...     for err in errors:
        ...         print(err)
    """
    
    errors = []
    
    print("\n" + "="*60)
    print("VALIDATING DATASET SCHEMA")
    print("="*60)
    
    # 1. Check for critical files
    print("\n1. Checking for critical files...")
    critical_files = ['projects', 'activities', 'dependencies', 'activity_states', 'construction_memory']
    for filename in critical_files:
        if filename not in data:
            errors.append(f"Missing critical file: {filename}.csv")
            print(f"  ✗ {filename}.csv missing")
        else:
            print(f"  ✓ {filename}.csv loaded")
    
    # 2. Check row counts for key tables
    print("\n2. Checking row counts...")
    for table_name, (min_rows, max_rows) in EXPECTED_ROW_COUNTS.items():
        if table_name in data:
            row_count = len(data[table_name])
            if row_count < min_rows or row_count > max_rows:
                errors.append(
                    f"{table_name}.csv has {row_count:,} rows, "
                    f"expected {min_rows:,}-{max_rows:,}"
                )
                print(f"  ⚠️  {table_name}: {row_count:,} rows (expected {min_rows:,}-{max_rows:,})")
            else:
                print(f"  ✓ {table_name}: {row_count:,} rows")
    
    # 3. Check required columns
    print("\n3. Checking required columns...")
    for table_name, required_cols in REQUIRED_COLUMNS.items():
        if table_name in data:
            df = data[table_name]
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                errors.append(
                    f"{table_name}.csv missing columns: {', '.join(missing_cols)}"
                )
                print(f"  ✗ {table_name}: missing {missing_cols}")
            else:
                print(f"  ✓ {table_name}: all required columns present")
    
    # 4. Check for nulls in critical columns
    print("\n4. Checking for nulls in critical columns...")
    critical_nulls = {
        'projects': ['project_id'],
        'activities': ['project_id', 'activity_id'],
        'dependencies': ['project_id', 'predecessor_id', 'successor_id'],
    }
    
    for table_name, columns in critical_nulls.items():
        if table_name in data:
            df = data[table_name]
            for col in columns:
                if col in df.columns:
                    null_count = df[col].isna().sum()
                    if null_count > 0:
                        errors.append(
                            f"{table_name}.csv has {null_count:,} nulls in {col}"
                        )
                        print(f"  ✗ {table_name}.{col}: {null_count:,} nulls")
                    else:
                        print(f"  ✓ {table_name}.{col}: no nulls")
    
    # 5. Check for duplicates
    print("\n5. Checking for duplicate records...")
    for table_name in ['projects', 'activities', 'dependencies']:
        if table_name in data:
            df = data[table_name]
            if table_name == 'projects':
                dup_count = df.duplicated(subset=['project_id']).sum()
            elif table_name == 'activities':
                dup_count = df.duplicated(subset=['project_id', 'activity_id']).sum()
            elif table_name == 'dependencies':
                dup_count = df.duplicated(subset=['project_id', 'predecessor_id', 'successor_id']).sum()
            
            if dup_count > 0:
                errors.append(f"{table_name}.csv has {dup_count:,} duplicate records")
                print(f"  ✗ {table_name}: {dup_count:,} duplicates")
            else:
                print(f"  ✓ {table_name}: no duplicates")
    
    # 6. Check referential integrity
    print("\n6. Checking referential integrity...")
    if 'projects' in data and 'activities' in data:
        valid_projects = set(data['projects']['project_id'].unique())
        activity_projects = set(data['activities']['project_id'].unique())
        invalid = activity_projects - valid_projects
        if invalid:
            errors.append(
                f"activities.csv references {len(invalid)} invalid project_ids"
            )
            print(f"  ✗ activities: {len(invalid)} invalid project references")
        else:
            print("  ✓ activities: all project_ids are valid")
    
    if 'activities' in data and 'dependencies' in data:
        valid_activities = set(data['activities'][['project_id', 'activity_id']]
                              .apply(lambda row: f"{row['project_id']}_{row['activity_id']}", axis=1))
        dep_df = data['dependencies'].copy()
        dep_df['activity_key'] = dep_df['project_id'].astype(str) + '_' + dep_df['predecessor_id'].astype(str)
        invalid_pred = dep_df[~dep_df['activity_key'].isin(valid_activities)]
        
        if len(invalid_pred) > 0:
            errors.append(
                f"dependencies.csv has {len(invalid_pred):,} invalid predecessor_ids"
            )
            print(f"  ✗ dependencies: {len(invalid_pred):,} invalid predecessor references")
        else:
            print("  ✓ dependencies: all predecessor_ids are valid")
    
    # Summary
    print("\n" + "="*60)
    if errors:
        print(f"❌ VALIDATION FAILED ({len(errors)} errors)")
        print("="*60)
        return False, errors
    else:
        print("✅ VALIDATION PASSED")
        print("="*60)
        return True, []


# ============================================================================
# CONVENIENCE GETTER FUNCTIONS
# ============================================================================

def get_projects(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Return projects table."""
    return data.get('projects', pd.DataFrame())


def get_activities(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Return activities table."""
    return data.get('activities', pd.DataFrame())


def get_dependencies(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Return dependencies table."""
    return data.get('dependencies', pd.DataFrame())


def get_activity_states(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Return activity_states table (4M+ records)."""
    return data.get('activity_states', pd.DataFrame())


def get_construction_memory(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Return construction_memory table (training labels for ML)."""
    return data.get('construction_memory', pd.DataFrame())


def get_events(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Return events table."""
    return data.get('events', pd.DataFrame())


def get_environment(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Return environment table (weather, external factors)."""
    return data.get('environment', pd.DataFrame())


def get_decisions(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Return decisions table."""
    return data.get('decisions', pd.DataFrame())


def get_resources(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Return resources table."""
    return data.get('resources', pd.DataFrame())


def get_resource_allocation(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Return resource_allocation table."""
    return data.get('resource_allocation', pd.DataFrame())


def get_procurement(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Return procurement table."""
    return data.get('procurement', pd.DataFrame())


def get_rework(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Return rework table."""
    return data.get('rework', pd.DataFrame())


def get_counterfactuals(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Return counterfactuals table (what-if scenarios)."""
    return data.get('counterfactuals', pd.DataFrame())


def get_intervention_options(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Return intervention_options table (mitigation strategies)."""
    return data.get('intervention_options', pd.DataFrame())


def get_friction_canonical(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Return friction_canonical table (friction/resistance factors)."""
    return data.get('friction_canonical', pd.DataFrame())


def get_friction_summary_canonical(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Return friction_summary_canonical table."""
    return data.get('friction_summary_canonical', pd.DataFrame())


def get_master_projects_canonical(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Return master_projects_canonical table."""
    return data.get('master_projects_canonical', pd.DataFrame())


def get_outcomes(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Return outcomes table."""
    return data.get('outcomes', pd.DataFrame())


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

if __name__ == '__main__':
    """
    Test the loader when run directly.
    Usage: uv run python src/data/loader.py
    """
    print("\n" + "="*60)
    print("SCOPE v0.2 DATASET LOADER TEST")
    print("="*60 + "\n")
    
    try:
        # Load dataset
        print("Loading dataset...")
        data = load_raw_dataset('data/raw')
        
        # Validate schema
        print("\nValidating schema...")
        valid, errors = validate_schema(data)
        
        if valid:
            print("\n✅ All checks passed! Dataset is ready.")
            print("\nDataset Summary:")
            print(f"  Projects: {len(get_projects(data)):,}")
            print(f"  Activities: {len(get_activities(data)):,}")
            print(f"  Dependencies: {len(get_dependencies(data)):,}")
            print(f"  Activity States: {len(get_activity_states(data)):,}")
            print(f"  Construction Memory: {len(get_construction_memory(data)):,}")
        else:
            print(f"\n❌ Validation failed with {len(errors)} errors")
            for error in errors:
                print(f"  - {error}")
    
    except (pd.errors.EmptyDataError, pd.errors.ParserError, OSError) as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()