import os
import glob
import random
from pathlib import Path

# Get the absolute path to the SynthHuman directory
SYNTH_HUMAN_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "SynthHuman")

def get_all_ids():
    """Get all unique IDs from the files in the SynthHuman directory."""
    # Get all files matching the patterns for both png and exr
    pattern_png = os.path.join(SYNTH_HUMAN_DIR, "*_*.png")
    pattern_exr = os.path.join(SYNTH_HUMAN_DIR, "*_*.exr")
    files = glob.glob(pattern_png) + glob.glob(pattern_exr)
    
    if not files:
        print(f"No files found matching patterns:")
        print(f"  PNG pattern: {pattern_png}")
        print(f"  EXR pattern: {pattern_exr}")
        print(f"Checking if directory exists: {SYNTH_HUMAN_DIR}")
        print(f"Directory exists: {os.path.exists(SYNTH_HUMAN_DIR)}")
        return []
    
    # Extract unique IDs from filenames
    ids = set()
    for file in files:
        # Extract the numeric ID from the filename
        # Example: from "SynthHuman/depth_0259999.exr" get "0259999"
        filename = os.path.basename(file)
        id_part = filename.split('_')[1].split('.')[0]
        ids.add(id_part)
    
    return sorted(list(ids))

def get_files_for_id(id_num):
    """Get all files associated with a specific ID."""
    patterns = [
        f"depth_{id_num}.exr",
        f"alpha_{id_num}.png",
        f"normal_{id_num}.exr",
        f"rgb_{id_num}.png"
    ]
    return [os.path.join(SYNTH_HUMAN_DIR, pattern) for pattern in patterns]

def remove_random_data(percentage=80, dry_run=True):
    """
    Remove a random percentage of the data.
    
    Args:
        percentage (int): Percentage of data to remove (0-100)
        dry_run (bool): If True, only show what would be deleted without actually deleting
    """
    # Get all unique IDs
    all_ids = get_all_ids()
    
    if not all_ids:
        print("No files found to process!")
        return
        
    total_ids = len(all_ids)
    
    # Calculate how many IDs to remove
    num_to_remove = int(total_ids * (percentage / 100))
    
    # Randomly select IDs to remove
    ids_to_remove = random.sample(all_ids, num_to_remove)
    
    # Get all files to remove
    files_to_remove = []
    for id_num in ids_to_remove:
        files_to_remove.extend(get_files_for_id(id_num))
    
    # Print summary
    print(f"\nTotal unique IDs found: {total_ids}")
    print(f"Number of IDs to remove: {num_to_remove}")
    print(f"Total files to remove: {len(files_to_remove)}")
    
    # Show first few examples
    print("\nExample files that will be removed (showing first 2 IDs):")
    for id_num in ids_to_remove[:2]:
        print(f"\nFiles for ID {id_num}:")
        for file in get_files_for_id(id_num):
            print(f"  - {file}")
    
    if dry_run:
        print("\nThis was a dry run. No files were actually deleted.")
        print("To actually delete the files, run with dry_run=False")
        return
    
    # If not a dry run, actually delete the files
    print("\nDeleting files...")
    for file in files_to_remove:
        try:
            os.remove(file)
            print(f"Deleted: {file}")
        except Exception as e:
            print(f"Error deleting {file}: {e}")
    
    print("\nDeletion complete!")

if __name__ == "__main__":
    # First do a dry run to show what will be deleted
    print("Performing dry run...")
    remove_random_data(percentage=80, dry_run=True)
    
    # Ask for confirmation before actual deletion
    response = input("\nWould you like to proceed with actual deletion? (yes/no): ")
    if response.lower() == 'yes':
        print("\nProceeding with actual deletion...")
        remove_random_data(percentage=80, dry_run=False)
    else:
        print("\nAborted. No files were deleted.")
