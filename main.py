from file_indexer import FileIndexer
from llm_orchestrator import LLMOrchestrator
import argparse
import os
import sys
from pathlib import Path
from typing import Set, List
from tqdm import tqdm
import time

class SystemIndexer:
    """Handles system-wide file indexing with configurable skip patterns."""

    # Move skip patterns to class constants with clear names
    SYSTEM_SKIP_DIRS = {
        '/proc', '/sys', '/run', '/dev', '/tmp',
        '/var/tmp', '/var/cache', '/var/run', '/var/lock',
        '/lost+found', '/.snapshots'
    }

    DEV_SKIP_PATTERNS = {
        '.git', '__pycache__', 'node_modules',
        '.venv', 'venv', '.env', '.idea', '.vscode'
    }

    def __init__(self, skip_hidden: bool = False):
        """Initialize indexer with configurable hidden file handling.
        
        Args:
            skip_hidden: Whether to skip hidden files/directories
        """
        self.indexed_count = 0
        self.skipped_count = 0
        self.error_count = 0
        self.total_bytes_processed = 0
        self.start_time = None
        self.skip_hidden = skip_hidden

    def should_skip_dir(self, dir_path: str) -> bool:
        """Determine if directory should be skipped during indexing.
        
        Args:
            dir_path: Full path to directory
            
        Returns:
            bool: True if directory should be skipped
        """
        dir_name = os.path.basename(dir_path)
        
        if self.skip_hidden and dir_name.startswith('.'):
            return True
            
        if any(pattern in dir_path for pattern in self.DEV_SKIP_PATTERNS):
            return True
        
        return any(dir_path.startswith(skip_dir) for skip_dir in self.SYSTEM_SKIP_DIRS)

    def should_skip_file(self, file_name: str) -> bool:
        """Check if file should be skipped"""
        return self.skip_hidden and file_name.startswith('.')

    def get_user_home(self) -> str:
        """Get user's home directory"""
        return str(Path.home())

    def format_size(self, size_bytes):
        """Convert bytes to human readable format"""
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.2f} {unit}"
            size_bytes /= 1024.0

    def get_dir_stats(self, path: str) -> tuple[int, int]:
        """Get total number of files and total size in directory"""
        total_files = 0
        total_size = 0
        for root, dirs, files in os.walk(path, topdown=True):
            dirs[:] = [d for d in dirs if not self.should_skip_dir(os.path.join(root, d))]
            total_files += len(files)
            for file in files:
                try:
                    file_path = os.path.join(root, file)
                    if os.access(file_path, os.R_OK):
                        total_size += os.path.getsize(file_path)
                except Exception:
                    continue
        return total_files, total_size

    def index_system(self, indexer: FileIndexer, paths: List[str], show_progress: bool = True):
        """Index files in specified paths with progress tracking.
        
        Args:
            indexer: FileIndexer instance to use
            paths: List of paths to index
            show_progress: Whether to show progress bar
        """
        self.start_time = time.time()

        if show_progress:
            total_files, total_size = self._get_total_stats(paths)
            print(f"Found {total_files:,} files (Total size: {self.format_size(total_size)})")

        with tqdm(total=total_files, disable=not show_progress,
                 desc="Indexing files", unit="file") as progress_bar:
            self._process_paths(paths, indexer, progress_bar, show_progress)

        if show_progress:
            self._print_summary()

    def _get_total_stats(self, paths: List[str]) -> tuple[int, int]:
        """Get total file count and size for all paths.
        
        Args:
            paths: List of paths to analyze
            
        Returns:
            tuple: (total_files, total_size)
        """
        print("Scanning directories...")
        total_files = total_size = 0
        for path in paths:
            files, size = self.get_dir_stats(path)
            total_files += files
            total_size += size
        return total_files, total_size

    def _process_paths(self, paths: List[str], indexer: FileIndexer, 
                      progress_bar: tqdm, show_progress: bool):
        """Process all paths for indexing.
        
        Args:
            paths: List of paths to process
            indexer: FileIndexer instance
            progress_bar: tqdm progress bar
            show_progress: Whether to show progress
        """
        for base_path in paths:
            try:
                self._walk_path(base_path, indexer, progress_bar, show_progress)
            except Exception as err:
                if show_progress:
                    tqdm.write(f"Error accessing {base_path}: {err}")

    def _walk_path(self, base_path: str, indexer: FileIndexer, 
                   progress_bar: tqdm, show_progress: bool):
        """Walk through the directory structure and index files.
        
        Args:
            base_path: Base path to start walking from
            indexer: FileIndexer instance
            progress_bar: tqdm progress bar
            show_progress: Whether to show progress
        """
        for root, dirs, files in os.walk(base_path, topdown=True):
            # Modify dirs in-place to skip unwanted directories
            dirs[:] = [d for d in dirs if not self.should_skip_dir(os.path.join(root, d))]
            
            # Filter files based on hidden status
            files = [f for f in files if not self.should_skip_file(f)]

            for file in files:
                file_path = os.path.join(root, file)
                try:
                    if os.access(file_path, os.R_OK):
                        file_size = os.path.getsize(file_path)
                        
                        # Update progress bar description with current file
                        progress_bar.set_postfix_str(f"Processing: {os.path.basename(file_path)}")
                        
                        # Index the file
                        if indexer.index_file(file_path):
                            self.indexed_count += 1
                            self.total_bytes_processed += file_size
                        else:
                            self.skipped_count += 1
                    else:
                        self.skipped_count += 1
                except Exception as e:
                    self.error_count += 1
                    if show_progress:
                        tqdm.write(f"Error processing {file_path}: {str(e)}")
                finally:
                    progress_bar.update(1)

    def _print_summary(self):
        """Print summary of indexing process."""
        elapsed_time = time.time() - self.start_time
        hours = int(elapsed_time // 3600)
        minutes = int((elapsed_time % 3600) // 60)
        seconds = int(elapsed_time % 60)
        
        print("\nIndexing Summary:")
        print(f"├─ Time elapsed: {hours:02d}:{minutes:02d}:{seconds:02d}")
        print(f"├─ Files processed: {self.indexed_count:,}")
        print(f"├─ Total data processed: {self.format_size(self.total_bytes_processed)}")
        print(f"├─ Average speed: {self.format_size(self.total_bytes_processed/elapsed_time)}/s")
        print(f"├─ Files skipped: {self.skipped_count:,}")
        print(f"└─ Errors encountered: {self.error_count:,}")

def main():
    """Main entry point for the file indexing and search application."""
    args = _parse_arguments()
    
    indexer = FileIndexer()
    llm = LLMOrchestrator()
    system_indexer = SystemIndexer(skip_hidden=args.skip_hidden)
    
    if args.index:
        _handle_indexing(args, indexer, system_indexer)
    elif args.query:
        _handle_search(args, indexer, llm)
    else:
        _print_usage()

def _parse_arguments() -> argparse.Namespace:
    """Parse and return command line arguments."""
    parser = argparse.ArgumentParser(description='File Search and LLM Assistant')
    parser.add_argument('--index', action='store_true', help='Index files in the directory')
    parser.add_argument('--system', action='store_true', help='Index the whole system (requires root)')
    parser.add_argument('--home', action='store_true', help='Index user home directory')
    parser.add_argument('--directory', type=str, help='Directory to index')
    parser.add_argument('--query', type=str, help='Search query')
    parser.add_argument('--quiet', action='store_true', help='Hide progress output')
    parser.add_argument('--skip-hidden', action='store_true', help='Skip hidden files and directories (starting with .)')
    
    return parser.parse_args()

def _handle_indexing(args: argparse.Namespace, indexer: FileIndexer, system_indexer: SystemIndexer):
    """Handle indexing of files."""
    paths_to_index = []
    
    if args.system:
        if os.geteuid() != 0:
            print("Error: Root privileges required for system indexing")
            print("Please run with sudo")
            sys.exit(1)
        paths_to_index.append('/')
        
    elif args.home:
        paths_to_index.append(system_indexer.get_user_home())
        
    elif args.directory:
        if not os.path.exists(args.directory):
            print(f"Error: Directory not found: {args.directory}")
            sys.exit(1)
        paths_to_index.append(args.directory)
    else:
        print("Error: Please specify what to index using --system, --home, or --directory")
        sys.exit(1)
    
    if paths_to_index:
        print(f"Starting indexing of: {', '.join(paths_to_index)}")
        system_indexer.index_system(indexer, paths_to_index, not args.quiet)
        print("Saving index...")
        if indexer.save_index():
            print("Indexing complete! You can now search using --query")
        else:
            print("Error saving index!")

def _handle_search(args: argparse.Namespace, indexer: FileIndexer, llm: LLMOrchestrator):
    """Handle searching for files."""
    if not indexer.load_index():
        print("\nNo index found. Please index some files first using one of these commands:")
        print("  Index home directory:   python main.py --index --home")
        print("  Index specific folder:  python main.py --index --directory /path/to/folder")
        print("  Index whole system:     sudo python main.py --index --system")
        return
            
    print("Searching for files...")
    results = indexer.search(args.query)
    
    if results:
        response = llm.generate_response(args.query, results)
        print("\nAI Assistant Response:")
        print(response)
    else:
        print("No matching files found")

def _print_usage():
    """Print usage information."""
    print("Please specify either --index to index files or --query to search")
    print("For help, use: python main.py --help")

if __name__ == "__main__":
    main() 