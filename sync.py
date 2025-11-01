#!/usr/bin/env python3
"""
SSH Data Sync Tool for LLM Distillery

Syncs datasets and reports between local development machine and remote server
without bloating git repository. Uses SCP (SSH copy) for reliable cross-platform
file transfer.

Setup:
    1. Copy sync_config.example.json to sync_config.json
    2. Edit sync_config.json with your server details
    3. Ensure SSH access to server is configured

Usage:
    python sync.py pull           # Pull data FROM server TO local
    python sync.py push           # Push data FROM local TO server
    python sync.py status         # Show what would be synced
    python sync.py pull-code      # Pull code changes from git
    python sync.py push-code      # Push code changes to git
    python sync.py full-sync      # Pull code, push code, pull data

Common Workflows:
    # Start work session: get latest code and data
    python sync.py pull-code && python sync.py pull

    # End work session: push code changes
    python sync.py push-code

    # Deploy to server: push code, then run on server
    python sync.py push-code
    # (then SSH to server and run batch_labeler)

    # Retrieve results: pull data from server
    python sync.py pull
"""

import json
import subprocess
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Optional
import platform


class SyncTool:
    """SSH-based data sync for LLM distillery."""

    def __init__(self, config_path: str = "sync_config.json"):
        """Load configuration from JSON file."""
        config_file = Path(config_path)

        if not config_file.exists():
            print(f"ERROR: Config file not found: {config_path}")
            print(f"\nPlease create {config_path} from sync_config.example.json")
            print("\nExample config:")
            print("  cp sync_config.example.json sync_config.json")
            print("  # Edit sync_config.json with your server details")
            sys.exit(1)

        with open(config_file, 'r') as f:
            self.config = json.load(f)

        self.remote = self.config['remote']
        self.sync_config = self.config['sync']
        self.ssh_config = self.config.get('ssh', {})

        # Detect if we're on Windows
        self.is_windows = platform.system() == 'Windows'

    def _build_scp_cmd(
        self,
        source: str,
        dest: str
    ) -> List[str]:
        """
        Build scp command with proper options.

        Args:
            source: Source path (local or remote, e.g., "user@host:/path" or "/local/path")
            dest: Destination path (local or remote)

        Returns:
            Command as list of strings
        """
        cmd = ['scp', '-r', '-p']  # Recursive, preserve timestamps

        # Add port if not default
        if self.remote['port'] != 22:
            cmd.extend(['-P', str(self.remote['port'])])

        # Add SSH key if specified
        if self.ssh_config.get('key_path'):
            cmd.extend(['-i', self.ssh_config['key_path']])

        # Add compression
        cmd.append('-C')

        # Source and destination
        cmd.extend([source, dest])

        return cmd

    def _get_remote_path(self, subdir: str = "") -> str:
        """Get full remote path (user@host:/path/subdir)."""
        user = self.remote['user']
        host = self.remote['host']
        path = self.remote['remote_path']

        if subdir:
            path = f"{path.rstrip('/')}/{subdir.lstrip('/')}"

        return f"{user}@{host}:{path}"

    def pull_data(self, dry_run: bool = False) -> bool:
        """
        Pull data FROM server TO local using SCP.

        Args:
            dry_run: If True, only show what would be synced (doesn't transfer)

        Returns:
            True if successful
        """
        print("="*70)
        print("PULL DATA FROM SERVER")
        print("="*70)

        if dry_run:
            print("DRY RUN MODE - no files will be transferred\n")

        success = True

        for data_dir in self.sync_config['data_dirs']:
            # Remote source (add trailing /* to copy contents)
            remote_source = self._get_remote_path(data_dir.rstrip('/') + '/*')

            # Local destination
            local_dest = Path.cwd() / data_dir

            # Ensure local dir exists
            local_dest.mkdir(parents=True, exist_ok=True)

            print(f"\nSyncing: {data_dir}")
            print(f"   From: {self._get_remote_path(data_dir)}")
            print(f"   To:   {local_dest}")

            if dry_run:
                print(f"   Would sync {data_dir}")
                continue

            # Build and run scp command
            cmd = self._build_scp_cmd(
                source=remote_source,
                dest=str(local_dest)
            )

            try:
                print(f"   Running: {' '.join(cmd[:4])} ...")
                print(f"   (This may take a while for large datasets...)")
                result = subprocess.run(
                    cmd,
                    check=True,
                    stderr=subprocess.PIPE,  # Capture only errors, let stdout show progress
                    text=True
                )
                print(f"   Successfully synced {data_dir}")

            except subprocess.CalledProcessError as e:
                print(f"   ERROR: Failed to sync {data_dir}")
                if e.stderr:
                    print(f"   {e.stderr.strip()}")
                success = False
            except FileNotFoundError:
                print(f"   ERROR: scp not found. Please ensure SSH is installed.")
                return False

        print("\n" + "="*70)
        if dry_run:
            print("DRY RUN COMPLETE - no files were transferred")
        elif success:
            print("DATA PULL COMPLETE")
        else:
            print("WARNING: DATA PULL COMPLETED WITH ERRORS")
        print("="*70 + "\n")

        return success

    def push_data(self, dry_run: bool = False) -> bool:
        """
        Push data FROM local TO server using SCP.

        Args:
            dry_run: If True, only show what would be synced (doesn't transfer)

        Returns:
            True if successful
        """
        print("="*70)
        print("PUSH DATA TO SERVER")
        print("="*70)

        if dry_run:
            print("DRY RUN MODE - no files will be transferred\n")

        # Check if local data exists
        has_data = False
        for data_dir in self.sync_config['data_dirs']:
            local_path = Path.cwd() / data_dir
            if local_path.exists():
                has_data = True
                break

        if not has_data:
            print("WARNING: No local data directories found. Nothing to push.")
            return True

        success = True

        for data_dir in self.sync_config['data_dirs']:
            local_source = Path.cwd() / data_dir

            # Check if local dir exists
            if not local_source.exists():
                print(f"\nSkipping {data_dir} (not found locally)")
                continue

            # Remote destination
            remote_dest = self._get_remote_path(data_dir.rstrip('/'))

            print(f"\nSyncing: {data_dir}")
            print(f"   From: {local_source}")
            print(f"   To:   {remote_dest}")

            if dry_run:
                print(f"   Would sync {data_dir}")
                continue

            # Build and run scp command (copy contents with /*)
            local_source_contents = str(local_source / '*')
            cmd = self._build_scp_cmd(
                source=local_source_contents,
                dest=remote_dest
            )

            try:
                print(f"   Running: {' '.join(cmd[:4])} ...")
                print(f"   (This may take a while for large datasets...)")
                result = subprocess.run(
                    cmd,
                    check=True,
                    stderr=subprocess.PIPE,  # Capture only errors, let stdout show progress
                    text=True,
                    shell=self.is_windows  # Need shell on Windows for glob expansion
                )
                print(f"   Successfully synced {data_dir}")

            except subprocess.CalledProcessError as e:
                print(f"   ERROR: Failed to sync {data_dir}")
                if e.stderr:
                    print(f"   {e.stderr.strip()}")
                success = False
            except FileNotFoundError:
                print(f"   ERROR: scp not found. Please ensure SSH is installed.")
                return False

        print("\n" + "="*70)
        if dry_run:
            print("DRY RUN COMPLETE - no files were transferred")
        elif success:
            print("DATA PUSH COMPLETE")
        else:
            print("WARNING: DATA PUSH COMPLETED WITH ERRORS")
        print("="*70 + "\n")

        return success

    def status(self) -> bool:
        """Show what would be synced (dry run for both directions)."""
        print("="*70)
        print("SYNC STATUS")
        print("="*70)
        print()

        print("Checking what would be PULLED from server...\n")
        self.pull_data(dry_run=True)

        print("\n" + "-"*70 + "\n")

        print("Checking what would be PUSHED to server...\n")
        self.push_data(dry_run=True)

        return True

    def pull_code(self) -> bool:
        """Pull code changes from git."""
        print("="*70)
        print("📥 PULL CODE FROM GIT")
        print("="*70)
        print()

        try:
            # Check if there are uncommitted changes
            result = subprocess.run(
                ['git', 'status', '--porcelain'],
                check=True,
                capture_output=True,
                text=True
            )

            if result.stdout.strip():
                print("⚠️  Warning: You have uncommitted changes:")
                subprocess.run(['git', 'status', '--short'], check=True)
                print()
                response = input("Continue with git pull? [y/N]: ")
                if response.lower() != 'y':
                    print("❌ Aborted")
                    return False

            # Pull from git
            subprocess.run(['git', 'pull'], check=True)

            print("\n✅ CODE PULL COMPLETE\n")
            return True

        except subprocess.CalledProcessError as e:
            print(f"❌ Git pull failed: {e}")
            return False

    def push_code(self) -> bool:
        """Push code changes to git."""
        print("="*70)
        print("📤 PUSH CODE TO GIT")
        print("="*70)
        print()

        try:
            # Check if there are changes to commit
            result = subprocess.run(
                ['git', 'status', '--porcelain'],
                check=True,
                capture_output=True,
                text=True
            )

            if not result.stdout.strip():
                print("✓ No changes to commit")
            else:
                print("📝 Current changes:")
                subprocess.run(['git', 'status', '--short'], check=True)
                print()

            # Push to git
            subprocess.run(['git', 'push'], check=True)

            print("\n✅ CODE PUSH COMPLETE\n")
            return True

        except subprocess.CalledProcessError as e:
            print(f"❌ Git push failed: {e}")
            return False

    def full_sync(self) -> bool:
        """Full sync: pull code, push code, pull data."""
        print("="*70)
        print("🔄 FULL SYNC")
        print("="*70)
        print()

        steps = [
            ("Pull code from git", self.pull_code),
            ("Push code to git", self.push_code),
            ("Pull data from server", self.pull_data),
        ]

        for step_name, step_func in steps:
            print(f"\n▶️  {step_name}...")
            if not step_func():
                print(f"\n❌ Full sync failed at: {step_name}")
                return False

        print("\n" + "="*70)
        print("✅ FULL SYNC COMPLETE")
        print("="*70 + "\n")
        return True


def main():
    parser = argparse.ArgumentParser(
        description='SSH Data Sync Tool for LLM Distillery',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    # Pull data
    pull_parser = subparsers.add_parser('pull', help='Pull data from server to local')
    pull_parser.add_argument('--dry-run', action='store_true', help='Show what would be transferred without actually transferring')

    # Push data
    push_parser = subparsers.add_parser('push', help='Push data from local to server')
    push_parser.add_argument('--dry-run', action='store_true', help='Show what would be transferred without actually transferring')

    # Status
    subparsers.add_parser('status', help='Show sync status (dry run for both directions)')

    # Pull code
    subparsers.add_parser('pull-code', help='Pull code changes from git')

    # Push code
    subparsers.add_parser('push-code', help='Push code changes to git')

    # Full sync
    subparsers.add_parser('full-sync', help='Full sync: pull code, push code, pull data')

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Create sync tool
    try:
        sync = SyncTool()
    except Exception as e:
        print(f"❌ Failed to initialize sync tool: {e}")
        sys.exit(1)

    # Execute command
    commands = {
        'pull': lambda: sync.pull_data(dry_run=args.dry_run),
        'push': lambda: sync.push_data(dry_run=args.dry_run),
        'status': sync.status,
        'pull-code': sync.pull_code,
        'push-code': sync.push_code,
        'full-sync': sync.full_sync,
    }

    success = commands[args.command]()
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
