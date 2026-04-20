import asyncio
import io
import os
import pathlib
import tempfile
import zipfile

from sky.data import storage_utils
from sky.server import server
from sky.skylet import constants


def test_zip_files_and_folders(skyignore_dir):
    log_file = io.StringIO()
    with tempfile.NamedTemporaryFile('wb+', suffix='.zip') as f:
        storage_utils.zip_files_and_folders([skyignore_dir], f, log_file)
        # Print out all files in the zip
        f.seek(0)
        with zipfile.ZipFile(f, 'r') as zipf:
            actual_zipped_files = zipf.namelist()

        expected_zipped_files = [
            '', 'ln-keep.py', 'ln-dir-keep.py', 'dir/subdir/ln-keep.py',
            constants.SKY_IGNORE_FILE, 'dir/subdir/remove.py', 'keep.py',
            'dir/keep.txt', 'dir/keep.a', 'dir/subdir/keep.b', 'ln-folder',
            'empty-folder/', 'dir/', 'dir/subdir/', 'dir/subdir/remove_dir/'
        ]

        expected_zipped_file_paths = []
        for filename in expected_zipped_files:
            file_path = os.path.join(skyignore_dir, filename)
            if 'ln' not in filename:
                file_path = file_path.lstrip('/')
            expected_zipped_file_paths.append(file_path)

        for file in actual_zipped_files:
            assert file in expected_zipped_file_paths, (
                file, expected_zipped_file_paths)
        assert len(actual_zipped_files) == len(expected_zipped_file_paths)
        # Check the log file correctly logs the zipped files
        log_file.seek(0)
        log_file_content = log_file.read()
        assert f'Zipped {skyignore_dir}' in log_file_content


def test_unzip_file(skyignore_dir, tmp_path):
    """Test server.unzip_file function."""
    # Create a temporary zip file
    zip_path = tmp_path / 'test.zip'
    # Zip the test directory
    storage_utils.zip_files_and_folders([skyignore_dir], zip_path,
                                        io.StringIO())

    excluded_files = storage_utils.get_excluded_files(skyignore_dir)

    # Create a temporary directory to unzip into
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir_path = pathlib.Path(temp_dir)

        # Call server.unzip_file
        asyncio.run(server.unzip_file(zip_path, temp_dir_path))

        # Verify the zip file was deleted
        assert not zip_path.exists()

        # Get list of files in original directory
        original_files = []
        for root, dirs, files in os.walk(skyignore_dir):
            rel_root = os.path.relpath(root, skyignore_dir)
            if rel_root == '.':
                rel_root = ''

            # Add directories
            for d in dirs:
                path = os.path.join(rel_root, d).rstrip('/')
                if path and path not in excluded_files:
                    original_files.append(path)

            # Add files
            for f in files:
                path = os.path.join(rel_root, f)
                if path not in excluded_files:
                    original_files.append(path)

        # Get list of files in unzipped directory
        unzipped_files = []
        unzipped_dir = os.path.join(str(temp_dir_path),
                                    str(skyignore_dir).lstrip('/'))
        unzipped_dir = pathlib.Path(unzipped_dir)
        print('unzipped_dir', unzipped_dir)
        for root, dirs, files in os.walk(unzipped_dir):
            rel_root = os.path.relpath(root, unzipped_dir)
            if rel_root == '.':
                rel_root = ''
            # Add directories
            for d in dirs:
                path = os.path.join(rel_root, d).rstrip('/')
                if path:
                    unzipped_files.append(path)

            # Add files
            for f in files:
                path = os.path.join(rel_root, f)
                unzipped_files.append(path)

        # Verify files match
        assert sorted(original_files) == sorted(unzipped_files)

        # Verify symlinks are preserved
        assert (unzipped_dir / 'ln-keep.py').is_symlink()
        assert (unzipped_dir / 'ln-dir-keep.py').is_symlink()
        assert (unzipped_dir / 'dir/subdir/ln-keep.py').is_symlink()
        assert (unzipped_dir / 'ln-folder').is_symlink()

        # Verify empty folders are preserved
        assert (unzipped_dir / 'empty-folder').is_dir()
        assert not any((unzipped_dir / 'empty-folder').iterdir())


def _make_zip_with_one_file(zip_path: pathlib.Path, member_rel_path: str,
                            body: bytes) -> None:
    """Write a tiny zip containing a single file member with the given body."""
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_STORED) as zf:
        zf.writestr(member_rel_path, body)


def test_unzip_file_leaves_no_tmp_files(tmp_path):
    """The write-then-rename path must clean up its .tmp.* scratch files."""
    member = 'data/payload.bin'
    body = b'0123456789' * 2048  # 20 KB
    zip_path = tmp_path / 'src.zip'
    _make_zip_with_one_file(zip_path, member, body)

    staging = tmp_path / 'staging'
    staging.mkdir()

    asyncio.run(server.unzip_file(zip_path, staging))

    # Final file is there with the correct bytes
    final = staging / member
    assert final.read_bytes() == body
    # Zip was removed by the unzip_file finally block
    assert not zip_path.exists()
    # No .tmp.* debris left behind next to the final file
    leftover = list((staging / 'data').glob('payload.bin.tmp.*'))
    assert leftover == [], f'tmp files leaked: {leftover}'


def test_unzip_file_reader_with_open_fd_sees_old_bytes(tmp_path):
    """Atomic replace: a reader holding an fd on the old file sees the OLD
    bytes in full even after unzip_file rewrites the path with new content.

    This is the invariant that protects the concurrent aws-cli S3 sync from
    IncompleteBody when a second submission overwrites the same staging file.
    """
    member = 'data/payload.bin'
    old_body = b'A' * 4096
    new_body = b'B' * 4096

    staging = tmp_path / 'staging'
    (staging / 'data').mkdir(parents=True)

    # Plant the "old" file at the final path and open a read fd on it,
    # simulating aws-cli mid-sync reading the staging file.
    final = staging / member
    final.write_bytes(old_body)
    with open(final, 'rb') as reader_fd:
        # Now run unzip_file with a zip containing DIFFERENT bytes for the
        # same member — this must not affect the open reader_fd.
        zip_path = tmp_path / 'src.zip'
        _make_zip_with_one_file(zip_path, member, new_body)
        asyncio.run(server.unzip_file(zip_path, staging))

        # The reader_fd, which was opened BEFORE the unzip, must still see the
        # complete old bytes — POSIX inode semantics via os.replace().
        assert reader_fd.read() == old_body

    # A fresh read after the rename sees the new bytes.
    assert final.read_bytes() == new_body
