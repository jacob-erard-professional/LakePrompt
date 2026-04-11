from __future__ import annotations

import hashlib
import json
import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import Request, urlopen


USER_AGENT = "LakePrompt/1.0"


@dataclass
class _PreparedLake:
    """
    Description of a prepared local CSV lake derived from a remote source.

    Attributes:
        source_url: Original remote URL requested by the user.
        prepared_dir: Local directory containing normalized CSV files.
        source_type: Detected source kind such as `csv`, `zip`, or `github`.
    """
    source_url: str
    prepared_dir: Path
    source_type: str


class _DataLakePreparer:
    """
    Download and normalize supported remote sources into a local CSV lake.

    This class is needed because many useful datasets are published as
    archives or repositories instead of as a clean local CSV directory.

    Supported inputs:
    - direct CSV links
    - ZIP archives containing CSV files
    - GitHub repository URLs
    """

    def __init__(self, cache_root: str | Path | None = None):
        """
        Initialize the preparation cache directory.

        Args:
            cache_root: Optional directory used to store downloaded and
                normalized source material.

        Returns:
            None.
        """
        self.cache_root = Path(cache_root or ".lakeprompt_cache").expanduser()
        self.cache_root.mkdir(parents=True, exist_ok=True)

    def prepare(self, source_url: str) -> _PreparedLake:
        """
        Prepare a remote source as a local CSV lake.

        Args:
            source_url: Remote HTTP(S) URL pointing to a CSV, ZIP archive,
                or GitHub repository.

        Returns:
            A `_PreparedLake` describing the prepared local dataset.
        """
        source_type = self._detect_source_type(source_url)
        lake_id = hashlib.sha256(source_url.encode("utf-8")).hexdigest()[:16]
        target_dir = self.cache_root / lake_id / "lake"

        if target_dir.exists() and any(target_dir.glob("*.csv")):
            return _PreparedLake(
                source_url=source_url,
                prepared_dir=target_dir,
                source_type=source_type,
            )

        if target_dir.exists():
            shutil.rmtree(target_dir)
        target_dir.mkdir(parents=True, exist_ok=True)

        if source_type == "csv":
            self._prepare_csv(source_url, target_dir)
        elif source_type == "zip":
            self._prepare_zip(source_url, target_dir)
        elif source_type == "github":
            self._prepare_github_repo(source_url, target_dir)
        else:
            raise ValueError(f"Unsupported data lake source: {source_url}")

        csv_files = list(target_dir.glob("*.csv"))
        if not csv_files:
            raise ValueError(f"No CSV files were prepared from source: {source_url}")

        return _PreparedLake(
            source_url=source_url,
            prepared_dir=target_dir,
            source_type=source_type,
        )

    def _detect_source_type(self, source_url: str) -> str:
        """
        Classify a remote URL into one of the supported source types.

        Args:
            source_url: Candidate remote source URL.

        Returns:
            One of `csv`, `zip`, or `github`.
        """
        parsed = urlparse(source_url)
        lower_path = parsed.path.lower()

        if parsed.scheme not in {"http", "https"}:
            raise ValueError("Only http and https URLs are supported.")
        if parsed.netloc == "github.com":
            return "github"
        if lower_path.endswith(".csv"):
            return "csv"
        if lower_path.endswith(".zip"):
            return "zip"
        raise ValueError(
            "Unsupported URL. Expected a GitHub repo URL, direct .csv link, or direct .zip link."
        )

    def _prepare_csv(self, source_url: str, target_dir: Path) -> None:
        """
        Download a single CSV into the target lake directory.

        Args:
            source_url: Direct URL to a CSV file.
            target_dir: Destination directory for prepared lake files.

        Returns:
            None.
        """
        destination = target_dir / self._csv_name_from_url(source_url)
        self._download_to_path(source_url, destination)

    def _prepare_zip(self, source_url: str, target_dir: Path) -> None:
        """
        Download a ZIP archive and extract its CSV files.

        Args:
            source_url: Direct URL to a ZIP archive.
            target_dir: Destination directory for prepared CSV files.

        Returns:
            None.
        """
        archive_path = target_dir.parent / "source.zip"
        extract_dir = target_dir.parent / "extract"
        self._download_to_path(source_url, archive_path)
        self._extract_csvs_from_zip(archive_path, extract_dir, target_dir)

    def _prepare_github_repo(self, source_url: str, target_dir: Path) -> None:
        """
        Download a GitHub repository archive and extract its CSV files.

        Args:
            source_url: GitHub repository URL, optionally including a ref.
            target_dir: Destination directory for prepared CSV files.

        Returns:
            None.
        """
        owner, repo, ref = self._parse_github_repo_url(source_url)
        archive_url = self._resolve_github_archive_url(owner, repo, ref)
        archive_path = target_dir.parent / "repo.zip"
        extract_dir = target_dir.parent / "extract"
        self._download_to_path(archive_url, archive_path)
        self._extract_csvs_from_zip(archive_path, extract_dir, target_dir)

    def _extract_csvs_from_zip(
        self,
        archive_path: Path,
        extract_dir: Path,
        target_dir: Path,
    ) -> None:
        """
        Extract all CSV files from a downloaded ZIP archive.

        Args:
            archive_path: Downloaded archive file.
            extract_dir: Temporary extraction directory.
            target_dir: Final directory that should receive normalized CSVs.

        Returns:
            None.
        """
        if extract_dir.exists():
            shutil.rmtree(extract_dir)
        extract_dir.mkdir(parents=True, exist_ok=True)

        with zipfile.ZipFile(archive_path) as handle:
            handle.extractall(extract_dir)

        copied = 0
        used_names: set[str] = set()
        for csv_path in sorted(extract_dir.rglob("*.csv")):
            name = self._unique_csv_name(csv_path, used_names)
            shutil.copy2(csv_path, target_dir / name)
            copied += 1

        if copied == 0:
            raise ValueError(f"No CSV files found in archive: {archive_path}")

    def _unique_csv_name(self, csv_path: Path, used_names: set[str]) -> str:
        """
        Build a collision-resistant normalized CSV filename.

        Args:
            csv_path: Original extracted CSV path.
            used_names: Filenames already allocated in this preparation run.

        Returns:
            A unique filename suitable for the prepared lake directory.
        """
        parts = list(csv_path.parts[-3:])
        stemmed = "__".join(Path(part).stem for part in parts[:-1] + [csv_path.name])
        candidate = f"{stemmed}.csv"
        counter = 2
        while candidate in used_names:
            candidate = f"{stemmed}_{counter}.csv"
            counter += 1
        used_names.add(candidate)
        return candidate

    def _download_to_path(self, source_url: str, destination: Path) -> None:
        """
        Download a remote resource to a local path.

        Args:
            source_url: Remote URL to fetch.
            destination: Local output path.

        Returns:
            None.
        """
        destination.parent.mkdir(parents=True, exist_ok=True)
        request = Request(source_url, headers={"User-Agent": USER_AGENT})
        with urlopen(request) as response, destination.open("wb") as handle:
            shutil.copyfileobj(response, handle)

    def _resolve_github_archive_url(self, owner: str, repo: str, ref: str | None) -> str:
        """
        Resolve the ZIP archive URL for a GitHub repository reference.

        Args:
            owner: GitHub repository owner.
            repo: GitHub repository name.
            ref: Optional branch or ref name.

        Returns:
            A codeload ZIP URL for the selected branch.
        """
        branch = ref or self._fetch_github_default_branch(owner, repo)
        return f"https://codeload.github.com/{owner}/{repo}/zip/refs/heads/{branch}"

    def _fetch_github_default_branch(self, owner: str, repo: str) -> str:
        """
        Look up the default branch for a GitHub repository.

        Args:
            owner: GitHub repository owner.
            repo: GitHub repository name.

        Returns:
            The repository's default branch name.
        """
        api_url = f"https://api.github.com/repos/{owner}/{repo}"
        request = Request(
            api_url,
            headers={
                "Accept": "application/vnd.github+json",
                "User-Agent": USER_AGENT,
            },
        )
        with urlopen(request) as response:
            payload = json.loads(response.read().decode("utf-8"))
        branch = payload.get("default_branch")
        if not isinstance(branch, str) or not branch.strip():
            raise ValueError(f"Could not determine default branch for GitHub repo: {owner}/{repo}")
        return branch

    @staticmethod
    def _parse_github_repo_url(source_url: str) -> tuple[str, str, str | None]:
        """
        Parse a GitHub repository URL into owner, repo, and optional ref.

        Args:
            source_url: GitHub URL supplied by the user.

        Returns:
            A tuple of `(owner, repo, ref)`.
        """
        parsed = urlparse(source_url)
        parts = [part for part in parsed.path.strip("/").split("/") if part]
        if len(parts) < 2:
            raise ValueError(f"Invalid GitHub repository URL: {source_url}")

        owner, repo = parts[0], parts[1]
        ref = None
        if len(parts) >= 4 and parts[2] == "tree":
            ref = parts[3]

        return owner, repo, ref

    @staticmethod
    def _csv_name_from_url(source_url: str) -> str:
        """
        Derive a local CSV filename from a direct CSV URL.

        Args:
            source_url: Direct CSV URL.

        Returns:
            A filename ending in `.csv`.
        """
        parsed = urlparse(source_url)
        filename = Path(parsed.path).name or "dataset.csv"
        if not filename.lower().endswith(".csv"):
            filename = f"{filename}.csv"
        return filename
