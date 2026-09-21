"""Read the signed fastMRI archive URLs out of ``.env``, and check they are still alive.

NYU signs these links for about two weeks, so the usual failure is a 403 on a link that
worked last sprint. Two jobs, one file:

* printing them one per line is what lets a Slurm array index them, since ``.env`` keeps
  every URL inside a single variable and an array task needs exactly one;
* ``--check`` verifies each one is live without downloading it, standing in for issue
  #77's "answers 200 to a HEAD request": a real HEAD fails on these specific links
  regardless of validity (see :func:`head_status`), so this checks the signature the
  way an actual download would instead.

Parsing goes through :func:`cyfm.data.download.load_env` and the same comma-or-newline
splitting the auto-downloader uses, so there is one ``.env`` format rather than two.

Usage::

    uv run python scripts/data/fastmri_urls.py --check
    uv run python scripts/data/fastmri_urls.py --var FASTMRI_MINI_URLS
"""

from __future__ import annotations

import argparse
import sys
import urllib.error
import urllib.request

from cyfm.data.download import load_env

TIMEOUT = 30


def archive_name(url: str) -> str:
    """The archive a signed URL points at, with the query string stripped."""
    return url.split("?", 1)[0].rsplit("/", 1)[-1]


def urls_from_env(env_path: str, variable: str) -> list[str]:
    """Every URL in one ``.env`` variable, in order.

    Args:
        env_path: The ``.env`` file to read.
        variable: Which key holds the URLs, e.g. ``FASTMRI_FULL_URLS``.

    Returns:
        URLs with surrounding whitespace removed and blanks dropped.
    """
    raw = load_env(env_path).get(variable) or ""
    return [url.strip() for url in raw.replace("\n", ",").split(",") if url.strip()]


def head_status(url: str) -> tuple[int | None, str]:
    """Check one URL without downloading it, returning its status and a short note.

    Not actually a HEAD: NYU's presigned S3 links use AWS's SigV2 scheme, whose
    signature covers the HTTP verb, so a link signed for GET returns 403 to a genuine
    HEAD regardless of whether it has expired -- confirmed against a live link, which
    answered 403 to HEAD and 206 to this. A single-byte range request verifies the
    signature the same way a real download would while transferring nothing.
    """
    request = urllib.request.Request(url, headers={"Range": "bytes=0-0"})  # noqa: S310
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT) as response:  # noqa: S310
            # 206 is the expected answer to a range request; a plain 200 means the
            # server ignored the range but the signature still checked out.
            return response.status, "ok"
    except urllib.error.HTTPError as error:
        reason = "expired or wrong signature" if error.code == 403 else str(error.reason)
        return error.code, reason
    except (urllib.error.URLError, TimeoutError) as error:
        return None, f"unreachable: {error}"


def main() -> int:
    """Print or check the fastMRI download URLs."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--env", default=".env", help="The .env file holding the links.")
    parser.add_argument("--var", default="FASTMRI_FULL_URLS", help="Which key to read.")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Verify each URL's signature and exit non-zero unless every one is live.",
    )
    args = parser.parse_args()

    urls = urls_from_env(args.env, args.var)
    if not urls:
        print(f"{args.var} is empty or absent in {args.env}", file=sys.stderr)
        return 2

    if not args.check:
        for url in urls:
            print(url)
        return 0

    # 200 if the server ignores the range, 206 if it honours it -- either means the
    # signature was accepted.
    failed = 0
    for position, url in enumerate(urls):
        status, note = head_status(url)
        if status not in (200, 206):
            failed += 1
        shown = status if status is not None else "---"
        print(f"[{position:>2}] {shown} {archive_name(url)}  {note}")

    print(f"\n{len(urls) - failed}/{len(urls)} links are live")
    if failed:
        print("Refresh them at https://fastmri.med.nyu.edu/ and update .env (#77).")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
