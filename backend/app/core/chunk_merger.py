"""
Sliding-window chunk merge for overlapped ASR outputs.

When continuous audio (rap, music, no-silence) is split into fixed-duration
chunks for cloud API calls, the resulting segments may have jittery or
disjointed timestamps at chunk boundaries. This module finds the best
alignment between overlapping chunk transcriptions via text matching,
then cuts at the midpoint of the matched overlap region.

Reference: Groq API Cookbook chunked ASR merging strategy.
"""
import difflib
from typing import List, Optional, Tuple

from .engine_base import Segment, TranscribeResult


class ChunkMerger:
    """Merge overlapping ASR chunk results into a clean continuous sequence.

    Uses sliding-window text matching to find the best alignment point
    between consecutive chunks, then joins at the midpoint.
    """

    def __init__(self, min_matches: int = 2, fuzzy_threshold: float = 0.7):
        self.min_matches = min_matches
        self.fuzzy_threshold = fuzzy_threshold

    def merge(
        self,
        chunks: List[TranscribeResult],
        overlap_duration: float = 10.0,
    ) -> TranscribeResult:
        """Merge multiple overlapping transcriptions.

        Args:
            chunks: List of TranscribeResult from consecutive overlapping chunks.
            overlap_duration: Expected overlap in seconds (used to extract overlap region).
        """
        if not chunks:
            return TranscribeResult()
        if len(chunks) == 1:
            return chunks[0]

        merged_segments: List[Segment] = list(chunks[0].segments)

        for i in range(1, len(chunks)):
            right_segments = chunks[i].segments
            if not right_segments:
                continue
            if not merged_segments:
                merged_segments = right_segments
                continue

            merged_segments = self._merge_pair(
                merged_segments, right_segments, overlap_duration
            )

        return TranscribeResult(segments=merged_segments)

    def _merge_pair(
        self,
        left: List[Segment],
        right: List[Segment],
        overlap_duration: float,
    ) -> List[Segment]:
        """Merge a single pair of consecutive chunk results."""
        if not left or not right:
            return left or right

        # Extract overlap regions
        left_overlap = self._extract_overlap_end(left, overlap_duration)
        right_overlap = self._extract_overlap_start(right, overlap_duration)

        if not left_overlap or not right_overlap:
            return left + right

        # Find best alignment
        alignment = self._find_alignment(left_overlap, right_overlap)

        if alignment is None:
            # Fallback: cut by time boundary
            cut_idx = len(left)
            right_start = right[0].start
            for j in range(len(left) - 1, -1, -1):
                if left[j].end <= right_start:
                    cut_idx = j + 1
                    break
            return left[:cut_idx] + right

        l_match_start, l_match_end, r_match_start, r_match_end, matches = alignment

        # Cut at midpoint of matched region
        l_offset = len(left) - len(left_overlap)
        l_mid = (l_match_start + l_match_end) // 2
        cut_point = l_offset + l_mid

        r_mid = (r_match_start + r_match_end) // 2

        return left[:cut_point] + right[r_mid:]

    def _extract_overlap_end(self, segments: List[Segment], duration: float) -> List[Segment]:
        """Extract segments from the end of a sequence (the overlap tail)."""
        if not segments:
            return []
        threshold = segments[-1].end - duration
        result = []
        for s in segments:
            if s.start >= threshold:
                result.append(s)
        return result

    def _extract_overlap_start(self, segments: List[Segment], duration: float) -> List[Segment]:
        """Extract segments from the start of a sequence (the overlap head)."""
        if not segments:
            return []
        threshold = segments[0].start + duration
        result = []
        for s in segments:
            if s.end <= threshold:
                result.append(s)
        return result

    def _find_alignment(
        self,
        left: List[Segment],
        right: List[Segment],
    ) -> Optional[Tuple[int, int, int, int, int]]:
        """Sliding-window alignment search.

        Returns:
            (l_start, l_end, r_start, r_end, match_count) or None
        """
        best_score = 0.0
        best_result = None
        L, R = len(left), len(right)

        for i in range(1, L + R + 1):
            epsilon = i / 10000.0  # bias toward longer matches

            l_start = max(0, L - i)
            l_end = min(L, L + R - i)
            r_start = max(0, i - L)
            r_end = min(R, i)

            left_slice = left[l_start:l_end]
            right_slice = right[r_start:r_end]

            if not left_slice or not right_slice:
                continue

            matches = sum(
                1 for a, b in zip(left_slice, right_slice)
                if difflib.SequenceMatcher(None, a.text, b.text).ratio() > self.fuzzy_threshold
            )

            if matches >= self.min_matches:
                score = matches / i + epsilon
                if score > best_score:
                    best_score = score
                    best_result = (l_start, l_end, r_start, r_end, matches)

        return best_result
