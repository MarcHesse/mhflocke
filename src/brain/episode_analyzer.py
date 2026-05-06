"""
MH-FLOCKE — EpisodeAnalyzer (Meta-Learning Loop Phase A)
=========================================================
Analyzes navigation episodes (scent found / missed) to extract
insights about what makes the creature successful.

Design reference: docs/DESIGN_AUTONOMOUS_LOOP.md, Verbindung 1
Biology: Hippocampal replay + prefrontal evaluation (Botvinick 2019)

The analyzer compares successful vs unsuccessful episodes and
identifies key differentiating factors. Insights are stored as
dictionaries that can be fed into the Concept Graph (Phase B).

Author: Marc Hesse
License: Apache 2.0
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
import time
import logging

logger = logging.getLogger(__name__)


@dataclass
class NavigationEvent:
    """A single navigation event (light found or missed)."""
    event_type: str          # 'found' or 'missed' or 'timeout'
    timestamp: float         # time.time()
    step: int                # global training step
    context: Dict[str, float] = field(default_factory=dict)


@dataclass
class Insight:
    """A learned observation from episode comparison."""
    insight_type: str        # 'correlation', 'threshold', 'pattern'
    factor: str              # which context key matters
    description: str         # human-readable
    confidence: float        # 0-1, based on evidence count
    recommendation: str      # what to change
    evidence_for: int = 0    # how many episodes support this
    evidence_against: int = 0
    created_step: int = 0
    last_updated_step: int = 0


class EpisodeAnalyzer:
    """Compares successful vs unsuccessful navigation episodes.

    Phase A of the Meta-Learning Loop (DESIGN_AUTONOMOUS_LOOP.md).
    Runs after each scent/light event and identifies what made
    the creature successful or unsuccessful.

    Feeds into: Concept Graph (Phase B) via get_new_insights().
    """

    def __init__(self, min_events_for_analysis: int = 4,
                 max_events: int = 200,
                 confidence_threshold: float = 0.6):
        """
        Args:
            min_events_for_analysis: Need at least this many events
                (with at least 1 success and 1 failure) before analyzing
            max_events: Maximum events to keep in history
            confidence_threshold: Minimum confidence to report an insight
        """
        self.min_events = min_events_for_analysis
        self.max_events = max_events
        self.confidence_threshold = confidence_threshold

        self.events: List[NavigationEvent] = []
        self.insights: Dict[str, Insight] = {}  # keyed by factor name
        self._pending_insights: List[Insight] = []  # new, not yet consumed

        # Context keys to analyze (must be numeric)
        self.analysis_keys = [
            'smell_strength',       # light intensity at event
            'dist_to_light',        # distance to target
            'gait_quality',         # GQ score
            'heading_error',        # angular offset from target
            'steering_offset',      # PID output
            'upright',              # stability
            'velocity',             # forward speed
            'cpg_weight',           # CPG vs actor mix
            'actor_competence',     # actor skill level
            'steps_since_last',     # steps between events
            'cumulative_turn',      # total turning in degrees
        ]

    def record_event(self, event_type: str, context: Dict[str, float],
                     step: int = 0):
        """Record a navigation event with context.

        Args:
            event_type: 'found' (reached light), 'missed' (light expired
                        or creature moved too far), 'timeout' (no progress)
            context: Dictionary of numeric context values at event time
            step: Current training step
        """
        event = NavigationEvent(
            event_type=event_type,
            timestamp=time.time(),
            step=step,
            context=context,
        )
        self.events.append(event)

        # Evict old events
        if len(self.events) > self.max_events:
            self.events = self.events[-self.max_events:]

        # Run analysis after each event
        self._analyze()

    def _analyze(self):
        """Compare successful vs unsuccessful events."""
        successes = [e for e in self.events if e.event_type == 'found']
        failures = [e for e in self.events
                    if e.event_type in ('missed', 'timeout')]

        if len(successes) < 1 or len(failures) < 1:
            return  # Need at least one of each
        if len(self.events) < self.min_events:
            return  # Not enough data

        # For each context key, compare means between success/failure
        for key in self.analysis_keys:
            success_vals = [e.context.get(key) for e in successes
                           if key in e.context and e.context[key] is not None]
            failure_vals = [e.context.get(key) for e in failures
                           if key in e.context and e.context[key] is not None]

            if len(success_vals) < 1 or len(failure_vals) < 1:
                continue

            s_mean = sum(success_vals) / len(success_vals)
            f_mean = sum(failure_vals) / len(failure_vals)
            diff = s_mean - f_mean

            # Is the difference meaningful?
            # Use coefficient of variation as noise estimate
            all_vals = success_vals + failure_vals
            overall_mean = sum(all_vals) / len(all_vals)
            if abs(overall_mean) < 1e-6:
                continue  # Can't normalize

            relative_diff = abs(diff) / max(abs(overall_mean), 0.01)

            if relative_diff < 0.15:
                continue  # Less than 15% difference — not meaningful

            # Calculate confidence from sample size
            n_total = len(success_vals) + len(failure_vals)
            confidence = min(1.0, n_total / 20.0) * min(1.0, relative_diff)

            if confidence < self.confidence_threshold:
                continue

            # Generate or update insight
            direction = 'higher' if diff > 0 else 'lower'
            description = (
                f"Successful navigation has {direction} {key} "
                f"(success: {s_mean:.3f}, failure: {f_mean:.3f}, "
                f"diff: {diff:+.3f})"
            )

            # Recommendation based on direction and factor
            recommendation = self._generate_recommendation(
                key, direction, s_mean, f_mean)

            insight = Insight(
                insight_type='correlation',
                factor=key,
                description=description,
                confidence=confidence,
                recommendation=recommendation,
                evidence_for=len(success_vals),
                evidence_against=len(failure_vals),
                created_step=(self.insights[key].created_step
                              if key in self.insights
                              else self.events[-1].step),
                last_updated_step=self.events[-1].step,
            )

            # Check if this is new or updated
            old = self.insights.get(key)
            if old is None or abs(old.confidence - confidence) > 0.1:
                self._pending_insights.append(insight)
                logger.info(f"[INSIGHT] {description} (conf={confidence:.2f})")

            self.insights[key] = insight

    def _generate_recommendation(self, key: str, direction: str,
                                  s_mean: float, f_mean: float) -> str:
        """Generate actionable recommendation from insight."""
        recommendations = {
            'gait_quality': {
                'higher': 'Maintain high gait quality — stable gait improves navigation',
                'lower': 'Gait quality may not correlate with navigation success',
            },
            'heading_error': {
                'lower': 'Lower heading error improves success — strengthen PID steering',
                'higher': 'Some heading variation may aid exploration',
            },
            'velocity': {
                'higher': 'Higher velocity correlates with success — increase stride',
                'lower': 'Slower movement may improve precision',
            },
            'steering_offset': {
                'lower': 'Less steering correction at success — dog is well-aimed',
                'higher': 'Active steering correction correlates with finding targets',
            },
            'steps_since_last': {
                'lower': 'Shorter intervals between finds — creature is efficient',
                'higher': 'Longer search periods lead to finds — patience pays',
            },
            'smell_strength': {
                'higher': 'Higher light intensity at event — creature approaches closely',
                'lower': 'Success at lower intensity — creature detects from further',
            },
            'cumulative_turn': {
                'higher': 'More turning correlates with success — exploration helps',
                'lower': 'Less turning correlates with success — direct paths work',
            },
            'cpg_weight': {
                'lower': 'Lower CPG weight (more actor) correlates with success',
                'higher': 'Higher CPG weight (less actor) correlates with success',
            },
        }

        if key in recommendations and direction in recommendations[key]:
            return recommendations[key][direction]
        return f"Factor '{key}' is {direction} during successful navigation"

    def get_new_insights(self) -> List[Insight]:
        """Return and clear pending new/updated insights.

        These should be fed into the Concept Graph (Phase B).
        """
        result = list(self._pending_insights)
        self._pending_insights.clear()
        return result

    def get_all_insights(self) -> Dict[str, Insight]:
        """Return all current insights (for FLOG/persistence)."""
        return dict(self.insights)

    def get_success_rate(self) -> float:
        """Return fraction of events that were successful."""
        if not self.events:
            return 0.0
        found = sum(1 for e in self.events if e.event_type == 'found')
        return found / len(self.events)

    def stats(self) -> Dict[str, Any]:
        """Return stats for FLOG logging."""
        return {
            'episode_analyzer_events': len(self.events),
            'episode_analyzer_insights': len(self.insights),
            'episode_analyzer_success_rate': self.get_success_rate(),
            'episode_analyzer_pending': len(self._pending_insights),
            'episode_analyzer_top_factor': (
                max(self.insights.values(), key=lambda i: i.confidence).factor
                if self.insights else 'none'
            ),
            'episode_analyzer_top_confidence': (
                max(i.confidence for i in self.insights.values())
                if self.insights else 0.0
            ),
        }

    def save_state(self) -> Dict[str, Any]:
        """Serialize state for persistence in brain.pt."""
        return {
            'events': [
                {
                    'event_type': e.event_type,
                    'timestamp': e.timestamp,
                    'step': e.step,
                    'context': e.context,
                }
                for e in self.events[-50:]  # Keep last 50 events
            ],
            'insights': {
                k: {
                    'insight_type': v.insight_type,
                    'factor': v.factor,
                    'description': v.description,
                    'confidence': v.confidence,
                    'recommendation': v.recommendation,
                    'evidence_for': v.evidence_for,
                    'evidence_against': v.evidence_against,
                    'created_step': v.created_step,
                    'last_updated_step': v.last_updated_step,
                }
                for k, v in self.insights.items()
            },
        }

    def load_state(self, state: Dict[str, Any]):
        """Restore state from brain.pt."""
        if 'events' in state:
            self.events = [
                NavigationEvent(**e) for e in state['events']
            ]
        if 'insights' in state:
            self.insights = {
                k: Insight(**v) for k, v in state['insights'].items()
            }
        logger.info(
            f"[EpisodeAnalyzer] Restored {len(self.events)} events, "
            f"{len(self.insights)} insights"
        )
