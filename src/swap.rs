//! Swap policy for recognizer rebuilds (DESIGN_VOSK_REBUILD_BOUNDARIES).
//!
//! Pure decision logic — no Vosk, no IO — so the boundary rules are
//! table-testable. The policy tracks two word sets: `applied` (behind the
//! live dynamic recognizer) and `pending` (arrived while the decoder may
//! hold utterance state; swapped in at the next safe boundary). Word
//! lists are canonical (sorted, deduped, `[unk]` included) before they
//! reach the policy — main.rs owns canonicalization.

#[derive(Debug, PartialEq)]
pub enum UpdateAction {
    /// The set already matches the live or queued grammar — apply the
    /// force-finalize tweak only, no rebuild.
    TweakOnly,
    /// The decoder is idle; rebuild and swap immediately.
    SwapNow(Vec<String>),
    /// The decoder may hold utterance state; the set is parked and will
    /// swap at the next safe boundary.
    Deferred,
}

#[derive(Debug, Default)]
pub struct SwapPolicy {
    /// Set behind the live dynamic recognizer. None = startup recognizer
    /// is live (pre-first-update) — in that state no incoming set is
    /// "unchanged", matching the old guard's
    /// `dynamic_recognizer.is_some()` requirement.
    applied: Option<Vec<String>>,
    pending: Option<Vec<String>>,
}

impl SwapPolicy {
    pub fn new() -> Self {
        Self::default()
    }

    /// Decide what to do with an incoming (canonical) word set.
    ///
    /// `decoder_idle` is the caller's judgment: session mode with no
    /// active session. Continuous mode is never idle — its boundaries
    /// are the 0.8s force-finalize ticks.
    pub fn on_vocabulary_update(
        &mut self,
        words: Vec<String>,
        decoder_idle: bool,
    ) -> UpdateAction {
        if self.applied.as_ref() == Some(&words) {
            // Matches the LIVE set — anything still pending is moot
            // (the vocabulary changed back before we ever swapped).
            self.pending = None;
            return UpdateAction::TweakOnly;
        }
        if self.pending.as_ref() == Some(&words) {
            // Exactly this set is already queued.
            return UpdateAction::TweakOnly;
        }
        if decoder_idle {
            UpdateAction::SwapNow(words)
        } else {
            self.pending = Some(words);
            UpdateAction::Deferred
        }
    }

    /// Called at a safe boundary (finalized / force-finalized /
    /// audio_stop / audio_start). Pops the pending set if one is parked.
    pub fn take_pending(&mut self) -> Option<Vec<String>> {
        self.pending.take()
    }

    /// Re-park a set after a failed build so the next boundary retries.
    pub fn restore_pending(&mut self, words: Vec<String>) {
        self.pending = Some(words);
    }

    /// Record a successful build + swap.
    pub fn note_applied(&mut self, words: Vec<String>) {
        self.applied = Some(words);
    }

    /// The set behind the live recognizer, for delta logging.
    pub fn applied(&self) -> Option<&[String]> {
        self.applied.as_deref()
    }

    #[cfg(test)]
    pub fn has_pending(&self) -> bool {
        self.pending.is_some()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn w(items: &[&str]) -> Vec<String> {
        items.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn idle_update_swaps_now() {
        let mut p = SwapPolicy::new();
        assert_eq!(
            p.on_vocabulary_update(w(&["a", "b"]), true),
            UpdateAction::SwapNow(w(&["a", "b"])),
        );
    }

    #[test]
    fn busy_update_defers_and_boundary_pops_once() {
        let mut p = SwapPolicy::new();
        assert_eq!(p.on_vocabulary_update(w(&["a"]), false), UpdateAction::Deferred);
        assert_eq!(p.take_pending(), Some(w(&["a"])));
        assert_eq!(p.take_pending(), None);
    }

    #[test]
    fn unchanged_set_is_tweak_only() {
        let mut p = SwapPolicy::new();
        p.note_applied(w(&["a", "b"]));
        assert_eq!(p.on_vocabulary_update(w(&["a", "b"]), false), UpdateAction::TweakOnly);
        assert!(!p.has_pending());
    }

    #[test]
    fn update_matching_live_set_cancels_pending() {
        // live=A, pending=B, update=A — the change was undone before any
        // boundary; swapping to B later would apply a stale vocabulary.
        let mut p = SwapPolicy::new();
        p.note_applied(w(&["a"]));
        assert_eq!(p.on_vocabulary_update(w(&["b"]), false), UpdateAction::Deferred);
        assert_eq!(p.on_vocabulary_update(w(&["a"]), false), UpdateAction::TweakOnly);
        assert_eq!(p.take_pending(), None);
    }

    #[test]
    fn update_matching_pending_set_is_tweak_only_and_keeps_pending() {
        let mut p = SwapPolicy::new();
        p.note_applied(w(&["a"]));
        assert_eq!(p.on_vocabulary_update(w(&["b"]), false), UpdateAction::Deferred);
        assert_eq!(p.on_vocabulary_update(w(&["b"]), false), UpdateAction::TweakOnly);
        assert_eq!(p.take_pending(), Some(w(&["b"])));
    }

    #[test]
    fn newer_pending_replaces_older() {
        let mut p = SwapPolicy::new();
        p.note_applied(w(&["a"]));
        assert_eq!(p.on_vocabulary_update(w(&["b"]), false), UpdateAction::Deferred);
        assert_eq!(p.on_vocabulary_update(w(&["c"]), false), UpdateAction::Deferred);
        assert_eq!(p.take_pending(), Some(w(&["c"])));
        assert_eq!(p.take_pending(), None);
    }

    #[test]
    fn failed_build_can_repark_for_retry() {
        let mut p = SwapPolicy::new();
        assert_eq!(p.on_vocabulary_update(w(&["b"]), false), UpdateAction::Deferred);
        let taken = p.take_pending().unwrap();
        p.restore_pending(taken);
        assert_eq!(p.take_pending(), Some(w(&["b"])));
    }

    #[test]
    fn idle_update_does_not_linger_as_pending() {
        let mut p = SwapPolicy::new();
        let action = p.on_vocabulary_update(w(&["a"]), true);
        assert!(matches!(action, UpdateAction::SwapNow(_)));
        assert!(!p.has_pending());
    }
}
