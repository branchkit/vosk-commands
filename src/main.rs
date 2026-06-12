use std::path::PathBuf;
use std::{env, process};

use branch_actuator::pipeline::events::{
    event_type, AudioChunk, AudioFormat, Capability, FlowCredit, Transcript,
    TranscriptAlternative,
};
use branch_actuator::pipeline::wire::{Event, Reader, Writer};
use serde_json::Value;
use tokio::io::{stdin, stdout};
use vosk::{CompleteResult, DecodingState, Model, Recognizer};

mod swap;
use swap::{SwapPolicy, UpdateAction};

const SAMPLE_RATE: f32 = 16000.0;

// Stage log line: leading RFC3339-millis UTC timestamp matching actuator.log's
// `[2026-06-01T05:55:13.116Z]` prefix so the two logs correlate on one clock,
// then the `[vosk_commands]` tag. Use this instead of bare `eprintln!`.
macro_rules! vlog {
    ($($arg:tt)*) => {{
        eprintln!(
            "[{}] [vosk_commands] {}",
            chrono::Utc::now().format("%Y-%m-%dT%H:%M:%S%.3fZ"),
            format_args!($($arg)*)
        );
    }};
}

#[tokio::main(flavor = "current_thread")]
async fn main() {
    if let Err(e) = run().await {
        vlog!("fatal: {e}");
        process::exit(1);
    }
}

async fn run() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().skip(1).collect();
    let stage_args = parse_args(&args)?;
    let lifecycle_mode = stage_args.lifecycle_mode;

    // Resolve relative paths against exe directory
    let model_path = resolve_relative_to_exe(&stage_args.model);
    let model = Model::new(model_path.to_str().unwrap_or(&model_path.to_string_lossy()))
        .ok_or_else(|| format!("failed to load model from {}", model_path.display()))?;

    let grammar: Vec<String> = if let Some(path) = stage_args.grammar {
        let resolved = resolve_relative_to_exe(&path);
        let content = std::fs::read_to_string(&resolved)
            .map_err(|e| format!("failed to read grammar file {}: {e}", resolved.display()))?;
        serde_json::from_str(&content)
            .map_err(|e| format!("grammar file must be a JSON array of strings: {e}"))?
    } else {
        vec!["[unk]".into()]
    };

    let grammar_refs: Vec<&str> = grammar.iter().map(|s| s.as_str()).collect();

    let mut full_recognizer = Recognizer::new_with_grammar(&model, SAMPLE_RATE, &grammar_refs)
        .ok_or("failed to create recognizer with grammar")?;
    full_recognizer.set_partial_words(true);
    full_recognizer.set_words(true);
    full_recognizer.set_max_alternatives(3);

    // Dynamic recognizer is (re)created on every `vocabulary_update` event
    // and dropped on `recognizer_reset`. When None, `full_recognizer` (the
    // startup-loaded grammar) is the active one.
    //
    // What "dynamic" actually holds varies by sender — and historically
    // the variable name was "narrowed_recognizer", which was misleading:
    //
    //   - Voice plugin's LockForSpeak / LockForSandbox push a *truly
    //     narrow* word list (13 stop phrases / sandbox test words). These
    //     are acoustic-task modes; narrowing is the point.
    //   - Voice plugin's Init/Refresh push the full union grammar
    //     (every command's words + plugin HWMs). Not narrowing — refresh.
    //   - `vocabulary.commit` from any plugin (browser hints, etc.) goes
    //     through the actuator's `send_vocabulary_update` path which
    //     builds the full union vocab. Not narrowing — refresh.
    //
    // The recognizer slot is the same in all cases; only the word count
    // tells you whether you're in an acoustic-task narrow mode vs. a
    // routine refresh. Operator: read the word count, not the variable
    // name.
    let mut dynamic_recognizer: Option<Recognizer> = None;

    macro_rules! active_rec {
        () => {
            dynamic_recognizer.as_mut().unwrap_or(&mut full_recognizer)
        };
    }

    let mut reader = Reader::new(stdin());
    let mut writer = Writer::new(stdout());

    // Capability handshake
    let cap = Capability {
        stage_type: "command_recognition".into(),
        stage_name: "vosk".into(),
        audio_formats: vec![AudioFormat::PCM_16K_MONO],
        lifecycle_modes: vec!["persistent".into()],
        feature_flags: serde_json::Map::new(),
    };
    writer
        .write_event(&Event::new(
            event_type::CAPABILITY,
            serde_json::to_value(&cap)?,
        ))
        .await?;

    // Grant initial credit upstream (continuous mode needs more to cover startup)
    let initial_frames = if lifecycle_mode == LifecycleMode::Continuous { 64 } else { 16 };
    let initial_credit = FlowCredit {
        frames: initial_frames,
        session_id: String::new(),
    };
    writer
        .write_event(&Event::new(
            event_type::FLOW_CREDIT,
            serde_json::to_value(&initial_credit)?,
        ))
        .await?;

    if lifecycle_mode == LifecycleMode::Continuous {
        vlog!("continuous mode: no VAD gating, force-finalize every 0.8s");
    }

    let mut current_session: Option<String> = None;
    let mut frames_since_credit: u32 = 0;
    let mut skip_next_reset = false;
    let mut samples_since_finalize: u32 = 0;
    let mut last_partial: Option<String> = None;
    let mut force_finalize_samples: u32 = (SAMPLE_RATE * 0.8) as u32;
    const DEFAULT_FORCE_FINALIZE_SAMPLES: u32 = (SAMPLE_RATE * 0.8) as u32;
    // Swap policy: tracks the applied word set (redundant-rebuild guard)
    // and parks genuinely-new sets that arrive mid-decode for the next
    // safe boundary (DESIGN_VOSK_REBUILD_BOUNDARIES).
    let mut policy = SwapPolicy::new();

    loop {
        let event = match reader.read_event().await? {
            Some(e) => e,
            None => break,
        };

        match event.event_type.as_str() {
            "vocabulary_update" => {
                if let Some(words) = event.data.get("words").and_then(|v| v.as_array()) {
                    let mut new_grammar: Vec<String> = words.iter()
                        .filter_map(|v| v.as_str().map(String::from))
                        .collect();
                    // The platform sends a pure engine-agnostic word list;
                    // Vosk's unknown-word sentinel is this stage's concern.
                    // Append before sort/dedup so the unchanged-set guard
                    // sees a canonical set either way.
                    new_grammar.push("[unk]".to_string());
                    // Canonicalize to a sorted, deduped SET before the unchanged-set
                    // guard below — the guard must compare word-set membership, not
                    // list order or duplicates. Upstream occasionally emits a word
                    // twice (a `cancels_bridge` command word like "dismiss"/"over"
                    // appended while a hint bridge is active), which toggles the list
                    // length (e.g. 299<->297) without changing the set. Comparing raw
                    // lists, that toggle defeats the guard and forces a recognizer
                    // rebuild mid-utterance — landing while the user speaks the second
                    // word of a two-word codeword, which endpoints the recognizer and
                    // drops the tail (the "say it twice on a fresh page" truncation).
                    new_grammar.sort();
                    new_grammar.dedup();
                    // The Vosk grammar is the union *word* set, not the matcher's
                    // per-context narrowing. A scan storm (e.g. browser hints after a
                    // page nav) re-pushes commands constantly but rarely changes the
                    // word set — codewords reuse words already in vocab — so most
                    // updates arrive byte-for-byte identical (the word list is sorted
                    // upstream). Rebuilding the recognizer on an unchanged set is pure
                    // cost: it drops in-progress audio and truncates a command spoken
                    // across the rebuild (the "say it twice on a fresh page" bug).
                    // Skip the rebuild when the word set is unchanged; still honor a
                    // force-finalize change since that needs no new recognizer.
                    let force_finalize = || -> u32 {
                        match event.data.get("force_finalize_ms").and_then(|v| v.as_u64()) {
                            Some(0) => 0,
                            Some(ms) => (SAMPLE_RATE * ms as f32 / 1000.0) as u32,
                            None => {
                                if event.data.get("force_finalize").and_then(|v| v.as_bool()) == Some(false) {
                                    0
                                } else {
                                    DEFAULT_FORCE_FINALIZE_SAMPLES
                                }
                            }
                        }
                    };
                    // Force-finalize is loop state, not recognizer state —
                    // apply it on receipt regardless of whether the word
                    // set swaps now, later, or not at all.
                    force_finalize_samples = force_finalize();
                    // A genuinely-new set rebuilds immediately only when
                    // the decoder holds no utterance state; mid-decode it
                    // is parked and swapped at the next safe boundary, so
                    // the in-flight utterance keeps decoding against the
                    // grammar it started with instead of being truncated.
                    // Continuous mode is never idle — its boundaries are
                    // the force-finalize ticks.
                    let decoder_idle = lifecycle_mode != LifecycleMode::Continuous
                        && current_session.is_none();
                    match policy.on_vocabulary_update(new_grammar, decoder_idle) {
                        UpdateAction::TweakOnly => {}
                        UpdateAction::SwapNow(words) => {
                            if build_and_swap(
                                &model, words, &mut dynamic_recognizer, &mut policy,
                                &mut samples_since_finalize, &mut last_partial,
                                "idle", ff_ms(force_finalize_samples),
                            ).is_ok() {
                                skip_next_reset = true;
                            }
                        }
                        UpdateAction::Deferred => {
                            vlog!(
                                "vocabulary change parked — decoder active; swapping at next boundary (force_finalize_ms={} applied now)",
                                ff_ms(force_finalize_samples),
                            );
                        }
                    }
                }
            }
            "recognizer_reset" => {
                dynamic_recognizer = None;
                policy.on_reset();
                full_recognizer.reset();
                let silence = vec![0i16; (SAMPLE_RATE * 0.3) as usize];
                let _ = full_recognizer.accept_waveform(&silence);
                samples_since_finalize = 0;
                force_finalize_samples = DEFAULT_FORCE_FINALIZE_SAMPLES;
                vlog!("recognizer reset → startup grammar (cached)");
            }
            "audio_start" => {
                if lifecycle_mode == LifecycleMode::Continuous {
                    // In continuous mode, ignore session signals — audio flows continuously
                    continue;
                }
                if let Some(sid) = event.data.get("session_id").and_then(Value::as_str) {
                    current_session = Some(sid.to_string());
                    vlog!("session start: {}", &sid[..8.min(sid.len())]);
                }
                // Boundary: nothing decoded yet this session — a parked
                // vocabulary can swap in before any speech arrives.
                if let Some(words) = policy.take_pending() {
                    match build_and_swap(
                        &model, words, &mut dynamic_recognizer, &mut policy,
                        &mut samples_since_finalize, &mut last_partial,
                        "audio_start", ff_ms(force_finalize_samples),
                    ) {
                        Ok(()) => skip_next_reset = true,
                        Err(words) => policy.restore_pending(words),
                    }
                }
                if skip_next_reset {
                    skip_next_reset = false;
                } else {
                    active_rec!().reset();
                }
                // Pre-feed silence so the decoder has context before speech onset.
                // Kaldi models need ~300ms of silence to anchor word boundaries.
                let silence = vec![0i16; (SAMPLE_RATE * 0.3) as usize];
                let _ = active_rec!().accept_waveform(&silence);
                frames_since_credit = 0;
                samples_since_finalize = 0;
                writer
                    .write_event(&Event::new(
                        event_type::FLOW_CREDIT,
                        serde_json::to_value(&FlowCredit {
                            frames: 16,
                            session_id: String::new(),
                        })?,
                    ))
                    .await?;
            }
            "audio_chunk" => {
                let chunk: AudioChunk = serde_json::from_value(event.data.clone())?;
                let payload = &event.payload;

                // Decode i16 PCM
                let samples: Vec<i16> = payload
                    .chunks_exact(2)
                    .map(|b| i16::from_le_bytes([b[0], b[1]]))
                    .collect();

                // Feed to Vosk
                samples_since_finalize += samples.len() as u32;
                let is_dynamic = dynamic_recognizer.is_some();
                let state = active_rec!().accept_waveform(&samples);

                let force_finalize = force_finalize_samples > 0
                    && matches!(state, Ok(DecodingState::Running))
                    && samples_since_finalize >= force_finalize_samples;

                let result = match state {
                    Ok(DecodingState::Finalized) => {
                        samples_since_finalize = 0;
                        Some(active_rec!().result())
                    }
                    Ok(DecodingState::Running) if force_finalize => {
                        samples_since_finalize = 0;
                        Some(active_rec!().final_result())
                    }
                    Ok(DecodingState::Running) => {
                        let pr = active_rec!().partial_result();
                        let text = pr.partial.trim();
                        if !text.is_empty() {
                            last_partial = Some(strip_unk(text));
                        }
                        None
                    }
                    Ok(DecodingState::Failed) => {
                        vlog!("decoding failed");
                        None
                    }
                    Err(e) => {
                        vlog!("accept error: {e}");
                        None
                    }
                };

                let finalize_boundary = result.is_some();
                if let Some(result) = result {
                    let forced = force_finalize;
                    let (text, confidence, alts) = extract_best_result(&result)
                        .unwrap_or_else(|| (String::new(), 0.0, vec![]));
                    let session_id = current_session.clone()
                        .unwrap_or_else(|| chunk.session_id.clone());
                    let partial_hint = if text.is_empty() {
                        last_partial.take()
                    } else {
                        last_partial = None;
                        None
                    };
                    if !text.is_empty() {
                        let tag = if forced { " (forced)" } else { "" };
                        let rec_tag = if is_dynamic { " [dynamic]" } else { " [startup]" };
                        vlog!("recognized{tag}{rec_tag}: \"{text}\" conf={confidence:.2}");
                    } else {
                        let tag = if forced { " (forced)" } else { "" };
                        let rec_tag = if is_dynamic { " [dynamic]" } else { " [startup]" };
                        let hint = partial_hint.as_deref().unwrap_or("");
                        vlog!("empty{tag}{rec_tag}: conf={confidence:.2} last_partial=\"{hint}\"");
                    }
                    writer
                        .write_event(&Event::new(
                            event_type::TRANSCRIPT,
                            serde_json::to_value(&Transcript {
                                session_id,
                                text,
                                is_final: false,
                                partial: false,
                                confidence: Some(confidence),
                                alternatives: if alts.is_empty() { None } else { Some(alts) },
                                last_partial: partial_hint,
                            })?,
                        ))
                        .await?;
                }

                // Boundary: a final (natural or forced) was just emitted,
                // so the decoder holds no utterance state — a parked
                // vocabulary swaps in cleanly. No skip_next_reset here:
                // the new recognizer will consume audio for the rest of
                // the hold, so the next audio_start must reset as usual.
                if finalize_boundary {
                    if let Some(words) = policy.take_pending() {
                        let reason = if force_finalize { "force_finalized" } else { "finalized" };
                        if let Err(words) = build_and_swap(
                            &model, words, &mut dynamic_recognizer, &mut policy,
                            &mut samples_since_finalize, &mut last_partial,
                            reason, ff_ms(force_finalize_samples),
                        ) {
                            policy.restore_pending(words);
                        }
                    }
                }

                // Replenish credit
                frames_since_credit += 1;
                if frames_since_credit >= 4 {
                    frames_since_credit = 0;
                    let replenish = if lifecycle_mode == LifecycleMode::Continuous { 8 } else { 4 };
                    writer
                        .write_event(&Event::new(
                            event_type::FLOW_CREDIT,
                            serde_json::to_value(&FlowCredit {
                                frames: replenish,
                                session_id: chunk.session_id,
                            })?,
                        ))
                        .await?;
                }
            }
            "audio_stop" => {
                if lifecycle_mode == LifecycleMode::Continuous {
                    continue;
                }
                let result = active_rec!().final_result();
                let (text, confidence, alts) = extract_best_result(&result)
                    .unwrap_or_else(|| (String::new(), 0.0, vec![]));
                let partial_hint = if text.is_empty() {
                    last_partial.take()
                } else {
                    last_partial = None;
                    None
                };
                if !text.is_empty() {
                    vlog!("final: \"{text}\" conf={confidence:.2}");
                } else {
                    let hint = partial_hint.as_deref().unwrap_or("");
                    vlog!("final: empty conf={confidence:.2} last_partial=\"{hint}\"");
                }
                let session_id = current_session.take().unwrap_or_default();
                writer
                    .write_event(&Event::new(
                        event_type::TRANSCRIPT,
                        serde_json::to_value(&Transcript {
                            session_id,
                            text,
                            is_final: true,
                            partial: false,
                            confidence: Some(confidence),
                            alternatives: if alts.is_empty() { None } else { Some(alts) },
                            last_partial: partial_hint,
                        })?,
                    ))
                    .await?;
                current_session = None;
                // Boundary: final_result was just taken; no audio flows
                // until the next audio_start, so the fresh recognizer
                // stays virgin — skip its reset next session.
                if let Some(words) = policy.take_pending() {
                    match build_and_swap(
                        &model, words, &mut dynamic_recognizer, &mut policy,
                        &mut samples_since_finalize, &mut last_partial,
                        "audio_stop", ff_ms(force_finalize_samples),
                    ) {
                        Ok(()) => skip_next_reset = true,
                        Err(words) => policy.restore_pending(words),
                    }
                }
            }
            _ => {}
        }
    }

    Ok(())
}

/// Display form of the force-finalize threshold for log lines.
fn ff_ms(force_finalize_samples: u32) -> u32 {
    if force_finalize_samples == 0 {
        0
    } else {
        (force_finalize_samples as f32 / SAMPLE_RATE * 1000.0) as u32
    }
}

/// Build a recognizer for `words`, pre-feed warm-up silence, and swap it
/// in. Logs the word-set delta plus build cost (the phase-0 measurement
/// for DESIGN_VOSK_REBUILD_BOUNDARIES). On failure the old recognizer
/// stays live and the words come back in `Err` so the caller can re-park
/// them for a retry at the next boundary.
#[allow(clippy::too_many_arguments)]
fn build_and_swap(
    model: &Model,
    words: Vec<String>,
    dynamic_recognizer: &mut Option<Recognizer>,
    policy: &mut SwapPolicy,
    samples_since_finalize: &mut u32,
    last_partial: &mut Option<String>,
    reason: &str,
    ff_ms: u32,
) -> Result<(), Vec<String>> {
    let refs: Vec<&str> = words.iter().map(|s| s.as_str()).collect();
    let t0 = std::time::Instant::now();
    let Some(mut new_rec) = Recognizer::new_with_grammar(model, SAMPLE_RATE, &refs) else {
        vlog!("recognizer build FAILED ({} words, at {})", words.len(), reason);
        return Err(words);
    };
    new_rec.set_partial_words(true);
    new_rec.set_words(true);
    new_rec.set_max_alternatives(3);
    let built = t0.elapsed();
    let silence = vec![0i16; (SAMPLE_RATE * 0.3) as usize];
    let _ = new_rec.accept_waveform(&silence);
    let total = t0.elapsed();
    *dynamic_recognizer = Some(new_rec);
    *samples_since_finalize = 0;
    *last_partial = None;
    // Log the word-set DELTA, not the full grammar — rebuilds only fire
    // on real changes (the actuator dedups unchanged broadcasts upstream),
    // and the full ~300-word dump per rebuild was log noise.
    match policy.applied() {
        Some(old) => {
            let old_set: std::collections::HashSet<&str> = old.iter().map(|s| s.as_str()).collect();
            let new_set: std::collections::HashSet<&str> = words.iter().map(|s| s.as_str()).collect();
            let added: Vec<&str> = words.iter().map(|s| s.as_str()).filter(|w| !old_set.contains(w)).collect();
            let removed: Vec<&str> = old.iter().map(|s| s.as_str()).filter(|w| !new_set.contains(w)).collect();
            vlog!(
                "vocabulary updated to {} words (+{} -{}, force_finalize_ms={}, swap at {}, build {:.1?} + prefeed {:.1?}) added={:?} removed={:?}",
                words.len(), added.len(), removed.len(), ff_ms, reason,
                built, total - built, added, removed,
            );
        }
        None => {
            vlog!(
                "initial vocabulary — {} words (force_finalize_ms={}, swap at {}, build {:.1?} + prefeed {:.1?})",
                words.len(), ff_ms, reason, built, total - built,
            );
        }
    }
    policy.note_applied(words);
    Ok(())
}

/// Parse CLI args: supports both positional and flag-style.
/// Positional: `vosk_commands <model-path> [grammar.json]`
/// Flags: `vosk_commands --model <path> --grammar <path>`
struct StageArgs {
    model: String,
    grammar: Option<String>,
    lifecycle_mode: LifecycleMode,
}

#[derive(Clone, Copy, PartialEq)]
enum LifecycleMode {
    Session,
    Continuous,
}

fn parse_args(args: &[String]) -> Result<StageArgs, String> {
    let mut model = None;
    let mut grammar = None;
    let mut lifecycle_mode = None;
    let mut positional = Vec::new();
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--model" => {
                i += 1;
                model = Some(args.get(i).ok_or("--model requires a value")?.clone());
            }
            "--grammar" => {
                i += 1;
                grammar = Some(args.get(i).ok_or("--grammar requires a value")?.clone());
            }
            "--lifecycle_mode" => {
                i += 1;
                lifecycle_mode = Some(args.get(i).ok_or("--lifecycle_mode requires a value")?.clone());
            }
            other if !other.starts_with("--") => {
                positional.push(other.to_string());
            }
            _ => {} // ignore unknown flags
        }
        i += 1;
    }
    // Flag-style takes precedence, fall back to positional
    let model_path = model
        .or_else(|| positional.first().cloned())
        .ok_or_else(|| "usage: vosk_commands <model-path> [grammar.json] or --model <path> [--grammar <path>]".to_string())?;
    let grammar_path = grammar.or_else(|| positional.get(1).cloned());
    let mode = match lifecycle_mode.as_deref() {
        Some("continuous") => LifecycleMode::Continuous,
        _ => LifecycleMode::Session,
    };
    Ok(StageArgs { model: model_path, grammar: grammar_path, lifecycle_mode: mode })
}

fn extract_best_result(result: &CompleteResult) -> Option<(String, f32, Vec<TranscriptAlternative>)> {
    match result {
        CompleteResult::Multiple(multi) => {
            let alts: Vec<TranscriptAlternative> = multi.alternatives.iter()
                .map(|a| TranscriptAlternative {
                    text: strip_unk(a.text.trim()),
                    confidence: a.confidence,
                })
                .filter(|a| !a.text.is_empty())
                .collect();
            let (text, confidence) = if let Some(best) = alts.first() {
                (best.text.clone(), best.confidence)
            } else {
                let best = multi.alternatives.first()?;
                (String::new(), best.confidence)
            };
            Some((text, confidence, alts))
        }
        CompleteResult::Single(single) => {
            let text = strip_unk(single.text.trim());
            Some((text, 1.0, vec![]))
        }
    }
}

fn strip_unk(text: &str) -> String {
    text.split_whitespace()
        .filter(|w| *w != "[unk]")
        .collect::<Vec<_>>()
        .join(" ")
}

fn resolve_relative_to_exe(path: &str) -> PathBuf {
    let p = PathBuf::from(path);
    if p.is_absolute() || p.exists() {
        return p;
    }
    // Application Support models directory (primary location)
    if let Some(home) = env::var_os("HOME") {
        let name = if env::var("BRANCHKIT_DEV").is_ok() { "BranchKitDev" } else { "BranchKit" };
        let app_support = PathBuf::from(home)
            .join("Library/Application Support")
            .join(name)
            .join("models/vosk")
            .join(&p);
        if app_support.exists() {
            return app_support;
        }
    }
    if let Ok(exe) = env::current_exe() {
        if let Some(exe_dir) = exe.parent() {
            let resolved = exe_dir.join(&p);
            if resolved.exists() {
                return resolved;
            }
            // Legacy: macOS app bundle Resources/ (migration period)
            let resources = exe_dir.join("../Resources").join(&p);
            if resources.exists() {
                return resources;
            }
        }
    }
    p
}
