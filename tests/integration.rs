//! End-to-end integration test: wav_source → vosk_commands via the
//! source-driven pipeline orchestrator. Requires VOSK_MODEL_PATH env
//! var pointing to a model directory (e.g., vosk-model-small-en-us-0.15).
//!
//! Run with: cargo test -p vosk-commands --test integration -- --ignored
//!
//! macOS local runs also need DYLD_FALLBACK_LIBRARY_PATH=stages/vosk-commands/lib:
//! libvosk.dylib's install name is bare (no @rpath/), which defeats the
//! baked rpaths; the app bundle fixes it with install_name_tool, and on
//! Linux the ELF rpath resolves libvosk.so as-is.

use std::time::Duration;

use branch_actuator::pipeline::{events::AudioFormat, orchestrator::PipelineOrchestrator};
use tokio::time::timeout;

fn vosk_stage_path() -> String {
    env!("CARGO_BIN_EXE_vosk_commands").to_string()
}

fn wav_source_path() -> String {
    // wav_source is built as part of branch-actuator package
    let dir = std::path::Path::new(env!("CARGO_BIN_EXE_vosk_commands"))
        .parent()
        .unwrap();
    dir.join("wav_source").to_string_lossy().to_string()
}

fn model_path() -> String {
    std::env::var("VOSK_MODEL_PATH").expect(
        "set VOSK_MODEL_PATH to a vosk model directory (e.g., vosk-model-small-en-us-0.15)",
    )
}

fn fixture(name: &str) -> String {
    format!("{}/tests/fixtures/{name}", env!("CARGO_MANIFEST_DIR"))
}

/// Full pipeline: wav_source reads a WAV of spoken command words →
/// vosk_commands recognizes them against a fixed grammar → orchestrator
/// surfaces the recognitions.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore]
async fn wav_source_to_vosk_commands_recognizes_command_words() {
    let model = model_path();
    let wav = fixture("test_commands.wav");
    let grammar = fixture("grammar.json");
    let vosk = vosk_stage_path();
    let source = wav_source_path();

    let result = timeout(
        Duration::from_secs(30),
        PipelineOrchestrator::run_source_driven_pipeline(
            (&source, &[&wav]),
            (&vosk, &["--model", &model, "--grammar", &grammar]),
            AudioFormat::PCM_16K_MONO,
        ),
    )
    .await
    .expect("pipeline must finish within 30s")
    .expect("orchestrator must succeed");

    // Vosk finalizes per utterance: recognitions arrive as mid-stream
    // segments; the audio_stop final carries only whatever decoded last.
    let recognized = result
        .segments
        .iter()
        .map(String::as_str)
        .chain(std::iter::once(result.transcript.as_str()))
        .collect::<Vec<_>>()
        .join(" ");
    for word in ["browser", "scroll", "click"] {
        assert!(
            recognized.contains(word),
            "expected {word:?} in recognized text, got: {recognized:?}"
        );
    }
    assert_eq!(result.session_id.len(), 36);
}
